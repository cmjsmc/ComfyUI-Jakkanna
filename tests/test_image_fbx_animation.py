import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import threading
import types
import unittest
from unittest import mock

import cv2
import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "scail2_animate"
sys.path.insert(0, str(TOOLS))

import audit_fbx
import build_driver
import camera_presets
import make_greenfield_workflow
import make_workflow
import render
import run_all
import workflow_builder
from _env import package_version, release_identity, safe_name


ROOT = Path(__file__).resolve().parents[1]


def load_image_fbx_node():
    """Import the node module against stubs, so the suite runs without ComfyUI."""
    package = sys.modules.get("ComfyUI_Jakkanna")
    if package is None:
        package = types.ModuleType("ComfyUI_Jakkanna")
        package.__path__ = [str(ROOT)]
        sys.modules[package.__name__] = package
    nodes_package = sys.modules.get("ComfyUI_Jakkanna.nodes")
    if nodes_package is None:
        nodes_package = types.ModuleType("ComfyUI_Jakkanna.nodes")
        nodes_package.__path__ = [str(ROOT / "nodes")]
        sys.modules[nodes_package.__name__] = nodes_package

    folder_paths = sys.modules.setdefault("folder_paths", types.ModuleType("folder_paths"))
    folder_paths.get_input_directory = tempfile.gettempdir
    folder_paths.get_temp_directory = tempfile.gettempdir
    folder_paths.get_user_directory = tempfile.gettempdir
    if not hasattr(folder_paths, "folder_names_and_paths"):
        folder_paths.folder_names_and_paths = {}

    comfy = sys.modules.get("comfy")
    if comfy is None:
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        sys.modules["comfy"] = comfy
    model_management = sys.modules.get("comfy.model_management")
    if model_management is None:
        model_management = types.ModuleType("comfy.model_management")
        sys.modules["comfy.model_management"] = model_management
        comfy.model_management = model_management
    model_management.unload_all_models = lambda: None
    model_management.throw_exception_if_processing_interrupted = lambda: None

    utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total, node_id=None):
            self.total = total
            self.updates = []

        def update_absolute(self, value, total=None, preview=None):
            self.updates.append((value, total))

    utils.ProgressBar = ProgressBar
    comfy.utils = utils
    sys.modules["comfy.utils"] = utils

    return importlib.import_module("ComfyUI_Jakkanna.nodes.image_fbx_animation")


image_fbx_animation = load_image_fbx_node()


class CameraPresetTests(unittest.TestCase):
    def test_presets_are_stable_and_explicit(self):
        self.assertEqual(
            camera_presets.CAMERA_PRESETS,
            (
                "Auto Director",
                "Classic Full Shot",
                "American / Cowboy Shot",
                "Medium Shot",
                "Medium Close-Up",
                "Classic Close-Up",
                "Dutch Left (10°)",
                "Dutch Right (10°)",
            ),
        )
        self.assertEqual(
            camera_presets.camera_preset_spec("Medium Shot")["allowed_border_clipping"],
            ("bottom", "left", "right"),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported camera preset"):
            camera_presets.camera_preset_spec("Bird's-eye View")

    def test_camera_margin_resolves_before_cache_identity(self):
        self.assertEqual(
            camera_presets.resolve_camera_margin("Auto Director", 0.06),
            0.06,
        )
        self.assertEqual(
            camera_presets.resolve_camera_margin("American / Cowboy Shot", 0.06),
            0.08,
        )

    def test_shot_rays_selects_the_upper_region_per_frame(self):
        frames = [
            np.array([[-2.0, -4.0], [0.0, -4.0], [2.0, -4.0],
                      [-2.0, 6.0], [2.0, 6.0]]),
            np.array([[-3.0, -2.0], [0.0, -2.0], [3.0, -2.0],
                      [-3.0, 8.0], [3.0, 8.0]]),
        ]

        rays = build_driver.shot_rays(frames, 0.5)

        self.assertEqual(rays.shape, (10, 2))
        np.testing.assert_array_equal(rays[:5, 1], [-4.0, -4.0, -4.0, 1.0, 1.0])
        np.testing.assert_array_equal(rays[5:, 1], [-2.0, -2.0, -2.0, 3.0, 3.0])
        self.assertGreater(
            build_driver.minimum_safe_focal(704, 1280, 0.5),
            build_driver.minimum_safe_focal(704, 1280, 1.0),
        )

    def test_cropped_camera_honours_the_vertical_shot_line(self):
        rays = np.array([
            [-2.0, -0.5],
            [2.0, -0.5],
            [-0.25, 0.5],
            [0.25, 0.5],
        ])

        focal, principal = build_driver.fit_cropped_camera(rays, 704, 1280, 0.08)
        projected_y = rays[:, 1] * focal + principal[1]

        self.assertAlmostEqual(projected_y.min(), 1280 * 0.08)
        self.assertAlmostEqual(projected_y.max(), 1280 * 0.92)
        self.assertGreater(focal * (rays[:, 0].max() - rays[:, 0].min()), 704)

    def test_roll_is_identical_for_geometry_and_reference_reprojection(self):
        fit = type("Fit", (), {"image_size": (100, 80), "focal_length": 100.0})()
        principal = (60.0, 70.0)
        point = np.array([[10.0, -20.0, -100.0]])
        projected, _depth = render.project(
            point,
            np.zeros(3),
            100.0,
            200,
            200,
            principal,
            10.0,
        )

        with mock.patch.object(cv2, "warpAffine", return_value=np.zeros((200, 200, 3))) as warp:
            build_driver.reproject_reference(
                np.zeros((80, 100, 3), dtype=np.uint8),
                fit,
                100.0,
                principal,
                200,
                200,
                10.0,
            )
        matrix = warp.call_args.args[1]
        reference_pixel = np.array([60.0, 60.0, 1.0])

        np.testing.assert_allclose(matrix @ reference_pixel, projected[0], atol=1e-12)


class RunIdentityTests(unittest.TestCase):
    def test_run_identity_covers_inputs_settings_and_release_identity(self):
        settings = {"frames": 81, "fps": 16.0, "root_motion": "preserve"}
        baseline = run_all.run_identity("image", "fbx", settings, "1.3.0+commit")

        self.assertEqual(
            baseline,
            run_all.run_identity("image", "fbx", dict(settings), "1.3.0+commit"),
        )
        self.assertNotEqual(
            baseline,
            run_all.run_identity("other", "fbx", settings, "1.3.0+commit"),
        )
        self.assertNotEqual(
            baseline,
            run_all.run_identity(
                "image",
                "fbx",
                {**settings, "root_motion": "in_place"},
                "1.3.0+commit",
            ),
        )
        self.assertNotEqual(
            baseline,
            run_all.run_identity(
                "image",
                "fbx",
                {**settings, "camera_preset": "Medium Shot"},
                "1.3.0+commit",
            ),
        )
        self.assertNotEqual(
            baseline,
            run_all.run_identity("image", "fbx", settings, "1.3.1+other"),
        )
        self.assertNotEqual(
            baseline,
            run_all.run_identity(
                "image", "fbx", settings, "1.3.0+commit", "subject-mask"
            ),
        )

    def test_fit_cache_key_covers_release_identity_threshold_and_mask(self):
        baseline = run_all.fit_cache_key("image", "1.3.0+commit", 0.8)

        self.assertEqual(
            baseline,
            run_all.fit_cache_key("image", "1.3.0+commit", 0.8),
        )
        self.assertNotEqual(
            baseline,
            run_all.fit_cache_key("image", "1.3.1+other", 0.8),
        )
        self.assertNotEqual(
            baseline,
            run_all.fit_cache_key("image", "1.3.0+commit", 0.7),
        )
        self.assertNotEqual(
            baseline,
            run_all.fit_cache_key(
                "image", "1.3.0+commit", 0.8, "subject-mask"
            ),
        )

    def test_existing_run_must_match_its_full_identity(self):
        settings = {"frames": 81, "fps": 16.0}
        manifest = {
            "schema": "jakkanna.image_fbx_animation.v1",
            "run_id": "person-dance-abc",
            "inputs": {
                "image": {"sha256": "image"},
                "fbx": {"sha256": "fbx"},
            },
            "settings": settings,
            "software": {"jakkanna_release_identity": "1.3.0+commit"},
        }
        self.assertTrue(
            run_all.existing_run_matches(
                manifest, "person-dance-abc", "image", "fbx", settings, "1.3.0+commit"
            )
        )
        changed = json.loads(json.dumps(manifest))
        changed["inputs"]["fbx"]["sha256"] = "other"
        self.assertFalse(
            run_all.existing_run_matches(
                changed, "person-dance-abc", "image", "fbx", settings, "1.3.0+commit"
            )
        )
        changed = json.loads(json.dumps(manifest))
        changed["software"]["jakkanna_release_identity"] = "1.3.1+other"
        self.assertFalse(
            run_all.existing_run_matches(
                changed, "person-dance-abc", "image", "fbx", settings, "1.3.0+commit"
            )
        )
        changed = json.loads(json.dumps(manifest))
        changed["inputs"]["subject_mask"] = {"sha256": "subject-mask"}
        self.assertTrue(
            run_all.existing_run_matches(
                changed,
                "person-dance-abc",
                "image",
                "fbx",
                settings,
                "1.3.0+commit",
                "subject-mask",
            )
        )
        self.assertFalse(
            run_all.existing_run_matches(
                manifest,
                "person-dance-abc",
                "image",
                "fbx",
                settings,
                "1.3.0+commit",
                "subject-mask",
            )
        )

    def test_registry_archive_identity_uses_version_without_git(self):
        failed = type("Result", (), {"returncode": 128, "stdout": ""})()
        with mock.patch("_env.subprocess.run", return_value=failed):
            self.assertEqual(release_identity(), package_version())

    def test_source_checkout_identity_includes_version_and_commit(self):
        commit = "a" * 40
        successful = type("Result", (), {"returncode": 0, "stdout": f"{commit}\n"})()
        with mock.patch("_env.subprocess.run", return_value=successful):
            self.assertEqual(release_identity(), f"{package_version()}+{commit}")

    def test_safe_run_name_is_bounded_and_filesystem_safe(self):
        self.assertEqual(safe_name(" Luc & DANCE.fbx "), "luc-dance-fbx")
        self.assertEqual(len(safe_name("a" * 100)), 64)
        with self.assertRaises(ValueError):
            safe_name(" !!! ")

    def test_automatic_timing_only_fits_short_clips(self):
        self.assertEqual(audit_fbx.automatic_timing(4.0), "fit_clip")
        self.assertEqual(audit_fbx.automatic_timing(5.0), "original_speed")
        self.assertEqual(audit_fbx.automatic_timing(15.0), "original_speed")

    def test_sampling_times_uses_source_speed_or_fits_short_clip(self):
        class FakeRetargeter:
            def __init__(self, duration):
                self.clip = type("Clip", (), {"duration": duration})()

            def sample_times(self, frames, fps, start):
                return start + np.arange(frames) / fps

        long_times, long_policy = build_driver.sampling_times(
            FakeRetargeter(15.0), 81, 16.0, 0.0, "auto"
        )
        short_times, short_policy = build_driver.sampling_times(
            FakeRetargeter(2.0), 81, 16.0, 0.0, "auto"
        )
        self.assertEqual(long_policy, "original_speed")
        self.assertEqual(long_times[-1], 5.0)
        self.assertEqual(short_policy, "fit_clip")
        self.assertEqual(short_times[-1], 2.0)


class PreparedArtifactTests(unittest.TestCase):
    def test_rgb16_hash_matches_comfy_tensor_quantization(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.png"
            bgr = np.array([[[0, 127, 255], [255, 1, 0]]], dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), bgr))
            rgb = bgr[..., ::-1]
            expected = hashlib.sha256(
                (rgb.astype(np.dtype("<u2")) * 257).tobytes()
            ).hexdigest()
            self.assertEqual(build_driver.rgb16_sha256([path]), expected)

    def test_load_run_requires_scoped_schema_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_root = Path(temporary)
            run_dir = input_root / "person-dance-abc"
            run_dir.mkdir()
            for relative in ("driver", "driving_mask"):
                (run_dir / relative).mkdir()
            (run_dir / "registered_reference.png").touch()
            report = {
                "schema": "jakkanna.image_fbx_animation.v1",
                "run_id": run_dir.name,
                "settings": {"width": 704, "height": 1280},
            }
            (run_dir / "preprocess_manifest.json").write_text(json.dumps(report))

            with mock.patch.object(make_workflow, "INPUT_ROOT", input_root):
                loaded_dir, loaded_report = make_workflow.load_run(run_dir)
            self.assertEqual(loaded_dir, run_dir)
            self.assertEqual(loaded_report, report)

            report["run_id"] = "other"
            (run_dir / "preprocess_manifest.json").write_text(json.dumps(report))
            with mock.patch.object(make_workflow, "INPUT_ROOT", input_root):
                with self.assertRaisesRegex(ValueError, "run_id"):
                    make_workflow.load_run(run_dir)


class WorkflowContractTests(unittest.TestCase):
    def setUp(self):
        self.graph = make_workflow.workflow(
            "jakkanna_animate/person-dance-abc",
            '{"schema":"jakkanna.image_fbx_animation.v1"}',
            "jakkanna_animate/person-dance-abc",
            "",
            "person",
            42,
            704,
            1280,
        )

    def test_graph_uses_prepared_batches_without_tracking_or_batch_adapters(self):
        node_types = {node_type for node_type, _ in self.graph.values()}
        self.assertNotIn("SAM3_VideoTrack", node_types)
        self.assertNotIn("ImageListToImageBatch", node_types)
        # The prepared driver is already an 81-frame batch. Wrapping it in a VIDEO only
        # to have the raw-input gate re-derive a frame rate this graph typed in asserted
        # a constant; the prepared gate now publishes the canvas, rate, and report.
        self.assertNotIn("CreateVideo", {
            node_type for name, (node_type, _) in self.graph.items() if name.startswith("V")
        })
        self.assertNotIn("JakkannaSCAIL2ProductionInputValidate", node_types)
        self.assertEqual(self.graph["C1"][1]["width"], ["V3", 4])
        self.assertEqual(self.graph["C1"][1]["height"], ["V3", 5])
        self.assertEqual(self.graph["O4"][1]["input_report"], ["V3", 3])
        self.assertEqual(self.graph["O1"][1]["fps"], ["V3", 6])
        self.assertEqual(
            self.graph["C1"][1]["pose_video"],
            ["D1", 0],
        )
        self.assertEqual(
            self.graph["C1"][1]["pose_video_mask"],
            ["D2", 0],
        )
        self.assertFalse(self.graph["C1"][1]["replacement_mode"])
        self.assertEqual(self.graph["C1"][1]["length"], ["V3", 2])

    def test_graph_binds_preprocess_provenance_to_raw_inputs(self):
        manifest = self.graph["O4"][1]
        prepared_gate = self.graph["V3"][1]
        self.assertEqual(prepared_gate["preprocess_report"], ["P0", 0])
        self.assertEqual(prepared_gate["source_images"], ["D1", 0])
        self.assertEqual(prepared_gate["driving_mask_images"], ["D2", 0])
        self.assertEqual(manifest["preprocess_report"], ["P0", 0])
        self.assertEqual(manifest["source_images"], ["D1", 0])
        self.assertEqual(manifest["reference_image"], ["R2", 0])

    def test_live_contract_builder_distinguishes_widgets_and_forced_inputs(self):
        info = {
            "Example": {
                "input": {
                    "required": {
                        "literal": ("INT", {"default": 3}),
                        "connected": ("STRING", {"forceInput": True}),
                    },
                    "optional": {},
                },
                "output": ["STRING"],
                "output_name": ["value"],
            }
        }
        inputs, outputs = workflow_builder.node_spec(info, "Example")
        self.assertTrue(inputs[0]["widget"])
        self.assertFalse(inputs[1]["widget"])
        self.assertEqual(outputs, [{"name": "value", "type": "STRING"}])


class GreenfieldWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.graph = make_greenfield_workflow.greenfield_workflow(
            "ref_pic/luc.jpg",
            "dance.fbx",
            "preserve the subject",
            42,
        )

    def test_exposes_only_image_fbx_and_prompt_as_intended_controls(self):
        self.assertEqual(make_greenfield_workflow.VISIBLE, {"I1", "F1", "P1"})
        self.assertEqual(self.graph["I1"], ("LoadImage", {"image": "ref_pic/luc.jpg"}))
        self.assertEqual(self.graph["F1"][1]["fbx_file"], "dance.fbx")
        self.assertEqual(self.graph["P1"][1]["value"], "preserve the subject")

    def test_prepared_outputs_replace_all_run_directory_loaders(self):
        node_types = {node_type for node_type, _inputs in self.graph.values()}
        self.assertNotIn("VHS_LoadImages", node_types)
        self.assertEqual(self.graph["V3"][1]["source_images"], ["F1", 0])
        self.assertEqual(self.graph["V3"][1]["driving_mask_images"], ["F1", 1])
        self.assertEqual(self.graph["V3"][1]["registered_reference"], ["F1", 2])
        self.assertEqual(self.graph["V3"][1]["preprocess_report"], ["F1", 3])

    def test_sam31_mask_guides_the_prefit_without_a_user_control(self):
        self.assertEqual(self.graph["A1"][0], "SAM3_Detect")
        self.assertEqual(self.graph["A1"][1]["model"], ["S1", 0])
        self.assertEqual(self.graph["A1"][1]["image"], ["I1", 0])
        self.assertEqual(self.graph["A1"][1]["conditioning"], ["S2", 0])
        self.assertTrue(self.graph["A1"][1]["individual_masks"])
        self.assertEqual(self.graph["A2"][1]["masks"], ["A1", 0])
        self.assertEqual(self.graph["F1"][1]["subject_mask"], ["A2", 0])
        self.assertNotIn("A1", make_greenfield_workflow.VISIBLE)
        self.assertNotIn("A2", make_greenfield_workflow.VISIBLE)

    def test_prepared_validation_remains_on_sampling_dependency_path(self):
        self.assertEqual(self.graph["C1"][1]["length"], ["V3", 2])
        self.assertEqual(self.graph["C1"][1]["pose_video"], ["F1", 0])
        self.assertEqual(self.graph["C1"][1]["pose_video_mask"], ["F1", 1])

    def test_camera_preset_is_always_emitted_and_defaults_to_auto(self):
        # The node renders camera_preset as a widget from its own INPUT_TYPES, so
        # the control is visible whether or not the graph names it. Emitting it
        # explicitly keeps the shipped file honest about what the user can change.
        self.assertEqual(self.graph["F1"][1]["camera_preset"], camera_presets.AUTO_CAMERA_PRESET)
        self.assertEqual(
            make_greenfield_workflow.TITLES["F1"],
            "2–3 · FBX animation and camera preset",
        )
        self.assertEqual(make_greenfield_workflow.TITLES["P1"], "4 · Customization prompt")

    def test_camera_preset_can_be_baked_and_is_validated(self):
        graph = make_greenfield_workflow.greenfield_workflow(
            "ref_pic/luc.jpg",
            "dance.fbx",
            "preserve the subject",
            42,
            "American / Cowboy Shot",
        )
        self.assertEqual(graph["F1"][1]["camera_preset"], "American / Cowboy Shot")

        with self.assertRaisesRegex(ValueError, "Unsupported camera preset"):
            make_greenfield_workflow.greenfield_workflow(
                "ref_pic/luc.jpg", "dance.fbx", "", 42, "Bird's-eye View"
            )

    def test_registration_floor_is_wired_into_the_prepared_track_gate(self):
        self.assertEqual(self.graph["V3"][1]["minimum_registration_iou"], 0.35)


class PreparationChildProcessTests(unittest.TestCase):
    """The preparation child holds its own CUDA context, so it must never outlive the node."""

    def _spawn(self, body):
        process = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(body)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        self.addCleanup(process.stdout.close)
        return process

    def test_terminate_stops_a_running_child(self):
        process = self._spawn("import time; time.sleep(120)")
        image_fbx_animation._terminate(process)
        self.assertIsNotNone(process.poll())

    def test_terminate_escalates_when_the_child_ignores_sigterm(self):
        if os.name != "posix":
            self.skipTest("SIGTERM handling is POSIX-specific")
        process = self._spawn(
            """
            import signal, sys, time
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            print("ready", flush=True)
            time.sleep(120)
            """
        )
        self.assertEqual(process.stdout.readline().strip(), "ready")

        with mock.patch.object(image_fbx_animation, "_TERMINATE_GRACE_SECONDS", 0.5):
            image_fbx_animation._terminate(process)

        self.assertIsNotNone(process.poll())

    def test_terminate_is_a_no_op_for_a_child_that_already_exited(self):
        process = self._spawn("pass")
        process.wait()
        image_fbx_animation._terminate(process)
        self.assertEqual(process.returncode, 0)


class PreparationProgressTests(unittest.TestCase):
    class FakeProcess:
        """A child that emits `lines`, then either ends or keeps the pipe open."""

        def __init__(self, lines, block=False):
            self._lines = list(lines)
            self._block = block
            self.released = threading.Event()

        @property
        def stdout(self):
            def generate():
                for line in self._lines:
                    yield line
                if self._block:
                    self.released.wait(30)
            return generate()

    def test_mirrors_output_and_reports_rendered_frames(self):
        process = self.FakeProcess([
            "=== fit_reference.py --image ...\n",
            "[driver] clip 'dance.fbx' 6.20s\n",
            "  frame   0/81  t= 0.000s  coverage= 8.11%\n",
            "  frame  80/81  t= 5.000s  coverage= 9.02%\n",
        ])
        bar = image_fbx_animation.comfy.utils.ProgressBar(81)
        printed = []
        with mock.patch.object(
            image_fbx_animation.comfy.utils, "ProgressBar", return_value=bar
        ):
            with mock.patch("builtins.print", lambda *a, **k: printed.append(a[0])):
                image_fbx_animation._mirror_progress(process, 81)

        # Every line reaches the log, but only the frame lines advance the bar, and the
        # counter is 0-based on the wire.
        self.assertEqual(len(printed), 4)
        self.assertEqual(bar.updates, [(1, 81), (81, 81)])

    def test_cancellation_is_noticed_while_the_child_is_still_talking(self):
        process = self.FakeProcess(["  frame   0/81  t= 0.000s\n"], block=True)
        self.addCleanup(process.released.set)

        calls = []

        def interrupt_on_third_poll():
            calls.append(1)
            if len(calls) >= 3:
                raise KeyboardInterrupt("queue cancelled")

        with mock.patch.object(
            image_fbx_animation.comfy.model_management,
            "throw_exception_if_processing_interrupted",
            interrupt_on_third_poll,
        ):
            with mock.patch("builtins.print"):
                with self.assertRaises(KeyboardInterrupt):
                    image_fbx_animation._mirror_progress(process, 81)

    def test_drain_always_signals_end_of_stream(self):
        import queue as queue_module

        sink = queue_module.Queue()
        image_fbx_animation._drain(iter(["one\n", "two\n"]), sink)
        self.assertEqual(sink.get_nowait(), "one\n")
        self.assertEqual(sink.get_nowait(), "two\n")
        self.assertIsNone(sink.get_nowait())

        def exploding():
            yield "one\n"
            raise ValueError("pipe closed under us")

        sink = queue_module.Queue()
        image_fbx_animation._drain(exploding(), sink)
        self.assertEqual(sink.get_nowait(), "one\n")
        self.assertIsNone(sink.get_nowait())


if __name__ == "__main__":
    unittest.main()
