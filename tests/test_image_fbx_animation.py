import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "scail2_animate"
sys.path.insert(0, str(TOOLS))

import audit_fbx
import build_driver
import camera_presets
import make_camera_preset_workflow
import make_greenfield_workflow
import make_workflow
import render
import run_all
import workflow_builder
from _env import package_version, release_identity, safe_name


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
            ("bottom",),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported camera preset"):
            camera_presets.camera_preset_spec("Bird's-eye View")

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

    def test_fit_cache_key_covers_release_identity_and_threshold(self):
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
        self.assertEqual(self.graph["V1"][1]["images"], ["F1", 0])
        self.assertEqual(self.graph["V3"][1]["driving_mask_images"], ["F1", 1])
        self.assertEqual(self.graph["V3"][1]["registered_reference"], ["F1", 2])
        self.assertEqual(self.graph["V3"][1]["preprocess_report"], ["F1", 3])

    def test_prepared_validation_remains_on_sampling_dependency_path(self):
        self.assertEqual(self.graph["C1"][1]["length"], ["V3", 2])
        self.assertEqual(self.graph["C1"][1]["pose_video"], ["F1", 0])
        self.assertEqual(self.graph["C1"][1]["pose_video_mask"], ["F1", 1])

    def test_camera_preset_workflow_exposes_the_fourth_control(self):
        graph = make_camera_preset_workflow.camera_preset_workflow(
            "ref_pic/luc.jpg",
            "dance.fbx",
            "American / Cowboy Shot",
            "preserve the subject",
            42,
        )

        self.assertEqual(
            make_camera_preset_workflow.VISIBLE,
            {"I1", "F1", "P1"},
        )
        self.assertEqual(
            graph["F1"][1]["camera_preset"],
            "American / Cowboy Shot",
        )
        self.assertEqual(
            make_camera_preset_workflow.TITLES["P1"],
            "4 · Customization prompt",
        )


if __name__ == "__main__":
    unittest.main()
