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

import build_driver
import make_workflow
import run_all
import workflow_builder
from _env import safe_name


class RunIdentityTests(unittest.TestCase):
    def test_run_identity_covers_inputs_settings_and_commit(self):
        settings = {"frames": 81, "fps": 16.0, "root_motion": "preserve"}
        baseline = run_all.run_identity("image", "fbx", settings, "commit")

        self.assertEqual(baseline, run_all.run_identity("image", "fbx", dict(settings), "commit"))
        self.assertNotEqual(baseline, run_all.run_identity("other", "fbx", settings, "commit"))
        self.assertNotEqual(
            baseline,
            run_all.run_identity("image", "fbx", {**settings, "root_motion": "in_place"}, "commit"),
        )
        self.assertNotEqual(baseline, run_all.run_identity("image", "fbx", settings, "other"))

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
            "software": {"jakkanna_commit": "commit"},
        }
        self.assertTrue(
            run_all.existing_run_matches(
                manifest, "person-dance-abc", "image", "fbx", settings, "commit"
            )
        )
        changed = json.loads(json.dumps(manifest))
        changed["inputs"]["fbx"]["sha256"] = "other"
        self.assertFalse(
            run_all.existing_run_matches(
                changed, "person-dance-abc", "image", "fbx", settings, "commit"
            )
        )

    def test_safe_run_name_is_bounded_and_filesystem_safe(self):
        self.assertEqual(safe_name(" Luc & DANCE.fbx "), "luc-dance-fbx")
        self.assertEqual(len(safe_name("a" * 100)), 64)
        with self.assertRaises(ValueError):
            safe_name(" !!! ")


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


if __name__ == "__main__":
    unittest.main()
