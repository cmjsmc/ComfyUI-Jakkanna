import json
import importlib
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(ROOT))
for path in (ROOT, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from comfy.ldm.sam3.tracker import pack_masks

package = types.ModuleType("ComfyUI_Jakkanna")
package.__path__ = [ROOT]
sys.modules.setdefault(package.__name__, package)
nodes_package = types.ModuleType("ComfyUI_Jakkanna.nodes")
nodes_package.__path__ = [os.path.join(ROOT, "nodes")]
sys.modules.setdefault(nodes_package.__name__, nodes_package)
scail2_production = importlib.import_module("ComfyUI_Jakkanna.nodes.scail2_production")


class TensorVideo:
    def __init__(self, images, frame_rate=16):
        self.components = SimpleNamespace(images=images, frame_rate=frame_rate, audio=None)

    def get_components(self):
        return self.components


class ScailInputValidationTests(unittest.TestCase):
    def test_validates_raw_tensor_components_without_video_encoding(self):
        images = torch.zeros((1, 1, 1, 3), dtype=torch.uint8).expand(81, 1280, 704, 3)
        reference = torch.zeros((1, 1, 1, 3), dtype=torch.uint8).expand(1, 1280, 704, 3)
        validator = scail2_production.JakkannaSCAIL2ProductionInputValidate()

        with mock.patch.object(scail2_production, "_hash_tensor", return_value="raw-hash"):
            width, height, frames, fps, report_json = validator.validate(
                TensorVideo(images),
                reference,
            )

        report = json.loads(report_json)
        self.assertEqual((width, height, frames, fps), (704, 1280, 81, 16.0))
        self.assertEqual(report["source_kind"], "image_tensor_batch")
        self.assertEqual(report["source_rgb16_sha256"], "raw-hash")
        self.assertNotIn("source_sha256", report)
        self.assertNotIn("decoded_rgb8_sha256", report)

    def test_rejects_noncanonical_tensor_shape_before_hashing(self):
        validator = scail2_production.JakkannaSCAIL2ProductionInputValidate()
        with self.assertRaisesRegex(ValueError, "exactly 81 frames"):
            validator.validate(
                TensorVideo(torch.zeros((2, 32, 32, 3))),
                torch.zeros((1, 32, 32, 3)),
            )

    def test_rejects_nonfinite_reference(self):
        images = torch.zeros((1, 1, 1, 3)).expand(81, 1280, 704, 3)
        reference = torch.zeros((1, 1280, 704, 3))
        reference[0, 0, 0, 0] = torch.nan
        validator = scail2_production.JakkannaSCAIL2ProductionInputValidate()

        with self.assertRaisesRegex(ValueError, "non-finite"):
            validator.validate(TensorVideo(images), reference)


class ScailTrackValidationTests(unittest.TestCase):
    def _track(self, masks):
        return {
            "packed_masks": pack_masks(masks),
            "orig_size": masks.shape[-2:],
        }

    def test_rejects_subject_at_canvas_edge(self):
        masks = torch.zeros((3, 1, 24, 24), dtype=torch.bool)
        masks[:, :, 4:20, 0:8] = True
        reference = torch.zeros((1, 24, 24))
        reference[:, 4:20, 4:20] = 1

        with self.assertRaisesRegex(ValueError, "canvas edge"):
            scail2_production.JakkannaSCAIL2TrackValidate().validate(
                self._track(masks),
                reference,
                3,
                2.0,
                0.15,
                0.1,
                0.01,
            )

    def test_reports_subject_border_margin(self):
        masks = torch.zeros((3, 1, 24, 24), dtype=torch.bool)
        masks[:, :, 4:20, 4:20] = True
        reference = torch.zeros((1, 24, 24))
        reference[:, 4:20, 4:20] = 1

        _indices, report_json = scail2_production.JakkannaSCAIL2TrackValidate().validate(
            self._track(masks),
            reference,
            3,
            2.0,
            0.15,
            0.1,
            0.01,
        )

        self.assertAlmostEqual(json.loads(report_json)["minimum_border_margin"], 4 / 24)

    def test_validates_reference_against_original_canvas_not_internal_masks(self):
        masks = torch.zeros((3, 1, 24, 24), dtype=torch.bool)
        masks[:, :, 4:20, 4:20] = True
        track = self._track(masks)
        track["orig_size"] = (32, 18)
        reference = torch.zeros((1, 32, 18))
        reference[:, 4:28, 3:15] = 1

        _indices, report_json = scail2_production.JakkannaSCAIL2TrackValidate().validate(
            track,
            reference,
            3,
            2.0,
            0.15,
            0.1,
            0.01,
        )

        report = json.loads(report_json)
        self.assertEqual(report["tracking_mask_resolution"], [24, 24])
        self.assertEqual(report["source_canvas_resolution"], [18, 32])


class PreparedTrackValidationTests(unittest.TestCase):
    def _inputs(self):
        source = torch.zeros((3, 24, 24, 3))
        driving_masks = torch.zeros_like(source)
        driving_masks[:, 4:20, 4:20, 2] = 1
        registered_reference = torch.zeros((1, 24, 24, 3))
        reference_mask = torch.zeros((1, 24, 24))
        reference_mask[:, 4:20, 4:20] = 1
        report = {
            "schema": "jakkanna.image_fbx_animation.v1",
            "run_id": "test-run",
            "registration": {"frame0_subject_iou_visible": 0.9},
            "settings": {"frames": 3, "fps": 16.0, "width": 24, "height": 24},
            "artifacts": {
                "driver": {
                    "rgb16_sha256": scail2_production._hash_tensor(source),
                },
                "driving_mask": {
                    "rgb16_sha256": scail2_production._hash_tensor(driving_masks),
                },
                "registered_reference": {
                    "rgb16_sha256": scail2_production._hash_tensor(registered_reference),
                },
            },
        }
        return source, driving_masks, registered_reference, reference_mask, json.dumps(report)

    def test_validates_exact_prepared_palette_and_provenance(self):
        source, driving_masks, registered_reference, reference_mask, report = self._inputs()
        initial_json, track_json, frames = scail2_production.JakkannaSCAIL2PreparedTrackValidate().validate(
            source,
            driving_masks,
            registered_reference,
            reference_mask,
            3,
            report,
            2.0,
            0.15,
            0.1,
            0.01,
        )

        initial = json.loads(initial_json)
        track = json.loads(track_json)
        self.assertEqual(initial["source"], "prepared_render")
        self.assertEqual(track["source"], "prepared_render")
        self.assertEqual(track["preprocess_run_id"], "test-run")
        self.assertEqual(track["source_canvas_resolution"], [24, 24])
        self.assertEqual(frames, 3)

    def test_rejects_non_scail_palette(self):
        source, driving_masks, registered_reference, reference_mask, report = self._inputs()
        driving_masks[0, 0, 0, 0] = 1

        with self.assertRaisesRegex(ValueError, "only black and RGB blue"):
            scail2_production.JakkannaSCAIL2PreparedTrackValidate().validate(
                source,
                driving_masks,
                registered_reference,
                reference_mask,
                3,
                report,
                2.0,
                0.15,
                0.1,
                0.01,
            )

    def test_rejects_preprocessing_hash_mismatch(self):
        source, driving_masks, registered_reference, reference_mask, report_json = self._inputs()
        report = json.loads(report_json)
        report["artifacts"]["driving_mask"]["rgb16_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "do not match"):
            scail2_production.JakkannaSCAIL2PreparedTrackValidate().validate(
                source,
                driving_masks,
                registered_reference,
                reference_mask,
                3,
                json.dumps(report),
                2.0,
                0.15,
                0.1,
                0.01,
            )

    def test_rejects_poor_visible_subject_registration(self):
        source, driving_masks, registered_reference, reference_mask, report_json = self._inputs()
        report = json.loads(report_json)
        report["registration"]["frame0_subject_iou_visible"] = 0.4

        with self.assertRaisesRegex(ValueError, "visible-subject IoU"):
            scail2_production.JakkannaSCAIL2PreparedTrackValidate().validate(
                source,
                driving_masks,
                registered_reference,
                reference_mask,
                3,
                json.dumps(report),
                2.0,
                0.15,
                0.1,
                0.01,
            )


class ManifestHelperTests(unittest.TestCase):
    def test_git_commit_supports_packed_refs(self):
        with tempfile.TemporaryDirectory() as directory:
            git_dir = os.path.join(directory, ".git")
            os.makedirs(git_dir)
            with open(os.path.join(git_dir, "HEAD"), "w", encoding="utf-8") as handle:
                handle.write("ref: refs/heads/main\n")
            with open(os.path.join(git_dir, "packed-refs"), "w", encoding="utf-8") as handle:
                handle.write("1" * 40 + " refs/heads/main\n")

            self.assertEqual(scail2_production._git_commit(directory), "1" * 40)

    def test_graph_settings_record_only_literal_inputs(self):
        prompt = {
            "1": {
                "class_type": "BasicScheduler",
                "inputs": {"scheduler": "simple", "steps": 6, "denoise": 1.0, "model": ["2", 0]},
            },
            "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "model.safetensors"}},
        }

        self.assertEqual(
            scail2_production._graph_settings(prompt),
            [{
                "node_id": "1",
                "class_type": "BasicScheduler",
                "inputs": {"scheduler": "simple", "steps": 6, "denoise": 1.0},
            }],
        )

    def test_graph_settings_record_flow_unipc_topology(self):
        prompt = {
            "42": {
                "class_type": "JakkannaSCAIL2FlowUniPC",
                "inputs": {"steps": 40, "shift": 3.0},
            },
        }

        self.assertEqual(
            scail2_production._graph_settings(prompt),
            [{
                "node_id": "42",
                "class_type": "JakkannaSCAIL2FlowUniPC",
                "inputs": {"steps": 40, "shift": 3.0},
            }],
        )


if __name__ == "__main__":
    unittest.main()
