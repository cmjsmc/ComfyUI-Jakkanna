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


def pack_masks(masks):
    binary = masks > 0
    shifts = torch.arange(8, device=masks.device)
    return (binary.view(*masks.shape[:-1], -1, 8) * (1 << shifts)).sum(-1).byte()


def load_scail2_production_module():
    package = types.ModuleType("ComfyUI_Jakkanna")
    package.__path__ = [ROOT]
    sys.modules[package.__name__] = package
    nodes_package = types.ModuleType("ComfyUI_Jakkanna.nodes")
    nodes_package.__path__ = [os.path.join(ROOT, "nodes")]
    sys.modules[nodes_package.__name__] = nodes_package

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.folder_names_and_paths = {}
    folder_paths.get_temp_directory = tempfile.gettempdir
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    model_sampling = types.ModuleType("comfy.model_sampling")
    samplers = types.ModuleType("comfy.samplers")
    comfy.model_management = model_management
    comfy.model_sampling = model_sampling
    comfy.samplers = samplers
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy.model_sampling"] = model_sampling
    sys.modules["comfy.samplers"] = samplers

    ldm = types.ModuleType("comfy.ldm")
    ldm.__path__ = []
    sam3 = types.ModuleType("comfy.ldm.sam3")
    sam3.__path__ = []
    tracker = types.ModuleType("comfy.ldm.sam3.tracker")

    def unpack_masks(packed):
        bits = torch.tensor(
            [1, 2, 4, 8, 16, 32, 64, 128],
            dtype=torch.uint8,
            device=packed.device,
        )
        return (packed.unsqueeze(-1) & bits).bool().view(*packed.shape[:-1], -1)

    tracker.unpack_masks = unpack_masks
    sys.modules["comfy.ldm"] = ldm
    sys.modules["comfy.ldm.sam3"] = sam3
    sys.modules["comfy.ldm.sam3.tracker"] = tracker

    flow_unipc = types.ModuleType("ComfyUI_Jakkanna.nodes.scail2_flow_unipc")
    flow_unipc.FlowUniPCMultistepScheduler = object
    sys.modules[flow_unipc.__name__] = flow_unipc
    return importlib.import_module("ComfyUI_Jakkanna.nodes.scail2_production")


scail2_production = load_scail2_production_module()


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
            "settings": {
                "frames": 3,
                "fps": 16.0,
                "width": 24,
                "height": 24,
                "timing": "auto",
                "resolved_timing": "fit_clip",
                "playback_speed_ratio": 0.4,
                "root_motion": "auto",
                "resolved_root_motion": "in_place",
            },
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

    def _set_camera_preset(self, driving_masks, report_json, preset, allowed_edges):
        report = json.loads(report_json)
        report["settings"]["camera_preset"] = preset
        report["registration"]["camera_preset"] = {
            "requested": preset,
            "vertical_fraction": 0.58,
            "roll_degrees": 0.0,
            "allowed_border_clipping": allowed_edges,
        }
        report["artifacts"]["driving_mask"]["rgb16_sha256"] = (
            scail2_production._hash_tensor(driving_masks)
        )
        return json.dumps(report)

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

    def test_rejects_a_body_fit_that_does_not_land_on_the_subject(self):
        source, driving_masks, registered_reference, reference_mask, report_json = self._inputs()
        report = json.loads(report_json)
        report["registration"]["frame0_subject_iou_visible"] = 0.01

        with self.assertRaisesRegex(ValueError, "registration IoU is 0.010"):
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

    def test_registration_floor_admits_occlusion_and_can_be_disabled(self):
        source, driving_masks, registered_reference, reference_mask, report_json = self._inputs()
        report = json.loads(report_json)
        # Clothing and occlusion legitimately depress the score; the floor is
        # loose enough that only a gross fit failure trips it.
        report["registration"]["frame0_subject_iou_visible"] = 0.5

        _initial, track_json, frames = (
            scail2_production.JakkannaSCAIL2PreparedTrackValidate().validate(
                source, driving_masks, registered_reference, reference_mask,
                3, json.dumps(report), 2.0, 0.15, 0.1, 0.01,
            )
        )
        track = json.loads(track_json)
        self.assertEqual(track["preprocess_run_id"], "test-run")
        self.assertEqual(track["frame0_subject_iou_visible"], 0.5)
        self.assertEqual(track["minimum_registration_iou"], 0.35)
        self.assertEqual(frames, 3)

        report["registration"]["frame0_subject_iou_visible"] = 0.01
        _initial, track_json, _frames = (
            scail2_production.JakkannaSCAIL2PreparedTrackValidate().validate(
                source, driving_masks, registered_reference, reference_mask,
                3, json.dumps(report), 2.0, 0.15, 0.1, 0.01, 0.0,
            )
        )
        self.assertEqual(json.loads(track_json)["frame0_subject_iou_visible"], 0.01)

    def test_rejects_a_report_with_no_registration_score(self):
        source, driving_masks, registered_reference, reference_mask, report_json = self._inputs()
        report = json.loads(report_json)
        del report["registration"]["frame0_subject_iou_visible"]

        with self.assertRaisesRegex(ValueError, "no frame-zero registration IoU"):
            scail2_production.JakkannaSCAIL2PreparedTrackValidate().validate(
                source, driving_masks, registered_reference, reference_mask,
                3, json.dumps(report), 2.0, 0.15, 0.1, 0.01,
            )

    def test_surfaces_the_timing_and_root_motion_preparation_resolved(self):
        source, driving_masks, registered_reference, reference_mask, report = self._inputs()

        _initial, track_json, _frames = (
            scail2_production.JakkannaSCAIL2PreparedTrackValidate().validate(
                source, driving_masks, registered_reference, reference_mask,
                3, report, 2.0, 0.15, 0.1, 0.01,
            )
        )

        track = json.loads(track_json)
        # Preparation was asked for "auto" and silently chose in-place motion and
        # a stretched clip. Both decisions have to reach the operator.
        self.assertEqual(track["root_motion_requested"], "auto")
        self.assertEqual(track["root_motion_resolved"], "in_place")
        self.assertEqual(track["timing_resolved"], "fit_clip")
        self.assertEqual(track["playback_speed_ratio"], 0.4)
        self.assertEqual(track["camera_preset"], "Auto Director")

    def test_medium_shot_allows_declared_bottom_and_side_clipping(self):
        source, driving_masks, registered_reference, reference_mask, report = self._inputs()
        driving_masks.zero_()
        driving_masks[:, 4:24, 0:20, 2] = 1
        report = self._set_camera_preset(
            driving_masks, report, "Medium Shot", ["bottom", "left", "right"]
        )

        _initial, track_json, _frames = (
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
        )

        self.assertEqual(
            json.loads(track_json)["allowed_border_clipping"],
            ["bottom", "left", "right"],
        )

    def test_medium_shot_still_rejects_top_clipping(self):
        source, driving_masks, registered_reference, reference_mask, report = self._inputs()
        driving_masks.zero_()
        driving_masks[:, 0:20, 4:20, 2] = 1
        report = self._set_camera_preset(
            driving_masks, report, "Medium Shot", ["bottom", "left", "right"]
        )

        with self.assertRaisesRegex(ValueError, "canvas edge"):
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

    def test_rejects_camera_preset_provenance_that_expands_crop_edges(self):
        source, driving_masks, registered_reference, reference_mask, report = self._inputs()
        report = self._set_camera_preset(
            driving_masks, report, "Medium Shot", ["top", "bottom", "left", "right"]
        )

        with self.assertRaisesRegex(ValueError, "invalid border-clipping provenance"):
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
