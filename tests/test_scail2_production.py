import json
import importlib
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
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
    folder_paths.get_user_directory = tempfile.gettempdir
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    model_sampling = types.ModuleType("comfy.model_sampling")
    samplers = types.ModuleType("comfy.samplers")
    # Faithful reimplementation of the two upstream pieces the sigma check depends on.
    # SamplerGateTests pins it against values read off a live ComfyUI, so a drift in
    # either direction shows up as a test failure rather than a silently weaker check.
    class ModelSamplingDiscreteFlow:
        def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1000):
            self.shift = shift
            t = torch.arange(1, timesteps + 1, 1) / timesteps
            self.sigmas = shift * t / (1 + (shift - 1) * t)

    def simple_scheduler(model_sampling_, steps):
        s = model_sampling_.sigmas
        ss = len(s) / steps
        sigs = [float(s[-(1 + int(x * ss))]) for x in range(steps)]
        sigs.append(0.0)
        return torch.FloatTensor(sigs)

    model_sampling.ModelSamplingDiscreteFlow = ModelSamplingDiscreteFlow
    samplers.simple_scheduler = simple_scheduler

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
    def setUp(self):
        # The track metrics are checked on a 24x24 canvas so the mask geometry stays
        # readable. Declare that canvas rather than relaxing the production whitelist.
        patch = mock.patch.object(scail2_production, "_SCAIL2_CANVASES", {(24, 24)})
        patch.start()
        self.addCleanup(patch.stop)

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
        initial_json, track_json, frames, *_ = scail2_production.JakkannaSCAIL2PreparedTrackValidate().validate(
            source,
            driving_masks,
            registered_reference,
            reference_mask,
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

        _initial, track_json, frames, *_ = (
            scail2_production.JakkannaSCAIL2PreparedTrackValidate().validate(
                source, driving_masks, registered_reference, reference_mask,
                json.dumps(report), 2.0, 0.15, 0.1, 0.01,
            )
        )
        track = json.loads(track_json)
        self.assertEqual(track["preprocess_run_id"], "test-run")
        self.assertEqual(track["frame0_subject_iou_visible"], 0.5)
        self.assertEqual(track["minimum_registration_iou"], 0.35)
        self.assertEqual(frames, 3)

        report["registration"]["frame0_subject_iou_visible"] = 0.01
        _initial, track_json, _frames, *_ = (
            scail2_production.JakkannaSCAIL2PreparedTrackValidate().validate(
                source, driving_masks, registered_reference, reference_mask,
                json.dumps(report), 2.0, 0.15, 0.1, 0.01, 0.0,
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
                json.dumps(report), 2.0, 0.15, 0.1, 0.01,
            )

    def test_surfaces_the_timing_and_root_motion_preparation_resolved(self):
        source, driving_masks, registered_reference, reference_mask, report = self._inputs()

        _initial, track_json, _frames, *_ = (
            scail2_production.JakkannaSCAIL2PreparedTrackValidate().validate(
                source, driving_masks, registered_reference, reference_mask,
                report, 2.0, 0.15, 0.1, 0.01,
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

        _initial, track_json, _frames, *_ = (
            scail2_production.JakkannaSCAIL2PreparedTrackValidate().validate(
                source,
                driving_masks,
                registered_reference,
                reference_mask,
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
                report,
                2.0,
                0.15,
                0.1,
                0.01,
            )


class PreparedInputContractTests(unittest.TestCase):
    """The prepared gate replaces the CreateVideo -> InputValidate round trip, so it has
    to publish the canvas, rate, and input report the rest of the graph consumed."""

    def setUp(self):
        patch = mock.patch.object(scail2_production, "_SCAIL2_CANVASES", {(24, 24)})
        patch.start()
        self.addCleanup(patch.stop)
        self.inputs = PreparedTrackValidationTests._inputs(self)

    def _validate(self, **kwargs):
        source, driving_masks, registered_reference, reference_mask, report = self.inputs
        return scail2_production.JakkannaSCAIL2PreparedTrackValidate().validate(
            source, driving_masks, registered_reference, reference_mask,
            report, 2.0, 0.15, 0.1, 0.01, **kwargs,
        )

    def test_publishes_canvas_rate_and_input_report(self):
        _initial, _track, frames, input_json, width, height, fps = self._validate()

        self.assertEqual((width, height, frames, fps), (24, 24, 3, 16.0))
        report = json.loads(input_json)
        self.assertEqual(report["source_kind"], "prepared_image_batch")
        self.assertEqual(report["fps"], "16/1")
        self.assertEqual(report["canonical_fps_enforced"], 16)
        self.assertEqual(report["frames"], 3)
        self.assertEqual(report["preprocess_run_id"], "test-run")
        # The manifest reconciles its source digest against this field.
        self.assertEqual(
            report["source_rgb16_sha256"],
            json.loads(_track)["source_rgb16_sha256"],
        )

    def test_expected_frames_is_optional_but_still_cross_checks(self):
        _i, _t, frames, *_ = self._validate(expected_frames=3)
        self.assertEqual(frames, 3)

        with self.assertRaisesRegex(ValueError, "expected 7"):
            self._validate(expected_frames=7)

    def test_rejects_a_canvas_scail2_was_not_trained_on(self):
        # The production whitelist, not the one setUp declares for the metric fixtures.
        with mock.patch.object(
            scail2_production, "_SCAIL2_CANVASES", {(704, 1280), (1280, 704)}
        ):
            with self.assertRaisesRegex(ValueError, "704x1280 or 1280x704, got 24x24"):
                self._validate()


class ManifestDigestReuseTests(unittest.TestCase):
    """Every consumer of a link receives the same tensor object, so re-hashing the source
    and driving mask in the manifest cannot detect anything the prepared gate missed - it
    only re-reads several hundred megabytes."""

    SOURCE_DIGEST = "source-digest"
    MASK_DIGEST = "mask-digest"

    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.home, True)
        models = os.path.join(self.home, "models")
        os.makedirs(models)
        for name in ("unet", "lora", "sam", "clip", "vision", "vae"):
            with open(os.path.join(models, name), "wb") as handle:
                handle.write(b"x")

        folder_paths = scail2_production.folder_paths
        stubs = {
            "get_full_path": lambda category, name: os.path.join(models, name),
            "get_output_directory": lambda: self.home,
            "get_save_image_path": lambda prefix, out, w, h: (self.home, "manifest", 1, "", prefix),
            "__file__": os.path.join(self.home, "folder_paths.py"),
        }
        for name, value in stubs.items():
            patch = mock.patch.object(folder_paths, name, value, create=True)
            patch.start()
            self.addCleanup(patch.stop)

        mm = scail2_production.comfy.model_management
        for name in ("sage_attention_enabled", "flash_attention_enabled",
                     "xformers_enabled", "pytorch_attention_enabled"):
            patch = mock.patch.object(mm, name, lambda: False, create=True)
            patch.start()
            self.addCleanup(patch.stop)

    def _save(self, track_report, **overrides):
        images = torch.zeros((1, 8, 8, 3))
        kwargs = dict(
            output_images=images,
            pose_video_mask=images,
            reference_image_mask=images,
            input_report=json.dumps({"source_rgb16_sha256": self.SOURCE_DIGEST}),
            reference_report=json.dumps({"output_rgb16_sha256": "reference-digest"}),
            source_initial_mask_report="{}",
            reference_mask_report="{}",
            track_report=track_report,
            noise_report=json.dumps({"seed": 7}),
            positive_prompt="",
            negative_prompt="",
            seed=7,
            diffusion_model="unet",
            dpo_lora="lora",
            sam_checkpoint="sam",
            text_encoder="clip",
            clip_vision="vision",
            vae="vae",
            attention_backend="PyTorch SDPA",
            hash_model_files=False,
            filename_prefix="test/manifest",
            source_images=images,
        )
        kwargs.update(overrides)
        result = scail2_production.JakkannaSCAIL2ProductionManifest().save(**kwargs)
        with open(result["result"][0], encoding="utf-8") as handle:
            return json.load(handle)

    def test_prepared_digests_are_carried_forward_not_recomputed(self):
        prepared = json.dumps({
            "source_rgb16_sha256": self.SOURCE_DIGEST,
            "driving_mask_rgb16_sha256": self.MASK_DIGEST,
        })
        hashed = []
        real_hash = scail2_production._hash_tensor

        def recording_hash(tensor, binary=False):
            hashed.append(tuple(tensor.shape))
            return real_hash(tensor, binary)

        with mock.patch.object(scail2_production, "_hash_tensor", recording_hash):
            manifest = self._save(prepared)

        self.assertEqual(manifest["source_capture_rgb16_sha256"], self.SOURCE_DIGEST)
        self.assertEqual(manifest["pose_video_mask_rgb16_sha256"], self.MASK_DIGEST)
        # Only the decoded output and the reference mask are still hashed here.
        self.assertEqual(len(hashed), 2)

    def test_falls_back_to_hashing_when_the_track_carries_no_digest(self):
        # The non-prepared path - JakkannaSCAIL2TrackValidate - publishes no rgb16
        # digests, so the manifest must still derive them from the tensors it was given.
        real = scail2_production._hash_tensor(torch.zeros((1, 8, 8, 3))[..., :3])

        manifest = self._save(
            json.dumps({"frames": 81}),
            input_report=json.dumps({"source_rgb16_sha256": real}),
        )

        self.assertEqual(manifest["source_capture_rgb16_sha256"], real)
        self.assertNotEqual(manifest["pose_video_mask_rgb16_sha256"], self.MASK_DIGEST)

    def test_reports_that_disagree_on_the_source_digest_are_rejected(self):
        prepared = json.dumps({
            "source_rgb16_sha256": "a-different-digest",
            "driving_mask_rgb16_sha256": self.MASK_DIGEST,
        })
        with self.assertRaisesRegex(ValueError, "does not match the validated production input"):
            self._save(prepared)


class SamplerGateTests(unittest.TestCase):
    """Every mismatch the manifest used to report after VAE decode is knowable from the
    prompt graph and the sigma schedule, so the gate has to catch it before sampling."""

    def _prompt(self, **overrides):
        prompt = {
            "1": {"class_type": "BasicScheduler",
                  "inputs": {"scheduler": "simple", "steps": 6, "denoise": 1.0}},
            "2": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "3": {"class_type": "CFGGuider", "inputs": {"cfg": 1.0}},
            "4": {"class_type": "WanSCAILToVideo",
                  "inputs": {"replacement_mode": False, "pose_strength": 1.0,
                             "pose_start": 0.0, "pose_end": 1.0}},
            "5": {"class_type": "CLIPVisionEncode", "inputs": {"crop": "none"}},
            "6": {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"lora_name": "dpo.safetensors", "strength_model": 1.0}},
            "7": {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"lora_name": "accel.safetensors", "strength_model": 0.8}},
            "8": {"class_type": "JakkannaSCAIL2ProductionManifest",
                  "inputs": {"steps": 6, "cfg": 1.0, "mode": "animation",
                             "sampler_name": "Euler",
                             "scheduler_name": "simple (shift-8 flow schedule)",
                             "flow_shift": 8.0, "effective_flow_shift": 0.0,
                             "dpo_lora": "dpo.safetensors",
                             "acceleration_lora": "accel.safetensors",
                             "acceleration_lora_strength": 0.8}},
        }
        for node_id, inputs in overrides.items():
            prompt[node_id]["inputs"].update(inputs)
        return prompt

    def _sigmas(self, shift=8.0, steps=6):
        return scail2_production._simple_flow_sigmas(shift, steps)

    def test_schedule_matches_a_live_comfyui(self):
        # Read off a running ComfyUI for WAN21_SCAIL2, whose sampling_settings inherit
        # WAN21_T2V shift=8.0. If upstream retunes that default this is where it shows.
        self.assertEqual(
            [round(float(s), 6) for s in self._sigmas()],
            [1.0, 0.975724, 0.941259, 0.888889, 0.800479, 0.615952, 0.0],
        )

    def test_passes_a_consistent_graph_through_untouched(self):
        gate = scail2_production.JakkannaSCAIL2SamplerValidate()
        sigmas = self._sigmas()

        passed, report_json = gate.validate(sigmas, self._prompt())

        self.assertIs(passed, sigmas)
        report = json.loads(report_json)
        self.assertTrue(report["validated_before_sampling"])
        self.assertEqual(report["steps"], 6)
        self.assertEqual(report["mode"], "animation")
        self.assertEqual(report["dpo_lora_strength"], 1.0)
        self.assertEqual(report["effective_flow_shift"], 8.0)
        self.assertEqual(report["sigma_schedule"][0], 1.0)
        self.assertEqual(report["sigma_schedule"][-1], 0.0)

    def test_rejects_a_schedule_built_at_the_wrong_shift(self):
        gate = scail2_production.JakkannaSCAIL2SamplerValidate()
        with self.assertRaisesRegex(ValueError, "not the Euler/simple flow schedule"):
            gate.validate(self._sigmas(shift=5.0), self._prompt())

    def test_rejects_a_graph_whose_cfg_drifted_from_the_manifest(self):
        gate = scail2_production.JakkannaSCAIL2SamplerValidate()
        with self.assertRaisesRegex(ValueError, "does not match CFGGuider"):
            gate.validate(self._sigmas(), self._prompt(**{"3": {"cfg": 5.0}}))

    def test_rejects_replacement_mode_declared_as_animation(self):
        gate = scail2_production.JakkannaSCAIL2SamplerValidate()
        with self.assertRaisesRegex(ValueError, "does not match WanSCAILToVideo mode"):
            gate.validate(self._sigmas(), self._prompt(**{"4": {"replacement_mode": True}}))

    def test_rejects_an_acceleration_lora_at_an_undeclared_strength(self):
        gate = scail2_production.JakkannaSCAIL2SamplerValidate()
        with self.assertRaisesRegex(ValueError, "acceleration LoRA strength"):
            gate.validate(self._sigmas(), self._prompt(**{"7": {"strength_model": 0.5}}))

    def test_rejects_a_second_scheduler(self):
        prompt = self._prompt()
        prompt["9"] = {"class_type": "BasicScheduler",
                       "inputs": {"scheduler": "simple", "steps": 6}}
        gate = scail2_production.JakkannaSCAIL2SamplerValidate()
        with self.assertRaisesRegex(ValueError, "exactly one BasicScheduler"):
            gate.validate(self._sigmas(), prompt)

    def test_requires_the_executed_prompt_graph(self):
        gate = scail2_production.JakkannaSCAIL2SamplerValidate()
        with self.assertRaisesRegex(ValueError, "needs the executed prompt graph"):
            gate.validate(self._sigmas(), None)

    def test_gate_and_manifest_agree_because_they_share_one_reconciliation(self):
        prompt = self._prompt(**{"3": {"cfg": 5.0}})
        declared = scail2_production._declared_manifest_settings(prompt)

        with self.assertRaisesRegex(ValueError, "does not match CFGGuider"):
            scail2_production._resolve_sampler_graph(prompt, declared, self._sigmas())
        with self.assertRaisesRegex(ValueError, "does not match CFGGuider"):
            scail2_production._resolve_sampler_graph(prompt, declared, None)


class ModelHashCacheTests(unittest.TestCase):
    """Hashing the manifest's model set costs tens of gigabytes of reads; the cache
    that avoids it has to survive a restart."""

    def _sandbox(self):
        home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, home, True)
        user_dir = os.path.join(home, "user")
        temp_dir = os.path.join(home, "temp")
        os.makedirs(user_dir)
        os.makedirs(temp_dir)
        weights = os.path.join(home, "model.safetensors")
        with open(weights, "wb") as handle:
            handle.write(b"pretend weights")
        # Patch the module object the node itself holds: another test module installs
        # its own folder_paths stub into sys.modules, so the registry is not reliable.
        folder_paths = scail2_production.folder_paths
        patches = (
            mock.patch.object(folder_paths, "get_user_directory", lambda: user_dir, create=True),
            mock.patch.object(folder_paths, "get_temp_directory", lambda: temp_dir, create=True),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return user_dir, temp_dir, weights

    def test_cache_is_not_kept_in_the_directory_comfyui_wipes(self):
        user_dir, temp_dir, weights = self._sandbox()

        digest = scail2_production._hash_file_cached(weights)
        cache_path = scail2_production._hash_cache_path()
        self.assertTrue(cache_path.is_file())
        self.assertEqual(Path(user_dir), Path(os.path.commonpath([user_dir, cache_path])))
        self.assertNotIn(Path(temp_dir), cache_path.parents)

        # main.py rmtrees the temp directory on both startup and shutdown.
        shutil.rmtree(temp_dir)
        self.assertTrue(cache_path.is_file())

        with mock.patch.object(
            scail2_production,
            "_hash_file",
            side_effect=AssertionError("re-read a model whose hash was already cached"),
        ):
            self.assertEqual(scail2_production._hash_file_cached(weights), digest)

    def test_cache_entry_is_rejected_when_the_file_changes(self):
        _user_dir, _temp_dir, weights = self._sandbox()
        first = scail2_production._hash_file_cached(weights)

        with open(weights, "wb") as handle:
            handle.write(b"different weights entirely")
        second = scail2_production._hash_file_cached(weights)

        self.assertNotEqual(first, second)

    def test_cache_directory_is_created_on_demand(self):
        user_dir, _temp_dir, weights = self._sandbox()
        shutil.rmtree(user_dir)

        scail2_production._hash_file_cached(weights)
        self.assertTrue(scail2_production._hash_cache_path().is_file())


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
