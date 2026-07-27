import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch

import comfy.model_management
import folder_paths

from ..camera_presets import (
    AUTO_CAMERA_PRESET,
    CAMERA_PRESETS,
    camera_preset_spec,
)


_JAKKANNA_ROOT = Path(__file__).resolve().parents[1]
_RUN_ALL = _JAKKANNA_ROOT / "tools" / "scail2_animate" / "run_all.py"
_MINIMUM_IMAGE_EDGE = 720
_MAXIMUM_IMAGE_PIXELS = 4096 * 4096


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _animation_root():
    return Path(folder_paths.get_input_directory()).resolve() / "animate"


def _fbx_files():
    root = _animation_root()
    if not root.is_dir():
        return []
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".fbx"
    )


def _resolve_fbx(name):
    root = _animation_root().resolve()
    path = (root / name).resolve()
    if root != path.parent and root not in path.parents:
        raise ValueError(f"FBX path escapes the animation input directory: {name!r}")
    if not path.is_file() or path.suffix.lower() != ".fbx":
        raise ValueError(f"FBX animation does not exist under {root}: {name!r}")
    return path


def _tensor_rgb16_sha256(image):
    data = (
        image.detach()
        .to(device="cpu", dtype=torch.float32)[..., :3]
        .clamp(0, 1)
        .mul(65535)
        .round()
        .to(torch.uint16)
        .contiguous()
    )
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def _write_source_image(image):
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError("Reference must provide an IMAGE tensor batch")
    if image.shape[0] != 1:
        raise ValueError(f"Reference must contain exactly one image, got {image.shape[0]}")
    height, width = image.shape[1:3]
    if min(width, height) < _MINIMUM_IMAGE_EDGE:
        raise ValueError(
            f"Reference must be high resolution with a shortest edge of at least "
            f"{_MINIMUM_IMAGE_EDGE}px, got {width}x{height}"
        )
    if width * height > _MAXIMUM_IMAGE_PIXELS:
        raise ValueError(
            f"Reference exceeds the supported {_MAXIMUM_IMAGE_PIXELS}-pixel limit: {width}x{height}"
        )
    if not torch.isfinite(image).all():
        raise ValueError("Reference contains non-finite values")
    minimum, maximum = torch.aminmax(image)
    if minimum < 0 or maximum > 1:
        raise ValueError(
            f"Reference IMAGE values must be in [0, 1], got "
            f"[{float(minimum.item()):g}, {float(maximum.item()):g}]"
        )

    rgb = (
        image[0, ..., :3]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .mul(255)
        .round()
        .to(torch.uint8)
        .contiguous()
    )
    rgb8_sha256 = hashlib.sha256(rgb.numpy().tobytes()).hexdigest()
    source_dir = Path(folder_paths.get_input_directory()) / "jakkanna_animate" / "_source_cache"
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / f"{rgb8_sha256}.png"
    if path.exists():
        cached = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if cached is None:
            raise ValueError(f"Cached workflow image cannot be decoded: {path}")
        cached_rgb = np.ascontiguousarray(cached[..., ::-1])
        if hashlib.sha256(cached_rgb.tobytes()).hexdigest() != rgb8_sha256:
            raise ValueError(f"Cached workflow image does not match its content identity: {path}")
    elif not cv2.imwrite(str(path), rgb.numpy()[..., ::-1]):
        raise RuntimeError(f"Failed to write normalized workflow image: {path}")
    return path, {
        "width": width,
        "height": height,
        "workflow_input_rgb16_sha256": _tensor_rgb16_sha256(image),
        "normalized_rgb8_sha256": rgb8_sha256,
    }


def _write_subject_mask(mask, width, height):
    if not isinstance(mask, torch.Tensor) or mask.ndim != 3:
        raise ValueError("Subject mask must provide a MASK tensor batch")
    if tuple(mask.shape) != (1, height, width):
        raise ValueError(
            f"Subject mask must contain one {width}x{height} mask, got {tuple(mask.shape)}"
        )
    data = mask.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(data).all():
        raise ValueError("Subject mask contains non-finite values")
    if not torch.all((data == 0) | (data == 1)):
        raise ValueError("Subject mask must be binary")
    binary = data[0].to(torch.uint8).contiguous().numpy()
    if not binary.any():
        raise ValueError("Subject mask is empty")

    binary_sha256 = hashlib.sha256(binary.tobytes()).hexdigest()
    mask_dir = Path(folder_paths.get_input_directory()) / "jakkanna_animate" / "_subject_mask_cache"
    mask_dir.mkdir(parents=True, exist_ok=True)
    path = mask_dir / f"{binary_sha256}.png"
    if path.exists():
        cached = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if cached is None or not np.array_equal(cached > 127, binary > 0):
            raise ValueError(f"Cached workflow subject mask does not match its content identity: {path}")
    elif not cv2.imwrite(str(path), binary * 255):
        raise RuntimeError(f"Failed to write workflow subject mask: {path}")
    return path, {
        "width": width,
        "height": height,
        "binary_sha256": binary_sha256,
        "coverage": float(binary.mean()),
    }


def _load_rgb_image(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not decode prepared image: {path}")
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    return torch.from_numpy(rgb).to(torch.float32).div_(255).unsqueeze(0)


def _load_rgb_sequence(run_dir, artifact, expected_frames):
    directory = run_dir / artifact["path"]
    entries = artifact.get("files", [])
    if len(entries) != expected_frames:
        raise ValueError(
            f"Prepared sequence {directory} contains {len(entries)} manifest entries; "
            f"expected {expected_frames}"
        )
    batch = None
    for index, entry in enumerate(entries):
        path = directory / entry["name"]
        if _file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"Prepared frame does not match its manifest: {path}")
        frame = _load_rgb_image(path)
        if batch is None:
            batch = torch.empty(
                (expected_frames, frame.shape[1], frame.shape[2], 3),
                dtype=torch.float32,
            )
        if frame.shape[1:] != batch.shape[1:]:
            raise ValueError(f"Prepared sequence contains inconsistent frame geometry: {path}")
        batch[index].copy_(frame[0])
    return batch


class JakkannaImageFBXPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        files = _fbx_files()
        return {
            "required": {
                "image": ("IMAGE",),
                "fbx_file": (files or ["No FBX files found under input/animate"],),
            },
            "optional": {
                "camera_preset": (list(CAMERA_PRESETS), {"default": AUTO_CAMERA_PRESET}),
                "subject_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("driver", "driving_mask", "registered_reference", "preprocess_report")
    FUNCTION = "prepare"
    CATEGORY = "Jakkanna/SCAIL-2"
    DESCRIPTION = (
        "Automatically fits one high-resolution subject, retargets the selected "
        "input/animate FBX, and prepares the registered SCAIL-2 driver. A connected "
        "subject mask guides the otherwise automatic body fit."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, image, fbx_file, camera_preset=AUTO_CAMERA_PRESET, subject_mask=None):
        try:
            _resolve_fbx(fbx_file)
            camera_preset_spec(camera_preset)
        except ValueError as error:
            return str(error)
        return True

    @classmethod
    def IS_CHANGED(cls, image, fbx_file, camera_preset=AUTO_CAMERA_PRESET, subject_mask=None):
        try:
            return f"{_file_sha256(_resolve_fbx(fbx_file))}:{camera_preset}"
        except (OSError, ValueError):
            return float("nan")

    def prepare(self, image, fbx_file, camera_preset=AUTO_CAMERA_PRESET, subject_mask=None):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Jakkanna image + FBX preparation requires an NVIDIA CUDA GPU"
            )
        camera_preset_spec(camera_preset)
        source_path, source_metadata = _write_source_image(image)
        subject_mask_path = None
        subject_mask_metadata = None
        if subject_mask is not None:
            subject_mask_path, subject_mask_metadata = _write_subject_mask(
                subject_mask,
                source_metadata["width"],
                source_metadata["height"],
            )
        fbx_path = _resolve_fbx(fbx_file)
        run_name = f"greenfield-{source_metadata['normalized_rgb8_sha256'][:12]}-{fbx_path.stem}"

        temp_dir = Path(folder_paths.get_temp_directory())
        temp_dir.mkdir(parents=True, exist_ok=True)
        handle, result_name = tempfile.mkstemp(prefix="jakkanna-image-fbx-", suffix=".json", dir=temp_dir)
        os.close(handle)
        result_path = Path(result_name)
        command = [
            sys.executable,
            str(_RUN_ALL),
            "--image", str(source_path),
            "--fbx", str(fbx_path),
            "--name", run_name,
            "--timing", "auto",
            "--root-motion", "auto",
            "--camera-preset", camera_preset,
            "--skip-verify",
            "--result-file", str(result_path),
        ]
        if subject_mask_path is not None:
            command.extend(("--subject-mask", str(subject_mask_path)))

        comfy.model_management.unload_all_models()
        try:
            process = subprocess.Popen(
                command,
                cwd=_RUN_ALL.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError("Image/FBX preparation did not expose its progress stream")
            for line in process.stdout:
                print(line, end="")
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"Image/FBX preparation failed with exit code {return_code}")
            result = json.loads(result_path.read_text())
        finally:
            result_path.unlink(missing_ok=True)

        run_dir = Path(result["run_dir"]).resolve()
        input_root = Path(folder_paths.get_input_directory()).resolve()
        if input_root not in run_dir.parents:
            raise ValueError(f"Prepared run escaped the ComfyUI input directory: {run_dir}")
        report_path = run_dir / "preprocess_manifest.json"
        report = json.loads(report_path.read_text())
        if report.get("schema") != "jakkanna.image_fbx_animation.v1":
            raise ValueError("Prepared run has an unsupported manifest schema")
        if subject_mask_path is not None:
            prepared_mask = report.get("inputs", {}).get("subject_mask", {})
            if prepared_mask.get("sha256") != _file_sha256(subject_mask_path):
                raise ValueError("Prepared run subject mask does not match the workflow input")
        report["workflow_input"] = {
            **source_metadata,
            "fbx_file": fbx_path.relative_to(_animation_root().resolve()).as_posix(),
            "fbx_sha256": _file_sha256(fbx_path),
            "camera_preset": camera_preset,
        }
        if subject_mask_metadata is not None:
            report["workflow_input"]["subject_mask"] = subject_mask_metadata

        frames = report.get("settings", {}).get("frames")
        if frames != 81:
            raise ValueError(f"Prepared run contains {frames} frames; expected 81")
        artifacts = report.get("artifacts", {})
        driver = _load_rgb_sequence(run_dir, artifacts["driver"], frames)
        driving_mask = _load_rgb_sequence(run_dir, artifacts["driving_mask"], frames)
        registered_reference = _load_rgb_image(run_dir / artifacts["registered_reference"]["path"])
        return (
            driver,
            driving_mask,
            registered_reference,
            json.dumps(report, sort_keys=True, separators=(",", ":")),
        )


NODE_CLASS_MAPPINGS = {
    "JakkannaImageFBXPrepare": JakkannaImageFBXPrepare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JakkannaImageFBXPrepare": "Jakkanna Image + FBX Prepare",
}
