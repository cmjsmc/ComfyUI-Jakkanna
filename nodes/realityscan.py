from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import cv2
import torch
import torch.nn.functional as F

import folder_paths
from comfy.ldm.sam3.tracker import unpack_masks


_DATASET_PART_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_root(prefix):
    relative = Path(prefix)
    if relative.is_absolute() or not relative.parts or any(
        part in ("", ".", "..") or not _DATASET_PART_RE.fullmatch(part)
        for part in relative.parts
    ):
        raise ValueError("dataset_prefix must be a relative path containing only letters, numbers, '.', '_', '-' and '/'")

    output_root = Path(folder_paths.get_output_directory()).resolve()
    base = (output_root / relative).resolve()
    if not base.is_relative_to(output_root):
        raise ValueError("dataset_prefix must remain inside the ComfyUI output directory")
    base.mkdir(parents=True, exist_ok=True)

    counter = 1
    while True:
        root = base / f"run_{counter:05d}"
        try:
            root.mkdir()
            return root
        except FileExistsError:
            counter += 1


class JakkannaRealityScanImageSetSave:
    FRAME_COUNT = 81
    WIDTH = 704
    HEIGHT = 1280
    FPS = 16

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "track_data": ("SAM3_TRACK_DATA",),
                "track_report": ("STRING", {"forceInput": True}),
                "expected_frames": ("INT", {"forceInput": True}),
                "dataset_prefix": (
                    "STRING",
                    {"default": "Jakkanna/RealityScan/turntable"},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dataset_path", "manifest_path")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Jakkanna/RealityScan"

    def save(self, images, track_data, track_report, expected_frames, dataset_prefix):
        if expected_frames != self.FRAME_COUNT:
            raise ValueError(f"RealityScan turntable must contain exactly {self.FRAME_COUNT} frames")
        if images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("RealityScan images must have shape [frames, height, width, channels]")
        if images.shape[0] != self.FRAME_COUNT:
            raise ValueError(f"RealityScan image batch has {images.shape[0]} frames; expected {self.FRAME_COUNT}")
        if (images.shape[2], images.shape[1]) != (self.WIDTH, self.HEIGHT):
            raise ValueError(
                f"RealityScan image frames must be {self.WIDTH}x{self.HEIGHT}; "
                f"got {images.shape[2]}x{images.shape[1]}"
            )

        packed = track_data.get("packed_masks")
        if not isinstance(packed, torch.Tensor):
            raise ValueError("SAM 3.1 produced no final output track")
        if packed.ndim != 4 or packed.shape[:2] != (self.FRAME_COUNT, 1):
            objects = packed.shape[1] if packed.ndim >= 2 else 0
            frames = packed.shape[0] if packed.ndim >= 1 else 0
            raise ValueError(
                f"SAM 3.1 final track must contain one subject across {self.FRAME_COUNT} frames; "
                f"got {objects} subjects across {frames} frames"
            )

        orig_size = track_data.get("orig_size")
        if not isinstance(orig_size, (tuple, list)) or tuple(orig_size) != (self.HEIGHT, self.WIDTH):
            raise ValueError(
                f"SAM 3.1 original frame size must be {self.WIDTH}x{self.HEIGHT}; "
                f"got {orig_size}"
            )

        try:
            parsed_track_report = json.loads(track_report)
        except (TypeError, ValueError):
            raise ValueError("Validated SAM 3.1 track report is invalid") from None
        if parsed_track_report.get("frames") != self.FRAME_COUNT or parsed_track_report.get("objects") != 1:
            raise ValueError("Validated SAM 3.1 track report does not describe the required 81-frame single-subject track")

        native_mask_height = packed.shape[-2]
        native_mask_width = packed.shape[-1] * 8
        projected_masks = []
        mask_areas = []
        for index in range(self.FRAME_COUNT):
            mask = unpack_masks(packed[index:index + 1]).float()
            if mask.shape[-2:] != (self.HEIGHT, self.WIDTH):
                mask = F.interpolate(
                    mask,
                    size=(self.HEIGHT, self.WIDTH),
                    mode="bilinear",
                    align_corners=False,
                )
            mask = (mask[0, 0] > 0.5).detach().to(device="cpu").numpy()
            area_pixels = int(mask.sum())
            if area_pixels == 0:
                raise ValueError(f"SAM 3.1 final track is empty on frame {index}")
            projected_masks.append(mask)
            mask_areas.append(area_pixels)

        root = _dataset_root(dataset_prefix)
        layers = root / "layers"
        geometry = layers / ".geometry"
        masks = layers / ".mask"
        geometry.mkdir(parents=True)
        masks.mkdir()

        frames = []
        for index in range(self.FRAME_COUNT):
            mask = projected_masks[index]
            rgb = images[index, :, :, :3].detach().to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(rgb).all():
                raise ValueError(f"RealityScan image frame {index} contains non-finite pixels")
            rgb = rgb.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
            rgb[~mask] = 0

            filename = f"frame_{index:04d}.png"
            geometry_path = geometry / filename
            mask_path = masks / filename
            if not cv2.imwrite(str(geometry_path), rgb[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 4]):
                raise RuntimeError(f"Failed to write RealityScan image: {geometry_path}")
            if not cv2.imwrite(str(mask_path), mask.astype("uint8") * 255, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
                raise RuntimeError(f"Failed to write RealityScan mask: {mask_path}")

            frames.append({
                "index": index,
                "filename": filename,
                "turntable_angle_degrees": index * 360.0 / self.FRAME_COUNT,
                "mask_area_pixels": mask_areas[index],
                "geometry_sha256": _hash_file(geometry_path),
                "mask_sha256": _hash_file(mask_path),
            })

        manifest = {
            "schema": "jakkanna.realityscan.turntable.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(root),
            "layers_path": str(layers),
            "geometry_layer": ".geometry",
            "mask_layer": ".mask",
            "frame_count": self.FRAME_COUNT,
            "width": self.WIDTH,
            "height": self.HEIGHT,
            "fps": self.FPS,
            "turntable_degrees": 360,
            "turntable_step_degrees": 360.0 / self.FRAME_COUNT,
            "duplicate_endpoint": False,
            "realityscan_camera_priors": {
                "type": "position",
                "circle_radius_meters": 4.5,
                "circle_height_meters": 1.0,
            },
            "background": "black outside the binary SAM 3.1 mask",
            "sam3_native_mask_width": native_mask_width,
            "sam3_native_mask_height": native_mask_height,
            "sam3_mask_projection": "bilinear resize to original frame size, thresholded at 0.5",
            "track_report": parsed_track_report,
            "frames": frames,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        return str(root), str(manifest_path)


NODE_CLASS_MAPPINGS = {
    "JakkannaRealityScanImageSetSave": JakkannaRealityScanImageSetSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JakkannaRealityScanImageSetSave": "Jakkanna RealityScan Save Image Set",
}
