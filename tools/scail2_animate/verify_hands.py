#!/usr/bin/env python3
"""High-resolution crops of head and both hands to judge fine alignment."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from _env import bootstrap

bootstrap()

from mhr_rig import BODY_JOINTS, MhrRig
from render import Rasteriser, camera_space, project

CROPS = {
    "head": ("head", 520),
    "hand_l": ("hand_l", 620),
    "hand_r": ("hand_r", 620),
}


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pose_json = json.loads(args.fit.read_text())
    rig = MhrRig()
    fit = rig.reference_fit(pose_json)
    verts = rig.vertices(fit.joint_params, fit)

    photo = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if photo is None:
        raise FileNotFoundError(f"could not read {args.image}")
    h, w = photo.shape[:2]

    pts, depth = project(verts, fit.cam_t, fit.focal_length, w, h)
    raster = Rasteriser(rig.faces, device=rig.device, supersample=1)
    rgb, mask = raster.shade(camera_space(verts, fit.cam_t), pts, depth, w, h)

    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    overlay = photo.astype(np.float32) * (1 - 0.55 * alpha) + rgb[..., ::-1].astype(np.float32) * (0.55 * alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    joint_pos_cm, _, _ = rig.world_state(fit.joint_params)
    jpts, _ = project(joint_pos_cm / 100.0, fit.cam_t, fit.focal_length, w, h)

    args.out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for label, (joint, half) in CROPS.items():
        cx, cy = jpts[BODY_JOINTS[joint]]
        x0, y0 = int(cx - half), int(cy - half)
        x1, y1 = x0 + 2 * half, y0 + 2 * half
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        crop_photo = cv2.resize(photo[y0:y1, x0:x1], (512, 512))
        crop_over = cv2.resize(overlay[y0:y1, x0:x1], (512, 512))
        tiles.append(np.vstack([crop_photo, crop_over]))
        print(f"[hands] {label}: joint at ({cx:.0f}, {cy:.0f}) crop {x0},{y0}-{x1},{y1}")

    sheet = np.hstack(tiles)
    write_image(args.out / "detail_crops.png", sheet)
    print(f"[hands] wrote {args.out / 'detail_crops.png'}  (top: photo, bottom: mannequin overlay)")


if __name__ == "__main__":
    main()
