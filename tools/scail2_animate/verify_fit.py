#!/usr/bin/env python3
"""Render the SAM 3D Body fit back onto the reference photo to judge alignment."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from _env import bootstrap

bootstrap()

from mhr_rig import BODY_JOINTS, MhrRig
from render import Rasteriser, camera_space, project


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.25, help="preview downscale")
    parser.add_argument("--cull", choices=["neg", "pos"], default="neg")
    args = parser.parse_args()

    pose_json = json.loads(args.fit.read_text())
    rig = MhrRig()
    fit = rig.reference_fit(pose_json)
    verts = rig.vertices(fit.joint_params, fit)

    photo = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if photo is None:
        raise FileNotFoundError(f"could not read {args.image}")
    full_h, full_w = photo.shape[:2]

    # Render at preview scale: focal and principal point scale with the image.
    scale = args.scale
    w, h = int(round(full_w * scale)), int(round(full_h * scale))
    focal = fit.focal_length * scale

    pts, depth = project(verts, fit.cam_t, focal, w, h)
    raster = Rasteriser(rig.faces, device=rig.device, supersample=2)
    if args.cull == "pos":
        raster.faces = raster.faces[:, [0, 2, 1]]
    rgb, mask = raster.shade(camera_space(verts, fit.cam_t), pts, depth, w, h)

    small = cv2.resize(photo, (w, h), interpolation=cv2.INTER_AREA)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    overlay = small.astype(np.float32) * (1 - 0.62 * alpha) + rgb[..., ::-1].astype(np.float32) * (0.62 * alpha)

    # 2D joint markers for the joints that define the alignment quality bar.
    joint_pos_cm, _, _ = rig.world_state(fit.joint_params)
    jpts, _ = project(joint_pos_cm / 100.0, fit.cam_t, focal, w, h)
    for name, idx in BODY_JOINTS.items():
        x, y = int(round(jpts[idx, 0])), int(round(jpts[idx, 1]))
        if not (0 <= x < w and 0 <= y < h):
            continue
        colour = (0, 255, 255) if name.endswith("_l") else (255, 128, 0)
        cv2.circle(overlay, (x, y), 4, colour, -1, cv2.LINE_AA)

    args.out.mkdir(parents=True, exist_ok=True)
    write_image(args.out / "overlay.png", np.clip(overlay, 0, 255).astype(np.uint8))
    write_image(args.out / "mannequin.png", rgb[..., ::-1])
    write_image(args.out / "silhouette.png", mask)
    print(f"[verify] wrote overlay/mannequin/silhouette to {args.out} at {w}x{h}")
    print(f"[verify] mannequin covers {float((mask > 127).mean()) * 100:.2f}% of frame")


if __name__ == "__main__":
    main()
