#!/usr/bin/env python3
"""Render the 81-frame mannequin driver for SCAIL-2, aligned to the reference photo.

Outputs, under ComfyUI's input directory so the workflow can load them:
  <pose_dir>/frame_%04d.png   flat-shaded mannequin on black  (pose_video)
  <mask_dir>/frame_%04d.png   SCAIL-2 animation-mode colour mask, blue on black

The render camera is the SAM 3D Body camera from the reference photo, rescaled
for a centre crop to the output aspect -- the same crop WanSCAILToVideo applies
to the reference image -- so driver and reference share one framing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from _env import COMFY_ROOT, bootstrap, file_sha256, repository_commit

bootstrap()

from fbx import FbxClip
from mhr_rig import MhrRig
from render import Rasteriser, camera_space, project
from retarget import Retargeter

# SCAIL-2's palette: identity 0 is blue. Animation mode wants a black background.
SUBJECT_BGR = (255, 0, 0)


def write_image(path: Path, image: np.ndarray, parameters: list[int] | None = None) -> None:
    if not cv2.imwrite(str(path), image, parameters or []):
        raise RuntimeError(f"failed to write image: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--fbx", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--start", type=float, default=0.0, help="clip start time, seconds")
    parser.add_argument("--supersample", type=int, default=3)
    parser.add_argument("--bbox-threshold", type=float, default=0.8)
    parser.add_argument("--framing", choices=("full_body", "reference_crop"), default="full_body",
                        help="full_body pulls the camera back so the whole clip fits the canvas; "
                             "reference_crop keeps the photograph's own framing")
    parser.add_argument("--margin", type=float, default=0.06,
                        help="fraction of the canvas kept clear around the body (full_body only)")
    parser.add_argument("--ease-frames", type=int, default=12,
                        help="frames spent easing out of the photograph's pose into the clip; "
                             "0 starts on the clip's own pose")
    parser.add_argument("--root-motion", choices=("preserve", "in_place"), default="preserve")
    return parser.parse_args()


def rgb16_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read generated image {path}")
        rgb16 = (bgr[..., ::-1].astype(np.dtype("<u2")) * 257).copy()
        digest.update(rgb16.tobytes())
    return digest.hexdigest()


def artifact(path: Path, run_dir: Path) -> dict:
    return {
        "path": str(path.relative_to(run_dir)),
        "file_sha256": file_sha256(path),
        "rgb16_sha256": rgb16_sha256([path]),
    }


def source_input(path: Path) -> dict:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(COMFY_ROOT)
    except ValueError:
        relative = Path(resolved.name)
    return {"path": str(relative), "sha256": file_sha256(resolved)}


def reference_crop_camera(fit, width: int, height: int) -> tuple[float, tuple[float, float]]:
    """Focal and principal point for a centre crop of the reference to width:height."""
    ref_w, ref_h = fit.image_size
    if (ref_w / ref_h) > (width / height):
        scale = height / ref_h          # crop the sides
    else:
        scale = width / ref_w           # crop top and bottom
    # The crop is centred, so the principal point stays at the render centre.
    return fit.focal_length * scale, (width * 0.5, height * 0.5)


def fit_body_camera(rays: np.ndarray, width: int, height: int,
                    margin: float = 0.06) -> tuple[float, tuple[float, float]]:
    """Camera that frames every supplied normalised ray inside width x height.

    Only the focal length and principal point change; the camera centre stays put.
    That keeps the projection a pure scale-and-translate of the reference camera,
    so the photograph's pixels can be mapped into this canvas exactly.
    """
    x_lo, y_lo = rays.min(axis=0)
    x_hi, y_hi = rays.max(axis=0)
    pad_x, pad_y = width * margin, height * margin
    focal = min((width - 2 * pad_x) / max(x_hi - x_lo, 1e-9),
                (height - 2 * pad_y) / max(y_hi - y_lo, 1e-9))
    cx = width * 0.5 - focal * (x_lo + x_hi) * 0.5
    cy = height * 0.5 - focal * (y_lo + y_hi) * 0.5
    return focal, (cx, cy)


def reproject_reference(photo: np.ndarray, fit, focal: float,
                        principal: tuple[float, float],
                        width: int, height: int) -> np.ndarray:
    """Map the reference photograph into a virtual camera that shares its centre.

    u' = (f'/f) * (u - cx) + cx', which is an affine warp, so the subject stays
    pixel-registered with anything rendered through the same virtual camera.
    """
    ref_w, ref_h = fit.image_size
    scale = focal / fit.focal_length
    matrix = np.array([
        [scale, 0.0, principal[0] - scale * ref_w * 0.5],
        [0.0, scale, principal[1] - scale * ref_h * 0.5],
    ], dtype=np.float64)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    return cv2.warpAffine(photo, matrix, (width, height), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def main() -> None:
    args = parse_args()
    run_id = args.run_id or args.run_dir.name
    if not args.image.is_file():
        raise FileNotFoundError(f"reference image does not exist: {args.image}")
    if not args.fbx.is_file():
        raise FileNotFoundError(f"FBX animation does not exist: {args.fbx}")
    if not args.fit.is_file():
        raise FileNotFoundError(f"SAM 3D Body fit does not exist: {args.fit}")
    if (args.width, args.height) not in {(704, 1280), (1280, 704)}:
        raise ValueError("SCAIL-2 prepared drivers must be 704x1280 or 1280x704")
    if args.frames != 81:
        raise ValueError("SCAIL-2 prepared drivers must contain exactly 81 frames")
    if args.fps != 16.0:
        raise ValueError("SCAIL-2 prepared drivers must be exactly 16 fps")
    if not 0 <= args.margin < 0.5:
        raise ValueError("margin must be in [0, 0.5)")
    if args.ease_frames < 0:
        raise ValueError("ease-frames must be nonnegative")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    pose_dir = args.run_dir / "driver"
    mask_dir = args.run_dir / "driving_mask"
    review_dir = args.run_dir / "review"
    for directory in (pose_dir, mask_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"prepared output directory is not empty: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    pose_json = json.loads(args.fit.read_text())
    rig = MhrRig()
    fit = rig.reference_fit(pose_json)
    clip = FbxClip(args.fbx)
    retargeter = Retargeter(rig, fit, clip)

    print(f"[driver] clip '{args.fbx.name}' {clip.duration:.2f}s, "
          f"{args.frames} frames @ {args.fps} fps from {args.start:.2f}s, "
          f"ease {args.ease_frames} frames")
    times = retargeter.sample_times(args.frames, args.fps, args.start)

    # Pose every frame first: the full-body camera has to see the whole clip
    # before it can choose a focal length that keeps the body inside the canvas.
    posed = []
    rays = []
    for index, time in enumerate(times):
        params = retargeter.joint_params_at(
            time, reference_time=args.start,
            follow_root=args.root_motion == "preserve",
            blend=retargeter.ease(index, args.ease_frames),
        )
        verts = rig.vertices(params, fit)
        posed.append(verts)
        cam = camera_space(verts, fit.cam_t)
        rays.append(cam[:, :2] / np.maximum(cam[:, 2:3], 1e-4))
    rays = np.concatenate(rays, axis=0)

    if args.framing == "full_body":
        focal, principal = fit_body_camera(rays, args.width, args.height, args.margin)
    else:
        focal, principal = reference_crop_camera(fit, args.width, args.height)
    print(f"[driver] reference {fit.image_size} focal={fit.focal_length:.1f} -> "
          f"{args.framing} {args.width}x{args.height} focal={focal:.2f} "
          f"principal=({principal[0]:.1f}, {principal[1]:.1f})")

    raster = Rasteriser(rig.faces, device=rig.device, supersample=args.supersample)

    preview_frames = []
    for index, verts in enumerate(posed):
        time = times[index]
        pts, depth = project(verts, fit.cam_t, focal, args.width, args.height, principal)
        rgb, mask = raster.shade(camera_space(verts, fit.cam_t), pts, depth,
                                 args.width, args.height)

        write_image(pose_dir / f"frame_{index:04d}.png", rgb[..., ::-1])

        hard = (mask > 127)
        colour = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        colour[hard] = SUBJECT_BGR
        write_image(mask_dir / f"frame_{index:04d}.png", colour)

        if index % 8 == 0:
            preview_frames.append(rgb[..., ::-1])
        if index % 10 == 0 or index == len(times) - 1:
            print(f"  frame {index:3d}/{len(times)}  t={time:6.3f}s  "
                  f"coverage={hard.mean() * 100:5.2f}%")

    if preview_frames:
        columns = min(4, len(preview_frames))
        blank = np.zeros_like(preview_frames[0])
        rows = [preview_frames[i:i + columns] for i in range(0, len(preview_frames), columns)]
        rows = [row + [blank] * (columns - len(row)) for row in rows]
        contact = np.vstack([np.hstack(row) for row in rows])
        write_image(review_dir / "driver_contact_sheet.jpg", contact)

    # The reference photograph, mapped into the same virtual camera as the driver.
    # This is what the workflow loads: SAM 3.1 segments the subject out of it, so
    # reference and driver share one canvas, one camera and one framing.
    photo = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if photo is None:
        raise FileNotFoundError(f"could not read {args.image}")
    reference = reproject_reference(photo, fit, focal, principal, args.width, args.height)
    reference_path = args.run_dir / "registered_reference.png"
    write_image(reference_path, reference)

    first = cv2.imread(str(pose_dir / "frame_0000.png"))
    if first is None:
        raise FileNotFoundError(f"could not read generated driver frame: {pose_dir / 'frame_0000.png'}")
    overlay = np.clip(reference.astype(np.float32) * 0.55 + first.astype(np.float32) * 0.75, 0, 255)
    write_image(review_dir / "driver_frame0_over_reference.jpg", overlay.astype(np.uint8))

    # How well frame 0 registers with the subject decides whether SCAIL-2 only has
    # to change appearance or geometry as well. Segment the subject the same way
    # the fit did, reproject it, and measure the overlap against driver frame 0.
    from jakkanna_pack.jakkanna_sam3d.processing.birefnet_mask import auto_mask_bgr

    subject_mask, _ = auto_mask_bgr(photo)
    subject_sil = reproject_reference((subject_mask > 0.5).astype(np.uint8) * 255,
                                      fit, focal, principal, args.width, args.height) > 127
    if subject_sil.ndim == 3:
        subject_sil = subject_sil[..., 0]
    driver_sil = cv2.imread(str(pose_dir / "frame_0000.png"), cv2.IMREAD_GRAYSCALE) > 24
    iou = float((driver_sil & subject_sil).sum()) / max(int((driver_sil | subject_sil).sum()), 1)
    visible = driver_sil & (reference.max(axis=2) > 0)
    iou_visible = (float((visible & subject_sil).sum())
                   / max(int((visible | subject_sil).sum()), 1))
    print(f"[driver] frame 0 vs photographed subject: IoU = {iou:.3f} over the whole body, "
          f"{iou_visible:.3f} over the part the photo actually shows")

    driver_paths = sorted(pose_dir.glob("frame_*.png"))
    mask_paths = sorted(mask_dir.glob("frame_*.png"))
    if len(driver_paths) != args.frames or len(mask_paths) != args.frames:
        raise RuntimeError(
            f"prepared output is incomplete: expected {args.frames} driver and mask frames, "
            f"got {len(driver_paths)} and {len(mask_paths)}"
        )
    manifest = {
        "schema": "jakkanna.image_fbx_animation.v1",
        "run_id": run_id,
        "inputs": {
            "image": source_input(args.image),
            "fbx": source_input(args.fbx),
        },
        "settings": {
            "frames": args.frames,
            "fps": args.fps,
            "start_seconds": args.start,
            "width": args.width,
            "height": args.height,
            "framing": args.framing,
            "margin": args.margin,
            "ease_frames": args.ease_frames,
            "root_motion": args.root_motion,
            "supersample": args.supersample,
            "bbox_threshold": args.bbox_threshold,
        },
        "registration": {
            "render_focal": focal,
            "render_principal": list(principal),
            "reference_focal": fit.focal_length,
            "reference_image_size": list(fit.image_size),
            "camera_translation": fit.cam_t.tolist(),
            "frame0_subject_iou": iou,
            "frame0_subject_iou_visible": iou_visible,
        },
        "artifacts": {
            "fit": {
                "path": str(args.fit.relative_to(args.run_dir)) if args.run_dir in args.fit.parents else args.fit.name,
                "file_sha256": file_sha256(args.fit),
            },
            "driver": {
                "path": "driver",
                "files": [{"name": path.name, "sha256": file_sha256(path)} for path in driver_paths],
                "rgb16_sha256": rgb16_sha256(driver_paths),
            },
            "driving_mask": {
                "path": "driving_mask",
                "files": [{"name": path.name, "sha256": file_sha256(path)} for path in mask_paths],
                "rgb16_sha256": rgb16_sha256(mask_paths),
                "palette_rgb": [[0, 0, 0], [0, 0, 255]],
            },
            "registered_reference": artifact(reference_path, args.run_dir),
        },
        "software": {
            "jakkanna_commit": repository_commit(),
        },
    }
    manifest_path = args.run_dir / "preprocess_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[driver] wrote {args.frames} frames to {pose_dir} and {mask_dir}")
    print(f"[driver] manifest at {manifest_path}")


if __name__ == "__main__":
    main()
