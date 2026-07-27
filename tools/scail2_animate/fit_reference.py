#!/usr/bin/env python3
"""Fit SAM 3D Body (MHR) to the reference photograph.

Produces the alignment anchor for the whole pipeline: the subject's 3D body
pose, hand/finger articulation, shape and the camera that reprojects it onto
the photo. Everything downstream (retarget, render) is expressed in this
camera so the mannequin lands exactly where the character is in the frame.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from _env import bootstrap

bootstrap()


def load_image_tensor(path: Path) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).unsqueeze(0)


def load_mask_tensor(path: Path, image: torch.Tensor) -> torch.Tensor:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"could not read {path}")
    if mask.shape != tuple(image.shape[1:3]):
        raise ValueError(
            f"subject mask is {mask.shape[1]}x{mask.shape[0]}; "
            f"reference image is {image.shape[2]}x{image.shape[1]}"
        )
    return torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bbox-threshold", type=float, default=0.8)
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"reference image does not exist: {args.image}")
    args.out.mkdir(parents=True, exist_ok=True)

    from jakkanna_pack.jakkanna_sam3d.processing.load_model import LoadSAM3DBodyModel
    from jakkanna_pack.jakkanna_sam3d.processing.process import SAM3DBodyProcessToJson

    image = load_image_tensor(args.image)
    print(f"[fit] image {args.image}  {tuple(image.shape)}")
    mask = load_mask_tensor(args.mask, image) if args.mask is not None else None
    if mask is not None:
        print(f"[fit] subject mask {args.mask}  coverage={float(mask.mean()):.6f}")

    model = LoadSAM3DBodyModel().load_model("Auto")[0]
    pose_json = SAM3DBodyProcessToJson().process_to_json(
        model=model,
        image=image,
        bbox_threshold=args.bbox_threshold,
        inference_type="full",
        debug_scale=True,
        mask=mask,
    )[0]

    payload = json.loads(pose_json)
    out_path = args.out / "pose.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[fit] wrote {out_path}")

    for key, value in payload.items():
        if isinstance(value, list):
            arr = np.asarray(value, dtype=object)
            print(f"  {key}: list shape~{np.shape(value)}")
        elif isinstance(value, dict):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
