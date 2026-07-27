#!/usr/bin/env python3
"""Prove that driving MHR through joint parameters reproduces SAM 3D Body exactly.

If this matches, we can replace individual joint rotations with animation data and
trust that everything else about the reference fit is preserved.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _env import bootstrap

bootstrap()

from mhr_rig import BODY_JOINTS, MhrRig, quat_mul


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    args = parser.parse_args()
    pose_json = json.loads(args.fit.read_text())
    rig = MhrRig()

    print(f"joints={rig.num_joints}  faces={rig.faces.shape}")
    print("parents monotonic (parent < child):",
          bool(np.all(rig.parents[1:] < np.arange(1, rig.num_joints))))

    fit = rig.reference_fit(pose_json)
    print(f"model_params {fit.model_params.shape}  joint_params {fit.joint_params.shape}")
    print(f"cam_t={fit.cam_t}  focal={fit.focal_length}  image={fit.image_size}")

    # Reference: what SAM 3D Body itself produced for this pose.
    from jakkanna_pack.jakkanna_sam3d.processing.process import _to_batched_tensor
    dev = rig.device
    ref_out = rig.head.mhr_forward(
        global_trans=torch.zeros((1, 3), dtype=torch.float32, device=dev),
        global_rot=_to_batched_tensor(pose_json.get("global_rot"), dev, width=3),
        body_pose_params=_to_batched_tensor(pose_json.get("body_pose_params"), dev, width=133),
        hand_pose_params=_to_batched_tensor(pose_json.get("hand_pose_params"), dev, width=108),
        scale_params=_to_batched_tensor(pose_json.get("scale_params"), dev, width=rig.head.num_scale_comps),
        shape_params=_to_batched_tensor(pose_json.get("shape_params"), dev, width=rig.head.num_shape_comps),
        expr_params=torch.zeros((1, rig.head.num_face_comps), dtype=torch.float32, device=dev),
        return_joint_coords=True,
    )
    ref_verts = ref_out[0][0].cpu().numpy()
    ref_coords = ref_out[1][0].cpu().numpy()

    # Ours, rebuilt from joint parameters.
    ours = rig.vertices(fit.joint_params, fit)
    print(f"\nvertex agreement: max |delta| = {np.abs(ours - ref_verts).max():.3e} m")

    pos_cm, world_q, scale = rig.world_state(fit.joint_params)
    print(f"joint coord agreement: max |delta| = {np.abs(pos_cm / 100.0 - ref_coords).max():.3e} m")

    # Cross-check world quaternions against the rig's own skeleton state.
    skel = rig.skeleton_state(fit.joint_params)[0].cpu().numpy()
    print(f"skel_state t agreement:  max |delta| = {np.abs(skel[:, 0:3] - pos_cm).max():.3e} cm")
    dot = np.abs(np.sum(skel[:, 3:7] * world_q, axis=1))
    print(f"skel_state q agreement:  min |dot|   = {dot.min():.6f}")

    # Round-trip: ask for the fit's own world rotations back and expect a no-op.
    targets = {idx: world_q[idx] for idx in BODY_JOINTS.values()}
    rebuilt = rig.set_world_rotations(fit.joint_params, targets)
    rebuilt_verts = rig.vertices(rebuilt, fit)
    print(f"\nworld-rotation round trip: max |delta| = {np.abs(rebuilt_verts - ours).max():.3e} m")

    for name, idx in sorted(BODY_JOINTS.items(), key=lambda kv: kv[1]):
        p = pos_cm[idx] / 100.0
        print(f"  {name:12s} idx={idx:3d} parent={rig.parents[idx]:3d} pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})")


if __name__ == "__main__":
    main()
