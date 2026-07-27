#!/usr/bin/env python3
"""Direct control of the MHR (Momentum Human Rig) used by SAM 3D Body.

SAM 3D Body predicts MHR *model parameters*. Momentum expands those into
127 x 7 joint parameters (tx, ty, tz, rx, ry, rz, log2 scale), turns them into a
per-joint world skeleton state (translation, quaternion xyzw, scale) and skins
the mesh with linear blend skinning.

Working at the joint-parameter level lets us keep everything SAM 3D Body
inferred about the reference photo -- body shape, bone scales, hand pose and the
camera -- while replacing only the rotations we want to drive from an external
animation clip.

Conventions, read off the TorchScript rig itself:
  local_q = prerotation * euler_xyz_to_quaternion(params[3:6])
  world_q[j] = world_q[parent] * local_q[j]
  quaternions are (x, y, z, w); euler XYZ composes as R = Rz @ Ry @ Rx
  the rig works in centimetres; SAM 3D Body divides by 100 on the way out
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# MHR joint indices that Jakkanna's SAM 3D bridge identifies by name. The rig
# ships no joint_names table, so this is the authoritative body mapping.
BODY_JOINTS = {
    "pelvis": 1,
    "thigh_l": 2, "calf_l": 3, "foot_l": 4,
    "thigh_r": 18, "calf_r": 19, "foot_r": 20,
    "spine_01": 35, "spine_02": 36, "spine_03": 37,
    "clavicle_r": 38, "upperarm_r": 39, "lowerarm_r": 40, "hand_r": 42,
    "clavicle_l": 74, "upperarm_l": 75, "lowerarm_l": 76, "hand_l": 78,
    "neck_01": 110, "head": 113,
}

ROOT_JOINT = 0
CM_PER_M = 100.0


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for (..., 4) xyzw quaternions."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    """Inverse of the rig's euler_xyz_to_quaternion (extrinsic XYZ, R = Rz Ry Rx)."""
    return Rotation.from_quat(np.asarray(q, dtype=np.float64)).as_euler("xyz")


@dataclass
class ReferenceFit:
    """The SAM 3D Body fit of the reference photo, in rig-native form."""
    joint_params: np.ndarray      # (127, 7)
    shape_params: np.ndarray      # (1, 45) identity coefficients
    expr_params: np.ndarray       # (1, 72)
    model_params: np.ndarray      # (1, P) momentum model parameters
    cam_t: np.ndarray             # (3,) metres, SAM 3D camera translation
    focal_length: float           # pixels, in reference-image space
    image_size: tuple[int, int]   # (width, height) of the reference image


class MhrRig:
    """Thin wrapper over the SAM 3D Body MHR rig."""

    def __init__(self, device: str | None = None):
        from jakkanna_pack.jakkanna_sam3d.processing.load_model import LoadSAM3DBodyModel
        from jakkanna_pack.jakkanna_sam3d.processing.process import _load_sam3d_model

        loaded = _load_sam3d_model(LoadSAM3DBodyModel().load_model("Auto")[0])
        self.device = torch.device(device or loaded["device"])
        self.model = loaded["model"]
        self.head = self.model.head_pose
        self.mhr = self.head.mhr
        self.character = self.mhr.character_torch

        skeleton = self.character.skeleton
        self.parents = skeleton.joint_parents.cpu().numpy().astype(np.int64)
        self.prerotations = skeleton.joint_prerotations.cpu().numpy().astype(np.float64)
        self.translation_offsets = skeleton.joint_translation_offsets.cpu().numpy().astype(np.float64)
        self.num_joints = int(self.parents.shape[0])
        self.faces = self.head.faces.cpu().numpy().astype(np.int64)

    # -- fitting -----------------------------------------------------------
    def reference_fit(self, pose_json: dict) -> ReferenceFit:
        """Re-run mhr_forward on a stored SAM 3D pose to recover model parameters."""
        from jakkanna_pack.jakkanna_sam3d.processing.process import _to_batched_tensor

        dev = self.device
        head = self.head
        shape_params = _to_batched_tensor(pose_json.get("shape_params"), dev, width=head.num_shape_comps)
        scale_params = _to_batched_tensor(pose_json.get("scale_params"), dev, width=head.num_scale_comps)
        expr_params = torch.zeros((1, head.num_face_comps), dtype=torch.float32, device=dev)

        out = head.mhr_forward(
            global_trans=torch.zeros((1, 3), dtype=torch.float32, device=dev),
            global_rot=_to_batched_tensor(pose_json.get("global_rot"), dev, width=3),
            body_pose_params=_to_batched_tensor(pose_json.get("body_pose_params"), dev, width=133),
            hand_pose_params=_to_batched_tensor(pose_json.get("hand_pose_params"), dev, width=108),
            scale_params=scale_params,
            shape_params=shape_params,
            expr_params=expr_params,
            return_model_params=True,
        )
        model_params = next(t for t in out[1:] if t.ndim == 2 and t.shape[-1] > 3)

        joint_params = self.joint_params_from_model_params(model_params, shape_params)
        image_size = pose_json.get("image_size", {})
        return ReferenceFit(
            joint_params=joint_params,
            shape_params=shape_params.cpu().numpy(),
            expr_params=expr_params.cpu().numpy(),
            model_params=model_params.cpu().numpy(),
            cam_t=np.asarray(pose_json["camera"], dtype=np.float64).reshape(3),
            focal_length=float(pose_json["focal_length"]),
            image_size=(int(image_size.get("width", 0)), int(image_size.get("height", 0))),
        )

    def joint_params_from_model_params(self, model_params, shape_params) -> np.ndarray:
        """Expand momentum model parameters into (127, 7) joint parameters."""
        full = torch.cat([model_params, torch.zeros_like(shape_params)], dim=1)
        joint_params = self.character.model_parameters_to_joint_parameters(full)
        return joint_params.reshape(-1, self.num_joints, 7)[0].cpu().numpy().astype(np.float64)

    # -- kinematics --------------------------------------------------------
    def local_state(self, joint_params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(translation, quaternion, scale) of each joint relative to its parent."""
        trans = joint_params[:, 0:3] + self.translation_offsets
        quat = quat_mul(self.prerotations, Rotation.from_euler("xyz", joint_params[:, 3:6]).as_quat())
        scale = np.exp(joint_params[:, 6] * np.log(2.0))
        return trans, quat, scale

    def world_state(self, joint_params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(position, quaternion, scale) of each joint in world space, centimetres."""
        lt, lq, ls = self.local_state(joint_params)
        pos = np.zeros_like(lt)
        quat = np.zeros_like(lq)
        scale = np.zeros_like(ls)
        for j in range(self.num_joints):
            p = self.parents[j]
            if p < 0:
                pos[j], quat[j], scale[j] = lt[j], lq[j], ls[j]
                continue
            rotated = Rotation.from_quat(quat[p]).apply(lt[j] * scale[p])
            pos[j] = pos[p] + rotated
            quat[j] = quat_mul(quat[p], lq[j])
            scale[j] = scale[p] * ls[j]
        return pos, quat, scale

    def set_world_rotations(self, joint_params: np.ndarray, targets: dict[int, np.ndarray]) -> np.ndarray:
        """Return joint parameters whose world rotations match `targets` for the
        given joints, leaving every other joint's local rotation untouched.

        Joints are processed parent-first so each solve sees its final parent.
        """
        out = joint_params.copy()
        _, local_q, _ = self.local_state(out)
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        world_q = np.zeros_like(local_q)
        for j in range(self.num_joints):
            p = self.parents[j]
            parent_q = world_q[p] if p >= 0 else identity
            if j in targets:
                desired = targets[j] / np.linalg.norm(targets[j])
                euler_q = quat_mul(quat_conj(self.prerotations[j]), quat_mul(quat_conj(parent_q), desired))
                out[j, 3:6] = quat_to_euler_xyz(euler_q)
                world_q[j] = desired
            else:
                world_q[j] = quat_mul(parent_q, local_q[j])
        return out

    # -- posing ------------------------------------------------------------
    def skeleton_state(self, joint_params: np.ndarray) -> torch.Tensor:
        jp = torch.as_tensor(joint_params.reshape(1, -1), dtype=torch.float32, device=self.device)
        return self.character.joint_parameters_to_skeleton_state(jp)

    def vertices(self, joint_params: np.ndarray, fit: ReferenceFit,
                 apply_correctives: bool = True) -> np.ndarray:
        """Skin the mesh for the given joint parameters. Returns metres."""
        dev = self.device
        shape = torch.as_tensor(fit.shape_params, dtype=torch.float32, device=dev)
        expr = torch.as_tensor(fit.expr_params, dtype=torch.float32, device=dev)
        jp = torch.as_tensor(joint_params.reshape(1, -1), dtype=torch.float32, device=dev)

        skel_state = self.character.joint_parameters_to_skeleton_state(jp)
        unposed = self.character.blend_shape.forward(shape)
        unposed = unposed + self.mhr.face_expressions_model.forward(expr)
        if apply_correctives:
            unposed = unposed + self.mhr.pose_correctives_model.forward(jp)
        verts = self.character.skin_points(skel_state, unposed)
        return (verts[0] / CM_PER_M).cpu().numpy().astype(np.float64)
