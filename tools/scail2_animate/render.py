#!/usr/bin/env python3
"""Z-buffered mesh rasteriser for the mannequin driver.

Uses the same pinhole convention as SAM 3D Body's own viewer, so a mesh rendered
here lands exactly where the subject sits in the reference photograph:

    camera space = (x, -y, -z) + cam_t
    u = x * f / z + cx,   v = y * f / z + cy

Triangles are bucketed by bounding-box size and rasterised exactly over their
own pixel box, so coverage does not depend on triangle area. Visibility is
resolved by a scatter-min over a packed depth/face key, which handles
self-occlusion (an arm crossing the torso) correctly.
"""
from __future__ import annotations

import numpy as np
import torch

MANNEQUIN_RGB = (206, 214, 222)
FACE_ID_BITS = 20


def project(vertices: np.ndarray, cam_t: np.ndarray, focal: float,
            width: int, height: int,
            principal: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """World-space metres -> (pixels, depth). Mirrors SAM 3D Body's viewer."""
    verts = np.asarray(vertices, dtype=np.float64).copy()
    verts[:, 1] *= -1.0
    verts[:, 2] *= -1.0
    verts += np.asarray(cam_t, dtype=np.float64).reshape(1, 3)

    cx, cy = principal if principal is not None else (width * 0.5, height * 0.5)
    z = np.maximum(verts[:, 2], 1e-4)
    u = verts[:, 0] * focal / z + cx
    v = verts[:, 1] * focal / z + cy
    return np.stack([u, v], axis=1), verts[:, 2]


def camera_space(vertices: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64).copy()
    verts[:, 1] *= -1.0
    verts[:, 2] *= -1.0
    return verts + np.asarray(cam_t, dtype=np.float64).reshape(1, 3)


class Rasteriser:
    """Exact triangle rasteriser with a depth buffer, on GPU."""

    def __init__(self, faces: np.ndarray, device: torch.device | str = "cuda",
                 supersample: int = 2, max_box: int = 256):
        self.device = torch.device(device)
        self.faces = torch.as_tensor(faces, dtype=torch.long, device=self.device)
        self.supersample = int(supersample)
        self.max_box = int(max_box)
        if self.faces.shape[0] >= (1 << FACE_ID_BITS):
            raise ValueError("face count exceeds the packed depth/face key layout")

    def render(self, vertices_px: np.ndarray, depth: np.ndarray,
               width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (face id per pixel, depth per pixel); face id -1 is background."""
        ss = self.supersample
        w, h = width * ss, height * ss

        px = torch.as_tensor(vertices_px, dtype=torch.float32, device=self.device) * ss
        pz = torch.as_tensor(depth, dtype=torch.float32, device=self.device)

        tri = px[self.faces]                                     # (F, 3, 2)
        tri_z = pz[self.faces]                                   # (F, 3)

        edge1 = tri[:, 1] - tri[:, 0]
        edge2 = tri[:, 2] - tri[:, 0]
        area = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]

        lo = tri.amin(dim=1).floor()
        hi = tri.amax(dim=1).ceil()
        keep = (
            (area < 0)                                           # front facing
            & (tri_z > 1e-3).all(dim=1)                          # in front of camera
            & (hi[:, 0] >= 0) & (lo[:, 0] < w)
            & (hi[:, 1] >= 0) & (lo[:, 1] < h)
        )

        buffer = torch.full((h * w,), 1 << 60, dtype=torch.long, device=self.device)
        face_index = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if face_index.numel():
            box = (hi - lo).amax(dim=1)[face_index].clamp(min=1)
            # Bucket by power-of-two box size so each batch shares one offset grid.
            tier = torch.clamp(torch.ceil(torch.log2(box.float() + 1.0)), 0, 12).long()
            for t in torch.unique(tier).tolist():
                sel = face_index[tier == t]
                k = min(1 << t, self.max_box) + 1
                self._rasterise_bucket(buffer, tri[sel], tri_z[sel], lo[sel], sel, k, w, h)

        hit = buffer < (1 << 60)
        face_buf = torch.where(hit, buffer & ((1 << FACE_ID_BITS) - 1),
                               torch.full_like(buffer, -1))
        depth_buf = torch.where(hit, (buffer >> FACE_ID_BITS).float() / 4096.0,
                                torch.full_like(buffer, float("inf"), dtype=torch.float32))
        return face_buf.view(h, w).cpu().numpy(), depth_buf.view(h, w).cpu().numpy()

    def _rasterise_bucket(self, buffer, tri, tri_z, lo, face_ids, k, w, h,
                          chunk_samples: int = 1 << 24) -> None:
        """Rasterise one bucket of triangles over a shared k x k pixel box."""
        steps = torch.arange(k, device=self.device, dtype=torch.float32)
        oy, ox = torch.meshgrid(steps, steps, indexing="ij")
        offsets = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=1)   # (K, 2)

        per_tri = offsets.shape[0]
        stride = max(1, chunk_samples // max(per_tri, 1))
        for start in range(0, tri.shape[0], stride):
            end = min(start + stride, tri.shape[0])
            t = tri[start:end]
            tz = tri_z[start:end]
            ids = face_ids[start:end]

            # Pixel centres inside each triangle's own bounding box.
            pts = lo[start:end].unsqueeze(1) + offsets.unsqueeze(0) + 0.5   # (T, K, 2)

            p0, p1, p2 = t[:, 0:1], t[:, 1:2], t[:, 2:3]
            w0 = self._cross(p1 - pts, p2 - pts)
            w1 = self._cross(p2 - pts, p0 - pts)
            w2 = self._cross(p0 - pts, p1 - pts)
            total = (w0 + w1 + w2)

            xi = pts[..., 0].long()
            yi = pts[..., 1].long()
            inside = (
                (w0 <= 0) & (w1 <= 0) & (w2 <= 0)                 # front faces wind negative
                & (total.abs() > 1e-9)
                & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            )
            if not bool(inside.any()):
                continue

            denom = torch.where(total.abs() > 1e-9, total, torch.ones_like(total))
            b0, b1, b2 = w0 / denom, w1 / denom, w2 / denom
            # Perspective-correct depth.
            inv_z = (b0 / tz[:, 0:1] + b1 / tz[:, 1:2] + b2 / tz[:, 2:3]).clamp(min=1e-6)
            z = 1.0 / inv_z

            flat = (yi * w + xi)[inside]
            z_key = (z[inside] * 4096.0).clamp(0, float((1 << 40) - 1)).long()
            key = (z_key << FACE_ID_BITS) | ids.unsqueeze(1).expand_as(z)[inside]
            buffer.scatter_reduce_(0, flat, key, reduce="amin")

    @staticmethod
    def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def shade(self, vertices_cam: np.ndarray, vertices_px: np.ndarray, depth: np.ndarray,
              width: int, height: int,
              base_rgb: tuple[int, int, int] = MANNEQUIN_RGB,
              background: tuple[int, int, int] = (0, 0, 0)) -> tuple[np.ndarray, np.ndarray]:
        """Flat-shaded mannequin render plus its exact silhouette mask.

        `vertices_cam` must already be in camera space (see `camera_space`).
        Returns (rgb uint8 [H, W, 3], mask uint8 [H, W]).
        """
        ss = self.supersample
        face_buf, _ = self.render(vertices_px, depth, width, height)

        faces = self.faces.cpu().numpy()
        tri = np.asarray(vertices_cam, dtype=np.float64)[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

        # Camera sits at the origin looking down +z, so front faces have n_z < 0.
        key_light = np.array([-0.35, -0.45, -1.0])
        key_light /= np.linalg.norm(key_light)
        fill_light = np.array([0.55, -0.10, -1.0])
        fill_light /= np.linalg.norm(fill_light)

        lambert = np.clip(normals @ key_light, 0.0, 1.0)
        fill = np.clip(normals @ fill_light, 0.0, 1.0)
        shade = np.clip(0.30 + 0.55 * lambert + 0.22 * fill, 0.16, 1.0)

        face_rgb = np.clip(np.asarray(base_rgb, dtype=np.float64)[None, :] * shade[:, None], 0, 255)

        hit = face_buf >= 0
        rgb_hi = np.empty((*face_buf.shape, 3), dtype=np.float64)
        rgb_hi[:] = np.asarray(background, dtype=np.float64)
        rgb_hi[hit] = face_rgb[face_buf[hit]]
        mask_hi = hit.astype(np.float64)

        if ss > 1:
            rgb = rgb_hi.reshape(height, ss, width, ss, 3).mean(axis=(1, 3))
            mask = mask_hi.reshape(height, ss, width, ss).mean(axis=(1, 3))
        else:
            rgb, mask = rgb_hi, mask_hi

        return (np.clip(rgb, 0, 255).astype(np.uint8),
                (np.clip(mask, 0, 1) * 255).astype(np.uint8))
