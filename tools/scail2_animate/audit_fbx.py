#!/usr/bin/env python3
"""Audit Mixamo-compatible FBX files for the greenfield SCAIL-2 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fbx import FbxClip
from retarget import JOINT_MAP


def automatic_timing(duration: float, frames: int = 81, fps: float = 16.0) -> str:
    required_end = (frames - 1) / fps
    return "original_speed" if duration + 1e-6 >= required_end else "fit_clip"


def audit(path: Path) -> dict:
    clip = FbxClip(path)
    required = set(JOINT_MAP.values())
    by_name = {
        bone.name.replace("mixamorig:", ""): bone.uid
        for bone in clip.bones.values()
    }
    missing = sorted(required - set(by_name))
    mapped_animated = {
        clip.bones[uid].name.replace("mixamorig:", "")
        for uid in clip.animated_bones()
        if uid in clip.bones
    } & required

    maximum_channel_samples = 0
    timeline = []
    for channels in clip.curves.values():
        for axes in channels.values():
            for times, _values in axes.values():
                maximum_channel_samples = max(maximum_channel_samples, len(times))
                timeline.extend(float(value) for value in times)

    transforms_finite = True
    determinant_range = [float("inf"), float("-inf")]
    for time in np.linspace(0.0, clip.duration, min(9, max(2, int(clip.duration) + 1))):
        for matrix in clip.world_matrices(float(time)).values():
            transforms_finite = transforms_finite and bool(np.isfinite(matrix).all())
            determinant = float(np.linalg.det(matrix[:3, :3]))
            determinant_range[0] = min(determinant_range[0], determinant)
            determinant_range[1] = max(determinant_range[1], determinant)

    hips_travel = None
    hips_maximum_excursion = None
    hips_uid = by_name.get("Hips")
    if hips_uid is not None:
        start = clip.world_matrices(0.0)[hips_uid][:3, 3]
        end = clip.world_matrices(clip.duration)[hips_uid][:3, 3]
        hips_travel = float(np.linalg.norm(end - start))
        hips_maximum_excursion = max(
            float(np.linalg.norm(clip.world_matrices(float(time))[hips_uid][:3, 3] - start))
            for time in np.linspace(0.0, clip.duration, 81)
        )

    timing = automatic_timing(clip.duration)
    compatible = (
        not missing
        and len(mapped_animated) == len(required)
        and maximum_channel_samples >= 81
        and transforms_finite
        and determinant_range[0] > 0
    )
    return {
        "file": path.name,
        "file_size_bytes": path.stat().st_size,
        "fbx_version": clip.version,
        "bones": len(clip.bones),
        "animated_bones": len(clip.animated_bones()),
        "mapped_bones": len(required) - len(missing),
        "mapped_animated_bones": len(mapped_animated),
        "missing_mapped_bones": missing,
        "duration_seconds": clip.duration,
        "timeline_samples": len(set(timeline)),
        "maximum_channel_samples": maximum_channel_samples,
        "transforms_finite": transforms_finite,
        "determinant_range": determinant_range,
        "hips_endpoint_travel": hips_travel,
        "hips_maximum_excursion": hips_maximum_excursion,
        "automatic_timing": timing,
        "compatible": compatible,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path.home() / "Code" / "ComfyUI" / "input" / "animate",
    )
    args = parser.parse_args()
    paths = sorted(args.directory.glob("*.fbx"), key=lambda path: path.name.lower())
    reports = []
    for path in paths:
        try:
            reports.append(audit(path))
        except Exception as error:
            reports.append({
                "file": path.name,
                "compatible": False,
                "error": f"{type(error).__name__}: {error}",
            })
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
