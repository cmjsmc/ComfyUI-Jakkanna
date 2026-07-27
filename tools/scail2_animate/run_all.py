#!/usr/bin/env python3
"""Fit a reference image and render a registered FBX driver for SCAIL-2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from _env import (
    COMFY_ROOT,
    INPUT_ROOT,
    RUN_ROOT,
    file_sha256,
    package_version,
    release_identity,
    safe_name,
)


HERE = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(HERE / script), *args]
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    result = subprocess.run(command, cwd=HERE)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed with exit code {result.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--fbx", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--timing", choices=("original_speed", "auto", "fit_clip"), default="original_speed")
    parser.add_argument("--framing", choices=("full_body", "reference_crop"), default="full_body")
    parser.add_argument("--margin", type=float, default=0.06)
    parser.add_argument("--ease-frames", type=int, default=12)
    parser.add_argument("--root-motion", choices=("preserve", "in_place", "auto"), default="preserve")
    parser.add_argument("--supersample", type=int, default=3)
    parser.add_argument("--bbox-threshold", type=float, default=0.8)
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--result-file", type=Path)
    return parser.parse_args()


def run_identity(image_hash: str, fbx_hash: str, settings: dict, identity: str) -> str:
    payload = {
        "image_sha256": image_hash,
        "fbx_sha256": fbx_hash,
        "settings": settings,
        "jakkanna_release_identity": identity,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fit_cache_key(image_hash: str, identity: str, bbox_threshold: float) -> str:
    return hashlib.sha256(
        f"{image_hash}:{identity}:{bbox_threshold:g}".encode()
    ).hexdigest()[:16]


def existing_run_matches(
    manifest: dict,
    run_id: str,
    image_hash: str,
    fbx_hash: str,
    settings: dict,
    identity: str,
) -> bool:
    return (
        manifest.get("schema") == "jakkanna.image_fbx_animation.v1"
        and manifest.get("run_id") == run_id
        and manifest.get("inputs", {}).get("image", {}).get("sha256") == image_hash
        and manifest.get("inputs", {}).get("fbx", {}).get("sha256") == fbx_hash
        and all(manifest.get("settings", {}).get(name) == value for name, value in settings.items())
        and manifest.get("software", {}).get("jakkanna_release_identity") == identity
    )


def write_result(path: Path | None, run_dir: Path) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_dir": str(run_dir)}, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    image = args.image.resolve()
    fbx = args.fbx.resolve()
    output_root = args.output_root.resolve()
    if not image.is_file():
        raise FileNotFoundError(f"reference image does not exist: {image}")
    if not fbx.is_file():
        raise FileNotFoundError(f"FBX animation does not exist: {fbx}")
    if output_root != INPUT_ROOT and INPUT_ROOT not in output_root.parents:
        raise ValueError(f"output-root must be inside {INPUT_ROOT}")

    image_hash = file_sha256(image)
    fbx_hash = file_sha256(fbx)
    version = package_version()
    identity = release_identity()
    settings = {
        "frames": args.frames,
        "fps": args.fps,
        "start_seconds": args.start,
        "timing": args.timing,
        "width": args.width,
        "height": args.height,
        "framing": args.framing,
        "margin": args.margin,
        "ease_frames": args.ease_frames,
        "root_motion": args.root_motion,
        "supersample": args.supersample,
        "bbox_threshold": args.bbox_threshold,
    }
    digest = run_identity(image_hash, fbx_hash, settings, identity)
    prefix = safe_name(args.name or f"{image.stem}-{fbx.stem}")
    run_id = f"{prefix}-{digest[:12]}"
    run_dir = output_root / run_id
    manifest_path = run_dir / "preprocess_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if not existing_run_matches(manifest, run_id, image_hash, fbx_hash, settings, identity):
            raise ValueError(f"existing prepared run does not match its content identity: {run_dir}")
        print(f"Prepared run already exists: {run_dir}")
        write_result(args.result_file, run_dir)
        return
    if run_dir.exists():
        raise FileExistsError(
            f"incomplete prepared run exists without a manifest: {run_dir}; "
            "move it aside after reviewing any diagnostics, then retry"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    staging_dir = output_root / f".{run_id}.partial-{os.getpid()}"
    staging_dir.mkdir()
    review_dir = staging_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    fit_key = fit_cache_key(image_hash, identity, args.bbox_threshold)
    fit_cache = output_root / "_fit_cache" / fit_key
    cached_pose = fit_cache / "pose.json"
    if not cached_pose.is_file():
        run(
            "fit_reference.py",
            "--image", str(image),
            "--out", str(fit_cache),
            "--bbox-threshold", str(args.bbox_threshold),
        )
    else:
        print(f"=== reusing content-keyed fit {cached_pose}")

    fit_dir = staging_dir / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    fit_path = fit_dir / "pose.json"
    if not fit_path.exists():
        shutil.copy2(cached_pose, fit_path)

    if not args.skip_verify:
        common = (
            "--image", str(image),
            "--fit", str(fit_path),
            "--out", str(review_dir),
        )
        run("verify_fit.py", *common)
        run("verify_hands.py", *common)

    run(
        "build_driver.py",
        "--image", str(image),
        "--fbx", str(fbx),
        "--fit", str(fit_path),
        "--run-dir", str(staging_dir),
        "--run-id", run_id,
        "--width", str(args.width),
        "--height", str(args.height),
        "--frames", str(args.frames),
        "--fps", str(args.fps),
        "--start", str(args.start),
        "--timing", args.timing,
        "--framing", args.framing,
        "--margin", str(args.margin),
        "--ease-frames", str(args.ease_frames),
        "--root-motion", args.root_motion,
        "--supersample", str(args.supersample),
        "--bbox-threshold", str(args.bbox_threshold),
        "--jakkanna-version", version,
        "--jakkanna-release-identity", identity,
    )

    staging_dir.rename(run_dir)
    write_result(args.result_file, run_dir)
    relative_run = run_dir.relative_to(COMFY_ROOT)
    print(
        "\nPrepared input ready:\n"
        f"  {relative_run}\n"
        "Review the fit overlays and driver contact sheet before generating or running the workflow.\n"
        f"Next: {sys.executable} {HERE / 'make_workflow.py'} --run-dir {run_dir}"
    )


if __name__ == "__main__":
    main()
