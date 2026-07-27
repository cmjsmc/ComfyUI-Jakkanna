#!/usr/bin/env python3
"""Retire superseded prepared runs from the image/FBX cache.

Preparation is content-addressed on the inputs, the settings, *and* the Jakkanna
release identity. Keying on the release means a commit that cannot change a
pixel still mints a fresh run directory, so the cache accumulates one full
81-frame render per commit per input combination and never sheds any of it.

The runs are not interchangeable, though: outputs from different commits really
do differ, so this keeps the most recent run for each combination of inputs and
settings rather than deduplicating. Older ones were produced by code that is no
longer installed, and regenerating one costs a body fit plus an 81-frame render.

Dry run by default. Nothing is deleted without --apply.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from _env import COMFY_ROOT, INPUT_ROOT, RUN_ROOT


# The settings run_all.py hashes into the run identity, minus the release. Two
# runs agreeing on all of these differ only by the code that produced them.
IDENTITY_SETTINGS = (
    "frames", "fps", "start_seconds", "timing", "width", "height", "framing",
    "camera_preset", "margin", "ease_frames", "root_motion", "supersample",
    "bbox_threshold",
)

CACHE_DIRS = ("_source_cache", "_subject_mask_cache", "_fit_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=RUN_ROOT,
                        help="prepared-run directory (default: input/jakkanna_animate)")
    parser.add_argument("--keep", type=int, default=1,
                        help="most recent runs to keep per input+settings combination")
    parser.add_argument("--workflows", type=Path, nargs="*",
                        help="directories of workflow JSON to scan for run references; "
                             "a referenced run is never retired")
    parser.add_argument("--apply", action="store_true",
                        help="actually delete; without it nothing is touched")
    return parser.parse_args()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_runs(root: Path) -> list[dict]:
    runs = []
    for manifest_path in sorted(root.glob("*/preprocess_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        if manifest.get("schema") != "jakkanna.image_fbx_animation.v1":
            continue
        inputs = manifest.get("inputs", {})
        settings = manifest.get("settings", {})
        runs.append({
            "path": manifest_path.parent,
            "mtime": manifest_path.stat().st_mtime,
            "size": directory_size(manifest_path.parent),
            "identity": manifest.get("software", {}).get("jakkanna_release_identity"),
            "image": inputs.get("image", {}).get("path"),
            "subject_mask": inputs.get("subject_mask", {}).get("path"),
            "key": json.dumps({
                "image": inputs.get("image", {}).get("sha256"),
                "fbx": inputs.get("fbx", {}).get("sha256"),
                "subject_mask": inputs.get("subject_mask", {}).get("sha256"),
                "settings": {name: settings.get(name) for name in IDENTITY_SETTINGS},
            }, sort_keys=True),
        })
    return runs


def workflow_references(directories: list[Path]) -> str:
    """Concatenated workflow JSON, searched for run directory names."""
    text = ""
    for directory in directories or ():
        if not directory.is_dir():
            continue
        for workflow in sorted(directory.glob("*.json")):
            try:
                text += workflow.read_text(errors="replace")
            except OSError:
                continue
    return text


def select(runs: list[dict], keep: int, workflow_text: str) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        grouped.setdefault(run["key"], []).append(run)

    kept, retire = [], []
    for group in grouped.values():
        group.sort(key=lambda run: run["mtime"], reverse=True)
        for index, run in enumerate(group):
            if index < keep:
                run["reason"] = "newest" if index == 0 else f"kept ({index + 1} of {keep})"
                kept.append(run)
            elif run["path"].name in workflow_text:
                run["reason"] = "referenced by a workflow"
                kept.append(run)
            else:
                retire.append(run)
    kept.sort(key=lambda run: run["path"].name)
    retire.sort(key=lambda run: run["path"].name)
    return kept, retire


def orphaned_cache_entries(root: Path, surviving: list[dict]) -> list[Path]:
    """Normalised inputs no surviving run refers to any more."""
    live = set()
    for run in surviving:
        for key in ("image", "subject_mask"):
            value = run.get(key)
            if value:
                live.add(Path(value).name)
    orphans = []
    for name in ("_source_cache", "_subject_mask_cache"):
        directory = root / name
        if not directory.is_dir():
            continue
        orphans.extend(entry for entry in sorted(directory.glob("*.png"))
                       if entry.name not in live)
    return orphans


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if root != INPUT_ROOT and INPUT_ROOT not in root.parents:
        raise ValueError(f"run root must be inside {INPUT_ROOT}")
    if args.keep < 1:
        raise ValueError("--keep must be at least 1: every combination keeps a current run")

    directories = args.workflows
    if directories is None:
        directories = [
            COMFY_ROOT / "user" / "default" / "workflows",
            Path(__file__).resolve().parents[2] / "workflows",
        ]
    workflow_text = workflow_references(directories)

    runs = load_runs(root)
    if not runs:
        print(f"No prepared runs under {root}")
        return
    kept, retire = select(runs, args.keep, workflow_text)
    orphans = orphaned_cache_entries(root, kept)

    print(f"{len(runs)} prepared run(s) under {root.relative_to(COMFY_ROOT)}\n")
    print(f"keeping {len(kept)}:")
    for run in kept:
        print(f"  {run['path'].name:<48}{run['size'] / 2**20:7.1f} MiB  {run['reason']}")
    print(f"\nretiring {len(retire)}:")
    for run in retire:
        identity = run["identity"] or "unrecorded release"
        print(f"  {run['path'].name:<48}{run['size'] / 2**20:7.1f} MiB  superseded ({identity})")
    if orphans:
        print(f"\norphaned normalised inputs ({len(orphans)}):")
        for entry in orphans:
            print(f"  {entry.relative_to(root)}  {entry.stat().st_size / 2**20:.1f} MiB")

    reclaim = sum(run["size"] for run in retire) + sum(e.stat().st_size for e in orphans)
    print(f"\nreclaimable: {reclaim / 2**20:.0f} MiB")
    if not retire and not orphans:
        return
    if not args.apply:
        print("\nDry run. Re-run with --apply to delete.")
        return

    for run in retire:
        shutil.rmtree(run["path"])
        print(f"removed {run['path'].name}")
    for entry in orphans:
        entry.unlink()
        print(f"removed {entry.relative_to(root)}")
    print(f"\nreclaimed {reclaim / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
