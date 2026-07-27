"""Shared paths and ComfyUI import bootstrap for image/FBX animation tools."""

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path


JAKKANNA_DIR = Path(__file__).resolve().parents[2]
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", JAKKANNA_DIR.parents[1])).resolve()
INPUT_ROOT = COMFY_ROOT / "input"
RUN_ROOT = INPUT_ROOT / "jakkanna_animate"
JAKKANNA_PKG = "jakkanna_pack"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=JAKKANNA_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and len(commit) == 40 else "unknown"


def package_version() -> str:
    match = re.search(
        r'(?m)^version\s*=\s*"([^"]+)"',
        (JAKKANNA_DIR / "pyproject.toml").read_text(encoding="utf-8"),
    )
    if match is None:
        raise ValueError("Jakkanna package version is missing from pyproject.toml")
    version = match.group(1)
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[+-][0-9A-Za-z.-]+)?", version):
        raise ValueError(f"invalid Jakkanna package version: {version!r}")
    return version


def release_identity() -> str:
    version = package_version()
    commit = repository_commit()
    return f"{version}+{commit}" if commit != "unknown" else version


def safe_name(value: str) -> str:
    name = _SAFE_NAME.sub("-", value).strip("-_").lower()
    if not name:
        raise ValueError("run name must contain at least one letter or number")
    return name[:64]


def bootstrap() -> None:
    """Expose ComfyUI and this hyphenated custom-node package to tool scripts."""
    import types

    entry = str(COMFY_ROOT)
    if entry not in sys.path:
        sys.path.insert(0, entry)

    if JAKKANNA_PKG not in sys.modules:
        package = types.ModuleType(JAKKANNA_PKG)
        package.__path__ = [str(JAKKANNA_DIR)]
        sys.modules[JAKKANNA_PKG] = package
