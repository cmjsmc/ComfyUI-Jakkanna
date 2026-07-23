#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--texture-size 512|1024|2048|4096] WORKFLOW DATASET [OUTPUT.glb]"
}

texture_size=2048
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ "${1:-}" == "--texture-size" ]]; then
    if [[ $# -lt 2 ]]; then
        usage >&2
        exit 2
    fi
    texture_size="$2"
    shift 2
fi
if [[ "$texture_size" != 512 && "$texture_size" != 1024 && "$texture_size" != 2048 && "$texture_size" != 4096 ]]; then
    echo "Texture size must be 512, 1024, 2048, or 4096." >&2
    exit 2
fi
if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi

workflow="$(realpath "$1")"
dataset="$(realpath "$2")"
output="$(realpath -m "${3:-$dataset/JakkannaSCAILProxy.glb}")"
if [[ "${output,,}" != *.glb ]]; then
    echo "Output model must use the .glb extension." >&2
    exit 1
fi
if [[ ! -f "$workflow" ]]; then
    echo "Workflow does not exist: $workflow" >&2
    exit 1
fi
if [[ ! -f "$dataset/manifest.json" ]]; then
    echo "Dataset does not contain manifest.json: $dataset" >&2
    exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
comfy_root="$(realpath "$repo_root/../..")"
python="$comfy_root/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    echo "ComfyUI virtual environment Python is missing: $python" >&2
    exit 1
fi

proxy="$dataset/JakkannaMannequinProxy.glb"
metadata="$dataset/JakkannaMannequinProxy.json"
if [[ "$output" == "$proxy" ]]; then
    echo "Output model must differ from the intermediate mannequin proxy." >&2
    exit 1
fi

alignment_report="$dataset/RealityScanAlignmentReport.txt"
if [[ -f "$alignment_report" ]]; then
    registered="$(
        sed -n 's/^JAKKANNA_REGISTERED_CAMERAS=//p' "$alignment_report" |
            tail -n 1 |
            tr -d '\r'
    )"
    if [[ "$registered" != "81" ]]; then
        echo "RealityScan validation report does not contain an 81-camera alignment." >&2
        exit 1
    fi
fi

"$python" "$repo_root/tools/export_posed_mesh.py" \
    "$workflow" \
    --pose-index 0 \
    --output "$proxy" \
    --joints "$metadata"

"$python" "$repo_root/tools/bake_scail_turntable.py" \
    "$proxy" \
    "$metadata" \
    "$dataset" \
    --output "$output" \
    --texture-size "$texture_size"

if [[ -f "$alignment_report" ]]; then
    echo "RealityScan validation: 81/81 cameras"
else
    echo "RealityScan validation: not run (optional)"
fi
