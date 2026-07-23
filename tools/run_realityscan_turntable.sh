#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--align-only] [--dry-run] DATASET [OUTPUT.glb]"
}

mode="reconstruct"
dry_run=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --align-only)
            mode="align"
            shift
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 2
fi
if ! command -v realityscan-cli >/dev/null 2>&1; then
    echo "realityscan-cli is not installed." >&2
    exit 1
fi

dataset="$(realpath "$1")"
layers="$dataset/layers"
geometry="$layers/.geometry"
masks="$layers/.mask"
if [[ ! -f "$dataset/manifest.json" || ! -d "$geometry" || ! -d "$masks" ]]; then
    echo "Dataset must contain manifest.json, layers/.geometry, and layers/.mask." >&2
    exit 1
fi

geometry_count="$(find "$geometry" -maxdepth 1 -type f -name 'frame_*.png' | wc -l)"
mask_count="$(find "$masks" -maxdepth 1 -type f -name 'frame_*.png' | wc -l)"
if [[ "$geometry_count" -ne 81 || "$mask_count" -ne 81 ]]; then
    echo "RealityScan dataset must contain 81 geometry images and 81 masks; got $geometry_count and $mask_count." >&2
    exit 1
fi
while IFS= read -r image; do
    if [[ ! -f "$masks/$(basename "$image")" ]]; then
        echo "Missing mask for $(basename "$image")." >&2
        exit 1
    fi
done < <(find "$geometry" -maxdepth 1 -type f -name 'frame_*.png' | sort)

project="$dataset/JakkannaRealityScan.rsproj"
export_path="${2:-$dataset/JakkannaRealityScan.glb}"
export_path="$(realpath -m "$export_path")"
if [[ "${export_path,,}" != *.glb ]]; then
    echo "Output model must use the .glb extension." >&2
    exit 1
fi

to_wine_path() {
    local path="$1"
    printf 'Z:%s' "${path//\//\\}"
}

wine_layers="$(to_wine_path "$layers")"
wine_project="$(to_wine_path "$project")"
wine_export="$(to_wine_path "$export_path")"
crash_report="$dataset/RealityScanCrashReport.txt"
wine_crash_report="$(to_wine_path "$crash_report")"

command=(
    realityscan-cli
    -headless
    -silent "$wine_crash_report"
    -newScene
    -set "appQuitOnError=true"
    -set "suppressErrors=true"
    -set "appIncSubdirs=true"
    -set "sfmFeatureDetectionQuality=High"
    -set "sfmImagesOverlap=High"
    -set "sfmImageDownscaleFactor=1"
    -set "sfmMaxFeatureReprojectionError=2.0"
    -set "sfmDetectorSensitivity=High"
    -set "MvsGeometryGpuAccel=true"
    -addFolder "$wine_layers"
    -selectAllImages
    -setConstantCalibrationGroups
    -editInputSelection "inpMaskOpts=3"
    -align
    -selectMaximalComponent
    -setReconstructionRegionAuto
    -save "$wine_project"
)

if [[ "$mode" == "reconstruct" ]]; then
    command+=(
        -calculateNormalModel
        -calculateTexture
        -save "$wine_project"
        -exportSelectedModel "$wine_export"
    )
fi
command+=(-quit)

if "$dry_run"; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$(dirname "$export_path")"
DISPLAY="${DISPLAY:-:0}" "${command[@]}"

if [[ ! -s "$project" ]]; then
    echo "RealityScan did not create the expected project: $project" >&2
    exit 1
fi
if [[ "$mode" == "reconstruct" && ! -s "$export_path" ]]; then
    echo "RealityScan did not create the expected GLB: $export_path" >&2
    exit 1
fi

echo "RealityScan project: $project"
if [[ "$mode" == "reconstruct" ]]; then
    echo "RealityScan model: $export_path"
fi
