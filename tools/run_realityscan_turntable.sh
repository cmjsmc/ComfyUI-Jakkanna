#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--align-only|--resume] [--dry-run] DATASET [OUTPUT.glb]"
}

mode="reconstruct"
dry_run=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --align-only)
            mode="align"
            shift
            ;;
        --resume)
            mode="resume"
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

base_command=(
    realityscan-cli
    -headless
    -silent "$wine_crash_report"
    -set "appQuitOnError=true"
    -set "suppressErrors=true"
)

alignment_command=("${base_command[@]}")
if [[ "$mode" != "resume" ]]; then
    alignment_command+=(
        -newScene
        -set "appIncSubdirs=true"
        -set "sfmFeatureDetectionQuality=High"
        -set "sfmImagesOverlap=High"
        -set "sfmImageDownscaleFactor=1"
        -set "sfmMaxFeatureReprojectionError=2.0"
        -set "sfmDetectorSensitivity=High"
        -set "sfmEnableCameraPrior=true"
        -set "sfmCameraPriorAccuracyX=0.05"
        -set "sfmCameraPriorAccuracyY=0.05"
        -set "sfmCameraPriorAccuracyZ=0.05"
        -set "sfmCameraPriorWeight=10.0"
        -set "sfmMergeGeoreferencedComponents=true"
        -set "MvsGeometryGpuAccel=true"
        -addFolder "$wine_layers"
        -selectAllImages
        -setConstantCalibrationGroups
        -editInputSelection "inpMaskOpts=3"
    )

    for index in $(seq 0 80); do
        read -r x y < <(
            awk -v i="$index" 'BEGIN {
                angle = i * 2 * atan2(0, -1) / 81
                printf "%.12f %.12f\n", 4.5 * sin(angle), 4.5 * cos(angle)
            }'
        )
        printf -v filename "frame_%04d.png" "$index"
        wine_image="$(to_wine_path "$geometry/$filename")"
        alignment_command+=(
            -deselectAllImages
            -selectImage "$wine_image"
            -editInputSelection "inpPose=1"
            -editInputSelection "inpTx=$x"
            -editInputSelection "inpTy=$y"
            -editInputSelection "inpTz=1.0"
        )
    done

    alignment_command+=(
        -align
        -selectMaximalComponent
        -setReconstructionRegionAuto
        -save "$wine_project"
        -printReport 'JAKKANNA_REGISTERED_CAMERAS=$(cameraCount)$[n]'
        -quit
    )
else
    if [[ ! -s "$project" ]]; then
        echo "Cannot resume because the RealityScan project does not exist: $project" >&2
        exit 1
    fi
    alignment_command+=(
        -load "$wine_project"
        -selectMaximalComponent
        -printReport 'JAKKANNA_REGISTERED_CAMERAS=$(cameraCount)$[n]'
        -quit
    )
fi

reconstruction_command=(
    "${base_command[@]}"
    -load "$wine_project"
    -selectMaximalComponent
    -setReconstructionRegionAuto
    -calculateNormalModel
    -calculateTexture
    -save "$wine_project"
    -exportSelectedModel "$wine_export"
    -quit
)

if "$dry_run"; then
    printf '%q ' "${alignment_command[@]}"
    printf '\n'
    if [[ "$mode" != "align" ]]; then
        printf '%q ' "${reconstruction_command[@]}"
        printf '\n'
    fi
    exit 0
fi

mkdir -p "$(dirname "$export_path")"
alignment_report="$dataset/RealityScanAlignmentReport.txt"
DISPLAY="${DISPLAY:-:0}" "${alignment_command[@]}" | tee "$alignment_report"

if [[ ! -s "$project" ]]; then
    echo "RealityScan did not create the expected project: $project" >&2
    exit 1
fi

registered_cameras="$(
    sed -n 's/^JAKKANNA_REGISTERED_CAMERAS=//p' "$alignment_report" | tail -n 1 | tr -d '\r'
)"
if [[ "$registered_cameras" != "81" ]]; then
    echo "RealityScan aligned ${registered_cameras:-an unknown number of} cameras; expected all 81. Reconstruction was not started." >&2
    exit 1
fi

echo "RealityScan aligned all 81 cameras."
echo "RealityScan project: $project"
if [[ "$mode" == "align" ]]; then
    exit 0
fi

DISPLAY="${DISPLAY:-:0}" "${reconstruction_command[@]}"
if [[ ! -s "$export_path" ]]; then
    echo "RealityScan did not create the expected GLB: $export_path" >&2
    exit 1
fi

echo "RealityScan model: $export_path"
