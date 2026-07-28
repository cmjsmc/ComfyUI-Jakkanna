export const DEFAULT_CAPTURE_CAMERA_PARAMS = Object.freeze({
    offset_x: 0,
    offset_y: 0,
    zoom: 1,
    yaw_deg: 0,
    pitch_deg: 0,
});

function finiteNumber(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
}

export function resolveCaptureCameraParams(params = {}, fallback = DEFAULT_CAPTURE_CAMERA_PARAMS) {
    const resolvedFallback = {
        ...DEFAULT_CAPTURE_CAMERA_PARAMS,
        ...(fallback && typeof fallback === "object" ? fallback : {}),
    };
    const source = params && typeof params === "object" ? params : {};
    return {
        offset_x: finiteNumber(source.offset_x, finiteNumber(resolvedFallback.offset_x, 0)),
        offset_y: finiteNumber(source.offset_y, finiteNumber(resolvedFallback.offset_y, 0)),
        zoom: finiteNumber(source.zoom, finiteNumber(resolvedFallback.zoom, 1)),
        yaw_deg: finiteNumber(source.yaw_deg, finiteNumber(resolvedFallback.yaw_deg, 0)),
        pitch_deg: finiteNumber(source.pitch_deg, finiteNumber(resolvedFallback.pitch_deg, 0)),
    };
}

export function poseWithCameraParams(pose, cameraParams, fallback = DEFAULT_CAPTURE_CAMERA_PARAMS) {
    return {
        ...(pose && typeof pose === "object" ? pose : {}),
        cameraParams: resolveCaptureCameraParams(cameraParams, fallback),
    };
}
