import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
    DEFAULT_CAPTURE_CAMERA_PARAMS,
    poseWithCameraParams,
    resolveCaptureCameraParams,
} from "../web/jakkanna_pose_camera.mjs";


const poseStudioSource = await readFile(
    new URL("../web/jakkanna_pose_studio.js", import.meta.url),
    "utf8",
);

function methodSource(signature, nextSignature) {
    const start = poseStudioSource.indexOf(signature);
    assert.notEqual(start, -1, `missing method: ${signature}`);
    const end = poseStudioSource.indexOf(nextSignature, start + signature.length);
    assert.notEqual(end, -1, `missing boundary after: ${signature}`);
    return poseStudioSource.slice(start, end);
}

test("camera parameters preserve exact zero values and fill missing fields", () => {
    assert.deepEqual(
        resolveCaptureCameraParams(
            { offset_x: 0, zoom: 0.5, yaw_deg: -12 },
            { offset_x: 9, offset_y: 4, zoom: 2, yaw_deg: 7, pitch_deg: 3 },
        ),
        { offset_x: 0, offset_y: 4, zoom: 0.5, yaw_deg: -12, pitch_deg: 3 },
    );
    assert.deepEqual(resolveCaptureCameraParams(null), DEFAULT_CAPTURE_CAMERA_PARAMS);
});

test("saving camera parameters does not mutate the source pose", () => {
    const source = { bones: { head: [1, 2, 3] }, prompt: "turn left" };
    const stored = poseWithCameraParams(source, {
        offset_x: -1,
        offset_y: 2,
        zoom: 1.25,
        yaw_deg: 30,
        pitch_deg: -5,
    });

    assert.notEqual(stored, source);
    assert.equal(source.cameraParams, undefined);
    assert.deepEqual(stored.bones, source.bones);
    assert.deepEqual(stored.cameraParams, {
        offset_x: -1,
        offset_y: 2,
        zoom: 1.25,
        yaw_deg: 30,
        pitch_deg: -5,
    });
});

test("tab deletion restores the surviving pose camera before rendering", () => {
    const method = methodSource("deleteTab(targetIndex = -1)", "\n\n\n    resetCurrentPose()");
    const restore = method.indexOf("this.restoreActivePoseCameraParams({ updateViewer: false })");
    const render = method.indexOf("this.viewer.setPose");
    assert.ok(restore >= 0, "deleted tabs must restore the surviving pose camera");
    assert.ok(render > restore, "camera state must be restored before the surviving pose renders");
});

test("exports and library loads retain camera parameters", () => {
    const exportMethod = methodSource("exportPose(type, customName)", "\n    importPose()");
    assert.match(exportMethod, /poseWithCameraParams\(this\.viewer\.getPose\(\), this\.currentCameraParams\(\)\)/);
    assert.match(exportMethod, /cameraParams: this\.poses\[this\.activeTab\]\.cameraParams/);

    const libraryMethod = methodSource("async loadFromLibrary(poseOrName)", "\n    showSettingsModal()");
    const sourceCamera = libraryMethod.indexOf("const savedCamera = resolveCaptureCameraParams");
    const viewerPose = libraryMethod.indexOf("this.viewer.setPose");
    const storedPose = libraryMethod.indexOf("poseWithCameraParams", viewerPose);
    assert.ok(sourceCamera >= 0 && sourceCamera < viewerPose);
    assert.ok(storedPose > viewerPose);
});
