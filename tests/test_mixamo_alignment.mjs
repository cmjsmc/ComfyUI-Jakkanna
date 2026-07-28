import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import * as THREE from "../web/three.module.js";
import { buildMixamoWorldKeypoints } from "../web/jakkanna_mixamo_import.js";


function sourceBone(point) {
    const position = new THREE.Vector3(...point);
    return {
        position,
        getWorldPosition(target) {
            return target.copy(position);
        },
    };
}

function sourceSkeleton(points) {
    const bones = {};
    const normalizedBones = {};
    for (const [name, point] of Object.entries(points)) {
        const bone = sourceBone(point);
        bones[name] = bone;
        normalizedBones[name.toLowerCase()] = bone;
    }
    return { bones, normalizedBones };
}

function viewerWithRestPoints(points) {
    return {
        THREE,
        _getBoneWorldPositionForImport(name) {
            const point = points[name];
            return point ? new THREE.Vector3(...point) : null;
        },
    };
}

test("Mixamo keypoints include toe bases and finger joints at target lengths", () => {
    const source = sourceSkeleton({
        Hips: [0, 1, 0],
        Neck: [0, 2, 0],
        Head: [0, 2.3, 0],
        LeftArm: [0.3, 1.9, 0],
        LeftForeArm: [0.8, 1.9, 0],
        LeftHand: [1.2, 1.9, 0],
        RightArm: [-0.3, 1.9, 0],
        RightForeArm: [-0.8, 1.9, 0],
        RightHand: [-1.2, 1.9, 0],
        LeftUpLeg: [0.2, 1, 0],
        LeftLeg: [0.2, 0.5, 0],
        LeftFoot: [0.2, 0, 0],
        LeftToeBase: [0.2, 0, 0.3],
        RightUpLeg: [-0.2, 1, 0],
        RightLeg: [-0.2, 0.5, 0],
        RightFoot: [-0.2, 0, 0],
        RightToeBase: [-0.2, 0, 0.3],
        LeftHandIndex1: [1.3, 1.9, 0],
        LeftHandIndex2: [1.4, 1.9, 0],
        LeftHandIndex3: [1.5, 1.9, 0],
        RightHandIndex1: [-1.3, 1.9, 0],
        RightHandIndex2: [-1.4, 1.9, 0],
        RightHandIndex3: [-1.5, 1.9, 0],
    });
    const rest = {
        pelvis: [0, 1, 0],
        neck_01: [0, 2, 0],
        head: [0, 2.3, 0],
        upperarm_l: [0.3, 1.9, 0],
        lowerarm_l: [0.8, 1.9, 0],
        hand_l: [1.2, 1.9, 0],
        upperarm_r: [-0.3, 1.9, 0],
        lowerarm_r: [-0.8, 1.9, 0],
        hand_r: [-1.2, 1.9, 0],
        thigh_l: [0.2, 1, 0],
        calf_l: [0.2, 0.5, 0],
        foot_l: [0.2, 0, 0],
        ball_l: [0.2, 0, 0.25],
        thigh_r: [-0.2, 1, 0],
        calf_r: [-0.2, 0.5, 0],
        foot_r: [-0.2, 0, 0],
        ball_r: [-0.2, 0, 0.25],
        index_01_l: [1.3, 1.9, 0],
        index_02_l: [1.4, 1.9, 0],
        index_03_l: [1.5, 1.9, 0],
        index_01_r: [-1.3, 1.9, 0],
        index_02_r: [-1.4, 1.9, 0],
        index_03_r: [-1.5, 1.9, 0],
    };

    const result = buildMixamoWorldKeypoints(source, viewerWithRestPoints(rest));
    assert.ok(result);
    assert.ok(result.worldKps.left_big_toe);
    assert.ok(result.worldKps.right_big_toe);
    assert.ok(result.worldKps.index_01_l);
    assert.ok(result.worldKps.index_03_r);
    assert.ok(Math.abs(result.worldKps.left_big_toe.distanceTo(result.worldKps.left_ankle) - 0.25) < 1e-8);
    assert.ok(Math.abs(result.worldKps.index_02_l.distanceTo(result.worldKps.index_01_l) - 0.1) < 1e-8);
});

test("Mixamo import uses shared alignment without weakening sampling failures", async () => {
    const source = await readFile(
        new URL("../web/jakkanna_mixamo_import.js", import.meta.url),
        "utf8",
    );
    assert.match(source, /viewer\.applyWorldKeypointImport\(mixamoKeypoints\.worldKps/);
    assert.match(source, /if \(!aligned\) \{\s*throw new Error\(`Could not align the sampled frame/);
    assert.doesNotMatch(source, /if \(!aligned\) continue/);
    assert.match(source, /const frameCount = Math\.min\(frameLimit, requestedFrames\)/);
    assert.doesNotMatch(source, /times\.push\(safeDuration\)/);
});
