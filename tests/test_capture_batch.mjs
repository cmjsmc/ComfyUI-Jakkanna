import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";


const coreSource = await readFile(
    new URL("../web/jakkanna_pose_studio_core.js", import.meta.url),
    "utf8",
);
const studioSource = await readFile(
    new URL("../web/jakkanna_pose_studio.js", import.meta.url),
    "utf8",
);

const start = coreSource.indexOf("beginCaptureBatch(width, height)");
const end = coreSource.indexOf("\n    capture(width, height", start);
assert.ok(start >= 0 && end > start);
const CaptureBatchHarness = Function(`"use strict"; return class { ${coreSource.slice(start, end)} }`)();

function createHarness() {
    const sizeCalls = [];
    const ratioCalls = [];
    const renderCalls = [];
    const harness = new CaptureBatchHarness();
    harness.initialized = true;
    harness._captureBatch = null;
    harness.THREE = {
        Vector2: class {
            constructor() {
                this.x = 0;
                this.y = 0;
            }
        },
    };
    harness.scene = {};
    harness.camera = {};
    harness.renderer = {
        getSize(target) {
            target.x = 800;
            target.y = 600;
        },
        getPixelRatio() {
            return 2;
        },
        setPixelRatio(value) {
            ratioCalls.push(value);
        },
        setSize(width, height, updateStyle) {
            sizeCalls.push([width, height, updateStyle]);
        },
        render(scene, camera) {
            renderCalls.push([scene, camera]);
        },
    };
    return { harness, sizeCalls, ratioCalls, renderCalls };
}

test("capture batch resizes once and restores the renderer once", () => {
    const { harness, sizeCalls, ratioCalls, renderCalls } = createHarness();

    assert.equal(harness.beginCaptureBatch(1024, 768), true);
    assert.equal(harness.beginCaptureBatch(1024, 768), false);
    assert.deepEqual(sizeCalls, [[1024, 768, false]]);
    assert.deepEqual(ratioCalls, [1]);

    harness.endCaptureBatch();
    assert.deepEqual(sizeCalls, [
        [1024, 768, false],
        [800, 600, false],
    ]);
    assert.deepEqual(ratioCalls, [1, 2]);
    assert.deepEqual(renderCalls, [[harness.scene, harness.camera]]);
    assert.equal(harness._captureBatch, null);
});

test("full pose capture brackets the loop with one renderer batch", () => {
    const syncStart = studioSource.indexOf("syncToNode(fullCapture = false, options = {})");
    const syncEnd = studioSource.indexOf("\n    loadFromNode()", syncStart);
    const syncMethod = studioSource.slice(syncStart, syncEnd);
    const begin = syncMethod.indexOf("this.viewer.beginCaptureBatch");
    const loop = syncMethod.indexOf("for (let i = 0; i < this.poses.length; i++)");
    const endBatch = syncMethod.indexOf("this.viewer.endCaptureBatch");
    assert.ok(begin >= 0 && begin < loop);
    assert.ok(endBatch > loop);
});
