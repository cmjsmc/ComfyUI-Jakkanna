#!/usr/bin/env python3
"""Generate the four-input greenfield image-plus-FBX camera-preset workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _env import COMFY_ROOT, bootstrap

bootstrap()

from jakkanna_pack.camera_presets import AUTO_CAMERA_PRESET, CAMERA_PRESETS
import make_greenfield_workflow
import workflow_builder


NOTE = """# Jakkanna Greenfield Image + FBX Camera Presets

Only four inputs are intended to be changed:

1. **Reference image** — upload one high-resolution image containing one person.
2. **FBX animation** — select a compatible file from `ComfyUI/input/animate`.
3. **Camera preset** — choose a production-safe classic shot or Dutch roll.
4. **Customization prompt** — describe the finished video, especially the scene
   and any clothing or footwear that is hidden or outside the reference frame.

Requires ComfyUI 0.27.0 or newer, an NVIDIA CUDA GPU, the model files listed in
`docs/IMAGE_FBX_ANIMATION.md`, and ComfyUI-VideoHelperSuite 1.7.9 or newer.

Queue the workflow after selecting those inputs. The collapsed nodes are the
automatic preparation, validation, SCAIL-2 generation, and provenance pipeline;
they do not need adjustment.

Full, American, medium, and close shots preserve the original view axis and
change only optical framing. Dutch presets add a fixed 10-degree optical-axis
roll. Cropped shots may intentionally continue below the bottom of the canvas,
and peripheral limbs may continue through a side edge as they would in a
conventional close shot; top/head clearance remains active. The same fixed
camera transform is applied to the reference, driver, and mask for all 81
frames.

The workflow intentionally omits high angle, low angle, profile, bird's-eye,
worm's-eye, over-the-shoulder, and point-of-view presets. Those require
novel-view reconstruction or synthesis that a single reference photograph
cannot provide reliably."""


TITLES = {
    **make_greenfield_workflow.TITLES,
    "F1": "2–3 · FBX animation and camera preset",
    "P1": "4 · Customization prompt",
}

COLUMNS = make_greenfield_workflow.COLUMNS
VISIBLE = make_greenfield_workflow.VISIBLE


def camera_preset_workflow(
    image: str = "ref_pic/luc.jpg",
    fbx_file: str = "dance.fbx",
    camera_preset: str = AUTO_CAMERA_PRESET,
    prompt: str = make_greenfield_workflow.DEFAULT_PROMPT,
    seed: int = 260724,
) -> dict:
    if camera_preset not in CAMERA_PRESETS:
        raise ValueError(
            f"Unsupported camera preset {camera_preset!r}; "
            f"choose one of: {', '.join(CAMERA_PRESETS)}"
        )
    return make_greenfield_workflow.greenfield_workflow(
        image,
        fbx_file,
        prompt,
        seed,
        camera_preset,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--image", default="ref_pic/luc.jpg")
    parser.add_argument("--fbx", default="dance.fbx")
    parser.add_argument("--camera-preset", choices=CAMERA_PRESETS, default=AUTO_CAMERA_PRESET)
    parser.add_argument("--prompt", default=make_greenfield_workflow.DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=260724)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--api-output", type=Path)
    args = parser.parse_args()

    graph = camera_preset_workflow(
        args.image,
        args.fbx,
        args.camera_preset,
        args.prompt,
        args.seed,
    )
    ui = workflow_builder.build(
        args.server,
        graph,
        NOTE,
        TITLES,
        COLUMNS,
        "jakkanna-greenfield-image-fbx-camera-presets",
        collapsed=set(graph) - VISIBLE,
    )
    api = workflow_builder.to_api(args.server, graph)

    workflow_dir = COMFY_ROOT / "user" / "default" / "workflows"
    output = args.output or workflow_dir / "Jakkanna_Image_FBX_Animate_Camera_Presets.json"
    api_output = args.api_output or output.with_suffix(".api.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ui, indent=2) + "\n")
    api_output.write_text(json.dumps({"workflow": ui, "prompt": api}, indent=2) + "\n")
    print(f"[workflow] wrote {output} ({len(ui['nodes'])} nodes, {len(ui['links'])} links)")
    print(f"[workflow] wrote {api_output} ({len(api)} API nodes)")


if __name__ == "__main__":
    main()
