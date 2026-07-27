#!/usr/bin/env python3
"""Generate the four-input greenfield image-plus-FBX SCAIL-2 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _env import COMFY_ROOT, bootstrap
import make_workflow
import workflow_builder

bootstrap()

from jakkanna_pack.camera_presets import AUTO_CAMERA_PRESET, CAMERA_PRESETS


DEFAULT_PROMPT = (
    "Full-body animation of the person, preserving their face, hairstyle, "
    "clothing, colours, and identity."
)

NOTE = """# Jakkanna Greenfield Image + FBX Animation

Only four inputs are intended to be changed:

1. **Reference image** — upload one high-resolution image containing one person.
2. **FBX animation** — select a compatible file from `ComfyUI/input/animate`.
3. **Camera preset** — choose a production-safe classic shot or Dutch roll.
   `Auto Director` frames the whole body automatically and is the default.
4. **Customization prompt** — describe the finished video, especially the scene
   and any clothing or footwear that is hidden or outside the reference frame.

Requires ComfyUI 0.27.0 or newer, an NVIDIA CUDA GPU, the model files listed in
`docs/IMAGE_FBX_ANIMATION.md`, and ComfyUI-VideoHelperSuite 1.7.9 or newer.

Queue the workflow after selecting those inputs. The collapsed nodes are the
automatic preparation, validation, SCAIL-2 generation, and provenance pipeline;
they do not need adjustment.

SAM 3.1 isolates the person before SAM 3D Body fitting so scene objects do not
distort the automatic body bounds. The same checkpoint segments the registered
reference after fitting. No mask input or adjustment is required.

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
cannot provide reliably.

Short FBX clips are automatically stretched to the 81-frame output. Clips at
least five seconds long retain original speed and use their first five seconds.
Root motion is preserved when it can be framed at a useful subject scale.
High-travel clips are automatically retargeted in-place when a static preserved
camera would make the subject too small. Whichever timing and root-motion
policy preparation resolves is reported in the track report preview, along with
the frame-zero registration score that gates the run."""


def greenfield_workflow(
    image: str = "ref_pic/luc.jpg",
    fbx_file: str = "dance.fbx",
    prompt: str = DEFAULT_PROMPT,
    seed: int = 260724,
    camera_preset: str = AUTO_CAMERA_PRESET,
) -> dict:
    if camera_preset not in CAMERA_PRESETS:
        raise ValueError(
            f"Unsupported camera preset {camera_preset!r}; "
            f"choose one of: {', '.join(CAMERA_PRESETS)}"
        )
    graph = make_workflow.workflow(
        "unused",
        "{}",
        "jakkanna_greenfield",
        prompt,
        "person",
        seed,
        704,
        1280,
    )
    del graph["R1"]
    del graph["D1"]
    del graph["D2"]
    del graph["P0"]

    replacements = {
        "R1": ("F1", 2),
        "D1": ("F1", 0),
        "D2": ("F1", 1),
        "P0": ("F1", 3),
    }
    for _node_type, inputs in graph.values():
        for name, value in inputs.items():
            if isinstance(value, list) and len(value) == 2 and value[0] in replacements:
                source, slot = replacements[value[0]]
                inputs[name] = [source, slot]

    prepare_inputs = {
        "image": ["I1", 0],
        "subject_mask": ["A2", 0],
        "fbx_file": fbx_file,
        "camera_preset": camera_preset,
    }

    return {
        "I1": ("LoadImage", {"image": image}),
        "A1": ("SAM3_Detect", {
            "model": ["S1", 0],
            "image": ["I1", 0],
            "conditioning": ["S2", 0],
            "threshold": 0.5,
            "refine_iterations": 2,
            "individual_masks": True,
        }),
        "A2": ("JakkannaSCAIL2SingleSubjectMask", {
            "masks": ["A1", 0],
            "minimum_area_percent": 1.0,
            "maximum_area_percent": 95.0,
        }),
        "F1": ("JakkannaImageFBXPrepare", prepare_inputs),
        **graph,
    }


TITLES = {
    **make_workflow.TITLES,
    "I1": "1 · Reference image",
    "A1": "Automatic original-image person segmentation",
    "A2": "Single pre-fit subject gate",
    "F1": "2–3 · FBX animation and camera preset",
    "P1": "4 · Customization prompt",
    "O2": "Review MP4",
    "O3": "Lossless master",
}

COLUMNS = {
    **make_workflow.COLUMNS,
    "I1": 0,
    "F1": 0,
    "P1": 0,
    "A1": 1,
    "A2": 1,
    "N1": 1,
}

VISIBLE = {"I1", "F1", "P1"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--image", default="ref_pic/luc.jpg")
    parser.add_argument("--fbx", default="dance.fbx")
    parser.add_argument("--camera-preset", choices=CAMERA_PRESETS, default=AUTO_CAMERA_PRESET)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=260724)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--api-output", type=Path)
    args = parser.parse_args()

    graph = greenfield_workflow(
        args.image, args.fbx, args.prompt, args.seed, args.camera_preset
    )
    ui = workflow_builder.build(
        args.server,
        graph,
        NOTE,
        TITLES,
        COLUMNS,
        "jakkanna-greenfield-image-fbx-scail2-animate",
        collapsed=set(graph) - VISIBLE,
    )
    api = workflow_builder.to_api(args.server, graph)

    workflow_dir = COMFY_ROOT / "user" / "default" / "workflows"
    output = args.output or workflow_dir / "Jakkanna_Image_FBX_Animate_Greenfield.json"
    api_output = args.api_output or output.with_suffix(".api.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ui, indent=2) + "\n")
    api_output.write_text(json.dumps({"workflow": ui, "prompt": api}, indent=2) + "\n")
    print(f"[workflow] wrote {output} ({len(ui['nodes'])} nodes, {len(ui['links'])} links)")
    print(f"[workflow] wrote {api_output} ({len(api)} API nodes)")


if __name__ == "__main__":
    main()
