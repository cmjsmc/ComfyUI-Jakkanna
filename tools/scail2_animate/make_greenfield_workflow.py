#!/usr/bin/env python3
"""Generate the three-input greenfield image-plus-FBX SCAIL-2 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _env import COMFY_ROOT
import make_workflow
import workflow_builder


DEFAULT_PROMPT = (
    "Full-body animation of the person, preserving their face, hairstyle, "
    "clothing, colours, and identity."
)

NOTE = """# Jakkanna Greenfield Image + FBX Animation

Only three inputs are intended to be changed:

1. **Reference image** — upload one high-resolution image containing one person.
2. **FBX animation** — select a compatible file from `ComfyUI/input/animate`.
3. **Customization prompt** — describe the finished video, especially the scene
   and any clothing or footwear that is hidden or outside the reference frame.

Queue the workflow after selecting those inputs. The collapsed nodes are the
automatic preparation, validation, SCAIL-2 generation, and provenance pipeline;
they do not need adjustment.

Short FBX clips are automatically stretched to the 81-frame output. Clips at
least five seconds long retain original speed and use their first five seconds.
Root motion is preserved when it can be framed at a useful subject scale.
High-travel clips are automatically retargeted in-place when a static preserved
camera would make the subject too small."""


def greenfield_workflow(
    image: str = "ref_pic/luc.jpg",
    fbx_file: str = "dance.fbx",
    prompt: str = DEFAULT_PROMPT,
    seed: int = 260724,
) -> dict:
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

    return {
        "I1": ("LoadImage", {"image": image}),
        "F1": ("JakkannaImageFBXPrepare", {
            "image": ["I1", 0],
            "fbx_file": fbx_file,
        }),
        **graph,
    }


TITLES = {
    **make_workflow.TITLES,
    "I1": "1 · Reference image",
    "F1": "2 · FBX animation",
    "P1": "3 · Customization prompt",
    "O2": "Review MP4",
    "O3": "Lossless master",
}

COLUMNS = {
    **make_workflow.COLUMNS,
    "I1": 0,
    "F1": 0,
    "P1": 0,
    "N1": 1,
}

VISIBLE = {"I1", "F1", "P1"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--image", default="ref_pic/luc.jpg")
    parser.add_argument("--fbx", default="dance.fbx")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=260724)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--api-output", type=Path)
    args = parser.parse_args()

    graph = greenfield_workflow(args.image, args.fbx, args.prompt, args.seed)
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
