#!/usr/bin/env python3
"""Generate a hardened image-plus-FBX SCAIL-2 workflow for one prepared run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _env import COMFY_ROOT, INPUT_ROOT
import workflow_builder


NEGATIVE = ""

NOTE = """# Jakkanna Registered Image + FBX + SCAIL-2 Animate

This workflow animates one registered human reference with a compatible binary
Mixamo FBX. It loads the run-scoped artifacts produced by
`tools/scail2_animate/run_all.py`; it does not load the original photograph
directly.

Before queueing, review `review/overlay.png`, `review/detail_crops.png`,
`review/driver_frame0_over_reference.jpg`, and
`review/driver_contact_sheet.jpg` in the prepared run directory.

The graph validates the raw 81-frame 16 fps driver, exact black/blue prepared
mask palette, adjacent-frame continuity, canvas clearance, single reference
subject, preprocessing hashes, sampler settings, and source/reference tensor
hashes. SCAIL-2 runs animation mode (`replacement_mode = false`).

The fast profile is FP8-scaled SCAIL-2 + DPO LoRA 1.0 + LightX2V 0.8,
Euler/simple, six steps, CFG 1.0. Empty positive text is valid but leaves scene
generation unconstrained. Describe the finished video when a particular scene
or unseen clothing/anatomy must be steered. Run ComfyUI with `./run.sh`."""


def workflow(
    run_relative: str,
    preprocess_report: str,
    output_prefix: str,
    prompt: str,
    subject: str,
    seed: int,
    width: int,
    height: int,
) -> dict:
    reference = f"{run_relative}/registered_reference.png"
    driver = f"{run_relative}/driver"
    driving_mask = f"{run_relative}/driving_mask"
    return {
        "R1": ("LoadImage", {"image": reference}),
        "D1": ("VHS_LoadImages", {"directory": driver}),
        "D2": ("VHS_LoadImages", {"directory": driving_mask}),
        "P0": ("PrimitiveStringMultiline", {"value": preprocess_report}),
        "P1": ("PrimitiveStringMultiline", {"value": prompt}),
        "N1": ("PrimitiveStringMultiline", {"value": NEGATIVE}),

        "S1": ("CheckpointLoaderSimple", {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}),
        "S2": ("CLIPTextEncode", {"clip": ["S1", 1], "text": subject}),
        "S3": ("SAM3_Detect", {
            "model": ["S1", 0],
            "image": ["R1", 0],
            "conditioning": ["S2", 0],
            "threshold": 0.5,
            "refine_iterations": 2,
            "individual_masks": True,
        }),
        "S4": ("JakkannaSCAIL2SingleSubjectMask", {
            "masks": ["S3", 0],
            "minimum_area_percent": 1.0,
            "maximum_area_percent": 95.0,
        }),

        "M1": ("MaskToImage", {"mask": ["S4", 0]}),
        "M2": ("ImageInvert", {"image": ["M1", 0]}),
        "M3": ("EmptyImage", {"width": 64, "height": 64, "batch_size": 1, "color": 255}),
        "M4": ("ImageBlend", {
            "image1": ["M2", 0],
            "image2": ["M3", 0],
            "blend_factor": 1.0,
            "blend_mode": "screen",
        }),
        "B1": ("EmptyImage", {"width": width, "height": height, "batch_size": 1, "color": 0}),
        "R2": ("ImageCompositeMasked", {
            "destination": ["B1", 0],
            "source": ["R1", 0],
            "x": 0,
            "y": 0,
            "resize_source": False,
            "mask": ["S4", 0],
        }),
        "R3": ("JakkannaSCAIL2ReferencePNG16Save", {
            "images": ["R2", 0],
            "filename_prefix": f"{output_prefix}/reference/SCAIL2_reference",
        }),

        "V1": ("CreateVideo", {"images": ["D1", 0], "fps": 16.0, "bit_depth": 8}),
        "V2": ("JakkannaSCAIL2ProductionInputValidate", {
            "source_video": ["V1", 0],
            "reference_image": ["R2", 0],
        }),
        "V3": ("JakkannaSCAIL2PreparedTrackValidate", {
            "source_images": ["D1", 0],
            "driving_mask_images": ["D2", 0],
            "registered_reference": ["R1", 0],
            "reference_mask": ["S4", 0],
            "expected_frames": ["V2", 2],
            "preprocess_report": ["P0", 0],
            "maximum_area_step_ratio": 2.0,
            "maximum_centroid_jump": 0.15,
            "minimum_adjacent_iou": 0.1,
            "minimum_border_margin": 0.005,
            "minimum_registration_iou": 0.35,
        }),

        "W1": ("UNETLoader", {
            "unet_name": "wan2.1_14B_SCAIL_2_fp8_scaled.safetensors",
            "weight_dtype": "default",
        }),
        "W2": ("LoraLoaderModelOnly", {
            "model": ["W1", 0],
            "lora_name": "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors",
            "strength_model": 1.0,
        }),
        "W3": ("LoraLoaderModelOnly", {
            "model": ["W2", 0],
            "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            "strength_model": 0.8,
        }),
        "W4": ("CLIPLoader", {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "wan",
            "device": "default",
        }),
        "W5": ("CLIPTextEncode", {"clip": ["W4", 0], "text": ["P1", 0]}),
        "W6": ("CLIPTextEncode", {"clip": ["W4", 0], "text": ["N1", 0]}),
        "W7": ("VAELoader", {"vae_name": "wan_2.1_vae.safetensors"}),
        "W8": ("CLIPVisionLoader", {"clip_name": "clip_vision_h.safetensors"}),
        "W9": ("CLIPVisionEncode", {
            "clip_vision": ["W8", 0],
            "image": ["R2", 0],
            "crop": "none",
        }),

        "C1": ("WanSCAILToVideo", {
            "positive": ["W5", 0],
            "negative": ["W6", 0],
            "vae": ["W7", 0],
            "width": ["V2", 0],
            "height": ["V2", 1],
            "length": ["V3", 2],
            "batch_size": 1,
            "pose_video": ["D1", 0],
            "pose_video_mask": ["D2", 0],
            "replacement_mode": False,
            "pose_strength": 1.0,
            "pose_start": 0.0,
            "pose_end": 1.0,
            "reference_image": ["R2", 0],
            "reference_image_mask": ["M4", 0],
            "clip_vision_output": ["W9", 0],
            "video_frame_offset": 0,
            "previous_frame_count": 5,
        }),
        "C2": ("CFGGuider", {
            "model": ["W3", 0],
            "positive": ["C1", 0],
            "negative": ["C1", 1],
            "cfg": 1.0,
        }),
        "C3": ("KSamplerSelect", {"sampler_name": "euler"}),
        "C4": ("BasicScheduler", {
            "model": ["W3", 0],
            "scheduler": "simple",
            "steps": 6,
            "denoise": 1.0,
        }),
        "C5": ("PrimitiveInt", {"value": seed}),
        "C6": ("JakkannaSCAIL2UpstreamNoise", {
            "latent": ["C1", 2],
            "noise_seed": ["C5", 0],
        }),
        "C7": ("SamplerCustomAdvanced", {
            "noise": ["C6", 0],
            "guider": ["C2", 0],
            "sampler": ["C3", 0],
            "sigmas": ["C4", 0],
            "latent_image": ["C1", 2],
        }),
        "C8": ("VAEDecode", {"samples": ["C7", 1], "vae": ["W7", 0]}),

        "O1": ("CreateVideo", {
            "images": ["C8", 0],
            "fps": ["V2", 3],
            "bit_depth": 10,
        }),
        "O2": ("SaveVideo", {
            "video": ["O1", 0],
            "filename_prefix": f"{output_prefix}/review/SCAIL2_Animate",
            "format": "mp4",
            "codec": "h264",
        }),
        "O3": ("VHS_VideoCombine", {
            "images": ["C8", 0],
            "frame_rate": ["V2", 3],
            "loop_count": 0,
            "filename_prefix": f"{output_prefix}/master/SCAIL2_Animate",
            "format": "video/ffv1-rgb-mkv.json",
            "pingpong": False,
            "save_output": True,
            "level": "3",
            "coder": "1",
            "context": "1",
            "gop_size": 1,
            "slices": "16",
            "slicecrc": "1",
            "pix_fmt": "rgb48le",
            "save_metadata": True,
            "trim_to_audio": False,
        }),
        "O4": ("JakkannaSCAIL2ProductionManifest", {
            "output_images": ["C8", 0],
            "pose_video_mask": ["D2", 0],
            "reference_image_mask": ["M4", 0],
            "input_report": ["V2", 4],
            "reference_report": ["R3", 0],
            "source_initial_mask_report": ["V3", 0],
            "reference_mask_report": ["S4", 1],
            "track_report": ["V3", 1],
            "noise_report": ["C6", 1],
            "positive_prompt": ["P1", 0],
            "negative_prompt": ["N1", 0],
            "seed": ["C5", 0],
            "diffusion_model": "wan2.1_14B_SCAIL_2_fp8_scaled.safetensors",
            "dpo_lora": "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors",
            "sam_checkpoint": "sam3.1_multiplex_fp16.safetensors",
            "text_encoder": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "clip_vision": "clip_vision_h.safetensors",
            "vae": "wan_2.1_vae.safetensors",
            "attention_backend": "PyTorch SDPA",
            "hash_model_files": True,
            "filename_prefix": f"{output_prefix}/manifests/SCAIL2_Animate",
            "acceleration_lora": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            "acceleration_lora_strength": 0.8,
            "sampler_name": "Euler",
            "scheduler_name": "simple (shift-8 flow schedule)",
            "steps": 6,
            "flow_shift": 8.0,
            "cfg": 1.0,
            "effective_flow_shift": 0.0,
            "mode": "animation",
            "sigmas": ["C4", 0],
            "source_images": ["D1", 0],
            "reference_image": ["R2", 0],
            "preprocess_report": ["P0", 0],
        }),
        "Q1": ("PreviewImage", {"images": ["R2", 0]}),
        "Q2": ("PreviewImage", {"images": ["M4", 0]}),
        "Q3": ("PreviewAny", {"source": ["V3", 1]}),
    }


TITLES = {
    "R1": "Registered photograph on the driver canvas",
    "D1": "Registered 81-frame FBX driver",
    "D2": "Pixel-exact black/blue driving masks",
    "P0": "Content-addressed preprocessing report",
    "P1": "Finished-video prompt — optional",
    "S2": "Reference subject concept",
    "S4": "Single reference-subject gate",
    "R2": "Subject-on-black SCAIL reference",
    "R3": "Lossless reference and provenance",
    "V2": "Raw 81-frame / 16 fps input gate",
    "V3": "Prepared track, palette, and provenance gate",
    "C1": "SCAIL-2 animation mode",
    "C4": "Six-step native shift-8 schedule",
    "O2": "Review MP4",
    "O3": "Lossless FFV1 master",
    "O4": "Production manifest",
}

COLUMNS = {
    **{name: 0 for name in ("R1", "D1", "D2", "P0", "P1", "N1")},
    **{name: 1 for name in ("S1", "S2", "S3", "S4", "M1", "M2", "M3", "M4", "B1", "R2", "R3")},
    **{name: 2 for name in ("V1", "V2", "V3", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9")},
    **{name: 3 for name in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")},
    **{name: 4 for name in ("O1", "O2", "O3", "O4", "Q1", "Q2", "Q3")},
}


def load_run(run_dir: Path) -> tuple[Path, dict]:
    run_dir = run_dir.resolve()
    if INPUT_ROOT not in run_dir.parents:
        raise ValueError(f"run-dir must be inside {INPUT_ROOT}")
    report_path = run_dir / "preprocess_manifest.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"preprocessing report does not exist: {report_path}")
    report = json.loads(report_path.read_text())
    if report.get("schema") != "jakkanna.image_fbx_animation.v1":
        raise ValueError("preprocessing report has an unsupported schema")
    if report.get("run_id") != run_dir.name:
        raise ValueError("preprocessing report run_id does not match its directory")
    for relative in ("registered_reference.png", "driver", "driving_mask"):
        if not (run_dir / relative).exists():
            raise FileNotFoundError(f"prepared artifact is missing: {run_dir / relative}")
    return run_dir, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--subject", default="person")
    parser.add_argument("--seed", type=int, default=260724)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--api-output", type=Path)
    args = parser.parse_args()

    run_dir, report = load_run(args.run_dir)
    run_relative = run_dir.relative_to(INPUT_ROOT).as_posix()
    run_id = report["run_id"]
    graph = workflow(
        run_relative,
        json.dumps(report, sort_keys=True, separators=(",", ":")),
        f"jakkanna_animate/{run_id}",
        args.prompt,
        args.subject,
        args.seed,
        report["settings"]["width"],
        report["settings"]["height"],
    )
    ui = workflow_builder.build(
        args.server,
        graph,
        NOTE,
        TITLES,
        COLUMNS,
        f"jakkanna-image-fbx-scail2-animate-{run_id}",
    )
    prompt = workflow_builder.to_api(args.server, graph)

    workflow_dir = COMFY_ROOT / "user/default/workflows"
    output = args.output or workflow_dir / f"Jakkanna_Image_FBX_SCAIL2_Animate_{run_id}_fast.json"
    api_output = args.api_output or output.with_suffix(".api.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ui, indent=2) + "\n")
    api_output.write_text(json.dumps({"workflow": ui, "prompt": prompt}, indent=2) + "\n")
    print(f"[workflow] wrote {output} ({len(ui['nodes'])} nodes, {len(ui['links'])} links)")
    print(f"[workflow] wrote {api_output} ({len(prompt)} API nodes)")


if __name__ == "__main__":
    main()
