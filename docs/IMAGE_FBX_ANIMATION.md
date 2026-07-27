# Registered Image + FBX Animation

Jakkanna can prepare a single-human reference image and a binary
Mixamo-compatible FBX for SCAIL-2 animation mode. The tool fits the Momentum
Human Rig used by SAM 3D Body, preserves the inferred shape, camera, root
position, and hand articulation, retargets the supported FBX body joints, and
renders an exact 81-frame driver and black/blue mask sequence.

The original photograph is not wired directly to SCAIL-2. It is reprojected
through the fitted camera so the reference and driver share one canvas and
frame-zero geometry.

## Greenfield workflow

The bundled
[`Jakkanna_Image_FBX_Animate_Greenfield.json`](../workflows/Jakkanna_Image_FBX_Animate_Greenfield.json)
is the normal end-user entry point. Only three expanded controls are intended
to be changed:

1. upload or select one high-resolution single-subject image;
2. select an FBX from `ComfyUI/input/animate`; and
3. describe the finished animation.

All preparation and generation nodes are collapsed and preconfigured. Queueing
the workflow fits the subject, prepares or reuses a content-addressed driver,
validates registration and animation continuity, and generates the review MP4,
lossless master, reference, and production manifest.

Describe any important clothing, footwear, or anatomy that is hidden or outside
the photograph. Those details cannot be recovered from pixels that are not
present.

The sibling
[`Jakkanna_Image_FBX_Animate_Camera_Presets.json`](../workflows/Jakkanna_Image_FBX_Animate_Camera_Presets.json)
adds one intended control: a production-safe camera preset. The choices are:

- **Auto Director** and **Classic Full Shot** for complete-body framing;
- **American / Cowboy Shot**, **Medium Shot**, **Medium Close-Up**, and
  **Classic Close-Up** for progressively tighter optical framing; and
- **Dutch Left (10°)** and **Dutch Right (10°)** for a fixed optical-axis roll.

All presets use one fixed camera for the full clip and apply the same transform
to the registered photograph, rendered driver, and masks. Cropped presets may
intentionally continue below the bottom edge; top and side canvas-clearance
checks remain mandatory. An explicit selection is recorded in cache identity
and provenance and is never silently replaced.

High angle, low angle, profile, bird's-eye, worm's-eye,
over-the-shoulder, and point-of-view presets are intentionally omitted. A
single photograph does not contain the unseen appearance needed to synthesize
those views reliably. Supporting them correctly requires a separate
novel-view reconstruction or synthesis stage.

## Installation requirements

- ComfyUI 0.27.0 or newer.
- Python 3.10 or newer.
- An NVIDIA CUDA GPU. The complete workflow was validated on a 32 GiB RTX 5090
  and used about 27 GiB of VRAM. Lower-VRAM configurations have not been
  validated.
- ComfyUI-VideoHelperSuite 1.7.9 or newer for the review MP4 and lossless FFV1
  master.
- Approximately 30 GB for the seven SCAIL-2 generation files below, plus about
  3 GB for SAM 3D Body and BiRefNet.

Place the generation files at these exact paths:

| ComfyUI path | Source |
| --- | --- |
| `models/diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors` | [Comfy-Org/SCAIL-2](https://huggingface.co/Comfy-Org/SCAIL-2/resolve/main/diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors) |
| `models/loras/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors` | [Comfy-Org/SCAIL-2](https://huggingface.co/Comfy-Org/SCAIL-2/resolve/main/loras/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors) |
| `models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` | [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors) |
| `models/checkpoints/sam3.1_multiplex_fp16.safetensors` | [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors) |
| `models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` | [Comfy-Org/Wan 2.1](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) |
| `models/clip_vision/clip_vision_h.safetensors` | [Comfy-Org/Wan 2.1](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors) |
| `models/vae/wan_2.1_vae.safetensors` | [Comfy-Org/Wan 2.1](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors) |

The first preparation run downloads two auxiliary snapshots when they are not
already installed:

- `jetjodh/sam-3d-body-dinov3` to `models/sam3dbody`;
- `ZhengPeng7/BiRefNet_lite` to `models/birefnet/BiRefNet_lite`.

This first-run network access is initiated only when the user queues the
workflow. Subsequent runs use the local snapshots.

Create `ComfyUI/input/animate`, copy compatible `.fbx` files into it, and
refresh ComfyUI before selecting an animation. The reference image is selected
with the ordinary `Load Image` control and does not require a fixed filename.

## Supported contract

- One clearly detectable human subject.
- Binary FBX with the standard 20-bone Mixamo body mapping.
- At least 81 animation samples. Clips shorter than five seconds are
  automatically time-fitted; longer clips retain original speed and use their
  first five seconds.
- 704 × 1280 or 1280 × 704 output.
- Automatic root-motion policy. Motion is preserved unless the required static
  camera would make the person too small, in which case it is retargeted
  in-place and recorded as such in the manifest.

Other skeleton profiles, ASCII FBX, multiple people, non-human subjects, and
severely occluded people are rejected or outside the supported contract.

## Prepare a run

Use the ComfyUI virtual environment:

```bash
cd ~/Code/ComfyUI/custom_nodes/ComfyUI-Jakkanna

~/Code/ComfyUI/.venv/bin/python tools/scail2_animate/run_all.py \
  --image ~/Code/ComfyUI/input/ref_pic/luc.jpg \
  --fbx ~/Code/ComfyUI/input/animate/dance.fbx \
  --name luc-dance \
  --root-motion preserve \
  --camera-preset "American / Cowboy Shot"
```

The run identity covers the image and FBX hashes, selected camera preset,
camera/render settings, root motion, Jakkanna package version, and the Git
commit when installed from a source checkout. Outputs are isolated under:

```text
ComfyUI/input/jakkanna_animate/<name>-<identity>/
  fit/pose.json
  review/overlay.png
  review/detail_crops.png
  review/driver_frame0_over_reference.jpg
  review/driver_contact_sheet.jpg
  driver/frame_0000.png …
  driving_mask/frame_0000.png …
  registered_reference.png
  preprocess_manifest.json
```

Fits are cached separately by reference-image hash, fit threshold, and Jakkanna
release identity. Registry upgrades and different photographs cannot silently
reuse an earlier fit.

Review all four images in `review/` before running SCAIL-2. A good frame-zero
overlay means SCAIL-2 can concentrate on appearance transfer instead of
repairing geometry.

`full_body` framing keeps the complete clip inside the canvas. Use
`--framing reference_crop` when retaining the photograph's crop matters more
than showing motion outside it. The default root-motion policy is `preserve`;
select `in_place` only for intentionally stationary motion. Explicit camera
presets require `full_body` because their crop and clearance contract is
defined by the fitted body geometry.

## Generate the workflow

Start ComfyUI through the repository wrapper:

```bash
cd ~/Code/ComfyUI
./run.sh
```

Then generate both UI and API workflows against the live node contracts:

```bash
~/Code/ComfyUI/.venv/bin/python \
  custom_nodes/ComfyUI-Jakkanna/tools/scail2_animate/make_workflow.py \
  --run-dir ~/Code/ComfyUI/input/jakkanna_animate/<name>-<identity>
```

The generated workflow uses current Jakkanna validation nodes, verifies the
prepared mask palette and continuity, embeds the preprocessing report, hashes
the raw source/reference tensors, and writes review, FFV1 master, reference,
and production-manifest outputs.

Submit the same graph directly with:

```bash
~/Code/ComfyUI/.venv/bin/python \
  custom_nodes/ComfyUI-Jakkanna/tools/scail2_animate/run_workflow.py \
  --run-dir ~/Code/ComfyUI/input/jakkanna_animate/<name>-<identity> \
  --prompt "Describe the finished video"
```

An empty prompt is permitted, but then SCAIL-2 chooses the scene and unseen
appearance. Cropped photographs necessarily leave missing anatomy or clothing
for the model to synthesize.

## Provenance

`preprocess_manifest.json` contains:

- input image and FBX SHA-256 hashes;
- frame, FPS, canvas, framing, camera-preset, ease, and root-motion settings;
- camera and registration measurements;
- every generated file hash;
- canonical RGB16 hashes for the driver, mask batch, and registered reference;
- the exact two-colour driving-mask palette; and
- the Jakkanna commit.

The ComfyUI production manifest embeds this report and rejects source or mask
tensors that no longer match it.
