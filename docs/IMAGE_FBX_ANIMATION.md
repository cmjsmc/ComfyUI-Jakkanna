# Registered Image + FBX Animation

Jakkanna can prepare a single-human reference image and a binary
Mixamo-compatible FBX for SCAIL-2 animation mode. The tool fits the Momentum
Human Rig used by SAM 3D Body, preserves the inferred shape, camera, root
position, and hand articulation, retargets the supported FBX body joints, and
renders an exact 81-frame driver and black/blue mask sequence.

The original photograph is not wired directly to SCAIL-2. It is reprojected
through the fitted camera so the reference and driver share one canvas and
frame-zero geometry.

In the Greenfield workflows, the installed SAM 3.1 checkpoint first segments
the `person` in the original photograph. That binary mask guides SAM 3D Body's
body bounds and reconstruction, excluding scene objects such as vehicles or
furniture. After camera registration, the same checkpoint segments the
registered reference used by SCAIL-2. Both stages are automatic and require no
mask editing or additional user input.

## Greenfield workflow

The bundled
[`Jakkanna_Image_FBX_Animate_Greenfield.json`](../workflows/Jakkanna_Image_FBX_Animate_Greenfield.json)
is the normal end-user entry point. Only four expanded controls are intended
to be changed:

1. upload or select one high-resolution single-subject image;
2. select an FBX from `ComfyUI/input/animate`;
3. choose a camera preset (`Auto Director` by default); and
4. describe the finished animation.

All preparation and generation nodes are collapsed and preconfigured. Queueing
the workflow fits the subject, prepares or reuses a content-addressed driver,
validates registration and animation continuity, and generates the review MP4,
lossless master, reference, and production manifest.

Preparation runs as a separate process that loads SAM 3D Body into a CUDA
context of its own, so the first queue on a new image takes minutes. The node
reports the driver render as node progress, and cancelling the queue stops the
child rather than leaving it holding the GPU.

Describe any important clothing, footwear, or anatomy that is hidden or outside
the photograph. Those details cannot be recovered from pixels that are not
present.

## Camera presets

The camera preset is a production-safe shot choice. The options are:

- **Auto Director** and **Classic Full Shot** for complete-body framing;
- **American / Cowboy Shot**, **Medium Shot**, **Medium Close-Up**, and
  **Classic Close-Up** for progressively tighter optical framing; and
- **Dutch Left (10°)** and **Dutch Right (10°)** for a fixed optical-axis roll.

All presets use one fixed camera for the full clip and apply the same transform
to the registered photograph, rendered driver, and masks. Cropped presets may
intentionally continue below the bottom edge, and peripheral limbs may
continue through a side edge as they would in a conventional close shot.
Top/head canvas clearance remains mandatory. The selection is recorded in cache
identity and provenance, and preparation never substitutes a different preset.

Preparation does resolve two other choices on its own: `auto` timing stretches
a clip shorter than five seconds across the 81 frames, and `auto` root motion
falls back to an in-place retarget when preserving travel would frame the
subject too small. Both decisions, together with the frame-zero registration
score, are reported in the track report preview node on every run — check them
before accepting a result.

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
  3 GB for SAM 3D Body and the optional BiRefNet fallback.

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

The first Greenfield preparation run downloads SAM 3D Body when it is not
already installed:

- `jetjodh/sam-3d-body-dinov3` to `models/sam3dbody`;

This first-run network access is initiated only when the user queues the
workflow. Subsequent runs use the local snapshot. The standalone
`run_all.py` command retains a BiRefNet fallback when `--subject-mask` is not
provided; that fallback downloads `ZhengPeng7/BiRefNet_lite` to
`models/birefnet/BiRefNet_lite`.

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
references without enough visible torso and limb evidence for a body fit are
rejected or outside the supported contract. Hidden or out-of-frame anatomy is
inferred and should be described in the customization prompt.

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

### Sampler gate

`Jakkanna SCAIL-2 Validate Sampler` sits between the scheduler and the sampler.
It reconciles the sigma schedule, sampler, scheduler, CFG, generation mode and
LoRA strengths against the settings the production manifest declares, then
passes the schedule through unchanged.

The manifest performs the same reconciliation when it is written, so a graph
assembled without the gate is still checked. The difference is when: without
the gate, a mismatch — a second `CLIPVisionEncode`, a CFG that drifted from the
declared value, or a sigma schedule built at the wrong flow shift — surfaces
only after the latent has been sampled and decoded. Both nodes share one
implementation, so they cannot disagree.

### Registration gate

`Jakkanna SCAIL-2 Validate Prepared Track` refuses to run generation when the
fitted mannequin does not land on the photographed subject. `minimum_registration_iou`
(default `0.35`) is a floor on the frame-zero silhouette overlap measured over
the region the photograph actually shows. Usable fits measure roughly 0.58 to
0.88 — an occluded, seated, or partly cropped subject sits at the bottom of
that range with registration still sound. A body fit at the wrong depth, scale,
or orientation collapses far below the floor, which is where a run should stop
rather than spend a full 81-frame generation.

Because that spread is wide, the default is deliberately loose: an earlier
0.65 gate rejected working runs and was removed entirely, leaving no floor at
all. `0.35` sits below every fit observed to be usable. If
a run trips the floor, open `review/driver_frame0_over_reference.jpg` in the
prepared run directory first. Lower the threshold only once that overlay
confirms the mismatch is legitimate; set it to `0` to record the score without
gating.

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

## Retiring superseded runs

Preparation is content-addressed on the inputs, the settings, and the Jakkanna
release identity. Keying on the release means a commit that cannot change a
pixel still mints a fresh run directory, so `input/jakkanna_animate` grows by
one full 81-frame render per commit per input combination.

The runs are not interchangeable — outputs from different commits genuinely
differ — so `tools/scail2_animate/prune_runs.py` retires superseded ones rather
than deduplicating. It keeps the most recent run for every combination of
inputs and settings, and never touches a run that a workflow JSON still loads.

```bash
~/Code/ComfyUI/.venv/bin/python \
  custom_nodes/ComfyUI-Jakkanna/tools/scail2_animate/prune_runs.py
```

It is a dry run by default and prints exactly what it would keep and retire.
Add `--apply` to perform it, or `--keep 2` to retain a generation of history per
combination. Normalised inputs under `_source_cache` and `_subject_mask_cache`
that no surviving run refers to are reported and removed alongside.

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

With `hash_model_files` enabled the manifest also records a SHA-256 for every
loaded model, which is tens of gigabytes of reads. Those digests are cached by
path, size, and mtime in `ComfyUI/user/jakkanna/scail2_model_hash_cache.json`
so the cost is paid once per model file rather than once per ComfyUI restart.
Delete that file to force a re-hash.
