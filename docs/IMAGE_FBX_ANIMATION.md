# Registered Image + FBX Animation

Jakkanna can prepare a single-human reference image and a binary
Mixamo-compatible FBX for SCAIL-2 animation mode. The tool fits the Momentum
Human Rig used by SAM 3D Body, preserves the inferred shape, camera, root
position, and hand articulation, retargets the supported FBX body joints, and
renders an exact 81-frame driver and black/blue mask sequence.

The original photograph is not wired directly to SCAIL-2. It is reprojected
through the fitted camera so the reference and driver share one canvas and
frame-zero geometry.

## Supported contract

- One clearly detectable human subject.
- Binary FBX with the standard 20-bone Mixamo body mapping.
- A clip spanning the requested 81-frame, 16 fps window.
- 704 × 1280 or 1280 × 704 output.
- Explicit `preserve` or `in_place` root-motion policy.

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
  --root-motion preserve
```

The run identity covers the image and FBX hashes, camera/render settings, root
motion, and Jakkanna commit. Outputs are isolated under:

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
commit. A different photograph cannot silently reuse an earlier fit.

Review all four images in `review/` before running SCAIL-2. A good frame-zero
overlay means SCAIL-2 can concentrate on appearance transfer instead of
repairing geometry.

`full_body` framing keeps the complete clip inside the canvas. Use
`--framing reference_crop` when retaining the photograph's crop matters more
than showing motion outside it. The default root-motion policy is `preserve`;
select `in_place` only for intentionally stationary motion.

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
- frame, FPS, canvas, framing, ease, and root-motion settings;
- camera and registration measurements;
- every generated file hash;
- canonical RGB16 hashes for the driver, mask batch, and registered reference;
- the exact two-colour driving-mask palette; and
- the Jakkanna commit.

The ComfyUI production manifest embeds this report and rejects source or mask
tensors that no longer match it.
