# FBX Input Audit

`ComfyUI/input/animate` was audited on 2026-07-27 with
`tools/scail2_animate/audit_fbx.py`. All 18 files:

- are binary FBX 7.7 files;
- contain 67 skeleton bones and 52 animated bones;
- contain and animate all 20 bones required by Jakkanna's Mixamo retarget
  profile;
- contain at least 96 samples on their densest animation channel;
- produce finite sampled world transforms with positive rotation/scale
  determinants; and
- pass the greenfield workflow's structural compatibility contract.

The workflow emits 81 frames at 16 fps. Clips shorter than five seconds are
automatically time-fitted across those frames. Longer clips retain original
speed and use their first five seconds.

| FBX | Duration | Maximum channel samples | Automatic timing | Maximum hips excursion* | Result |
| --- | ---: | ---: | --- | ---: | --- |
| Au To Role.fbx | 4.000 s | 298 | Fit clip | 507.12 | Pass |
| Belly Dance.fbx | 19.633 s | 1255 | Original speed | 106.07 | Pass |
| Booty Hip Hop Dance.fbx | 4.883 s | 294 | Fit clip | 144.99 | Pass |
| Boxing.fbx | 1.983 s | 120 | Fit clip | 53.23 | Pass |
| dance.fbx | 15.183 s | 912 | Original speed | 34.94 | Pass |
| Dancing Maraschino Step.fbx | 3.217 s | 194 | Fit clip | 22.88 | Pass |
| Female Peek And Aim.fbx | 2.583 s | 156 | Fit clip | 23.71 | Pass |
| Hip Hop Dancing_1.fbx | 13.633 s | 819 | Original speed | 62.71 | Pass |
| Hip Hop Dancing_2.fbx | 4.467 s | 269 | Fit clip | 43.95 | Pass |
| Hip Hop Dancing_3.fbx | 6.333 s | 381 | Original speed | 50.00 | Pass |
| Hip Hop Dancing_4.fbx | 15.650 s | 988 | Original speed | 31.97 | Pass |
| Kicking.fbx | 2.583 s | 156 | Fit clip | 159.70 | Pass |
| Kip Up.fbx | 2.017 s | 164 | Fit clip | 69.15 | Pass |
| Macarena Dance.fbx | 8.217 s | 520 | Original speed | 37.26 | Pass |
| Mma Kick.fbx | 1.583 s | 96 | Fit clip | 43.48 | Pass |
| Punch To Elbow Combo.fbx | 4.983 s | 322 | Fit clip | 276.11 | Pass |
| Snake Hip Hop Dance.fbx | 15.250 s | 947 | Original speed | 36.24 | Pass |
| Surprise Uppercut.fbx | 3.483 s | 210 | Fit clip | 168.79 | Pass |

\* Raw FBX coordinate units, measured from the initial hips position over 81
uniform samples of the complete source clip.

## Framing audit

Every clip was also retargeted to the fitted `luc.jpg` body and projected
through the automatic 704 × 1280 camera. Preserved root motion produced a
useful static camera for 13 clips. Five high-travel clips could make the person
too small or make the static camera numerically degenerate:

- `Au To Role.fbx`
- `Booty Hip Hop Dance.fbx`
- `Kicking.fbx`
- `Punch To Elbow Combo.fbx`
- `Surprise Uppercut.fbx`

The greenfield workflow therefore evaluates the actual fitted subject and
sampled motion at runtime. It preserves root motion when the resulting focal
length remains above the safe framing floor and automatically retargets the
clip in-place otherwise. This decision is recorded in the preprocessing and
production manifests; it is not a user setting.

Re-run the structural audit after adding or replacing animation files:

```bash
~/Code/ComfyUI/.venv/bin/python \
  custom_nodes/ComfyUI-Jakkanna/tools/scail2_animate/audit_fbx.py \
  ~/Code/ComfyUI/input/animate
```
