# Inference Technical Guide

[Back to README](../README.md)

This guide describes the multi-step inference recipe implemented by this
release. Quick-start commands and official model downloads are in the README.

## Local model layout

```text
models/
  TaoMate-FVSR.safetensors  # complete DiT, including audio/reference projections
  Wan2.2_VAE.pth
  GPEN-BFR-512.pth         # --scale 1
  GPEN-BFR-1024.pth        # --scale 2; official ModelScope pytorch_model.pt
  RetinaFace-R50.pth
  ParseNet-latest.pth
  dwpose/
    yolox_l_8xb8-300e_coco.py
    yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth
    rtmw-x_8xb320-270e_cocktail14-384x288.py
    rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth
  hubert/
    config.json
    preprocessor_config.json
    pytorch_model.bin      # or the upstream model.safetensors
```

`--model-dir models` uses this layout without downloading. Individual paths
can be selected with `--checkpoint`, `--vae`, `--hubert`, `--gpen-weights`,
`--gpen-detector`, and `--gpen-parser`. Checkpoint loading is strict.
Only the GPEN checkpoint for the selected scale is required. `--gpen-weights`
must match that scale's architecture; custom model-source JSON files should
provide `gpen_weights` for 512 and `gpen_weights_1024` for 1024.
With `--reference`, GPEN assets and the GPEN source checkout are not needed.
DWPose is still needed unless `--cropped` is used. The paste mask is included
in the package at `livevsr/assets/paste_face_mask.png`; it does not belong in
the downloaded-model layout. `--blend-mask` selects a custom mask if needed.

Without a local override, model addresses come from
[`model_sources.json`](../livevsr/model_sources.json). Hugging Face assets use
the Hub cache; official direct-download assets use `~/.cache/livevsr/upstream/`.
`--cache-dir` overrides the download cache root (direct downloads use its
`upstream/` subdirectory). `--offline` disables downloads for both source types.
The default checkpoint source is `TaoLiveAIGC/TaoMate-FVSR` on Hugging Face.
While that repository is private, authenticate with `hf auth login` using an
account/token with read access. Existing checkpoints under another filename
remain usable through `--checkpoint`; `--model-dir` expects the layout above.

## Input and reference preparation

The default CLI accepts the original constant-frame-rate scene video and
returns a composited scene at its original dimensions and 25 fps. It decodes
native frames, extracts DWPose landmarks, and applies shot-local crop filtering
before mapping both scene frames and boxes to 25 fps with the same floor index.
An ordered 25-fps PNG directory works with `--audio`.

Use `--cropped` for prepared 512×512 face videos/PNG frames: it bypasses scene
extraction and paste-back and returns a 512×512 face video.

GPEN reference preparation uses GPEN-BFR-512 for 1× and GPEN-BFR-1024 for 2×,
with one official face alignment, one generator pass by default, and one inverse
mapping with the native soft-mask blend. Alignment, generation and the resulting
reference all use the selected 512 or 1024 resolution. The
surrounding crop is preserved; there is no background SR or second alignment.
`--gpen-passes 2` applies two generator passes in the same aligned coordinates.

For an externally edited first crop:

```bash
python infer.py --input inputs/face.mp4 --cropped --audio inputs/target.wav \
  --edited-reference inputs/edited_first_crop.png --gpen-repo third_party/GPEN \
  --model-dir models --output outputs/dubbed.mp4
```

`--edited-reference` applies GPEN preparation; `--reference` skips it. Edit the
first frame of the current model crop while preserving identity, pose, mouth
shape, and framing. Regenerate the reference if the crop geometry changes.
No GPT-Image or other image-editing API is called by the runtime.

At 2×, input RGB frames are bicubic-upsampled before VAE encoding. The DiT and
decoder run at the target resolution. The first model crop is resized to 1024
with Lanczos, then aligned, enhanced once by GPEN-BFR-1024, and inversely mapped
onto that 1024 crop. This 1024 reference is encoded directly, without reducing
it to 512 or substituting an upsampled GPEN-BFR-512 result. Generated
1024×1024 frames are reduced to 512×512 with `cv2.INTER_AREA` before any color
matching or mask blend. Full-scene output keeps the original scene dimensions;
cropped output is also 512×512. `--scale` controls model resolution, not final
scene scaling.

## Sampling and guidance

Defaults are 10 Euler steps, noise 0.8, audio CFG 4.0, reference CFG 1.7,
schedule shift 5.0, and seed 42. The initial latent mixture is
`(1 - noise) × LR + noise × Gaussian`.

The default guidance uses four DiT forwards per denoising step. Audio guidance
uses zero raw HuBERT features for its negative branch. Reference guidance uses
an entirely zero negative reference canvas. Audio guidance is computed inside
each reference branch; reference guidance then combines them in FP32 before
casting back to BF16.

## Temporal windows and reference slots

Windows use 20 latent frames (77 RGB positions), a 16-frame payload overlap,
and a 56-frame stride. Overlapping latent states are replayed at each matching
noise level; the VAE decoder stays continuous across windows.

Reference slots contain:

- **Slot 0:** an independently encoded identity frame.
- **Slot 1:** the second latent of the fixed reference repeated five times.
- **Slot 5 (K2):** the previous window's last emitted uint8 frame, independently encoded.

K2 has amplitude 1 in the positive canvas and is not shared with the negative
branch. LR is VAE-encoded on a continuous global timeline. From the second
window, only LR slots 0 and 1 reuse the first window's LR latents. Motion slots,
including slot 5, retain their global-time LR. This replaced LR initializes the
noisy state. Noise stays on the global timeline and does not reuse first-window
noise. End padding is discarded; generated tail frames are not cloned.

This release implements the multi-step, bidirectional recipe. Older one-step
checkpoints and other LR-conditioning modes are not interchangeable with it.

## Default full-scene pipeline

`livevsr.scene` runs these stages from the same installed CLI:

1. Decode native frames and run YOLOX-L / RTMW-X; map landmarks to normalized
   DWPose `[N, 1, 134, 3]` order with the current RTMW score convention.
2. Detect cuts and reset crop filtering within each shot. Apply pose filtering,
   five landmark-smoothing passes, forehead extension and 1.45× square expansion.
   Select fixed-box, fixed-size moving-box or dynamic-box smoothing using the
   adaptive crop thresholds (0.16 extent tolerance / 0.10 side variation).
3. Resample frames and boxes together to 25 fps; crop and resize with PIL
   BILINEAR to 512×512. The first such crop supplies the scale-matched GPEN
   reference: 512 for 1× and 1024 for 2×.
4. Restore at 512 or 1024 according to `--scale`, then return to RGB512.
5. Match each generated color channel to the source crop. Blend with the bundled
   grayscale mask, resized to 512 with OpenCV linear interpolation. Map the
   mixed crop to the original bbox using `cv2.INTER_LINEAR` and encode with audio.

The background outside the bbox is unchanged before video encoding. Cuts reset
crop filters only, not model history or the streaming decoder. The original
paste mask is bundled and selected by default; no custom feathering or private
SDK is needed. Use `--pose-python` for a separate OpenMMLab environment;
`--pose-device` controls its CPU/GPU device. Pose extraction runs before the
restoration model is loaded.

### Python geometry/compositing helpers

`livevsr.crop` and `livevsr.composite` are included in the standard installation.
These helpers remain available for callers with externally extracted tracks.
The default CLI supplies its tracks automatically and loads the bundled mask.

```python
from PIL import Image
from livevsr.crop import detect_cuts, shot_local_boxes, crop_frame
from livevsr.composite import paste_head

# frames_rgb: decoded RGB frames; raw_dwpose: same native frame order.
# Tracks: normalized x/y/score, shape [N, 1, 134, 3], in DWPose order.
# Raw COCO-WholeBody 133-point output is not interchangeable.
cuts = detect_cuts(frames_rgb)
boxes = shot_local_boxes(raw_dwpose, width, height, cuts, adaptive=True)
crops = [crop_frame(rgb, box) for rgb, box in zip(frames_rgb, boxes)]
Image.fromarray(crops[0]).save('first_crop.png')
```

The crop uses 1.45× head expansion, square framing, and PIL BILINEAR
512 resizing. Pose filtering, landmark smoothing, and bbox smoothing run within
each shot, with no face rotation/template alignment. The cut heuristic is
grayscale mean difference >20 at 240×135; review cuts around flashes and fast
motion. Model history, K2, overlap, and decoder cache do not reset at a cut.

When converting source fps to 25, calculate native-frame boxes first, then
resample frames, boxes, and crops using the same index mapping. Pass the ordered
crops and matching audio to the CLI.

For scene output, call `paste_head(scene_rgb, source_crop, generated_crop,
box, alpha)` for each matching frame. It histogram-matches each generated color
channel to the source crop, applies the supplied soft mask, then maps the head
to its bbox. No ground-truth background is used. The default CLI uses the bundled
original mask; a different mask changes the blend. At 2×, reduce the generated head to 512
before blending with the 512 source crop and mask. The scene keeps its original
dimensions. Preserve raw head outputs separately and mux the target audio.

## Memory and local data handling

- `--cpu-offload` moves inactive models to CPU. 2× increases spatial attention cost.
- 2× uses tiled VAE encoding and height-striped decoder convolutions with kernel
  overlap; temporal decoder caches are preserved.
- Temporary decoded frames and predictions are stored beside the output and
  removed on completion or a handled error. Forced termination can leave them
  for manual removal.
- Inputs are not uploaded. Network requests are only for model downloads.
- Use normal Hugging Face login/environment configuration for private models;
  keep tokens out of the repository and model URLs.
- Existing outputs require `--overwrite`. Encoding and full decoding are checked
  before replacement. Inputs cannot be overwritten.

## Tests

```bash
python -m unittest discover -s tests -v
python infer.py --help
```

See [VALIDATION.md](../VALIDATION.md) for previously measured runtime checks.
