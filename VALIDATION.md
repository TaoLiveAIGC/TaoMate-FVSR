# Validation

These checks validate the inference release, not restoration quality or benchmark
performance. All inference checks use local model assets.

## Test isolation and README results — 2026-11-03

- Fixed the GPEN unit test's module isolation: OpenCV is imported before the
  temporary `sys.modules` patch and remains the same module after it exits.
- The standard `python -m unittest discover -s tests -v` command passed
  **33/33 tests** in both the PyTorch 2.8.0 / OpenCV 4.11.0 environment that
  previously failed and the PyTorch 2.9.0 release-validation environment.
  Neither run used an external OpenCV preload workaround; GPUs were disabled.
- README restoration, replacement-audio and ablation tables were checked
  cell-for-cell against the manuscript tables: 13, 11 and 9 data rows,
  respectively. Values and displayed precision are unchanged. The README
  retains the replacement-audio subset selection and reference protocols.
- Only tests and documentation changed. Model weights, inference behavior and
  the package version remain unchanged; no new GPU benchmark was run.

## 0.3.3 — inference-oriented audio naming, 2026-11-03

- **33/33 unit tests passed**, including the renamed audio interfaces,
  projection parameter paths and zero-audio condition.
- The wheel built and installed independently with `audio_conditioning.py`.
  Imports and the compatibility check ran outside the source checkout.
- All **886 checkpoint tensor names and shapes** match both the previous
  release and the existing model file. Strict checkpoint loading passed.
- With the real checkpoint on H20, two small-latent DiT forward checks
  (conditioned audio and zero audio) matched version 0.3.2 exactly in BF16:
  maximum absolute difference **0**, with `rtol=0` and `atol=0`.
- HuBERT window alignment at offsets 0 and 56 remained exactly equal.
  Sampling, GPEN resolution selection, crop/paste behavior and weights did
  not change.

## 0.3.2 — scale-matched GPEN references, 2026-11-03

- **31/31 unit tests passed**, including scale-specific GPEN asset selection,
  official 1024 download resolution, and 512/1024 alignment/output geometry.
- The wheel built and installed independently. The installed CLI was run
  outside the source checkout with a model directory containing GPEN-BFR-1024
  but no GPEN-BFR-512 and no external paste mask.
- The official ModelScope `pytorch_model.pt` loaded into GPEN's 1024 network.
  The runtime produced a **1024×1024 reference with one GPEN pass**, then loaded
  the full restoration checkpoint, VAE and HuBERT.
- A 73-native-frame, two-shot clip produced **77 frames across two windows** at
  25 fps, 768×1152, with 3.08 seconds of video and audio. The 2× model output
  was reduced to RGB512 before full-scene paste-back. Exact frame count,
  dimensions, duration and complete FFmpeg decoding passed on one H20.
- This new runtime check used **two denoising steps** as a short functional
  regression; the default remains ten. It is not a new quality benchmark or
  a repeat of the full-length tests below.
- The official 1024 URL also returned a PyTorch checkpoint through the same
  Python URL-opening mechanism used by the automatic downloader. Third-party
  weights remain at their original publisher URLs.

## 0.3.1 — bundled paste mask, 2026-11-03

- **29/29 unit tests passed**, including full-scene CLI validation without a
  mask in the model directory, plus explicit custom-mask selection.
- The wheel built and installed separately. Outside the source checkout, the
  installed CLI selected its bundled mask and passed component validation.
- The bundled 7,015-byte grayscale mask is byte-for-byte identical to the mask
  used for the full-video checks below; the wheel contains the same file.
- Cropping, histogram matching, compositing, sampling and model weights are
  unchanged. This update removes the separate mask-file setup requirement.

## 0.3.0 — default full-scene pipeline, 2026-11-03

The CLI now crops and pastes full videos by default. `--scale 2` runs the model
at 1024 and reduces predictions to RGB512 **before** color matching and blending;
the full-scene output is not enlarged. `--cropped` retains the face-only entry.

- **28/28 unit tests passed**, including source-frame mapping, shot-local
  adaptive crop filtering, DWPose point ordering and 1024→512-before-paste order.
- The new wheel built, installed separately, and ran its console entry point
  outside the checkout. All model and mask assets were local/offline.
- On two real landmark tracks (452 frames), adaptive crop boxes matched the
  current preprocessing implementation exactly. Separate 51-frame trajectories
  for each of its three crop modes also matched (153 boxes).
- Histogram/mask blending matched the existing implementation pixel-for-pixel
  with the same local mask. These checks do not establish equivalence for a
  different detector, mask or reference image.
- The YOLOX-L and RTMW-X original publisher checkpoint links returned HTTP200.

The full-video check uses a 584-frame, 23.36-second 1080×1920 source with audio,
the installed CLI, default 10 steps, one GPEN pass and no supplied reference.
YOLOX/RTMW preprocessing uses the separate OpenMMLab environment documented in
the README, on the same H20 GPU before restoration loading.

| Scale | Model resolution | Frames / windows | Composited output | Status |
|---|---|---|---|---|
| 1× | 512×512 | 584 / 11 | 1080×1920, 25 fps, audio | Passed: exact frame count/duration and complete FFmpeg decode |
| 2× | 1024×1024 → 512×512 | 584 / 11 | 1080×1920, 25 fps, audio | Passed: exact frame count/duration and complete FFmpeg decode |

Both full runs finished successfully. Each output contains 584 video frames and
23.36 seconds of audio. The model runs at 1024 for 2×, but encoding follows the
RGB512 blend/paste step and retains the original 1080×1920 scene dimensions.

## Hugging Face source and local-weight inference — 2026-11-03

- **23/23 unit tests passed**, including the released Hugging Face checkpoint
  configuration and offline cache resolution.
- The wheel built and installed into a separate directory. Both GPU checks
  below used its installed `livevsr` command outside the source checkout.
- Only the README assets were supplied through `--model-dir models --offline`:
  `TaoMate-FVSR.safetensors`, Wan2.2 VAE, HuBERT Large, GPEN-BFR-512,
  RetinaFace-R50, and ParseNet. No external reference image was supplied;
  the input first frame went through the default **one GPEN pass**.
- Both runs strictly loaded all 886 DiT tensors, VAE, and HuBERT. They used one
  NVIDIA H20 (96 GB) each, Python 3.10.20, PyTorch 2.9.0+cu128, and
  Transformers 5.13.1.

| Installed CLI | Input frames | Steps | Output | Result |
|---|---:|---:|---|---|
| `--scale 1` | 73 | 10 | 512×512, 25 fps, audio | Passed, including two temporal windows |
| `--scale 2` | 5 | 2 | 1024×1024, 25 fps, audio | Passed, short smoke test |

Both outputs passed complete FFmpeg decoding and exact frame-count checks.
Inputs were ordered PNG frames with a separate audio file. The 2× run is a
short functionality check, not a full-length/default-step or quality benchmark.

The default checkpoint now points to `TaoLiveAIGC/TaoMate-FVSR` on Hugging Face.
Authenticated metadata access confirmed the private repository and the
10,326,266,408-byte checkpoint. The GPU host could not connect directly to
Hugging Face, so these runs validate the documented local-weight flow, not
online downloading on that host.

## Documentation and model sources — 2026-11-02

- **22/22 unit tests passed**, including official-URL resolution, cached offline
  reuse, offline cache misses, and interrupted-download cleanup.
- CLI help, README/guide relative links, shell example syntax, and diff formatting passed.
- Official GPEN, RetinaFace, ParseNet, Wan VAE, and the listed HuBERT file URLs
  returned HTTP 200 on link checks. This check did not download their full weights.
- At that time, the Taomatefvsr release address was unset (`checkpoint.repo_id: null`).
  Third-party models now use their original publisher addresses.
- Sampling, reference preparation defaults, and model execution are unchanged;
  no new GPU inference benchmark was run for this documentation/download update.

## 0.2.0 — current recipe, 2026-11-01

The current GPEN default is **one pass** in both the CLI and Python API.
The runtime results below retain the pass counts used in the original runs.

- **18/18 unit tests passed**, including first-window LR prefix reuse, joint K2
  negative-branch semantics, shot-local smoothing, short shots, crop sizing,
  lossless PNG input, histogram matching and bbox paste.
- The crop port produced **exactly equal integer boxes** to the accepted
  shot-local pipeline on two clips, 226 source frames each (452 total).
- Official align-once / two-generator-pass / inverse-once GPEN preparation
  produced **pixel-identical** output to the accepted reference adapter on one
  first-frame crop. This does not measure perceptual improvement.
- Compilation, CLI help and a `livevsr-0.2.0` wheel build passed with modern
  setuptools. macOS system Python 3.9 is not a supported build environment;
  use the documented Python 3.10+ requirement.
- The wheel installed in a separate environment and its console entry point
  ran outside the source checkout.
- Joint reference/K2 guidance matched the accepted nested audio/reference
  implementation exactly in BF16 (`rtol=0`, `atol=0`).

Current GPU smoke checks use one NVIDIA H20 (96 GB) per process:

| Entry | Scale | Frames | Steps | Result |
|---|---:|---:|---:|---|
| Python pipeline, prepared reference | 1× | 73 | 10 | Passed: 512×512, two windows, first-window LR prefix and continuous decoder |
| Python pipeline, prepared reference | 2× | 5 | 2 | Passed: 1024×1024, 25 fps, audio and full FFmpeg decode |
| Installed CLI, edited-reference branch + GPEN2 + CPU offload | 1× | 5 | 2 | Passed: 512×512, separate audio, offline local assets |

The 2× check is a short smoke test, not a new full-length/default-step benchmark.
All three videos passed complete FFmpeg decoding and contained the exact
requested frame count at 25 fps with audio. The installed CLI removed its
temporary media directory. These runs loaded all 886 DiT tensors strictly.

The scene helpers accept externally extracted DWPose tracks and a supplied
soft mask. Detector deployment and private SDK mask redistribution are outside
this package. Full-scene encoded-video parity is not implied by box or blend
function parity. GPT-Image editing is external and was not invoked by these tests.

## 0.1.0 — historical checks, 2026-10-28

The following results belong to the original release, before the LR-prefix and
reference-preparation update; they are retained as historical coverage only.

### Automated checks

- **10/10 unit tests passed** with PyTorch available: window coverage (including
  short clips and partial tails), overlap indices, audio/reference guidance,
  schedule endpoints, decoder convolution stripes, overwrite protection, local
  assets, and unconfigured Hugging Face placeholders.
- Python compilation and command-line help passed.
- A `livevsr-0.1.0` wheel was built with setuptools 80.10.2, installed in a separate
  virtual environment, and invoked outside the source checkout. The model
  definitions, model-address configuration, and console entry point were present.

### GPU runtime

Hardware: NVIDIA H20 (96 GB), one GPU per inference process. Environment:
Python 3.10, PyTorch 2.9.0+cu128, Transformers 5.13.1, NumPy 2.2.6,
Pillow 12.3.0, and safetensors 0.8.0.

| Entry | Scale | Frames | Steps | Reference | Result |
|---|---:|---:|---:|---|---|
| Python pipeline | 1× | 73 | 10 | GPEN, one pass | Passed: 512×512, two windows |
| Python pipeline | 2× | 5 | 2 | GPEN, one pass | Passed: 1024×1024 |
| Installed CLI + CPU offload | 1× | 5 | 2 | Prepared image + separate audio | Passed: 512×512 |
| Installed CLI | 2× | 73 | 10 | GPEN, two passes | Passed: 1024×1024, two windows |

All successful runs produced exactly the requested frame count, 25 fps video,
and an audio stream. The output also passed a complete FFmpeg decode. The
installed CLI removed its temporary media directories after completion. GPEN
preparation, strict loading of the complete 886-tensor DiT, VAE, and HuBERT ran
with real weights. A small-latent DiT forward matched the original implementation
exactly (`rtol=0`, `atol=0`); this is a core-forward check, not a claim of identical
end-to-end videos across platforms.

At the time of the 0.1.0 checks, hosted LiveVSR/GPEN downloads were unverified
because their repository IDs were placeholders. Other GPU
types, dependency versions, and long-video memory requirements have not been
characterized by these smoke tests.
