# TaoMate-FVSR for ComfyUI

English | [简体中文](README.zh-CN.md) | [Original project](https://github.com/TaoLiveAIGC/TaoMate-FVSR/tree/main)

Restore faces in generated videos with **CPU face cropping → GPEN reference → TaoMate-FVSR → paste back with audio**. Reuses the original BF16 weights without conversion and targets a 16 GB GPU budget. Full-scene output keeps the input dimensions; it is not whole-video upscaling.

**Experimental version.** Install manually from the `comfyui` branch using the instructions below. ComfyUI Manager registration is not claimed.

## Requirements

- Working ComfyUI with CUDA PyTorch and BF16-capable NVIDIA hardware. Tested on Linux / Python 3.10 / PyTorch 2.9.0+cu128; Windows installation is not yet verified.
- Git, FFmpeg and FFprobe on PATH.
- About **14.89 GB** for ten model/config files, plus disk space for frame caches and output.
- **32 GB system RAM recommended**; the longest tested clip used about 17.75 GiB of process memory.
- 16 GB VRAM is the first target. Measurements are from an H20 with a memory budget, **not a physical 16 GB consumer GPU**. 8/12 GB operation is not validated.

No H3, VOSR2, DWPose or MMCV is needed for this workflow.

## Install

From `ComfyUI/custom_nodes`:

```bash
git clone --depth 1 --single-branch --branch comfyui https://github.com/TaoLiveAIGC/TaoMate-FVSR.git ComfyUI-TaoMate-FVSR
cd ComfyUI-TaoMate-FVSR
python -m pip install -r requirements.txt
```

Use **ComfyUI's own Python environment**, not an unrelated system Python. Requirements install the pinned upstream `livevsr` package and Ultralytics. Keep ComfyUI's working CUDA PyTorch installation.

For Windows portable, open Command Prompt in `ComfyUI_windows_portable`:

```bat
git clone --depth 1 --single-branch --branch comfyui https://github.com/TaoLiveAIGC/TaoMate-FVSR.git ComfyUI\custom_nodes\ComfyUI-TaoMate-FVSR
.\python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-TaoMate-FVSR\requirements.txt
cd ComfyUI\custom_nodes\ComfyUI-TaoMate-FVSR
..\..\..\python_embeded\python.exe scripts\download_models.py --dest ..\..\models\TaoMate-FVSR --gpen-source-dir ..\..\models\TaoMate-GPEN-source
```

These are expected Windows paths, not a completed Windows acceptance test. Git and FFmpeg/FFprobe must also be available.

## Reuse the existing weights

There is **no separate ComfyUI checkpoint**. The original TaoMate-FVSR checkpoint, VAE, HuBERT and GPEN weights are reused. YOLO face detection adds a roughly 6.23 MB file. No quantization or checkpoint conversion is needed.

From the node directory, using ComfyUI's Python:

```bash
python scripts/download_models.py --dest ../../models/TaoMate-FVSR --plan
python scripts/download_models.py --dest ../../models/TaoMate-FVSR --gpen-source-dir ../../models/TaoMate-GPEN-source
python scripts/download_models.py --dest ../../models/TaoMate-FVSR --check --gpen-source-dir ../../models/TaoMate-GPEN-source
```

On Windows portable, replace `python` here with `..\..\..\python_embeded\python.exe`. `--plan` lists assets without networking. Downloads are explicit; importing nodes or opening a workflow does not download models. The tool checks fixed sizes and SHA256 hashes, skips valid files and resumes partial downloads. Invalid existing files are reported, not overwritten.

Useful download options:

| Option | Purpose |
| --- | --- |
| `--endpoint https://hf-mirror.com` | Mirror HF assets with unchanged revisions/hashes; GPEN OSS and GitHub URLs are unchanged. |
| `--reuse /path/to/existing/models` | Verify then symlink existing files. Repeat for other directories, including HuBERT. No disk scan. Symlink permission required. |
| `--check` | Verify only; no download or repair. |
| `--only checkpoint_config` | Select one asset ID from the manifest. |

Expected layout:

```text
ComfyUI/models/
├── TaoMate-FVSR/
│   ├── TaoMate-FVSR.safetensors
│   ├── config.json
│   ├── Wan2.2_VAE.pth
│   ├── hubert/
│   │   ├── config.json
│   │   ├── preprocessor_config.json
│   │   └── pytorch_model.bin
│   ├── GPEN-BFR-512.pth
│   ├── RetinaFace-R50.pth
│   ├── ParseNet-latest.pth
│   └── face_yolov8n.pt
└── TaoMate-GPEN-source/       # source code, separate from weights
```

Select `(root)` in **Load Model**. Set the GPEN node's `gpen_repo` to the **absolute path of TaoMate-GPEN-source**, not the weights directory. Alternatively, set `local_directory` to an existing complete model folder, or use the `taomate` / `TaoMate-FVSR` key in ComfyUI's `extra_model_paths.yaml`.

For manual downloads, [assets.json](assets.json) lists official URLs, exact revisions, destinations, sizes and SHA256 hashes. GPEN source is pinned to `2c736702983368847fb544d234a22ac7cff25802`. More downloader details: [model setup](docs/model-setup.md).

## Start with a 16 GB budget

Add `--disable-cuda-malloc` to your usual launch command. From the ComfyUI directory:

```bash
python main.py --disable-cuda-malloc
```

Windows portable equivalent, from its root directory:

```bat
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --disable-cuda-malloc
```

This enables PyTorch's native allocator so the default **15 GiB allocator budget** can be enforced, leaving space for CUDA overhead. With `cudaMallocAsync`, offloading still runs but a hard budget is not claimed. The budget does not guarantee against every out-of-memory condition; avoid concurrent GPU generation jobs.

Restart ComfyUI after installation. The five nodes appear under `TaoMate FVSR`.

## First workflow

Drag [yolo-gpen-workflow.json](examples/yolo-gpen-workflow.json) into the canvas. It connects **Load Model**, **Prepare Video**, **GPEN Reference**, **Restore Face Video**, and **Save / Paste Back Video**, plus crop/reference previews and an output-path preview.

1. Set `video_path` to a file accessible **on the ComfyUI server**. Choose `yolo_face`, keep `crop_scale=0.8` and `max_frames=77` for a short first run. Set the model and GPEN source paths. Leave `detector_path` empty when YOLO is in the model folder.
2. **Preview first:** mute both **Save / Paste Back Video** and its downstream **Preview as Text / PreviewAny** output node, then queue. Crop and GPEN image previews remain active. For crop-only checking, use a separate Prepare Video → Preview Image graph. Check the face location and matching reference framing.
3. Unmute both output nodes and queue again. Keep 10 steps initially. The MP4 is saved under ComfyUI's `output` folder; its actual path appears on the canvas.
4. Once the short result looks right, set `max_frames=0` for the full clip. The longest fully tested input is **14.88 seconds**, not an unlimited-length guarantee.

Leave `audio_path` blank for original audio; a supplied file replaces it and provides audio conditioning. Output audio is AAC with source channels and sample rate; short audio is padded with silence. Video is normalized to 25 fps.

Already have a face crop? Use [cropped-face-workflow.json](examples/cropped-face-workflow.json), `cropped_face` mode and a **single 512×512 reference matching the first crop**. Output is 512×512. For a stationary face, `manual_roi` uses a fixed square `roi_x/y/side` and pastes back to original scene dimensions.

API versions: [YOLO + GPEN](examples/yolo-gpen-api.json), [cropped face](examples/cropped-face-api.json). Replace their paths and submit `{"prompt": <workflow contents>}` to ComfyUI's `/prompt`.

## Settings and boundaries

| Setting | Meaning |
| --- | --- |
| `crop_scale=0.8` | Expanded YOLO crop side multiplier. Adjust after previewing; not an upscale factor. |
| `max_frames=77 / 0` | Short trial / full clip. |
| `budget_gib=15` | Allocator budget with the native allocator. |
| `steps=10` | Tested default. |
| `audio_path` | Blank for original audio, or local replacement file. |
| `gpen_repo` | GPEN source checkout, not weights. |

The first version supports **one person in one shot, 512-pixel face inference**. Suspected cuts are rejected: split into shots first. Missing detections retain frame timing and leave those frames unchanged at paste-back. Tracking is geometric, not identity recognition. Inspect very small faces, turns and crossings.

`crop_scale` differs from upstream `scale2`, which runs a 1024 face then returns to 512 for paste-back. This node does not expose `scale2` or double the scene resolution.

Keep reference protocols separate:

- **Default GPEN:** CPU GPEN enhances the first crop; the default workflow.
- **Enhanced reference (+):** a matching enhanced image supplied by the user.
- **Paper GT reference:** a research evaluation protocol, not a reference available for arbitrary user videos.

Frames/output are cached on disk rather than passed as a whole-clip Comfy IMAGE tensor. Disk and host memory usage still grow with duration. Save outputs before clearing ComfyUI's temporary folder. See [node details](docs/node-usage.md).

## Validation

Linux / NVIDIA H20 / original BF16 / 10 steps:

| Test | Result |
| --- | --- |
| YOLO + default GPEN + restore + native paste-back + audio | Passed in actual ComfyUI execution |
| Continuous 14.88 s input | 372 frames, 7 windows; full output decode passed |
| Output | 1344×768, 25 fps; audio and video both 14.88 s |
| Peak GPU process memory | 15,334 MiB (14.975 GiB) |
| Peak main-process host memory | About 17.75 GiB |
| Cancel inference, then rerun | Passed |
| Longer clips / physical 16 GB consumer GPU / Windows | Not yet validated |

The 14.88-second test took **745 seconds including slow cold loading and disk reads**; this is not steady-state throughput. Earlier 72/148-frame and multi-window runs also passed. Browser operation and clean-machine installation still need acceptance testing; backend execution and workflow schema validation do not replace those checks.

## Troubleshooting

- **Missing nodes/import error:** install into the Python that starts ComfyUI, inspect its startup log and restart.
- **Missing model:** check the layout and run `--check`. `gpen_repo` needs source code; `local_directory` needs weights.
- **Tiny face/bad crop:** preview and tune `crop_scale`, or try a fixed `manual_roi` for a stationary face. Cropping cannot guarantee recovery of missing detail.
- **Shot-change error:** split at cuts; one reference and temporal history should not cross unrelated shots.
- **Slow first run:** model loading, CPU GPEN and frame caching add cost. Cold-run timing is not a speed promise.
- **Out of memory:** check the launch flag and other GPU jobs; start with a short clip. Fewer steps do not remove model-weight memory requirements. Lower-memory GPUs remain unvalidated.
- **Cancel seems delayed:** denoising supports cancellation; CPU GPEN and model loading do not offer per-layer immediate cancellation.

## Sources and licenses

Uses [TaoMate-FVSR](https://github.com/TaoLiveAIGC/TaoMate-FVSR/tree/main), [GPEN](https://github.com/yangxy/GPEN), [Ultralytics](https://github.com/ultralytics/ultralytics), and model sources in [assets.json](assets.json). Their respective code and weight licenses continue to apply. Upstream `livevsr`: `bc924d8572c4e5050cdc8a86e7fab13d30a327db`. Main checkpoint HF revision: `a37c97010a2af15d561f2515481134a8015cc300`.
