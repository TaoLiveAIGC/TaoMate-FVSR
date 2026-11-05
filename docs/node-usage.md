# Node reference

All paths are paths on the machine running ComfyUI. FFmpeg and FFprobe must be available on PATH. Workflows operate on files rather than storing an entire video in a ComfyUI IMAGE batch.

| Node | Purpose |
| --- | --- |
| Load Model | Select `models/TaoMate-FVSR`, an extra model path under `taomate` / `TaoMate-FVSR`, or `local_directory`. Loading is lazy. |
| Prepare Video / Preview Crop | Decode to 25 fps and prepare 512×512 crops. Choose `yolo_face`, `manual_roi`, or already-cropped `cropped_face`. Preview the first crop before restoration. |
| GPEN Reference (CPU) | Generate the default reference from the first crop using separately installed GPEN source/weights. Set `gpen_repo` to that source directory. |
| Restore Face Video | Restore using original BF16 weights, a single RGB reference and optional replacement audio. Default inference uses 10 steps and a 15 GiB allocator budget. |
| Save / Paste Back Video | Save under ComfyUI output and return the MP4 path. Full-scene modes retain original dimensions; cropped-face mode saves 512×512 video. |

`max_frames=0` processes the complete clip; examples limit the first run to 77 frames. A manual ROI is a fixed square (`x`, `y`, `side`) and does not track movement. `detector_path` can explicitly locate YOLO weights; when empty, the node searches the TaoMate model directory and `models/ultralytics`.

The GPEN reference is the default protocol. A user-enhanced reference is the **+ protocol**; a ground-truth reference used for paper evaluation is a separate protocol. An external reference should be a single 512×512 RGB image matching the current crop composition. Reusing a reference from a different crop can misalign the result.

The runtime unloads upstream ComfyUI GPU models before restoration and unloads its GPU components after success, failure or cancellation. CPU model caches can remain. The 16 GB target was tested as a budget on an H20, not on every consumer 16 GB card. Keep sufficient system memory and temporary disk space available.

The native `livevsr.composite.paste_head` performs per-channel histogram matching against the corresponding source crop, then blends using the upstream static face mask and linear interpolation. Pixels outside the crop rectangle stay unchanged before video encoding. Missing-face frames retain the original image.

Original audio channels and sample rate are retained when encoding to AAC. `audio_path` substitutes another audio track; model conditioning separately uses 16 kHz mono audio. Short audio is padded with silence. The processing/output frame rate is 25 fps.

Frames and intermediate results are cached under ComfyUI's temporary `taomate/<random ID>` directory. Save outputs before clearing temporary data; long clips require additional disk space and system memory. The node currently supports single-shot, single-person input, with scale1/512 face restoration. Model scale2 would mean 1024 face inference followed by 512 paste-back, not doubling the scene; it is not exposed here.

## Workflows

Drag [YOLO + GPEN workflow](../examples/yolo-gpen-workflow.json) into ComfyUI for automatic cropping and default reference generation. Fill in your video, model, GPEN source and detector paths. Run the crop/reference previews first, then restore and save.

[Cropped-face workflow](../examples/cropped-face-workflow.json) accepts an already cropped video and an external reference through Load Image. Classify results according to the reference protocol used.

The corresponding `*-api.json` files are API prompts, not draggable canvas workflows. Replace their placeholders and submit as `{"prompt": <JSON contents>}` to ComfyUI's `/prompt` endpoint. Both UI workflows display the final MP4 path with Preview as Text.
