# Model setup

Run these commands from this custom node's directory using ComfyUI's Python:

```bash
python scripts/download_models.py --dest ../../models/TaoMate-FVSR --plan
python scripts/download_models.py --dest ../../models/TaoMate-FVSR --gpen-source-dir ../../models/TaoMate-GPEN-source
python scripts/download_models.py --dest ../../models/TaoMate-FVSR --check --gpen-source-dir ../../models/TaoMate-GPEN-source
```

Select `(root)` in Load Model. Set the GPEN node's `gpen_repo` to the absolute path of `TaoMate-GPEN-source`. The downloader uses the Python standard library; importing nodes or opening workflows never starts downloads.

The default manifest contains ten files totaling approximately 14.89 GB: original TaoMate-FVSR weights/configuration, Wan VAE, HuBERT weights/configuration/preprocessor, GPEN512, RetinaFace, ParseNet and YOLO face weights. No converted checkpoint is required. DWPose/MMCV and optional GPEN1024 are not included.

## Existing files and network options

Add `--reuse /path/to/existing/models` to reuse verified files. Repeat this option for multiple directories, including a HuBERT directory if needed. Only known filenames and manifest-relative paths are checked; the downloader does not recursively scan storage. Matching size and SHA256 are required before creating a symlink. If symlink creation is unavailable, use an existing complete model directory through `local_directory` or ComfyUI's `extra_model_paths.yaml` instead.

The default Hugging Face endpoint is official. An optional `--endpoint https://hf-mirror.com` changes only that endpoint; revision and SHA256 checks remain fixed. This does not change GPEN OSS or GitHub URLs. Use `--only checkpoint_config` to download/check one manifest asset; other IDs are in [assets.json](../assets.json).

`--plan` does not access the network or hash large files. `--check` verifies without downloading or repairing. Valid destination files are skipped; mismatched existing files cause an error and are preserved. Downloads use `.part` files, bounded retries and HTTP Range resume. A server that ignores Range triggers a fresh partial download. Complete files must pass size and SHA256 checks before publication. The destination filesystem must support hard links for atomic creation. Do not run concurrent downloaders against the same destination. An invalid complete partial is preserved for inspection; move it aside before retrying.

## GPEN source and provenance

`--gpen-source-dir` clones [official GPEN](https://github.com/yangxy/GPEN) at `2c736702983368847fb544d234a22ac7cff25802`, with implicit LFS downloads disabled. Existing source directories are only checked for the expected HEAD and clean tracked files; they are not updated or cleaned. GPEN runs in a separate CPU process.

The manifest records source URLs, revisions, byte sizes, SHA256 and digest provenance. `local_verified_sha256` identifies a digest calculated from verified downloaded content, not a publisher-signed checksum. Model licenses remain separate from this repository's source license; see [third-party notices](../THIRD_PARTY_NOTICES.md).
