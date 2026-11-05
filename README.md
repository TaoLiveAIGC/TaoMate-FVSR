# <img src="https://taoliveaigc.github.io/TaoMate-FVSR/assets/favicon.svg" width="40" height="40" alt="" /> TaoMate-FVSR

**Audio-Guided Facial Video Restoration**

[![arXiv](https://img.shields.io/badge/arXiv-2611.03393-red.svg)](https://arxiv.org/abs/2611.03393)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://taoliveaigc.github.io/TaoMate-FVSR/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/TaoLiveAIGC/TaoMate-FVSR)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Experimental-blue)](https://github.com/TaoLiveAIGC/TaoMate-FVSR/tree/comfyui)
[![Code License](https://img.shields.io/badge/Code-Apache--2.0-green)](LICENSE)

[Demo](#demo) · [Results](#experimental-results) ·
[Installation](#installation) · [Model Downloads](#model-downloads) ·
[Inference](#inference) · [Citation](#citation) ·
[Technical Guide](docs/inference.md)

**Prefer ComfyUI?** Use the dedicated [ComfyUI branch](https://github.com/TaoLiveAIGC/TaoMate-FVSR/tree/comfyui) for custom nodes, installation instructions, model setup and ready-to-import workflows: [English guide](https://github.com/TaoLiveAIGC/TaoMate-FVSR/blob/comfyui/README.md) · [中文安装与使用指南](https://github.com/TaoLiveAIGC/TaoMate-FVSR/blob/comfyui/README.zh-CN.md). It reuses the original TaoMate-FVSR weights; no converted checkpoint is required.

TaoMate-FVSR restores face videos with audio conditioning, sparse visual
references, and generated-frame history. It supports restoration with the
original audio and visual dubbing with replacement speech.

- **Full-video inference:** DWPose crop → face restoration → color/mask blend → paste-back.
- **1× / 2× inference:** run the model at 512×512 or 1024×1024, then return to
  512×512 before compositing. The output scene retains its original dimensions.
- **Audio-guided generation:** use the video's original audio or provide target speech.
- **Flexible references:** prepare the first frame with GPEN, use an externally
  edited first frame, or supply a finished reference image. Default enhancement
  uses one GPEN-BFR-512 pass for 1× and one GPEN-BFR-1024 pass for 2×.

The `main` branch contains the standalone inference runtime; the `comfyui`
branch provides the ComfyUI integration. Model weights are downloaded
separately, and input media is supplied locally.

## News

- **2026-11-03:** 🎉 Inference code released, including full-scene crop/paste-back, scale-matched GPEN references, and 1× / 2× inference through one entry point.
- **2026-11-02:** Our [paper](https://arxiv.org/abs/2611.03393) is available on arXiv.

## Demo

**The Gallery** — TaoMate-FVSR+ restoration example from the project page.

<table>
  <thead>
    <tr><th align="center">Input</th><th align="center">TaoMate-FVSR+</th></tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><a href="https://taoliveaigc.github.io/TaoMate-FVSR/#top"><img src="https://taoliveaigc.github.io/TaoMate-FVSR/assets/media/h3-768p-20261102/ours/gallery__original/input.jpg" alt="The Gallery — input face crop" width="320"></a></td>
      <td align="center"><a href="https://taoliveaigc.github.io/TaoMate-FVSR/#top"><img src="https://taoliveaigc.github.io/TaoMate-FVSR/assets/media/h3-768p-20261102/ours/gallery__original/ours.jpg" alt="The Gallery — TaoMate-FVSR+ restored face crop" width="320"></a></td>
    </tr>
  </tbody>
</table>

[▶ Watch the synchronized comparison](https://taoliveaigc.github.io/TaoMate-FVSR/#top)
· [Restored video (MP4)](https://raw.githubusercontent.com/TaoLiveAIGC/TaoMate-FVSR/media/assets/media/h3-768p-20261102/ours/gallery__original/ours.mp4)
· [More demos](https://taoliveaigc.github.io/TaoMate-FVSR/#collection)

*The “+” variant uses a GPT-Image-enhanced reference frame. Click either preview to play the before/after comparison.*

## Experimental Results

Results from our [paper](https://arxiv.org/pdf/2611.03393). ↑ indicates higher
is better; ↓ indicates lower is better. Scores are video-wise means, and GT
rows provide undegraded-source context rather than a competing method.
<p><strong>Bold</strong> marks the best method result; <ins>underline</ins> marks the second-best.
GT is excluded from both rankings.</p>

### Facial restoration under original audio

The 150-video benchmark includes MEAD, HDTF, TalkVid, and our internal
talking-head dataset. All methods receive the same training-style degradation,
including 4× downsampling followed by resizing back to 512×512. Scores are
computed on complete prepared frames using native uncompressed RGB outputs,
averaged equally over common valid clips.

<table>
<thead>
  <tr><th align="left">Method</th><th align="right">PSNR ↑</th><th align="right">SSIM ↑</th><th align="right">LPIPS ↓</th><th align="right">DISTS ↓</th><th align="right">Sync-C ↑</th><th align="right">Sync-D ↓</th><th align="right">MUSIQ ↑</th><th align="right">CLIP-IQA ↑</th><th align="right">FasterVQA ↑</th><th align="right">DOVER ↑</th></tr>
</thead>
<tbody>
  <tr><td align="left">GT</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">6.959</td><td align="right">7.863</td><td align="right">62.70</td><td align="right">0.4525</td><td align="right">0.7735</td><td align="right">0.6148</td></tr>
  <tr><td align="left">GPEN</td><td align="right">26.77</td><td align="right">0.7970</td><td align="right">0.2267</td><td align="right">0.1825</td><td align="right">6.309</td><td align="right">8.288</td><td align="right"><ins>76.47</ins></td><td align="right"><ins>0.7032</ins></td><td align="right"><ins>0.8931</ins></td><td align="right"><ins>0.8505</ins></td></tr>
  <tr><td align="left">CodeFormer</td><td align="right">27.38</td><td align="right">0.8045</td><td align="right">0.2195</td><td align="right">0.1876</td><td align="right">5.856</td><td align="right">8.693</td><td align="right">74.42</td><td align="right">0.6649</td><td align="right">0.8854</td><td align="right">0.7954</td></tr>
  <tr><td align="left">InstantRestore</td><td align="right">27.42</td><td align="right">0.8283</td><td align="right"><ins>0.1380</ins></td><td align="right"><ins>0.1421</ins></td><td align="right">6.209</td><td align="right">8.403</td><td align="right">68.11</td><td align="right">0.5166</td><td align="right">0.8562</td><td align="right">0.7704</td></tr>
  <tr><td align="left">SparkVSR</td><td align="right"><strong>30.62</strong></td><td align="right"><strong>0.8863</strong></td><td align="right">0.1755</td><td align="right">0.1586</td><td align="right">6.545</td><td align="right">8.170</td><td align="right">65.66</td><td align="right">0.4417</td><td align="right">0.8459</td><td align="right">0.7467</td></tr>
  <tr><td align="left">RG-FVR</td><td align="right">26.69</td><td align="right">0.8308</td><td align="right">0.1645</td><td align="right">0.1461</td><td align="right">5.685</td><td align="right">9.024</td><td align="right">70.51</td><td align="right">0.5246</td><td align="right">0.8688</td><td align="right">0.7407</td></tr>
  <tr><td align="left">SeedVR2-3B</td><td align="right">27.49</td><td align="right">0.8456</td><td align="right">0.2227</td><td align="right">0.1630</td><td align="right">6.377</td><td align="right">8.339</td><td align="right">57.36</td><td align="right">0.4412</td><td align="right">0.7548</td><td align="right">0.5601</td></tr>
  <tr><td align="left">SeedVR2-7B</td><td align="right">28.60</td><td align="right">0.8584</td><td align="right">0.2046</td><td align="right">0.1525</td><td align="right">6.359</td><td align="right">8.319</td><td align="right">54.30</td><td align="right">0.3939</td><td align="right">0.7433</td><td align="right">0.5212</td></tr>
  <tr><td align="left">FlashVSR-Tiny</td><td align="right">27.74</td><td align="right">0.8497</td><td align="right">0.1563</td><td align="right">0.1428</td><td align="right">6.490</td><td align="right">8.275</td><td align="right">70.19</td><td align="right">0.5395</td><td align="right">0.8810</td><td align="right">0.7709</td></tr>
  <tr><td align="left">FlashVSR-Full</td><td align="right">27.52</td><td align="right">0.8435</td><td align="right">0.1652</td><td align="right">0.1458</td><td align="right">6.468</td><td align="right">8.316</td><td align="right">71.23</td><td align="right">0.5776</td><td align="right">0.8820</td><td align="right">0.7729</td></tr>
  <tr><td align="left">FlashVSR+</td><td align="right">26.68</td><td align="right">0.8379</td><td align="right">0.1652</td><td align="right">0.1473</td><td align="right">6.524</td><td align="right">8.230</td><td align="right">69.83</td><td align="right">0.4947</td><td align="right">0.8775</td><td align="right">0.7851</td></tr>
  <tr><td align="left">Ours</td><td align="right"><ins>28.67</ins></td><td align="right"><ins>0.8836</ins></td><td align="right"><strong>0.0793</strong></td><td align="right"><strong>0.0858</strong></td><td align="right"><ins>7.413</ins></td><td align="right"><ins>7.561</ins></td><td align="right">66.84</td><td align="right">0.4908</td><td align="right">0.8245</td><td align="right">0.7178</td></tr>
  <tr><td align="left">Ours+</td><td align="right">23.52</td><td align="right">0.7525</td><td align="right">0.2402</td><td align="right">0.1902</td><td align="right"><strong>7.453</strong></td><td align="right"><strong>7.520</strong></td><td align="right"><strong>77.62</strong></td><td align="right"><strong>0.7081</strong></td><td align="right"><strong>0.8997</strong></td><td align="right"><strong>0.8823</strong></td></tr>
</tbody>
</table>

Reference-capable baselines and Ours use the same undegraded GT first frame.
Ours+ uses a GPT-Image-enhanced version of that frame, with the model and sampler
unchanged. Methods without a reference interface retain their supported inputs.

### Visual dubbing under replacement audio

This table uses the 30-clip subset selected post hoc from 44 valid clips by
Ours+ Sync-C. All pipelines use those same clips and replacement-audio pairs,
with the same degraded sources and no additional degradation between stages.
Arrows indicate execution order: restoration first, then dubbing. Since there
is no paired target-mouth video for replacement speech, this table reports
synchronization and no-reference quality metrics only.

<table>
<thead>
  <tr><th align="left">Pipeline</th><th align="right">Sync-C ↑</th><th align="right">Sync-D ↓</th><th align="right">MUSIQ ↑</th><th align="right">CLIP-IQA ↑</th><th align="right">FasterVQA ↑</th><th align="right">DOVER ↑</th></tr>
</thead>
<tbody>
  <tr><td align="left">FlashVSR Full → X-Dub</td><td align="right">6.137</td><td align="right">9.332</td><td align="right">73.25</td><td align="right"><ins>0.6582</ins></td><td align="right"><ins>0.8750</ins></td><td align="right"><ins>0.7845</ins></td></tr>
  <tr><td align="left">FlashVSR Full → KeySync</td><td align="right">7.347</td><td align="right">7.840</td><td align="right">68.07</td><td align="right">0.4908</td><td align="right">0.8736</td><td align="right">0.7647</td></tr>
  <tr><td align="left">FlashVSR Full → TBDub Teacher</td><td align="right">7.736</td><td align="right">7.543</td><td align="right">70.18</td><td align="right">0.6208</td><td align="right">0.8650</td><td align="right">0.7179</td></tr>
  <tr><td align="left">FlashVSR Full → TBDub Student</td><td align="right">7.469</td><td align="right">8.196</td><td align="right">69.72</td><td align="right">0.5834</td><td align="right">0.8641</td><td align="right">0.7309</td></tr>
  <tr><td align="left">FlashVSR Full → LipForcing</td><td align="right"><ins>7.917</ins></td><td align="right"><ins>7.186</ins></td><td align="right">66.18</td><td align="right">0.4973</td><td align="right">0.8476</td><td align="right">0.6958</td></tr>
  <tr><td align="left">SparkVSR → X-Dub</td><td align="right">5.853</td><td align="right">9.409</td><td align="right"><ins>74.12</ins></td><td align="right">0.6335</td><td align="right">0.8596</td><td align="right">0.7822</td></tr>
  <tr><td align="left">SparkVSR → KeySync</td><td align="right">7.175</td><td align="right">7.964</td><td align="right">62.17</td><td align="right">0.4271</td><td align="right">0.8318</td><td align="right">0.7199</td></tr>
  <tr><td align="left">SparkVSR → TBDub Teacher</td><td align="right">7.208</td><td align="right">7.817</td><td align="right">67.94</td><td align="right">0.5383</td><td align="right">0.8389</td><td align="right">0.7030</td></tr>
  <tr><td align="left">SparkVSR → TBDub Student</td><td align="right">7.090</td><td align="right">8.401</td><td align="right">67.25</td><td align="right">0.4987</td><td align="right">0.8412</td><td align="right">0.7174</td></tr>
  <tr><td align="left">SparkVSR → LipForcing</td><td align="right">7.858</td><td align="right">7.224</td><td align="right">60.70</td><td align="right">0.4270</td><td align="right">0.8079</td><td align="right">0.6517</td></tr>
  <tr><td align="left">Ours+</td><td align="right"><strong>8.175</strong></td><td align="right"><strong>7.083</strong></td><td align="right"><strong>77.56</strong></td><td align="right"><strong>0.7070</strong></td><td align="right"><strong>0.8958</strong></td><td align="right"><strong>0.8709</strong></td></tr>
</tbody>
</table>

SparkVSR and Ours+ share the same GPT-Image-enhanced GT first frame.
The two comparisons above use fixed-reference motion conditioning, seed 42,
source noise 0.8, ten Euler steps, shift 5, audio guidance 4.0, and reference
guidance 1.5. These are the paper's evaluation settings; the default CLI uses
GPEN reference preparation and reference guidance 1.7, as documented below.

### Ablation study

Video-wise means on the same 30 videos: 15 from MEAD and 15 from HDTF. All
variants use the unenhanced GT first frame. X-Dub (adapted) and DirectMotion
compare alternative source-conditioning and reference-injection designs,
respectively; w/o Audio removes audio conditioning.

<table>
<thead>
  <tr><th align="left">Metric</th><th align="right">GT</th><th align="right">Ours</th><th align="right">w/o Audio</th><th align="right">DirectMotion</th><th align="right">X-Dub (adapted)</th></tr>
</thead>
<tbody>
  <tr><td align="left">PSNR ↑</td><td align="right">—</td><td align="right"><ins>29.10</ins></td><td align="right"><strong>29.36</strong></td><td align="right">28.24</td><td align="right">26.63</td></tr>
  <tr><td align="left">SSIM ↑</td><td align="right">—</td><td align="right"><ins>0.8852</ins></td><td align="right"><strong>0.8889</strong></td><td align="right">0.8600</td><td align="right">0.8415</td></tr>
  <tr><td align="left">LPIPS ↓</td><td align="right">—</td><td align="right"><strong>0.0850</strong></td><td align="right"><ins>0.0856</ins></td><td align="right">0.1341</td><td align="right">0.1523</td></tr>
  <tr><td align="left">MUSIQ ↑</td><td align="right">63.26</td><td align="right">68.19</td><td align="right">69.09</td><td align="right"><ins>69.49</ins></td><td align="right"><strong>70.08</strong></td></tr>
  <tr><td align="left">CLIP-IQA ↑</td><td align="right">0.4971</td><td align="right">0.5591</td><td align="right">0.5709</td><td align="right"><ins>0.5807</ins></td><td align="right"><strong>0.6145</strong></td></tr>
  <tr><td align="left">FasterVQA ↑</td><td align="right">0.7880</td><td align="right">0.8352</td><td align="right"><ins>0.8432</ins></td><td align="right"><strong>0.8595</strong></td><td align="right">0.8354</td></tr>
  <tr><td align="left">DOVER ↑</td><td align="right">0.6392</td><td align="right">0.7367</td><td align="right"><ins>0.7567</ins></td><td align="right"><strong>0.7602</strong></td><td align="right">0.7232</td></tr>
  <tr><td align="left">Sync-C ↑</td><td align="right">7.762</td><td align="right"><strong>7.621</strong></td><td align="right">4.283</td><td align="right">7.254</td><td align="right"><ins>7.407</ins></td></tr>
  <tr><td align="left">Sync-D ↓</td><td align="right">7.374</td><td align="right"><strong>7.560</strong></td><td align="right">10.524</td><td align="right"><ins>7.875</ins></td><td align="right">8.144</td></tr>
</tbody>
</table>

## Installation

### Requirements

- Linux and Python 3.10+
- An NVIDIA CUDA GPU with BF16 support
- CUDA-compatible PyTorch 2.6+ and torchvision 0.21+
- FFmpeg / FFprobe and Git
- A C++ build toolchain and Ninja for GPEN's custom operators

The current runtime has been smoke-tested on H20. 2× inference needs more GPU
memory; `--cpu-offload` trades speed for lower model residency. See
[validation details](VALIDATION.md) for tested configurations.

### Set up the environment

Clone this repository and install a CUDA-compatible PyTorch build for your GPU,
then set up the inference package:

```bash
git clone https://github.com/TaoLiveAIGC/TaoMate-FVSR.git
cd TaoMate-FVSR

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install PyTorch / torchvision for your CUDA version before this step.
pip install -e '.[reference]'
git clone https://github.com/yangxy/GPEN.git third_party/GPEN
```

Follow [GPEN's installation instructions](https://github.com/yangxy/GPEN#usage)
for its custom operators. Crop/compositing code and its Python dependencies are
included in the package; prepare the DWPose models below for full-scene inputs.

## Model Downloads

Our checkpoint is hosted on Hugging Face. Third-party assets below link directly
to their original publishers.

| Model | Purpose | Official download | Local path |
|---|---|---|---|
| TaoMate-FVSR | Audio-conditioned restoration model | [TaoMate-FVSR.safetensors](https://huggingface.co/TaoLiveAIGC/TaoMate-FVSR/resolve/main/TaoMate-FVSR.safetensors) | `models/TaoMate-FVSR.safetensors` |
| TaoMate-FVSR configuration | Inference settings | [config.json](https://huggingface.co/TaoLiveAIGC/TaoMate-FVSR/resolve/main/config.json) | `models/config.json` |
| Wan2.2 VAE | Video encoding and decoding | [Wan-AI / Wan2.2 VAE](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main/Wan2.2_VAE.pth) | `models/Wan2.2_VAE.pth` |
| HuBERT Large | Audio features | [facebook/hubert-large-ll60k](https://huggingface.co/facebook/hubert-large-ll60k) | `models/hubert/` |
| GPEN-BFR-512 | First-frame enhancement for 1× | [GPEN-BFR-512.pth](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512.pth) | `models/GPEN-BFR-512.pth` |
| GPEN-BFR-1024 | First-frame enhancement for 2× | [pytorch_model.pt — official GPEN-1024](https://www.modelscope.cn/models/damo/cv_gpen_image-portrait-enhancement-hires/resolve/master/pytorch_model.pt) | `models/GPEN-BFR-1024.pth` |
| RetinaFace-R50 | GPEN face detection and alignment | [RetinaFace-R50.pth](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/RetinaFace-R50.pth) | `models/RetinaFace-R50.pth` |
| ParseNet | GPEN face mask | [ParseNet-latest.pth](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/ParseNet-latest.pth) | `models/ParseNet-latest.pth` |
| YOLOX-L | DWPose person detection | [OpenMMLab YOLOX-L](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth) | `models/dwpose/` |
| RTMW-X | DWPose whole-body landmarks | [OpenMMLab RTMW-X](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth) | `models/dwpose/` |

### Download third-party weights

<details>
<summary><strong>Download commands — Wan VAE, HuBERT, GPEN and face components</strong></summary>

```bash
mkdir -p models/hubert

hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir models
hf download facebook/hubert-large-ll60k \
  config.json preprocessor_config.json pytorch_model.bin --local-dir models/hubert

curl -fL https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/RetinaFace-R50.pth \
  -o models/RetinaFace-R50.pth
curl -fL https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/ParseNet-latest.pth \
  -o models/ParseNet-latest.pth
```

Download the GPEN checkpoint matching `--scale` (download both if using both scales):

```bash
mkdir -p models

# --scale 1: GPEN-BFR-512
curl -fL https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512.pth \
  -o models/GPEN-BFR-512.pth

# --scale 2: GPEN-BFR-1024, published under the name pytorch_model.pt
curl -fL https://www.modelscope.cn/models/damo/cv_gpen_image-portrait-enhancement-hires/resolve/master/pytorch_model.pt \
  -o models/GPEN-BFR-1024.pth
```

For 2×, the runtime reads **`models/GPEN-BFR-1024.pth`**. Keep this local
filename even though the publisher's download URL ends in `pytorch_model.pt`.
The 1024 model is linked from the [official GPEN repository](https://github.com/yangxy/GPEN)
to its ModelScope release; no third-party weights are rehosted here.

</details>

### TaoMate-FVSR checkpoint

The checkpoint is [TaoMate-FVSR.safetensors](https://huggingface.co/TaoLiveAIGC/TaoMate-FVSR/resolve/main/TaoMate-FVSR.safetensors)
in the public `TaoLiveAIGC/TaoMate-FVSR` model repository. Download the checkpoint
and its inference configuration together for the local `--model-dir models` flow:

```bash
hf download TaoLiveAIGC/TaoMate-FVSR \
  TaoMate-FVSR.safetensors config.json --local-dir models
```

The Hub's [download statistics](https://huggingface.co/docs/hub/models-download-stats)
count requests to `config.json`; weight-only downloads are not counted under the
current configuration. Both the command above and the CLI's automatic online
checkpoint download fetch the weights and configuration. `config.json` documents
the inference settings; it does not override CLI arguments. Local checkpoints and
`--offline` reuse remain unchanged and do not require this configuration file.

Omit `--model-dir` and `--checkpoint` to download our model automatically using
[`livevsr/model_sources.json`](livevsr/model_sources.json). Wan VAE and HuBERT
download from their official Hugging Face repositories; GPEN assets use the
official URLs above. Use `--model-dir models` for the fully local layout, or
`--offline` for existing local files and caches. A checkpoint with another local
filename can still be supplied with `--checkpoint /path/to/model.safetensors`.
The checkpoint includes the audio and reference-conditioning weights; a base
Wan checkpoint or a LoRA alone is not sufficient. No model weights are stored
in this Git repository.

### Full-scene components

The default path also needs YOLOX/RTMW; the compositing mask is bundled.

<details>
<summary><strong>Required DWPose setup for full-scene inputs</strong></summary>

Install MMDetection and MMPose following the
[MMPose installation guide](https://mmpose.readthedocs.io/en/latest/installation.html).
An independent pose environment is supported through `--pose-python /path/to/pose-env/bin/python`;
this keeps OpenMMLab's PyTorch/MMCV requirements separate from the restoration
environment. The pose environment used for validation has PyTorch 2.5.1+cu121,
MMCV 2.1.0, MMDetection 3.3.0 and MMPose 1.3.2.

In that environment, download the upstream weights and resolved configs:

```bash
mim download mmdet --config yolox_l_8xb8-300e_coco --dest models/dwpose
mim download mmpose --config rtmw-x_8xb320-270e_cocktail14-384x288 --dest models/dwpose
```

The original grayscale paste mask is included at
[`livevsr/assets/paste_face_mask.png`](livevsr/assets/paste_face_mask.png) and in
the installed wheel. It is selected automatically, including outside the
checkout: no separate mask download or private SDK is needed. Use
`--blend-mask /path/to/custom.png` only to override it. The runtime implements
histogram matching and blending directly and uses the bundled mask unchanged.
`--pose-model-dir` overrides the YOLOX/RTMW directory. DWPose configs and weights
must still be prepared locally even when restoration weights download automatically.

</details>

## Inference

Pass the **original full-scene video**. The runtime extracts the head with
DWPose, restores it, and pastes it into the original scene at **25 fps**, with
the selected audio. An ordered 25-fps PNG directory also works with `--audio`.

### Restore a video

```bash
python infer.py --input inputs/video.mp4 --output outputs/restored.mp4 \
  --model-dir models --offline --gpen-repo third_party/GPEN --scale 1
```

This command uses only the local weights listed above. To download assets
automatically instead, omit `--model-dir models --offline`. After installation,
`livevsr` is equivalent to `python infer.py` and can run outside the checkout.
Use the same entry point with `--scale 1` (512 inference) or `--scale 2`
(512→1024 inference→512). With `--scale 2`, the first crop is resized to 1024,
enhanced once by **GPEN-BFR-1024**, and retained at 1024 for reference encoding;
it is not a resized GPEN-BFR-512 result. Generated frames are reduced with area
resampling **before** histogram matching and mask blending. Neither option
enlarges the background or changes the final scene dimensions.

<details>
<summary><strong>Separate pose environments and already-cropped inputs</strong></summary>

Add `--pose-python /path/to/pose-env/bin/python` when using a separate pose
environment. DWPose uses CPU by default; `--pose-device cuda:0` enables GPU
preprocessing, which finishes before the restoration model is loaded.

For an already-cropped **512×512** input, add `--cropped`. This skips DWPose and
scene paste-back and returns the restored 512×512 face video. It also reduces
2× model output back to 512; no scene models or paste mask are needed.

</details>

### Visual dubbing with target audio

```bash
python infer.py --input inputs/video.mp4 --audio inputs/target.wav \
  --output outputs/dubbed.mp4 --model-dir models --gpen-repo third_party/GPEN
```

Without `--audio`, the original audio is used. Target audio is padded or trimmed
to the processed video duration.

### Reference options

| Input | Reference preparation |
|---|---|
| Default | Enhance the first input crop once: **GPEN-BFR-512 for 1×; GPEN-BFR-1024 for 2×** |
| `--edited-reference inputs/edited_first_crop.png` | Apply the scale-matched GPEN model to an externally edited first crop, such as a GPT-Image edit |
| `--reference inputs/reference.png` | Use the finished model-crop reference directly; skip GPEN, but retain DWPose for full-scene inputs |

For example, use a finished reference with replacement audio:

```bash
python infer.py --input inputs/face.mp4 --audio inputs/target.wav \
  --cropped --reference inputs/reference.png --output outputs/dubbed.mp4 --model-dir models
```

References should preserve the first crop's identity, pose, mouth shape, and
framing. `--reference` and `--edited-reference` are mutually exclusive. Image
editing is performed externally; this package does not call an image-editing API.

### Main settings

| Option | Default | Description |
|---|---:|---|
| `--scale` | 1 | Model scale: 1 or 2; selects GPEN-512 or GPEN-1024; return to 512 before paste |
| `--cropped` | off | Skip full-scene crop/paste for prepared 512×512 inputs |
| `--steps` | 10 | Euler denoising steps |
| `--noise` | 0.8 | Initial latent noise mixture |
| `--audio-cfg` | 4.0 | Audio guidance |
| `--ref-cfg` | 1.7 | Joint reference/history guidance |
| `--gpen-passes` | 1 | GPEN generator passes; 2 is optional |
| `--seed` | 42 | Random seed |

Run `python infer.py --help` for all options. Windowing, reference injection,
memory handling, and full-scene integration are documented in the
[technical guide](docs/inference.md). Inputs are processed locally and are not uploaded.

## Repository Layout

```text
infer.py                    Command-line entry point
livevsr/                    Inference runtime and model definitions
livevsr/pose.py              YOLOX/RTMW inference adapter
livevsr/scene.py             Native-frame crop and RGB512 paste orchestration
livevsr/models/audio_conditioning.py  Temporal audio projection and attention
livevsr/assets/              Bundled paste mask
livevsr/model_sources.json   Hugging Face checkpoint and official component URLs
docs/inference.md           Advanced settings and scene integration
tests/                      Unit tests
```

The Python package and installed command remain named `livevsr`.

## Citation

If you find TaoMate-FVSR useful in your research, please cite:

```bibtex
@misc{li2026taomatefvsr,
  title={TaoMate-FVSR: Audio-Guided Facial Video Restoration},
  author={Xinyang Li and Zeran Xu and Qijun Gan and Chenwei Zhang and Meiguang Jin and Junfeng Ma},
  year={2026},
  eprint={2611.03393},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2611.03393}
}
```

## Acknowledgments

We thank the authors and maintainers of [Wan2.2](https://github.com/Wan-Video/Wan2.2),
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), and
[X-Dub](https://github.com/KlingAIResearch/X-Dub) for their video-model and
audio-conditioning implementations; [GPEN](https://github.com/yangxy/GPEN),
[RetinaFace](https://github.com/biubug6/Pytorch_Retinaface), and ParseNet (distributed
with GPEN) for reference-frame preparation; and
[HuBERT](https://huggingface.co/facebook/hubert-large-ll60k) for audio representations.
We thank [DWPose](https://github.com/IDEA-Research/DWPose),
[MMDetection](https://github.com/open-mmlab/mmdetection), and
[MMPose](https://github.com/open-mmlab/mmpose) for person detection and landmark estimation.
