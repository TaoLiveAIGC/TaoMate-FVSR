# TaoMate-FVSR ComfyUI 节点

[English](README.md) | 简体中文 | [原始项目](https://github.com/TaoLiveAIGC/TaoMate-FVSR/tree/main)

通过 **CPU 自动裁脸 → GPEN 参考图 → TaoMate-FVSR 修复 → 带音轨贴回**，修复生成视频中的人脸。原始 BF16 权重直接复用，无需转换；第一阶段以 16 GB 显存为目标。全景输出保持输入画幅，不是整段视频放大。

**测试版。**请按下文从 `comfyui` 分支手动安装；不代表已注册 ComfyUI Manager。

## 环境要求

- 已能运行的 ComfyUI、CUDA PyTorch，以及支持 BF16 的 NVIDIA 显卡。已测试 Linux / Python 3.10 / PyTorch 2.9.0+cu128，Windows 安装尚未验收。
- Git、FFmpeg、FFprobe 已加入 PATH。
- 十个模型及配置文件约 **14.89 GB**，另需预留逐帧缓存和输出空间。
- **建议 32 GB 系统内存**；最长实测视频的主进程内存峰值约 17.75 GiB。
- 首先覆盖 16 GB 显存预算。下文数据来自 H20 上的限额测试，**不等同于消费级 16 GB 显卡实测**；8/12 GB 暂未验证。

这套流程不需要 H3、VOSR2、DWPose 或 MMCV。

## 安装节点

在 `ComfyUI/custom_nodes` 目录执行：

```bash
git clone --depth 1 --single-branch --branch comfyui https://github.com/TaoLiveAIGC/TaoMate-FVSR.git ComfyUI-TaoMate-FVSR
cd ComfyUI-TaoMate-FVSR
python -m pip install -r requirements.txt
```

必须使用 **ComfyUI 自己的 Python 环境**，不要误装到其他系统 Python。依赖会安装固定版本的原始 `livevsr` 包和 Ultralytics；保留 ComfyUI 已正常工作的 CUDA PyTorch。

Windows 便携版：在 `ComfyUI_windows_portable` 根目录打开命令提示符：

```bat
git clone --depth 1 --single-branch --branch comfyui https://github.com/TaoLiveAIGC/TaoMate-FVSR.git ComfyUI\custom_nodes\ComfyUI-TaoMate-FVSR
.\python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-TaoMate-FVSR\requirements.txt
cd ComfyUI\custom_nodes\ComfyUI-TaoMate-FVSR
..\..\..\python_embeded\python.exe scripts\download_models.py --dest ..\..\models\TaoMate-FVSR --gpen-source-dir ..\..\models\TaoMate-GPEN-source
```

以上是按便携版目录结构给出的命令，不代表已经完成 Windows 验收。Windows 也需要 Git、FFmpeg 和 FFprobe。

## 复用原有权重

**没有专用的 ComfyUI 主权重。**TaoMate-FVSR 主权重、VAE、HuBERT 和 GPEN 都沿用原文件；自动裁脸另需约 6.23 MB 的 YOLO 人脸检测文件。无需量化或转换主权重。

在节点目录中，用 ComfyUI 的 Python 执行：

```bash
python scripts/download_models.py --dest ../../models/TaoMate-FVSR --plan
python scripts/download_models.py --dest ../../models/TaoMate-FVSR --gpen-source-dir ../../models/TaoMate-GPEN-source
python scripts/download_models.py --dest ../../models/TaoMate-FVSR --check --gpen-source-dir ../../models/TaoMate-GPEN-source
```

Windows 便携版将这里的 `python` 换成 `..\..\..\python_embeded\python.exe`。`--plan` 只列清单，不联网；导入节点或打开工作流也不会自动下载。下载工具检查固定文件大小和 SHA256，已有正确文件会跳过，未完成下载支持续传。发现已有文件校验失败会报错，不覆盖原文件。

下载命令可追加：

| 参数 | 用途 |
| --- | --- |
| `--endpoint https://hf-mirror.com` | 使用 HF 镜像，revision/hash 不变；GPEN 的 OSS 和 GitHub 地址不变。 |
| `--reuse /path/to/existing/models` | 核验已有文件后建立软链接；可重复指定多个目录，包括 HuBERT 目录。不扫描磁盘，需要软链接权限。 |
| `--check` | 仅检查，不下载、不修复。 |
| `--only checkpoint_config` | 按资产 ID 选择清单中的单项。 |

准备后的目录：

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
└── TaoMate-GPEN-source/       # GPEN 源码，与权重分开
```

**Load Model** 选择 `(root)`；GPEN 节点的 `gpen_repo` 填 **TaoMate-GPEN-source 的绝对路径**，不要填权重目录。已有完整模型目录也可以直接填入 `local_directory`，或通过 ComfyUI 的 `extra_model_paths.yaml` 中 `taomate` / `TaoMate-FVSR` 键配置。

手动下载时，查阅 [assets.json](assets.json) 中各文件的官方地址、固定 revision、保存路径、大小及 SHA256。GPEN 源码固定为 `2c736702983368847fb544d234a22ac7cff25802`。详细规则见 [模型准备](docs/model-setup.md)。

## 按 16 GB 预算启动

在原有 ComfyUI 启动命令中加上 `--disable-cuda-malloc`。例如在 ComfyUI 根目录执行：

```bash
python main.py --disable-cuda-malloc
```

Windows 便携版对应命令，在便携版根目录执行：

```bat
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --disable-cuda-malloc
```

这会使用 PyTorch 原生分配器，使默认 **15 GiB 分配预算**能够生效，为 CUDA 等开销留空间。使用 `cudaMallocAsync` 时仍会卸载组件，但不宣称硬性预算生效。预算不能保证所有情况下都不溢出；不要在同一张卡上并行运行其他生成任务。

安装后重启 ComfyUI，五个节点位于 `TaoMate FVSR` 分类。

## 跑第一个视频

将 [yolo-gpen-workflow.json](examples/yolo-gpen-workflow.json) 拖入画布。工作流包含 **Load Model → Prepare Video → GPEN Reference → Restore Face Video → Save / Paste Back Video** 的模型与视频连接，以及裁脸、参考图预览和输出路径预览。

1. `video_path` 填 **ComfyUI 服务所在机器能访问**的视频路径。选择 `yolo_face`，首次保留 `crop_scale=0.8`、`max_frames=77`。填写模型目录和 GPEN 源码路径；YOLO 已在模型目录中时，`detector_path` 可留空。
2. **先看裁脸：**将 **Save / Paste Back Video** 及其下游 **Preview as Text / PreviewAny** 输出节点同时设为 Mute，再排队。裁脸和 GPEN 图像预览仍会运行；若只想检查裁脸，可单独连接 Prepare Video → Preview Image。确认人脸位置正确、参考图构图与裁脸一致。
3. 取消两个输出节点的 Mute，再运行完整流程。首次使用默认 10 步。MP4 保存到 ComfyUI 的 `output` 目录，画布上会显示实际文件路径。
4. 短片效果合适后，将 `max_frames=0` 处理整片。目前完整实测最长 **14.88 秒**，不代表已验证无限长度。

`audio_path` 留空保留原音轨；指定文件则替换音轨，并用于音频条件。输出编码为 AAC，保留来源的声道及采样率；音频不足补静音。视频统一为 25 fps。

已有裁好的人脸视频，可用 [cropped-face-workflow.json](examples/cropped-face-workflow.json)，选择 `cropped_face` 并提供**与首帧构图一致的单张 512×512 参考图**，输出为 512×512。对位置固定的人脸，也可用 `manual_roi` 指定固定方框 `roi_x/y/side`，结果贴回原画幅。

API 工作流：[YOLO + GPEN](examples/yolo-gpen-api.json)、[已裁脸输入](examples/cropped-face-api.json)。替换其中的路径，以 `{"prompt": <工作流内容>}` 提交到 ComfyUI 的 `/prompt`。

## 参数及适用范围

| 参数 | 含义 |
| --- | --- |
| `crop_scale=0.8` | YOLO 扩展裁框的边长系数，预览后调整；不是输出放大倍数。 |
| `max_frames=77 / 0` | 短片试跑 / 整片。 |
| `budget_gib=15` | 原生分配器下的显存分配预算。 |
| `steps=10` | 已验证的默认修复步数。 |
| `audio_path` | 留空用原音，或填本地替换音频。 |
| `gpen_repo` | GPEN 源码目录，不是权重目录。 |

首版面向 **单人、单镜头、512 人脸推理**。检测到疑似切镜会报错，请先按镜头拆段。缺检帧保留原时间索引，最终贴回时这些帧保持原图。跟踪基于几何关联，不是身份识别；极小脸、转头及多人交叉仍需检查。

`crop_scale` 不同于原模型的 `scale2`：后者以 1024 人脸推理，再回到 512 贴回。本节点不提供 `scale2`，不会把全景宽高放大两倍。

参考图协议应分别说明：

- **默认 GPEN：**CPU GPEN 增强首帧裁脸，是默认工作流。
- **增强参考图（+）：**用户提供构图匹配的增强图。
- **论文 GT 参考图：**研究评测协议，不是任意用户视频都能取得的参考图。

逐帧和输出使用磁盘缓存，不将整段视频作为 Comfy IMAGE 张量传递。但磁盘和系统内存需求仍随片长增长；清理 ComfyUI 临时目录前请保存结果。更多行为见 [节点使用](docs/node-usage.md)。

## 已验证情况

环境为 Linux / NVIDIA H20 / 原始 BF16 / 10 步：

| 项目 | 结果 |
| --- | --- |
| YOLO + 默认 GPEN + 修复 + 原生贴回 + 音轨 | 真实 ComfyUI 执行通过 |
| 连续 14.88 秒输入 | 372 帧、7 个窗口，全片解码通过 |
| 输出 | 1344×768、25 fps；音频和视频均为 14.88 秒 |
| GPU 进程显存峰值 | 15,334 MiB（14.975 GiB） |
| 主进程系统内存峰值 | 约 17.75 GiB |
| 推理中取消，再次运行 | 通过 |
| 更长视频 / 消费级 16 GB 显卡 / Windows | 尚未验证 |

14.88 秒测试耗时 **745 秒，包含较慢的冷加载和读盘**，不能作为稳定推理吞吐。此前 72/148 帧及多窗口运行也已通过。浏览器实际操作与干净机器安装仍待验收；后台执行和工作流结构验证不等于完成这两项测试。

## 常见问题

- **节点不显示或导入失败：**确认依赖装在启动 ComfyUI 的 Python 中，查看启动日志并重启。
- **提示缺少模型：**按上方目录检查并运行 `--check`。`gpen_repo` 指向源码，`local_directory` 指向权重。
- **脸太小或裁剪不对：**先预览并调整 `crop_scale`；固定位置的人脸可试 `manual_roi`。调整裁框不能保证恢复原本不存在的细节。
- **提示切镜：**按镜头拆段，避免同一参考图和时序历史跨越不相关镜头。
- **第一次运行很慢：**模型加载、CPU GPEN、逐帧缓存都有开销；冷启动时间不是速度承诺。
- **显存不足：**检查启动参数和其他 GPU 任务，先试短片。减少步数不能免除模型权重的显存需求，较低显存卡尚未验收。
- **取消没有立即结束：**去噪期间支持取消；CPU GPEN 和模型加载不提供逐层即时取消。

## 来源与许可

使用了 [TaoMate-FVSR](https://github.com/TaoLiveAIGC/TaoMate-FVSR/tree/main)、[GPEN](https://github.com/yangxy/GPEN)、[Ultralytics](https://github.com/ultralytics/ultralytics) 及 [assets.json](assets.json) 列出的模型来源，各自代码和权重许可继续适用。原始 `livevsr` 固定为 `bc924d8572c4e5050cdc8a86e7fab13d30a327db`；主权重 HF revision 为 `a37c97010a2af15d561f2515481134a8015cc300`。
