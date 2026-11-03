# Third-party components

The Wan transformer/VAE components are derived from
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and the
[Wan models](https://github.com/Wan-Video/Wan2.2), under Apache-2.0.
The Apache-2.0 license is retained in `LICENSE`. This distribution extracts the
needed inference definitions, selects dense attention, adds audio/reference
conditioning, and provides standalone model loading for inference.
The [X-Dub](https://github.com/KlingAIResearch/X-Dub)-compatible audio
architecture and feature alignment are retained for compatibility with the
Taomatefvsr checkpoint, hosted separately at
[TaoLiveAIGC/TaoMate-FVSR](https://huggingface.co/TaoLiveAIGC/TaoMate-FVSR).

[GPEN](https://github.com/yangxy/GPEN) is installed separately, not vendored.
The reference adapter invokes its `FaceGAN`, RetinaFace detector, five-point
alignment and ParseNet mask for local reference preparation. It uses the
upstream mask postprocessing recipe without loading the background-SR model.
GPEN-BFR-512 is used for 1× references; GPEN-BFR-1024 for 2× references comes
from the author's [ModelScope high-resolution release](https://www.modelscope.cn/models/damo/cv_gpen_image-portrait-enhancement-hires/summary).
GPEN and its released weights remain subject to their own upstream terms;
this project's license does not grant rights to redistribute those weights.

[HuBERT](https://huggingface.co/facebook/hubert-large-ll60k) uses the Hugging Face
Transformers implementation and the selected model's terms.
Third-party weights download from their original publishers, not a project
mirror. Model-weight licenses are separate from source-code licenses. The
Taomatefvsr model card and weight license should accompany its release.

The scene pipeline uses YOLOX-L through MMDetection and RTMW-X through MMPose,
with the DWPose landmark ordering. These dependencies, configs and weights are
obtained from their upstream publishers, not redistributed here. The crop
helper also accepts externally extracted DWPose tracks. The compositing helper
uses scikit-image histogram matching and the bundled static paste mask. No
private blending SDK code is included or required.
