"""HuBERT feature extraction and alignment for 77-frame inference windows."""

from __future__ import annotations

import subprocess

import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor


def _decode_audio_16k_mono(path: str) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            path,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return (
        np.frombuffer(result.stdout, dtype=np.int16)
        .astype(np.float32)
        .reshape(-1)
        / 32768.0
    )


class HubertAudioProcessor:
    """Produce the ``[B, 20, 8, 4096]`` audio condition for video inference.

    HuBERT-large hidden layers 9--12 are concatenated.  A 77-frame clip is
    aligned to Wan VAE38's 20 temporal tokens with causal audio windows.
    HuBERT runs as a separate, fixed feature extractor and is loaded
    independently of the DiT checkpoint.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: torch.device | str,
        num_frames: int = 77,
        layer_indices=(9, 10, 11, 12),
    ):
        self.device = torch.device(device)
        self.num_frames = int(num_frames)
        if self.num_frames != 77:
            raise ValueError("Audio conditioning requires 77 frames")
        self.token_frames = (self.num_frames - 1) // 4 + 1
        self.layer_indices = tuple(int(index) for index in layer_indices)
        if self.layer_indices != (9, 10, 11, 12):
            raise ValueError("HuBERT layers must be exactly 9,10,11,12")

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.model = HubertModel.from_pretrained(
            model_path,
            local_files_only=True,
        ).to(device=self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.feature_extractor._freeze_parameters()

    def to(self, device: torch.device | str):
        device = torch.device(device)
        if device != self.device:
            self.model.to(device=device)
            self.device = device
        return self

    @torch.no_grad()
    def extract_full_features(self, audio_path: str) -> torch.Tensor:
        audio = _decode_audio_16k_mono(audio_path)
        values = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_values.to(device=self.device, dtype=self.model.dtype)
        hidden_states = self.model(
            values,
            output_hidden_states=True,
        ).hidden_states
        selected = torch.stack(
            [hidden_states[index][0] for index in self.layer_indices],
            dim=1,
        )
        return selected.float()

    @staticmethod
    def _load_feature_file(path: str) -> torch.Tensor:
        feature = torch.from_numpy(np.load(path).astype(np.float32))
        if feature.ndim == 4 and feature.shape[0] == 1:
            feature = feature[0]
        return feature

    @staticmethod
    def _slice_window(
        feature: torch.Tensor,
        video_frame_index: int,
        *,
        first_token: bool,
    ) -> torch.Tensor:
        if feature.ndim != 3 or tuple(feature.shape[1:]) != (4, 1024):
            raise ValueError(
                "raw HuBERT feature must have shape [F,4,1024], got "
                f"{tuple(feature.shape)}"
            )
        flattened = feature.reshape(feature.shape[0], 4096)
        center = int(video_frame_index) * 2
        left = center - 6 if first_token else center
        right = center + 2 if first_token else center + 8
        indices = torch.arange(left, right, device=flattened.device)
        indices = indices.clamp_(0, flattened.shape[0] - 1)
        return flattened.index_select(0, indices)

    def crop_clip_features(
        self,
        feature: torch.Tensor,
        frame_start: int,
    ) -> torch.Tensor:
        if (
            feature.ndim == 3
            and tuple(feature.shape) == (self.token_frames, 8, 4096)
        ):
            return feature
        windows = []
        for token_index in range(self.token_frames):
            if token_index == 0:
                video_index = int(frame_start)
                first_token = True
            else:
                video_index = int(frame_start) + (token_index - 1) * 4 + 1
                first_token = False
            windows.append(
                self._slice_window(
                    feature,
                    video_index,
                    first_token=first_token,
                )
            )
        return torch.stack(windows, dim=0)

    @torch.no_grad()
    def build_condition(
        self,
        *,
        audio_path: str | None,
        audio_feature_path: str | None,
        audio_valid: bool,
        frame_start: int,
        output_device: torch.device | str,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if not audio_valid:
            return torch.zeros(
                (1, self.token_frames, 8, 4096),
                device=output_device,
                dtype=output_dtype,
            )
        if audio_feature_path:
            full_feature = self._load_feature_file(audio_feature_path).to(
                device=self.device
            )
        else:
            if not audio_path:
                raise ValueError("audio_valid sample requires audio_path")
            full_feature = self.extract_full_features(audio_path)
        clip_feature = self.crop_clip_features(full_feature, int(frame_start))
        if not torch.isfinite(clip_feature).all():
            raise ValueError("HuBERT audio feature contains NaN or Inf")
        return clip_feature.unsqueeze(0).to(
            device=output_device,
            dtype=output_dtype,
        )
