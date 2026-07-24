"""Frozen whole-image OpenCLIP color selector for PACIT revision 3.

The public scoring call accepts only canonical RGB tensors. Pose, slot names,
pose masks, blind-color statistics, and D0 features are absent from the call
graph by construction.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from asset_oracle_core import (
    ACTIVE_PROPOSALS_PER_IMAGE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    compute_centered_color_drop,
)
from prompt_spec import flattened_prompts


CLIP_MODEL_NAME = "ViT-L-14"
CLIP_SIZE = 224
LETTERBOX_WIDTH = 75
LETTERBOX_LEFT = 74
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class FrozenWholeImageColorSelector:
    """Standard whole-image encoder plus the frozen centered-color scorer."""

    def __init__(
        self,
        checkpoint,
        checkpoint_sha256,
        device,
        *,
        microbatch=4,
    ):
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if sha256_file(checkpoint) != str(checkpoint_sha256):
            raise RuntimeError("CLIP checkpoint SHA256 mismatch")
        if int(microbatch) <= 0:
            raise ValueError("CLIP microbatch must be positive")

        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=str(checkpoint)
        )
        model = model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        normalizers = [
            transform
            for transform in preprocess.transforms
            if hasattr(transform, "mean") and hasattr(transform, "std")
        ]
        if len(normalizers) != 1:
            raise RuntimeError("could not identify CLIP normalization")
        if any(
            abs(float(actual) - expected) > 1e-8
            for actual, expected in zip(normalizers[0].mean, CLIP_MEAN)
        ) or any(
            abs(float(actual) - expected) > 1e-8
            for actual, expected in zip(normalizers[0].std, CLIP_STD)
        ):
            raise RuntimeError("CLIP normalization mismatch")

        prompts, layout = flattened_prompts()
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        with torch.inference_mode():
            text = model.encode_text(tokenizer(prompts).to(device))
        text = F.normalize(text.float(), dim=-1)
        color_text = []
        for prompt_indices in layout:
            color_text.append(text[prompt_indices].mean(dim=0))
        self.color_text = F.normalize(
            torch.stack(color_text, dim=0), dim=-1
        )
        self.model = model
        self.device = torch.device(device)
        self.microbatch = int(microbatch)
        self.mean = torch.tensor(
            CLIP_MEAN, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            CLIP_STD, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)

    def _normalize_whole_rgb(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1:] != (
            3,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        ):
            raise ValueError("RGB must have shape [N,3,384,128]")
        if rgb.device != self.device:
            raise ValueError("RGB must be on the frozen CLIP device")
        resized = F.interpolate(
            rgb.float(),
            size=(CLIP_SIZE, LETTERBOX_WIDTH),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)
        canvas = self.mean.expand(
            len(rgb), 3, CLIP_SIZE, CLIP_SIZE
        ).clone()
        canvas[
            :,
            :,
            :,
            LETTERBOX_LEFT : LETTERBOX_LEFT + LETTERBOX_WIDTH,
        ] = resized
        return (canvas - self.mean) / self.std

    @torch.inference_mode()
    def _encode_whole_rgb(self, rgb):
        features = []
        for start in range(0, len(rgb), self.microbatch):
            stop = min(start + self.microbatch, len(rgb))
            normalized = self._normalize_whole_rgb(rgb[start:stop])
            encoded = self.model.encode_image(normalized)
            features.append(F.normalize(encoded.float(), dim=-1))
        output = torch.cat(features, dim=0)
        if not torch.isfinite(output).all():
            raise RuntimeError("nonfinite whole-image CLIP feature")
        return output

    @torch.inference_mode()
    def __call__(self, original_rgb, edited_rgb):
        if original_rgb.shape != (3, IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError("original_rgb must have shape [3,384,128]")
        if edited_rgb.shape != (
            ACTIVE_PROPOSALS_PER_IMAGE,
            3,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        ):
            raise ValueError("edited_rgb must have shape [7,3,384,128]")
        original = self._encode_whole_rgb(original_rgb.unsqueeze(0))[0]
        edited = self._encode_whole_rgb(edited_rgb)
        return compute_centered_color_drop(
            original, edited, self.color_text
        )
