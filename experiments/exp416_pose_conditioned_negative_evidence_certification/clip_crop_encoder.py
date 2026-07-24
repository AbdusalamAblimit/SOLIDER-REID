#!/usr/bin/env python3
"""Sealed OpenCLIP image encoder for the exp416 fuel audit.

The public runtime accepts canonical RGB tensors and fixed rectangles only.
It has no language branch, intervention selector, or structural-input API.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


IMAGE_HEIGHT = 384
IMAGE_WIDTH = 128
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_OUTPUT_DIM = 768
CLIP_SIZE = 224
WHOLE_LETTERBOX_WIDTH = 75
WHOLE_LETTERBOX_LEFT = 74
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SEALED_CLIP_SHA256 = (
    "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e"
)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_rgb(rgb):
    if not torch.is_tensor(rgb):
        raise TypeError("RGB input must be a tensor")
    if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (
        3,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    ):
        raise ValueError("RGB must have shape [N,3,384,128]")
    if not rgb.is_floating_point():
        raise ValueError("RGB must be floating point in [0,1]")
    if not bool(torch.isfinite(rgb).all()):
        raise ValueError("RGB contains non-finite values")
    if rgb.numel() and (
        float(rgb.detach().min()) < 0.0
        or float(rgb.detach().max()) > 1.0
    ):
        raise ValueError("RGB must lie in [0,1]")


def _validate_fixed_rectangles(rectangles, batch_size):
    boxes = torch.as_tensor(rectangles)
    if boxes.ndim != 3 or boxes.shape[0] != int(batch_size) or boxes.shape[2] != 4:
        raise ValueError("rectangles must have shape [N,S,4]")
    if boxes.is_floating_point():
        if not bool(torch.isfinite(boxes).all()):
            raise ValueError("rectangles contain non-finite values")
        rounded = boxes.round()
        if not bool(torch.equal(boxes, rounded)):
            raise ValueError("rectangles must contain integer coordinates")
        boxes = rounded
    boxes = boxes.to(dtype=torch.int64, device="cpu")
    if boxes.shape[1] <= 0:
        raise ValueError("at least one rectangle slot is required")
    top, left, height, width = boxes.unbind(dim=-1)
    if bool((height <= 0).any()) or bool((width <= 0).any()):
        raise ValueError("rectangle dimensions must be positive")
    if (
        bool((top < 0).any())
        or bool((left < 0).any())
        or bool((top + height > IMAGE_HEIGHT).any())
        or bool((left + width > IMAGE_WIDTH).any())
    ):
        raise ValueError("rectangle leaves the canonical RGB canvas")
    dimensions = boxes[:, :, 2:4]
    if not bool((dimensions == dimensions[:1]).all()):
        raise ValueError(
            "each slot must keep one fixed height/width across the batch"
        )
    return boxes


def _rectangle_crop_batch(rgb, rectangles):
    """Return row-major `[image, slot]` crops resized to the CLIP square."""
    _validate_rgb(rgb)
    boxes = _validate_fixed_rectangles(rectangles, len(rgb))
    crops = []
    for row in range(len(rgb)):
        for slot in range(boxes.shape[1]):
            top, left, height, width = (
                int(value) for value in boxes[row, slot].tolist()
            )
            crop = rgb[
                row : row + 1,
                :,
                top : top + height,
                left : left + width,
            ].float()
            crop = F.interpolate(
                crop,
                size=(CLIP_SIZE, CLIP_SIZE),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp(0.0, 1.0)
            crops.append(crop[0])
    output = torch.stack(crops, dim=0)
    expected = (len(rgb) * boxes.shape[1], 3, CLIP_SIZE, CLIP_SIZE)
    if tuple(output.shape) != expected or not bool(torch.isfinite(output).all()):
        raise RuntimeError("rectangle crop batch contract failed")
    return output, int(boxes.shape[1])


def _whole_image_letterbox(rgb, mean):
    """Use the sealed full-person letterbox without dropping head or feet."""
    _validate_rgb(rgb)
    resized = F.interpolate(
        rgb.float(),
        size=(CLIP_SIZE, WHOLE_LETTERBOX_WIDTH),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0.0, 1.0)
    canvas = mean.expand(len(rgb), 3, CLIP_SIZE, CLIP_SIZE).clone()
    canvas[
        :,
        :,
        :,
        WHOLE_LETTERBOX_LEFT : WHOLE_LETTERBOX_LEFT + WHOLE_LETTERBOX_WIDTH,
    ] = resized
    return canvas


class FrozenClipCropEncoder:
    """Strict frozen ViT-L/14 encoder for local crops and full-person images."""

    def __init__(
        self,
        checkpoint,
        checkpoint_sha256,
        device,
        *,
        microbatch=8,
    ):
        configured = Path(checkpoint).expanduser()
        if not configured.is_absolute():
            raise ValueError("CLIP checkpoint path must be absolute")
        resolved = configured.resolve(strict=True)
        if resolved != configured:
            raise RuntimeError("CLIP checkpoint path must be canonical")
        if str(checkpoint_sha256) != SEALED_CLIP_SHA256:
            raise RuntimeError("exp416 requires the sealed CLIP checkpoint SHA")
        if sha256_file(resolved) != SEALED_CLIP_SHA256:
            raise RuntimeError("CLIP checkpoint SHA256 mismatch")
        if int(microbatch) <= 0:
            raise ValueError("CLIP microbatch must be positive")

        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=str(resolved)
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
            raise RuntimeError("could not identify OpenCLIP normalization")
        if any(
            abs(float(actual) - expected) > 1e-8
            for actual, expected in zip(normalizers[0].mean, CLIP_MEAN)
        ) or any(
            abs(float(actual) - expected) > 1e-8
            for actual, expected in zip(normalizers[0].std, CLIP_STD)
        ):
            raise RuntimeError("OpenCLIP normalization mismatch")
        if getattr(model.visual, "output_dim", None) != CLIP_OUTPUT_DIM:
            raise RuntimeError("expected ViT-L/14 output dimension 768")

        self.model = model
        self.device = torch.device(device)
        self.microbatch = int(microbatch)
        self.mean = torch.tensor(
            CLIP_MEAN, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            CLIP_STD, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.checkpoint_path = resolved
        self.checkpoint_sha256 = SEALED_CLIP_SHA256

    def _assert_runtime(self):
        if self.model is None:
            raise RuntimeError("CLIP encoder is closed")
        if self.model.training:
            raise RuntimeError("frozen CLIP model left eval mode")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("frozen CLIP parameter requires a gradient")

    @torch.inference_mode()
    def _encode_square_batch(self, square_rgb):
        self._assert_runtime()
        if square_rgb.ndim != 4 or tuple(square_rgb.shape[1:]) != (
            3,
            CLIP_SIZE,
            CLIP_SIZE,
        ):
            raise ValueError("square RGB must have shape [N,3,224,224]")
        outputs = []
        for start in range(0, len(square_rgb), self.microbatch):
            stop = min(start + self.microbatch, len(square_rgb))
            batch = square_rgb[start:stop].to(
                self.device, dtype=torch.float32
            )
            normalized = (batch - self.mean) / self.std
            encoded = self.model.encode_image(normalized)
            encoded = F.normalize(encoded.float(), dim=-1)
            outputs.append(encoded)
        if not outputs:
            raise ValueError("cannot encode an empty RGB batch")
        output = torch.cat(outputs, dim=0)
        if output.ndim != 2 or output.shape[0] != len(square_rgb):
            raise RuntimeError("OpenCLIP output order/shape mismatch")
        if output.shape[1] != CLIP_OUTPUT_DIM:
            raise RuntimeError("unexpected OpenCLIP feature dimension")
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("non-finite OpenCLIP feature")
        norms = output.norm(dim=-1)
        if not bool(
            torch.allclose(
                norms,
                torch.ones_like(norms),
                atol=1e-5,
                rtol=1e-5,
            )
        ):
            raise RuntimeError("OpenCLIP features are not L2 normalized")
        return output

    @torch.inference_mode()
    def encode_rectangles(self, rgb, rectangles):
        """Encode `[N,S,4]` fixed rectangles and return `[N,S,768]` FP32."""
        if rgb.device.type != "cpu":
            rgb = rgb.detach().cpu()
        crops, slot_count = _rectangle_crop_batch(rgb, rectangles)
        encoded = self._encode_square_batch(crops)
        output = encoded.reshape(len(rgb), slot_count, CLIP_OUTPUT_DIM)
        if output.dtype != torch.float32:
            raise RuntimeError("rectangle OpenCLIP output must be FP32")
        return output

    @torch.inference_mode()
    def encode_whole_images(self, rgb):
        """Encode canonical full-person images in input order as `[N,768]`."""
        _validate_rgb(rgb)
        if rgb.device.type != "cpu":
            rgb = rgb.detach().cpu()
        mean = self.mean.detach().cpu()
        square = _whole_image_letterbox(rgb, mean)
        output = self._encode_square_batch(square)
        if output.dtype != torch.float32:
            raise RuntimeError("whole-image OpenCLIP output must be FP32")
        return output

    def close(self):
        if self.model is not None:
            self.model = None
        if sha256_file(self.checkpoint_path) != self.checkpoint_sha256:
            raise RuntimeError("CLIP checkpoint changed during use")


class _MockClip(torch.nn.Module):
    output_dim = CLIP_OUTPUT_DIM

    def __init__(self):
        super().__init__()
        self.register_parameter(
            "sentinel", torch.nn.Parameter(torch.ones(()), requires_grad=False)
        )

    def encode_image(self, value):
        means = value.float().mean(dim=(2, 3))
        repeated = means.repeat(1, CLIP_OUTPUT_DIM // means.shape[1] + 1)
        return repeated[:, :CLIP_OUTPUT_DIM]


def _mock_encoder():
    encoder = object.__new__(FrozenClipCropEncoder)
    encoder.model = _MockClip().eval()
    encoder.device = torch.device("cpu")
    encoder.microbatch = 2
    encoder.mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(
        1, 3, 1, 1
    )
    encoder.std = torch.tensor(CLIP_STD, dtype=torch.float32).view(
        1, 3, 1, 1
    )
    encoder.checkpoint_path = None
    encoder.checkpoint_sha256 = ""
    return encoder


def run_self_test():
    rgb = torch.zeros(2, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    rgb[0, 0, :192] = 1.0
    rgb[0, 1, 192:] = 1.0
    rgb[1, 2, :192] = 1.0
    rgb[1, :, 192:] = torch.tensor((0.3, 0.6, 0.9)).view(3, 1, 1)
    rectangles = torch.tensor(
        [
            [[0, 0, 192, 64], [192, 64, 192, 64]],
            [[0, 64, 192, 64], [192, 0, 192, 64]],
        ],
        dtype=torch.int64,
    )
    encoder = _mock_encoder()
    local = encoder.encode_rectangles(rgb, rectangles)
    whole = encoder.encode_whole_images(rgb)
    assert local.shape == (2, 2, CLIP_OUTPUT_DIM)
    assert whole.shape == (2, CLIP_OUTPUT_DIM)
    assert local.dtype == whole.dtype == torch.float32
    assert torch.allclose(local.norm(dim=-1), torch.ones(2, 2), atol=1e-5)
    assert torch.allclose(whole.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert not torch.equal(local[0, 0], local[0, 1])
    assert not torch.equal(local[0, 0], local[1, 0])
    assert not encoder.model.training
    assert not any(
        parameter.requires_grad for parameter in encoder.model.parameters()
    )
    bad = rectangles.clone()
    bad[1, 0, 2] -= 1
    try:
        encoder.encode_rectangles(rgb, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("variable per-slot rectangle size was accepted")
    print("clip_crop_encoder self-test PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("only --self-test is supported by this helper")
    run_self_test()


if __name__ == "__main__":
    main()
