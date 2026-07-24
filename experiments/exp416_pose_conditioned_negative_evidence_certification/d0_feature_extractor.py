#!/usr/bin/env python3
"""Strict read-only sealed-D0 feature extractor for the exp416 fuel audit."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


IMAGE_HEIGHT = 384
IMAGE_WIDTH = 128
SEALED_D0_CHECKPOINT_SHA256 = (
    "59017755d61370754aa2e852a487d8e242fcee8814685f77f5388ba3a430e069"
)
SEALED_D0_CONFIG_SHA256 = (
    "510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b"
)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_state(payload):
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError("unexpected D0 checkpoint payload")
    return payload


class ExplodingPose(dict):
    """Sentinel that makes any attempted external structural read fatal."""

    accesses = 0

    @classmethod
    def reset(cls):
        cls.accesses = 0

    def _fail(self, operation):
        type(self).accesses += 1
        raise RuntimeError("sealed D0 accessed external structure via " + operation)

    def __getitem__(self, key):
        del key
        return self._fail("getitem")

    def get(self, key, default=None):
        del key, default
        return self._fail("get")

    def __iter__(self):
        return self._fail("iter")

    def __contains__(self, key):
        del key
        return self._fail("contains")

    def keys(self):
        return self._fail("keys")

    def items(self):
        return self._fail("items")

    def values(self):
        return self._fail("values")


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


def _validate_rectangles(rectangles, batch_size):
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


def fractional_rectangle_masks(
    rectangles,
    batch_size,
    *,
    feature_hw,
    device,
):
    """Area-resample canonical binary rectangles to fractional fmap masks."""
    boxes = _validate_rectangles(rectangles, batch_size)
    masks = torch.zeros(
        len(boxes),
        boxes.shape[1],
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        dtype=torch.float32,
    )
    for row in range(len(boxes)):
        for slot in range(boxes.shape[1]):
            top, left, height, width = (
                int(value) for value in boxes[row, slot].tolist()
            )
            masks[
                row,
                slot,
                top : top + height,
                left : left + width,
            ] = 1.0
    masks = F.interpolate(
        masks,
        size=tuple(map(int, feature_hw)),
        mode="area",
    ).to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(masks).all()):
        raise RuntimeError("fractional rectangle mask is non-finite")
    if bool((masks < 0.0).any()) or bool((masks > 1.0).any()):
        raise RuntimeError("fractional rectangle mask leaves [0,1]")
    return masks


def pool_fractional_rectangles(feature_map, rectangles):
    """Pool and normalize one feature for every canonical rectangle."""
    if (
        not torch.is_tensor(feature_map)
        or feature_map.ndim != 4
        or feature_map.shape[0] <= 0
    ):
        raise ValueError("feature_map must have shape [N,C,H,W]")
    if not bool(torch.isfinite(feature_map).all()):
        raise ValueError("feature_map contains non-finite values")
    masks = fractional_rectangle_masks(
        rectangles,
        len(feature_map),
        feature_hw=feature_map.shape[-2:],
        device=feature_map.device,
    )
    mass = masks.flatten(2).sum(dim=-1)
    valid = mass > 0.0
    pooled = torch.einsum(
        "nchw,nshw->nsc", feature_map.float(), masks
    ) / mass.clamp_min(1e-12)[..., None]
    pooled = F.normalize(pooled.float(), dim=-1)
    pooled = torch.where(valid[..., None], pooled, torch.zeros_like(pooled))
    if not bool(torch.isfinite(pooled).all()):
        raise RuntimeError("pooled D0 slot feature is non-finite")
    if bool(valid.any()):
        norms = pooled.norm(dim=-1)[valid]
        if not bool(
            torch.allclose(
                norms,
                torch.ones_like(norms),
                atol=1e-5,
                rtol=1e-5,
            )
        ):
            raise RuntimeError("pooled D0 slot feature is not L2 normalized")
    return pooled, valid, masks


class SealedD0FeatureExtractor:
    """Frozen clean-D0 eval path returning global and final-map features."""

    def __init__(
        self,
        *,
        config_path,
        config_sha256,
        checkpoint_path,
        checkpoint_sha256,
        dataset,
        device,
        microbatch=8,
    ):
        configured_cfg = Path(config_path).expanduser()
        configured_ckpt = Path(checkpoint_path).expanduser()
        if not configured_cfg.is_absolute() or not configured_ckpt.is_absolute():
            raise ValueError("D0 config/checkpoint paths must be absolute")
        resolved_cfg = configured_cfg.resolve(strict=True)
        resolved_ckpt = configured_ckpt.resolve(strict=True)
        if resolved_cfg != configured_cfg or resolved_ckpt != configured_ckpt:
            raise RuntimeError("D0 config/checkpoint paths must be canonical")
        if str(config_sha256) != SEALED_D0_CONFIG_SHA256:
            raise RuntimeError("exp416 requires the sealed D0 config SHA")
        if str(checkpoint_sha256) != SEALED_D0_CHECKPOINT_SHA256:
            raise RuntimeError("exp416 requires the sealed D0 checkpoint SHA")
        if sha256_file(resolved_cfg) != SEALED_D0_CONFIG_SHA256:
            raise RuntimeError("D0 config SHA256 mismatch")
        if sha256_file(resolved_ckpt) != SEALED_D0_CHECKPOINT_SHA256:
            raise RuntimeError("D0 checkpoint SHA256 mismatch")
        if int(microbatch) <= 0:
            raise ValueError("D0 microbatch must be positive")

        from config import cfg
        from model import make_model

        local_cfg = cfg.clone()
        local_cfg.merge_from_file(str(resolved_cfg))
        local_cfg.defrost()
        local_cfg.MODEL.PRETRAIN_PATH = ""
        local_cfg.freeze()
        model = make_model(
            local_cfg,
            num_class=int(dataset.num_train_pids),
            camera_num=int(dataset.num_train_cams),
            view_num=int(dataset.num_train_vids),
            semantic_weight=local_cfg.MODEL.SEMANTIC_WEIGHT,
        )
        payload = torch.load(str(resolved_ckpt), map_location="cpu")
        incompatible = model.load_state_dict(
            _checkpoint_state(payload), strict=True
        )
        del payload
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("strict D0 state load failed")
        for name, value in model.state_dict().items():
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise RuntimeError("non-finite sealed D0 state: " + name)
        model = model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self.model = model
        self.device = torch.device(device)
        self.microbatch = int(microbatch)
        self.mean = torch.tensor(
            local_cfg.INPUT.PIXEL_MEAN,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            local_cfg.INPUT.PIXEL_STD,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self.config_path = resolved_cfg
        self.config_sha256 = SEALED_D0_CONFIG_SHA256
        self.checkpoint_path = resolved_ckpt
        self.checkpoint_sha256 = SEALED_D0_CHECKPOINT_SHA256

    def _assert_runtime(self):
        if self.model is None:
            raise RuntimeError("D0 extractor is closed")
        if self.model.training:
            raise RuntimeError("sealed D0 model left eval mode")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("sealed D0 parameter requires a gradient")

    @torch.inference_mode()
    def encode(self, rgb, rectangles=None):
        """Return FP32 L2 global, raw last fmap, and optional pooled slots."""
        self._assert_runtime()
        _validate_rgb(rgb)
        global_parts = []
        map_parts = []
        ExplodingPose.reset()
        sentinel = ExplodingPose()
        for start in range(0, len(rgb), self.microbatch):
            stop = min(start + self.microbatch, len(rgb))
            batch = rgb[start:stop].to(self.device, dtype=torch.float32)
            normalized = (batch - self.mean) / self.std
            output = self.model(
                normalized,
                pose_batch=sentinel,
                tapf_epoch=None,
            )
            if (
                not isinstance(output, tuple)
                or len(output) != 2
                or not torch.is_tensor(output[0])
                or output[0].ndim != 2
                or not isinstance(output[1], (tuple, list))
                or not output[1]
                or not torch.is_tensor(output[1][-1])
                or output[1][-1].ndim != 4
            ):
                raise RuntimeError("unexpected sealed D0 eval output")
            descriptor = F.normalize(output[0].float(), dim=-1)
            final_map = output[1][-1].float()
            if (
                descriptor.shape[0] != len(batch)
                or final_map.shape[0] != len(batch)
                or not bool(torch.isfinite(descriptor).all())
                or not bool(torch.isfinite(final_map).all())
            ):
                raise RuntimeError("invalid sealed D0 feature output")
            global_parts.append(descriptor)
            map_parts.append(final_map)
        if ExplodingPose.accesses != 0:
            raise RuntimeError("sealed D0 consumed external structure")
        if not global_parts:
            raise ValueError("cannot encode an empty RGB batch")
        global_features = torch.cat(global_parts, dim=0)
        last_feature_map = torch.cat(map_parts, dim=0)
        if global_features.shape[0] != len(rgb) or last_feature_map.shape[0] != len(
            rgb
        ):
            raise RuntimeError("sealed D0 batch order/shape mismatch")
        global_norms = global_features.norm(dim=-1)
        if not bool(
            torch.allclose(
                global_norms,
                torch.ones_like(global_norms),
                atol=1e-5,
                rtol=1e-5,
            )
        ):
            raise RuntimeError("sealed D0 global descriptor is not L2 normalized")

        slot_features = None
        slot_valid = None
        fractional_masks = None
        if rectangles is not None:
            slot_features, slot_valid, fractional_masks = (
                pool_fractional_rectangles(last_feature_map, rectangles)
            )
        return {
            "global_features": global_features,
            "last_feature_map": last_feature_map,
            "slot_features": slot_features,
            "slot_valid": slot_valid,
            "fractional_masks": fractional_masks,
        }

    def close(self):
        if self.model is not None:
            self.model = None
        if sha256_file(self.config_path) != self.config_sha256:
            raise RuntimeError("D0 config changed during use")
        if sha256_file(self.checkpoint_path) != self.checkpoint_sha256:
            raise RuntimeError("D0 checkpoint changed during use")


class _MockD0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter(
            "sentinel", torch.nn.Parameter(torch.ones(()), requires_grad=False)
        )

    def forward(self, value, *, pose_batch, tapf_epoch):
        del pose_batch, tapf_epoch
        final_map = F.interpolate(
            value.float(), size=(12, 4), mode="area"
        )
        descriptor = torch.cat(
            (
                value.mean(dim=(2, 3)),
                value.square().mean(dim=(2, 3)),
            ),
            dim=1,
        )
        return descriptor, [final_map]


def _mock_extractor():
    extractor = object.__new__(SealedD0FeatureExtractor)
    extractor.model = _MockD0().eval()
    extractor.device = torch.device("cpu")
    extractor.microbatch = 1
    extractor.mean = torch.zeros(1, 3, 1, 1)
    extractor.std = torch.ones(1, 3, 1, 1)
    extractor.config_path = None
    extractor.config_sha256 = ""
    extractor.checkpoint_path = None
    extractor.checkpoint_sha256 = ""
    return extractor


def run_self_test():
    rgb = torch.zeros(2, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    rgb[0, 0] = 1.0
    rgb[0, 1, :192] = 0.5
    rgb[1, 1] = 1.0
    rgb[1, 2, 192:] = 0.75
    rectangles = torch.tensor(
        [
            [[0, 0, 192, 64], [192, 64, 192, 64]],
            [[0, 64, 192, 64], [192, 0, 192, 64]],
        ],
        dtype=torch.int64,
    )
    extractor = _mock_extractor()
    before = {
        name: value.detach().clone()
        for name, value in extractor.model.state_dict().items()
    }
    output = extractor.encode(rgb, rectangles)
    assert output["global_features"].shape == (2, 6)
    assert output["last_feature_map"].shape == (2, 3, 12, 4)
    assert output["slot_features"].shape == (2, 2, 3)
    assert output["slot_valid"].shape == (2, 2)
    assert output["fractional_masks"].shape == (2, 2, 12, 4)
    assert output["global_features"].dtype == torch.float32
    assert output["last_feature_map"].dtype == torch.float32
    assert torch.allclose(
        output["global_features"].norm(dim=-1),
        torch.ones(2),
        atol=1e-5,
    )
    assert torch.allclose(
        output["slot_features"].norm(dim=-1),
        torch.ones(2, 2),
        atol=1e-5,
    )
    assert not torch.equal(
        output["global_features"][0], output["global_features"][1]
    )
    assert before.keys() == extractor.model.state_dict().keys()
    for name, expected in before.items():
        assert torch.equal(extractor.model.state_dict()[name], expected)
    assert not extractor.model.training
    assert not any(
        parameter.requires_grad for parameter in extractor.model.parameters()
    )
    sentinel = ExplodingPose()
    ExplodingPose.reset()
    try:
        sentinel.get("forbidden")
    except RuntimeError:
        pass
    else:
        raise AssertionError("external-structure sentinel did not explode")
    assert ExplodingPose.accesses == 1
    bad = rectangles.clone()
    bad[1, 1, 3] += 1
    try:
        pool_fractional_rectangles(
            output["last_feature_map"], bad
        )
    except ValueError:
        pass
    else:
        raise AssertionError("variable per-slot rectangle size was accepted")
    print("d0_feature_extractor self-test PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("only --self-test is supported by this helper")
    run_self_test()


if __name__ == "__main__":
    main()
