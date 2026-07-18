#!/usr/bin/env python3
"""CPU/static contracts for the exp392 Phase 0B2 teacher interface.

This script deliberately does not load CLIP weights or any ReID checkpoint.
It checks only the geometry, branch-order and additive-attention contracts that
must hold before an OpenCLIP-specific implementation is allowed.
"""

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


REGIONS = 5
HEADS = 16
LEAK_RATIO = 0.01


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_partition(raw):
    """Normalize nonnegative raw supports without assigning empty pixels."""
    if raw.ndim != 4 or raw.shape[1] != REGIONS:
        raise ValueError("raw support must have shape [B,5,H,W]")
    if bool((raw < 0).any()):
        raise ValueError("raw support must be nonnegative")
    total = raw.sum(dim=1, keepdim=True)
    return torch.where(total > 0, raw / total.clamp_min(1e-12), raw)


def regional_log_prior(masks, leak_ratio=LEAK_RATIO):
    """Create a bounded-total-leak log prior for CLS-to-patch logits.

    masks is [B,R,P].  Invalid all-zero slots return a zero prior plus a false
    validity bit; callers must skip their CLS readout instead of using it.
    """
    if masks.ndim != 3 or masks.shape[1] != REGIONS:
        raise ValueError("masks must have shape [B,5,P]")
    if bool((masks < 0).any()):
        raise ValueError("masks must be nonnegative")
    patch_count = masks.shape[-1]
    mass = masks.sum(dim=-1, keepdim=True)
    maximum = masks.amax(dim=-1, keepdim=True)
    valid = maximum.squeeze(-1) > 0
    delta = float(leak_ratio) * mass / float(patch_count)
    prior = torch.log(
        (masks + delta).clamp_min(torch.finfo(masks.dtype).tiny)
        / (maximum + delta).clamp_min(torch.finfo(masks.dtype).tiny)
    )
    prior = torch.where(valid[..., None], prior, torch.zeros_like(prior))
    return prior, valid


def make_cls_patch_attn_mask(prior, heads=HEADS):
    """Return PyTorch MHA additive mask [B*R*heads,L,L]."""
    if prior.ndim != 3 or prior.shape[1] != REGIONS:
        raise ValueError("prior must have shape [B,5,P]")
    batch, regions, patches = prior.shape
    sequence = patches + 1
    mask = torch.zeros(
        batch * regions,
        sequence,
        sequence,
        dtype=prior.dtype,
        device=prior.device,
    )
    mask[:, 0, 1:] = prior.reshape(batch * regions, patches)
    return mask.repeat_interleave(int(heads), dim=0)


def expand_region_branches(tokens, regions=REGIONS):
    """Copy one official sequence into independent region branches."""
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B,L,D]")
    batch, sequence, width = tokens.shape
    return (
        tokens[:, None]
        .expand(batch, int(regions), sequence, width)
        .reshape(batch * int(regions), sequence, width)
        .clone()
    )


def transform_marker_and_pose(
    original_height=192,
    original_width=64,
    point_xy=(20, 50),
    flipped=True,
    crop_top=13,
    crop_left=7,
):
    """Exercise resize->flip->pad->crop for both RGB marker and pose point."""
    if (original_width, original_height) != (64, 192):
        raise ValueError("synthetic pose image size mismatch")
    x, y = map(float, point_xy)
    image = torch.zeros(1, 1, original_height, original_width)
    image[0, 0, int(y), int(x)] = 1.0
    image = F.interpolate(image, size=(384, 128), mode="nearest")
    x *= 128.0 / float(original_width)
    y *= 384.0 / float(original_height)
    if flipped:
        image = image.flip(-1)
        x = 127.0 - x
    image = F.pad(image, (10, 10, 10, 10))
    x += 10.0
    y += 10.0
    image = image[:, :, crop_top:crop_top + 384, crop_left:crop_left + 128]
    x -= float(crop_left)
    y -= float(crop_top)
    coordinates = image[0, 0].nonzero(as_tuple=False).float()
    marker_yx = coordinates.mean(dim=0)
    marker_xy = torch.stack((marker_yx[1], marker_yx[0]))
    pose_xy = torch.tensor((x, y), dtype=torch.float32)
    grid_pose_xy = torch.tensor(
        (x * 31.0 / 127.0, y * 95.0 / 383.0), dtype=torch.float32
    )
    pooled = F.avg_pool2d(image, kernel_size=4, stride=4)
    pooled_yx = torch.nonzero(
        pooled[0, 0] == pooled[0, 0].max(), as_tuple=False
    ).float().mean(dim=0)
    pooled_xy = torch.stack((pooled_yx[1], pooled_yx[0]))
    return {
        "pose_xy": pose_xy,
        "marker_xy": marker_xy,
        "pose_marker_max_abs": float((pose_xy - marker_xy).abs().max()),
        "grid_pose_xy": grid_pose_xy,
        "pooled_marker_xy": pooled_xy,
        "grid_marker_max_abs": float(
            (grid_pose_xy - pooled_xy).abs().max()
        ),
    }


def run_audit():
    torch.manual_seed(20260718)
    checks = {}

    raw = torch.rand(3, REGIONS, 11, 7)
    raw[:, :, 0, 0] = 0
    partition = normalize_partition(raw)
    sum_by_pixel = partition.sum(dim=1)
    checks["partition_empty_exact"] = bool(
        torch.equal(partition[:, :, 0, 0], torch.zeros(3, REGIONS))
    )
    checks["partition_nonempty_sum_max_error"] = float(
        (sum_by_pixel[:, 1:] - 1.0).abs().max()
    )

    patches = 256
    binary = torch.zeros(1, REGIONS, patches)
    binary[:, :, :12] = 1.0
    prior, valid = regional_log_prior(binary)
    unnormalized = prior.exp()
    inside = unnormalized[:, :, :12].sum(dim=-1)
    outside = unnormalized[:, :, 12:].sum(dim=-1)
    leak_fraction = outside / inside
    checks["total_background_leak_max"] = float(leak_fraction.max())
    checks["all_sparse_slots_valid"] = bool(valid.all())

    all_one = torch.ones(2, REGIONS, patches)
    all_one_prior, all_one_valid = regional_log_prior(all_one)
    checks["all_one_prior_max_abs"] = float(all_one_prior.abs().max())
    checks["all_one_valid"] = bool(all_one_valid.all())

    zero = torch.zeros(2, REGIONS, patches)
    zero_prior, zero_valid = regional_log_prior(zero)
    checks["zero_prior_exact"] = bool(torch.equal(zero_prior, zero))
    checks["zero_all_invalid"] = bool((~zero_valid).all())

    directional = torch.zeros(1, REGIONS, 4)
    directional[:, :, 2] = 1.0
    directional_prior, _ = regional_log_prior(directional)
    additive = make_cls_patch_attn_mask(directional_prior, heads=2)
    base_logits = torch.zeros_like(additive)
    attention = torch.softmax(base_logits + additive, dim=-1)
    base_attention = torch.softmax(base_logits, dim=-1)
    cls_rows = attention[:, 0]
    target_probability = cls_rows[:, 3]
    background_probability = torch.stack(
        (cls_rows[:, 1], cls_rows[:, 2], cls_rows[:, 4]), dim=-1
    ).amax(dim=-1)
    checks["single_patch_beats_background"] = bool(
        (target_probability > background_probability).all()
    )
    checks["cls_self_unchanged_vs_target"] = bool(
        torch.equal(cls_rows[:, 0], target_probability)
    )
    checks["non_cls_rows_exact"] = bool(
        torch.equal(attention[:, 1:], base_attention[:, 1:])
    )
    nonzero = (additive != 0).nonzero(as_tuple=False)
    checks["mask_only_cls_query"] = bool((nonzero[:, 1] == 0).all())
    checks["mask_excludes_cls_key"] = bool((nonzero[:, 2] > 0).all())

    tokens = torch.arange(2 * 6 * 3, dtype=torch.float32).view(2, 6, 3)
    expanded = expand_region_branches(tokens)
    restored = expanded.view(2, REGIONS, 6, 3)
    checks["branch_expand_restore_exact"] = bool(
        torch.equal(restored, tokens[:, None].expand_as(restored))
    )

    geometry = transform_marker_and_pose()
    checks["geometry_pose_marker_max_abs"] = geometry["pose_marker_max_abs"]
    checks["geometry_grid_marker_max_abs"] = geometry["grid_marker_max_abs"]

    gates = {
        "partition": (
            checks["partition_empty_exact"]
            and checks["partition_nonempty_sum_max_error"] <= 1e-6
        ),
        "leak_budget": checks["total_background_leak_max"] <= LEAK_RATIO,
        "all_one": (
            checks["all_one_prior_max_abs"] == 0.0
            and checks["all_one_valid"]
        ),
        "null": checks["zero_prior_exact"] and checks["zero_all_invalid"],
        "attention_direction": (
            checks["single_patch_beats_background"]
            and checks["cls_self_unchanged_vs_target"]
            and checks["non_cls_rows_exact"]
            and checks["mask_only_cls_query"]
            and checks["mask_excludes_cls_key"]
        ),
        "branch_order": checks["branch_expand_restore_exact"],
        "geometry_chain": (
            checks["geometry_pose_marker_max_abs"] <= 0.51
            and checks["geometry_grid_marker_max_abs"] <= 1.0
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "torch_version": torch.__version__,
        "dtype": "float32",
        "regions": REGIONS,
        "heads": HEADS,
        "leak_ratio": LEAK_RATIO,
        "checks": checks,
        "gates": gates,
        "script_sha256": sha256_file(__file__),
    }


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
