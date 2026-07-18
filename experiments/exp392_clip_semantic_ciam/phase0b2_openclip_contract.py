#!/usr/bin/env python3
"""OpenCLIP-specific contract smoke for exp392 Phase 0B2 PC-MBCLS.

The script uses synthetic images and masks only.  It does not read a ReID
dataset, build an optimizer, or start training.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


REGIONS = 5
SPLIT_BLOCK = 20
LEAK_RATIO = 0.01


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regional_log_prior(masks, leak_ratio=LEAK_RATIO):
    if masks.ndim != 3 or masks.shape[1] != REGIONS:
        raise ValueError("masks must have shape [B,5,P]")
    if bool((masks < 0).any()):
        raise ValueError("masks must be nonnegative")
    mass = masks.sum(dim=-1, keepdim=True)
    maximum = masks.amax(dim=-1, keepdim=True)
    valid = maximum.squeeze(-1) > 0
    delta = float(leak_ratio) * mass / float(masks.shape[-1])
    prior = torch.log(
        (masks + delta).clamp_min(torch.finfo(masks.dtype).tiny)
        / (maximum + delta).clamp_min(torch.finfo(masks.dtype).tiny)
    )
    prior = torch.where(valid[..., None], prior, torch.zeros_like(prior))
    return prior, valid


def expand_region_branches(tokens):
    batch, sequence, width = tokens.shape
    return (
        tokens[:, None]
        .expand(batch, REGIONS, sequence, width)
        .reshape(batch * REGIONS, sequence, width)
        .clone()
    )


def additive_cls_mask(prior, heads):
    if prior.ndim == 3:
        flat = prior.reshape(-1, prior.shape[-1])
    elif prior.ndim == 2:
        flat = prior
    else:
        raise ValueError("prior must have shape [B,R,P] or [N,P]")
    branches, patches = flat.shape
    sequence = patches + 1
    mask = torch.zeros(
        branches,
        sequence,
        sequence,
        device=prior.device,
        dtype=prior.dtype,
    )
    mask[:, 0, 1:] = flat
    return mask.repeat_interleave(int(heads), dim=0)


def project_pooled(visual, tokens):
    pooled, _ = visual._pool(tokens)
    if visual.proj is not None:
        pooled = pooled @ visual.proj
    return F.normalize(pooled.float(), dim=-1)


@torch.no_grad()
def forward_shared_trunk(visual, images, split_block=SPLIT_BLOCK):
    tokens = visual._embeds(images)
    blocks = visual.transformer.resblocks
    for block in blocks[:split_block]:
        tokens = block(tokens)
    return tokens


@torch.no_grad()
def forward_official_tail(visual, shared, split_block=SPLIT_BLOCK):
    tokens = shared
    for block in visual.transformer.resblocks[split_block:]:
        tokens = block(tokens)
    return project_pooled(visual, tokens)


@torch.no_grad()
def forward_regions(visual, shared, masks, split_block=SPLIT_BLOCK):
    batch = shared.shape[0]
    flat_masks = masks.flatten(2).float()
    prior, valid = regional_log_prior(flat_masks)
    branches = expand_region_branches(shared)
    flat_valid = valid.reshape(-1)
    output = torch.zeros(
        batch * REGIONS,
        visual.output_dim,
        dtype=torch.float32,
        device=shared.device,
    )
    if bool(flat_valid.any()):
        selected = branches[flat_valid]
        selected_prior = prior.reshape(batch * REGIONS, -1)[flat_valid]
        all_one = bool((selected_prior == 0).all())
        if all_one:
            attention_mask = None
        else:
            heads = visual.transformer.resblocks[0].attn.num_heads
            attention_mask = additive_cls_mask(selected_prior, heads=heads)
        for block in visual.transformer.resblocks[split_block:]:
            selected = block(selected, attn_mask=attention_mask)
        output[flat_valid] = project_pooled(visual, selected)
    return output.view(batch, REGIONS, -1), valid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    import open_clip

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained=str(checkpoint)
    )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.visual
    if len(visual.transformer.resblocks) != 24:
        raise RuntimeError("Expected 24 ViT-L/14 blocks")
    if tuple(visual.grid_size) != (16, 16):
        raise RuntimeError("Expected 16x16 ViT-L/14 patch grid")

    images = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        official = F.normalize(visual(images).float(), dim=-1)
        shared = forward_shared_trunk(visual, images)
        manual = forward_official_tail(visual, shared.clone())

        all_one_masks = torch.ones(1, REGIONS, 16, 16, device=device)
        all_one, all_one_valid = forward_regions(
            visual, shared.clone(), all_one_masks
        )
        all_one_repeat, _ = forward_regions(
            visual, shared.clone(), all_one_masks
        )

        sparse_masks = torch.zeros(1, REGIONS, 16, 16, device=device)
        for region in range(REGIONS):
            top = region * 3
            sparse_masks[:, region, top:min(top + 4, 16), 4:12] = 1.0
        sparse, sparse_valid = forward_regions(
            visual, shared.clone(), sparse_masks
        )
        sparse_repeat, _ = forward_regions(
            visual, shared.clone(), sparse_masks
        )

        zero_masks = torch.zeros(1, REGIONS, 16, 16, device=device)
        zero, zero_valid = forward_regions(visual, shared.clone(), zero_masks)

    official_vs_manual = float((official - manual).abs().max().item())
    all_one_vs_official = float(
        (all_one - official[:, None]).abs().max().item()
    )
    sparse_change = float(
        (sparse - all_one).abs().amax(dim=-1).min().item()
    )
    checks = {
        "official_vs_manual_max_abs": official_vs_manual,
        "all_one_vs_official_max_abs": all_one_vs_official,
        "all_one_repeat_exact": bool(torch.equal(all_one, all_one_repeat)),
        "all_one_all_valid": bool(all_one_valid.all()),
        "sparse_repeat_exact": bool(torch.equal(sparse, sparse_repeat)),
        "sparse_all_valid": bool(sparse_valid.all()),
        "sparse_min_region_max_abs_change": sparse_change,
        "zero_output_exact": bool(torch.equal(zero, torch.zeros_like(zero))),
        "zero_all_invalid": bool((~zero_valid).all()),
        "finite": bool(
            torch.isfinite(official).all()
            and torch.isfinite(all_one).all()
            and torch.isfinite(sparse).all()
        ),
    }
    gates = {
        "manual_24_block_parity": official_vs_manual <= 1e-6,
        "all_one_parity": all_one_vs_official <= 1e-6,
        "repeat": (
            checks["all_one_repeat_exact"] and checks["sparse_repeat_exact"]
        ),
        "validity": (
            checks["all_one_all_valid"]
            and checks["sparse_all_valid"]
            and checks["zero_all_invalid"]
        ),
        "sparse_effective": sparse_change > 1e-6,
        "null": checks["zero_output_exact"],
        "finite": checks["finite"],
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "checks": checks,
        "gates": gates,
        "device": str(device),
        "torch_version": torch.__version__,
        "open_clip_version": open_clip.__version__,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "script_sha256": sha256_file(__file__),
        "split_block": SPLIT_BLOCK,
        "regions": REGIONS,
        "leak_ratio": LEAK_RATIO,
        "attention_mask_semantics": "additive CLS-query to patch-key only",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
