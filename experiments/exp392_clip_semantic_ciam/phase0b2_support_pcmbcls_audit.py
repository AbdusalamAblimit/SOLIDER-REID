#!/usr/bin/env python3
"""Phase 0B2-SI PC-MBCLS slot-support readout audit.

The audit keeps the B2-SC support prompts and nested occlusions fixed.  It
changes only the frozen CLIP readout from crop-global CLS to pose-conditioned
multi-block CLS on the aligned full pedestrian RGB.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REGIONS = 5
EXPECTED_ONTOLOGY_SCRIPT_SHA256 = (
    "b0d5ce6a53e94d09fa5d15c338392ea31437eee036299256c424ed30489028ca"
)
EXPECTED_SUPPORT_SCRIPT_SHA256 = (
    "692a3662d0de9613a6a1c573d2d86bfd7f40b3082f215d005aa4f8857869496a"
)
EXPECTED_PCMBCLS_SCRIPT_SHA256 = (
    "7206dc13bf69b5666b54169ae3333f838c48b16d0c963512e7c67d906354c2c7"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aspect_letterbox(images, masks, mean):
    resized_images = F.interpolate(
        images.float(),
        size=(224, 75),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    clip_images = mean.expand(images.shape[0], 3, 224, 224).clone()
    clip_images[:, :, :, 74:149] = resized_images.clamp(0.0, 1.0)

    resized_masks = F.interpolate(
        masks.float(), size=(224, 75), mode="nearest"
    )
    clip_masks = torch.zeros(
        masks.shape[0],
        masks.shape[1],
        224,
        224,
        dtype=torch.float32,
    )
    clip_masks[:, :, :, 74:149] = resized_masks
    grid_masks = F.avg_pool2d(clip_masks, kernel_size=14, stride=14)
    if grid_masks.shape[-2:] != (16, 16):
        raise RuntimeError("CLIP mask grid mismatch")
    return clip_images, clip_masks, grid_masks


def pairwise_pixel_product_max(masks):
    values = [
        masks[:, left].mul(masks[:, right]).max()
        for left in range(REGIONS)
        for right in range(left + 1, REGIONS)
    ]
    return float(torch.stack(values).max()) if values else 0.0


class FullImageOcclusionBuilder:
    def __init__(self, ontology_module, mean, levels):
        self.ontology = ontology_module
        self.mean = mean
        self.levels = tuple(float(value) for value in levels)

    def __call__(self, images, masks, valid):
        full_masks = F.interpolate(
            masks.float(), size=(384, 128), mode="nearest"
        )
        variants = []
        variant_masks = []
        targets = []
        realized = []
        target_error = []
        support_pixels = []
        source_image_index = []
        for image_index in range(images.shape[0]):
            for region in range(REGIONS):
                if not bool(valid[image_index, region]):
                    continue
                region_mask = full_masks[image_index, region]
                support = region_mask > (0.05 * region_mask.max())
                coordinates = support.nonzero(as_tuple=False)
                if len(coordinates) == 0:
                    raise RuntimeError("Pose-valid full-image slot is empty")
                count = int(len(coordinates))
                level_realized = []
                level_error = []
                for level in self.levels:
                    erase_count = int(round(level * count))
                    variant = images[image_index].clone()
                    if erase_count > 0:
                        chosen = coordinates[:erase_count]
                        variant[:, chosen[:, 0], chosen[:, 1]] = (
                            self.mean[0, :, 0, 0, None]
                        )
                    variants.append(variant)
                    variant_masks.append(masks[image_index])
                    level_realized.append(erase_count / float(count))
                    level_error.append(
                        abs(erase_count / float(count) - level)
                    )
                targets.append(region)
                realized.append(level_realized)
                target_error.append(level_error)
                support_pixels.append(count)
                source_image_index.append(image_index)
        return {
            "images": torch.stack(variants),
            "masks": torch.stack(variant_masks),
            "targets": torch.as_tensor(targets, dtype=torch.long),
            "realized": torch.as_tensor(realized, dtype=torch.float64),
            "target_error": torch.as_tensor(
                target_error, dtype=torch.float64
            ),
            "support_pixels": torch.as_tensor(
                support_pixels, dtype=torch.long
            ),
            "source_image_index": torch.as_tensor(
                source_image_index, dtype=torch.long
            ),
        }


class FrozenSupportPCMBCLS:
    def __init__(
        self,
        ontology_module,
        support_module,
        pcmbcls_module,
        checkpoint,
        device,
        clip_batch,
    ):
        import open_clip

        self.ontology = ontology_module
        self.support = support_module
        self.pcmbcls = pcmbcls_module
        self.device = device
        self.clip_batch = int(clip_batch)
        self.model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=str(checkpoint)
        )
        self.model = self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.visual = self.model.visual
        if len(self.visual.transformer.resblocks) != 24:
            raise RuntimeError("Expected 24 ViT-L/14 blocks")
        if tuple(self.visual.grid_size) != (16, 16):
            raise RuntimeError("Expected 16x16 ViT-L/14 patch grid")

        prompts = []
        for phrase in ontology_module.DISJOINT_REGION_PHRASES:
            for visible, occluded in support_module.SUPPORT_PROMPT_PAIRS:
                prompts.extend((visible.format(phrase), occluded.format(phrase)))
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            text = self.model.encode_text(tokens)
        text = F.normalize(text.float(), dim=-1)
        text = text.view(
            REGIONS, len(support_module.SUPPORT_PROMPT_PAIRS), 2, -1
        )
        self.text = F.normalize(text.mean(1), dim=-1)
        self.prompt_payload = {
            "regions": list(ontology_module.REGION_NAMES),
            "region_phrases": list(
                ontology_module.DISJOINT_REGION_PHRASES
            ),
            "state_order": ["visible", "occluded"],
            "template_pairs": [
                list(pair) for pair in support_module.SUPPORT_PROMPT_PAIRS
            ],
            "prompts": prompts,
            "temperature": support_module.TEMPERATURE,
        }

        normalizers = [
            transform for transform in preprocess.transforms
            if hasattr(transform, "mean") and hasattr(transform, "std")
        ]
        if len(normalizers) != 1:
            raise RuntimeError("Could not identify official CLIP Normalize")
        official_mean = tuple(float(value) for value in normalizers[0].mean)
        official_std = tuple(float(value) for value in normalizers[0].std)
        if not np.allclose(
            official_mean, ontology_module.CLIP_MEAN, atol=1e-8, rtol=0.0
        ):
            raise RuntimeError("CLIP mean mismatch")
        if not np.allclose(
            official_std, ontology_module.CLIP_STD, atol=1e-8, rtol=0.0
        ):
            raise RuntimeError("CLIP std mismatch")
        self.mean = torch.as_tensor(
            ontology_module.CLIP_MEAN, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.std = torch.as_tensor(
            ontology_module.CLIP_STD, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.preprocess_repr = repr(preprocess)
        if "BICUBIC" not in self.preprocess_repr.upper():
            raise RuntimeError("Official CLIP preprocess is not bicubic")

    @torch.no_grad()
    def _encode_chunk(self, images, masks):
        clip_images, pixel_masks, grid_masks = aspect_letterbox(
            images, masks, self.mean
        )
        pixel_product_max = pairwise_pixel_product_max(pixel_masks)
        normalized = ((clip_images - self.mean) / self.std).to(self.device)
        grid_masks = grid_masks.to(self.device)
        shared = self.pcmbcls.forward_shared_trunk(
            self.visual, normalized
        )
        global_features = self.pcmbcls.forward_official_tail(
            self.visual, shared.clone()
        )
        region_features, region_valid = self.pcmbcls.forward_regions(
            self.visual, shared.clone(), grid_masks
        )
        region_logits = torch.einsum(
            "brd,rsd->brs", region_features, self.text
        )
        region_q_visible = torch.softmax(
            region_logits / self.support.TEMPERATURE, dim=-1
        )[..., 0]
        zero_features, zero_valid = self.pcmbcls.forward_regions(
            self.visual,
            shared[:1].clone(),
            torch.zeros(
                1, REGIONS, 16, 16, device=self.device
            ),
        )
        null_exact = bool(
            torch.equal(zero_features, torch.zeros_like(zero_features))
            and (~zero_valid).all()
        )
        return {
            "global_features": global_features.cpu(),
            "region_q_visible": region_q_visible.cpu(),
            "region_valid": region_valid.cpu(),
            "pixel_product_max": pixel_product_max,
            "null_exact": null_exact,
        }

    @torch.no_grad()
    def encode(self, images, masks, targets, repeat_check):
        global_features = []
        region_q_visible = []
        region_valid = []
        pixel_product_max = 0.0
        null_exact = True
        elapsed = 0.0
        first_chunk_cache = None
        for start in range(0, len(images), self.clip_batch):
            stop = min(start + self.clip_batch, len(images))
            began = time.perf_counter()
            encoded = self._encode_chunk(
                images[start:stop], masks[start:stop]
            )
            elapsed += time.perf_counter() - began
            global_features.append(encoded["global_features"])
            region_q_visible.append(encoded["region_q_visible"])
            region_valid.append(encoded["region_valid"])
            pixel_product_max = max(
                pixel_product_max, encoded["pixel_product_max"]
            )
            null_exact = null_exact and encoded["null_exact"]
            if start == 0:
                first_chunk_cache = encoded

        repeat_exact = True
        if repeat_check:
            stop = min(self.clip_batch, len(images))
            repeated = self._encode_chunk(images[:stop], masks[:stop])
            repeat_exact = bool(
                torch.equal(
                    first_chunk_cache["global_features"],
                    repeated["global_features"],
                )
                and torch.equal(
                    first_chunk_cache["region_q_visible"],
                    repeated["region_q_visible"],
                )
                and torch.equal(
                    first_chunk_cache["region_valid"],
                    repeated["region_valid"],
                )
            )
        global_features = torch.cat(global_features, dim=0)
        region_q_visible = torch.cat(region_q_visible, dim=0)
        region_valid = torch.cat(region_valid, dim=0)
        target_text = self.text.index_select(
            0, targets.to(self.device)
        ).cpu()
        global_logits = torch.einsum(
            "nd,nsd->ns", global_features, target_text
        )
        global_q_visible = torch.softmax(
            global_logits / self.support.TEMPERATURE, dim=-1
        )[..., 0]
        target_q_visible = region_q_visible.gather(
            1, targets[:, None]
        ).squeeze(1)
        target_valid = region_valid.gather(
            1, targets[:, None]
        ).squeeze(1)
        onehot = F.one_hot(targets, num_classes=REGIONS).bool()
        non_target_valid = region_valid & (~onehot)
        non_target_q_visible = (
            region_q_visible.masked_fill(~non_target_valid, 0.0).sum(1)
            / non_target_valid.sum(1).clamp_min(1).float()
        )
        return {
            "target_q_visible": target_q_visible,
            "non_target_q_visible": non_target_q_visible,
            "global_q_visible": global_q_visible,
            "target_valid": target_valid,
            "repeat_exact": repeat_exact,
            "pixel_product_max": pixel_product_max,
            "null_exact": null_exact,
            "elapsed_seconds": elapsed,
        }


def summarize(
    support_module,
    target_q,
    non_target_q,
    global_q,
    targets,
    realized,
    pids,
    repeats,
    seed,
):
    from scipy.stats import spearmanr

    levels = list(support_module.LEVELS)
    per_region = []
    macro_by_level = []
    for level_index in range(len(levels)):
        class_means = []
        for region in range(REGIONS):
            keep = targets == region
            class_means.append(
                float(target_q[keep, level_index].mean())
            )
        macro_by_level.append(float(np.mean(class_means)))

    for region in range(REGIONS):
        keep = targets == region
        target_values = target_q[keep].numpy()
        non_target_values = non_target_q[keep].numpy()
        global_values = global_q[keep].numpy()
        overlaps = realized[keep].numpy()
        region_pids = pids[keep].numpy()
        target_delta = target_values[:, 0] - target_values[:, -1]
        non_target_delta = (
            non_target_values[:, 0] - non_target_values[:, -1]
        )
        global_delta = global_values[:, 0] - global_values[:, -1]
        correlation = float(
            spearmanr(
                overlaps.reshape(-1), -target_values.reshape(-1)
            ).statistic
        )
        per_region.append({
            "region": int(region),
            "samples": int(keep.sum()),
            "target_q_visible_mean_by_level": [
                float(target_values[:, index].mean())
                for index in range(len(levels))
            ],
            "non_target_q_visible_mean_by_level": [
                float(non_target_values[:, index].mean())
                for index in range(len(levels))
            ],
            "global_q_visible_mean_by_level": [
                float(global_values[:, index].mean())
                for index in range(len(levels))
            ],
            "spearman_overlap_negative_target_q": correlation,
            "target_level0_minus_level75": support_module.clustered_mean(
                target_delta, region_pids, repeats, seed + region + 1
            ),
            "target_minus_non_target_delta": (
                support_module.clustered_mean(
                    target_delta - non_target_delta,
                    region_pids,
                    repeats,
                    seed + region + 11,
                )
            ),
            "target_minus_global_delta": support_module.clustered_mean(
                target_delta - global_delta,
                region_pids,
                repeats,
                seed + region + 21,
            ),
        })
    macro_adjacent = [
        macro_by_level[index] - macro_by_level[index + 1]
        for index in range(len(levels) - 1)
    ]
    macro_target_minus_global = float(np.mean([
        item["target_minus_global_delta"]["mean"]
        for item in per_region
    ]))
    return {
        "levels": levels,
        "macro_target_q_visible_by_level": macro_by_level,
        "macro_target_adjacent_delta": macro_adjacent,
        "macro_target_minus_global_delta": macro_target_minus_global,
        "per_region": per_region,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--phase0b-script", required=True)
    parser.add_argument("--ontology-script", required=True)
    parser.add_argument("--support-script", required=True)
    parser.add_argument("--pcmbcls-script", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pose-artifact", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--clip-batch", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def main():
    args = parse_args()
    audit_script_path = Path(__file__).resolve()
    repo_root = Path(args.repo_root).resolve()
    phase0b_script = Path(args.phase0b_script).resolve()
    ontology_script = Path(args.ontology_script).resolve()
    support_script = Path(args.support_script).resolve()
    pcmbcls_script = Path(args.pcmbcls_script).resolve()
    data_root = Path(args.data_root).resolve()
    pose_artifact = Path(args.pose_artifact).resolve()
    clip_checkpoint = Path(args.clip_checkpoint).resolve()
    expected = {
        ontology_script: EXPECTED_ONTOLOGY_SCRIPT_SHA256,
        support_script: EXPECTED_SUPPORT_SCRIPT_SHA256,
        pcmbcls_script: EXPECTED_PCMBCLS_SCRIPT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError("Dependency script SHA mismatch: %s" % path)
    ontology = load_module("exp392_b2_ontology", ontology_script)
    support = load_module("exp392_b2_support", support_script)
    pcmbcls = load_module("exp392_b2_pcmbcls", pcmbcls_script)
    if sha256_file(phase0b_script) != ontology.EXPECTED_PHASE0B_SCRIPT_SHA256:
        raise RuntimeError("Phase 0B audit script SHA mismatch")
    base = load_module("exp392_phase0b", phase0b_script)
    contract = ontology.ontology_static_contract(base, "hard-owner")
    if contract["status"] != "PASS":
        raise RuntimeError("Hard-owner ontology static contract failed")
    if sha256_file(pose_artifact / "manifest.json") != base.EXPECTED_POSE_MANIFEST_SHA256:
        raise RuntimeError("Pose manifest SHA mismatch")
    if sha256_file(clip_checkpoint) != base.EXPECTED_CLIP_SHA256:
        raise RuntimeError("CLIP checkpoint SHA mismatch")
    runtime_sha = {
        relative: sha256_file(repo_root / relative)
        for relative in base.EXPECTED_RUNTIME_SHA256
    }
    if runtime_sha != base.EXPECTED_RUNTIME_SHA256:
        raise RuntimeError("Phase 0B minimal runtime SHA mismatch")

    os.chdir(str(repo_root))
    sys.path.insert(0, str(repo_root))
    from datasets.occluded_duke import OccludedDuke
    from datasets.pose_targets import PoseTargetStore

    base.set_seed(args.seed)
    dataset = OccludedDuke(root=str(data_root), verbose=False)
    records = list(dataset.train)
    if len(records) != 15618:
        raise RuntimeError("Unexpected official train size: %d" % len(records))
    if args.max_samples > 0:
        records = records[:args.max_samples]
    pose_store = PoseTargetStore(
        pose_artifact, base.EXPECTED_POSE_MANIFEST_SHA256
    )
    audit_dataset = ontology.RecipientDataset(
        base,
        records,
        pose_store,
        args.seed,
        verify_sha=True,
        partition_mode="hard-owner",
    )
    loader = torch.utils.data.DataLoader(
        audit_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=ontology.collate_recipient,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )
    device = torch.device(args.device)
    teacher = FrozenSupportPCMBCLS(
        ontology,
        support,
        pcmbcls,
        clip_checkpoint,
        device,
        args.clip_batch,
    )
    builder = FullImageOcclusionBuilder(
        ontology, teacher.mean, support.LEVELS
    )

    target_q_all = []
    non_target_q_all = []
    global_q_all = []
    target_all = []
    realized_all = []
    error_all = []
    pixels_all = []
    pid_all = []
    repeat_exact = True
    target_valid_exact = True
    null_exact = True
    pixel_product_max = 0.0
    forward_seconds = 0.0
    manifest_digest = hashlib.sha256()
    for batch_index, batch in enumerate(loader):
        for relative_path, image_sha in zip(
            batch["relative_path"], batch["image_sha256"]
        ):
            manifest_digest.update(relative_path.encode("utf-8"))
            manifest_digest.update(b"\0")
            manifest_digest.update(image_sha.encode("ascii"))
        built = builder(
            batch["pre"], batch["masks"], batch["valid"].bool()
        )
        encoded = teacher.encode(
            built["images"],
            built["masks"],
            built["targets"].repeat_interleave(len(support.LEVELS)),
            repeat_check=batch_index == 0,
        )
        slot_count = len(built["targets"])
        level_count = len(support.LEVELS)
        target_q_all.append(
            encoded["target_q_visible"].view(slot_count, level_count)
        )
        non_target_q_all.append(
            encoded["non_target_q_visible"].view(
                slot_count, level_count
            )
        )
        global_q_all.append(
            encoded["global_q_visible"].view(slot_count, level_count)
        )
        target_all.append(built["targets"])
        realized_all.append(built["realized"])
        error_all.append(built["target_error"])
        pixels_all.append(built["support_pixels"])
        pid_all.append(
            batch["pid"].index_select(
                0, built["source_image_index"]
            )
        )
        target_valid_exact = target_valid_exact and bool(
            encoded["target_valid"].all()
        )
        repeat_exact = repeat_exact and encoded["repeat_exact"]
        null_exact = null_exact and encoded["null_exact"]
        pixel_product_max = max(
            pixel_product_max, encoded["pixel_product_max"]
        )
        forward_seconds += encoded["elapsed_seconds"]

    target_q = torch.cat(target_q_all, dim=0)
    non_target_q = torch.cat(non_target_q_all, dim=0)
    global_q = torch.cat(global_q_all, dim=0)
    targets = torch.cat(target_all, dim=0)
    realized = torch.cat(realized_all, dim=0)
    target_error = torch.cat(error_all, dim=0)
    support_pixels = torch.cat(pixels_all, dim=0)
    pids = torch.cat(pid_all, dim=0)
    summary = summarize(
        support,
        target_q,
        non_target_q,
        global_q,
        targets,
        realized,
        pids,
        args.bootstrap_repeats,
        args.seed,
    )
    finite = bool(
        torch.isfinite(target_q).all()
        and torch.isfinite(non_target_q).all()
        and torch.isfinite(global_q).all()
        and torch.isfinite(realized).all()
    )
    max_target_error = float(target_error.max())
    max_allowed_error = float((1.0 / support_pixels.double()).max())
    gates = {
        "finite": finite,
        "repeat_exact": bool(repeat_exact),
        "null_exact": bool(null_exact),
        "target_valid_exact": bool(target_valid_exact),
        "hard_owner_pixel_product_exact_zero": pixel_product_max == 0.0,
        "level_target_error_within_one_pixel": (
            max_target_error <= max_allowed_error + 1e-12
        ),
        "realized_overlap_strictly_increasing": bool(
            (realized[:, 1:] > realized[:, :-1]).all()
        ),
        "all_region_positive_target_spearman": all(
            item["spearman_overlap_negative_target_q"] > 0.0
            for item in summary["per_region"]
        ),
        "all_region_positive_target_level0_minus_level75": all(
            item["target_level0_minus_level75"]["mean"] > 0.0
            for item in summary["per_region"]
        ),
        "macro_all_target_adjacent_monotonic": all(
            value > 0.0
            for value in summary["macro_target_adjacent_delta"]
        ),
        "all_region_target_delta_gt_non_target": all(
            item["target_minus_non_target_delta"]["mean"] > 0.0
            for item in summary["per_region"]
        ),
        "macro_target_delta_gt_global": (
            summary["macro_target_minus_global_delta"] > 0.0
        ),
    }
    full_audit = args.max_samples <= 0
    verdict = (
        "B2_SI_SMOKE_PASS"
        if not full_audit and all(gates.values())
        else "B2_SI_SMOKE_FAIL"
        if not full_audit
        else "B2_SI_FULL_REQUIRES_SEPARATE_PROTOCOL"
    )
    result = {
        "status": "EXP392_PHASE0B2_SI_COMPLETE",
        "scope": "full" if full_audit else "smoke",
        "verdict": verdict,
        "formal_training_authorized": False,
        "full_teacher_audit_authorized": verdict == "B2_SI_SMOKE_PASS",
        "execution": {
            "repo_root": str(repo_root),
            "source_commit": args.source_commit,
            "phase0b_script_sha256": sha256_file(phase0b_script),
            "ontology_script_sha256": sha256_file(ontology_script),
            "support_script_sha256": sha256_file(support_script),
            "pcmbcls_script_sha256": sha256_file(pcmbcls_script),
            "audit_script_sha256": sha256_file(audit_script_path),
            "runtime_sha256": runtime_sha,
            "pose_manifest_sha256": base.EXPECTED_POSE_MANIFEST_SHA256,
            "clip_checkpoint_sha256": base.EXPECTED_CLIP_SHA256,
            "device": str(device),
            "torch_version": torch.__version__,
            "open_clip_version": __import__("open_clip").__version__,
            "images": len(records),
            "valid_target_slots": len(target_q),
            "encoded_full_image_variants": (
                len(target_q) * len(support.LEVELS)
            ),
            "batch_size": args.batch_size,
            "clip_batch": args.clip_batch,
            "workers": args.workers,
            "seed": args.seed,
            "sample_manifest_sha256": manifest_digest.hexdigest(),
            "pcmbcls_forward_seconds": forward_seconds,
        },
        "readout": {
            "name": "PC-MBCLS",
            "split_block": pcmbcls.SPLIT_BLOCK,
            "leak_ratio": pcmbcls.LEAK_RATIO,
            "geometry": "aspect-letterbox 224x75 in x=[74,149)",
            "image_interpolation": "bicubic antialias",
            "hard_owner_mask_interpolation": "nearest then 14x14 average pool",
            "pixel_product_max": pixel_product_max,
            "official_global_cls_control": True,
            "non_target_slot_control": True,
        },
        "prompt_payload": teacher.prompt_payload,
        "prompt_sha256": sha256_json(teacher.prompt_payload),
        "open_clip_preprocess_repr": teacher.preprocess_repr,
        "support_pixels": {
            "min": int(support_pixels.min()),
            "median": float(support_pixels.double().median()),
            "max": int(support_pixels.max()),
            "max_target_error": max_target_error,
            "max_allowed_one_pixel_error": max_allowed_error,
        },
        "summary": summary,
        "gates": gates,
    }
    write_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "verdict": verdict,
        "images": len(records),
        "valid_target_slots": len(target_q),
        "encoded_full_image_variants": (
            len(target_q) * len(support.LEVELS)
        ),
        "summary": summary,
        "gates": gates,
        "output_sha256": sha256_file(args.output),
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if verdict == "B2_SI_SMOKE_PASS" else 1)


if __name__ == "__main__":
    main()
