#!/usr/bin/env python3
"""Phase 0B2-SC crop-global slot-support feasibility audit.

This script never builds a ReID model, optimizer, or training loss.  It keeps
the hard-owner anatomy and frozen crop-global CLIP readout fixed, then tests
whether slot-conditioned visible/occluded text scores respond monotonically to
three deterministic nested occlusion levels.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REGIONS = 5
LEVELS = (0.0, 0.25, 0.50, 0.75)
TEMPERATURE = 0.07
EXPECTED_ONTOLOGY_SCRIPT_SHA256 = (
    "b0d5ce6a53e94d09fa5d15c338392ea31437eee036299256c424ed30489028ca"
)
SUPPORT_PROMPT_PAIRS = (
    (
        "a photo of a person with clearly visible {}",
        "a photo of a person with an occluded or obstructed {}",
    ),
    (
        "the person's {} is clearly visible and unobstructed",
        "the person's {} is hidden or obstructed",
    ),
    (
        "clear visual evidence of the person's {}",
        "weak visual evidence of the person's {}",
    ),
    (
        "a fully observable human {}",
        "a heavily obscured human {}",
    ),
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


def clustered_mean(values, pids, repeats, seed):
    values = np.asarray(values, dtype=np.float64)
    pids = np.asarray(pids, dtype=np.int64)
    if len(values) == 0:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "pids": 0,
            "samples": 0,
            "repeats": int(repeats),
        }
    unique = np.unique(pids)
    per_pid = np.asarray([values[pids == pid].mean() for pid in unique])
    rng = np.random.RandomState(seed)
    draws = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sample = rng.randint(0, len(per_pid), size=len(per_pid))
        draws[index] = per_pid[sample].mean()
    return {
        "mean": float(per_pid.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "pids": int(len(unique)),
        "samples": int(len(values)),
        "repeats": int(repeats),
    }


class FrozenSupportCropTeacher:
    def __init__(self, ontology_module, checkpoint, device, clip_batch):
        import open_clip

        self.ontology = ontology_module
        self.device = device
        self.clip_batch = int(clip_batch)
        self.model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=str(checkpoint)
        )
        self.model = self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        prompts = []
        for phrase in ontology_module.DISJOINT_REGION_PHRASES:
            for visible, occluded in SUPPORT_PROMPT_PAIRS:
                prompts.extend((visible.format(phrase), occluded.format(phrase)))
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            text = self.model.encode_text(tokens)
        text = F.normalize(text.float(), dim=-1)
        text = text.view(REGIONS, len(SUPPORT_PROMPT_PAIRS), 2, -1)
        self.text = F.normalize(text.mean(1), dim=-1).cpu()
        self.prompt_payload = {
            "regions": list(ontology_module.REGION_NAMES),
            "region_phrases": list(ontology_module.DISJOINT_REGION_PHRASES),
            "state_order": ["visible", "occluded"],
            "template_pairs": [list(pair) for pair in SUPPORT_PROMPT_PAIRS],
            "prompts": prompts,
            "temperature": TEMPERATURE,
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
            raise RuntimeError("CLIP mean does not match official preprocess")
        if not np.allclose(
            official_std, ontology_module.CLIP_STD, atol=1e-8, rtol=0.0
        ):
            raise RuntimeError("CLIP std does not match official preprocess")
        self.mean = torch.as_tensor(
            ontology_module.CLIP_MEAN, dtype=torch.float32
        ).view(3, 1, 1)
        self.std = torch.as_tensor(
            ontology_module.CLIP_STD, dtype=torch.float32
        ).view(3, 1, 1)
        self.preprocess_repr = repr(preprocess)
        if "BICUBIC" not in self.preprocess_repr.upper():
            raise RuntimeError("Official CLIP preprocess is not bicubic")

    def _crop_image_and_support(self, image, mask):
        height, width = image.shape[-2:]
        full_mask = F.interpolate(
            mask[None, None].float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        selected = full_mask > (0.05 * full_mask.max())
        coordinates = selected.nonzero(as_tuple=False)
        if len(coordinates) == 0:
            return None
        y0, x0 = coordinates.amin(dim=0).tolist()
        y1, x1 = coordinates.amax(dim=0).tolist()
        y1 += 1
        x1 += 1
        box_h = max(y1 - y0, 1)
        box_w = max(x1 - x0, 1)
        y_pad = max(int(round(0.15 * box_h)), 1)
        x_pad = max(int(round(0.15 * box_w)), 1)
        y0 = max(y0 - y_pad, 0)
        y1 = min(y1 + y_pad, height)
        x0 = max(x0 - x_pad, 0)
        x1 = min(x1 + x_pad, width)

        crop = image[:, y0:y1, x0:x1][None]
        crop_mask = full_mask[y0:y1, x0:x1][None, None]
        scale = min(224.0 / crop.shape[-2], 224.0 / crop.shape[-1])
        resized_h = max(int(round(crop.shape[-2] * scale)), 1)
        resized_w = max(int(round(crop.shape[-1] * scale)), 1)
        crop = F.interpolate(
            crop,
            size=(resized_h, resized_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )[0].clamp(0.0, 1.0)
        crop_mask = F.interpolate(
            crop_mask,
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        canvas = self.mean.expand(3, 224, 224).clone()
        support = torch.zeros(224, 224, dtype=torch.bool)
        top = (224 - resized_h) // 2
        left = (224 - resized_w) // 2
        canvas[:, top:top + resized_h, left:left + resized_w] = crop
        crop_support = crop_mask > (0.05 * crop_mask.max())
        support[top:top + resized_h, left:left + resized_w] = crop_support
        if not bool(support.any()):
            raise RuntimeError("Pose-valid slot has empty crop support")
        return canvas, support

    def _nested_occlusions(self, canvas, support):
        coordinates = support.nonzero(as_tuple=False)
        count = int(len(coordinates))
        variants = []
        realized = []
        target_error = []
        for level in LEVELS:
            erase_count = int(round(float(level) * count))
            variant = canvas.clone()
            if erase_count > 0:
                chosen = coordinates[:erase_count]
                variant[:, chosen[:, 0], chosen[:, 1]] = self.mean[:, 0, 0, None]
            variants.append(variant)
            actual = erase_count / float(count)
            realized.append(actual)
            target_error.append(abs(actual - float(level)))
        return variants, realized, target_error, count

    @torch.no_grad()
    def _encode(self, images):
        normalized = (images - self.mean[None]) / self.std[None]
        encoded = []
        for start in range(0, len(normalized), self.clip_batch):
            batch = normalized[start:start + self.clip_batch].to(self.device)
            encoded.append(
                F.normalize(self.model.encode_image(batch).float(), dim=-1).cpu()
            )
        return torch.cat(encoded, dim=0)

    @torch.no_grad()
    def encode_batch(self, images, masks, valid, repeat_check):
        owners = masks.argmax(dim=1)
        variants = []
        regions = []
        realized = []
        target_error = []
        support_pixels = []
        for image_index in range(images.shape[0]):
            for region in range(REGIONS):
                if not bool(valid[image_index, region]):
                    continue
                hard_owner = masks[image_index, region] * (
                    owners[image_index] == region
                ).float()
                item = self._crop_image_and_support(
                    images[image_index], hard_owner
                )
                if item is None:
                    raise RuntimeError("Pose-valid slot has no crop")
                crop, support = item
                level_images, level_realized, level_error, pixel_count = (
                    self._nested_occlusions(crop, support)
                )
                variants.extend(level_images)
                regions.append(region)
                realized.append(level_realized)
                target_error.append(level_error)
                support_pixels.append(pixel_count)

        stacked = torch.stack(variants)
        features = self._encode(stacked)
        repeat_exact = True
        if repeat_check:
            check_count = min(self.clip_batch, len(stacked))
            repeated = self._encode(stacked[:check_count])
            repeat_exact = bool(torch.equal(features[:check_count], repeated))
        features = features.view(len(regions), len(LEVELS), -1)
        region_index = torch.as_tensor(regions, dtype=torch.long)
        text = self.text.index_select(0, region_index)
        logits = torch.einsum("nld,nsd->nls", features, text)
        q_visible = torch.softmax(logits / TEMPERATURE, dim=-1)[..., 0]
        return {
            "q_visible": q_visible,
            "regions": region_index,
            "realized": torch.as_tensor(realized, dtype=torch.float64),
            "target_error": torch.as_tensor(target_error, dtype=torch.float64),
            "support_pixels": torch.as_tensor(support_pixels, dtype=torch.long),
            "repeat_exact": repeat_exact,
        }


def summarize(q_visible, regions, realized, pids, repeats, seed):
    from scipy.stats import spearmanr

    levels = list(LEVELS)
    per_region = []
    macro_by_level = []
    for level_index in range(len(levels)):
        class_means = []
        for region in range(REGIONS):
            keep = regions == region
            class_means.append(float(q_visible[keep, level_index].mean()))
        macro_by_level.append(float(np.mean(class_means)))

    for region in range(REGIONS):
        keep = regions == region
        values = q_visible[keep].numpy()
        overlaps = realized[keep].numpy()
        region_pids = pids[keep].numpy()
        correlation = float(
            spearmanr(overlaps.reshape(-1), -values.reshape(-1)).statistic
        )
        delta = values[:, 0] - values[:, -1]
        per_region.append({
            "region": int(region),
            "samples": int(keep.sum()),
            "q_visible_mean_by_level": [
                float(values[:, index].mean())
                for index in range(len(levels))
            ],
            "spearman_overlap_negative_q_visible": correlation,
            "q_visible_level0_minus_level75": clustered_mean(
                delta, region_pids, repeats, seed + region + 1
            ),
        })
    adjacent_delta = [
        macro_by_level[index] - macro_by_level[index + 1]
        for index in range(len(levels) - 1)
    ]
    return {
        "levels": levels,
        "macro_q_visible_by_level": macro_by_level,
        "macro_adjacent_delta": adjacent_delta,
        "per_region": per_region,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--phase0b-script", required=True)
    parser.add_argument("--ontology-script", required=True)
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
    data_root = Path(args.data_root).resolve()
    pose_artifact = Path(args.pose_artifact).resolve()
    clip_checkpoint = Path(args.clip_checkpoint).resolve()

    if sha256_file(ontology_script) != EXPECTED_ONTOLOGY_SCRIPT_SHA256:
        raise RuntimeError("Ontology audit script SHA mismatch")
    ontology = load_module("exp392_phase0b2_ontology", ontology_script)
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
    teacher = FrozenSupportCropTeacher(
        ontology, clip_checkpoint, device, args.clip_batch
    )

    q_all = []
    region_all = []
    realized_all = []
    error_all = []
    pixels_all = []
    pid_all = []
    original_valid_count = 0
    repeat_exact = True
    manifest_digest = hashlib.sha256()
    for batch_index, batch in enumerate(loader):
        for relative_path, image_sha in zip(
            batch["relative_path"], batch["image_sha256"]
        ):
            manifest_digest.update(relative_path.encode("utf-8"))
            manifest_digest.update(b"\0")
            manifest_digest.update(image_sha.encode("ascii"))
        encoded = teacher.encode_batch(
            batch["pre"],
            batch["masks"],
            batch["valid"].bool(),
            repeat_check=batch_index == 0,
        )
        q_all.append(encoded["q_visible"])
        region_all.append(encoded["regions"])
        realized_all.append(encoded["realized"])
        error_all.append(encoded["target_error"])
        pixels_all.append(encoded["support_pixels"])
        pid_all.append(
            batch["pid"].repeat_interleave(
                batch["valid"].bool().sum(dim=1)
            )
        )
        original_valid_count += int(batch["valid"].sum())
        repeat_exact = repeat_exact and encoded["repeat_exact"]

    q_visible = torch.cat(q_all, dim=0)
    regions = torch.cat(region_all, dim=0)
    realized = torch.cat(realized_all, dim=0)
    target_error = torch.cat(error_all, dim=0)
    support_pixels = torch.cat(pixels_all, dim=0)
    pids = torch.cat(pid_all, dim=0)
    if len(q_visible) != original_valid_count:
        raise RuntimeError("Crop construction changed valid coverage")
    summary = summarize(
        q_visible,
        regions,
        realized,
        pids,
        args.bootstrap_repeats,
        args.seed,
    )
    finite = bool(
        torch.isfinite(q_visible).all()
        and torch.isfinite(realized).all()
        and torch.isfinite(target_error).all()
    )
    max_allowed_error = float((1.0 / support_pixels.double()).max())
    max_target_error = float(target_error.max())
    gates = {
        "coverage_exact": len(q_visible) == original_valid_count,
        "finite": finite,
        "repeat_exact": bool(repeat_exact),
        "level_target_error_within_one_pixel": (
            max_target_error <= max_allowed_error + 1e-12
        ),
        "realized_overlap_strictly_increasing": bool(
            (realized[:, 1:] > realized[:, :-1]).all()
        ),
        "all_region_positive_spearman": all(
            item["spearman_overlap_negative_q_visible"] > 0.0
            for item in summary["per_region"]
        ),
        "all_region_positive_level0_minus_level75": all(
            item["q_visible_level0_minus_level75"]["mean"] > 0.0
            for item in summary["per_region"]
        ),
        "macro_all_adjacent_monotonic": all(
            value > 0.0 for value in summary["macro_adjacent_delta"]
        ),
    }
    full_audit = args.max_samples <= 0
    verdict = (
        "B2_SC_SMOKE_PASS"
        if not full_audit and all(gates.values())
        else "B2_SC_SMOKE_FAIL"
        if not full_audit
        else "B2_SC_FULL_REQUIRES_SEPARATE_PROTOCOL"
    )
    result = {
        "status": "EXP392_PHASE0B2_SC_COMPLETE",
        "scope": "full" if full_audit else "smoke",
        "verdict": verdict,
        "formal_training_authorized": False,
        "phase0b2_si_authorized": verdict == "B2_SC_SMOKE_PASS",
        "execution": {
            "repo_root": str(repo_root),
            "source_commit": args.source_commit,
            "phase0b_script": str(phase0b_script),
            "phase0b_script_sha256": sha256_file(phase0b_script),
            "ontology_script": str(ontology_script),
            "ontology_script_sha256": sha256_file(ontology_script),
            "audit_script_sha256": sha256_file(audit_script_path),
            "runtime_sha256": runtime_sha,
            "data_root": str(data_root),
            "pose_artifact": str(pose_artifact),
            "pose_manifest_sha256": base.EXPECTED_POSE_MANIFEST_SHA256,
            "clip_checkpoint": str(clip_checkpoint),
            "clip_checkpoint_sha256": base.EXPECTED_CLIP_SHA256,
            "device": str(device),
            "torch_version": torch.__version__,
            "open_clip_version": __import__("open_clip").__version__,
            "images": len(records),
            "valid_slot_crops": len(q_visible),
            "encoded_crop_variants": len(q_visible) * len(LEVELS),
            "batch_size": args.batch_size,
            "clip_batch": args.clip_batch,
            "workers": args.workers,
            "seed": args.seed,
            "sample_manifest_sha256": manifest_digest.hexdigest(),
        },
        "ontology": {
            "partition_mode": "hard-owner",
            "static_contract": contract,
            "overlap_levels": list(LEVELS),
            "occlusion_order": "row-major nested support pixels",
            "replacement": "CLIP pixel mean",
            "crop_bbox_recomputed_after_occlusion": False,
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
        "valid_slot_crops": len(q_visible),
        "encoded_crop_variants": len(q_visible) * len(LEVELS),
        "macro_q_visible_by_level": summary["macro_q_visible_by_level"],
        "macro_adjacent_delta": summary["macro_adjacent_delta"],
        "per_region": summary["per_region"],
        "gates": gates,
        "output_sha256": sha256_file(args.output),
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if verdict == "B2_SC_SMOKE_PASS" else 1)


if __name__ == "__main__":
    main()
