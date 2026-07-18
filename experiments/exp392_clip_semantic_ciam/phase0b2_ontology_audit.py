#!/usr/bin/env python3
"""Phase 0B2-O ontology-only audit with frozen crop-global CLIP.

This audit changes the five-region ontology while keeping the diagnostic task
as five body-part names.  It never builds a ReID model or optimizer.
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
REGION_NAMES = (
    "head_face",
    "central_torso",
    "arms_hands",
    "upper_legs_thighs",
    "lower_legs_feet",
)
REGION_JOINTS = (
    (0, 1, 2, 3, 4),
    (5, 6, 11, 12),
    (7, 8, 9, 10),
    (13, 14),
    (15, 16),
)
REGION_SEGMENTS = (
    ((0, 1), (0, 2), (1, 3), (2, 4)),
    ((5, 6), (5, 11), (6, 12), (11, 12)),
    ((5, 7), (7, 9), (6, 8), (8, 10)),
    ((11, 13), (12, 14)),
    ((13, 15), (14, 16)),
)
REGION_PHRASES = (
    "head and face",
    "torso and upper body",
    "arms and hands",
    "upper legs and thighs",
    "lower legs and feet",
)
PROMPT_TEMPLATES = (
    "a photo of the {} of a person",
    "the {} region of a pedestrian",
    "a close view of a person's {}",
    "human {}",
)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
EXPECTED_PHASE0B_SCRIPT_SHA256 = (
    "03b8f707bc6f189dd3de34505af82e63f7ee71bd23d70b6e9663aee318afcd70"
)
SEGMENT_INTERIOR_FRACTION = 0.15


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


def load_phase0b_module(path):
    spec = importlib.util.spec_from_file_location("exp392_phase0b", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExclusiveRegionRenderer:
    def __init__(self, base_module):
        self.base = base_module.RegionRenderer(height=96, width=32, sigma=1.5)

    def _segment_map(self, keypoints, reliability, segments, interior):
        if not segments:
            return torch.zeros(
                self.base.height, self.base.width, dtype=torch.float32
            )
        result = torch.zeros(
            self.base.height, self.base.width, dtype=torch.float32
        )
        px = self.base.grid_x[0]
        py = self.base.grid_y[0]
        for left, right in segments:
            ax = keypoints[left, 0] * (self.base.width - 1) / 127.0
            ay = keypoints[left, 1] * (self.base.height - 1) / 383.0
            bx = keypoints[right, 0] * (self.base.width - 1) / 127.0
            by = keypoints[right, 1] * (self.base.height - 1) / 383.0
            dx = bx - ax
            dy = by - ay
            denominator = (dx * dx + dy * dy).clamp_min(1e-6)
            projection = ((px - ax) * dx + (py - ay) * dy) / denominator
            if interior:
                projection = projection.clamp(
                    SEGMENT_INTERIOR_FRACTION,
                    1.0 - SEGMENT_INTERIOR_FRACTION,
                )
            else:
                projection = projection.clamp(0.0, 1.0)
            nearest_x = ax + projection * dx
            nearest_y = ay + projection * dy
            distance = (
                (px - nearest_x).square() + (py - nearest_y).square()
            )
            amplitude = torch.minimum(reliability[left], reliability[right])
            tube = torch.exp(
                -distance / (2.0 * self.base.sigma ** 2)
            ) * amplitude
            result = torch.maximum(result, tube)
        return result

    def __call__(self, keypoints, scores, valid):
        reliability = valid.float() * scores.float().clamp(0.0, 1.0)
        joints = self.base._joint_maps(keypoints.float(), reliability)
        raw = []
        confidence = []
        region_valid = []
        for region, (joint_ids, segments) in enumerate(
            zip(REGION_JOINTS, REGION_SEGMENTS)
        ):
            index = torch.as_tensor(joint_ids, dtype=torch.long)
            joint_mask = joints.index_select(0, index).amax(0)
            segment_mask = self._segment_map(
                keypoints.float(),
                reliability,
                segments,
                interior=region >= 2,
            )
            raw.append(torch.maximum(joint_mask, segment_mask))
            values = reliability.index_select(0, index)
            confidence.append(values.mean())
            region_valid.append(values.max() > 0)
        raw = torch.stack(raw, dim=0)
        total = raw.sum(dim=0, keepdim=True)
        masks = raw / total.clamp_min(1.0)
        return (
            masks,
            torch.stack(confidence, dim=0),
            torch.stack(region_valid, dim=0),
        )


def ontology_static_contract(base_module):
    renderer = ExclusiveRegionRenderer(base_module)
    keypoints = torch.tensor([
        [64, 40], [58, 36], [70, 36], [52, 40], [76, 40],
        [44, 100], [84, 100], [30, 160], [98, 160], [20, 220], [108, 220],
        [50, 220], [78, 220], [50, 300], [78, 300], [50, 370], [78, 370],
    ], dtype=torch.float32)
    scores = torch.ones(17)
    valid = torch.ones(17, dtype=torch.bool)
    masks, _, region_valid = renderer(keypoints, scores, valid)

    empty_masks, _, empty_valid = renderer(
        keypoints, torch.zeros(17), torch.zeros(17, dtype=torch.bool)
    )
    flip_index = torch.as_tensor(base_module.COCO17_FLIP, dtype=torch.long)
    flipped_keypoints = keypoints.clone()
    flipped_keypoints[:, 0] = 127.0 - flipped_keypoints[:, 0]
    flipped_keypoints = flipped_keypoints.index_select(0, flip_index)
    flipped_masks, _, flipped_valid = renderer(
        flipped_keypoints,
        scores.index_select(0, flip_index),
        valid.index_select(0, flip_index),
    )

    joint_owner = {}
    owner_counts = torch.zeros(17, dtype=torch.int64)
    onehot_exact = True
    for region, joint_ids in enumerate(REGION_JOINTS):
        for joint in joint_ids:
            owner_counts[joint] += 1
            joint_owner[joint] = region
    for joint in range(17):
        one_valid = torch.zeros(17, dtype=torch.bool)
        one_valid[joint] = True
        one_masks, _, _ = renderer(keypoints, scores, one_valid)
        active_regions = one_masks.flatten(1).sum(1) > 1e-8
        expected = torch.zeros(REGIONS, dtype=torch.bool)
        expected[joint_owner[joint]] = True
        onehot_exact = onehot_exact and bool(torch.equal(active_regions, expected))

    boundary_checks = []
    for joint, owner, competitor in ((5, 1, 2), (11, 1, 3), (13, 3, 4)):
        x = int(round(float(keypoints[joint, 0]) * 31.0 / 127.0))
        y = int(round(float(keypoints[joint, 1]) * 95.0 / 383.0))
        boundary_checks.append(
            float(masks[owner, y, x]) > float(masks[competitor, y, x])
        )
    segment_path_nonempty = []
    segment_path_finite = []
    segment_single_endpoint_zero = []
    trimmed_midpoint_beats_endpoint = []
    for region, segments in enumerate(REGION_SEGMENTS):
        left, right = segments[0]
        reliability = torch.zeros(17)
        reliability[left] = 1.0
        reliability[right] = 1.0
        direct = renderer._segment_map(
            keypoints,
            reliability,
            (segments[0],),
            interior=region >= 2,
        )
        segment_path_nonempty.append(float(direct.sum()) > 0.0)
        segment_path_finite.append(bool(torch.isfinite(direct).all()))
        one_endpoint = reliability.clone()
        one_endpoint[right] = 0.0
        direct_one_endpoint = renderer._segment_map(
            keypoints,
            one_endpoint,
            (segments[0],),
            interior=region >= 2,
        )
        segment_single_endpoint_zero.append(
            bool(torch.equal(direct_one_endpoint, torch.zeros_like(direct_one_endpoint)))
        )
        if region >= 2:
            midpoint = 0.5 * (keypoints[left] + keypoints[right])
            midpoint_x = int(round(float(midpoint[0]) * 31.0 / 127.0))
            midpoint_y = int(round(float(midpoint[1]) * 95.0 / 383.0))
            endpoint_x = int(round(float(keypoints[left, 0]) * 31.0 / 127.0))
            endpoint_y = int(round(float(keypoints[left, 1]) * 95.0 / 383.0))
            trimmed_midpoint_beats_endpoint.append(
                float(direct[midpoint_y, midpoint_x])
                > float(direct[endpoint_y, endpoint_x])
            )

    checks = {
        "joint_owner_count_exact": bool((owner_counts == 1).all()),
        "onehot_joint_region_exact": bool(onehot_exact),
        "empty_mask_exact": bool(torch.equal(empty_masks, torch.zeros_like(empty_masks))),
        "empty_valid_exact": bool(torch.equal(empty_valid, torch.zeros_like(empty_valid))),
        "all_regions_valid": bool(region_valid.all()),
        "flip_valid_exact": bool(torch.equal(flipped_valid, region_valid)),
        "flip_mask_max_abs": float(
            (flipped_masks - masks.flip(-1)).abs().max()
        ),
        "sum_region_support_max": float(masks.sum(0).max()),
        "boundary_owner_dominance": bool(all(boundary_checks)),
        "segment_path_nonempty": bool(all(segment_path_nonempty)),
        "segment_path_finite": bool(all(segment_path_finite)),
        "segment_single_endpoint_zero": bool(all(segment_single_endpoint_zero)),
        "trimmed_midpoint_beats_endpoint": bool(
            all(trimmed_midpoint_beats_endpoint)
        ),
        "finite": bool(torch.isfinite(masks).all()),
    }
    gates = {
        "unique_joint_owner": checks["joint_owner_count_exact"],
        "onehot_joint": checks["onehot_joint_region_exact"],
        "empty": checks["empty_mask_exact"] and checks["empty_valid_exact"],
        "flip": (
            checks["flip_valid_exact"] and checks["flip_mask_max_abs"] <= 1e-6
        ),
        "bounded_partition": checks["sum_region_support_max"] <= 1.0 + 1e-6,
        "boundary_owner": checks["boundary_owner_dominance"],
        "segment_path": (
            checks["segment_path_nonempty"]
            and checks["segment_path_finite"]
            and checks["segment_single_endpoint_zero"]
            and checks["trimmed_midpoint_beats_endpoint"]
        ),
        "finite": checks["finite"],
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "checks": checks,
        "gates": gates,
        "segment_interior_fraction": SEGMENT_INTERIOR_FRACTION,
    }


class OntologyTransform:
    def __init__(self, base_module, seed):
        self.base = base_module.AuditTransform(seed=seed)
        self.base.renderer = ExclusiveRegionRenderer(base_module)

    def __call__(self, image, pose):
        if tuple(image.size) != tuple(pose.image_size):
            raise RuntimeError("RGB size does not match pose artifact")
        return self.base(image, pose)


class RecipientDataset(torch.utils.data.Dataset):
    def __init__(self, base_module, records, pose_store, seed, verify_sha):
        self.base_module = base_module
        self.records = list(records)
        self.pose_store = pose_store
        self.transform = OntologyTransform(base_module, seed)
        self.verify_sha = bool(verify_sha)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from datasets.bases import read_image

        image_path, pid, camid, trackid = self.records[index]
        del trackid
        image = read_image(image_path)
        pose = self.pose_store.get(
            image_path, verify_image_sha=self.verify_sha
        )
        item = self.transform(image, pose)
        item.update({
            "pid": int(pid),
            "camid": int(camid),
            "relative_path": pose.relative_path,
            "image_sha256": pose.image_sha256,
        })
        return item


def collate_recipient(items):
    return {
        "pre": torch.stack([item["pre"] for item in items]),
        "masks": torch.stack([item["masks"] for item in items]),
        "confidence": torch.stack([item["confidence"] for item in items]),
        "valid": torch.stack([item["valid"] for item in items]),
        "pid": torch.tensor([item["pid"] for item in items]),
        "camid": torch.tensor([item["camid"] for item in items]),
        "relative_path": tuple(item["relative_path"] for item in items),
        "image_sha256": tuple(item["image_sha256"] for item in items),
    }


class FrozenCropTeacher:
    def __init__(self, checkpoint, device, clip_batch):
        import open_clip

        self.device = device
        self.clip_batch = int(clip_batch)
        self.model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=str(checkpoint)
        )
        self.model = self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        prompts = [
            template.format(phrase)
            for phrase in REGION_PHRASES
            for template in PROMPT_TEMPLATES
        ]
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            text = self.model.encode_text(tokens)
        text = F.normalize(text.float(), dim=-1)
        text = text.view(len(REGION_NAMES), len(PROMPT_TEMPLATES), -1)
        self.text = F.normalize(text.mean(1), dim=-1)
        self.prompt_payload = {
            "region_names": list(REGION_NAMES),
            "region_phrases": list(REGION_PHRASES),
            "templates": list(PROMPT_TEMPLATES),
            "prompts": prompts,
        }
        self.preprocess_repr = repr(preprocess)
        normalizers = [
            transform for transform in preprocess.transforms
            if hasattr(transform, "mean") and hasattr(transform, "std")
        ]
        if len(normalizers) != 1:
            raise RuntimeError("Could not identify official CLIP Normalize")
        official_mean = tuple(float(value) for value in normalizers[0].mean)
        official_std = tuple(float(value) for value in normalizers[0].std)
        if not np.allclose(official_mean, CLIP_MEAN, atol=1e-8, rtol=0.0):
            raise RuntimeError("CLIP mean does not match official preprocess")
        if not np.allclose(official_std, CLIP_STD, atol=1e-8, rtol=0.0):
            raise RuntimeError("CLIP std does not match official preprocess")
        if "BICUBIC" not in self.preprocess_repr.upper():
            raise RuntimeError("Official CLIP preprocess is not bicubic")
        self.preprocess_contract = {
            "mean": list(official_mean),
            "std": list(official_std),
            "bicubic_in_repr": True,
            "manual_geometry_reason": (
                "reuse official interpolation/normalization without independent "
                "center crop so RGB and pose mask share one geometry"
            ),
        }

    def _crop_one(self, image, mask):
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
        mean = torch.as_tensor(CLIP_MEAN, dtype=crop.dtype).view(3, 1, 1)
        canvas = mean.expand(3, 224, 224).clone()
        top = (224 - resized_h) // 2
        left = (224 - resized_w) // 2
        canvas[:, top:top + resized_h, left:left + resized_w] = crop
        return canvas

    @torch.no_grad()
    def encode(self, images, masks, valid):
        crops = []
        indices = []
        owners = masks.argmax(dim=1)
        for image_index in range(images.shape[0]):
            for region in range(REGIONS):
                if not bool(valid[image_index, region]):
                    continue
                hard_owner = masks[image_index, region] * (
                    owners[image_index] == region
                ).float()
                crop = self._crop_one(images[image_index], hard_owner)
                if crop is None:
                    raise RuntimeError(
                        "Pose-valid slot has no hard-owner crop: image=%d region=%d"
                        % (image_index, region)
                    )
                crops.append(crop)
                indices.append((image_index, region))
        features = torch.zeros(
            images.shape[0], REGIONS, self.text.shape[-1], dtype=torch.float32
        )
        if crops:
            mean = torch.as_tensor(CLIP_MEAN).view(1, 3, 1, 1)
            std = torch.as_tensor(CLIP_STD).view(1, 3, 1, 1)
            crops = (torch.stack(crops) - mean) / std
            encoded = []
            for start in range(0, len(crops), self.clip_batch):
                batch = crops[start:start + self.clip_batch].to(self.device)
                encoded.append(
                    F.normalize(self.model.encode_image(batch).float(), dim=-1).cpu()
                )
            encoded = torch.cat(encoded, dim=0)
            for value, (image_index, region) in zip(encoded, indices):
                features[image_index, region] = value
        effective_valid = valid.bool() & (features.norm(dim=-1) > 0)
        if not bool(torch.equal(effective_valid, valid.bool())):
            raise RuntimeError("Crop construction changed pose-valid coverage")
        logits = torch.einsum("bkd,cd->bkc", features, self.text.cpu())
        return features, logits, effective_valid, len(crops)


def pairwise_overlap(masks, valid):
    flattened = masks.flatten(2).double()
    values = []
    for left in range(REGIONS):
        for right in range(left + 1, REGIONS):
            keep = valid[:, left] & valid[:, right]
            if not bool(keep.any()):
                continue
            intersection = torch.minimum(
                flattened[:, left], flattened[:, right]
            ).sum(-1)
            denominator = torch.minimum(
                flattened[:, left].sum(-1), flattened[:, right].sum(-1)
            ).clamp_min(1e-12)
            values.append((intersection / denominator)[keep])
    return torch.cat(values) if values else torch.zeros(0, dtype=torch.double)


def clustered_bootstrap(values, pids, repeats, seed):
    values = np.asarray(values, dtype=np.float64)
    pids = np.asarray(pids, dtype=np.int64)
    if len(values) == 0:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "pids": 0,
            "images": 0,
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
        "pids": int(len(per_pid)),
        "images": int(len(values)),
        "repeats": int(repeats),
    }


def clustered_macro_bootstrap(values, valid, pids, repeats, seed):
    """PID-cluster bootstrap with equal weight for each anatomical class."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    pids = np.asarray(pids, dtype=np.int64)
    unique = np.unique(pids)
    per_pid_class = np.full((len(unique), REGIONS), np.nan, dtype=np.float64)
    for pid_index, pid in enumerate(unique):
        pid_rows = pids == pid
        for region in range(REGIONS):
            keep = pid_rows & valid[:, region]
            if bool(keep.any()):
                per_pid_class[pid_index, region] = values[keep, region].mean()

    def macro(matrix):
        class_means = []
        for region in range(REGIONS):
            selected = matrix[:, region]
            selected = selected[np.isfinite(selected)]
            if len(selected) == 0:
                return None
            class_means.append(selected.mean())
        return float(np.mean(class_means))

    point = macro(per_pid_class)
    if point is None:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "pids": int(len(unique)),
            "images": int(len(values)),
            "repeats": int(repeats),
        }
    rng = np.random.RandomState(seed)
    draws = []
    for _ in range(repeats):
        sample = rng.randint(0, len(unique), size=len(unique))
        value = macro(per_pid_class[sample])
        if value is not None:
            draws.append(value)
    draws = np.asarray(draws, dtype=np.float64)
    return {
        "mean": point,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "pids": int(len(unique)),
        "images": int(len(values)),
        "repeats": int(len(draws)),
    }


def lower_gt(summary, threshold):
    return (
        summary["ci95_low"] is not None
        and summary["ci95_low"] > float(threshold)
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--phase0b-script", required=True)
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
    data_root = Path(args.data_root).resolve()
    pose_artifact = Path(args.pose_artifact).resolve()
    clip_checkpoint = Path(args.clip_checkpoint).resolve()
    if sha256_file(phase0b_script) != EXPECTED_PHASE0B_SCRIPT_SHA256:
        raise RuntimeError("Phase 0B audit script SHA mismatch")
    base = load_phase0b_module(phase0b_script)
    ontology_contract = ontology_static_contract(base)
    if ontology_contract["status"] != "PASS":
        raise RuntimeError(
            "Exclusive ontology static contract failed: %s"
            % json.dumps(ontology_contract, sort_keys=True)
        )
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
    audit_dataset = RecipientDataset(
        base, records, pose_store, args.seed, verify_sha=True
    )
    loader = torch.utils.data.DataLoader(
        audit_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_recipient,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )
    device = torch.device(args.device)
    teacher = FrozenCropTeacher(clip_checkpoint, device, args.clip_batch)
    logits_all = []
    valid_all = []
    original_valid_all = []
    pid_all = []
    overlap_all = []
    crop_count = 0
    manifest_digest = hashlib.sha256()
    for batch in loader:
        for relative_path, image_sha in zip(
            batch["relative_path"], batch["image_sha256"]
        ):
            manifest_digest.update(relative_path.encode("utf-8"))
            manifest_digest.update(b"\0")
            manifest_digest.update(image_sha.encode("ascii"))
        features, logits, effective_valid, count = teacher.encode(
            batch["pre"], batch["masks"], batch["valid"]
        )
        del features
        logits_all.append(logits)
        valid_all.append(effective_valid)
        original_valid_all.append(batch["valid"].bool())
        pid_all.append(batch["pid"])
        overlap_all.append(pairwise_overlap(batch["masks"], batch["valid"].bool()))
        crop_count += count
    logits = torch.cat(logits_all, dim=0)
    valid = torch.cat(valid_all, dim=0)
    original_valid = torch.cat(original_valid_all, dim=0)
    pids = torch.cat(pid_all, dim=0)
    overlaps = torch.cat(overlap_all, dim=0)
    labels = torch.arange(REGIONS).view(1, REGIONS)
    top1 = (logits.argmax(-1) == labels).double()
    expected = logits.gather(-1, labels[..., None].expand(len(logits), -1, 1)).squeeze(-1)
    alternative = logits.clone()
    alternative.scatter_(-1, labels[..., None].expand(len(logits), -1, 1), -float("inf"))
    margin = expected - alternative.amax(-1)
    summary = {
        "macro_top1": clustered_macro_bootstrap(
            top1.numpy(),
            valid.numpy(),
            pids.numpy(),
            args.bootstrap_repeats,
            args.seed,
        ),
        "raw_cosine_margin": clustered_macro_bootstrap(
            margin.numpy(),
            valid.numpy(),
            pids.numpy(),
            args.bootstrap_repeats,
            args.seed,
        ),
        "per_class": [],
    }
    for region in range(REGIONS):
        keep = valid[:, region]
        summary["per_class"].append({
            "region": REGION_NAMES[region],
            "original_valid_images": int(original_valid[:, region].sum()),
            "valid_images": int(keep.sum()),
            "coverage_drop": int(
                original_valid[:, region].sum() - keep.sum()
            ),
            "coverage_fraction": float(keep.double().mean()),
            "top1": clustered_bootstrap(
                top1[:, region][keep].numpy(),
                pids[keep].numpy(),
                args.bootstrap_repeats,
                args.seed + region + 1,
            ),
            "raw_cosine_margin": clustered_bootstrap(
                margin[:, region][keep].numpy(),
                pids[keep].numpy(),
                args.bootstrap_repeats,
                args.seed + region + 11,
            ),
        })
    full_audit = args.max_samples <= 0
    overlap_available = len(overlaps) > 0
    overlap_median = float(overlaps.median()) if overlap_available else None
    overlap_p95 = (
        float(torch.quantile(overlaps, 0.95)) if overlap_available else None
    )
    overlap_max = float(overlaps.max()) if overlap_available else None
    gates = {
        "macro_top1_lower_ge_035": lower_gt(summary["macro_top1"], 0.35 - 1e-12),
        "all_class_top1_lower_gt_020": all(
            lower_gt(item["top1"], 0.20) for item in summary["per_class"]
        ),
        "all_class_raw_margin_lower_gt_0": all(
            lower_gt(item["raw_cosine_margin"], 0.0)
            for item in summary["per_class"]
        ),
        "overlap_median_lt_010": (
            overlap_available and overlap_median < 0.10
        ),
        "overlap_p95_lt_025": overlap_available and overlap_p95 < 0.25,
        "every_class_has_samples": bool(valid.any(dim=0).all()),
        "coverage_exact": bool(torch.equal(valid, original_valid)),
        "finite": bool(torch.isfinite(logits).all()),
    }
    verdict = (
        "B2_O_SMOKE_PASS"
        if not full_audit
        and gates["every_class_has_samples"]
        and gates["coverage_exact"]
        and gates["finite"]
        and gates["overlap_median_lt_010"]
        and gates["overlap_p95_lt_025"]
        else "B2_O_GATE_PASS"
        if full_audit and all(gates.values())
        else "B2_O_GATE_FAIL"
        if full_audit
        else "B2_O_SMOKE_FAIL"
    )
    result = {
        "status": "EXP392_PHASE0B2_O_COMPLETE",
        "scope": "full" if full_audit else "smoke",
        "verdict": verdict,
        "formal_training_authorized": False,
        "phase0b2_i_authorized": full_audit and all(gates.values()),
        "execution": {
            "repo_root": str(repo_root),
            "source_commit": args.source_commit,
            "phase0b_script": str(phase0b_script),
            "phase0b_script_sha256": sha256_file(phase0b_script),
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
            "samples": len(records),
            "crops": crop_count,
            "batch_size": args.batch_size,
            "clip_batch": args.clip_batch,
            "workers": args.workers,
            "seed": args.seed,
            "sample_manifest_sha256": manifest_digest.hexdigest(),
        },
        "ontology": {
            "regions": list(REGION_NAMES),
            "region_joints": [list(value) for value in REGION_JOINTS],
            "region_segments": [
                [list(pair) for pair in value] for value in REGION_SEGMENTS
            ],
            "renderer": (
                "raw Gaussian/tube supports, unique joint owners, trimmed limb "
                "segments, amplitude-preserving cross-region partition"
            ),
            "static_contract": ontology_contract,
            "segment_interior_fraction": SEGMENT_INTERIOR_FRACTION,
            "crop_context_fraction": 0.15,
            "crop_path": "tight bbox -> aspect-preserving bicubic -> mean letterbox",
        },
        "prompt_payload": teacher.prompt_payload,
        "prompt_sha256": sha256_json(teacher.prompt_payload),
        "open_clip_preprocess_repr": teacher.preprocess_repr,
        "open_clip_preprocess_contract": teacher.preprocess_contract,
        "mask_overlap": {
            "count": int(len(overlaps)),
            "median": overlap_median,
            "p95": overlap_p95,
            "max": overlap_max,
        },
        "summary": summary,
        "gates": gates,
    }
    write_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "scope": result["scope"],
        "verdict": result["verdict"],
        "samples": len(records),
        "crops": crop_count,
        "macro_top1": summary["macro_top1"],
        "raw_cosine_margin": summary["raw_cosine_margin"],
        "mask_overlap": result["mask_overlap"],
        "output_sha256": sha256_file(args.output),
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if verdict in ("B2_O_SMOKE_PASS", "B2_O_GATE_PASS") else 1)


if __name__ == "__main__":
    main()
