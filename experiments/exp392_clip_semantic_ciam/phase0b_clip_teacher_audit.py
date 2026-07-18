#!/usr/bin/env python3
"""Teacher-only audit for the exp392 frozen CLIP dual-encoder proposal.

This script reads the official Occluded-Duke train RGB and the exp386 fresh
train-only pose artifact.  It never builds a ReID model or optimizer and never
writes query/gallery pose data.
"""

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from timm.data.random_erasing import RandomErasing
from torchvision.transforms import InterpolationMode


EXPECTED_POSE_MANIFEST_SHA256 = (
    "cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8"
)
EXPECTED_CLIP_SHA256 = (
    "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e"
)
EXPECTED_RUNTIME_SHA256 = {
    "datasets/bases.py": (
        "03d231558f46264e4cff0c251b9b728ab4971232ed6c4bb7324ce1964f139c2c"
    ),
    "datasets/occluded_duke.py": (
        "f0e7b25e75251643430b699d9c9969fae207c0a85c48855cd0404d61a4228f8e"
    ),
    "datasets/pose_targets.py": (
        "42f6e35eff2ad572445143cb3ecc5b6a22d856facc4453b989411300dec22624"
    ),
}
COCO17_FLIP = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9,
               12, 11, 14, 13, 16, 15)
REGION_NAMES = (
    "head_face",
    "torso",
    "arms_hands",
    "upper_legs",
    "lower_legs_feet",
)
REGION_JOINTS = (
    (0, 1, 2, 3, 4),
    (5, 6, 11, 12),
    (5, 6, 7, 8, 9, 10),
    (11, 12, 13, 14),
    (13, 14, 15, 16),
)
REGION_SEGMENTS = (
    ((0, 1), (0, 2), (1, 3), (2, 4)),
    ((5, 6), (5, 11), (6, 12), (11, 12)),
    ((5, 7), (7, 9), (6, 8), (8, 10)),
    ((11, 13), (12, 14), (11, 12)),
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
CHANNEL_CYCLE = (1, 2, 3, 4, 0)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


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


def seed_from_text(text, base_seed):
    digest = hashlib.sha256((str(base_seed) + "\0" + text).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2 ** 31)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RegionRenderer:
    def __init__(self, height=96, width=32, sigma=1.5):
        self.height = int(height)
        self.width = int(width)
        self.sigma = float(sigma)
        self.grid_y = torch.arange(self.height, dtype=torch.float32).view(
            1, self.height, 1
        )
        self.grid_x = torch.arange(self.width, dtype=torch.float32).view(
            1, 1, self.width
        )

    def _joint_maps(self, keypoints, reliability):
        x = keypoints[:, 0] * (self.width - 1) / 127.0
        y = keypoints[:, 1] * (self.height - 1) / 383.0
        distance = (self.grid_x - x[:, None, None]).square()
        distance = distance + (self.grid_y - y[:, None, None]).square()
        gaussian = torch.exp(-distance / (2.0 * self.sigma ** 2))
        return gaussian * reliability[:, None, None]

    def _segment_map(self, keypoints, reliability, segments):
        if not segments:
            return torch.zeros(self.height, self.width, dtype=torch.float32)
        result = torch.zeros(self.height, self.width, dtype=torch.float32)
        px = self.grid_x[0]
        py = self.grid_y[0]
        for left, right in segments:
            ax = keypoints[left, 0] * (self.width - 1) / 127.0
            ay = keypoints[left, 1] * (self.height - 1) / 383.0
            bx = keypoints[right, 0] * (self.width - 1) / 127.0
            by = keypoints[right, 1] * (self.height - 1) / 383.0
            dx = bx - ax
            dy = by - ay
            denominator = (dx * dx + dy * dy).clamp_min(1e-6)
            projection = ((px - ax) * dx + (py - ay) * dy) / denominator
            projection = projection.clamp(0.0, 1.0)
            nearest_x = ax + projection * dx
            nearest_y = ay + projection * dy
            distance = (px - nearest_x).square() + (py - nearest_y).square()
            amplitude = torch.minimum(reliability[left], reliability[right])
            tube = torch.exp(-distance / (2.0 * self.sigma ** 2)) * amplitude
            result = torch.maximum(result, tube)
        return result

    def __call__(self, keypoints, scores, valid):
        reliability = valid.float() * scores.float().clamp(0.0, 1.0)
        joints = self._joint_maps(keypoints.float(), reliability)
        masks = []
        confidence = []
        region_valid = []
        for joint_ids, segments in zip(REGION_JOINTS, REGION_SEGMENTS):
            index = torch.as_tensor(joint_ids, dtype=torch.long)
            joint_mask = joints.index_select(0, index).amax(0)
            segment_mask = self._segment_map(
                keypoints.float(), reliability, segments
            )
            masks.append(torch.maximum(joint_mask, segment_mask))
            values = reliability.index_select(0, index)
            confidence.append(values.mean())
            region_valid.append(values.max() > 0)
        return (
            torch.stack(masks, dim=0),
            torch.stack(confidence, dim=0),
            torch.stack(region_valid, dim=0),
        )


class AuditTransform:
    """One path-hash deterministic official geometry with pre/post-RE RGB."""

    def __init__(self, seed=20260718):
        self.seed = int(seed)
        self.renderer = RegionRenderer()
        self.eraser = RandomErasing(
            probability=0.5, mode="pixel", max_count=1, device="cpu"
        )

    def __call__(self, image, pose):
        keypoints = pose.keypoints.clone().float()
        scores = pose.scores.clone().float()
        valid = pose.valid.clone().bool()
        original_width, original_height = image.size
        image = TF.resize(
            image, [384, 128], interpolation=InterpolationMode.BICUBIC
        )
        keypoints[:, 0] *= 128.0 / float(original_width)
        keypoints[:, 1] *= 384.0 / float(original_height)

        local_seed = seed_from_text(pose.relative_path, self.seed)
        generator = torch.Generator().manual_seed(local_seed)
        flipped = bool(torch.rand(1, generator=generator).item() < 0.5)
        if flipped:
            image = TF.hflip(image)
            keypoints[:, 0] = 127.0 - keypoints[:, 0]
            index = torch.as_tensor(COCO17_FLIP, dtype=torch.long)
            keypoints = keypoints.index_select(0, index)
            scores = scores.index_select(0, index)
            valid = valid.index_select(0, index)

        image = TF.pad(image, [10, 10, 10, 10], fill=0)
        keypoints[:, 0] += 10.0
        keypoints[:, 1] += 10.0
        crop_top = int(torch.randint(0, 21, (1,), generator=generator).item())
        crop_left = int(torch.randint(0, 21, (1,), generator=generator).item())
        image = TF.crop(image, crop_top, crop_left, 384, 128)
        keypoints[:, 0] -= float(crop_left)
        keypoints[:, 1] -= float(crop_top)
        valid = (
            valid
            & (keypoints[:, 0] >= 0)
            & (keypoints[:, 0] <= 127)
            & (keypoints[:, 1] >= 0)
            & (keypoints[:, 1] <= 383)
        )
        pre_erasing = TF.to_tensor(image)
        normalized = (pre_erasing - 0.5) / 0.5
        rng_state = torch.get_rng_state()
        torch.manual_seed(local_seed ^ 0x5A5A5A5A)
        post_normalized = self.eraser(normalized.clone())
        torch.set_rng_state(rng_state)
        post_erasing = post_normalized * 0.5 + 0.5
        erase_mask = (normalized != post_normalized).any(0).float()
        erased = bool(erase_mask.any())
        masks, confidence, region_valid = self.renderer(
            keypoints, scores, valid
        )
        erase_mask_grid = F.avg_pool2d(
            erase_mask.view(1, 1, 384, 128), kernel_size=4, stride=4
        ).view(96, 32)
        mask_mass = masks.flatten(1).sum(1)
        erased_mass = (masks * erase_mask_grid[None]).flatten(1).sum(1)
        erase_overlap = erased_mass / mask_mass.clamp_min(1e-12)
        erase_overlap = torch.where(
            mask_mass > 0, erase_overlap, torch.zeros_like(erase_overlap)
        )
        return {
            "pre": pre_erasing,
            "post": post_erasing,
            "masks": masks,
            "confidence": confidence,
            "valid": region_valid,
            "erase_overlap": erase_overlap,
            "flipped": flipped,
            "erased": erased,
            "crop_top": crop_top,
            "crop_left": crop_left,
        }


class TeacherAuditDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records,
        pose_store,
        rgb_donor_map,
        mask_donor_map,
        seed,
        verify_sha,
    ):
        self.records = list(records)
        self.pose_store = pose_store
        self.rgb_donor_map = np.asarray(rgb_donor_map, dtype=np.int64)
        self.mask_donor_map = np.asarray(mask_donor_map, dtype=np.int64)
        self.transform = AuditTransform(seed=seed)
        self.verify_sha = bool(verify_sha)
        if self.rgb_donor_map.shape != (len(self.records),):
            raise ValueError("RGB donor map length mismatch")
        if self.mask_donor_map.shape != (len(self.records),):
            raise ValueError("Mask donor map length mismatch")

    def __len__(self):
        return len(self.records)

    def _load(self, index, verify):
        from datasets.bases import read_image

        image_path, pid, camid, trackid = self.records[index]
        del trackid
        image = read_image(image_path)
        pose = self.pose_store.get(image_path, verify_image_sha=verify)
        transformed = self.transform(image, pose)
        transformed.update({
            "pid": int(pid),
            "camid": int(camid),
            "relative_path": pose.relative_path,
            "image_sha256": pose.image_sha256,
        })
        return transformed

    def __getitem__(self, index):
        recipient = self._load(index, self.verify_sha)
        rgb_donor = self._load(int(self.rgb_donor_map[index]), False)
        mask_donor = self._load(int(self.mask_donor_map[index]), False)
        return recipient, rgb_donor, mask_donor


def collate_audit(batch):
    recipients, rgb_donors, mask_donors = zip(*batch)

    def stack(items):
        return {
            "pre": torch.stack([item["pre"] for item in items]),
            "post": torch.stack([item["post"] for item in items]),
            "masks": torch.stack([item["masks"] for item in items]),
            "confidence": torch.stack([item["confidence"] for item in items]),
            "valid": torch.stack([item["valid"] for item in items]),
            "erase_overlap": torch.stack(
                [item["erase_overlap"] for item in items]
            ),
            "pid": torch.tensor([item["pid"] for item in items]),
            "camid": torch.tensor([item["camid"] for item in items]),
            "relative_path": tuple(item["relative_path"] for item in items),
            "image_sha256": tuple(item["image_sha256"] for item in items),
            "erased": torch.tensor([item["erased"] for item in items]),
            "flipped": torch.tensor([item["flipped"] for item in items]),
            "crop_top": torch.tensor([item["crop_top"] for item in items]),
            "crop_left": torch.tensor([item["crop_left"] for item in items]),
        }

    return stack(recipients), stack(rgb_donors), stack(mask_donors)


def _donor_invariants(records, donor):
    pids = np.asarray([int(row[1]) for row in records])
    camids = np.asarray([int(row[2]) for row in records])
    donor = np.asarray(donor, dtype=np.int64)
    if donor.shape != (len(records),):
        raise RuntimeError("Donor map shape mismatch")
    if bool((donor < 0).any()) or bool((donor >= len(records)).any()):
        raise RuntimeError("Donor map index out of range")
    if bool((donor == np.arange(len(records))).any()):
        raise RuntimeError("Donor map has fixed point")
    if bool((pids[donor] == pids).any()):
        raise RuntimeError("Donor map has same-PID pair")
    return {
        "count": len(records),
        "same_camera_fraction": float((camids[donor] == camids).mean()),
        "different_pid_fraction": float((pids[donor] != pids).mean()),
        "no_fixed_points": bool(np.all(donor != np.arange(len(records)))),
    }


def build_rgb_donor_map(records):
    pids = np.asarray([int(row[1]) for row in records])
    camids = np.asarray([int(row[2]) for row in records])
    donor = np.full(len(records), -1, dtype=np.int64)
    for camera in np.unique(camids):
        group = np.flatnonzero(camids == camera)
        for position, recipient in enumerate(group):
            for offset in range(1, len(group)):
                candidate = group[(position + offset) % len(group)]
                if pids[candidate] != pids[recipient]:
                    donor[recipient] = candidate
                    break
    if bool((donor < 0).any()):
        raise RuntimeError("Could not build same-camera donor map")
    return donor, _donor_invariants(records, donor)


def pose_match_descriptors(records, pose_store):
    """Fixed raw-pose mask area/y-center/confidence descriptors."""
    renderer = RegionRenderer(height=48, width=16, sigma=0.75)
    descriptors = np.empty((len(records), 15), dtype=np.float64)
    for index, row in enumerate(records):
        pose = pose_store.get(row[0], verify_image_sha=False)
        keypoints = pose.keypoints.clone().float()
        width, height = pose.image_size
        keypoints[:, 0] *= 128.0 / float(width)
        keypoints[:, 1] *= 384.0 / float(height)
        masks, confidence, valid = renderer(
            keypoints, pose.scores, pose.valid
        )
        weights = masks.double()
        mass = weights.flatten(1).sum(1)
        y_axis = torch.linspace(0.0, 1.0, renderer.height).view(1, -1, 1)
        y_center = (weights * y_axis).flatten(1).sum(1) / mass.clamp_min(1e-12)
        area = (weights > 0.05).flatten(1).double().mean(1)
        descriptor = torch.stack(
            (area, y_center, confidence.double()), dim=1
        )
        descriptor = torch.where(
            valid[:, None], descriptor, torch.zeros_like(descriptor)
        )
        descriptors[index] = descriptor.flatten().numpy()
    if not np.isfinite(descriptors).all():
        raise RuntimeError("Non-finite pose matching descriptors")
    return descriptors


def build_mask_donor_map(records, descriptors):
    """Nearest different-PID donor matched on mask geometry and confidence."""
    from scipy.spatial import cKDTree

    descriptors = np.asarray(descriptors, dtype=np.float64)
    if descriptors.shape != (len(records), 15):
        raise RuntimeError("Unexpected pose matching descriptor shape")
    pids = np.asarray([int(row[1]) for row in records])
    camids = np.asarray([int(row[2]) for row in records])
    center = np.median(descriptors, axis=0)
    scale = np.median(np.abs(descriptors - center), axis=0) * 1.4826
    fallback = descriptors.std(axis=0)
    scale = np.where(scale > 1e-6, scale, fallback)
    scale = np.where(scale > 1e-6, scale, 1.0)
    standardized = (descriptors - center) / scale
    donor = np.full(len(records), -1, dtype=np.int64)
    matched_distance = np.full(len(records), np.nan, dtype=np.float64)

    for camera in np.unique(camids):
        group = np.flatnonzero(camids == camera)
        tree = cKDTree(standardized[group])
        neighbor_count = min(64, len(group))
        distances, positions = tree.query(
            standardized[group], k=neighbor_count
        )
        if neighbor_count == 1:
            distances = distances[:, None]
            positions = positions[:, None]
        for row_index, recipient in enumerate(group):
            candidates = group[np.asarray(positions[row_index], dtype=np.int64)]
            keep = (candidates != recipient) & (pids[candidates] != pids[recipient])
            if bool(keep.any()):
                first = int(np.flatnonzero(keep)[0])
                donor[recipient] = candidates[first]
                matched_distance[recipient] = float(distances[row_index][first])
                continue
            distances_one, positions_one = tree.query(
                standardized[recipient], k=len(group)
            )
            candidates = group[np.asarray(positions_one, dtype=np.int64)]
            keep = (candidates != recipient) & (pids[candidates] != pids[recipient])
            if not bool(keep.any()):
                raise RuntimeError("Could not find matched-mask donor")
            first = int(np.flatnonzero(keep)[0])
            donor[recipient] = candidates[first]
            matched_distance[recipient] = float(
                np.asarray(distances_one)[first]
            )

    summary = _donor_invariants(records, donor)
    delta = np.abs(descriptors - descriptors[donor]).reshape(-1, 5, 3)
    summary.update({
        "matching_features": ["mask_area", "vertical_center", "pose_confidence"],
        "standardized_distance_mean": float(matched_distance.mean()),
        "standardized_distance_median": float(np.median(matched_distance)),
        "standardized_distance_p95": float(np.quantile(matched_distance, 0.95)),
        "absolute_delta_mean_by_feature": {
            "mask_area": float(delta[:, :, 0].mean()),
            "vertical_center": float(delta[:, :, 1].mean()),
            "pose_confidence": float(delta[:, :, 2].mean()),
        },
    })
    return donor, summary


class FrozenClipTeacher:
    def __init__(self, checkpoint, device):
        import open_clip

        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=str(checkpoint)
        )
        self.model = self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.visual = self.model.visual
        self.captured = []
        self.last_raw_token_shape = None

        def hook(module, inputs, output):
            del module, inputs
            self.captured.append(output)

        self.handle = self.visual.transformer.resblocks[-1].register_forward_hook(
            hook
        )
        prompts = [
            template.format(phrase)
            for phrase in REGION_PHRASES
            for template in PROMPT_TEMPLATES
        ]
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            encoded = self.model.encode_text(tokens)
        encoded = F.normalize(encoded.float(), dim=-1)
        encoded = encoded.view(len(REGION_NAMES), len(PROMPT_TEMPLATES), -1)
        self.text = F.normalize(encoded.mean(1), dim=-1)
        self.prompt_payload = {
            "region_names": list(REGION_NAMES),
            "region_phrases": list(REGION_PHRASES),
            "templates": list(PROMPT_TEMPLATES),
            "prompts": prompts,
        }

    @torch.no_grad()
    def patch_tokens(self, images):
        mean = torch.as_tensor(CLIP_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.as_tensor(CLIP_STD, device=images.device).view(1, 3, 1, 1)
        normalized = (images.float() - mean) / std
        self.captured[:] = []
        with torch.cuda.amp.autocast():
            self.visual(normalized)
        if len(self.captured) != 1:
            raise RuntimeError("CLIP patch hook call count mismatch")
        tokens = self.captured.pop()
        if tokens.shape[0] != images.shape[0]:
            tokens = tokens.permute(1, 0, 2)
        if tokens.shape[1] != 257:
            raise RuntimeError("Expected ViT-L/14 16x16 patches")
        self.last_raw_token_shape = list(tokens.shape)
        with torch.cuda.amp.autocast():
            patch = self.visual.ln_post(tokens[:, 1:])
            if getattr(self.visual, "proj", None) is not None:
                patch = patch @ self.visual.proj
        return F.normalize(patch.float(), dim=-1)

    def close(self):
        self.handle.remove()


def clip_geometry(images, masks, kind):
    if kind == "square_stretch":
        clip_images = F.interpolate(
            images.float(), size=(224, 224), mode="bilinear", align_corners=False
        )
        clip_masks = F.interpolate(
            masks.float(), size=(224, 224), mode="bilinear", align_corners=False
        )
    elif kind == "aspect_letterbox":
        resized_images = F.interpolate(
            images.float(), size=(224, 75), mode="bilinear", align_corners=False
        )
        mean = torch.as_tensor(CLIP_MEAN, device=images.device).view(1, 3, 1, 1)
        clip_images = mean.expand(images.shape[0], 3, 224, 224).clone()
        clip_images[:, :, :, 74:149] = resized_images
        resized_masks = F.interpolate(
            masks.float(), size=(224, 75), mode="bilinear", align_corners=False
        )
        clip_masks = torch.zeros(
            masks.shape[0], masks.shape[1], 224, 224,
            device=masks.device, dtype=torch.float32
        )
        clip_masks[:, :, :, 74:149] = resized_masks
    else:
        raise ValueError("Unknown CLIP geometry: %s" % kind)
    grid_masks = F.avg_pool2d(clip_masks, kernel_size=14, stride=14)
    if grid_masks.shape[-2:] != (16, 16):
        raise RuntimeError("CLIP mask grid mismatch")
    return clip_images, grid_masks, clip_masks


def geometry_alignment_audit(kind, device):
    signal = torch.zeros(1, 1, 384, 128, device=device)
    signal[:, :, 73:311, 19:109] = 1.0
    images = signal.expand(-1, 3, -1, -1).contiguous()
    clip_images, grid_masks, clip_masks = clip_geometry(images, signal, kind)
    if kind == "square_stretch":
        compared_image = clip_images[:, :1]
        compared_mask = clip_masks
        transform = {
            "edge_coordinate_matrix": [
                [224.0 / 128.0, 0.0, 0.0],
                [0.0, 224.0 / 384.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "content_box_xyxy": [0, 0, 224, 224],
        }
    elif kind == "aspect_letterbox":
        compared_image = clip_images[:, :1, :, 74:149]
        compared_mask = clip_masks[:, :, :, 74:149]
        transform = {
            "edge_coordinate_matrix": [
                [75.0 / 128.0, 0.0, 74.0],
                [0.0, 224.0 / 384.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "content_box_xyxy": [74, 0, 149, 224],
            "padding_left_right": [74, 75],
        }
    else:
        raise ValueError(kind)
    max_abs = float((compared_image - compared_mask).abs().max().item())
    if max_abs != 0.0:
        raise RuntimeError("RGB/mask geometry alignment is not exact")
    return {
        "kind": kind,
        "rgb_mask_signal_max_abs": max_abs,
        "mask_grid_shape": list(grid_masks.shape[-2:]),
        "transform": transform,
        "pass": True,
    }


def fixed_band_masks(batch_size, device):
    masks = torch.zeros(batch_size, 5, 16, 16, device=device)
    boundaries = (0, 3, 6, 10, 13, 16)
    for region in range(5):
        masks[:, region, boundaries[region]:boundaries[region + 1], :] = 1.0
    return masks


def pool_regions(patch_tokens, masks):
    weights = masks.flatten(2).float()
    denominator = weights.sum(-1, keepdim=True)
    normalized = weights / denominator.clamp_min(1e-12)
    normalized = torch.where(
        denominator > 0, normalized, torch.zeros_like(normalized)
    )
    features = torch.einsum("bkn,bnd->bkd", normalized, patch_tokens.float())
    return F.normalize(features, dim=-1), denominator.squeeze(-1)


def distribution(features, text, temperature=0.07):
    logits = torch.einsum("bkd,cd->bkc", features.float(), text.float())
    return torch.softmax(logits / float(temperature), dim=-1)


def jsd(left, right):
    left = left.double().clamp_min(1e-12)
    right = right.double().clamp_min(1e-12)
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.log() - middle.log())).sum(-1)
        + (right * (right.log() - middle.log())).sum(-1)
    )


def entropy(q):
    value = q.double().clamp_min(1e-12)
    return -(value * value.log()).sum(-1)


def expected_margin(q):
    labels = torch.arange(5, device=q.device).view(1, 5, 1)
    expected = q.gather(-1, labels.expand(q.shape[0], -1, 1)).squeeze(-1)
    masked = q.clone()
    masked.scatter_(-1, labels.expand(q.shape[0], -1, 1), -1.0)
    alternative = masked.amax(-1)
    return expected - alternative


def per_image_mean(values, valid):
    values = values.double()
    valid = valid.bool()
    denominator = valid.sum(1)
    keep = denominator > 0
    result = (values * valid.double()).sum(1) / denominator.clamp_min(1)
    return result[keep].cpu().numpy()


def bootstrap_mean(values, seed=20260718, repeats=1000):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "repeats": repeats,
            "count": 0,
        }
    rng = np.random.RandomState(seed)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        draw = rng.randint(0, len(values), size=len(values))
        samples[index] = values[draw].mean()
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "repeats": repeats,
        "count": int(len(values)),
    }


def confusion_matrix(q, valid):
    prediction = q.argmax(-1)
    matrix = torch.zeros(5, 5, dtype=torch.int64)
    for expected in range(5):
        selected = prediction[:, expected][valid[:, expected]]
        matrix[expected] = torch.bincount(selected.cpu(), minlength=5)
    return matrix.tolist()


def summarize_arm(q, valid, repeats):
    q = q.float()
    valid = valid.bool()
    labels = torch.arange(5).view(1, 5)
    top1 = (q.argmax(-1) == labels).float()
    margin = expected_margin(q)
    top1_per_image = per_image_mean(top1, valid)
    margin_per_image = per_image_mean(margin, valid)
    per_class = []
    for region in range(5):
        keep = valid[:, region]
        per_class.append({
            "region": REGION_NAMES[region],
            "count": int(keep.sum()),
            "top1": float(top1[:, region][keep].mean()) if bool(keep.any()) else None,
            "margin": float(margin[:, region][keep].mean()) if bool(keep.any()) else None,
            "entropy": float(entropy(q[:, region])[keep].mean()) if bool(keep.any()) else None,
        })
    return {
        "macro_top1": bootstrap_mean(top1_per_image, repeats=repeats),
        "expected_margin": bootstrap_mean(margin_per_image, repeats=repeats),
        "entropy_mean": float(entropy(q)[valid].mean()),
        "per_class": per_class,
        "confusion": confusion_matrix(q, valid),
    }


def paired_arm(correct, counterfactual, valid, repeats):
    correct_top1 = (
        correct.argmax(-1) == torch.arange(5).view(1, 5)
    ).float()
    other_top1 = (
        counterfactual.argmax(-1) == torch.arange(5).view(1, 5)
    ).float()
    top1_delta = per_image_mean(correct_top1 - other_top1, valid)
    margin_delta = per_image_mean(
        expected_margin(correct) - expected_margin(counterfactual), valid
    )
    divergence = per_image_mean(jsd(correct, counterfactual), valid)
    return {
        "correct_minus_arm_top1": bootstrap_mean(
            top1_delta, repeats=repeats
        ),
        "correct_minus_arm_margin": bootstrap_mean(
            margin_delta, repeats=repeats
        ),
        "q_jsd": bootstrap_mean(divergence, repeats=repeats),
    }


def sample_specificity(q, valid):
    result = []
    rng = np.random.RandomState(20260718)
    for region in range(5):
        values = q[:, region][valid[:, region]].double()
        if len(values) < 2:
            result.append({
                "region": REGION_NAMES[region],
                "count": int(len(values)),
                "q_variance_mean": None,
                "centered_effective_rank": None,
                "sampled_pairwise_jsd_mean": None,
            })
            continue
        centered = values - values.mean(0, keepdim=True)
        covariance = centered.t().mm(centered) / max(len(values) - 1, 1)
        eigen = torch.linalg.eigvalsh(covariance).clamp_min(0)
        probability = eigen / eigen.sum().clamp_min(1e-12)
        nonzero = probability > 0
        effective_rank = float(
            torch.exp(-(probability[nonzero] * probability[nonzero].log()).sum())
        )
        pair_count = min(4096, max(len(values), 1))
        left = rng.randint(0, len(values), size=pair_count)
        right = rng.randint(0, len(values), size=pair_count)
        pair_jsd = jsd(values[left], values[right])
        result.append({
            "region": REGION_NAMES[region],
            "count": int(len(values)),
            "q_variance_mean": float(values.var(0, unbiased=False).mean()),
            "centered_effective_rank": effective_rank,
            "sampled_pairwise_jsd_mean": float(pair_jsd.mean()),
        })
    return result


def confidence_groups(q, valid, confidence):
    margin = expected_margin(q)
    ent = entropy(q)
    groups = {
        "high": valid & (confidence >= 0.7),
        "low": valid & (confidence < 0.3),
        "invalid": ~valid,
    }
    output = {}
    for name, mask in groups.items():
        output[name] = {
            "count": int(mask.sum()),
            "entropy": float(ent[mask].mean()) if bool(mask.any()) else None,
            "margin": float(margin[mask].mean()) if bool(mask.any()) else None,
        }
    try:
        from scipy.stats import spearmanr

        selected = valid.flatten().numpy()
        conf = confidence.flatten().numpy()[selected]
        margin_values = margin.flatten().numpy()[selected]
        entropy_values = ent.flatten().numpy()[selected]
        output["spearman_conf_margin"] = float(
            spearmanr(conf, margin_values).statistic
        )
        output["spearman_conf_negative_entropy"] = float(
            spearmanr(conf, -entropy_values).statistic
        )
    except Exception as error:
        output["spearman_error"] = repr(error)
    return output


def flip_summary(correct, flipped, valid, repeats):
    consistency = (
        correct.argmax(-1) == flipped.argmax(-1)
    ).float()
    return {
        "top1_consistency": bootstrap_mean(
            per_image_mean(consistency, valid), repeats=repeats
        ),
        "q_jsd": bootstrap_mean(
            per_image_mean(jsd(correct, flipped), valid), repeats=repeats
        ),
    }


def wrong_text_reverse_summary(q, valid, repeats):
    inverse = torch.argsort(torch.as_tensor(CHANNEL_CYCLE)).view(1, 5)
    recovered = (q.argmax(-1) == inverse).float()
    return {
        "top1_after_inverse_label_mapping": bootstrap_mean(
            per_image_mean(recovered, valid), repeats=repeats
        ),
        "prototype_bank_index_to_original_label": list(CHANNEL_CYCLE),
        "expected_inverse_bank_index": inverse.flatten().tolist(),
    }


def erase_summary(correct, post, valid, erased, erase_overlap, repeats):
    selected = valid & (erase_overlap >= 0.10)
    if not bool(selected.any()):
        return {
            "erased_images": int(erased.sum()),
            "regions_with_ge_10pct_mask_mass_erased": 0,
        }
    return {
        "erased_images": int(erased.sum()),
        "regions_with_ge_10pct_mask_mass_erased": int(selected.sum()),
        "q_jsd": bootstrap_mean(
            per_image_mean(jsd(correct, post), selected), repeats=repeats
        ),
        "post_minus_pre_entropy": bootstrap_mean(
            per_image_mean(entropy(post) - entropy(correct), selected),
            repeats=repeats,
        ),
        "pre_minus_post_margin": bootstrap_mean(
            per_image_mean(
                expected_margin(correct) - expected_margin(post), selected
            ),
            repeats=repeats,
        ),
    }


def spherical_kmeans(features, labels, iterations=20):
    features = F.normalize(features.float().cuda(), dim=-1)
    centers = [features[0]]
    minimum_distance = 1.0 - features.mv(centers[0])
    for _ in range(1, 5):
        index = int(minimum_distance.argmax().item())
        centers.append(features[index])
        distance = 1.0 - features.mv(centers[-1])
        minimum_distance = torch.minimum(minimum_distance, distance)
    centers = F.normalize(torch.stack(centers), dim=-1)
    assignment = None
    for _ in range(iterations):
        assignment = features.mm(centers.t()).argmax(1)
        updated = []
        for cluster in range(5):
            selected = features[assignment == cluster]
            updated.append(
                centers[cluster] if len(selected) == 0 else selected.mean(0)
            )
        centers = F.normalize(torch.stack(updated), dim=-1)
    assignment = features.mm(centers.t()).argmax(1).cpu().numpy()
    labels = np.asarray(labels, dtype=np.int64)
    confusion = np.zeros((5, 5), dtype=np.int64)
    for cluster, label in zip(assignment, labels):
        confusion[cluster, label] += 1
    best_accuracy = -1.0
    best_mapping = None
    for permutation in itertools.permutations(range(5)):
        correct = sum(confusion[cluster, permutation[cluster]] for cluster in range(5))
        accuracy = correct / float(len(labels))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = permutation
    return {
        "accuracy_after_best_permutation": best_accuracy,
        "cluster_to_region": list(best_mapping),
        "confusion_cluster_by_region": confusion.tolist(),
        "iterations": iterations,
        "samples": int(len(labels)),
    }


def geometry_buffers():
    arms = (
        "correct",
        "wrong_rgb",
        "wrong_mask",
        "channel_shuffle_mask",
        "wrong_text",
        "uniform_mask",
        "fixed_bands",
        "post_erasing",
        "horizontal_flip",
    )
    return {
        "q": {arm: [] for arm in arms},
        "visual": [],
        "mask_mass": [],
        "effective_patch_count": [],
        "padding_mass_fraction": [],
        "flip_mask_iou": [],
        "repeat_exact": [],
        "token_shapes": [],
    }


def finalize_geometry(
    buffer,
    valid,
    confidence,
    erased,
    erase_overlap,
    text_prototype_q,
    repeats,
):
    q = {
        arm: torch.cat(values, dim=0).float()
        for arm, values in buffer["q"].items()
    }
    visual = torch.cat(buffer["visual"], dim=0)
    mask_mass = torch.cat(buffer["mask_mass"], dim=0)
    patch_count = torch.cat(buffer["effective_patch_count"], dim=0)
    padding_fraction = torch.cat(buffer["padding_mass_fraction"], dim=0)
    flip_mask_iou = torch.cat(buffer["flip_mask_iou"], dim=0)
    for name, value in q.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("Non-finite teacher distribution: %s" % name)
    for name, value in (
        ("visual", visual),
        ("mask_mass", mask_mass),
        ("patch_count", patch_count),
        ("padding_fraction", padding_fraction),
        ("flip_mask_iou", flip_mask_iou),
    ):
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("Non-finite geometry buffer: %s" % name)
    effective_valid = valid.bool() & (mask_mass > 1e-8)
    correct = q["correct"]
    arm_summaries = {
        arm: summarize_arm(values, effective_valid, repeats)
        for arm, values in q.items()
    }
    paired = {
        arm: paired_arm(correct, values, effective_valid, repeats)
        for arm, values in q.items() if arm != "correct"
    }

    text_onehot = torch.eye(5).view(1, 5, 5).expand_as(correct)
    text_prototype = text_prototype_q.view(1, 5, 5).expand_as(correct)
    arm_summaries["text_only_onehot"] = summarize_arm(
        text_onehot, effective_valid, repeats
    )
    arm_summaries["text_only_prototype"] = summarize_arm(
        text_prototype, effective_valid, repeats
    )
    paired["text_only_onehot"] = paired_arm(
        correct, text_onehot, effective_valid, repeats
    )
    paired["text_only_prototype"] = paired_arm(
        correct, text_prototype, effective_valid, repeats
    )

    flat_visual = []
    flat_labels = []
    for region in range(5):
        selected = visual[:, region][effective_valid[:, region]]
        flat_visual.append(selected)
        flat_labels.extend([region] * len(selected))
    image_cluster = spherical_kmeans(
        torch.cat(flat_visual, dim=0), flat_labels
    )

    specificity = sample_specificity(correct, effective_valid)
    text_specificity = {
        "text_only_onehot": sample_specificity(text_onehot, effective_valid),
        "text_only_prototype": sample_specificity(
            text_prototype, effective_valid
        ),
    }
    groups = confidence_groups(correct, effective_valid, confidence)
    flip = flip_summary(
        correct, q["horizontal_flip"], effective_valid, repeats
    )
    erase = erase_summary(
        correct,
        q["post_erasing"],
        effective_valid,
        erased,
        erase_overlap,
        repeats,
    )
    flip_iou_selected = flip_mask_iou[effective_valid]
    coverage_summary = {
        "effective_patch_count_mean": float(
            patch_count[effective_valid].float().mean()
        ),
        "effective_patch_count_median": float(
            patch_count[effective_valid].float().median()
        ),
        "empty_region_fraction": float(
            (mask_mass <= 1e-8)[valid].float().mean()
        ),
        "mask_grid_mass_mean": float(mask_mass[effective_valid].mean()),
        "padding_mask_mass_fraction_mean": float(
            padding_fraction[effective_valid].mean()
        ),
        "padding_mask_mass_fraction_max": float(
            padding_fraction[effective_valid].max()
        ),
        "flip_mask_iou_mean": float(flip_iou_selected.mean()),
        "flip_mask_iou_median": float(flip_iou_selected.median()),
        "flip_mask_iou_min": float(flip_iou_selected.min()),
    }

    macro = arm_summaries["correct"]["macro_top1"]
    margin = arm_summaries["correct"]["expected_margin"]
    shuffle_margin = paired["channel_shuffle_mask"]["correct_minus_arm_margin"]
    wrong_mask_margin = paired["wrong_mask"]["correct_minus_arm_margin"]
    wrong_rgb_jsd = paired["wrong_rgb"]["q_jsd"]
    finite_rank = [
        row["centered_effective_rank"] for row in specificity
        if row["centered_effective_rank"] is not None
    ]
    finite_pair_jsd = [
        row["sampled_pairwise_jsd_mean"] for row in specificity
        if row["sampled_pairwise_jsd_mean"] is not None
    ]
    average_rank = float(np.mean(finite_rank)) if finite_rank else 0.0
    average_pair_jsd = (
        float(np.mean(finite_pair_jsd)) if finite_pair_jsd else 0.0
    )
    low = groups["low"]
    high = groups["high"]
    confidence_direction = (
        low["count"] > 0 and high["count"] > 0
        and low["entropy"] > high["entropy"]
        and low["margin"] < high["margin"]
    )
    erase_direction = (
        erase.get("regions_with_ge_10pct_mask_mass_erased", 0) > 0
        and erase["post_minus_pre_entropy"]["ci95_low"] > 0.0
        and erase["pre_minus_post_margin"]["ci95_low"] > 0.0
    )
    gates = {
        "correct_top1_lower_gt_chance": macro["ci95_low"] > 0.20,
        "correct_margin_positive": margin["ci95_low"] > 0.0,
        "correct_beats_channel_shuffle_margin": shuffle_margin["ci95_low"] > 0.0,
        "correct_beats_wrong_mask_margin": wrong_mask_margin["ci95_low"] > 0.0,
        "correct_beats_wrong_text_margin": paired["wrong_text"]
        ["correct_minus_arm_margin"]["ci95_low"] > 0.0,
        "wrong_rgb_sample_sensitive": wrong_rgb_jsd["ci95_low"] > 0.001,
        "distribution_not_constant": not (
            average_rank < 2.0 and average_pair_jsd < 0.01
        ),
        "confidence_direction": bool(confidence_direction),
        "synthetic_erasing_direction": bool(erase_direction),
        "flip_top1_ge_095": flip["top1_consistency"]["ci95_low"] >= 0.95,
        "flip_jsd_le_002": flip["q_jsd"]["ci95_high"] <= 0.02,
        "nonempty_regions": coverage_summary["empty_region_fraction"] == 0.0,
        "repeat_exact": bool(buffer["repeat_exact"]) and all(
            buffer["repeat_exact"]
        ),
    }
    return {
        "arms": arm_summaries,
        "paired": paired,
        "sample_specificity": specificity,
        "text_only_sample_specificity": text_specificity,
        "confidence_groups": groups,
        "flip": flip,
        "wrong_text_reverse": wrong_text_reverse_summary(
            q["wrong_text"], effective_valid, repeats
        ),
        "erasing": erase,
        "coverage": coverage_summary,
        "repeat_exact": {
            "all": gates["repeat_exact"],
            "checks": len(buffer["repeat_exact"]),
            "token_shapes": buffer["token_shapes"],
        },
        "image_only_cluster": image_cluster,
        "gates": gates,
        "pass": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pose-artifact", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--donor-map-output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    pose_artifact = Path(args.pose_artifact).resolve()
    clip_checkpoint = Path(args.clip_checkpoint).resolve()
    if sha256_file(pose_artifact / "manifest.json") != EXPECTED_POSE_MANIFEST_SHA256:
        raise RuntimeError("Pose manifest SHA mismatch")
    if sha256_file(clip_checkpoint) != EXPECTED_CLIP_SHA256:
        raise RuntimeError("CLIP checkpoint SHA mismatch")
    runtime_sha = {
        relative: sha256_file(repo_root / relative)
        for relative in EXPECTED_RUNTIME_SHA256
    }
    if runtime_sha != EXPECTED_RUNTIME_SHA256:
        raise RuntimeError("Phase 0B minimal runtime SHA mismatch")
    os.chdir(str(repo_root))
    sys.path.insert(0, str(repo_root))
    from datasets.occluded_duke import OccludedDuke
    from datasets.pose_targets import PoseTargetStore

    set_seed(args.seed)
    dataset = OccludedDuke(root=str(data_root), verbose=False)
    records = list(dataset.train)
    if len(records) != 15618:
        raise RuntimeError("Unexpected official train size: %d" % len(records))
    pose_store = PoseTargetStore(
        pose_artifact, EXPECTED_POSE_MANIFEST_SHA256
    )
    rgb_donor_map, rgb_donor_summary = build_rgb_donor_map(records)
    matching_descriptors = pose_match_descriptors(records, pose_store)
    mask_donor_map, mask_donor_summary = build_mask_donor_map(
        records, matching_descriptors
    )
    donor_payload = {
        "pose_manifest_sha256": EXPECTED_POSE_MANIFEST_SHA256,
        "seed": args.seed,
        "rgb_donor": {
            "summary": rgb_donor_summary,
            "donor_indices": rgb_donor_map.tolist(),
        },
        "matched_mask_donor": {
            "summary": mask_donor_summary,
            "donor_indices": mask_donor_map.tolist(),
            "matching_descriptor_sha256": hashlib.sha256(
                matching_descriptors.astype("<f8", copy=False).tobytes()
            ).hexdigest(),
        },
    }
    write_json(args.donor_map_output, donor_payload)
    donor_sha = sha256_file(args.donor_map_output)

    audit_dataset = TeacherAuditDataset(
        records,
        pose_store,
        rgb_donor_map,
        mask_donor_map,
        seed=args.seed,
        verify_sha=True,
    )
    if args.max_samples > 0:
        audit_dataset = torch.utils.data.Subset(
            audit_dataset, list(range(args.max_samples))
        )
    loader = torch.utils.data.DataLoader(
        audit_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_audit,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda")
    teacher = FrozenClipTeacher(clip_checkpoint, device)
    geometries = ("square_stretch", "aspect_letterbox")
    buffers = {geometry: geometry_buffers() for geometry in geometries}
    geometry_alignment = {
        geometry: geometry_alignment_audit(geometry, device)
        for geometry in geometries
    }
    text_prototype_q = distribution(
        teacher.text.unsqueeze(0), teacher.text
    ).squeeze(0).cpu()
    valid_all = []
    confidence_all = []
    erased_all = []
    erase_overlap_all = []
    manifest_digest = hashlib.sha256()
    samples_seen = 0
    cycle = torch.as_tensor(CHANNEL_CYCLE, device=device)

    for batch_index, (recipient, rgb_donor, mask_donor) in enumerate(loader):
        batch = recipient["pre"].shape[0]
        samples_seen += batch
        for relative_path, rgb_sha in zip(
            recipient["relative_path"], recipient["image_sha256"]
        ):
            manifest_digest.update(relative_path.encode("utf-8"))
            manifest_digest.update(b"\0")
            manifest_digest.update(rgb_sha.encode("ascii"))
        valid = recipient["valid"].bool()
        confidence = recipient["confidence"].float()
        valid_all.append(valid)
        confidence_all.append(confidence)
        erased_all.append(recipient["erased"].bool())
        erase_overlap_all.append(recipient["erase_overlap"].float())

        recipient_pre = recipient["pre"].to(device, non_blocking=True)
        recipient_post = recipient["post"].to(device, non_blocking=True)
        recipient_masks = recipient["masks"].to(device, non_blocking=True)
        donor_pre = rgb_donor["pre"].to(device, non_blocking=True)
        donor_masks = mask_donor["masks"].to(device, non_blocking=True)

        for geometry in geometries:
            rec_images, rec_grid, rec_clip_masks = clip_geometry(
                recipient_pre, recipient_masks, geometry
            )
            donor_images, donor_grid, _ = clip_geometry(
                donor_pre, donor_masks, geometry
            )
            post_images, _, _ = clip_geometry(
                recipient_post, recipient_masks, geometry
            )
            flip_images, flip_grid, _ = clip_geometry(
                recipient_pre.flip(-1), recipient_masks.flip(-1), geometry
            )

            paired_input = torch.cat((rec_images, donor_images), dim=0)
            paired_tokens = teacher.patch_tokens(paired_input)
            if batch_index == 0:
                repeated_tokens = teacher.patch_tokens(paired_input)
                buffers[geometry]["repeat_exact"].append(
                    bool(torch.equal(paired_tokens, repeated_tokens))
                )
                buffers[geometry]["token_shapes"].append(
                    {
                        "raw_with_cls": list(teacher.last_raw_token_shape),
                        "patch_only": list(paired_tokens.shape),
                    }
                )
            rec_tokens, donor_tokens = paired_tokens[:batch], paired_tokens[batch:]
            post_flip_tokens = teacher.patch_tokens(
                torch.cat((post_images, flip_images), dim=0)
            )
            post_tokens = post_flip_tokens[:batch]
            flip_tokens = post_flip_tokens[batch:]

            correct_features, mask_mass = pool_regions(rec_tokens, rec_grid)
            wrong_rgb_features, _ = pool_regions(donor_tokens, rec_grid)
            wrong_mask_features, _ = pool_regions(rec_tokens, donor_grid)
            shuffled_grid = rec_grid.index_select(1, cycle)
            shuffled_features, _ = pool_regions(rec_tokens, shuffled_grid)
            person_grid = rec_grid.amax(1, keepdim=True).expand(-1, 5, -1, -1)
            uniform_features, _ = pool_regions(rec_tokens, person_grid)
            bands = fixed_band_masks(batch, device)
            band_features, _ = pool_regions(rec_tokens, bands)
            post_features, _ = pool_regions(post_tokens, rec_grid)
            flip_features, _ = pool_regions(flip_tokens, flip_grid)

            q = buffers[geometry]["q"]
            q["correct"].append(
                distribution(correct_features, teacher.text).cpu()
            )
            q["wrong_rgb"].append(
                distribution(wrong_rgb_features, teacher.text).cpu()
            )
            q["wrong_mask"].append(
                distribution(wrong_mask_features, teacher.text).cpu()
            )
            q["channel_shuffle_mask"].append(
                distribution(shuffled_features, teacher.text).cpu()
            )
            q["wrong_text"].append(
                distribution(correct_features, teacher.text.index_select(0, cycle)).cpu()
            )
            q["uniform_mask"].append(
                distribution(uniform_features, teacher.text).cpu()
            )
            q["fixed_bands"].append(
                distribution(band_features, teacher.text).cpu()
            )
            q["post_erasing"].append(
                distribution(post_features, teacher.text).cpu()
            )
            q["horizontal_flip"].append(
                distribution(flip_features, teacher.text).cpu()
            )
            buffers[geometry]["visual"].append(
                correct_features.detach().half().cpu()
            )
            buffers[geometry]["mask_mass"].append(mask_mass.detach().cpu())
            buffers[geometry]["effective_patch_count"].append(
                (rec_grid.flatten(2) > 0.05).sum(-1).detach().cpu()
            )
            total_mask_mass = rec_clip_masks.flatten(2).sum(-1)
            if geometry == "aspect_letterbox":
                padding_mass = rec_clip_masks[:, :, :, :74].flatten(2).sum(-1)
                padding_mass = padding_mass + rec_clip_masks[
                    :, :, :, 149:
                ].flatten(2).sum(-1)
            else:
                padding_mass = torch.zeros_like(total_mask_mass)
            buffers[geometry]["padding_mass_fraction"].append(
                (padding_mass / total_mask_mass.clamp_min(1e-12))
                .detach().cpu()
            )
            reflected_flip_grid = flip_grid.flip(-1)
            intersection = torch.minimum(
                rec_grid, reflected_flip_grid
            ).flatten(2).sum(-1)
            union = torch.maximum(
                rec_grid, reflected_flip_grid
            ).flatten(2).sum(-1)
            buffers[geometry]["flip_mask_iou"].append(
                (intersection / union.clamp_min(1e-12)).detach().cpu()
            )

        if (batch_index + 1) % 25 == 0:
            print(
                "[Phase0B] batches=%d samples=%d" % (
                    batch_index + 1, samples_seen
                ),
                flush=True,
            )

    teacher.close()
    valid = torch.cat(valid_all, dim=0)
    confidence = torch.cat(confidence_all, dim=0)
    erased = torch.cat(erased_all, dim=0)
    erase_overlap = torch.cat(erase_overlap_all, dim=0)
    geometry_results = {
        geometry: finalize_geometry(
            buffers[geometry],
            valid,
            confidence,
            erased,
            erase_overlap,
            text_prototype_q,
            repeats=args.bootstrap_repeats,
        )
        for geometry in geometries
    }
    full_audit = args.max_samples <= 0
    passing = [name for name in geometries if geometry_results[name]["pass"]]
    if not full_audit:
        smoke_invariants = {
            "requested_sample_count_exact": samples_seen == args.max_samples,
            "geometry_alignment": all(
                value["pass"] for value in geometry_alignment.values()
            ),
            "repeat_exact": all(
                geometry_results[name]["repeat_exact"]["all"]
                for name in geometries
            ),
            "token_contract": all(
                geometry_results[name]["repeat_exact"]["token_shapes"]
                and geometry_results[name]["repeat_exact"]["token_shapes"][0]
                ["raw_with_cls"][0]
                == min(args.batch_size, args.max_samples) * 2
                and geometry_results[name]["repeat_exact"]["token_shapes"][0]
                ["raw_with_cls"][1] == 257
                and geometry_results[name]["repeat_exact"]["token_shapes"][0]
                ["patch_only"][1] == 256
                for name in geometries
            ),
        }
        verdict = (
            "CLIP_TEACHER_SMOKE_PASS"
            if all(smoke_invariants.values())
            else "CLIP_TEACHER_SMOKE_FAIL"
        )
        selected = None
    elif not passing:
        smoke_invariants = None
        verdict = "CURRENT_CLIP_TEACHER_NO_GO"
        selected = None
    elif len(passing) == 1:
        smoke_invariants = None
        verdict = "CLIP_TEACHER_GATE_PASS"
        selected = passing[0]
    else:
        smoke_invariants = None
        # Both satisfy every hard gate; choose the stronger mask-margin lower bound.
        selected = max(
            passing,
            key=lambda name: geometry_results[name]["paired"]
            ["channel_shuffle_mask"]["correct_minus_arm_margin"]["ci95_low"],
        )
        verdict = "CLIP_TEACHER_GATE_PASS"

    result = {
        "status": "EXP392_PHASE0B_COMPLETE",
        "verdict": verdict,
        "selected_geometry": selected,
        "formal_training_authorized": False,
        "phase0c_authorized": (
            full_audit and verdict == "CLIP_TEACHER_GATE_PASS"
        ),
        "audit_scope": "full" if full_audit else "smoke",
        "smoke_invariants": smoke_invariants,
        "execution": {
            "repo_root": str(repo_root),
            "source_commit": args.source_commit,
            "audit_script_sha256": sha256_file(Path(__file__).resolve()),
            "runtime_sha256": runtime_sha,
            "data_root": str(data_root),
            "pose_artifact": str(pose_artifact),
            "pose_manifest_sha256": EXPECTED_POSE_MANIFEST_SHA256,
            "clip_checkpoint": str(clip_checkpoint),
            "clip_checkpoint_sha256": EXPECTED_CLIP_SHA256,
            "open_clip_version": __import__("open_clip").__version__,
            "torch_version": torch.__version__,
            "samples": samples_seen,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "seed": args.seed,
            "sample_manifest_sha256": manifest_digest.hexdigest(),
            "donor_map_sha256": donor_sha,
        },
        "teacher": {
            "architecture": "ViT-L-14",
            "temperature": 0.07,
            "regions": list(REGION_NAMES),
            "region_joints": [list(value) for value in REGION_JOINTS],
            "region_segments": [
                [list(pair) for pair in value] for value in REGION_SEGMENTS
            ],
            "mask_grid": [96, 32],
            "sigma": 1.5,
            "prompt_payload": teacher.prompt_payload,
            "prompt_sha256": sha256_json(teacher.prompt_payload),
            "pre_erasing_primary": True,
        },
        "donor_summary": {
            "rgb": rgb_donor_summary,
            "matched_mask": mask_donor_summary,
        },
        "geometry_alignment": geometry_alignment,
        "geometries": geometry_results,
    }
    write_json(args.output, result)
    print("EXP392_PHASE0B_COMPLETE", flush=True)
    print("verdict=%s selected=%s" % (verdict, selected), flush=True)
    print("output_sha256=%s" % sha256_file(args.output), flush=True)


if __name__ == "__main__":
    main()
