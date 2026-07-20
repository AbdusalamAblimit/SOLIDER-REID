#!/usr/bin/env python3
"""Real frozen image+text teacher primitives for exp405 Phase 0B.

The module is import-safe: open_clip, checkpoints, data, and CUDA are touched
only when RegionIsolatedClipTeacher is explicitly constructed by the formal
measurement runner.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


HEIGHT = 384
WIDTH = 128
MASK_HEIGHT = 96
MASK_WIDTH = 32
REGIONS = 5
CLIP_SIZE = 224
CLIP_GRID = 16
CLIP_PATCH = 14
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
COCO17_FLIP = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)

REGION_NAMES = (
    "head",
    "upper_torso_arms",
    "lower_torso",
    "upper_legs",
    "lower_legs_feet",
)
REGION_JOINTS = (
    (0, 1, 2, 3, 4),
    (5, 6, 7, 8, 9, 10),
    (11, 12),
    (11, 12, 13, 14),
    (13, 14, 15, 16),
)
REGION_SEGMENTS = (
    ((0, 1), (0, 2), (1, 3), (2, 4)),
    ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10)),
    ((5, 11), (6, 12), (11, 12)),
    ((11, 13), (12, 14)),
    ((13, 15), (14, 16)),
)
REGION_PHRASES = (
    "head, face, and hair",
    "shoulders, chest, upper torso, arms, and hands",
    "abdomen, waist, hips, lower torso, and pelvis",
    "thighs and upper legs between the hips and knees",
    "lower legs and feet below the knees",
)
PART_PROMPT_TEMPLATES = (
    "a photo of the {} of a person",
    "the {} region of a pedestrian",
    "a close view of a person's {}",
    "human {}",
)
SUPPORT_PROMPT_PAIRS = (
    (
        "a photo of a person with clearly visible {}",
        "a photo of a person with occluded or obstructed {}",
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_from_path(relative_path: str, seed: int) -> int:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("relative_path must be a non-empty string")
    payload = "exp405/view/%d/%s" % (int(seed), relative_path)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def deterministic_geometry(relative_path: str, seed: int) -> dict[str, int | bool]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed_from_path(relative_path, seed))
    return {
        "flipped": bool(torch.rand((), generator=generator).item() < 0.5),
        "crop_top": int(torch.randint(0, 21, (), generator=generator).item()),
        "crop_left": int(torch.randint(0, 21, (), generator=generator).item()),
    }


def transform_pose(
    keypoints: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.Tensor,
    image_size: tuple[int, int],
    geometry: dict[str, int | bool],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if keypoints.shape != (17, 2) or scores.shape != (17,) or valid.shape != (17,):
        raise ValueError("expected one COCO-17 pose")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")
    width, height = map(int, image_size)
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    points = keypoints.detach().cpu().float().clone()
    confidence = scores.detach().cpu().float().clone()
    active = valid.detach().cpu().clone()
    if not torch.isfinite(points).all() or not torch.isfinite(confidence).all():
        raise ValueError("non-finite pose")
    points[:, 0] *= WIDTH / float(width)
    points[:, 1] *= HEIGHT / float(height)
    if bool(geometry["flipped"]):
        points[:, 0] = WIDTH - 1 - points[:, 0]
        index = torch.tensor(COCO17_FLIP, dtype=torch.long)
        points = points.index_select(0, index)
        confidence = confidence.index_select(0, index)
        active = active.index_select(0, index)
    points += 10.0
    points[:, 0] -= int(geometry["crop_left"])
    points[:, 1] -= int(geometry["crop_top"])
    active = (
        active
        & (points[:, 0] >= 0)
        & (points[:, 0] <= WIDTH - 1)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= HEIGHT - 1)
    )
    return points, confidence, active


def _segment_response(
    points: torch.Tensor,
    reliability: torch.Tensor,
    segments: Iterable[tuple[int, int]],
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    *,
    sigma: float,
    interior: bool,
) -> torch.Tensor:
    response = torch.zeros(MASK_HEIGHT, MASK_WIDTH, dtype=torch.float32)
    for left, right in segments:
        ax = points[left, 0] * (MASK_WIDTH - 1) / float(WIDTH - 1)
        ay = points[left, 1] * (MASK_HEIGHT - 1) / float(HEIGHT - 1)
        bx = points[right, 0] * (MASK_WIDTH - 1) / float(WIDTH - 1)
        by = points[right, 1] * (MASK_HEIGHT - 1) / float(HEIGHT - 1)
        dx = bx - ax
        dy = by - ay
        denominator = (dx.square() + dy.square()).clamp_min(1e-6)
        projection = ((grid_x - ax) * dx + (grid_y - ay) * dy) / denominator
        projection = projection.clamp(0.15, 0.85) if interior else projection.clamp(0.0, 1.0)
        nearest_x = ax + projection * dx
        nearest_y = ay + projection * dy
        distance = (grid_x - nearest_x).square() + (grid_y - nearest_y).square()
        amplitude = torch.minimum(reliability[left], reliability[right])
        tube = torch.exp(-distance / (2.0 * float(sigma) ** 2)) * amplitude
        response = torch.maximum(response, tube)
    return response


def render_anatomical_regions(
    keypoints: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.Tensor,
    *,
    sigma: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render mutually exclusive five-slot masks on the frozen 96x32 grid."""
    if keypoints.shape[-2:] != (17, 2) or scores.shape[-1:] != (17,):
        raise ValueError("expected batched COCO-17 tensors")
    if valid.shape != scores.shape or valid.dtype != torch.bool:
        raise ValueError("valid shape/dtype mismatch")
    if keypoints.shape[:-2] != scores.shape[:-1] or keypoints.ndim != 3:
        raise ValueError("pose batch mismatch")
    if not math.isfinite(float(sigma)) or float(sigma) <= 0:
        raise ValueError("sigma must be positive and finite")
    batch = len(keypoints)
    masks = []
    confidences = []
    region_validity = []
    grid_y = torch.arange(MASK_HEIGHT, dtype=torch.float32).view(MASK_HEIGHT, 1)
    grid_x = torch.arange(MASK_WIDTH, dtype=torch.float32).view(1, MASK_WIDTH)
    for row in range(batch):
        points = keypoints[row].detach().cpu().float()
        geometry = valid[row].detach().cpu().float()
        reliability = geometry * scores[row].detach().cpu().float().clamp(0, 1)
        point_x = points[:, 0] * (MASK_WIDTH - 1) / float(WIDTH - 1)
        point_y = points[:, 1] * (MASK_HEIGHT - 1) / float(HEIGHT - 1)
        distance = (
            (grid_x[None] - point_x[:, None, None]).square()
            + (grid_y[None] - point_y[:, None, None]).square()
        )
        joints = torch.exp(-distance / (2.0 * float(sigma) ** 2)) * geometry[:, None, None]
        raw = []
        confidence = []
        active = []
        for slot, (joint_ids, segments) in enumerate(zip(REGION_JOINTS, REGION_SEGMENTS)):
            index = torch.tensor(joint_ids, dtype=torch.long)
            joint_map = joints.index_select(0, index).amax(0)
            segment_map = _segment_response(
                points,
                geometry,
                segments,
                grid_x,
                grid_y,
                sigma=float(sigma),
                interior=slot in (1, 3, 4),
            )
            raw.append(torch.maximum(joint_map, segment_map))
            values = reliability.index_select(0, index)
            confidence.append(values.mean())
            active.append(geometry.index_select(0, index).any())
        raw = torch.stack(raw)
        total = raw.sum(0, keepdim=True)
        owner = raw.argmax(0, keepdim=True)
        hard = torch.zeros_like(raw).scatter_(0, owner, total.clamp(max=1.0))
        masks.append(hard)
        confidences.append(torch.stack(confidence))
        region_validity.append(torch.stack(active))
    masks = torch.stack(masks)
    confidence = torch.stack(confidences)
    geometry_valid = torch.stack(region_validity) & (masks.flatten(2).sum(-1) > 0)
    if not torch.isfinite(masks).all() or bool((masks.sum(1) > 1.000001).any()):
        raise RuntimeError("invalid hard-owner masks")
    return masks, confidence, geometry_valid


def isolated_attention_mask(patch_selected: torch.Tensor, heads: int) -> torch.Tensor:
    """Block all paths from selected CLS/patch tokens to non-target patches."""
    if patch_selected.ndim != 2 or patch_selected.dtype != torch.bool:
        raise ValueError("patch_selected must be a boolean matrix")
    if patch_selected.shape[1] != CLIP_GRID * CLIP_GRID:
        raise ValueError("expected a 16x16 CLIP patch grid")
    if int(heads) <= 0:
        raise ValueError("heads must be positive")
    selected = torch.cat(
        (torch.ones(len(patch_selected), 1, dtype=torch.bool, device=patch_selected.device), patch_selected),
        dim=1,
    )
    allowed = selected.unsqueeze(2) & selected.unsqueeze(1)
    identity = torch.eye(selected.shape[1], dtype=torch.bool, device=selected.device).unsqueeze(0)
    allowed |= identity
    mask = torch.zeros(allowed.shape, dtype=torch.float32, device=allowed.device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask.repeat_interleave(int(heads), dim=0)


class RegionIsolatedClipTeacher:
    """Frozen ViT-L/14 global and five-slot readout isolated from block one."""

    def __init__(
        self,
        checkpoint: str | Path,
        checkpoint_sha256: str,
        device: torch.device | str,
        *,
        microbatch: int = 1,
    ):
        unresolved_checkpoint = Path(checkpoint).expanduser()
        if unresolved_checkpoint.is_symlink():
            raise RuntimeError("CLIP checkpoint symlinks are forbidden")
        checkpoint = unresolved_checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if sha256_file(checkpoint) != checkpoint_sha256:
            raise RuntimeError("CLIP checkpoint SHA256 mismatch")
        if int(microbatch) <= 0:
            raise ValueError("microbatch must be positive")
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=str(checkpoint)
        )
        model = model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if len(model.visual.transformer.resblocks) != 24:
            raise RuntimeError("expected 24 ViT-L/14 blocks")
        if tuple(model.visual.grid_size) != (CLIP_GRID, CLIP_GRID):
            raise RuntimeError("expected a 16x16 CLIP grid")
        if getattr(model.visual, "pool_type", None) not in ("tok", "token"):
            raise RuntimeError("region isolation requires frozen CLS/token pooling")
        if getattr(model.visual, "attn_pool", None) is not None:
            raise RuntimeError("region isolation forbids an extra attention pool")
        normalizers = [
            transform for transform in preprocess.transforms
            if hasattr(transform, "mean") and hasattr(transform, "std")
        ]
        if len(normalizers) != 1:
            raise RuntimeError("could not identify official CLIP normalization")
        if any(abs(float(value) - expected) > 1e-8 for value, expected in zip(normalizers[0].mean, CLIP_MEAN)):
            raise RuntimeError("CLIP mean mismatch")
        if any(abs(float(value) - expected) > 1e-8 for value, expected in zip(normalizers[0].std, CLIP_STD)):
            raise RuntimeError("CLIP std mismatch")

        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        part_prompts = [
            template.format(phrase)
            for phrase in REGION_PHRASES
            for template in PART_PROMPT_TEMPLATES
        ]
        support_prompts = [
            template.format(phrase)
            for phrase in REGION_PHRASES
            for pair in SUPPORT_PROMPT_PAIRS
            for template in pair
        ]
        with torch.inference_mode():
            part = model.encode_text(tokenizer(part_prompts).to(device)).float()
            support = model.encode_text(tokenizer(support_prompts).to(device)).float()
        part = F.normalize(part, dim=-1).view(REGIONS, len(PART_PROMPT_TEMPLATES), -1)
        part = F.normalize(part.mean(1), dim=-1)
        support = F.normalize(support, dim=-1).view(
            REGIONS, len(SUPPORT_PROMPT_PAIRS), 2, -1
        )
        support = F.normalize(support.mean(1), dim=-1)
        logit_scale = float(model.logit_scale.detach().float().exp().item())
        if not math.isfinite(logit_scale) or logit_scale <= 0:
            raise RuntimeError("invalid native CLIP logit scale")

        self.visual = model.visual
        self.part_text = part
        self.visible_text = support[:, 0]
        self.occluded_text = support[:, 1]
        self.logit_scale = logit_scale
        self.device = torch.device(device)
        self.microbatch = int(microbatch)
        self.mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self.checkpoint_sha256 = checkpoint_sha256
        del model

    def _validate_inputs(
        self,
        rgb: torch.Tensor,
        masks: torch.Tensor,
        slots: torch.Tensor | None = None,
    ) -> None:
        if rgb.ndim != 4 or rgb.shape[1:] != (3, HEIGHT, WIDTH):
            raise ValueError("rgb must have shape Bx3x384x128")
        if masks.shape != (len(rgb), REGIONS, MASK_HEIGHT, MASK_WIDTH):
            raise ValueError("mask shape mismatch")
        values = (rgb, masks) if slots is None else (rgb, masks, slots)
        if any(value.device != self.device for value in values):
            raise ValueError("teacher inputs must be on the CLIP device")
        if not torch.isfinite(rgb).all() or not torch.isfinite(masks).all():
            raise ValueError("non-finite teacher input")
        if bool(((rgb < 0) | (rgb > 1)).any()) or bool(((masks < 0) | (masks > 1)).any()):
            raise ValueError("teacher inputs outside [0,1]")
        if slots is not None:
            if slots.shape != (len(rgb),) or slots.dtype not in (torch.int32, torch.int64):
                raise ValueError("slots must contain one integer per image")
            if bool(((slots < 0) | (slots >= REGIONS)).any()):
                raise ValueError("slot outside frozen taxonomy")

    def _letterbox(self, rgb: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        resized = F.interpolate(
            rgb.float(), size=(CLIP_SIZE, 75), mode="bicubic", align_corners=False, antialias=True
        ).clamp(0, 1)
        canvas = self.mean.expand(len(rgb), 3, CLIP_SIZE, CLIP_SIZE).clone()
        canvas[:, :, :, 74:149] = resized
        resized_masks = F.interpolate(masks.float(), size=(CLIP_SIZE, 75), mode="nearest")
        mask_canvas = torch.zeros(
            len(rgb), REGIONS, CLIP_SIZE, CLIP_SIZE, device=rgb.device, dtype=torch.float32
        )
        mask_canvas[:, :, :, 74:149] = resized_masks
        patch_mass = F.avg_pool2d(mask_canvas, kernel_size=CLIP_PATCH, stride=CLIP_PATCH)
        return (canvas - self.mean) / self.std, patch_mass.flatten(2) > 0

    def _project(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.visual._pool(tokens)
        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj
        return F.normalize(pooled.float(), dim=-1)

    def _encode_chunk(self, rgb: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized, selected = self._letterbox(rgb, masks)
        tokens = self.visual._embeds(normalized)
        global_tokens = tokens
        blocks = self.visual.transformer.resblocks
        for block in blocks:
            global_tokens = block(global_tokens)
        global_feature = self._project(global_tokens)

        batch, sequence, width = tokens.shape
        branches = tokens[:, None].expand(batch, REGIONS, sequence, width).reshape(
            batch * REGIONS, sequence, width
        ).clone()
        flat_selected = selected.reshape(batch * REGIONS, -1)
        valid = flat_selected.any(-1)
        output = torch.zeros(
            batch * REGIONS,
            self.visual.output_dim,
            device=self.device,
            dtype=torch.float32,
        )
        if bool(valid.any()):
            active = branches[valid]
            active_mask = flat_selected[valid]
            heads = int(blocks[0].attn.num_heads)
            attention = isolated_attention_mask(active_mask, heads)
            for block in blocks:
                active = block(active, attn_mask=attention)
            output[valid] = self._project(active)
        return {
            "global": global_feature,
            "visual": output.view(batch, REGIONS, -1),
            "readout_valid": valid.view(batch, REGIONS),
            "patch_selected": selected,
        }

    def _encode_selected_chunk(
        self,
        rgb: torch.Tensor,
        masks: torch.Tensor,
        slots: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        normalized, selected_all = self._letterbox(rgb, masks)
        index = torch.arange(len(rgb), device=self.device)
        selected = selected_all[index, slots]
        valid = selected.any(-1)
        tokens = self.visual._embeds(normalized)
        output = torch.zeros(
            len(rgb), self.visual.output_dim, device=self.device, dtype=torch.float32
        )
        if bool(valid.any()):
            active = tokens[valid]
            active_selected = selected[valid]
            blocks = self.visual.transformer.resblocks
            attention = isolated_attention_mask(
                active_selected, int(blocks[0].attn.num_heads)
            )
            for block in blocks:
                active = block(active, attn_mask=attention)
            output[valid] = self._project(active)
        return {
            "visual": output,
            "readout_valid": valid,
            "patch_selected": selected,
        }

    @torch.inference_mode()
    def encode(self, rgb: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_inputs(rgb, masks)
        chunks = []
        for start in range(0, len(rgb), self.microbatch):
            stop = min(start + self.microbatch, len(rgb))
            chunks.append(self._encode_chunk(rgb[start:stop], masks[start:stop]))
        result = {
            key: torch.cat([chunk[key] for chunk in chunks], dim=0)
            for key in chunks[0]
        }
        if not torch.isfinite(result["global"]).all() or not torch.isfinite(result["visual"]).all():
            raise RuntimeError("non-finite CLIP readout")
        return result

    @torch.inference_mode()
    def encode_selected(
        self,
        rgb: torch.Tensor,
        masks: torch.Tensor,
        slots: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(rgb, masks, slots)
        chunks = []
        for start in range(0, len(rgb), self.microbatch):
            stop = min(start + self.microbatch, len(rgb))
            chunks.append(
                self._encode_selected_chunk(
                    rgb[start:stop], masks[start:stop], slots[start:stop]
                )
            )
        result = {
            key: torch.cat([chunk[key] for chunk in chunks], dim=0)
            for key in chunks[0]
        }
        if not torch.isfinite(result["visual"]).all():
            raise RuntimeError("non-finite selected CLIP readout")
        return result
