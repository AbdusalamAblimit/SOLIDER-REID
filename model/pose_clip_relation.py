"""Training-only pose-indexed CLIP relation primitives for exp408."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_SIZE = 224
CLIP_GRID = 16
CLIP_PATCH = 14


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_pose_indexed_regions(
    keypoints,
    valid,
    *,
    image_hw,
    field_hw,
    sigma=1.5,
):
    """Render the frozen five-slot hard-owner ontology on an arbitrary grid."""
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 2):
        raise ValueError("keypoints must have shape [B,17,2]")
    if valid.shape != keypoints.shape[:2]:
        raise ValueError("valid must have shape [B,17]")
    if not float(sigma) > 0:
        raise ValueError("sigma must be positive")
    image_height, image_width = map(int, image_hw)
    field_height, field_width = map(int, field_hw)
    if min(image_height, image_width, field_height, field_width) <= 0:
        raise ValueError("image and field dimensions must be positive")

    geometry = valid.bool().float()
    points = keypoints.float()
    point_x = points[..., 0] * (field_width - 1) / float(max(image_width - 1, 1))
    point_y = points[..., 1] * (field_height - 1) / float(max(image_height - 1, 1))
    grid_y = torch.arange(
        field_height, device=points.device, dtype=torch.float32
    ).view(1, field_height, 1)
    grid_x = torch.arange(
        field_width, device=points.device, dtype=torch.float32
    ).view(1, 1, field_width)
    distance = (
        (grid_x[:, None] - point_x[..., None, None]).square()
        + (grid_y[:, None] - point_y[..., None, None]).square()
    )
    joints = torch.exp(-distance / (2.0 * float(sigma) ** 2))
    joints = joints * geometry[..., None, None]

    raw_regions = []
    region_validity = []
    for slot, (joint_ids, segments) in enumerate(
        zip(REGION_JOINTS, REGION_SEGMENTS)
    ):
        index = torch.as_tensor(joint_ids, device=points.device)
        joint_map = joints.index_select(1, index).amax(dim=1)
        segment_map = torch.zeros_like(joint_map)
        for left, right in segments:
            ax = point_x[:, left, None, None]
            ay = point_y[:, left, None, None]
            bx = point_x[:, right, None, None]
            by = point_y[:, right, None, None]
            dx = bx - ax
            dy = by - ay
            denominator = (dx.square() + dy.square()).clamp_min(1e-6)
            projection = (
                (grid_x - ax) * dx + (grid_y - ay) * dy
            ) / denominator
            projection = (
                projection.clamp(0.15, 0.85)
                if slot in (1, 3, 4)
                else projection.clamp(0.0, 1.0)
            )
            nearest_x = ax + projection * dx
            nearest_y = ay + projection * dy
            segment_distance = (
                (grid_x - nearest_x).square()
                + (grid_y - nearest_y).square()
            )
            amplitude = torch.minimum(
                geometry[:, left], geometry[:, right]
            )[:, None, None]
            tube = torch.exp(
                -segment_distance / (2.0 * float(sigma) ** 2)
            ) * amplitude
            segment_map = torch.maximum(segment_map, tube)
        raw_regions.append(torch.maximum(joint_map, segment_map))
        region_validity.append(
            geometry.index_select(1, index).bool().any(dim=1)
        )

    raw = torch.stack(raw_regions, dim=1)
    total = raw.sum(dim=1, keepdim=True)
    owner = raw.argmax(dim=1, keepdim=True)
    masks = torch.zeros_like(raw).scatter_(1, owner, total.clamp(max=1.0))
    region_valid = torch.stack(region_validity, dim=1)
    region_valid = region_valid & (masks.flatten(2).sum(dim=-1) > 0)
    if not bool(torch.isfinite(masks).all()):
        raise RuntimeError("non-finite anatomical masks")
    if bool((masks.sum(dim=1) > 1.000001).any()):
        raise RuntimeError("hard-owner masks overlap")
    return masks, region_valid


def mass_normalized_slot_pool(feature, masks):
    """Pool one feature per anatomical slot without detaching the feature."""
    if feature.ndim != 4 or masks.ndim != 4:
        raise ValueError("feature and masks must be rank four")
    if feature.shape[0] != masks.shape[0] or feature.shape[-2:] != masks.shape[-2:]:
        raise ValueError("feature/mask shape mismatch")
    mass = masks.float().flatten(2).sum(dim=-1)
    pooled = torch.einsum(
        "bchw,brhw->brc", feature.float(), masks.float()
    ) / mass.clamp_min(1e-6)[..., None]
    return pooled, mass > 0


def pose_clip_slot_relation_distance(student, teacher, valid):
    """Mean per-slot off-diagonal cosine Gram distance in FP32."""
    if student.ndim != 3 or teacher.ndim != 3:
        raise ValueError("student/teacher must have shape [B,R,D]")
    if student.shape[:2] != teacher.shape[:2] or valid.shape != student.shape[:2]:
        raise ValueError("relation batch/slot shape mismatch")
    distances = []
    for slot in range(student.shape[1]):
        active = valid[:, slot].bool()
        if int(active.sum()) < 2:
            continue
        slot_student = F.normalize(student[active, slot].float(), dim=-1)
        slot_teacher = F.normalize(teacher[active, slot].float(), dim=-1)
        student_relation = slot_student @ slot_student.transpose(0, 1)
        teacher_relation = slot_teacher @ slot_teacher.transpose(0, 1)
        off_diagonal = ~torch.eye(
            student_relation.shape[0],
            device=student_relation.device,
            dtype=torch.bool,
        )
        distances.append(
            F.mse_loss(
                student_relation[off_diagonal],
                teacher_relation[off_diagonal],
            )
        )
    if not distances:
        return student.sum() * 0.0
    return torch.stack(distances).mean()


def validated_different_pid_shift(identities, shift=4):
    """Validate the frozen identity-sampler cyclic shift."""
    if identities.ndim != 1 or len(identities) < 2:
        raise ValueError("identities must be a nontrivial vector")
    shift = int(shift)
    if shift <= 0 or shift >= len(identities):
        raise ValueError("frozen wrong-RGB shift is outside the batch")
    donor = torch.roll(identities, shifts=-shift, dims=0)
    if not bool((donor != identities).all()):
        raise RuntimeError("frozen wrong-RGB shift is not different-PID")
    return shift


def pose_clip_counterfactual_relation_loss(
    student,
    teacher,
    student_valid,
    teacher_valid,
    identities,
):
    """Correct relation plus margin-free ranking against three controls."""
    if student.shape[:2] != teacher.shape[:2]:
        raise ValueError("student/teacher batch and slot dimensions must match")
    if student_valid.shape != student.shape[:2] or teacher_valid.shape != student.shape[:2]:
        raise ValueError("student/teacher validity shape mismatch")
    shift = validated_different_pid_shift(identities, shift=4)
    wrong_teacher = torch.roll(teacher, shifts=-shift, dims=0)
    wrong_teacher_valid = torch.roll(teacher_valid, shifts=-shift, dims=0)
    weights = teacher_valid.float()
    generic_slot = (teacher.float() * weights[..., None]).sum(dim=0)
    generic_slot = generic_slot / weights.sum(dim=0).clamp_min(1.0)[..., None]
    generic_teacher = generic_slot.unsqueeze(0).expand_as(teacher)
    generic_available = weights.sum(dim=0) > 0
    common_valid = (
        student_valid.bool()
        & teacher_valid.bool()
        & wrong_teacher_valid.bool()
        & generic_available.unsqueeze(0)
    )
    correct = pose_clip_slot_relation_distance(student, teacher, common_valid)
    wrong = pose_clip_slot_relation_distance(student, wrong_teacher, common_valid)
    generic = pose_clip_slot_relation_distance(
        student, generic_teacher, common_valid
    )
    zero_teacher = torch.zeros_like(teacher)
    zero = pose_clip_slot_relation_distance(
        student, zero_teacher, common_valid
    )
    negatives = torch.stack((wrong, generic, zero))
    ranking = F.softplus(correct - negatives.detach()).mean()
    loss = correct + ranking
    values = torch.stack((loss, correct, wrong, generic, zero, ranking))
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("non-finite PICRD relation objective")
    return {
        "loss": loss,
        "correct": correct,
        "wrong_rgb": wrong,
        "generic": generic,
        "zero": zero,
        "ranking": ranking,
        "wrong_shift": shift,
        "common_valid": common_valid,
    }


class PoseClipRelationCache:
    """Strict in-memory loader for a fresh exp408 NPZ cache."""

    def __init__(self, path, expected_sha256):
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if not expected_sha256:
            raise ValueError("PICRD cache SHA256 is required")
        if sha256_file(path) != expected_sha256:
            raise RuntimeError("PICRD cache SHA256 mismatch")
        with np.load(str(path), allow_pickle=False) as arrays:
            required = {"schema", "relative_paths", "features", "valid"}
            if set(arrays.files) != required:
                raise RuntimeError("unexpected PICRD cache fields")
            schema = str(arrays["schema"].item())
            paths = arrays["relative_paths"]
            features = arrays["features"]
            valid = arrays["valid"]
            if schema != "exp408-picrd-cache-v1":
                raise RuntimeError("unsupported PICRD cache schema")
            if paths.ndim != 1 or paths.dtype.kind != "U":
                raise RuntimeError("PICRD paths must be a unicode vector")
            if features.shape != (len(paths), 5, 768) or features.dtype != np.float16:
                raise RuntimeError("PICRD feature shape/dtype mismatch")
            if valid.shape != (len(paths), 5) or valid.dtype != np.bool_:
                raise RuntimeError("PICRD validity shape/dtype mismatch")
            path_tuple = tuple(str(value) for value in paths.tolist())
            if len(set(path_tuple)) != len(path_tuple):
                raise RuntimeError("PICRD cache contains duplicate paths")
            if not np.isfinite(features).all():
                raise RuntimeError("PICRD cache contains non-finite features")
            self.features = torch.from_numpy(features.copy())
            self.valid = torch.from_numpy(valid.copy())
        self.path = path
        self.sha256 = expected_sha256
        self.paths = path_tuple
        self.index = {value: row for row, value in enumerate(path_tuple)}

    def __len__(self):
        return len(self.paths)

    def lookup(self, relative_paths):
        try:
            rows = [self.index[str(value)] for value in relative_paths]
        except KeyError as error:
            raise KeyError("PICRD cache path is missing: {}".format(error.args[0]))
        index = torch.as_tensor(rows, dtype=torch.long)
        return self.features.index_select(0, index), self.valid.index_select(0, index)


def isolated_attention_mask(patch_selected, heads):
    """Restrict CLS and selected patches to one anatomical region."""
    if patch_selected.ndim != 2 or patch_selected.dtype != torch.bool:
        raise ValueError("patch_selected must be a boolean matrix")
    if patch_selected.shape[1] != CLIP_GRID * CLIP_GRID:
        raise ValueError("expected a 16x16 CLIP patch grid")
    selected = torch.cat(
        (
            torch.ones(
                len(patch_selected),
                1,
                dtype=torch.bool,
                device=patch_selected.device,
            ),
            patch_selected,
        ),
        dim=1,
    )
    allowed = selected.unsqueeze(2) & selected.unsqueeze(1)
    allowed |= torch.eye(
        selected.shape[1], dtype=torch.bool, device=selected.device
    ).unsqueeze(0)
    mask = torch.zeros(
        allowed.shape, dtype=torch.float32, device=allowed.device
    )
    mask.masked_fill_(~allowed, float("-inf"))
    return mask.repeat_interleave(int(heads), dim=0)


class RegionIsolatedClipVisualTeacher:
    """Frozen ViT-L/14 five-slot visual readout isolated from block one."""

    def __init__(
        self,
        checkpoint,
        checkpoint_sha256,
        device,
        *,
        microbatch=1,
    ):
        checkpoint = Path(checkpoint).expanduser().resolve()
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
            raise RuntimeError("region isolation requires token pooling")
        if getattr(model.visual, "attn_pool", None) is not None:
            raise RuntimeError("region isolation forbids extra attention pooling")
        normalizers = [
            transform
            for transform in preprocess.transforms
            if hasattr(transform, "mean") and hasattr(transform, "std")
        ]
        if len(normalizers) != 1:
            raise RuntimeError("could not identify CLIP normalization")
        if any(
            abs(float(value) - expected) > 1e-8
            for value, expected in zip(normalizers[0].mean, CLIP_MEAN)
        ) or any(
            abs(float(value) - expected) > 1e-8
            for value, expected in zip(normalizers[0].std, CLIP_STD)
        ):
            raise RuntimeError("CLIP normalization mismatch")
        self.visual = model.visual
        self.device = torch.device(device)
        self.microbatch = int(microbatch)
        self.mean = torch.tensor(CLIP_MEAN, device=self.device).view(
            1, 3, 1, 1
        )
        self.std = torch.tensor(CLIP_STD, device=self.device).view(
            1, 3, 1, 1
        )
        del model

    def _letterbox(self, rgb, masks):
        resized = F.interpolate(
            rgb.float(),
            size=(CLIP_SIZE, 75),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0, 1)
        canvas = self.mean.expand(len(rgb), 3, CLIP_SIZE, CLIP_SIZE).clone()
        canvas[:, :, :, 74:149] = resized
        resized_masks = F.interpolate(
            masks.float(), size=(CLIP_SIZE, 75), mode="nearest"
        )
        mask_canvas = torch.zeros(
            len(rgb),
            5,
            CLIP_SIZE,
            CLIP_SIZE,
            device=rgb.device,
            dtype=torch.float32,
        )
        mask_canvas[:, :, :, 74:149] = resized_masks
        patch_mass = F.avg_pool2d(
            mask_canvas, kernel_size=CLIP_PATCH, stride=CLIP_PATCH
        )
        return (canvas - self.mean) / self.std, patch_mass.flatten(2) > 0

    def _project(self, tokens):
        pooled, _ = self.visual._pool(tokens)
        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj
        return F.normalize(pooled.float(), dim=-1)

    def _encode_chunk(self, rgb, masks):
        normalized, selected = self._letterbox(rgb, masks)
        tokens = self.visual._embeds(normalized)
        batch, sequence, width = tokens.shape
        branches = tokens[:, None].expand(
            batch, 5, sequence, width
        ).reshape(batch * 5, sequence, width).clone()
        flat_selected = selected.reshape(batch * 5, -1)
        valid = flat_selected.any(dim=-1)
        output = torch.zeros(
            batch * 5,
            self.visual.output_dim,
            device=self.device,
            dtype=torch.float32,
        )
        if bool(valid.any()):
            active = branches[valid]
            active_mask = flat_selected[valid]
            blocks = self.visual.transformer.resblocks
            attention = isolated_attention_mask(
                active_mask, int(blocks[0].attn.num_heads)
            )
            for block in blocks:
                active = block(active, attn_mask=attention)
            output[valid] = self._project(active)
        return output.view(batch, 5, -1), valid.view(batch, 5)

    @torch.inference_mode()
    def encode(self, rgb, masks):
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 384, 128):
            raise ValueError("rgb must have shape [B,3,384,128]")
        if masks.shape != (len(rgb), 5, 96, 32):
            raise ValueError("mask shape mismatch")
        if rgb.device != self.device or masks.device != self.device:
            raise ValueError("teacher inputs must be on its device")
        feature_parts = []
        valid_parts = []
        for start in range(0, len(rgb), self.microbatch):
            stop = min(start + self.microbatch, len(rgb))
            features, valid = self._encode_chunk(
                rgb[start:stop], masks[start:stop]
            )
            feature_parts.append(features)
            valid_parts.append(valid)
        features = torch.cat(feature_parts, dim=0)
        valid = torch.cat(valid_parts, dim=0)
        if not bool(torch.isfinite(features).all()):
            raise RuntimeError("non-finite CLIP slot features")
        return features, valid
