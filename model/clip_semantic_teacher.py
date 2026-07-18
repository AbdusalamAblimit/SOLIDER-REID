"""Frozen training-only PC-MBCLS teacher for semantic TAPF."""

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


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
    "head, face, and hair",
    "chest, abdomen, waist, and torso",
    "upper limbs, arms, elbows, forearms, wrists, and hands",
    "thighs between the hips and knees",
    "lower legs below the knees, including shins, calves, ankles, and feet",
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
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
TEMPERATURE = 0.07
REGIONS = 5
SPLIT_BLOCK = 20
LEAK_RATIO = 0.01


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_hard_owner_regions(
    keypoints,
    scores,
    valid,
    image_hw=(384, 128),
    field_hw=(96, 32),
    sigma=1.5,
):
    """Render the frozen five-slot ontology used by the B2-SI audit."""
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 2):
        raise ValueError("keypoints must have shape [B,17,2]")
    image_height, image_width = image_hw
    field_height, field_width = field_hw
    reliability = valid.bool().float() * scores.float().clamp(0.0, 1.0)
    points = keypoints.float().clone()
    points[..., 0] *= (field_width - 1) / float(max(image_width - 1, 1))
    points[..., 1] *= (field_height - 1) / float(max(image_height - 1, 1))
    grid_y = torch.arange(
        field_height, device=points.device, dtype=torch.float32
    ).view(1, 1, field_height, 1)
    grid_x = torch.arange(
        field_width, device=points.device, dtype=torch.float32
    ).view(1, 1, 1, field_width)
    joint_distance = (grid_x - points[..., 0, None, None]).square()
    joint_distance = joint_distance + (
        grid_y - points[..., 1, None, None]
    ).square()
    joints = torch.exp(-joint_distance / (2.0 * float(sigma) ** 2))
    joints = joints * reliability[..., None, None]

    px = grid_x[:, 0]
    py = grid_y[:, 0]
    raw = []
    region_valid = []
    for region, (joint_ids, segments) in enumerate(
        zip(REGION_JOINTS, REGION_SEGMENTS)
    ):
        index = torch.as_tensor(joint_ids, device=points.device)
        joint_mask = joints.index_select(1, index).amax(dim=1)
        segment_mask = torch.zeros_like(joint_mask)
        for left, right in segments:
            ax = points[:, left, 0, None, None]
            ay = points[:, left, 1, None, None]
            bx = points[:, right, 0, None, None]
            by = points[:, right, 1, None, None]
            dx = bx - ax
            dy = by - ay
            denominator = (dx.square() + dy.square()).clamp_min(1e-6)
            projection = ((px - ax) * dx + (py - ay) * dy) / denominator
            if region >= 2:
                projection = projection.clamp(0.15, 0.85)
            else:
                projection = projection.clamp(0.0, 1.0)
            nearest_x = ax + projection * dx
            nearest_y = ay + projection * dy
            distance = (px - nearest_x).square() + (py - nearest_y).square()
            amplitude = torch.minimum(
                reliability[:, left], reliability[:, right]
            )[:, None, None]
            tube = torch.exp(-distance / (2.0 * float(sigma) ** 2))
            segment_mask = torch.maximum(segment_mask, tube * amplitude)
        raw.append(torch.maximum(joint_mask, segment_mask))
        region_valid.append(reliability.index_select(1, index).amax(dim=1) > 0)

    raw = torch.stack(raw, dim=1)
    total = raw.sum(dim=1, keepdim=True)
    amplitude = total.clamp(max=1.0)
    owner = raw.argmax(dim=1, keepdim=True)
    masks = torch.zeros_like(raw).scatter_(1, owner, amplitude)
    return masks, torch.stack(region_valid, dim=1)


def _regional_log_prior(masks):
    flat = masks.flatten(2).float()
    mass = flat.sum(dim=-1, keepdim=True)
    maximum = flat.amax(dim=-1, keepdim=True)
    valid = maximum.squeeze(-1) > 0
    delta = LEAK_RATIO * mass / float(flat.shape[-1])
    tiny = torch.finfo(flat.dtype).tiny
    prior = torch.log(
        (flat + delta).clamp_min(tiny) / (maximum + delta).clamp_min(tiny)
    )
    prior = torch.where(valid[..., None], prior, torch.zeros_like(prior))
    return prior, valid


def _additive_cls_mask(prior, heads):
    branches, patches = prior.shape
    sequence = patches + 1
    mask = torch.zeros(
        branches,
        sequence,
        sequence,
        device=prior.device,
        dtype=prior.dtype,
    )
    mask[:, 0, 1:] = prior
    return mask.repeat_interleave(int(heads), dim=0)


def _project_pooled(visual, tokens):
    pooled, _ = visual._pool(tokens)
    if visual.proj is not None:
        pooled = pooled @ visual.proj
    return F.normalize(pooled.float(), dim=-1)


class FrozenClipSlotTeacher:
    """Non-registered teacher returning one aligned mask/q/valid slot state."""

    def __init__(self, checkpoint, checkpoint_sha256, device, microbatch=4):
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if sha256_file(checkpoint) != checkpoint_sha256:
            raise RuntimeError("CLIP checkpoint SHA mismatch")
        if microbatch <= 0:
            raise ValueError("CLIP microbatch must be positive")
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=str(checkpoint)
        )
        model = model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if len(model.visual.transformer.resblocks) != 24:
            raise RuntimeError("Expected 24 ViT-L/14 blocks")
        if tuple(model.visual.grid_size) != (16, 16):
            raise RuntimeError("Expected 16x16 ViT-L/14 patch grid")

        prompts = []
        for phrase in REGION_PHRASES:
            for visible, occluded in SUPPORT_PROMPT_PAIRS:
                prompts.extend((visible.format(phrase), occluded.format(phrase)))
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        with torch.inference_mode():
            text = model.encode_text(tokenizer(prompts).to(device))
        text = F.normalize(text.float(), dim=-1)
        text = text.view(REGIONS, len(SUPPORT_PROMPT_PAIRS), 2, -1)
        self.text = F.normalize(text.mean(dim=1), dim=-1)
        self.visual = model.visual
        del model

        normalizers = [
            transform
            for transform in preprocess.transforms
            if hasattr(transform, "mean") and hasattr(transform, "std")
        ]
        if len(normalizers) != 1:
            raise RuntimeError("Could not identify official CLIP normalization")
        if any(
            abs(float(actual) - expected) > 1e-8
            for actual, expected in zip(normalizers[0].mean, CLIP_MEAN)
        ):
            raise RuntimeError("Official CLIP mean mismatch")
        if any(
            abs(float(actual) - expected) > 1e-8
            for actual, expected in zip(normalizers[0].std, CLIP_STD)
        ):
            raise RuntimeError("Official CLIP std mismatch")
        self.mean = torch.as_tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.as_tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        self.device = torch.device(device)
        self.microbatch = int(microbatch)
        self.checkpoint_sha256 = checkpoint_sha256

    def _region_features(self, images, masks):
        tokens = self.visual._embeds(images)
        blocks = self.visual.transformer.resblocks
        for block in blocks[:SPLIT_BLOCK]:
            tokens = block(tokens)
        prior, valid = _regional_log_prior(masks)
        batch, sequence, width = tokens.shape
        branches = (
            tokens[:, None]
            .expand(batch, REGIONS, sequence, width)
            .reshape(batch * REGIONS, sequence, width)
            .clone()
        )
        flat_valid = valid.reshape(-1)
        output = torch.zeros(
            batch * REGIONS,
            self.visual.output_dim,
            device=tokens.device,
            dtype=torch.float32,
        )
        if bool(flat_valid.any()):
            selected = branches[flat_valid]
            selected_prior = prior.reshape(batch * REGIONS, -1)[flat_valid]
            attention_mask = None
            if not bool((selected_prior == 0).all()):
                heads = blocks[0].attn.num_heads
                attention_mask = _additive_cls_mask(selected_prior, heads)
            for block in blocks[SPLIT_BLOCK:]:
                selected = block(selected, attn_mask=attention_mask)
            output[flat_valid] = _project_pooled(self.visual, selected)
        return output.view(batch, REGIONS, -1), valid

    @torch.inference_mode()
    def __call__(self, teacher_rgb, keypoints, scores, valid):
        if teacher_rgb.ndim != 4 or teacher_rgb.shape[1:] != (3, 384, 128):
            raise ValueError("teacher_rgb must have shape [B,3,384,128]")
        masks, region_valid = render_hard_owner_regions(
            keypoints,
            scores,
            valid,
            image_hw=(384, 128),
            field_hw=(96, 32),
            sigma=1.5,
        )
        q_parts = []
        valid_parts = []
        for start in range(0, len(teacher_rgb), self.microbatch):
            stop = min(start + self.microbatch, len(teacher_rgb))
            rgb = teacher_rgb[start:stop].float()
            resized = F.interpolate(
                rgb,
                size=(224, 75),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            clip_images = self.mean.expand(stop - start, 3, 224, 224).clone()
            clip_images[:, :, :, 74:149] = resized.clamp(0.0, 1.0)
            region_masks = F.interpolate(
                masks[start:stop].float(), size=(224, 75), mode="nearest"
            )
            clip_masks = torch.zeros(
                stop - start,
                REGIONS,
                224,
                224,
                device=self.device,
                dtype=torch.float32,
            )
            clip_masks[:, :, :, 74:149] = region_masks
            grid_masks = F.avg_pool2d(clip_masks, kernel_size=14, stride=14)
            normalized = (clip_images - self.mean) / self.std
            features, readout_valid = self._region_features(
                normalized, grid_masks
            )
            logits = torch.einsum("brd,rsd->brs", features, self.text)
            q_visible = torch.softmax(logits / TEMPERATURE, dim=-1)[..., 0]
            q_parts.append(q_visible.float())
            valid_parts.append(readout_valid & region_valid[start:stop])
        q_visible = torch.cat(q_parts, dim=0)
        semantic_valid = torch.cat(valid_parts, dim=0)
        q_visible = torch.where(
            semantic_valid, q_visible, torch.zeros_like(q_visible)
        )
        if not bool(torch.isfinite(q_visible).all()):
            raise RuntimeError("Non-finite frozen CLIP targets")
        mean_q_visible = torch.zeros((), device=q_visible.device)
        if bool(semantic_valid.any()):
            mean_q_visible = q_visible[semantic_valid].mean()
        return {
            "q_visible": q_visible,
            "valid": semantic_valid,
            "mean_q_visible": mean_q_visible,
            "region_masks": masks,
        }
