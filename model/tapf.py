"""Minimal training-privileged, inference-RGB-only TAPF components."""

import torch
import torch.nn as nn
import torch.nn.functional as F


ANATOMICAL_REGION_JOINTS = (
    (0, 1, 2, 3, 4),
    (5, 6, 11, 12),
    (7, 8, 9, 10),
    (13, 14),
    (15, 16),
)


def render_pose_field(keypoints, scores, valid, image_hw, field_hw, sigma):
    """Render paired COCO-17 coordinates as reliability-scaled Gaussians."""
    image_height, image_width = image_hw
    field_height, field_width = field_hw
    if min(image_height, image_width, field_height, field_width) <= 0:
        raise ValueError("Image and field sizes must be positive")
    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive")
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 2):
        raise ValueError("keypoints must have shape [B, 17, 2]")
    if scores.shape != keypoints.shape[:2] or valid.shape != keypoints.shape[:2]:
        raise ValueError("scores and valid must have shape [B, 17]")

    keypoints = keypoints.float()
    scores = scores.float()
    valid = valid.bool()
    reliability = valid.float() * scores.clamp(0.0, 1.0)

    scale_x = (field_width - 1) / float(max(image_width - 1, 1))
    scale_y = (field_height - 1) / float(max(image_height - 1, 1))
    center_x = keypoints[..., 0] * scale_x
    center_y = keypoints[..., 1] * scale_y
    grid_y = torch.arange(
        field_height, device=keypoints.device, dtype=torch.float32
    ).view(1, 1, field_height, 1)
    grid_x = torch.arange(
        field_width, device=keypoints.device, dtype=torch.float32
    ).view(1, 1, 1, field_width)
    distance = (grid_x - center_x[..., None, None]).square()
    distance = distance + (grid_y - center_y[..., None, None]).square()
    gaussian = torch.exp(-distance / (2.0 * float(sigma) ** 2))
    teacher_field = gaussian * reliability[..., None, None]
    return gaussian, teacher_field, reliability


def aggregate_joint_field_to_regions(field):
    """Bind the 17 joint channels to five fixed, non-permutable regions."""
    if field.ndim != 4 or field.shape[1] != 17:
        raise ValueError("field must have shape [B,17,H,W]")
    return torch.stack(
        [field[:, indices].amax(dim=1) for indices in ANATOMICAL_REGION_JOINTS],
        dim=1,
    )


def build_matched_training_donor_map(identities, cameras):
    """Select the first cyclic same-camera, different-identity batch donor."""
    if identities.ndim != 1 or cameras.shape != identities.shape:
        raise ValueError("identities/cameras must be aligned one-dimensional tensors")
    count = identities.numel()
    if count < 2:
        return torch.full_like(identities, -1, dtype=torch.long)
    row = torch.arange(count, device=identities.device)
    offsets = torch.arange(1, count, device=identities.device)
    candidates = (row[:, None] + offsets[None, :]) % count
    matches = cameras[candidates].eq(cameras[:, None])
    matches = matches & identities[candidates].ne(identities[:, None])
    eligible = matches.any(dim=1)
    first_offset = matches.to(torch.int64).argmax(dim=1)
    selected = candidates.gather(1, first_offset[:, None]).squeeze(1)
    return torch.where(eligible, selected, torch.full_like(selected, -1))


class PoseAnchor(nn.Module):
    def __init__(self, in_channels=384, hidden_channels=128, joint_count=17):
        super().__init__()
        if hidden_channels % 16 != 0:
            raise ValueError("hidden_channels must be divisible by 16")
        self.joint_count = joint_count
        self.project = nn.Conv2d(in_channels, hidden_channels, 1)
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.norm = nn.GroupNorm(16, hidden_channels)
        self.activation = nn.GELU()
        self.head = nn.Conv2d(hidden_channels, 2 * joint_count, 1)

    def forward(self, feature):
        hidden = self.activation(self.norm(self.depthwise(self.project(feature))))
        output = self.head(hidden)
        heatmap_logits = output[:, : self.joint_count]
        confidence_logits = F.adaptive_avg_pool2d(
            output[:, self.joint_count :], output_size=1
        ).flatten(1)
        heatmaps = torch.sigmoid(heatmap_logits.float())
        confidence = torch.sigmoid(confidence_logits.float())
        field = heatmaps * confidence[..., None, None]
        return {
            "heatmap_logits": heatmap_logits,
            "confidence_logits": confidence_logits,
            "heatmaps": heatmaps,
            "confidence": confidence,
            "field": field,
        }


class PoseSpatialGate(nn.Module):
    def __init__(
        self,
        feature_channels=768,
        joint_count=17,
        hidden_channels=32,
        release=0.5,
    ):
        super().__init__()
        if hidden_channels % 8 != 0:
            raise ValueError("hidden_channels must be divisible by 8")
        if not 0.0 < release <= 1.0:
            raise ValueError("release must be in (0, 1]")
        self.feature_channels = feature_channels
        self.release = float(release)
        self.input_projection = nn.Conv2d(
            joint_count, hidden_channels, kernel_size=1, bias=False
        )
        self.norm = nn.GroupNorm(8, hidden_channels, affine=False)
        self.activation = nn.GELU()
        self.output_projection = nn.Conv2d(
            hidden_channels, feature_channels, kernel_size=1, bias=False
        )
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, tokens, hw_shape, field):
        height, width = hw_shape
        if tokens.ndim != 3 or tokens.shape[1] != height * width:
            raise ValueError("Token shape does not match hw_shape")
        if tokens.shape[2] != self.feature_channels:
            raise ValueError("Unexpected token channel count")
        resized = F.interpolate(
            field.float(), size=(height, width), mode="bilinear", align_corners=False
        )
        resized = resized.to(dtype=tokens.dtype)
        delta = self.output_projection(
            self.activation(self.norm(self.input_projection(resized)))
        )
        delta = delta.flatten(2).transpose(1, 2).contiguous()
        gate = 1.0 + self.release * torch.tanh(delta)
        return tokens * gate, delta


class SemanticPoseAnchor(nn.Module):
    """D0-compatible pose anchor plus one executable five-slot state."""

    def __init__(self, in_channels=384, hidden_channels=128):
        super().__init__()
        if hidden_channels % 16 != 0:
            raise ValueError("hidden_channels must be divisible by 16")
        self.project = nn.Conv2d(in_channels, hidden_channels, 1)
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.norm = nn.GroupNorm(16, hidden_channels)
        self.activation = nn.GELU()
        self.pose_head = nn.Conv2d(hidden_channels, 34, 1)
        self.region_mask_head = nn.Conv2d(hidden_channels, 5, 1)
        self.support_head = nn.Conv2d(hidden_channels, 5, 1)
        self.presence_head = nn.Conv2d(hidden_channels, 5, 1)
        nn.init.zeros_(self.region_mask_head.weight)
        nn.init.zeros_(self.region_mask_head.bias)
        nn.init.zeros_(self.support_head.weight)
        nn.init.zeros_(self.support_head.bias)
        nn.init.zeros_(self.presence_head.weight)
        nn.init.zeros_(self.presence_head.bias)

    def forward(self, feature):
        hidden = self.activation(self.norm(self.depthwise(self.project(feature))))
        pose_output = self.pose_head(hidden)
        heatmap_logits = pose_output[:, :17]
        confidence_logits = F.adaptive_avg_pool2d(
            pose_output[:, 17:], output_size=1
        ).flatten(1)
        region_mask_logits = self.region_mask_head(hidden)
        support_logits = F.adaptive_avg_pool2d(
            self.support_head(hidden), output_size=1
        ).flatten(1)
        presence_logits = F.adaptive_avg_pool2d(
            self.presence_head(hidden), output_size=1
        ).flatten(1)
        heatmaps = torch.sigmoid(heatmap_logits.float())
        confidence = torch.sigmoid(confidence_logits.float())
        joint_field = heatmaps * confidence[..., None, None]
        region_mask = torch.sigmoid(region_mask_logits.float())
        support = torch.sigmoid(support_logits.float())
        presence_probability = torch.sigmoid(presence_logits.float())
        presence_hard = (presence_probability > 0.5).float()
        presence = presence_hard.detach() - presence_probability.detach()
        presence = presence + presence_probability
        field = region_mask * support[..., None, None]
        field = field * presence[..., None, None]
        return {
            "heatmap_logits": heatmap_logits,
            "confidence_logits": confidence_logits,
            "heatmaps": heatmaps,
            "confidence": confidence,
            "joint_field": joint_field,
            "region_mask_logits": region_mask_logits,
            "region_mask": region_mask,
            "support_logits": support_logits,
            "support": support,
            "presence_logits": presence_logits,
            "presence_probability": presence_probability,
            "presence": presence,
            "field": field,
        }


class SemanticSpatialRouter(nn.Module):
    """Feature-dependent gather-transform-scatter router for fixed slots."""

    def __init__(
        self,
        feature_channels=768,
        region_count=5,
        rank=16,
        release=0.5,
        rezero=False,
    ):
        super().__init__()
        if feature_channels <= 0 or region_count <= 0 or rank <= 0:
            raise ValueError("Router dimensions must be positive")
        if not 0.0 < release <= 1.0:
            raise ValueError("release must be in (0, 1]")
        self.feature_channels = int(feature_channels)
        self.region_count = int(region_count)
        self.rank = int(rank)
        self.release = float(release)
        self.rezero = bool(rezero)
        self.token_projection = nn.Linear(feature_channels, rank, bias=False)
        self.context_projection = nn.Linear(feature_channels, rank, bias=False)
        if self.rezero:
            self.expert = nn.Parameter(
                torch.empty(region_count, rank, feature_channels)
            )
            cpu_rng_state = torch.get_rng_state()
            nn.init.normal_(self.expert, mean=0.0, std=0.02)
            torch.set_rng_state(cpu_rng_state)
            self.alpha_logit = nn.Parameter(torch.zeros(()))
        else:
            self.expert = nn.Parameter(
                torch.zeros(region_count, rank, feature_channels)
            )
            self.register_parameter("alpha_logit", None)

    def forward(self, tokens, hw_shape, mask, support):
        height, width = hw_shape
        if tokens.ndim != 3 or tokens.shape[1] != height * width:
            raise ValueError("Token shape does not match hw_shape")
        if tokens.shape[2] != self.feature_channels:
            raise ValueError("Unexpected token channel count")
        if mask.ndim != 4 or mask.shape[1] != self.region_count:
            raise ValueError("mask must have shape [B,R,H,W]")
        if support.shape != mask.shape[:2]:
            raise ValueError("support must have shape [B,R]")

        resized = F.interpolate(
            mask.float(), size=(height, width), mode="bilinear", align_corners=False
        ).flatten(2)
        resized = resized.to(dtype=tokens.dtype)
        support = support.to(dtype=tokens.dtype).clamp(0.0, 1.0)
        mass = resized.sum(dim=-1, keepdim=True)
        normalized = resized / mass.clamp_min(1e-6)
        normalized = torch.where(
            mass > 0, normalized, torch.zeros_like(normalized)
        )
        context = torch.einsum("brn,bnc->brc", normalized, tokens)
        token_latent = self.token_projection(tokens)
        context_latent = self.context_projection(context)
        hidden = F.gelu(token_latent[:, None] + context_latent[:, :, None])
        region_delta = torch.einsum("brnk,rkc->brnc", hidden, self.expert)
        scatter = resized * support[..., None]
        delta = torch.einsum("brn,brnc->bnc", scatter, region_delta)
        if self.rezero:
            applied_delta = torch.tanh(self.alpha_logit) * torch.tanh(delta)
            routed = tokens + self.release * applied_delta
            return routed, applied_delta
        routed = tokens + self.release * torch.tanh(delta)
        return routed, delta


class CleanTapfD0(nn.Module):
    def __init__(
        self,
        anchor_channels=384,
        anchor_hidden=128,
        consumer_channels=768,
        psg_hidden=32,
        gaussian_sigma=1.5,
        gate_release=0.5,
        teacher_epochs=5,
        handoff_epochs=5,
    ):
        super().__init__()
        if teacher_epochs < 0 or handoff_epochs <= 0:
            raise ValueError("Invalid TAPF handoff schedule")
        self.gaussian_sigma = float(gaussian_sigma)
        self.teacher_epochs = int(teacher_epochs)
        self.handoff_epochs = int(handoff_epochs)
        self.anchor = PoseAnchor(anchor_channels, anchor_hidden)
        self.psg_bank = nn.ModuleList(
            [
                PoseSpatialGate(
                    feature_channels=consumer_channels,
                    hidden_channels=psg_hidden,
                    release=gate_release,
                )
                for _ in range(2)
            ]
        )

    def student_fraction(self, epoch):
        if epoch is None:
            raise ValueError("Training TAPF requires an epoch")
        if epoch <= self.teacher_epochs:
            return 0.0
        fraction = (epoch - self.teacher_epochs) / float(self.handoff_epochs)
        return max(0.0, min(1.0, fraction))

    def _prepare_with_anchor(
        self, anchor, source_feature, pose_batch, image_hw, epoch, training
    ):
        prediction = anchor(source_feature.detach())
        student_field = prediction["field"]
        if not training:
            return {
                "consumer_field": student_field.detach(),
                "student_field": student_field,
                "pose_loss": None,
                "student_fraction": 1.0,
                "teacher_field": None,
                "reliability": None,
                "gate_deltas": [],
            }
        if pose_batch is None:
            raise ValueError("Training TAPF requires paired pose targets")

        gaussian, teacher_field, reliability = render_pose_field(
            pose_batch["keypoints"],
            pose_batch["scores"],
            pose_batch["valid"],
            image_hw=image_hw,
            field_hw=student_field.shape[-2:],
            sigma=self.gaussian_sigma,
        )
        heatmap_weight = reliability[..., None, None]
        heatmap_denominator = (
            heatmap_weight.sum() * gaussian.shape[-2] * gaussian.shape[-1]
        ).clamp_min(1.0)
        heatmap_loss = (
            (prediction["heatmaps"] - gaussian).square() * heatmap_weight
        ).sum() / heatmap_denominator
        confidence_loss = F.binary_cross_entropy_with_logits(
            prediction["confidence_logits"].float(), reliability
        )
        pose_loss = heatmap_loss + confidence_loss

        fraction = self.student_fraction(epoch)
        consumer_field = (
            (1.0 - fraction) * teacher_field + fraction * student_field
        ).detach()
        return {
            "consumer_field": consumer_field,
            "student_field": student_field,
            "pose_loss": pose_loss,
            "heatmap_loss": heatmap_loss,
            "confidence_loss": confidence_loss,
            "student_fraction": fraction,
            "teacher_field": teacher_field,
            "reliability": reliability,
            "gate_deltas": [],
        }

    def prepare(self, source_feature, pose_batch, image_hw, epoch, training):
        return self._prepare_with_anchor(
            self.anchor,
            source_feature,
            pose_batch=pose_batch,
            image_hw=image_hw,
            epoch=epoch,
            training=training,
        )

    def apply_gate(self, bank_index, tokens, hw_shape, state):
        gated, delta = self.psg_bank[bank_index](
            tokens, hw_shape, state["consumer_field"]
        )
        state["gate_deltas"].append(delta)
        return gated


class CleanSemanticTapfC0(nn.Module):
    """Single-stage CLIP-calibrated anatomical state with two live routers."""

    semantic = True

    def __init__(
        self,
        anchor_channels=384,
        anchor_hidden=128,
        consumer_channels=768,
        router_rank=16,
        router_rezero=False,
        gate_release=0.5,
        gaussian_sigma=1.5,
        teacher_epochs=5,
        handoff_epochs=5,
    ):
        super().__init__()
        if teacher_epochs < 0 or handoff_epochs <= 0:
            raise ValueError("Invalid TAPF handoff schedule")
        self.gaussian_sigma = float(gaussian_sigma)
        self.teacher_epochs = int(teacher_epochs)
        self.handoff_epochs = int(handoff_epochs)
        self.anchor = SemanticPoseAnchor(anchor_channels, anchor_hidden)
        self.psg_bank = nn.ModuleList(
            [
                SemanticSpatialRouter(
                    feature_channels=consumer_channels,
                    rank=router_rank,
                    release=gate_release,
                    rezero=router_rezero,
                )
                for _ in range(2)
            ]
        )

    def student_fraction(self, epoch):
        if epoch is None:
            raise ValueError("Training TAPF requires an epoch")
        if epoch <= self.teacher_epochs:
            return 0.0
        fraction = (epoch - self.teacher_epochs) / float(self.handoff_epochs)
        return max(0.0, min(1.0, fraction))

    def prepare(self, source_feature, pose_batch, image_hw, epoch, training):
        prediction = self.anchor(source_feature.detach())
        student_mask = prediction["region_mask"]
        student_support = prediction["support"]
        student_presence = prediction["presence"]
        student_execution_mask = student_mask * student_presence[..., None, None]
        student_execution_support = student_support * student_presence
        if not training:
            return {
                "consumer_mask": student_execution_mask.detach(),
                "consumer_support": student_support.detach(),
                "consumer_presence": student_presence.detach(),
                "mixed_mask": student_mask.detach(),
                "mixed_support": student_support.detach(),
                "mixed_presence": student_presence.detach(),
                "consumer_field": prediction["field"].detach(),
                "student_field": prediction["field"],
                "student_mask": student_mask,
                "student_support": student_support,
                "student_execution_support": student_execution_support,
                "student_presence": student_presence,
                "pose_loss": None,
                "semantic_loss": None,
                "region_mask_loss": None,
                "presence_loss": None,
                "q_loss": None,
                "student_fraction": 1.0,
                "teacher_field": None,
                "teacher_mask": None,
                "teacher_support": None,
                "teacher_presence": None,
                "reliability": student_execution_support,
                "gate_deltas": [],
            }
        if pose_batch is None:
            raise ValueError("Training semantic TAPF requires paired pose targets")
        required_semantic = {
            "semantic_q_visible",
            "semantic_valid",
            "semantic_teacher_mask",
        }
        missing_semantic = sorted(required_semantic.difference(pose_batch))
        if missing_semantic:
            raise ValueError(
                "Training semantic TAPF is missing frozen targets: "
                + ", ".join(missing_semantic)
            )

        gaussian, _, reliability = render_pose_field(
            pose_batch["keypoints"],
            pose_batch["scores"],
            pose_batch["valid"],
            image_hw=image_hw,
            field_hw=student_mask.shape[-2:],
            sigma=self.gaussian_sigma,
        )
        heatmap_weight = reliability[..., None, None]
        heatmap_denominator = (
            heatmap_weight.sum() * gaussian.shape[-2] * gaussian.shape[-1]
        ).clamp_min(1.0)
        heatmap_loss = (
            (prediction["heatmaps"] - gaussian).square() * heatmap_weight
        ).sum() / heatmap_denominator
        confidence_loss = F.binary_cross_entropy_with_logits(
            prediction["confidence_logits"].float(), reliability
        )

        teacher_support = pose_batch["semantic_q_visible"].float()
        semantic_valid = pose_batch["semantic_valid"].bool()
        if teacher_support.shape != prediction["support_logits"].shape:
            raise ValueError("Frozen CLIP target shape mismatch")
        if semantic_valid.shape != teacher_support.shape:
            raise ValueError("Frozen CLIP validity shape mismatch")

        high_resolution_teacher_mask = pose_batch[
            "semantic_teacher_mask"
        ].float()
        expected_high_resolution = tuple(
            dimension * 4 for dimension in student_mask.shape[-2:]
        )
        if high_resolution_teacher_mask.shape[:2] != student_mask.shape[:2]:
            raise ValueError("Frozen semantic mask batch/slot shape mismatch")
        if high_resolution_teacher_mask.shape[-2:] != expected_high_resolution:
            raise ValueError(
                "Frozen semantic mask must be exactly 4x the anchor resolution"
            )
        teacher_mask = F.avg_pool2d(
            high_resolution_teacher_mask, kernel_size=4, stride=4
        )
        if teacher_mask.shape != student_mask.shape:
            raise RuntimeError("Frozen semantic mask downsampling contract failed")

        q_values = F.binary_cross_entropy_with_logits(
            prediction["support_logits"].float(),
            teacher_support,
            reduction="none",
        )
        q_loss = (
            q_values * semantic_valid.float()
        ).sum() / semantic_valid.float().sum().clamp_min(1.0)
        region_mask_loss = F.binary_cross_entropy_with_logits(
            prediction["region_mask_logits"].float(), teacher_mask
        )
        teacher_presence = semantic_valid.float()
        presence_loss = F.binary_cross_entropy_with_logits(
            prediction["presence_logits"].float(), teacher_presence
        )
        semantic_loss = torch.stack(
            [region_mask_loss, presence_loss, q_loss]
        ).mean()
        pose_loss = heatmap_loss + confidence_loss + semantic_loss

        teacher_support = torch.where(
            semantic_valid, teacher_support, torch.zeros_like(teacher_support)
        )
        fraction = self.student_fraction(epoch)
        consumer_mask = (
            (1.0 - fraction) * teacher_mask + fraction * student_mask
        ).detach()
        consumer_support = (
            (1.0 - fraction) * teacher_support + fraction * student_support
        ).detach()
        consumer_presence = (
            (1.0 - fraction) * teacher_presence
            + fraction * student_presence
        ).detach()
        execution_mask = consumer_mask * consumer_presence[..., None, None]
        execution_support = consumer_support * consumer_presence
        consumer_field = execution_mask * consumer_support[..., None, None]
        return {
            "consumer_mask": execution_mask,
            "consumer_support": consumer_support,
            "consumer_presence": consumer_presence,
            "mixed_mask": consumer_mask,
            "mixed_support": consumer_support,
            "mixed_presence": consumer_presence,
            "consumer_field": consumer_field,
            "student_field": prediction["field"],
            "student_mask": student_mask,
            "student_support": student_support,
            "student_execution_support": student_execution_support,
            "student_presence": student_presence,
            "pose_loss": pose_loss,
            "heatmap_loss": heatmap_loss,
            "confidence_loss": confidence_loss,
            "semantic_loss": semantic_loss,
            "region_mask_loss": region_mask_loss,
            "presence_loss": presence_loss,
            "q_loss": q_loss,
            "student_fraction": fraction,
            "teacher_field": teacher_mask
            * teacher_support[..., None, None]
            * teacher_presence[..., None, None],
            "teacher_mask": teacher_mask,
            "teacher_support": teacher_support,
            "teacher_presence": teacher_presence,
            "reliability": execution_support,
            "gate_deltas": [],
        }

    def apply_gate(self, bank_index, tokens, hw_shape, state):
        routed, delta = self.psg_bank[bank_index](
            tokens,
            hw_shape,
            state["consumer_mask"],
            state["consumer_support"],
        )
        state["gate_deltas"].append(delta)
        return routed


def relation_gram_loss(student, teacher, valid):
    """MSE between complete valid-slot cosine relations, excluding diagonal."""
    if student.ndim != 3 or teacher.ndim != 3:
        raise ValueError("student/teacher evidence must have shape [B,R,D]")
    if student.shape[:2] != teacher.shape[:2]:
        raise ValueError("student/teacher evidence must share [B,R]")
    if valid.shape != student.shape[:2]:
        raise ValueError("evidence validity must have shape [B,R]")
    student = student.reshape(-1, student.shape[-1])
    teacher = teacher.reshape(-1, teacher.shape[-1])
    valid = valid.reshape(-1).bool()
    if int(valid.sum()) < 2:
        return student.sum() * 0.0
    student = F.normalize(student[valid].float(), dim=-1)
    teacher = F.normalize(teacher[valid].float(), dim=-1)
    student_relation = student @ student.transpose(0, 1)
    teacher_relation = teacher @ teacher.transpose(0, 1)
    off_diagonal = ~torch.eye(
        student_relation.shape[0],
        device=student_relation.device,
        dtype=torch.bool,
    )
    return F.mse_loss(
        student_relation[off_diagonal], teacher_relation[off_diagonal]
    )


class RichEvidencePoseAnchor(nn.Module):
    """Pose/mask/presence anchor with an independently owned evidence head."""

    def __init__(
        self,
        in_channels=384,
        hidden_channels=128,
        region_count=5,
        evidence_dim=16,
    ):
        super().__init__()
        if hidden_channels % 16 != 0:
            raise ValueError("hidden_channels must be divisible by 16")
        if region_count <= 0 or evidence_dim <= 0:
            raise ValueError("region/evidence dimensions must be positive")
        self.region_count = int(region_count)
        self.evidence_dim = int(evidence_dim)
        self.project = nn.Conv2d(in_channels, hidden_channels, 1)
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.norm = nn.GroupNorm(16, hidden_channels)
        self.activation = nn.GELU()
        self.pose_head = nn.Conv2d(hidden_channels, 34, 1)
        self.region_mask_head = nn.Conv2d(hidden_channels, region_count, 1)
        self.presence_head = nn.Conv2d(hidden_channels, region_count, 1)
        self.evidence_head = nn.Linear(
            hidden_channels, region_count * evidence_dim
        )

    def forward(self, feature):
        hidden = self.activation(self.norm(self.depthwise(self.project(feature))))
        pose_output = self.pose_head(hidden)
        heatmap_logits = pose_output[:, :17]
        confidence_logits = F.adaptive_avg_pool2d(
            pose_output[:, 17:], output_size=1
        ).flatten(1)
        region_mask_logits = self.region_mask_head(hidden)
        presence_logits = F.adaptive_avg_pool2d(
            self.presence_head(hidden), output_size=1
        ).flatten(1)
        evidence = self.evidence_head(
            F.adaptive_avg_pool2d(hidden.detach(), output_size=1).flatten(1)
        ).view(-1, self.region_count, self.evidence_dim)
        evidence = F.normalize(evidence.float(), dim=-1)

        heatmaps = torch.sigmoid(heatmap_logits.float())
        confidence = torch.sigmoid(confidence_logits.float())
        joint_field = heatmaps * confidence[..., None, None]
        region_mask = torch.sigmoid(region_mask_logits.float())
        presence_probability = torch.sigmoid(presence_logits.float())
        presence_hard = (presence_probability > 0.5).float()
        presence = presence_hard.detach() - presence_probability.detach()
        presence = presence + presence_probability
        field = region_mask * presence[..., None, None]
        return {
            "heatmap_logits": heatmap_logits,
            "confidence_logits": confidence_logits,
            "heatmaps": heatmaps,
            "confidence": confidence,
            "joint_field": joint_field,
            "region_mask_logits": region_mask_logits,
            "region_mask": region_mask,
            "presence_logits": presence_logits,
            "presence_probability": presence_probability,
            "presence": presence,
            "evidence": evidence,
            "field": field,
        }


class EvidenceBudgetRouter(nn.Module):
    """One inference-retained evidence router with a fixed residual budget."""

    def __init__(
        self,
        feature_channels=768,
        region_count=5,
        rank=16,
        evidence_dim=16,
    ):
        super().__init__()
        if min(feature_channels, region_count, rank, evidence_dim) <= 0:
            raise ValueError("Router dimensions must be positive")
        self.feature_channels = int(feature_channels)
        self.region_count = int(region_count)
        self.rank = int(rank)
        self.evidence_dim = int(evidence_dim)
        self.token_projection = nn.Linear(feature_channels, rank, bias=False)
        self.context_projection = nn.Linear(feature_channels, rank, bias=False)
        self.evidence_projection = nn.Linear(evidence_dim, rank, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Linear(rank, feature_channels, bias=False)
                for _ in range(region_count)
            ]
        )
        cpu_rng_state = torch.get_rng_state()
        for expert in self.experts:
            nn.init.normal_(expert.weight, mean=0.0, std=0.02)
        torch.set_rng_state(cpu_rng_state)

    def branch(self, tokens, hw_shape, mask, presence, evidence):
        height, width = hw_shape
        if tokens.ndim != 3 or tokens.shape[1] != height * width:
            raise ValueError("Token shape does not match hw_shape")
        if tokens.shape[2] != self.feature_channels:
            raise ValueError("Unexpected token channel count")
        if mask.ndim != 4 or mask.shape[1] != self.region_count:
            raise ValueError("mask must have shape [B,R,H,W]")
        if presence.shape != mask.shape[:2]:
            raise ValueError("presence must have shape [B,R]")
        if evidence.shape != (
            tokens.shape[0], self.region_count, self.evidence_dim
        ):
            raise ValueError("evidence must have shape [B,R,D]")

        resized = F.interpolate(
            mask.detach().float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).flatten(2)
        presence = presence.detach().float().clamp(0.0, 1.0)
        mass = resized.sum(dim=-1, keepdim=True)
        normalized_mask = resized / mass.clamp_min(1e-6)
        normalized_mask = torch.where(
            mass > 0, normalized_mask, torch.zeros_like(normalized_mask)
        )
        context = torch.einsum(
            "brn,bnc->brc", normalized_mask.to(tokens.dtype), tokens
        )
        token_hidden = self.token_projection(tokens)
        context_hidden = self.context_projection(context)
        evidence_hidden = self.evidence_projection(evidence)

        proposals = []
        normalized_proposals = []
        slot_valid = (mass.squeeze(-1) > 0) & (presence > 0)
        for slot, expert in enumerate(self.experts):
            hidden = token_hidden
            hidden = hidden + context_hidden[:, slot, None]
            hidden = hidden + evidence_hidden[:, slot, None]
            proposal = expert(F.gelu(hidden))
            proposal_float = proposal.float()
            denominator = (
                proposal_float.square().mean(dim=-1, keepdim=True).sqrt().detach()
                + 1e-6
            )
            normalized = (proposal_float / denominator).to(tokens.dtype)
            normalized = torch.where(
                slot_valid[:, slot, None, None],
                normalized,
                torch.zeros_like(normalized),
            )
            proposals.append(proposal)
            normalized_proposals.append(normalized)

        proposal = torch.stack(proposals, dim=1)
        normalized_proposal = torch.stack(normalized_proposals, dim=1)
        scatter = resized[:, :, :, None] * presence[:, :, None, None]
        unit_delta = (
            scatter.to(normalized_proposal.dtype) * normalized_proposal
        ).sum(dim=1)
        if not bool(torch.isfinite(unit_delta).all()):
            raise RuntimeError("Non-finite evidence-budget router output")
        return {
            "proposal": proposal,
            "normalized_proposal": normalized_proposal,
            "unit_delta": unit_delta,
            "mask": resized,
            "normalized_mask": normalized_mask,
            "mass": mass,
            "slot_valid": slot_valid,
        }

    def forward(self, tokens, hw_shape, mask, presence, evidence, rho):
        branch = self.branch(tokens, hw_shape, mask, presence, evidence)
        applied_delta = float(rho) * branch["unit_delta"]
        return tokens + applied_delta, applied_delta, branch


class EvidenceOwnedLowRankRouter(nn.Module):
    """Shared low-rank production operator with evidence-owned coefficients."""

    def __init__(
        self,
        feature_channels=768,
        region_count=5,
        rank=16,
        evidence_dim=16,
    ):
        super().__init__()
        if min(feature_channels, region_count, rank, evidence_dim) <= 0:
            raise ValueError("Router dimensions must be positive")
        self.feature_channels = int(feature_channels)
        self.region_count = int(region_count)
        self.rank = int(rank)
        self.evidence_dim = int(evidence_dim)
        self.down_projection = nn.Linear(feature_channels, rank, bias=False)
        self.context_projection = nn.Linear(feature_channels, rank, bias=False)
        self.evidence_projection = nn.Linear(evidence_dim, rank, bias=False)
        self.up_projection = nn.Linear(rank, feature_channels, bias=False)
        self.context_query = nn.Linear(feature_channels, evidence_dim, bias=False)
        self.evidence_key = nn.Linear(evidence_dim, evidence_dim, bias=False)

    def branch(self, tokens, hw_shape, mask, presence, evidence):
        height, width = hw_shape
        if tokens.ndim != 3 or tokens.shape[1] != height * width:
            raise ValueError("Token shape does not match hw_shape")
        if tokens.shape[2] != self.feature_channels:
            raise ValueError("Unexpected token channel count")
        if mask.ndim != 4 or mask.shape[1] != self.region_count:
            raise ValueError("mask must have shape [B,R,H,W]")
        if presence.shape != mask.shape[:2]:
            raise ValueError("presence must have shape [B,R]")
        if evidence.shape != (
            tokens.shape[0], self.region_count, self.evidence_dim
        ):
            raise ValueError("evidence must have shape [B,R,D]")

        resized = F.interpolate(
            mask.detach().float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).flatten(2)
        presence = presence.detach().float().clamp(0.0, 1.0)
        mass = resized.sum(dim=-1, keepdim=True)
        normalized_mask = resized / mass.clamp_min(1e-6)
        normalized_mask = torch.where(
            mass > 0, normalized_mask, torch.zeros_like(normalized_mask)
        )
        context = torch.einsum(
            "brn,bnc->brc", normalized_mask.to(tokens.dtype), tokens
        )

        token_hidden = self.down_projection(tokens)[:, None]
        context_hidden = self.context_projection(context)[:, :, None]
        coefficients = self.evidence_projection(evidence)
        query = F.normalize(self.context_query(context).float(), dim=-1)
        key_raw = self.evidence_key(evidence).float()
        key_norm = key_raw.norm(dim=-1)
        key = F.normalize(key_raw, dim=-1)
        similarity = (query * key).sum(dim=-1)
        compatibility = torch.where(
            key_norm > 0,
            similarity,
            torch.full_like(similarity, -1.0),
        )
        gate = torch.where(
            key_norm > 0,
            torch.sigmoid(compatibility),
            torch.zeros_like(compatibility),
        )

        hidden = (token_hidden + context_hidden) * coefficients[:, :, None]
        proposal = self.up_projection(F.gelu(hidden))
        proposal = proposal * gate[:, :, None, None].to(proposal.dtype)
        slot_valid = (mass.squeeze(-1) > 0) & (presence > 0)
        proposal = torch.where(
            slot_valid[:, :, None, None],
            proposal,
            torch.zeros_like(proposal),
        )
        scatter = resized[:, :, :, None] * presence[:, :, None, None]
        unit_delta = (scatter.to(proposal.dtype) * proposal).sum(dim=1)
        if not bool(torch.isfinite(unit_delta).all()):
            raise RuntimeError("Non-finite evidence-owned router output")
        return {
            "proposal": proposal,
            "unit_delta": unit_delta,
            "coefficients": coefficients,
            "compatibility": compatibility,
            "gate": gate,
            "context": context,
            "mask": resized,
            "normalized_mask": normalized_mask,
            "mass": mass,
            "slot_valid": slot_valid,
        }

    def forward(self, tokens, hw_shape, mask, presence, evidence, rho):
        branch = self.branch(tokens, hw_shape, mask, presence, evidence)
        applied_delta = float(rho) * branch["unit_delta"]
        return tokens + applied_delta, applied_delta, branch


class SemanticProductKernel(nn.Module):
    """Parameter-free semantic binding for one fixed Euclidean descriptor."""

    def __init__(self, feature_dim=768, groups=16):
        super().__init__()
        if feature_dim <= 0 or groups <= 0 or feature_dim % groups:
            raise ValueError(
                "Semantic product feature_dim must be divisible by groups"
            )
        self.feature_dim = int(feature_dim)
        self.groups = int(groups)
        self.group_width = self.feature_dim // self.groups

    def forward(self, global_feature, evidence, presence):
        if global_feature.ndim != 2 or global_feature.shape[1] != self.feature_dim:
            raise ValueError("Unexpected semantic product global feature shape")
        if evidence.ndim != 3 or evidence.shape[2] != self.groups:
            raise ValueError("Evidence must have shape [B,R,SPK_GROUPS]")
        if presence.shape != evidence.shape[:2]:
            raise ValueError("Presence must have shape [B,R]")
        if evidence.shape[0] != global_feature.shape[0]:
            raise ValueError("Semantic product batch mismatch")

        presence_float = presence.detach().float().clamp(0.0, 1.0)
        mass = presence_float.sum(dim=1, keepdim=True)
        pooled = (evidence.float() * presence_float[..., None]).sum(dim=1)
        pooled = pooled / mass.clamp_min(1.0)
        pooled = torch.where(mass > 0, pooled, torch.zeros_like(pooled))
        factor_float = self.groups * torch.softmax(pooled, dim=-1)
        factor = factor_float.to(dtype=global_feature.dtype)
        grouped = global_feature.reshape(
            global_feature.shape[0], self.groups, self.group_width
        )
        descriptor = (factor[..., None] * grouped).reshape_as(global_feature)
        if not bool(torch.isfinite(descriptor).all()):
            raise RuntimeError("Non-finite semantic product descriptor")
        return descriptor, factor_float


class CleanRichEvidenceBudgetTapf(nn.Module):
    """Single-stage rich-evidence TAPF with two independently budgeted routers."""

    semantic = True
    rich_evidence = True

    def __init__(
        self,
        anchor_channels=384,
        anchor_hidden=128,
        consumer_channels=768,
        router_rank=16,
        gate_release=0.5,
        gaussian_sigma=1.5,
        teacher_epochs=5,
        handoff_epochs=5,
        rho_star=0.08075544983148575,
        router_class=EvidenceBudgetRouter,
    ):
        super().__init__()
        if teacher_epochs < 0 or handoff_epochs <= 0:
            raise ValueError("Invalid TAPF handoff schedule")
        if rho_star < 0 or not torch.isfinite(torch.tensor(float(rho_star))):
            raise ValueError("Residual budget must be finite and nonnegative")
        if not 0.0 < gate_release <= 1.0:
            raise ValueError("gate_release must remain a valid legacy config value")
        self.gaussian_sigma = float(gaussian_sigma)
        self.teacher_epochs = int(teacher_epochs)
        self.handoff_epochs = int(handoff_epochs)
        self.rho_star = float(rho_star)
        self.anchor = RichEvidencePoseAnchor(
            anchor_channels, anchor_hidden, evidence_dim=16
        )
        self.psg_bank = nn.ModuleList(
            [
                router_class(
                    feature_channels=consumer_channels,
                    rank=router_rank,
                    evidence_dim=16,
                )
                for _ in range(2)
            ]
        )

    def student_fraction(self, epoch):
        if epoch is None:
            raise ValueError("Training TAPF requires an epoch")
        if epoch <= self.teacher_epochs:
            return 0.0
        fraction = (epoch - self.teacher_epochs) / float(self.handoff_epochs)
        return max(0.0, min(1.0, fraction))

    def rho_at_epoch(self, epoch, training):
        if not training:
            return self.rho_star
        if epoch is None or epoch < 0:
            raise ValueError("Training evidence-budget TAPF requires an epoch")
        if epoch <= self.teacher_epochs:
            return 0.0
        if epoch >= self.teacher_epochs + self.handoff_epochs:
            return self.rho_star
        progress = (epoch - self.teacher_epochs) / float(self.handoff_epochs)
        return self.rho_star * progress

    def prepare(self, source_feature, pose_batch, image_hw, epoch, training):
        prediction = self.anchor(source_feature.detach())
        student_mask = prediction["region_mask"]
        student_presence = prediction["presence"]
        student_evidence = prediction["evidence"]
        student_joint_field = prediction["joint_field"]
        rho = self.rho_at_epoch(epoch, training)
        if not training:
            consumer_mask = student_mask.detach()
            consumer_presence = student_presence.detach()
            return {
                "consumer_mask": consumer_mask,
                "consumer_presence": consumer_presence,
                "consumer_evidence": student_evidence.detach(),
                "student_mask": student_mask,
                "student_presence": student_presence,
                "student_evidence": student_evidence,
                "student_joint_field": student_joint_field,
                "consumer_joint_field": student_joint_field.detach(),
                "consumer_field": consumer_mask
                * consumer_presence[..., None, None],
                "pose_loss": None,
                "semantic_loss": None,
                "region_mask_loss": None,
                "presence_loss": None,
                "evidence_cos_loss": None,
                "evidence_relation_loss": None,
                "exec_loss": None,
                "q_loss": None,
                "student_fraction": 1.0,
                "teacher_field": None,
                "teacher_joint_field": None,
                "reliability": consumer_presence,
                "rho": rho,
                "gate_deltas": [],
                "router_branches": [],
                "exec_losses": [],
            }
        if pose_batch is None:
            raise ValueError("Training rich-evidence TAPF requires paired targets")
        required = {
            "semantic_teacher_evidence",
            "semantic_valid",
            "semantic_teacher_mask",
        }
        missing = sorted(required.difference(pose_batch))
        if missing:
            raise ValueError(
                "Training rich-evidence TAPF is missing frozen targets: "
                + ", ".join(missing)
            )

        gaussian, teacher_joint_field, reliability = render_pose_field(
            pose_batch["keypoints"],
            pose_batch["scores"],
            pose_batch["valid"],
            image_hw=image_hw,
            field_hw=student_mask.shape[-2:],
            sigma=self.gaussian_sigma,
        )
        heatmap_weight = reliability[..., None, None]
        heatmap_denominator = (
            heatmap_weight.sum() * gaussian.shape[-2] * gaussian.shape[-1]
        ).clamp_min(1.0)
        heatmap_loss = (
            (prediction["heatmaps"] - gaussian).square() * heatmap_weight
        ).sum() / heatmap_denominator
        confidence_loss = F.binary_cross_entropy_with_logits(
            prediction["confidence_logits"].float(), reliability
        )

        semantic_valid = pose_batch["semantic_valid"].bool()
        teacher_evidence = pose_batch["semantic_teacher_evidence"].float()
        if teacher_evidence.shape != student_evidence.shape:
            raise ValueError("Frozen rich evidence shape mismatch")
        if semantic_valid.shape != student_evidence.shape[:2]:
            raise ValueError("Frozen rich evidence validity shape mismatch")
        high_resolution_teacher_mask = pose_batch["semantic_teacher_mask"].float()
        expected_high_resolution = tuple(
            dimension * 4 for dimension in student_mask.shape[-2:]
        )
        if high_resolution_teacher_mask.shape[:2] != student_mask.shape[:2]:
            raise ValueError("Frozen rich mask batch/slot shape mismatch")
        if high_resolution_teacher_mask.shape[-2:] != expected_high_resolution:
            raise ValueError(
                "Frozen rich mask must be exactly 4x the anchor resolution"
            )
        teacher_mask = F.avg_pool2d(
            high_resolution_teacher_mask, kernel_size=4, stride=4
        )
        if teacher_mask.shape != student_mask.shape:
            raise RuntimeError("Frozen rich mask downsampling contract failed")
        teacher_presence = semantic_valid.float()
        region_mask_loss = F.binary_cross_entropy_with_logits(
            prediction["region_mask_logits"].float(), teacher_mask
        )
        presence_loss = F.binary_cross_entropy_with_logits(
            prediction["presence_logits"].float(), teacher_presence
        )
        cosine = 1.0 - F.cosine_similarity(
            student_evidence.float(), teacher_evidence, dim=-1
        )
        evidence_cos_loss = (
            cosine * semantic_valid.float()
        ).sum() / semantic_valid.float().sum().clamp_min(1.0)
        evidence_relation_loss = relation_gram_loss(
            student_evidence, teacher_evidence, semantic_valid
        )

        fraction = self.student_fraction(epoch)
        consumer_mask = (
            (1.0 - fraction) * teacher_mask + fraction * student_mask
        ).detach()
        consumer_presence = (
            (1.0 - fraction) * teacher_presence
            + fraction * student_presence
        ).detach()
        consumer_joint_field = (
            (1.0 - fraction) * teacher_joint_field
            + fraction * student_joint_field
        ).detach()
        semantic_loss_without_exec = torch.stack(
            [
                region_mask_loss,
                presence_loss,
                evidence_cos_loss,
                evidence_relation_loss,
            ]
        ).mean()
        pose_loss = heatmap_loss + confidence_loss + semantic_loss_without_exec
        return {
            "consumer_mask": consumer_mask,
            "consumer_presence": consumer_presence,
            "consumer_evidence": student_evidence.detach(),
            "student_mask": student_mask,
            "student_presence": student_presence,
            "student_evidence": student_evidence,
            "student_joint_field": student_joint_field,
            "consumer_joint_field": consumer_joint_field,
            "consumer_field": consumer_mask
            * consumer_presence[..., None, None],
            "pose_loss": pose_loss,
            "heatmap_loss": heatmap_loss,
            "confidence_loss": confidence_loss,
            "semantic_loss": semantic_loss_without_exec,
            "region_mask_loss": region_mask_loss,
            "presence_loss": presence_loss,
            "evidence_cos_loss": evidence_cos_loss,
            "evidence_relation_loss": evidence_relation_loss,
            "exec_loss": None,
            "q_loss": None,
            "student_fraction": fraction,
            "teacher_field": teacher_mask * teacher_presence[..., None, None],
            "teacher_joint_field": teacher_joint_field,
            "teacher_mask": teacher_mask,
            "teacher_presence": teacher_presence,
            "teacher_evidence": teacher_evidence.detach(),
            "semantic_valid": semantic_valid,
            "reliability": consumer_presence,
            "rho": rho,
            "gate_deltas": [],
            "router_branches": [],
            "exec_losses": [],
        }

    def apply_gate(self, bank_index, tokens, hw_shape, state):
        router = self.psg_bank[bank_index]
        routed, applied_delta, branch = router(
            tokens,
            hw_shape,
            state["consumer_mask"],
            state["consumer_presence"],
            state["consumer_evidence"],
            state["rho"],
        )
        state["gate_deltas"].append(applied_delta)
        state["router_branches"].append(branch)
        if state["pose_loss"] is not None:
            exec_branch = router.branch(
                tokens.detach(),
                hw_shape,
                state["consumer_mask"].detach(),
                state["consumer_presence"].detach(),
                state["student_evidence"],
            )
            pooled_proposal = torch.einsum(
                "brn,brnc->brc",
                exec_branch["normalized_mask"],
                exec_branch["proposal"].float(),
            )
            exec_valid = state["semantic_valid"] & (
                exec_branch["mass"].squeeze(-1) > 0
            )
            exec_loss = relation_gram_loss(
                pooled_proposal, state["teacher_evidence"], exec_valid
            )
            state["exec_losses"].append(exec_loss)
            state["exec_loss"] = torch.stack(state["exec_losses"]).mean()
            state["semantic_loss"] = torch.stack(
                [
                    state["region_mask_loss"],
                    state["presence_loss"],
                    state["evidence_cos_loss"],
                    state["evidence_relation_loss"],
                    state["exec_loss"],
                ]
            ).mean()
            state["pose_loss"] = (
                state["heatmap_loss"]
                + state["confidence_loss"]
                + state["semantic_loss"]
            )
        return routed


class CleanSemanticProductTapf(CleanRichEvidenceBudgetTapf):
    """Rich RGB student evidence with the original D0 spatial consumers."""

    semantic_product = True

    def __init__(
        self,
        anchor_channels=384,
        anchor_hidden=128,
        consumer_channels=768,
        psg_hidden=32,
        gate_release=0.5,
        gaussian_sigma=1.5,
        teacher_epochs=5,
        handoff_epochs=5,
    ):
        super().__init__(
            anchor_channels=anchor_channels,
            anchor_hidden=anchor_hidden,
            consumer_channels=consumer_channels,
            router_rank=16,
            gate_release=gate_release,
            gaussian_sigma=gaussian_sigma,
            teacher_epochs=teacher_epochs,
            handoff_epochs=handoff_epochs,
            rho_star=0.0,
        )
        self.psg_bank = nn.ModuleList(
            [
                PoseSpatialGate(
                    feature_channels=consumer_channels,
                    hidden_channels=psg_hidden,
                    release=gate_release,
                )
                for _ in range(2)
            ]
        )

    def prepare(self, source_feature, pose_batch, image_hw, epoch, training):
        state = super().prepare(
            source_feature,
            pose_batch,
            image_hw=image_hw,
            epoch=epoch,
            training=training,
        )
        state["consumer_field"] = state["consumer_joint_field"]
        return state

    def apply_gate(self, bank_index, tokens, hw_shape, state):
        routed, delta = self.psg_bank[bank_index](
            tokens,
            hw_shape,
            state["consumer_field"],
        )
        state["gate_deltas"].append(delta)
        return routed


class CleanEvidenceOperatorTapf(CleanRichEvidenceBudgetTapf):
    """ELO-CUR: evidence-owned operators with no-gradient reference replays."""

    counterfactual_operator = True
    reference_arm_names = ("wrong", "generic", "null")
    compatibility_margin = 0.10
    utility_margin = 0.05

    def __init__(self, *args, **kwargs):
        kwargs["router_class"] = EvidenceOwnedLowRankRouter
        super().__init__(*args, **kwargs)

    @staticmethod
    def _masked_mean(values, valid):
        valid = valid.to(device=values.device, dtype=torch.bool)
        if valid.shape != values.shape:
            raise ValueError("Masked mean received incompatible shapes")
        weight = valid.to(values.dtype)
        return (values * weight).sum() / weight.sum().clamp_min(1.0)

    def prepare(self, source_feature, pose_batch, image_hw, epoch, training):
        state = super().prepare(
            source_feature,
            pose_batch=pose_batch,
            image_hw=image_hw,
            epoch=epoch,
            training=training,
        )
        state.update(
            {
                "compatibility_loss": None,
                "cur_loss": None,
                "cur_component_losses": None,
                "compatibility_means": None,
                "compatibility_diagnostic_gaps": None,
                "correct_utility_mean": None,
                "reference_utility_means": None,
                "coefficient_std": None,
                "coefficient_effective_rank": None,
                "reference_descriptors": {},
                "reference_router_branches": {
                    name: [] for name in self.reference_arm_names
                },
                "reference_rng_exact": None,
                "counterfactual_finalized": False,
            }
        )
        if not training:
            state["donor_eligible"] = None
            state["donor_index"] = None
            state["reference_evidence"] = {}
            return state

        required = {"identity", "camera", "generic_evidence"}
        missing = sorted(required.difference(pose_batch))
        if missing:
            raise ValueError(
                "Training ELO-CUR is missing ownership inputs: "
                + ", ".join(missing)
            )
        identities = pose_batch["identity"].long()
        cameras = pose_batch["camera"].long()
        if identities.shape != (source_feature.shape[0],):
            raise ValueError("ELO-CUR identity shape mismatch")
        if cameras.shape != identities.shape:
            raise ValueError("ELO-CUR camera shape mismatch")

        donor = build_matched_training_donor_map(identities, cameras)
        donor_eligible = donor >= 0
        safe_donor = donor.clamp_min(0)
        student_evidence = state["student_evidence"]
        wrong = student_evidence[safe_donor].detach()
        wrong = torch.where(
            donor_eligible[:, None, None], wrong, torch.zeros_like(wrong)
        )
        generic = pose_batch["generic_evidence"].float()
        if generic.ndim == 2:
            generic = generic.unsqueeze(0)
        if generic.shape not in (
            (1, student_evidence.shape[1], student_evidence.shape[2]),
            student_evidence.shape,
        ):
            raise ValueError("Frozen generic evidence shape mismatch")
        generic = generic.to(student_evidence.device).expand_as(student_evidence)
        if not bool(torch.isfinite(generic).all()):
            raise RuntimeError("Frozen generic evidence is non-finite")
        generic = generic.detach()
        null = torch.zeros_like(student_evidence).detach()

        semantic_valid = state["semantic_valid"]
        donor_valid = semantic_valid[safe_donor]
        ownership_valid = semantic_valid & donor_valid
        ownership_valid = ownership_valid & donor_eligible[:, None]
        state["consumer_evidence"] = student_evidence
        state["identity"] = identities
        state["camera"] = cameras
        state["donor_index"] = donor
        state["donor_eligible"] = donor_eligible
        state["ownership_valid"] = ownership_valid
        state["reference_evidence"] = {
            "wrong": wrong,
            "generic": generic,
            "null": null,
        }
        return state

    def apply_gate(self, bank_index, tokens, hw_shape, state):
        router = self.psg_bank[bank_index]
        routed, applied_delta, branch = router(
            tokens,
            hw_shape,
            state["consumer_mask"],
            state["consumer_presence"],
            state["consumer_evidence"],
            state["rho"],
        )
        state["gate_deltas"].append(applied_delta)
        state["router_branches"].append(branch)
        return routed

    def apply_reference_gate(
        self, bank_index, tokens, hw_shape, state, arm_name
    ):
        if arm_name not in self.reference_arm_names:
            raise ValueError("Unknown ELO-CUR reference arm")
        router = self.psg_bank[bank_index]
        routed, _, branch = router(
            tokens,
            hw_shape,
            state["consumer_mask"],
            state["consumer_presence"],
            state["reference_evidence"][arm_name],
            state["rho"],
        )
        state["reference_router_branches"][arm_name].append(branch)
        return routed

    def record_reference_descriptor(self, state, arm_name, descriptor):
        if descriptor.requires_grad:
            raise RuntimeError("ELO-CUR reference descriptor retained autograd")
        state["reference_descriptors"][arm_name] = descriptor.detach()

    def finalize_counterfactual(self, correct_descriptor, state):
        if not self.training:
            raise RuntimeError("ELO-CUR finalization is training-only")
        if len(state["router_branches"]) != len(self.psg_bank):
            raise RuntimeError("Incomplete ELO-CUR correct execution")
        if set(state["reference_descriptors"]) != set(self.reference_arm_names):
            raise RuntimeError("Incomplete ELO-CUR reference execution")
        for name in self.reference_arm_names:
            if len(state["reference_router_branches"][name]) != len(self.psg_bank):
                raise RuntimeError("Incomplete ELO-CUR reference router execution")

        compatibility_losses = []
        compatibility_values = {
            name: [] for name in ("correct", *self.reference_arm_names)
        }
        wrong_generic_gaps = []
        generic_null_gaps = []
        for bank_index, correct_branch in enumerate(state["router_branches"]):
            reference = {
                name: state["reference_router_branches"][name][bank_index]
                for name in self.reference_arm_names
            }
            valid = state["ownership_valid"] & correct_branch["slot_valid"]
            correct = correct_branch["compatibility"].float()
            detached_reference = torch.stack(
                [reference[name]["compatibility"].float().detach()
                 for name in self.reference_arm_names],
                dim=0,
            )
            maximum_reference = detached_reference.max(dim=0).values
            compatibility_losses.append(
                self._masked_mean(
                    F.relu(
                        self.compatibility_margin
                        + maximum_reference
                        - correct
                    ),
                    valid,
                )
            )
            compatibility_values["correct"].append(
                self._masked_mean(correct.detach(), valid)
            )
            for arm_index, name in enumerate(self.reference_arm_names):
                compatibility_values[name].append(
                    self._masked_mean(detached_reference[arm_index], valid)
                )
            wrong_generic_gaps.append(
                self._masked_mean(
                    detached_reference[0] - detached_reference[1], valid
                )
            )
            generic_null_gaps.append(
                self._masked_mean(
                    detached_reference[1] - detached_reference[2], valid
                )
            )
        compatibility_loss = torch.stack(compatibility_losses).mean()

        identities = state["identity"]
        count = correct_descriptor.shape[0]
        same_identity = identities[:, None].eq(identities[None, :])
        same_identity.fill_diagonal_(False)
        positive_count = same_identity.sum(dim=1)
        positive_prototype = torch.matmul(
            same_identity.to(correct_descriptor.dtype), correct_descriptor
        )
        positive_prototype = positive_prototype / positive_count.clamp_min(1)[:, None]
        positive_prototype = positive_prototype.detach()
        sample_valid = state["donor_eligible"] & (positive_count > 0)
        sample_valid = sample_valid & state["ownership_valid"].any(dim=1)
        correct_utility = F.cosine_similarity(
            correct_descriptor.float(), positive_prototype.float(), dim=-1
        )
        cur_component_losses = {}
        reference_utility = {}
        for name in self.reference_arm_names:
            value = F.cosine_similarity(
                state["reference_descriptors"][name].float(),
                positive_prototype.float(),
                dim=-1,
            ).detach()
            reference_utility[name] = value
            cur_component_losses[name] = self._masked_mean(
                F.relu(self.utility_margin + value - correct_utility),
                sample_valid,
            )
        cur_loss = torch.stack(list(cur_component_losses.values())).mean()

        state["compatibility_loss"] = compatibility_loss
        state["cur_loss"] = cur_loss
        state["cur_component_losses"] = cur_component_losses
        state["compatibility_means"] = {
            name: torch.stack(values).mean()
            for name, values in compatibility_values.items()
        }
        state["compatibility_diagnostic_gaps"] = {
            "wrong_minus_generic": torch.stack(wrong_generic_gaps).mean(),
            "generic_minus_null": torch.stack(generic_null_gaps).mean(),
        }
        state["correct_utility_mean"] = self._masked_mean(
            correct_utility.detach(), sample_valid
        )
        state["reference_utility_means"] = {
            name: self._masked_mean(value, sample_valid)
            for name, value in reference_utility.items()
        }
        coefficients = torch.cat(
            [
                branch["coefficients"].detach().float().reshape(
                    -1, branch["coefficients"].shape[-1]
                )
                for branch in state["router_branches"]
            ],
            dim=0,
        )
        state["coefficient_std"] = coefficients.std(unbiased=False)
        rank_mass = coefficients.abs().mean(dim=0)
        rank_probability = rank_mass / rank_mass.sum().clamp_min(1e-12)
        rank_entropy = -(
            rank_probability
            * rank_probability.clamp_min(1e-12).log()
        ).sum()
        state["coefficient_effective_rank"] = rank_entropy.exp()
        state["semantic_loss"] = torch.stack(
            [
                state["region_mask_loss"],
                state["presence_loss"],
                state["evidence_cos_loss"],
                state["evidence_relation_loss"],
                compatibility_loss,
                cur_loss,
            ]
        ).mean()
        state["pose_loss"] = (
            state["heatmap_loss"]
            + state["confidence_loss"]
            + state["semantic_loss"]
        )
        state["counterfactual_finalized"] = True


class CleanTapfHt0(CleanTapfD0):
    """D0-preserving late TAPF plus one independent earlier hierarchy."""

    hierarchical = True

    def __init__(
        self,
        anchor_channels=384,
        anchor_hidden=128,
        consumer_channels=768,
        psg_hidden=32,
        gaussian_sigma=1.5,
        gate_release=0.5,
        teacher_epochs=5,
        handoff_epochs=5,
        early_anchor_channels=192,
        early_consumer_channels=384,
        early_consumer_count=6,
        pose_loss_reduction="sum",
    ):
        # Construct the complete D0 path first. This preserves every common
        # parameter value and initialization draw before adding early modules.
        super().__init__(
            anchor_channels=anchor_channels,
            anchor_hidden=anchor_hidden,
            consumer_channels=consumer_channels,
            psg_hidden=psg_hidden,
            gaussian_sigma=gaussian_sigma,
            gate_release=gate_release,
            teacher_epochs=teacher_epochs,
            handoff_epochs=handoff_epochs,
        )
        if early_consumer_count <= 0:
            raise ValueError("early_consumer_count must be positive")
        if pose_loss_reduction not in ("sum", "mean"):
            raise ValueError("pose_loss_reduction must be 'sum' or 'mean'")
        self.pose_loss_reduction = pose_loss_reduction
        self.early_anchor = PoseAnchor(early_anchor_channels, anchor_hidden)
        self.early_psg_bank = nn.ModuleList(
            [
                PoseSpatialGate(
                    feature_channels=early_consumer_channels,
                    hidden_channels=psg_hidden,
                    release=gate_release,
                )
                for _ in range(early_consumer_count)
            ]
        )

    def prepare_early(self, source_feature, pose_batch, image_hw, epoch, training):
        return super()._prepare_with_anchor(
            self.early_anchor,
            source_feature,
            pose_batch=pose_batch,
            image_hw=image_hw,
            epoch=epoch,
            training=training,
        )

    def apply_early_gate(self, bank_index, tokens, hw_shape, state):
        gated, delta = self.early_psg_bank[bank_index](
            tokens, hw_shape, state["consumer_field"]
        )
        state["gate_deltas"].append(delta)
        return gated

    def combine_states(self, early_state, late_state):
        combined = late_state
        combined["early_consumer_field"] = early_state["consumer_field"]
        combined["late_consumer_field"] = late_state["consumer_field"]
        combined["early_student_field"] = early_state["student_field"]
        combined["late_student_field"] = late_state["student_field"]
        combined["early_teacher_field"] = early_state["teacher_field"]
        combined["late_teacher_field"] = late_state["teacher_field"]
        combined["early_reliability"] = early_state["reliability"]
        combined["late_reliability"] = late_state["reliability"]
        combined["early_student_fraction"] = early_state["student_fraction"]
        combined["late_student_fraction"] = late_state["student_fraction"]
        combined["early_pose_loss"] = early_state["pose_loss"]
        combined["late_pose_loss"] = late_state["pose_loss"]
        if early_state["pose_loss"] is None:
            combined["pose_loss"] = None
        else:
            combined["pose_loss"] = early_state["pose_loss"] + late_state["pose_loss"]
            if self.pose_loss_reduction == "mean":
                combined["pose_loss"] = combined["pose_loss"] * 0.5
        combined["early_gate_deltas"] = early_state["gate_deltas"]
        combined["late_gate_deltas"] = []
        combined["gate_deltas"] = list(early_state["gate_deltas"])
        return combined

    def apply_gate(self, bank_index, tokens, hw_shape, state):
        gated, delta = self.psg_bank[bank_index](
            tokens, hw_shape, state["consumer_field"]
        )
        state["late_gate_deltas"].append(delta)
        state["gate_deltas"].append(delta)
        return gated
