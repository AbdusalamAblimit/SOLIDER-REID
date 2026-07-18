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
        self.token_projection = nn.Linear(feature_channels, rank, bias=False)
        self.context_projection = nn.Linear(feature_channels, rank, bias=False)
        self.expert = nn.Parameter(
            torch.zeros(region_count, rank, feature_channels)
        )

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
