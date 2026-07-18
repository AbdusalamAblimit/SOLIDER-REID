"""Minimal training-privileged, inference-RGB-only TAPF components."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            combined["pose_loss"] = (
                early_state["pose_loss"] + late_state["pose_loss"]
            )
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
