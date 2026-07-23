"""Pose-semantic continuous identity-region ranking for exp414 PSCIR."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


CONTROL_MODES = (
    "correct",
    "pose_only",
    "q_only",
    "text_shuffle",
    "all_edges",
)
_EDGE_LOCAL = ((0, 1), (0, 2), (1, 2))


def semantic_visibility_margin(clip_features, text_prototypes):
    """Return identity-free visible-minus-occluded CLIP margins."""
    if clip_features.ndim != 3 or clip_features.shape[1:] != (5, 768):
        raise ValueError("PSCIR CLIP features must have shape [B,5,768]")
    if text_prototypes.shape != (5, 2, 768):
        raise ValueError("PSCIR text prototypes must have shape [5,2,768]")
    feature = F.normalize(clip_features.float(), dim=-1)
    prototype = text_prototypes.float()
    if not bool(torch.isfinite(feature).all()):
        raise RuntimeError("PSCIR CLIP feature is non-finite")
    if not bool(torch.isfinite(prototype).all()):
        raise RuntimeError("PSCIR text prototype is non-finite")
    return torch.einsum(
        "brd,rd->br", feature, prototype[:, 0] - prototype[:, 1]
    )


def strict_support_ranks(visibility, semantic_margin, clip_valid):
    """Compute pose and semantic strict ranks inside LOO supports only."""
    if visibility.ndim != 4 or visibility.shape[-2:] != (3, 5):
        raise ValueError("PSCIR support visibility must have shape [B,C,3,5]")
    if semantic_margin.shape != visibility.shape:
        raise ValueError("PSCIR support semantic margin shape mismatch")
    if clip_valid.shape != visibility.shape:
        raise ValueError("PSCIR support validity shape mismatch")
    visibility = visibility.float()
    semantic_margin = semantic_margin.float()
    clip_valid = clip_valid.bool()
    if not bool(torch.isfinite(visibility).all()):
        raise RuntimeError("PSCIR visibility is non-finite")
    if not bool(torch.isfinite(semantic_margin).all()):
        raise RuntimeError("PSCIR semantic margin is non-finite")

    rank_v = (
        visibility.unsqueeze(3) > visibility.unsqueeze(2)
    ).sum(dim=3).to(torch.long)
    semantic_greater = (
        semantic_margin.unsqueeze(3) > semantic_margin.unsqueeze(2)
    )
    valid_peer = clip_valid.unsqueeze(2)
    rank_q = (semantic_greater & valid_peer).sum(dim=3).to(torch.long)
    rank_q = torch.where(clip_valid, rank_q, torch.zeros_like(rank_q))
    if not bool(
        ((rank_v >= 0) & (rank_v <= 2)).all()
        and ((rank_q >= 0) & (rank_q <= 2)).all()
    ):
        raise RuntimeError("PSCIR strict rank escaped [0,2]")
    return rank_v, rank_q


def build_continuous_region_from_signals(
    labels,
    visibility,
    semantic_margin,
    clip_valid,
    base_state,
    *,
    mode,
    reported_mode=None,
):
    """Build deterministic support-edge topology from scalar signals."""
    if mode not in {"correct", "pose_only", "q_only", "all_edges"}:
        raise ValueError("unsupported PSCIR signal mode")
    if labels.ndim != 1 or visibility.shape != (labels.numel(), 5):
        raise ValueError("PSCIR labels/visibility shape mismatch")
    if semantic_margin.shape != visibility.shape:
        raise ValueError("PSCIR semantic margin shape mismatch")
    if clip_valid.shape != visibility.shape:
        raise ValueError("PSCIR validity shape mismatch")
    if bool(base_state.get("use_owner_multiplicity", True)):
        raise RuntimeError("PSCIR requires the sealed zero-owner host")

    support = base_state["support_indices"].to(labels.device)
    class_labels = base_state["class_labels"].to(labels.device)
    positive_class = base_state["positive_class_indices"].to(labels.device)
    if support.ndim != 3 or support.shape[0] != labels.numel():
        raise ValueError("PSCIR base support shape mismatch")
    if support.shape[-1] != 3:
        raise RuntimeError("PSCIR requires exactly three LOO supports")
    if not bool((support[..., 1:] > support[..., :-1]).all()):
        raise RuntimeError("PSCIR requires ascending unique supports")
    if not bool(labels[support].eq(class_labels[None, :, None]).all()):
        raise RuntimeError("PSCIR base support crossed identity")

    support_visibility = visibility[support]
    support_semantic = semantic_margin[support]
    support_valid = clip_valid[support]
    rank_v, rank_q = strict_support_ranks(
        support_visibility, support_semantic, support_valid
    )
    edge_local = torch.as_tensor(
        _EDGE_LOCAL, dtype=torch.long, device=labels.device
    )
    edge_u = edge_local[:, 0]
    edge_v = edge_local[:, 1]
    delta_v = (
        rank_v[:, :, edge_u, :] - rank_v[:, :, edge_v, :]
    ).abs()
    delta_q = (
        rank_q[:, :, edge_u, :] - rank_q[:, :, edge_v, :]
    ).abs()
    if mode == "pose_only":
        edge_weight = delta_v.sum(dim=-1)
    elif mode == "q_only":
        edge_weight = delta_q.sum(dim=-1)
    else:
        edge_weight = (delta_v * delta_q).sum(dim=-1)

    batch, classes, _ = support.shape
    if mode == "all_edges":
        selected_local = edge_local.view(1, 1, 3, 2).expand(
            batch, classes, -1, -1
        )
        selected_edge_id = torch.arange(
            3, device=labels.device, dtype=torch.long
        ).view(1, 1, 3).expand(batch, classes, -1)
    else:
        # Supports are strictly ascending, so the fixed local edge order is
        # also the absolute-batch-index lexicographic order.  Multiplying the
        # integer weight by four and adding 3/2/1 makes every tie unique.
        tie_priority = torch.tensor(
            [3, 2, 1], device=labels.device, dtype=torch.long
        )
        priority = edge_weight.to(torch.long) * 4 + tie_priority
        selected_edge_id = torch.topk(
            priority, k=2, dim=-1, largest=True, sorted=True
        ).indices
        selected_local = edge_local[selected_edge_id]

    edge_indices = torch.stack(
        (
            support.gather(2, selected_local[..., 0]),
            support.gather(2, selected_local[..., 1]),
        ),
        dim=-1,
    )
    if not bool(
        labels[edge_indices].eq(
            class_labels[None, :, None, None]
        ).all()
    ):
        raise RuntimeError("PSCIR edge crossed identity")
    if not bool(
        (edge_indices[..., 0] < edge_indices[..., 1]).all()
    ):
        raise RuntimeError("PSCIR edge is not a canonical undirected pair")
    if mode != "all_edges":
        covered = (
            selected_local[..., None]
            == torch.arange(3, device=labels.device).view(1, 1, 1, 1, 3)
        ).any(dim=(2, 3))
        if not bool(covered.all()):
            raise RuntimeError("PSCIR MST failed to cover three supports")
    anchor = torch.arange(batch, device=labels.device)
    positive_edges = edge_indices[anchor, positive_class]
    if bool(positive_edges.eq(anchor[:, None, None]).any()):
        raise RuntimeError("PSCIR positive region contains anchor self")
    return {
        "support_indices": support,
        "edge_indices": edge_indices,
        "selected_edge_ids": selected_edge_id,
        "edge_weight": edge_weight,
        "class_labels": class_labels,
        "positive_class_indices": positive_class,
        "rank_v": rank_v,
        "rank_q": rank_q,
        "mode": reported_mode or mode,
        "use_owner_multiplicity": False,
    }


def build_pose_semantic_continuous_region(
    labels,
    visibility,
    clip_features,
    clip_valid,
    text_prototypes,
    base_state,
    *,
    mode="correct",
):
    """Build a PSCIR topology without reading excluded-image evidence."""
    if mode not in CONTROL_MODES:
        raise ValueError("unsupported PSCIR control mode")
    prototype = (
        torch.roll(text_prototypes, shifts=-1, dims=0)
        if mode == "text_shuffle"
        else text_prototypes
    )
    semantic = semantic_visibility_margin(clip_features, prototype)
    signal_mode = "correct" if mode == "text_shuffle" else mode
    return build_continuous_region_from_signals(
        labels,
        visibility,
        semantic,
        clip_valid,
        base_state,
        mode=signal_mode,
        reported_mode=mode,
    )


def continuous_region_distance(
    global_feat,
    edge_indices,
    *,
    normalize_feature=False,
):
    """Return anchor-to-polyline distances for every batch identity."""
    from .triplet_loss import normalize

    if global_feat.ndim != 2:
        raise ValueError("PSCIR global feature must have shape [B,D]")
    if (
        edge_indices.ndim != 4
        or edge_indices.shape[0] != global_feat.shape[0]
        or edge_indices.shape[-1] != 2
        or edge_indices.shape[-2] not in {2, 3}
    ):
        raise ValueError("PSCIR edge index shape mismatch")
    feature = global_feat.float()
    if normalize_feature:
        feature = normalize(feature, axis=-1)
    edge = edge_indices.to(feature.device)
    start = feature[edge[..., 0]]
    end = feature[edge[..., 1]]
    direction = end - start
    anchor = feature[:, None, None, :]
    denominator = direction.square().sum(dim=-1).clamp_min(1e-12)
    coefficient = (
        ((anchor - start) * direction).sum(dim=-1) / denominator
    ).clamp(0.0, 1.0)
    projection = start + coefficient[..., None] * direction
    segment_distance = torch.linalg.vector_norm(
        anchor - projection, dim=-1
    )
    edge_count = edge.shape[-2]
    region_distance = -(
        torch.logsumexp(-segment_distance, dim=-1)
        - math.log(edge_count)
    )
    if not bool(
        torch.isfinite(segment_distance).all()
        and torch.isfinite(region_distance).all()
    ):
        raise RuntimeError("PSCIR region distance is non-finite")
    return region_distance, segment_distance


def _listwise_region_loss(
    region_distance,
    labels,
    class_labels,
    positive_class,
):
    batch, classes = region_distance.shape
    anchor = torch.arange(batch, device=labels.device)
    positive_distance = region_distance[anchor, positive_class]
    negative_mask = class_labels[None, :].ne(labels[:, None])
    if not bool((negative_mask.sum(dim=1) == classes - 1).all()):
        raise RuntimeError("PSCIR negative identity cardinality drift")
    negative_distance = region_distance[negative_mask].view(
        batch, classes - 1
    )
    delta = positive_distance[:, None] - negative_distance
    log_mean_exp = torch.logsumexp(delta, dim=1) - math.log(classes - 1)
    loss = F.softplus(log_mean_exp).mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("PSCIR listwise region loss is non-finite")
    return loss, positive_distance, negative_distance


def pose_semantic_continuous_region_ranking_loss(
    global_feat,
    labels,
    region_state,
    *,
    normalize_feature=False,
):
    """Equal-weight sealed zero-owner and continuous-region objectives."""
    from .pose_clip_multi_positive_set import (
        pose_clip_identity_set_ranking_loss,
    )

    region_loss, region_diag = continuous_region_ranking_loss(
        global_feat,
        labels,
        region_state,
        normalize_feature=normalize_feature,
    )
    class_labels = region_state["class_labels"].to(labels.device)
    positive_class = region_state["positive_class_indices"].to(labels.device)
    zero_owner_state = {
        "support_indices": region_state["support_indices"],
        "owner_indices": region_state["support_indices"][:, :, :1].expand(
            -1, -1, 5
        ),
        "class_labels": class_labels,
        "positive_class_indices": positive_class,
        "use_owner_multiplicity": False,
    }
    zero_owner_loss, zero_diag = pose_clip_identity_set_ranking_loss(
        global_feat,
        labels,
        zero_owner_state,
        normalize_feature=normalize_feature,
    )
    loss = 0.5 * zero_owner_loss + 0.5 * region_loss
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("PSCIR combined metric loss is non-finite")
    return loss, {
        "loss": loss.detach(),
        "zero_owner_loss": zero_owner_loss.detach(),
        "region_loss": region_loss.detach(),
        "positive_distance": region_diag["positive_distance"],
        "negative_distance": region_diag["negative_distance"],
        "region_distance": region_diag["region_distance"],
        "segment_distance": region_diag["segment_distance"],
        "zero_owner_set_distance": zero_diag["set_distance"],
    }


def continuous_region_ranking_loss(
    global_feat,
    labels,
    region_state,
    *,
    normalize_feature=False,
):
    """Return only the differentiable PSCIR continuous-region objective."""
    if global_feat.ndim != 2 or global_feat.shape[0] != labels.numel():
        raise ValueError("PSCIR feature/label shape mismatch")
    edge_indices = region_state["edge_indices"].to(labels.device)
    class_labels = region_state["class_labels"].to(labels.device)
    positive_class = region_state["positive_class_indices"].to(labels.device)
    region_distance, segment_distance = continuous_region_distance(
        global_feat,
        edge_indices,
        normalize_feature=normalize_feature,
    )
    region_loss, positive, negative = _listwise_region_loss(
        region_distance, labels, class_labels, positive_class
    )
    return region_loss, {
        "loss": region_loss.detach(),
        "positive_distance": positive.detach(),
        "negative_distance": negative.detach(),
        "region_distance": region_distance.detach(),
        "segment_distance": segment_distance.detach(),
    }
