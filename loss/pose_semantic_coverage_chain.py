"""Pose-semantic complementary coverage-chain ranking for exp413 PSCCR."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


CONTROL_MODES = ("correct", "pose_only", "q_only", "text_shuffle")


def semantic_visibility_margin(clip_features, text_prototypes):
    """Return identity-free visible-minus-occluded CLIP margins."""
    if clip_features.ndim != 3 or clip_features.shape[1:] != (5, 768):
        raise ValueError("PSCCR CLIP features must have shape [B,5,768]")
    if text_prototypes.shape != (5, 2, 768):
        raise ValueError("PSCCR text prototypes must have shape [5,2,768]")
    feature = F.normalize(clip_features.float(), dim=-1)
    prototype = text_prototypes.float()
    if not bool(torch.isfinite(feature).all()):
        raise RuntimeError("PSCCR CLIP feature is non-finite")
    if not bool(torch.isfinite(prototype).all()):
        raise RuntimeError("PSCCR text prototype is non-finite")
    return torch.einsum(
        "brd,rd->br", feature, prototype[:, 0] - prototype[:, 1]
    )


def strict_support_reliability(
    visibility,
    semantic_margin,
    clip_valid,
    *,
    mode,
):
    """Compute strict ordinal reliability inside one LOO support only."""
    if mode not in {"correct", "pose_only", "q_only"}:
        raise ValueError("unsupported PSCCR reliability mode")
    if visibility.shape != (3, 5):
        raise ValueError("PSCCR support visibility must have shape [3,5]")
    if semantic_margin.shape != visibility.shape:
        raise ValueError("PSCCR support semantic margin shape mismatch")
    if clip_valid.shape != visibility.shape:
        raise ValueError("PSCCR support validity shape mismatch")
    visibility = visibility.float()
    semantic_margin = semantic_margin.float()
    clip_valid = clip_valid.bool()
    if not bool(torch.isfinite(visibility).all()):
        raise RuntimeError("PSCCR visibility is non-finite")
    if not bool(torch.isfinite(semantic_margin).all()):
        raise RuntimeError("PSCCR semantic margin is non-finite")

    rank_v = (
        visibility[:, None, :] > visibility[None, :, :]
    ).sum(dim=1).to(torch.long)
    semantic_greater = (
        semantic_margin[:, None, :] > semantic_margin[None, :, :]
    )
    rank_q = (
        semantic_greater & clip_valid[None, :, :]
    ).sum(dim=1).to(torch.long)
    rank_q = torch.where(clip_valid, rank_q, torch.zeros_like(rank_q))
    if mode == "pose_only":
        reliability = rank_v
    elif mode == "q_only":
        reliability = rank_q
    else:
        reliability = torch.minimum(rank_v, rank_q)
    if not bool(((reliability >= 0) & (reliability <= 2)).all()):
        raise RuntimeError("PSCCR strict rank escaped [0,2]")
    return reliability, rank_v, rank_q


def greedy_coverage_permutation(support_indices, reliability):
    """Order three supports by five-slot marginal max-coverage gain."""
    if support_indices.shape != (3,):
        raise ValueError("PSCCR support indices must have shape [3]")
    if reliability.shape != (3, 5):
        raise ValueError("PSCCR reliability must have shape [3,5]")
    if support_indices.unique().numel() != 3:
        raise RuntimeError("PSCCR support indices are not unique")
    reliability = reliability.to(torch.long)
    selected = []
    remaining = [0, 1, 2]
    covered = torch.zeros(5, dtype=torch.long, device=reliability.device)
    coverage_history = []
    while remaining:
        gains = []
        for local_index in remaining:
            candidate = torch.maximum(covered, reliability[local_index])
            gains.append((candidate - covered).sum())
        gains_tensor = torch.stack(gains)
        max_gain = gains_tensor.max()
        tied = [
            remaining[position]
            for position in range(len(remaining))
            if bool(gains_tensor[position] == max_gain)
        ]
        chosen = min(
            tied,
            key=lambda index: int(support_indices[index].item()),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        covered = torch.maximum(covered, reliability[chosen])
        coverage_history.append(covered.sum())
    chain = support_indices[
        torch.as_tensor(selected, dtype=torch.long, device=support_indices.device)
    ]
    coverage = torch.stack(coverage_history)
    if chain.unique().numel() != 3:
        raise RuntimeError("PSCCR coverage chain is not a permutation")
    if not bool((coverage[1:] >= coverage[:-1]).all()):
        raise RuntimeError("PSCCR coverage is not monotonic")
    return chain, coverage


def build_coverage_chain_from_signals(
    labels,
    visibility,
    semantic_margin,
    clip_valid,
    base_state,
    *,
    mode,
    reported_mode=None,
):
    """Build one support-only chain from already computed scalar signals."""
    if mode not in {"correct", "pose_only", "q_only"}:
        raise ValueError("unsupported PSCCR signal mode")
    if labels.ndim != 1 or visibility.shape != (labels.numel(), 5):
        raise ValueError("PSCCR labels/visibility shape mismatch")
    if semantic_margin.shape != visibility.shape:
        raise ValueError("PSCCR semantic margin shape mismatch")
    if clip_valid.shape != visibility.shape:
        raise ValueError("PSCCR validity shape mismatch")
    if bool(base_state.get("use_owner_multiplicity", True)):
        raise RuntimeError("PSCCR requires the sealed zero-owner host")
    support = base_state["support_indices"].to(labels.device)
    class_labels = base_state["class_labels"].to(labels.device)
    positive_class = base_state["positive_class_indices"].to(labels.device)
    if support.ndim != 3 or support.shape[0] != labels.numel():
        raise ValueError("PSCCR base support shape mismatch")
    if support.shape[-1] != 3:
        raise RuntimeError("PSCCR requires exactly three LOO supports")
    if not bool(
        labels[support].eq(class_labels[None, :, None]).all()
    ):
        raise RuntimeError("PSCCR base support crossed identity")

    batch, classes, _ = support.shape
    chain = torch.empty_like(support)
    coverage = torch.empty(
        batch, classes, 3, dtype=torch.long, device=labels.device
    )
    reliability = torch.empty(
        batch, classes, 3, 5, dtype=torch.long, device=labels.device
    )
    rank_v = torch.empty_like(reliability)
    rank_q = torch.empty_like(reliability)
    for anchor in range(batch):
        for class_index in range(classes):
            support_index = support[anchor, class_index]
            local_reliability, local_rank_v, local_rank_q = (
                strict_support_reliability(
                    visibility[support_index],
                    semantic_margin[support_index],
                    clip_valid[support_index],
                    mode=mode,
                )
            )
            local_chain, local_coverage = greedy_coverage_permutation(
                support_index, local_reliability
            )
            reliability[anchor, class_index] = local_reliability
            rank_v[anchor, class_index] = local_rank_v
            rank_q[anchor, class_index] = local_rank_q
            chain[anchor, class_index] = local_chain
            coverage[anchor, class_index] = local_coverage

    if not bool(
        labels[chain].eq(class_labels[None, :, None]).all()
    ):
        raise RuntimeError("PSCCR chain crossed identity")
    anchor_index = torch.arange(batch, device=labels.device)
    positive_chain = chain[anchor_index, positive_class]
    if bool(positive_chain.eq(anchor_index[:, None]).any()):
        raise RuntimeError("PSCCR positive chain contains anchor self")
    if not bool(
        torch.sort(chain, dim=-1).values.eq(
            torch.sort(support, dim=-1).values
        ).all()
    ):
        raise RuntimeError("PSCCR chain changed the three-support set")
    return {
        "support_indices": support,
        "chain_indices": chain,
        "class_labels": class_labels,
        "positive_class_indices": positive_class,
        "coverage": coverage,
        "reliability": reliability,
        "rank_v": rank_v,
        "rank_q": rank_q,
        "mode": reported_mode or mode,
        "use_owner_multiplicity": False,
    }


def build_pose_semantic_coverage_chain(
    labels,
    visibility,
    clip_features,
    clip_valid,
    text_prototypes,
    base_state,
    *,
    mode="correct",
):
    """Build a deterministic PSCCR chain without reading excluded evidence."""
    if mode not in CONTROL_MODES:
        raise ValueError("unsupported PSCCR control mode")
    prototype = (
        torch.roll(text_prototypes, shifts=-1, dims=0)
        if mode == "text_shuffle"
        else text_prototypes
    )
    semantic = semantic_visibility_margin(clip_features, prototype)
    signal_mode = "correct" if mode == "text_shuffle" else mode
    return build_coverage_chain_from_signals(
        labels,
        visibility,
        semantic,
        clip_valid,
        base_state,
        mode=signal_mode,
        reported_mode=mode,
    )


def _listwise_loss_from_set_distance(
    set_distance,
    labels,
    class_labels,
    positive_class,
):
    batch, classes = set_distance.shape
    anchor = torch.arange(batch, device=labels.device)
    positive_distance = set_distance[anchor, positive_class]
    negative_mask = class_labels[None, :].ne(labels[:, None])
    if not bool((negative_mask.sum(dim=1) == classes - 1).all()):
        raise RuntimeError("PSCCR negative identity cardinality drift")
    negative_distance = set_distance[negative_mask].view(batch, classes - 1)
    delta = positive_distance[:, None] - negative_distance
    log_mean_exp = torch.logsumexp(delta, dim=1) - math.log(classes - 1)
    loss = F.softplus(log_mean_exp).mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("PSCCR prefix loss is non-finite")
    return loss, positive_distance, negative_distance


def pose_semantic_coverage_chain_ranking_loss(
    global_feat,
    labels,
    chain_state,
    *,
    normalize_feature=False,
):
    """Mean the prefix-1/2 objectives with an exact zero-owner prefix-3."""
    from .pose_clip_multi_positive_set import (
        pose_clip_identity_set_ranking_loss,
    )
    from .triplet_loss import euclidean_dist, normalize

    if global_feat.ndim != 2 or global_feat.shape[0] != labels.numel():
        raise ValueError("PSCCR feature/label shape mismatch")
    feature = global_feat.float()
    if normalize_feature:
        feature = normalize(feature, axis=-1)
    distance = euclidean_dist(feature, feature)
    chain = chain_state["chain_indices"].to(labels.device)
    class_labels = chain_state["class_labels"].to(labels.device)
    positive_class = chain_state["positive_class_indices"].to(labels.device)
    batch, classes, support_count = chain.shape
    if support_count != 3:
        raise RuntimeError("PSCCR chain cardinality drift")
    chain_distance = distance.gather(1, chain.reshape(batch, -1)).view(
        batch, classes, support_count
    )
    prefix_set_distance = (
        chain_distance[..., 0],
        chain_distance[..., :2].mean(dim=-1),
    )
    prefix_losses = []
    positive_distances = []
    negative_distances = []
    for set_distance in prefix_set_distance:
        loss, positive, negative = _listwise_loss_from_set_distance(
            set_distance,
            labels,
            class_labels,
            positive_class,
        )
        prefix_losses.append(loss)
        positive_distances.append(positive)
        negative_distances.append(negative)

    zero_owner_state = {
        "support_indices": chain_state["support_indices"],
        "owner_indices": chain_state["support_indices"][:, :, :1].expand(
            -1, -1, 5
        ),
        "class_labels": class_labels,
        "positive_class_indices": positive_class,
        "use_owner_multiplicity": False,
    }
    prefix3_loss, prefix3_diag = pose_clip_identity_set_ranking_loss(
        global_feat,
        labels,
        zero_owner_state,
        normalize_feature=normalize_feature,
    )
    prefix_losses.append(prefix3_loss)
    positive_distances.append(prefix3_diag["positive_distance"])
    negative_distances.append(prefix3_diag["negative_distance"])
    loss = torch.stack(prefix_losses).mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("PSCCR chain loss is non-finite")
    return loss, {
        "loss": loss.detach(),
        "prefix_losses": torch.stack(
            [value.detach() for value in prefix_losses]
        ),
        "positive_distance_mean": torch.stack(
            [value.detach().mean() for value in positive_distances]
        ),
        "negative_distance_mean": torch.stack(
            [value.detach().mean() for value in negative_distances]
        ),
        "prefix3_set_distance": prefix3_diag["set_distance"],
    }
