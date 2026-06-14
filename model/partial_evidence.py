# encoding: utf-8
"""PARTIAL_EVIDENCE：把合成遮挡图当作部分身份观测来训练。

本模块只提供训练期的合成图监督校准。默认配置不开时，processor 不会导入本文件，
模型结构、前向路径和评测路径都保持基线。
"""

import torch
import torch.nn.functional as F

from loss.triplet_loss import euclidean_dist, normalize


def _first_tensor(x):
    if isinstance(x, (list, tuple)):
        return x[0]
    return x


def partial_evidence_ce_loss(logits, targets, evidence, min_keep=0.2, ls_max=0.2,
                             return_details=False):
    """逐样本按证据分设置 label smoothing 和 CE 权重。

    smoothing_i = LS_MAX * (1 - e_i)
    weight_i = MIN_KEEP + (1 - MIN_KEEP) * e_i
    """
    logits = _first_tensor(logits)
    if logits.numel() == 0:
        loss = logits.new_zeros(())
        if return_details:
            empty = logits.new_zeros((0,))
            return loss, {"weight": empty, "smoothing": empty, "per_sample": empty}
        return loss

    targets = targets.to(device=logits.device, dtype=torch.long)
    evidence = evidence.to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
    min_keep = float(min_keep)
    ls_max = float(ls_max)
    if not 0.0 <= min_keep <= 1.0:
        raise ValueError("PARTIAL_EVIDENCE.MIN_KEEP must be in [0, 1]")
    if not 0.0 <= ls_max < 1.0:
        raise ValueError("PARTIAL_EVIDENCE.LS_MAX must be in [0, 1)")

    smoothing = ls_max * (1.0 - evidence)
    weight = min_keep + (1.0 - min_keep) * evidence
    log_probs = F.log_softmax(logits, dim=1)
    soft_targets = torch.zeros_like(log_probs)
    soft_targets.scatter_(1, targets.unsqueeze(1), 1.0)
    soft_targets = (1.0 - smoothing.unsqueeze(1)) * soft_targets \
        + smoothing.unsqueeze(1) / float(logits.shape[1])
    per_sample = -(soft_targets * log_probs).sum(dim=1)
    loss = (per_sample * weight).mean()

    if return_details:
        return loss, {
            "weight": weight.detach(),
            "smoothing": smoothing.detach(),
            "per_sample": per_sample.detach(),
        }
    return loss


def partial_evidence_triplet_loss(synth_feat, synth_labels, synth_evidence,
                                  no_hardneg_below=0.4, base_margin=0.3,
                                  margin_scale=True, no_margin=False,
                                  normalize_feature=False, return_details=False):
    """只在合成图批内做部分证据 triplet。

    低证据合成样本不作为 anchor，也不作为 hard negative；它们仍保留在正样本集合里，
    因此同身份的其他 anchor 可以把它们作为部分观测正样本。NO_MARGIN=True 时沿用
    基线的 soft-margin triplet 形式；否则使用显式 margin，并可按 anchor 的证据分缩放。
    """
    synth_feat = _first_tensor(synth_feat)
    if synth_feat.numel() == 0:
        loss = synth_feat.new_zeros(())
        if return_details:
            return loss, _empty_triplet_details(synth_feat.device)
        return loss

    synth_labels = synth_labels.to(device=synth_feat.device, dtype=torch.long)
    synth_evidence = synth_evidence.to(device=synth_feat.device, dtype=synth_feat.dtype).clamp(0.0, 1.0)
    features = synth_feat
    if normalize_feature:
        features = normalize(features, axis=-1)
    labels = synth_labels
    evidence = synth_evidence

    low_evidence = evidence.lt(float(no_hardneg_below))
    anchor_eligible = ~low_evidence
    negative_eligible = ~low_evidence
    dist_mat = euclidean_dist(features, features)

    losses = []
    anchor_indices = []
    positive_indices = []
    negative_indices = []
    margins = []
    all_indices = torch.arange(features.shape[0], device=features.device)
    for anchor in all_indices[anchor_eligible].tolist():
        same_id = labels.eq(labels[anchor])
        pos_mask = same_id & all_indices.ne(anchor)
        neg_mask = labels.ne(labels[anchor]) & negative_eligible
        if not bool(pos_mask.any()) or not bool(neg_mask.any()):
            continue
        pos_candidates = all_indices[pos_mask]
        neg_candidates = all_indices[neg_mask]
        pos_rel = torch.argmax(dist_mat[anchor, pos_candidates])
        neg_rel = torch.argmin(dist_mat[anchor, neg_candidates])
        pos_idx = pos_candidates[pos_rel]
        neg_idx = neg_candidates[neg_rel]
        if no_margin:
            margin_t = dist_mat.new_zeros(())
            losses.append(F.soft_margin_loss(
                (dist_mat[anchor, neg_idx] - dist_mat[anchor, pos_idx]).view(1),
                dist_mat.new_ones(1),
                reduction='none',
            ).view(()))
        else:
            margin = float(base_margin)
            if margin_scale:
                margin = margin * float(evidence[anchor].detach().item())
            margin_t = dist_mat.new_tensor(margin)
            losses.append(F.relu(dist_mat[anchor, pos_idx] - dist_mat[anchor, neg_idx] + margin_t))
        anchor_indices.append(anchor)
        positive_indices.append(int(pos_idx.detach().item()))
        negative_indices.append(int(neg_idx.detach().item()))
        margins.append(margin_t)

    if losses:
        loss = torch.stack(losses).mean()
        anchor_tensor = torch.tensor(anchor_indices, dtype=torch.long, device=features.device)
        positive_tensor = torch.tensor(positive_indices, dtype=torch.long, device=features.device)
        negative_tensor = torch.tensor(negative_indices, dtype=torch.long, device=features.device)
        margin_tensor = torch.stack(margins).detach()
    else:
        loss = features.new_zeros(())
        anchor_tensor = torch.empty(0, dtype=torch.long, device=features.device)
        positive_tensor = torch.empty(0, dtype=torch.long, device=features.device)
        negative_tensor = torch.empty(0, dtype=torch.long, device=features.device)
        margin_tensor = features.new_zeros((0,))

    if return_details:
        return loss, {
            "anchor_indices": anchor_tensor,
            "positive_indices": positive_tensor,
            "negative_indices": negative_tensor,
            "margins": margin_tensor,
            "low_synth_indices": all_indices[low_evidence].detach(),
            "negative_eligible": negative_eligible.detach(),
            "evidence": evidence.detach(),
            "num_clean": torch.tensor(0, dtype=torch.long, device=features.device),
            "no_margin": bool(no_margin),
        }
    return loss


def _empty_triplet_details(device):
    return {
        "anchor_indices": torch.empty(0, dtype=torch.long, device=device),
        "positive_indices": torch.empty(0, dtype=torch.long, device=device),
        "negative_indices": torch.empty(0, dtype=torch.long, device=device),
        "margins": torch.empty(0, device=device),
        "low_synth_indices": torch.empty(0, dtype=torch.long, device=device),
        "negative_eligible": torch.empty(0, dtype=torch.bool, device=device),
        "evidence": torch.empty(0, device=device),
        "num_clean": torch.tensor(0, dtype=torch.long, device=device),
        "no_margin": False,
    }


def partial_evidence_training_loss(synth_score, synth_feat, targets, target_cam, occ_id, evidence, cfg,
                                   loss_fn=None, return_details=False):
    """计算合成图附加训练损失。干净图基线损失由 processor 单独照常计算。"""
    raw_synth_score = synth_score
    raw_synth_feat = synth_feat
    synth_score = _first_tensor(synth_score)
    synth_feat = _first_tensor(synth_feat)
    targets = targets.to(device=synth_score.device, dtype=torch.long)
    target_cam = target_cam.to(device=synth_score.device)
    occ_id = occ_id.to(device=synth_score.device)
    evidence = evidence.to(device=synth_score.device, dtype=synth_score.dtype).clamp(0.0, 1.0)

    if not bool(getattr(cfg.PARTIAL_EVIDENCE, 'CALIBRATE', True)):
        if loss_fn is None:
            raise ValueError("PARTIAL_EVIDENCE.CALIBRATE=False 需要传入原始 loss_fn。")
        loss = loss_fn(raw_synth_score, raw_synth_feat, targets, target_cam)
        if return_details:
            return loss, {
                "calibrate": False,
                "raw_loss": loss.detach(),
                "ce_loss": synth_score.new_zeros(()),
                "triplet_loss": synth_score.new_zeros(()),
                "weight": synth_score.new_zeros((0,)),
                "smoothing": synth_score.new_zeros((0,)),
                "per_sample_ce": synth_score.new_zeros((0,)),
                "triplet": _empty_triplet_details(synth_score.device),
                "occ_id": occ_id.detach(),
                "evidence": evidence.detach(),
            }
        return loss

    if synth_score.shape[0] == 0:
        loss = synth_score.new_zeros(())
        if return_details:
            return loss, {
                "calibrate": True,
                "ce_loss": loss.detach(),
                "triplet_loss": loss.detach(),
                "weight": synth_score.new_zeros((0,)),
                "smoothing": synth_score.new_zeros((0,)),
                "triplet": _empty_triplet_details(synth_score.device),
            }
        return loss

    ce_loss, ce_details = partial_evidence_ce_loss(
        synth_score,
        targets,
        evidence,
        min_keep=cfg.PARTIAL_EVIDENCE.MIN_KEEP,
        ls_max=cfg.PARTIAL_EVIDENCE.LS_MAX,
        return_details=True,
    )
    tri_loss, tri_details = partial_evidence_triplet_loss(
        synth_feat,
        targets,
        evidence,
        no_hardneg_below=cfg.PARTIAL_EVIDENCE.NO_HARDNEG_BELOW,
        base_margin=cfg.SOLVER.MARGIN,
        margin_scale=cfg.PARTIAL_EVIDENCE.MARGIN_SCALE,
        no_margin=cfg.MODEL.NO_MARGIN,
        normalize_feature=cfg.SOLVER.TRP_L2,
        return_details=True,
    )
    loss = cfg.MODEL.ID_LOSS_WEIGHT * ce_loss + cfg.MODEL.TRIPLET_LOSS_WEIGHT * tri_loss
    if return_details:
        details = {
            "calibrate": True,
            "ce_loss": ce_loss.detach(),
            "triplet_loss": tri_loss.detach(),
            "weight": ce_details["weight"],
            "smoothing": ce_details["smoothing"],
            "per_sample_ce": ce_details["per_sample"],
            "triplet": tri_details,
            "occ_id": occ_id.detach(),
            "evidence": evidence.detach(),
        }
        return loss, details
    return loss
