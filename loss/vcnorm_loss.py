"""VC-Norm alignment loss: visibility-conditioned per-keypoint statistic matching.

Core idea (the novel mechanism, distinct from OA-SD / BT-PKD):
    OA-SD distills the pooled GLOBAL feature (cosine); BT-PKD distills per-
    keypoint feature DIRECTION (cosine). Both pull INSTANCE features together.
    VC-Norm instead aligns the BATCH-LEVEL, per-keypoint, per-channel
    NORMALIZATION STATISTICS (first/second moments) of the occluded "student"
    path toward the clean "teacher" path. This removes the "occluded vs
    un-occluded" domain axis that the probe (scripts/vcnorm_probe.py) found in
    those very statistics, WITHOUT supervising individual identities — identity
    survives as the relative position of a token under the aligned statistics.

Why this does NOT erase identity (by construction):
    The loss operates on the per-keypoint mean/var aggregated OVER THE BATCH,
    not on which token belongs to which ID. It shifts the center/scale of a
    keypoint's token cloud to match the clean distribution; it does not pull any
    two specific tokens together. The relative geometry that carries identity is
    left to the ID/triplet losses.

WHICH student tokens enter the statistic (the load-bearing design choice — see
exp328 codex High-1):
    PLBOA occludes lower-body keypoints by (a) graying / pasting an occluder over
    the image region AND (b) zeroing that keypoint's score/visibility/heatmap
    (datasets/pose_dataset.py:871-874). The GCN token itself is NOT a 0-mask:
    keypoint COORDINATES are preserved, so the standard bilinear sampler
    (skeleton_gcn.py:_sample_keypoint_features) still returns a real, degenerate
    "occluded" feature — the backbone's response over the occluder pixels. THAT
    degenerate token is exactly the domain-shifted statistic VC-Norm must pull
    toward the clean teacher.

    The trap: PLBOA sets the occluded keypoint's score to 0, so weighting the
    student moment by the student score (the previous implementation) gives the
    occluded tokens ~zero weight — they never enter the batch mean/var and get
    no alignment gradient. VCA would then only "align" tokens the student already
    sees clearly, which is a no-op against the domain axis (the opposite of the
    goal). Fixed here: the student moment is computed over the OCCLUDED subset
    (teacher-visible AND student-occluded keypoints), weighted by the *teacher*
    score (the student score there is ~0 and useless as a weight). This is the
    cohort whose domain shift we want to collapse, and it is exactly the cohort
    the previous code dropped.

Same yardstick as the probe: per-channel diagonal-Gaussian first/second moment
matching (the exact statistic the probe measured as the separable shift).
"""

import torch
import torch.nn.functional as F


def _weighted_moments(feats, weights, eps=1e-5):
    """Per-keypoint, per-channel weighted mean/var over the batch.

    Args:
        feats: (B, K, C) tokens.
        weights: (B, K) non-negative weights.
        eps: numerical floor.

    Returns:
        mean: (K, C) per-keypoint per-channel weighted mean.
        var:  (K, C) per-keypoint per-channel weighted variance.
        wsum: (K,) total weight per keypoint (for validity gating).
    """
    # (B, K, 1) weights broadcast over channels.
    w = weights.clamp(min=0.0).unsqueeze(-1)  # (B, K, 1)
    wsum = w.sum(dim=0)  # (K, 1)
    wsum_safe = wsum.clamp(min=eps)

    mean = (w * feats).sum(dim=0) / wsum_safe  # (K, C)
    # E[x^2] - E[x]^2 (weighted).
    sq = (w * feats.pow(2)).sum(dim=0) / wsum_safe  # (K, C)
    var = (sq - mean.pow(2)).clamp(min=eps)  # (K, C)
    return mean, var, wsum.squeeze(-1)  # (K,C),(K,C),(K,)


def vcnorm_align_loss(student_kp, teacher_kp, student_scores, teacher_scores,
                      vis_thr=0.3, min_weight=1.0, eps=1e-5):
    """Align OCCLUDED-student per-keypoint normalization statistics to teacher's.

    Args:
        student_kp: (B, K, C) occluded-path per-keypoint tokens (grad).
        teacher_kp: (B, K, C) clean-path tokens (will be detached -> target).
        student_scores: (B, K) student visibility scores (PLBOA-occluded -> ~0).
        teacher_scores: (B, K) teacher visibility scores (clean -> mostly high).
        vis_thr: a keypoint is "visible" iff its score >= vis_thr. Teacher
            keypoints below vis_thr are skipped (no reliable clean target).
            Student keypoints below vis_thr are the OCCLUDED cohort we align.
        min_weight: a keypoint must accumulate at least this much weight on BOTH
            the teacher side (reliable target exists) AND the student-occluded
            side (the occluded cohort is actually present in this batch) to
            contribute. This is the dual min-count gate.
        eps: numerical floor.

    Returns:
        loss: scalar alignment loss (0 if no valid keypoint).
        stats: dict with before/after moment distance, valid-keypoint count, etc.

    Mechanism (codex High-1 fix): the student moment for keypoint k is taken over
    the tokens that are OCCLUDED in the student but VISIBLE in the teacher
    (``s_sc < vis_thr`` AND ``t_sc >= vis_thr``), weighted by the teacher score
    (the student score is ~0 there and useless as a weight). Pulling THAT
    occluded-token cloud toward the clean teacher cloud is what collapses the
    occlusion domain axis. Visible student tokens are intentionally NOT used to
    drive the student moment — they are not domain-shifted and aligning them is a
    no-op against the axis the probe found.
    """
    # Cast to float32 for stable moment statistics under AMP.
    s = student_kp.float()
    t = teacher_kp.float().detach()  # one-directional: teacher is the target
    s_sc = student_scores.float()
    t_sc = teacher_scores.float()

    B, K, C = s.shape

    # Teacher visibility mask (reliable clean target exists).
    t_vis = (t_sc >= vis_thr)                       # (B, K)
    # Student occlusion mask (PLBOA-occluded student token).
    s_occ = (s_sc < vis_thr)                        # (B, K)

    # Teacher target moments: over VISIBLE teacher keypoints, weighted by score.
    t_w = t_vis.float() * t_sc.clamp(min=0.0)       # (B, K)
    # Student moments: over the OCCLUDED-but-teacher-visible cohort. Weight by the
    # TEACHER score (student score is ~0 here so it cannot serve as a weight).
    # This is the codex High-1 fix: the occluded student tokens now enter the
    # batch statistic and receive the alignment gradient.
    s_w = (s_occ & t_vis).float() * t_sc.clamp(min=0.0)  # (B, K)

    s_mean, s_var, s_wsum = _weighted_moments(s, s_w, eps=eps)  # (K,C),(K,C),(K,)
    t_mean, t_var, t_wsum = _weighted_moments(t, t_w, eps=eps)  # (K,C),(K,C),(K,)

    # Dual validity gate (codex Medium-b): keypoint contributes only if BOTH
    #   - teacher accumulated enough visible weight (reliable target), AND
    #   - the occluded-student cohort accumulated enough weight (it exists here).
    valid_k = (t_wsum >= min_weight) & (s_wsum >= min_weight)  # (K,)
    if valid_k.sum() == 0:
        loss = (student_kp * 0.0).sum()  # keep graph, zero grad
        stats = {
            'vca_loss': 0.0, 'vca_valid_k': 0.0,
            'vca_mean_dist': 0.0, 'vca_std_dist': 0.0,
            'vca_occ_ratio': float((s_occ & t_vis).float().mean().item()),
        }
        return loss, stats

    # Diagonal-Gaussian first/second moment match (same family as the probe):
    #   mean term: ||mu_s - mu_t||^2 per channel
    #   scale term: (std_s - std_t)^2 per channel   (W2-style, scale-stable)
    mean_term = (s_mean - t_mean.detach()).pow(2)  # (K, C)
    std_term = (s_var.clamp(min=eps).sqrt()
                - t_var.detach().clamp(min=eps).sqrt()).pow(2)  # (K, C)

    per_k = (mean_term + std_term).mean(dim=1)  # (K,) avg over channels
    loss = per_k[valid_k].mean()

    with torch.no_grad():
        stats = {
            'vca_loss': float(loss.item()),
            'vca_valid_k': float(valid_k.sum().item()),
            'vca_mean_dist': float(mean_term[valid_k].mean().item()),
            'vca_std_dist': float(std_term[valid_k].mean().item()),
            # fraction of (token) slots that are the occluded-aligned cohort
            'vca_occ_ratio': float((s_occ & t_vis).float().mean().item()),
        }
    return loss, stats
