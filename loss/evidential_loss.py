"""Evidential Deep Learning loss for classification.

Replaces CrossEntropy with Dirichlet-based Bayes Risk + KL regularization.
Each sample outputs not just a class prediction, but an uncertainty estimate.

Reference: Sensoy et al., "Evidential Deep Learning to Quantify
Classification Uncertainty", NeurIPS 2018.
"""

import torch
import torch.nn.functional as F


def evidential_loss(logits, targets, num_classes, epoch, total_epochs,
                    kl_reg=0.1, anneal_ratio=0.6):
    """
    Args:
        logits: (B, K) raw logits from classifier (before softmax/softplus)
        targets: (B,) integer class labels
        num_classes: K
        epoch: current epoch (1-indexed)
        total_epochs: total training epochs
        kl_reg: maximum KL regularization weight
        anneal_ratio: fraction of training to reach full KL weight
    Returns:
        loss: scalar
        stats: dict with uncertainty and evidence metrics
    """
    # Evidence = softplus(logits) >= 0; Dirichlet params α = evidence + 1 >= 1
    evidence = F.softplus(logits.float())  # force float32 for numerical stability
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)  # Dirichlet strength

    # Bayes Risk: E_Dir[CE] ≈ Σ_k y_k * (log S - log α_k)
    one_hot = F.one_hot(targets, num_classes).float()
    bayes_risk = (one_hot * (torch.log(S) - torch.log(alpha))).sum(dim=1)

    # KL divergence: KL(Dir(α̃) || Dir(1,...,1))
    # α̃ removes evidence for true class (set to 1)
    alpha_tilde = evidence * (1.0 - one_hot) + 1.0  # remove true-class evidence

    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    kl = (
        torch.lgamma(S_tilde.squeeze(1))
        - torch.lgamma(alpha_tilde).sum(dim=1)
        - torch.lgamma(torch.tensor(float(num_classes), device=logits.device))
        + ((alpha_tilde - 1.0) *
           (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=1)
    )

    # KL annealing: 0 → 1 over anneal_ratio of training
    anneal_step = min(1.0, epoch / max(1, total_epochs * anneal_ratio))

    loss = (bayes_risk + kl_reg * anneal_step * kl).mean()

    # Per-sample uncertainty: u = K/S ∈ (0, 1]
    with torch.no_grad():
        uncertainty = (num_classes / S.squeeze(1))
        mean_evidence = evidence.sum(dim=1).mean().item()
        mean_uncertainty = uncertainty.mean().item()
        br_mean = bayes_risk.mean().item()
        kl_mean = kl.mean().item()

    stats = {
        'evidence': mean_evidence,
        'uncertainty': mean_uncertainty,
        'bayes_risk': br_mean,
        'kl': kl_mean,
        'anneal': anneal_step,
    }
    return loss, stats
