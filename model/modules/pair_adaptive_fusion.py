import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_distance_tensor(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.clamp_min_(0.0)


def common_support_distance(q_kp, g_kp, q_w, g_w, fallback=None, return_ratio=False):
    q_kp_t = q_kp.transpose(1, 0)  # (K, Q, C)
    g_kp_t = g_kp.transpose(1, 0)  # (K, G, C)
    dot = torch.matmul(q_kp_t, g_kp_t.transpose(2, 1))
    q_sq = q_kp_t.pow(2).sum(dim=-1)
    g_sq = g_kp_t.pow(2).sum(dim=-1)
    kp_dist = (q_sq.unsqueeze(2) - 2 * dot + g_sq.unsqueeze(1)).clamp_min_(0.0).sqrt_()

    weights = torch.sqrt(
        q_w.transpose(1, 0).unsqueeze(2) * g_w.transpose(1, 0).unsqueeze(1)
    )
    weight_sum = weights.sum(dim=0)
    masked = (kp_dist * weights).sum(dim=0) / weight_sum.clamp(min=1e-12)

    if fallback is not None:
        masked = torch.where(weight_sum > 0, masked, fallback)
    support_ratio = weight_sum / max(1, q_kp.shape[1])
    if return_ratio:
        return masked, support_ratio
    return masked


def build_pair_descriptors(global_dist, kp_dist, support_ratio, q_vis_mean, g_vis_mean):
    return torch.stack([
        global_dist,
        kp_dist,
        (global_dist - kp_dist).abs(),
        support_ratio,
        q_vis_mean,
        g_vis_mean,
    ], dim=-1)


def build_query_context_descriptors(base_dist, support_ratio, pair_change=None, valid_mask=None):
    """Build label-free query-level context usable in both train and test."""
    if valid_mask is None:
        valid_mask = torch.ones_like(base_dist, dtype=torch.bool)

    valid_float = valid_mask.float()
    valid_count = valid_float.sum(dim=1).clamp(min=1.0)

    row_mean = (base_dist * valid_float).sum(dim=1) / valid_count
    centered = (base_dist - row_mean.unsqueeze(1)) * valid_float
    row_std = torch.sqrt(centered.pow(2).sum(dim=1) / valid_count).clamp_min_(1e-12)

    inf_fill = torch.full_like(base_dist, float('inf'))
    row_min = torch.where(valid_mask, base_dist, inf_fill).min(dim=1).values
    row_min = torch.where(torch.isfinite(row_min), row_min, row_mean)

    row_support_mean = (support_ratio * valid_float).sum(dim=1) / valid_count
    if pair_change is None:
        row_change_mean = torch.zeros_like(row_mean)
    else:
        row_change_mean = (pair_change * valid_float).sum(dim=1) / valid_count

    row_ctx = torch.stack(
        [row_mean, row_std, row_min, row_support_mean, row_change_mean],
        dim=-1,
    )
    return row_ctx.unsqueeze(1).expand(-1, base_dist.shape[1], -1)


class PairAdaptiveFusionHead(nn.Module):
    """Predict how much each pair should trust common-support distance."""

    def __init__(self, input_dim=6, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, desc):
        alpha = torch.sigmoid(self.mlp(desc))
        return alpha.squeeze(-1)


class PairResidualScorer(nn.Module):
    """Predict a bounded pair-specific residual correction for the base distance."""

    def __init__(self, input_dim=6, hidden_dim=32, delta_scale=0.5):
        super().__init__()
        self.delta_scale = delta_scale
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, desc):
        delta = torch.tanh(self.mlp(desc)) * self.delta_scale
        return delta.squeeze(-1)
