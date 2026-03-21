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


def _normalized_rank(values, valid_mask, descending=False):
    if valid_mask is None:
        valid_mask = torch.ones_like(values, dtype=torch.bool)

    if descending:
        fill = torch.full_like(values, float('-inf'))
    else:
        fill = torch.full_like(values, float('inf'))
    work = torch.where(valid_mask, values, fill)
    order = torch.argsort(work, dim=1, descending=descending)
    base_rank = torch.arange(values.shape[1], device=values.device, dtype=values.dtype)
    base_rank = base_rank.unsqueeze(0).expand_as(values)
    rank = torch.zeros_like(values)
    rank.scatter_(1, order, base_rank)
    valid_count = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    rank = rank / (valid_count - 1.0).clamp(min=1.0)
    neutral = torch.full_like(rank, 0.5)
    return torch.where(valid_mask, rank, neutral)


def build_query_competition_descriptors(base_dist, kp_dist, support_ratio, valid_mask=None):
    """Build pair-specific competition context from a query's candidate set."""
    if valid_mask is None:
        valid_mask = torch.ones_like(base_dist, dtype=torch.bool)

    gain = base_dist - kp_dist
    valid_float = valid_mask.float()
    valid_count = valid_float.sum(dim=1, keepdim=True).clamp(min=1.0)

    gain_mean = (gain * valid_float).sum(dim=1, keepdim=True) / valid_count
    gain_centered = (gain - gain_mean) * valid_float
    gain_std = torch.sqrt(gain_centered.pow(2).sum(dim=1, keepdim=True) / valid_count).clamp_min_(1e-12)
    gain_z = torch.where(valid_mask, gain_centered / gain_std, torch.zeros_like(gain))

    base_rank = _normalized_rank(base_dist, valid_mask, descending=False)
    kp_rank = _normalized_rank(kp_dist, valid_mask, descending=False)
    support_rank = _normalized_rank(support_ratio, valid_mask, descending=True)
    gain_rank = _normalized_rank(gain, valid_mask, descending=True)

    return torch.stack(
        [base_rank, kp_rank, support_rank, gain_rank, gain_z],
        dim=-1,
    )


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


class PairResidualConfidenceScorer(nn.Module):
    """Predict residual correction and a confidence logit for applying it."""

    def __init__(self, input_dim=6, hidden_dim=32, delta_scale=0.5):
        super().__init__()
        self.delta_scale = delta_scale
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Linear(hidden_dim, 1)
        self.conf_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.conf_head.weight)
        nn.init.zeros_(self.conf_head.bias)

    def forward(self, desc):
        feat = self.backbone(desc)
        delta = torch.tanh(self.delta_head(feat)) * self.delta_scale
        conf_logits = self.conf_head(feat)
        return delta.squeeze(-1), conf_logits.squeeze(-1)
