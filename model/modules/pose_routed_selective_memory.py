"""Pose-routed recurrent memory for occluded person ReID.

Pose controls only which body-state slot receives an RGB token and how much
that token may overwrite the carried state.  RGB content alone produces the
candidate state, memory read query, and output residual.  This keeps the
mechanism distinct from pose FiLM / spatial feature gating.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseRoutedSelectiveMemory(nn.Module):
    """Bidirectional vertical scan with pose-routed body-part state slots."""

    # COCO-17: head, torso, left/right arm, left/right leg.  Shoulders and
    # hips intentionally anchor both the torso and their attached limb.
    PART_GROUPS = (
        (0, 1, 2, 3, 4),
        (5, 6, 11, 12),
        (5, 7, 9),
        (6, 8, 10),
        (11, 13, 15),
        (12, 14, 16),
    )

    VALID_ROUTING = {'parts', 'foreground_uniform', 'uniform', 'zero'}

    def __init__(self, feat_dim, state_dim=128, num_parts=6,
                 retention_init=0.95, residual_scale_init=1e-3,
                 routing='parts', bidirectional=True):
        super().__init__()
        if int(num_parts) != len(self.PART_GROUPS):
            raise ValueError(
                'PRSM currently requires six COCO anatomical state slots')
        if routing not in self.VALID_ROUTING:
            raise ValueError(
                'routing must be one of %s, got %r'
                % (sorted(self.VALID_ROUTING), routing))
        if not 0.0 < float(retention_init) < 1.0:
            raise ValueError('retention_init must be strictly between 0 and 1')

        self.feat_dim = int(feat_dim)
        self.state_dim = int(state_dim)
        self.num_parts = int(num_parts)
        self.routing = routing
        self.bidirectional = bool(bidirectional)

        self.input_norm = nn.LayerNorm(self.feat_dim)
        self.candidate_proj = nn.Linear(self.feat_dim, self.state_dim)
        self.read_query_proj = nn.Linear(
            self.feat_dim, self.state_dim, bias=False)
        self.part_keys = nn.Parameter(
            torch.empty(self.num_parts, self.state_dim))

        # Part-specific proposal transform is diagonal and cheap.  Zero delta
        # means every slot initially receives the same RGB candidate.
        self.part_scale_delta = nn.Parameter(
            torch.zeros(self.num_parts, self.state_dim))
        self.part_bias = nn.Parameter(
            torch.zeros(self.num_parts, self.state_dim))

        retention_logit = math.log(
            float(retention_init) / (1.0 - float(retention_init)))
        self.retention_logits = nn.Parameter(torch.full(
            (self.num_parts, self.state_dim), retention_logit))

        self.output_proj = nn.Linear(
            self.state_dim, self.feat_dim, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(
            float(residual_scale_init), dtype=torch.float32))

        nn.init.xavier_uniform_(self.candidate_proj.weight)
        nn.init.zeros_(self.candidate_proj.bias)
        nn.init.xavier_uniform_(self.read_query_proj.weight)
        nn.init.normal_(self.part_keys, std=self.state_dim ** -0.5)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def _pose_routes(self, heatmaps, size, batch, device, dtype):
        """Return soft-part routes [B,K,H,W] and write visibility [B,1,H,W]."""
        height, width = size
        uniform = torch.full(
            (batch, self.num_parts, height, width),
            1.0 / self.num_parts, device=device, dtype=dtype)

        if self.routing == 'uniform':
            visibility = torch.ones(
                batch, 1, height, width, device=device, dtype=dtype)
            return uniform, visibility
        if self.routing == 'zero':
            visibility = torch.zeros(
                batch, 1, height, width, device=device, dtype=dtype)
            return uniform, visibility
        if heatmaps is None:
            raise ValueError(
                'routing=%r requires a 17-channel pose heatmap' % self.routing)
        if heatmaps.ndim != 4 or heatmaps.shape[0] != batch \
                or heatmaps.shape[1] != 17:
            raise ValueError(
                'heatmaps must have shape [B,17,H,W], got %s'
                % (tuple(heatmaps.shape),))
        if not bool(torch.isfinite(heatmaps).all()):
            raise ValueError('heatmaps contain NaN/Inf')

        # ViTPose maps are activations rather than logits.  Clamping preserves
        # true zero as no-pose; a sigmoid would incorrectly turn it into 0.5.
        resized = F.interpolate(
            heatmaps.float(), size=size, mode='bilinear',
            align_corners=False).clamp_(min=0.0, max=1.0)
        part_maps = torch.stack([
            resized[:, indices].amax(dim=1)
            for indices in self.PART_GROUPS
        ], dim=1)
        visibility = part_maps.amax(dim=1, keepdim=True)

        if self.routing == 'foreground_uniform':
            return uniform, visibility.to(dtype=dtype)

        denominator = part_maps.sum(dim=1, keepdim=True)
        routes = part_maps / denominator.clamp_min(1e-8)
        routes = torch.where(
            denominator > 1e-8, routes,
            uniform.to(dtype=routes.dtype))
        return routes.to(dtype=dtype), visibility.to(dtype=dtype)

    def _scan(self, normalized, candidates, routes, visibility):
        """Scan one direction; all pose dependence is confined to ``write``."""
        batch_columns, length, _ = normalized.shape
        state = candidates.new_zeros(
            batch_columns, self.num_parts, self.state_dim)
        retention = torch.sigmoid(self.retention_logits).to(candidates.dtype)
        part_scale = (1.0 + torch.tanh(
            self.part_scale_delta)).to(candidates.dtype)
        part_bias = self.part_bias.to(candidates.dtype)
        part_keys = F.normalize(
            self.part_keys.float(), dim=-1).to(candidates.dtype)
        query_scale = self.state_dim ** -0.5

        reads = []
        entropy = candidates.new_zeros(())
        for token_index in range(length):
            candidate = candidates[:, token_index].unsqueeze(1)
            proposal = candidate * part_scale.unsqueeze(0) \
                + part_bias.unsqueeze(0)
            carried = state * retention.unsqueeze(0)

            # Read the carried *pre-write* state.  The current token cannot
            # take a memoryless shortcut through its own pose-gated write;
            # pose at t can affect only other tokens later in this direction.
            query = self.read_query_proj(
                normalized[:, token_index])
            logits = torch.einsum(
                'bd,kd->bk', query, part_keys) * query_scale
            read_weights = torch.softmax(logits.float(), dim=-1).to(
                candidates.dtype)
            read = (read_weights.unsqueeze(-1) * carried).sum(dim=1)
            reads.append(read)
            entropy = entropy + (-(read_weights.float()
                                    * read_weights.float().clamp_min(1e-8).log())
                                   .sum(dim=-1).mean()).to(entropy.dtype)

            write = (routes[:, token_index]
                     * visibility[:, token_index]).unsqueeze(-1)
            state = carried + write * (proposal - carried)

        return torch.stack(reads, dim=1), entropy / max(length, 1)

    def forward(self, features, heatmaps=None):
        if features.ndim != 4:
            raise ValueError(
                'features must have shape [B,C,H,W], got %s'
                % (tuple(features.shape),))
        batch, channels, height, width = features.shape
        if channels != self.feat_dim:
            raise ValueError(
                'feature channels mismatch: expected %d, got %d'
                % (self.feat_dim, channels))
        if not bool(torch.isfinite(features).all()):
            raise ValueError('features contain NaN/Inf')

        routes, visibility = self._pose_routes(
            heatmaps, (height, width), batch, features.device,
            features.dtype)

        # Each image column is one short anatomical sequence.
        tokens = features.permute(0, 3, 2, 1).reshape(
            batch * width, height, channels)
        route_tokens = routes.permute(0, 3, 2, 1).reshape(
            batch * width, height, self.num_parts)
        visibility_tokens = visibility.permute(0, 3, 2, 1).reshape(
            batch * width, height, 1)

        normalized = self.input_norm(tokens)
        candidates = F.silu(self.candidate_proj(normalized))
        forward_read, forward_entropy = self._scan(
            normalized, candidates, route_tokens, visibility_tokens)

        if self.bidirectional:
            reverse_read, reverse_entropy = self._scan(
                normalized.flip(1), candidates.flip(1),
                route_tokens.flip(1), visibility_tokens.flip(1))
            memory_read = 0.5 * (
                forward_read + reverse_read.flip(1))
            read_entropy = 0.5 * (
                forward_entropy + reverse_entropy)
        else:
            memory_read = forward_read
            read_entropy = forward_entropy

        delta = self.output_proj(memory_read)
        delta = delta.reshape(batch, width, height, channels).permute(
            0, 3, 2, 1).contiguous()
        output = features + self.residual_scale.to(features.dtype) * delta

        write_weights = routes * visibility
        stats = {
            'prsm_residual_scale': self.residual_scale.detach(),
            'prsm_retention_mean': torch.sigmoid(
                self.retention_logits.detach()).mean(),
            'prsm_visibility_mean': visibility.detach().float().mean(),
            'prsm_write_mean': write_weights.detach().float().mean(),
            'prsm_write_max': write_weights.detach().float().amax(),
            'prsm_read_entropy': read_entropy.detach().float(),
            'prsm_delta_ratio': (
                delta.detach().float().norm()
                / features.detach().float().norm().clamp_min(1e-8)),
        }
        return output, stats
