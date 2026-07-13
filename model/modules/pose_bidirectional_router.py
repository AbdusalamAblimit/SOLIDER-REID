"""Pose-supervised bidirectional structural routing (PBSR).

Pose is privileged training supervision only.  The representation path never
reads pose: it decomposes spatial tokens into learnable structural slots and
routes the slot messages back with the same assignment matrix.  Retrieval uses
the refined global feature, not the slots or part matching.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


BODY_PART_GROUPS = (
    (0, 1, 2, 3, 4),       # head
    (5, 6, 11, 12),        # torso
    (5, 7, 9),             # left arm
    (6, 8, 10),            # right arm
    (11, 13, 15),          # left leg
    (12, 14, 16),          # right leg
)


class PoseSupervisedBidirectionalRouter(nn.Module):
    """Coupled spatial-to-slot read and slot-to-spatial write routing."""

    def __init__(self, feat_dim, route_dim=256, num_slots=6, num_heads=4,
                 slot_mixer=True, writeback=True, coupled_write=True,
                 supervision='correct'):
        super().__init__()
        if route_dim % num_heads != 0:
            raise ValueError('route_dim must be divisible by num_heads')
        if num_slots < 1 or num_slots > len(BODY_PART_GROUPS):
            raise ValueError(
                f'num_slots must be in [1, {len(BODY_PART_GROUPS)}]')
        if supervision not in ('correct', 'uniform', 'shuffled', 'none'):
            raise ValueError(
                'supervision must be correct, uniform, shuffled, or none')

        self.feat_dim = int(feat_dim)
        self.route_dim = int(route_dim)
        self.num_body_slots = int(num_slots)
        self.num_slots = self.num_body_slots + 1  # background/reject
        self.num_heads = int(num_heads)
        self.head_dim = self.route_dim // self.num_heads
        self.use_slot_mixer = bool(slot_mixer)
        self.use_writeback = bool(writeback)
        self.coupled_write = bool(coupled_write)
        self.supervision = supervision

        self.slot_queries = nn.Parameter(
            torch.empty(self.num_slots, self.route_dim))
        self.key_proj = nn.Linear(self.feat_dim, self.route_dim)
        self.value_proj = nn.Linear(self.feat_dim, self.route_dim)

        if self.use_slot_mixer:
            self.slot_norm1 = nn.LayerNorm(self.route_dim)
            self.slot_attn = nn.MultiheadAttention(
                self.route_dim, self.num_heads, dropout=0.0,
                batch_first=True)
            self.slot_norm2 = nn.LayerNorm(self.route_dim)
            self.slot_ffn = nn.Sequential(
                nn.Linear(self.route_dim, self.route_dim * 2),
                nn.GELU(),
                nn.Linear(self.route_dim * 2, self.route_dim),
            )

        self.message_norm = nn.LayerNorm(self.route_dim)
        self.message_proj = nn.Linear(self.route_dim, self.route_dim)
        self.out_proj = nn.Linear(self.route_dim, self.feat_dim)
        # Make recomposition spatially meaningful before linear global pooling:
        # each routed message is modulated by the local token it is written to.
        # The update still vanishes exactly when write_scale is zero.
        self.token_norm = nn.LayerNorm(self.feat_dim)
        self.token_gate = nn.Sequential(
            nn.Linear(self.feat_dim, self.route_dim),
            nn.GELU(),
            nn.Linear(self.route_dim, self.feat_dim),
        )

        # A single zero gate gives an exact baseline at initialization while
        # retaining a non-zero first-step gradient for the gate itself.
        self.write_scale = nn.Parameter(torch.zeros(()))

        body_message_mask = torch.ones(self.num_slots)
        body_message_mask[-1] = 0.0  # background carries no identity message
        self.register_buffer(
            'body_message_mask', body_message_mask, persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.slot_queries, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _route(self, tokens):
        """Return per-head routes and projected keys/values.

        Args:
            tokens: B x N x C
        Returns:
            routes: B x H x S x N, softmax-normalized over N
            keys:   B x H x N x Dh
            values: B x H x N x Dh
        """
        batch, num_tokens, _ = tokens.shape
        keys = self.key_proj(tokens).reshape(
            batch, num_tokens, self.num_heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)
        values = self.value_proj(tokens).reshape(
            batch, num_tokens, self.num_heads, self.head_dim)
        values = values.permute(0, 2, 1, 3)

        queries = self.slot_queries.to(dtype=keys.dtype).reshape(
            self.num_slots, self.num_heads, self.head_dim)
        queries = queries.permute(1, 0, 2)
        logits = torch.einsum('hsd,bhnd->bhsn', queries, keys)
        logits = (logits / math.sqrt(self.head_dim)).clamp(-30.0, 30.0)
        routes = F.softmax(logits, dim=-1)
        return routes, keys, values

    def _read_slots(self, routes, values):
        slots_per_head = torch.einsum(
            'bhsn,bhnd->bhsd', routes, values)
        slots = slots_per_head.permute(0, 2, 1, 3).reshape(
            routes.shape[0], self.num_slots, self.route_dim)
        if self.use_slot_mixer:
            normed = self.slot_norm1(slots)
            slots = slots + self.slot_attn(normed, normed, normed,
                                           need_weights=False)[0]
            slots = slots + self.slot_ffn(self.slot_norm2(slots))
        return slots

    def _write_tokens(self, routes, keys, slots, tokens):
        messages = self.message_proj(self.message_norm(slots))
        messages = messages * self.body_message_mask.view(1, -1, 1)
        messages_h = messages.reshape(
            messages.shape[0], self.num_slots,
            self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.coupled_write:
            # Reuse the read assignment, normalized over slots for each token.
            write_weights = routes / routes.sum(dim=2, keepdim=True).clamp_min(1e-6)
            write_weights = write_weights.permute(0, 1, 3, 2)
        else:
            # Parameter-count-matched control: recompute token-to-slot weights
            # from existing keys and slot states, without a second Q/K module.
            slot_keys = slots.reshape(
                slots.shape[0], self.num_slots,
                self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            logits = torch.einsum('bhnd,bhsd->bhns', keys, slot_keys)
            logits = (logits / math.sqrt(self.head_dim)).clamp(-30.0, 30.0)
            write_weights = F.softmax(logits, dim=-1)

        delta_h = torch.einsum(
            'bhns,bhsd->bhnd', write_weights, messages_h)
        delta = delta_h.permute(0, 2, 1, 3).reshape(
            routes.shape[0], routes.shape[-1], self.route_dim)
        delta = self.out_proj(delta)
        local_gate = 2.0 * torch.sigmoid(
            self.token_gate(self.token_norm(tokens))
        )
        return delta * local_gate, write_weights

    def _pose_targets(self, heatmaps, height, width):
        """Build normalized body-part plus background routing targets."""
        heatmaps = F.interpolate(
            heatmaps.float(), size=(height, width),
            mode='bilinear', align_corners=False).clamp_min(0.0)
        body = []
        for group in BODY_PART_GROUPS[:self.num_body_slots]:
            body.append(heatmaps[:, group].amax(dim=1))
        body = torch.stack(body, dim=1)
        background = (1.0 - body.amax(dim=1, keepdim=True)).clamp_min(0.0)
        target = torch.cat([body, background], dim=1).flatten(2)
        mass = target.sum(dim=-1)
        valid = mass > 1e-6
        target = target / mass.unsqueeze(-1).clamp_min(1e-6)
        return target, valid

    def _route_supervision(self, feat_map, heatmaps):
        if self.supervision == 'none':
            return feat_map.new_zeros(())
        if heatmaps is None:
            raise RuntimeError(
                'PBSR route supervision requires target-person heatmaps in training')

        if self.supervision == 'shuffled':
            # Deterministic cross-image mismatch; does not consume global RNG.
            heatmaps = torch.roll(heatmaps, shifts=1, dims=0)

        batch, _, height, width = feat_map.shape
        if self.supervision == 'uniform':
            target = feat_map.new_full(
                (batch, self.num_slots, height * width),
                1.0 / (height * width), dtype=torch.float32)
            valid = torch.ones(
                batch, self.num_slots, dtype=torch.bool,
                device=feat_map.device)
        else:
            target, valid = self._pose_targets(heatmaps, height, width)

        detached_tokens = feat_map.detach().flatten(2).transpose(1, 2)
        supervised_routes, _, _ = self._route(detached_tokens)
        route_mean = supervised_routes.float().mean(dim=1).clamp_min(1e-8)
        kl_per_slot = (
            target * (target.clamp_min(1e-8).log() - route_mean.log())
        ).sum(dim=-1)
        return (kl_per_slot * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

    def forward(self, feat_map, heatmaps=None):
        """Refine a spatial feature map and return training diagnostics.

        ``heatmaps`` only contributes to the auxiliary loss while training.
        It never affects ``refined`` and is ignored in evaluation.
        """
        batch, channels, height, width = feat_map.shape
        if channels != self.feat_dim:
            raise ValueError(
                f'expected {self.feat_dim} channels, got {channels}')

        tokens = feat_map.flatten(2).transpose(1, 2)
        routes, keys, values = self._route(tokens)
        slots = self._read_slots(routes, values)

        if self.use_writeback:
            delta, write_weights = self._write_tokens(
                routes, keys, slots, tokens)
            scale = torch.tanh(self.write_scale).to(dtype=delta.dtype)
            refined_tokens = tokens + scale * delta
        else:
            delta = torch.zeros_like(tokens)
            write_weights = routes.permute(0, 1, 3, 2)
            refined_tokens = tokens

        refined = refined_tokens.transpose(1, 2).reshape(
            batch, channels, height, width)

        route_loss = feat_map.new_zeros(())
        if self.training:
            route_loss = self._route_supervision(feat_map, heatmaps)

        with torch.no_grad():
            route_mean = routes.float().mean(dim=1).clamp_min(1e-8)
            entropy = -(route_mean * route_mean.log()).sum(dim=-1).mean()
            background_share = write_weights.float()[..., -1].mean()
            stats = {
                'write_scale': float(torch.tanh(self.write_scale).item()),
                'route_entropy': float(entropy.item()),
                'background_share': float(background_share.item()),
                'delta_norm': float(delta.float().norm(dim=-1).mean().item()),
                'input_norm': float(tokens.float().norm(dim=-1).mean().item()),
                'slot_norm': float(slots.float().norm(dim=-1).mean().item()),
            }

        return refined, {
            'pbsr_route_loss': route_loss,
            'pbsr_stats': stats,
            'pbsr_routes': route_mean.detach(),
        }
