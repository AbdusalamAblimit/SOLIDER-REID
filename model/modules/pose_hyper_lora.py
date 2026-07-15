"""Per-token pose-conditioned dynamic low-rank transforms.

Unlike a conventional adapter with fixed projections, PoseHyperLoRA uses
the local 17-joint heatmap vector to mix separate banks of down- and
up-projection bases.  Each token therefore receives a pose-dependent
effective matrix while RGB features remain the transformed content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseHyperLoRA(nn.Module):
    """Apply a pose-generated low-rank residual to Swin tokens.

    For token ``x_n`` and local pose ``p_n``::

        A(p_n) = sum_m a_nm(p_n) A_m
        B(p_n) = sum_m b_nm(p_n) B_m
        y_n = x_n + alpha * visibility(p_n) * B(p_n) A(p_n) LN(x_n)

    The visibility factor is derived from the maximum joint confidence.
    Consequently an all-zero pose is an exact identity path, including when
    the pose MLP has non-zero biases after training.
    """

    def __init__(self, feat_dim, rank=4, num_bases=4, pose_hidden_dim=32,
                 residual_scale_init=1e-3, pose_channels=17,
                 factorization='basis'):
        super().__init__()
        if feat_dim <= 0 or rank <= 0 or num_bases <= 0:
            raise ValueError('feat_dim, rank, and num_bases must be positive')
        if pose_hidden_dim <= 0 or pose_channels <= 0:
            raise ValueError('pose_hidden_dim and pose_channels must be positive')

        self.feat_dim = int(feat_dim)
        self.rank = int(rank)
        self.num_bases = int(num_bases)
        self.pose_channels = int(pose_channels)
        self.factorization = str(factorization)
        if self.factorization not in ('basis', 'diagonal'):
            raise ValueError('factorization must be basis or diagonal')

        if self.factorization == 'basis':
            coefficient_dim = 2 * self.num_bases
        else:
            # Projection-matched exp071-style control: fixed A/B with a
            # pose-generated diagonal between them.  M*r keeps the large
            # projection parameter count equal to the factor-wise bank.
            self.effective_rank = self.num_bases * self.rank
            coefficient_dim = self.effective_rank

        self.norm = nn.LayerNorm(self.feat_dim)
        self.pose_mlp = nn.Sequential(
            nn.Linear(self.pose_channels, int(pose_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(pose_hidden_dim), coefficient_dim),
        )
        if self.factorization == 'basis':
            self.a_basis = nn.Parameter(torch.empty(
                self.num_bases, self.rank, self.feat_dim))
            self.b_basis = nn.Parameter(torch.empty(
                self.num_bases, self.feat_dim, self.rank))
        else:
            self.a_basis = nn.Parameter(torch.empty(
                self.effective_rank, self.feat_dim))
            self.b_basis = nn.Parameter(torch.empty(
                self.feat_dim, self.effective_rank))
        self.residual_scale = nn.Parameter(torch.tensor(
            float(residual_scale_init), dtype=torch.float32))

        if self.factorization == 'basis':
            # Each basis is an independent 2-D linear map.  Initialising the
            # whole 3-D bank at once makes PyTorch count M/r/C as convolutional
            # receptive dimensions and shrinks B by roughly sqrt(C), leaving
            # the FP16 residual numerically dead.
            for a_matrix, b_matrix in zip(self.a_basis, self.b_basis):
                nn.init.kaiming_uniform_(a_matrix, a=5 ** 0.5)
                nn.init.kaiming_uniform_(b_matrix, a=5 ** 0.5)
        else:
            nn.init.kaiming_uniform_(self.a_basis, a=5 ** 0.5)
            nn.init.kaiming_uniform_(self.b_basis, a=5 ** 0.5)
        nn.init.xavier_uniform_(self.pose_mlp[0].weight)
        nn.init.zeros_(self.pose_mlp[0].bias)
        # Alpha already provides the conservative near-identity start.  A
        # normal(0, 0.02) head on both A and B would attenuate the residual
        # quadratically and make the hypernetwork effectively dormant.
        nn.init.xavier_uniform_(self.pose_mlp[-1].weight)
        nn.init.zeros_(self.pose_mlp[-1].bias)

    def _local_pose(self, heatmaps, hw_shape, dtype):
        height, width = (int(hw_shape[0]), int(hw_shape[1]))
        if heatmaps.ndim != 4 or heatmaps.shape[1] != self.pose_channels:
            raise ValueError(
                'heatmaps must have shape (B, %d, H, W), got %s'
                % (self.pose_channels, tuple(heatmaps.shape)))
        if heatmaps.shape[-2:] != (height, width):
            heatmaps = F.interpolate(
                heatmaps.float(), size=(height, width), mode='bilinear',
                align_corners=False)
        # ViTPose heatmap magnitude carries joint confidence.  Clamping only
        # removes interpolation/estimator excursions; no sigmoid turns a true
        # zero pose into non-zero evidence.
        heatmaps = heatmaps.clamp(min=0.0, max=1.0).to(dtype=dtype)
        local_pose = heatmaps.permute(0, 2, 3, 1).reshape(
            heatmaps.shape[0], height * width, self.pose_channels)
        visibility = local_pose.amax(dim=-1, keepdim=True)
        return local_pose, visibility

    def forward(self, x, hw_shape, heatmaps, visibility_heatmaps=None):
        if x.ndim != 3 or x.shape[-1] != self.feat_dim:
            raise ValueError(
                'x must have shape (B, N, %d), got %s'
                % (self.feat_dim, tuple(x.shape)))
        height, width = (int(hw_shape[0]), int(hw_shape[1]))
        if x.shape[1] != height * width:
            raise ValueError('token count does not match hw_shape')

        if heatmaps is None:
            zero = x.new_zeros(())
            return x, {
                'visibility_mean': zero,
                'coefficient_abs_mean': zero,
                'delta_rms': zero,
                'residual_scale': self.residual_scale.detach().to(x),
            }

        if heatmaps.shape[0] != x.shape[0]:
            raise ValueError('heatmap batch size must match x batch size')
        local_pose, visibility = self._local_pose(
            heatmaps, (height, width), x.dtype)
        if visibility_heatmaps is not None:
            if visibility_heatmaps.shape[0] != x.shape[0]:
                raise ValueError(
                    'visibility heatmap batch size must match x batch size')
            _, visibility = self._local_pose(
                visibility_heatmaps, (height, width), x.dtype)
        coefficients = torch.tanh(self.pose_mlp(local_pose))

        normalized = self.norm(x)
        if self.factorization == 'basis':
            a_coeff, b_coeff = coefficients.chunk(2, dim=-1)
            # A_m z for every basis, followed by the per-token A mixture.
            down_all = torch.einsum(
                'bnc,mrc->bnmr', normalized, self.a_basis)
            low_rank = torch.einsum('bnmr,bnm->bnr', down_all, a_coeff)
            # Apply the independently generated B mixture without
            # materialising a BxNxMxC tensor.
            weighted_low_rank = low_rank.unsqueeze(2) * b_coeff.unsqueeze(-1)
            delta = torch.einsum(
                'bnmr,mcr->bnc', weighted_low_rank, self.b_basis)
        else:
            # exp071/PCL-style direct control:
            # B @ diag(f(P)) @ A @ LN(x), with the same explicit visibility.
            low_rank = torch.einsum(
                'bnc,rc->bnr', normalized, self.a_basis)
            delta = torch.einsum(
                'bnr,bnr,cr->bnc', low_rank, coefficients, self.b_basis)
        delta = delta * visibility
        output = x + self.residual_scale.to(x.dtype) * delta

        stats = {
            'visibility_mean': visibility.detach().float().mean(),
            'coefficient_abs_mean': coefficients.detach().float().abs().mean(),
            'delta_rms': delta.detach().float().square().mean().sqrt(),
            'residual_scale': self.residual_scale.detach().float(),
        }
        return output, stats
