"""VC-Norm: Visibility-Conditioned Normalization for per-part tokens.

Motivation (probe-confirmed, scripts/vcnorm_probe.py):
    Occlusion acts as an UNALIGNED domain factor. On a Market-trained model,
    occluded per-keypoint tokens have per-channel normalization statistics
    (mean/var) that drift into a separable sub-region vs visible tokens
    (KL~288, LDA-AUC~0.97 pre-GCN). This is a "occluded vs un-occluded" axis
    riding on top of the identity signal.

This module applies a VISIBILITY-CONDITIONED affine normalization to each
per-keypoint token: it LayerNorm-centers the token, then adds a per-channel
shift predicted from the keypoint's visibility score. The conditioning MLP is
zero-initialized so the module is an EXACT identity at training start and when
untrained — this guarantees the default (untrained) behavior reproduces the
baseline. The accompanying batch-level statistic-alignment loss
(loss/vcnorm_loss.py) is what actually collapses the domain axis; this module
gives the network a visibility-conditioned knob to do so without distorting the
identity-discriminative direction.

Train/test symmetry: the SAME module is applied in both training and inference
forward paths (wired in pose_backbone_model.py for both branches).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisibilityConditionedNorm(nn.Module):
    """Visibility-conditioned affine normalization on per-keypoint tokens.

    For each token x_k (per keypoint k, per image), with visibility score v_k:
        x_ln  = LayerNorm(x_k)                      # remove per-token scale drift
        g, b  = MLP(v_k)                            # per-channel gain / shift
        out   = x_k + g ⊙ (x_ln - x_k_centered) + b # residual, zero-init identity

    The residual is gated so that at init (zero-init MLP last layer) the output
    equals the input exactly. ``tanh`` bounds the gain/shift to keep the affine
    correction conservative (prevents identity erasure via runaway rewriting).

    Args:
        feat_dim: token channel dimension C.
        hidden: hidden width of the conditioning MLP.
        gain_scale: max magnitude of the bounded gain/shift (tanh range).
    """

    def __init__(self, feat_dim, hidden=64, gain_scale=1.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.gain_scale = float(gain_scale)

        # LayerNorm WITHOUT affine: pure centering/scaling reference. The affine
        # part is provided by the visibility-conditioned MLP instead, so the
        # only thing that breaks identity-at-init is the (zero-init) MLP.
        self.ln = nn.LayerNorm(feat_dim, elementwise_affine=False)

        # Conditioning MLP: scalar visibility -> (gain[C], shift[C]).
        self.cond = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * feat_dim),
        )
        # Zero-init last layer -> gain=shift=0 at start -> EXACT identity.
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)

    def forward(self, kp_feats, kp_scores):
        """
        Args:
            kp_feats: (B, K, C) per-keypoint tokens.
            kp_scores: (B, K) visibility/confidence scores in ~[0, 1].

        Returns:
            out: (B, K, C) visibility-conditioned tokens (identity at init).
            stats: dict with gain/shift magnitudes for logging (collapse check).
        """
        B, K, C = kp_feats.shape

        # Centered reference (LayerNorm has no affine -> elementwise standardized).
        x_ln = self.ln(kp_feats)  # (B, K, C)

        # Visibility-conditioned per-channel gain & shift.
        v = kp_scores.clamp(0.0, 1.0).reshape(B * K, 1)  # (B*K, 1)
        gb = self.cond(v)  # (B*K, 2C)
        gb = self.gain_scale * torch.tanh(gb)  # bound the correction
        gain, shift = gb[:, :C], gb[:, C:]  # (B*K, C) each
        gain = gain.view(B, K, C)
        shift = shift.view(B, K, C)

        # Residual, AMP-safe dtype match. Zero-init MLP -> out == kp_feats.
        out = kp_feats + (gain * (x_ln - kp_feats) + shift).to(kp_feats.dtype)

        with torch.no_grad():
            stats = {
                'vcn_gain_abs': float(gain.abs().mean().item()),
                'vcn_gain_std': float(gain.std(unbiased=False).item()),
                'vcn_shift_abs': float(shift.abs().mean().item()),
            }
        return out, stats
