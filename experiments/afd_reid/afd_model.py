# encoding: utf-8
"""
AFD-ReID model: ResNet50 backbone + GeM/avg pool + BNNeck + ID classifier.

Baseline (use_afd=False) is a standard strong ReID baseline (Luo et al. BoT):
    resnet50(IMAGENET1K_V1) -> last_stride=1 -> pool -> BNNeck -> linear classifier
    train returns (global_feat, logits); test returns L2-normalized BN feature.

AFD modules (use_afd=True) -- altitude-frequency decoupling, both switchable:
  * Frequency Reliability Router (afd_router):
      split a shallow feature map into low / mid / high frequency bands via 2D FFT,
      predict per-band reliability weights with a light gate (optionally conditioned
      on view/altitude), and recombine the bands. The recombined feature replaces the
      input to the rest of the backbone. Designed to *down-weight unreliable high-freq*
      content under aerial (low-resolution) views.
  * Cross-View Frequency Counterfactual (afd_cvfc):
      training-time regularizer (no test-time cost). Two switchable sub-mechanisms:
        - high-band dropout: zero the high-freq band of the shallow feature with prob p,
          producing a counterfactual view that must keep the same identity.
        - low/high consistency: penalize the distance between the embedding of the full
          feature and the embedding of its low-pass counterfactual, so identity relies
          on the stable low/mid band shared across A<->G.
      The CVFC forward exposes counterfactual features; the losses live in afd_train.py
      so the mechanism stays cleanly ablatable.

Every AFD piece is gated by flags so `use_afd=False` reproduces the baseline exactly
(the AFD submodules are not even constructed when use_afd=False).
"""
import copy
import math
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# --------------------------------------------------------------------------- #
# Swin-Small backbone (SOLIDER) -- optional, team asset, for SOTA push.
# Lazy-built ONLY when backbone='swin_small'; the resnet50 path never touches
# any of this so it stays byte-for-byte identical.
# --------------------------------------------------------------------------- #
# Repo root = .../SOLIDER-REID (this file lives at experiments/afd_reid/).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', '..'))


def _ensure_mmcv_stub():
    """The SOLIDER swin_transformer.py does, at import time,
        try:    from mmcv.runner import load_checkpoint as _load_checkpoint
        except ImportError: from mmengine.runner import load_checkpoint ...
    but `_load_checkpoint` is never actually called (init_weights uses
    torch.load directly).  The default python on the training boxes has neither
    mmcv nor mmengine, so the bare `import` would crash.  Register lightweight
    stub modules (with a dummy `load_checkpoint`) so the import line succeeds.
    Real mmcv/mmengine, if present, are left untouched.  Called ONLY from the
    swin branch -> resnet50 import path is unaffected.
    """
    for pkg, sub in (('mmcv', 'mmcv.runner'), ('mmengine', 'mmengine.runner')):
        try:
            __import__(sub)
            continue  # the real package is importable -> do not stub it
        except Exception:
            pass
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)
        if sub not in sys.modules:
            sys.modules[sub] = types.ModuleType(sub)
        # `from X.runner import load_checkpoint` needs the attribute to exist.
        sys.modules[sub].load_checkpoint = lambda *a, **k: None
        setattr(sys.modules[pkg], 'runner', sys.modules[sub])


class SwinBackboneReID(nn.Module):
    """Thin wrapper around the SOLIDER swin_small backbone for the AFD/OVLI
    trainer.

    Exposes the SAME hook contract as the resnet50 AFDModel:
      * `self.layer4` is an ``nn.Identity`` whose forward is fed the last-stage
        spatial map, so an OVLI ``model.layer4.register_forward_hook`` captures a
        ``(B, C, H, W)`` map WITHOUT detaching (gradient flows backbone->proj).
      * forward returns ``(feat_map, None)`` where feat_map is that same NCHW map
        (AFDModel pools it -> BNNeck), mirroring resnet's ``_forward_backbone``.

    swin_small: embed_dims=96, depths=(2,2,18,2), num_features=[96,192,384,768];
    last-stage channel = 768 (set as the model's in_planes).  For a 256x128 input
    the last map is (B,768,8,4); for 384x128 it is (B,768,12,4).
    """

    OUT_DIM = 768

    def __init__(self, img_size=(256, 128), pretrain_path='', semantic_weight=0.2,
                 drop_path_rate=0.1, iso_branch=False, iso_stage=3,
                 iso_trunk_recce=True):
        super().__init__()
        _ensure_mmcv_stub()
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        # import AFTER the stub is registered and repo root is on sys.path
        from model.backbones.swin_transformer import swin_small_patch4_window7_224

        # img_size is (H, W); the factory takes it through to PatchEmbed.
        self.swin = swin_small_patch4_window7_224(
            img_size=list(img_size),
            drop_path_rate=drop_path_rate,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            semantic_weight=semantic_weight,   # SOLIDER ReID default 0.2
            convert_weights=False,             # teacher ckpt is already in-repo layout
        )
        self.out_dim = self.swin.num_features[-1]   # 768 for swin_small
        if pretrain_path:
            # loads the SOLIDER 'teacher' checkpoint (backbone.* keys), strict=False
            self.swin.init_weights(pretrain_path)
        else:
            self.swin.init_weights(None)            # trunc-normal from scratch
        # Identity hook point so OVLI's model.layer4 forward-hook gets the NCHW map.
        self.layer4 = nn.Identity()

        # ---- AIRL gradient-isolated dual-branch (f_rec independent late stage) --
        # iso_branch=True forks a SECOND last-stage path (f_rec) off the shared
        # residual stream at the input of stage `iso_stage`.  The rec path is an
        # INDEPENDENT deep-copy of swin.stages[iso_stage:] (+ that stage's output
        # norm).  Two gradient regimes on the fork-point feed, governed by
        # iso_trunk_recce (the trunk-undersupervision FIX):
        #
        #   * DEGRADED (rec_only=True, the consistency pass): the fork feed is ALWAYS
        #     detach()ed -> the AIRL degradation-consistency gradient updates ONLY the
        #     rec copy + BNNeck_rec and NEVER reaches the shared trunk.  This is the
        #     isolation invariant that keeps f_rec a specialised "recover expert" and
        #     protects the clean trunk + f_full from being pulled toward degradation-
        #     robustness.  Holds for BOTH settings of iso_trunk_recce.
        #
        #   * CLEAN (rec_only=False, the main forward):
        #       - iso_trunk_recce=True  (default, the FIX): the fork feed is NOT
        #         detached, so f_rec's CLEAN ID-CE gradient FLOWS BACK into the shared
        #         trunk.  Diagnosis (codex consensus): the original full-detach iso cut
        #         the trunk's extra identity supervision (f_rec's clean ID-CE only
        #         updated the detached rec tail), leaving f_full WEAKER than even the
        #         fully-shared dual-branch (whose trunk saw both heads' ID-CE).
        #         Re-routing ONLY the clean ID-CE to the trunk restores that extra
        #         identity supervision -> strengthens f_full, while the degradation-
        #         consistency stays detached (above) -> f_rec stays specialised.
        #       - iso_trunk_recce=False (ablation): the clean fork feed is ALSO
        #         detached -> the ORIGINAL full-isolation iso (clean ID-CE + consistency
        #         both severed from the trunk).  Kept for the controlled comparison
        #         "does the clean-ID-CE trunk reflow help, or just any change?".
        #
        # OFF (iso_branch=False) -> nothing is built and the forward is byte-for-byte
        # the single-map baseline.
        self.iso_branch = bool(iso_branch)
        self.iso_stage = int(iso_stage)
        # iso_trunk_recce: whether the CLEAN rec ID-CE gradient reflows into the
        # shared trunk (True, the fix) or the clean fork feed is also detached
        # (False, original full-isolation ablation).  No effect when iso_branch off.
        self.iso_trunk_recce = bool(iso_trunk_recce)
        if self.iso_branch:
            n_stages = len(self.swin.stages)
            if not (1 <= self.iso_stage <= n_stages - 1):
                raise ValueError(
                    f"iso_stage must be in [1, {n_stages - 1}] (fork after the "
                    f"shared early stages, before the last); got {self.iso_stage}")
            # the rec branch re-runs stages [iso_stage .. last] on its OWN copy.
            # deep-copy preserves the pretrained weights as the f_rec init (same
            # starting point as f_full's stages -> divergence comes from training,
            # not from a random re-init that would cripple f_rec).
            self.rec_stages = nn.ModuleList(
                copy.deepcopy(self.swin.stages[i]) for i in range(self.iso_stage,
                                                                  n_stages))
            # the last output norm (norm{last}) applied to the rec last-stage map,
            # an independent copy so f_rec gets its own LayerNorm (matches the
            # f_full norm recipe; reshaped exactly like swin.forward does).
            last = n_stages - 1
            self.rec_norm = copy.deepcopy(getattr(self.swin, f'norm{last}'))
            # independent copies of the semantic-embed Linears for the rec stages
            # (frozen, requires_grad=False -- same as the trunk's; deep-copy keeps
            # the same frozen weights so the rec stream is modulated identically to
            # the trunk at init).  swin keeps one (w,b) pair PER stage index i; the
            # rec branch runs stages [iso_stage..last] so it needs those indices.
            if self.swin.semantic_weight >= 0:
                self.rec_semantic_embed_w = nn.ModuleList(
                    copy.deepcopy(self.swin.semantic_embed_w[i])
                    for i in range(self.iso_stage, n_stages))
                self.rec_semantic_embed_b = nn.ModuleList(
                    copy.deepcopy(self.swin.semantic_embed_b[i])
                    for i in range(self.iso_stage, n_stages))
                # the deep-copied Linears already carry requires_grad=False (the
                # trunk froze them); re-assert defensively so the rec semantic embed
                # is never trained even if a future deepcopy reset the flag.
                for p in self.rec_semantic_embed_w.parameters():
                    p.requires_grad = False
                for p in self.rec_semantic_embed_b.parameters():
                    p.requires_grad = False
            # Identity hook point for the rec last-stage map (mirrors self.layer4);
            # kept for parity / future hooks -- the rec map is a fresh path so OVLI's
            # single layer4 hook (on the f_full map) is unaffected.
            self.layer4_rec = nn.Identity()

    def _run_rec_stages(self, x, hw_shape, semantic_weight):
        """Run the INDEPENDENT rec copy of stages [iso_stage..last] on the residual
        stream `x` (the fork feed) and return the rec last-stage NCHW map.

        The caller (_forward_swin_split) decides whether `x` is detached: the DEGRADED
        consistency pass always passes a detached fork (gradient isolation), while the
        CLEAN pass with iso_trunk_recce=True passes a NON-detached fork so the clean
        f_rec ID-CE reflows into the shared trunk.  This method itself is agnostic to
        that choice -- it just runs the rec stages over whatever `x` it is given.

        Replicates SwinTransformer.forward's per-stage body EXACTLY (stage -> per-
        stage semantic-embed on the continuing stream -> final-stage norm + reshape)
        but over self.rec_stages / self.rec_norm / self.rec_semantic_embed_*, so the
        rec map is computed the same way f_full's map is -- the ONLY differences are
        (a) independent weights and (b) the fork input (detached or not per above).
        """
        n_stages = len(self.swin.stages)
        last = n_stages - 1
        rec_out = None
        for j, stage in enumerate(self.rec_stages):
            i = self.iso_stage + j               # absolute stage index
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin.semantic_weight >= 0:
                sw = self.rec_semantic_embed_w[j](semantic_weight).unsqueeze(1)
                sb = self.rec_semantic_embed_b[j](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb
            if i == last:
                out = self.rec_norm(out)
                out = out.view(-1, *out_hw_shape,
                               self.swin.num_features[i]).permute(
                                   0, 3, 1, 2).contiguous()
                rec_out = out
        return self.layer4_rec(rec_out)

    def _forward_swin_split(self, x, rec_only=False):
        """Replicate SwinTransformer.forward but ALSO branch the rec path.

        Returns (f_full_map, f_rec_map).  The shared patch_embed + ALL f_full stages
        run FIRST, exactly and in the same order as swin.forward (so even the
        training-time stochastic-depth / DropPath RNG sequence f_full sees is
        identical to the single-branch path -- the rec copy runs AFTER the full loop,
        not interleaved, so it cannot perturb f_full's RNG draws); the residual
        stream at the input of stage `iso_stage` is captured and fed through the
        independent rec stages afterward.  semantic_weight is built identically to
        swin.forward.

        Gradient regime on the rec fork feed (the trunk-undersupervision FIX):
          * rec_only=True (degraded consistency pass): the fork feed is ALWAYS
            DETACHED -> the consistency gradient cannot reach the shared trunk
            (the isolation invariant, independent of iso_trunk_recce).
          * rec_only=False (clean main pass): the fork feed is detached ONLY when
            self.iso_trunk_recce is False (original full-isolation ablation).  When
            iso_trunk_recce is True (the fix, default) the clean fork feed is NOT
            detached, so f_rec's CLEAN ID-CE gradient reflows into the shared trunk
            (extra identity supervision that strengthens f_full).  The
            degradation-consistency still uses the rec_only=True detached path, so
            it never reaches the trunk regardless of this setting.

        rec_only=True: skip the f_full BNNeck-side work entirely is done by the
        CALLER (it just ignores full_map); here rec_only additionally lets the
        degraded consistency pass avoid keeping the f_full map's grad graph -- we
        still must run the shared stages to REACH the fork point, but we do NOT need
        f_full's last-stage norm/grad, so full_map is returned detached to make the
        "f_full untouched by the degraded pass" contract explicit and cheap.
        """
        swin = self.swin
        # build the per-sample semantic weight exactly like SwinTransformer.forward
        if swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1) * swin.semantic_weight
            w = torch.cat([w, 1 - w], axis=-1)
            semantic_weight = w.to(x.device)
        else:
            semantic_weight = None

        x, hw_shape = swin.patch_embed(x)
        if swin.use_abs_pos_embed:
            x = x + swin.absolute_pos_embed
        x = swin.drop_after_pos(x)

        # Whether the rec fork feed is detached from the shared trunk:
        #   * degraded consistency pass (rec_only) -> ALWAYS detach (isolation
        #     invariant: consistency grad never reaches the trunk).
        #   * clean pass -> detach ONLY when iso_trunk_recce is False (original
        #     full-isolation ablation); when True (the fix) keep the graph so the
        #     clean rec ID-CE reflows into the trunk (extra identity supervision).
        detach_fork = bool(rec_only) or (not self.iso_trunk_recce)
        fork_x = None
        fork_hw = None
        full_map = None
        for i, stage in enumerate(swin.stages):
            # the residual stream `x` HERE is the input to stage i.  When i ==
            # iso_stage, snapshot this stream (the gradient-isolation boundary) to
            # feed the rec branch AFTER the full loop.  Detach per detach_fork above:
            # detached -> rec grad severed from trunk; non-detached -> clean rec
            # ID-CE flows back into the trunk (the trunk-undersupervision fix).
            if i == self.iso_stage:
                fork_x = x.detach() if detach_fork else x
                fork_hw = hw_shape
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if swin.semantic_weight >= 0:
                sw = swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * swin.softplus(sw) + sb
            if i in swin.out_indices and i == len(swin.stages) - 1:
                norm_layer = getattr(swin, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               swin.num_features[i]).permute(
                                   0, 3, 1, 2).contiguous()
                full_map = out
        full_map = self.layer4(full_map)        # Identity passthrough (OVLI hook)
        if rec_only:
            # the degraded consistency pass only needs f_rec; detach f_full's map so
            # no f_full grad graph is built and the contract "the degraded pass does
            # not train f_full" is explicit.  (running stats of self.bottleneck are
            # NOT updated for this pass because the caller never pools full_map -> no
            # BatchNorm forward on it; see AFDModel.forward rec_only path.)
            full_map = full_map.detach()
        # rec branch: independent late stages on the fork stream.  fork_x is detached
        # per detach_fork (degraded/ablation -> isolated; clean+fix -> grad reflows to
        # trunk).  The semantic weight is a FROZEN constant (no params), so detaching
        # it is harmless: it never blocks gradient through fork_x itself (the rec
        # multiply x*softplus(sw)+sb keeps x's graph).  Run AFTER the f_full loop so
        # f_full's RNG is unchanged.
        rec_map = self._run_rec_stages(
            fork_x, fork_hw,
            None if semantic_weight is None else semantic_weight.detach())
        return full_map, rec_map

    def forward(self, x, return_rec=False, rec_only=False):
        # Default path (return_rec=False OR iso off): SwinTransformer.forward ->
        # (global_feat(B,768), outs[list of NCHW maps]); take the last spatial map
        # and route it through self.layer4 so the OVLI hook fires (no detach -> grad
        # flows).  Byte-for-byte the original single-map behaviour.
        if not (self.iso_branch and return_rec):
            _gfeat, outs = self.swin(x)
            feat_map = self.layer4(outs[-1])        # (B,768,H,W), Identity passthrough
            return feat_map
        # iso dual-branch path: run the split forward -> (f_full map, f_rec map).
        # The rec map is computed through independent late stages.  The DEGRADED pass
        # (rec_only) forks off a DETACHED trunk so the consistency loss cannot perturb
        # the shared trunk; the CLEAN pass forks off a NON-detached trunk when
        # iso_trunk_recce=True (clean f_rec ID-CE reflows -> extra trunk supervision),
        # else detached (full-isolation ablation).  See _forward_swin_split.
        full_map, rec_map = self._forward_swin_split(x, rec_only=rec_only)
        return full_map, rec_map


# --------------------------------------------------------------------------- #
# weight init (BoT style)
# --------------------------------------------------------------------------- #
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# --------------------------------------------------------------------------- #
# GeM pooling
# --------------------------------------------------------------------------- #
class GeMPool(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)
        return x.flatten(1)


# --------------------------------------------------------------------------- #
# Frequency band decomposition (shared by Router and CVFC)
# --------------------------------------------------------------------------- #
def _band_masks(H, W, low_r=0.125, mid_r=0.30, device='cpu', dtype=torch.float32):
    """Build (low, mid, high) centered rectangular FFT-shifted masks on an HxW grid.

    low  : central box of half-size low_r*(H,W)
    mid  : ring between low_r and mid_r
    high : everything outside mid_r
    Masks sum to 1 everywhere -> band partition is exact and lossless.
    """
    cy, cx = H // 2, W // 2
    ry1, rx1 = max(1, int(H * low_r)), max(1, int(W * low_r))
    ry2, rx2 = max(ry1 + 1, int(H * mid_r)), max(rx1 + 1, int(W * mid_r))

    low = torch.zeros(H, W, device=device, dtype=dtype)
    low[cy - ry1:cy + ry1, cx - rx1:cx + rx1] = 1.0

    midbox = torch.zeros(H, W, device=device, dtype=dtype)
    midbox[cy - ry2:cy + ry2, cx - rx2:cx + rx2] = 1.0
    mid = midbox - low                     # ring
    high = 1.0 - midbox                     # outside
    return low, mid, high


def decompose_bands(x, low_r=0.125, mid_r=0.30):
    """Decompose a feature map x:(B,C,H,W) into (low, mid, high) spatial-domain maps.

    Sum of the three equals x (up to fp error). Done per-sample via 2D FFT over (H,W).
    """
    B, C, H, W = x.shape
    f = torch.fft.fftshift(torch.fft.fft2(x.float(), dim=(-2, -1)), dim=(-2, -1))
    low_m, mid_m, high_m = _band_masks(H, W, low_r, mid_r, device=x.device)
    out = []
    for m in (low_m, mid_m, high_m):
        fm = f * m.view(1, 1, H, W)
        band = torch.fft.ifft2(torch.fft.ifftshift(fm, dim=(-2, -1)), dim=(-2, -1)).real
        out.append(band.to(x.dtype))
    return out  # [low, mid, high]


# --------------------------------------------------------------------------- #
# AFD Module 1: Frequency Reliability Router
# --------------------------------------------------------------------------- #
class FrequencyReliabilityRouter(nn.Module):
    """Re-weight (low, mid, high) bands of a feature map by learned reliability.

    weight = softmax over 3 bands, optionally conditioned on view (aerial/ground).
    output = w_low*low + w_mid*mid + w_high*high   (residual-blended with input).

    cond_on_view: if True, the gate input is concatenated with a learned per-view
    embedding so aerial vs ground can learn different band reliabilities.
    """

    def __init__(self, channels, low_r=0.125, mid_r=0.30, cond_on_view=True,
                 residual=True):
        super().__init__()
        self.low_r = low_r
        self.mid_r = mid_r
        self.cond_on_view = cond_on_view
        self.residual = residual

        gate_in = channels + (2 if cond_on_view else 0)
        hidden = max(channels // 8, 16)
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        if cond_on_view:
            # view embedding: index 0 = Aerial, 1 = Ground
            self.view_emb = nn.Embedding(2, 2)
            nn.init.zeros_(self.view_emb.weight)
        # init last gate layer near-zero so initial weights ~ uniform (stable start)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x, view_idx=None):
        """x:(B,C,H,W); view_idx:(B,) long in {0,1} or None.

        Returns (recombined feature, band_weights(B,3)).
        """
        low, mid, high = decompose_bands(x, self.low_r, self.mid_r)
        ctx = F.adaptive_avg_pool2d(x, 1).flatten(1)          # (B,C)
        if self.cond_on_view and view_idx is not None:
            ctx = torch.cat([ctx, self.view_emb(view_idx)], dim=1)
        elif self.cond_on_view:
            # no view at hand (e.g. some test paths) -> zero pad
            ctx = torch.cat([ctx, ctx.new_zeros(ctx.size(0), 2)], dim=1)
        w = torch.softmax(self.gate(ctx), dim=1)              # (B,3)
        wl, wm, wh = w[:, 0], w[:, 1], w[:, 2]

        def b(weight, band):
            return weight.view(-1, 1, 1, 1) * band

        recomb = b(wl, low) + b(wm, mid) + b(wh, high)
        # scale by 3 so that uniform weights (1/3 each) reproduce the input map
        recomb = recomb * 3.0
        if self.residual:
            recomb = 0.5 * x + 0.5 * recomb
        return recomb, w


# --------------------------------------------------------------------------- #
# AFD Module 2: Cross-View Frequency Counterfactual (training-time)
# --------------------------------------------------------------------------- #
class CrossViewFrequencyCounterfactual(nn.Module):
    """Produce counterfactual feature maps for the consistency / dropout losses.

    Pure functional band ops -- no learnable params; kept as a module so the
    behavior is centralized and switchable. Used only in training.
    """

    def __init__(self, low_r=0.125, mid_r=0.30, high_drop_p=0.5):
        super().__init__()
        self.low_r = low_r
        self.mid_r = mid_r
        self.high_drop_p = high_drop_p

    def high_band_dropout(self, x):
        """Per-sample: with prob high_drop_p, drop the high band (keep low+mid)."""
        low, mid, high = decompose_bands(x, self.low_r, self.mid_r)
        B = x.size(0)
        keep = (torch.rand(B, device=x.device) >= self.high_drop_p).float()
        keep = keep.view(B, 1, 1, 1)
        return low + mid + keep * high

    def low_pass(self, x):
        """Keep only low+mid band (a stable, view-invariant counterfactual)."""
        low, mid, _ = decompose_bands(x, self.low_r, self.mid_r)
        return low + mid


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class AFDModel(nn.Module):
    def __init__(self, num_classes, last_stride=1, pool='gem',
                 pretrained=True,
                 use_afd=False,
                 afd_router=True, afd_cvfc=True,
                 afd_stage='layer1',
                 router_cond_view=True,
                 low_r=0.125, mid_r=0.30, high_drop_p=0.5,
                 backbone='resnet50',
                 swin_pretrain='', swin_semantic_weight=0.2,
                 img_size=(256, 128),
                 airl_dualbranch=False,
                 airl_dualbranch_iso=False, airl_iso_stage=3,
                 airl_iso_trunk_recce=True):
        super().__init__()
        self.backbone = backbone
        self.use_afd = use_afd
        # AIRL dual-branch: a SECOND BNNeck head (bottleneck_rec + classifier_rec)
        # over the SAME shared backbone feature map.  f_full (the original head)
        # keeps full-resolution identity evidence (protects G->A); f_rec (this
        # second head) additionally carries the AIRL ground-degradation
        # consistency at train time, so it learns identity evidence recoverable
        # under a low (aerial) pixel budget (serves A->G).  At eval the two
        # heads' cosine scores are SOFT-fused at the distance-matrix level
        # (cos = w*cos_rec + (1-w)*cos_full) -- ONE forward yields both features.
        # OFF (default) -> the second head is not even constructed and forward
        # returns exactly the single-head dict/eval tensor (byte-for-byte base).
        self.airl_dualbranch = bool(airl_dualbranch)
        # AIRL gradient-isolated dual-branch: the SAME two-head + soft-fusion idea
        # as airl_dualbranch, but f_rec is NOT a BNNeck over the shared global_feat;
        # it is a BNNeck over an INDEPENDENT late Swin stage forked off a DETACHED
        # trunk feature (see SwinBackboneReID.iso_branch).  This severs the f_rec
        # consistency gradient from the shared trunk so the clean trunk + f_full are
        # not pulled toward degradation-robustness -> the two heads re-diverge.
        # swin-only (the fork lives in the Swin stage list); mutually exclusive with
        # the shared airl_dualbranch (same eval/loss contract, different f_rec path).
        self.airl_dualbranch_iso = bool(airl_dualbranch_iso)
        self.airl_iso_stage = int(airl_iso_stage)
        # airl_iso_trunk_recce: route the CLEAN f_rec ID-CE gradient back into the
        # shared trunk (True, default = the trunk-undersupervision fix) vs the
        # original full-isolation iso where the clean fork feed is also detached
        # (False). The degradation-consistency stays trunk-isolated either way.
        self.airl_iso_trunk_recce = bool(airl_iso_trunk_recce)
        if self.airl_dualbranch_iso:
            assert not self.airl_dualbranch, (
                "airl_dualbranch_iso and airl_dualbranch are mutually exclusive "
                "(shared vs gradient-isolated f_rec; pick one).")
            assert backbone == 'swin_small', (
                "airl_dualbranch_iso requires backbone='swin_small' (the rec branch "
                "forks an independent Swin late stage).")
        self.afd_router = use_afd and afd_router
        self.afd_cvfc = use_afd and afd_cvfc
        self.afd_stage = afd_stage

        if backbone == 'resnet50':
            self.in_planes = 2048

            weights = 'IMAGENET1K_V1' if pretrained else None
            resnet = torchvision.models.resnet50(weights=weights)
            # ReID standard: last block stride 1 -> larger spatial map (16x8 for 256x128)
            if last_stride == 1:
                resnet.layer4[0].conv2.stride = (1, 1)
                resnet.layer4[0].downsample[0].stride = (1, 1)

            # split backbone so the router can be inserted after a shallow stage
            self.stem = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1 = resnet.layer1   # 256 ch
            self.layer2 = resnet.layer2   # 512 ch
            self.layer3 = resnet.layer3   # 1024 ch
            self.layer4 = resnet.layer4   # 2048 ch

            # channel count at the chosen insertion stage
            stage_ch = {'stem': 64, 'layer1': 256, 'layer2': 512}
            assert afd_stage in stage_ch, f"afd_stage must be one of {list(stage_ch)}"
            self.router_channels = stage_ch[afd_stage]

            if self.afd_router:
                self.router = FrequencyReliabilityRouter(
                    self.router_channels, low_r=low_r, mid_r=mid_r,
                    cond_on_view=router_cond_view)
            if self.afd_cvfc:
                self.cvfc = CrossViewFrequencyCounterfactual(
                    low_r=low_r, mid_r=mid_r, high_drop_p=high_drop_p)

            # pooling
            self.pool = GeMPool() if pool == 'gem' else None  # None -> avg in forward
        elif backbone == 'swin_small':
            # SOLIDER Swin-Small (team asset, SOTA push).  AFD frequency modules
            # insert at resnet shallow stages (stem/layer1/layer2) that do NOT
            # exist in Swin -> AFD is unsupported here (OVLI is the headline and
            # needs no AFD).  Enforce so a stray --use_afd cannot silently no-op.
            assert not use_afd, ("backbone='swin_small' does not support the AFD "
                                 "frequency modules (router/cvfc insert at resnet "
                                 "shallow stages). Run swin with --use_afd off "
                                 "(OVP/OVLI are independent of AFD).")
            self.backbone_swin = SwinBackboneReID(
                img_size=tuple(img_size), pretrain_path=swin_pretrain,
                semantic_weight=swin_semantic_weight,
                iso_branch=self.airl_dualbranch_iso,
                iso_stage=self.airl_iso_stage,
                iso_trunk_recce=self.airl_iso_trunk_recce)
            self.in_planes = self.backbone_swin.out_dim   # 768
            # OVLI hooks model.layer4 -> point it at the Swin Identity hook module
            # so the hook captures the (B,768,H,W) last-stage map.
            self.layer4 = self.backbone_swin.layer4
            # Swin's last map is LayerNorm'd (signed, ~half negative); GeM's
            # clamp(min=eps) would destroy the negative half -> force avg pooling
            # (== SOLIDER's native avgpool head over the same map).
            self.pool = None
        else:
            raise ValueError(f"unknown backbone '{backbone}' "
                             f"(expected 'resnet50' or 'swin_small')")

        # BNNeck (f_full -- the original head: full-resolution identity evidence)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        # AIRL dual-branch: a SECOND independent BNNeck head (f_rec).  Same structure
        # / init recipe as f_full (frozen-bias BNNeck + bias-free classifier), but its
        # OWN parameters so the two heads can specialise (f_rec absorbs the
        # degradation-consistency signal, f_full stays clean).
        #   * airl_dualbranch     : f_rec pools the SHARED global_feat (fully shared
        #                           trunk -> the gradient that collapsed the heads).
        #   * airl_dualbranch_iso : f_rec pools the INDEPENDENT rec last-stage map
        #                           (gradient-isolated trunk -> heads re-diverge).
        # Only built when one of the two is on -> the OFF model is structurally
        # identical to the single-head baseline (no extra params).
        if self.airl_dualbranch or self.airl_dualbranch_iso:
            self.bottleneck_rec = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_rec.bias.requires_grad_(False)
            self.bottleneck_rec.apply(weights_init_kaiming)
            self.classifier_rec = nn.Linear(self.in_planes, num_classes, bias=False)
            self.classifier_rec.apply(weights_init_classifier)

    # --- backbone forward with optional router insertion ------------------- #
    def _forward_backbone(self, x, view_idx=None, feat_override=None,
                          insert_router=False):
        """Run stem->layer4. If insert_router, apply router at self.afd_stage.

        feat_override: if given, a dict {stage: tensor} used to *replace* the
        feature at that stage (for counterfactual passes that re-enter mid-network).
        """
        band_w = None
        if self.backbone == 'swin_small':
            # Swin wrapper runs the full backbone and routes the last spatial map
            # through its Identity hook point (so the OVLI layer4 hook fires).
            # No AFD router/cvfc for swin -> band_w stays None.
            feat_map = self.backbone_swin(x)
            return feat_map, band_w
        x = self.stem(x)
        if self.afd_stage == 'stem':
            x = self._maybe_route(x, 'stem', view_idx, insert_router)
            x, band_w = x if isinstance(x, tuple) else (x, band_w)

        x = self.layer1(x)
        if self.afd_stage == 'layer1':
            x = self._maybe_route(x, 'layer1', view_idx, insert_router)
            x, band_w = x if isinstance(x, tuple) else (x, band_w)

        x = self.layer2(x)
        if self.afd_stage == 'layer2':
            x = self._maybe_route(x, 'layer2', view_idx, insert_router)
            x, band_w = x if isinstance(x, tuple) else (x, band_w)

        x = self.layer3(x)
        x = self.layer4(x)
        return x, band_w

    def _maybe_route(self, x, stage, view_idx, insert_router):
        if insert_router and self.afd_router and stage == self.afd_stage:
            return self.router(x, view_idx)   # returns (feat, w)
        return x

    def _pool(self, x):
        if self.pool is not None:
            return self.pool(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def _embed(self, x):
        """global feat -> BNNeck feat."""
        g = self._pool(x)
        bn = self.bottleneck(g)
        return g, bn

    def _embed_rec(self, x):
        """rec map -> pooled rec feat -> BNNeck_rec feat (independent f_rec head).

        Used only by the gradient-isolated dual-branch: the rec map already comes
        from a detached trunk + independent late stage, so pooling + bottleneck_rec
        here keeps the whole f_rec head isolated from the shared trunk.
        """
        g = self._pool(x)
        bn = self.bottleneck_rec(g)
        return g, bn

    # --- public forward ---------------------------------------------------- #
    def forward(self, x, view_idx=None, return_cvfc=False, return_dual=False,
                rec_only=False):
        """
        Train: returns dict with global_feat, bn_feat, logits, band_w,
               and (if return_cvfc & afd_cvfc) counterfactual embeddings.
               When airl_dualbranch is on, the dict ALSO carries the f_rec head's
               'bn_feat_rec' / 'logits_rec' (computed from the SAME pooled
               global_feat through the second BNNeck) so the train loop can add
               the f_rec ID-CE + degradation-consistency.
        Eval : returns the L2-normalized f_full BN feature (single head); when
               airl_dualbranch is on AND return_dual=True, returns the tuple
               (f_full_norm, f_rec_norm) so the dual-branch eval can SOFT-fuse
               the two cosine scores.  return_dual defaults to False, so the
               legacy single-feature eval path (extract_features) is unchanged.

               airl_dualbranch_iso: identical (f_full_norm, f_rec_norm) eval tuple
               and bn_feat_rec/logits_rec train keys, but f_rec is pooled from the
               INDEPENDENT rec last-stage map (gradient-isolated trunk) instead of
               the shared global_feat.  return_rec on the Swin backbone yields BOTH
               maps in ONE forward (split path).
        """
        # ---- AIRL gradient-isolated dual-branch path -------------------------- #
        # When iso is on we need BOTH the f_full map and the independent rec map.
        # The Swin split forward returns both in one pass; f_full pools the shared
        # map (bn_feat/global_feat) and f_rec pools the rec map through bottleneck_rec.
        # Fork-feed gradient regime (see _forward_swin_split): the DEGRADED pass
        # (rec_only) always detaches the fork so the consistency gradient never
        # reaches the trunk; the CLEAN pass detaches only when iso_trunk_recce is
        # False -- with the fix (True) the clean f_rec ID-CE reflows into the trunk
        # (extra identity supervision), while the consistency stays trunk-isolated.
        # `or rec_only` so the rec-only degraded contract is honoured even if a
        # caller invokes it under model.eval() (training=False, return_dual=False) --
        # otherwise want_iso would be False and the rec_only dict request would
        # silently fall through to the single f_full eval tensor.
        want_iso = self.airl_dualbranch_iso and (self.training or return_dual
                                                 or rec_only)
        if want_iso:
            full_map, rec_map = self.backbone_swin(
                x, return_rec=True, rec_only=rec_only)
            # rec_only (the degraded consistency pass): compute ONLY the f_rec head.
            # f_full's BNNeck is NOT run on the degraded images, so self.bottleneck's
            # running mean/var stay CLEAN (no degraded-ground stat leak into the
            # f_full eval head) -- f_full is a true clean expert -- and the f_full
            # pool+BN+classifier compute is skipped (cheaper).  The clean forward
            # (rec_only=False) still produces both heads as usual.
            if rec_only:
                _grec, bn_feat_rec = self._embed_rec(rec_map)
                return {
                    'bn_feat_rec': bn_feat_rec,
                    'logits_rec': self.classifier_rec(bn_feat_rec),
                }
            global_feat, bn_feat = self._embed(full_map)
            _grec, bn_feat_rec = self._embed_rec(rec_map)
            if not self.training:
                # eval: ONE forward -> two L2-normalized features (f_full, f_rec).
                return (F.normalize(bn_feat, dim=1),
                        F.normalize(bn_feat_rec, dim=1))
            out = {
                'global_feat': global_feat,   # f_full triplet (shared trunk)
                'bn_feat': bn_feat,
                'logits': self.classifier(bn_feat),
                'band_w': None,               # swin has no AFD band weights
                # f_rec head over the INDEPENDENT rec map (own ID-CE + AIRL
                # consistency in the train loop); pooled rec feat exposed too so the
                # smoke can confirm it is NOT the shared global_feat.
                'global_feat_rec': _grec,
                'bn_feat_rec': bn_feat_rec,
                'logits_rec': self.classifier_rec(bn_feat_rec),
            }
            # AFD CVFC is swin-incompatible (asserted off), so no cf block here.
            return out

        feat_map, band_w = self._forward_backbone(
            x, view_idx=view_idx, insert_router=self.afd_router)
        global_feat, bn_feat = self._embed(feat_map)

        if not self.training:
            f_full = F.normalize(bn_feat, dim=1)
            if return_dual and self.airl_dualbranch:
                # second head shares the SAME pooled global_feat -> ONE forward,
                # two L2-normalized features for the distmat-level soft fusion.
                bn_feat_rec = self.bottleneck_rec(global_feat)
                return f_full, F.normalize(bn_feat_rec, dim=1)
            return f_full

        out = {
            'global_feat': global_feat,   # for triplet (before BN)
            'bn_feat': bn_feat,           # BN feature
            'logits': self.classifier(bn_feat),
            'band_w': band_w,             # (B,3) or None
        }

        if self.airl_dualbranch:
            # f_rec head: its OWN BNNeck + classifier on the shared global_feat.
            # The train loop applies f_rec ID-CE (so f_rec is a valid identity
            # space) PLUS the AIRL degradation-consistency (so it is robust to a
            # low pixel budget).  global_feat is shared, so the global triplet is
            # NOT duplicated for f_rec (single backbone-level triplet).
            bn_feat_rec = self.bottleneck_rec(global_feat)
            out['bn_feat_rec'] = bn_feat_rec
            out['logits_rec'] = self.classifier_rec(bn_feat_rec)

        if return_cvfc and self.afd_cvfc:
            # build counterfactual at the SHALLOW stage, then run remainder.
            # We re-extract the shallow feature, apply CVFC, and finish forward.
            cf = self._forward_counterfactual(x, view_idx)
            out.update(cf)
        return out

    def _forward_counterfactual(self, x, view_idx):
        """Run two cheap counterfactual passes (high-drop, low-pass) from the
        shallow stage onward, returning their BN embeddings for consistency loss."""
        # get shallow feature up to (and including) the insertion stage
        stage = self.afd_stage
        shallow = self.stem(x)
        if stage in ('layer1', 'layer2'):
            shallow = self.layer1(shallow)
        if stage == 'layer2':
            shallow = self.layer2(shallow)

        def finish(feat):
            # continue from the stage AFTER the insertion point
            if stage == 'stem':
                h = self.layer1(feat); h = self.layer2(h)
            elif stage == 'layer1':
                h = self.layer2(feat)
            else:  # layer2
                h = feat
            h = self.layer3(h); h = self.layer4(h)
            _, bn = self._embed(h)
            return bn

        hd = self.cvfc.high_band_dropout(shallow)
        lp = self.cvfc.low_pass(shallow)
        return {
            'cf_highdrop_bn': finish(hd),
            'cf_lowpass_bn': finish(lp),
        }


def build_model(num_classes, args):
    """Factory from an argparse-like namespace.

    backbone defaults to 'resnet50' -> the existing BoT baseline is reproduced
    byte-for-byte (no new arg required of legacy callers).  backbone='swin_small'
    builds the SOLIDER Swin-Small backbone instead (img_size / swin_pretrain /
    swin_semantic_weight are read from args, with the SOLIDER ReID defaults).
    """
    return AFDModel(
        num_classes=num_classes,
        last_stride=getattr(args, 'last_stride', 1),
        pool=getattr(args, 'pool', 'gem'),
        pretrained=True,
        use_afd=getattr(args, 'use_afd', False),
        afd_router=getattr(args, 'afd_router', True),
        afd_cvfc=getattr(args, 'afd_cvfc', True),
        afd_stage=getattr(args, 'afd_stage', 'layer1'),
        router_cond_view=getattr(args, 'router_cond_view', True),
        low_r=getattr(args, 'low_r', 0.125),
        mid_r=getattr(args, 'mid_r', 0.30),
        high_drop_p=getattr(args, 'high_drop_p', 0.5),
        backbone=getattr(args, 'backbone', 'resnet50'),
        swin_pretrain=getattr(args, 'swin_pretrain', ''),
        swin_semantic_weight=getattr(args, 'swin_semantic_weight', 0.2),
        img_size=tuple(getattr(args, 'img_size', (256, 128))),
        airl_dualbranch=getattr(args, 'airl_dualbranch', False),
        airl_dualbranch_iso=getattr(args, 'airl_dualbranch_iso', False),
        airl_iso_stage=getattr(args, 'airl_iso_stage', 3),
        airl_iso_trunk_recce=getattr(args, 'airl_iso_trunk_recce', True),
    )
