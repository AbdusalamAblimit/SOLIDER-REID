# encoding: utf-8
"""
CVPB training loop on CARGO = the afd_reid baseline trainer + OVP-Mem / OVLI.

This reuses the dataset / model / eval from ../afd_reid unchanged.  Two optional
cross-view auxiliary mechanisms are bolted on (both default OFF -> the BoT
baseline is reproduced exactly).  They are NOT mutually exclusive: --ovp,
--ovli, and --ovp --ovli (both) are all valid; in `both` mode each loss keeps
its own warmup / lambda / diagnostics and the two terms are simply summed into
the same total (complementarity test: does prototype-memory + late-interaction
beat OVP-only?).

  --ovp   : OVP-Mem (Opposite-View Prototype Memory).  Per-pid per-view EMA
            prototype bank + opposite-view InfoNCE.  Known prior-art overlap
            (CMPC / MBCE / PDPA) -> kept only as an empirical auxiliary.

  --ovli  : OVLI (Opposite-View Late-Interaction Evidence Alignment).  The
            headline mechanism.  NO prototype / memory / EMA.  Instead a pure
            sample-to-sample, in-batch, opposite-view *retrieval* loss whose
            score is a hybrid of (a) global cosine and (b) a symmetric
            token-set late-interaction (ColBERT/MaxSim-style **partial**
            matching).  Framing: cross-view identity evidence is a partial
            token-set matching problem, not a global prototype alignment one --
            aerial<->ground has no 1-1 part correspondence, so a global
            prototype penalizes missing regions whereas partial MaxSim lets the
            tokens that *can* be matched carry the similarity.

  --acvp  : ACVP (Ambiguity-Calibrated opposite-View negative relaxation).  An
            OVLI calibration (requires --ovli).  Treats the opposite-view identity
            prototype ONLY as a DETACHED ambiguity SENSOR: it softens the
            unreliable NEGATIVES in the OVLI cross-view contrastive denominator and
            does NOT do any prototype-positive alignment (so it stays clear of the
            OVP / CMPC / PDPA prototype-contrast prior art).  Mechanism: maintain a
            detached per-pid per-view EMA prototype bank (its own OVPMemory, read
            detached); for an anchor i and an opposite-view negative j (different
            pid) measure how close j's opposite-view identity sits to i,
                delta_ij = cos(z_i, P[y_j, view_j]) - cos(z_i, P[y_i, view_j]),
            map it to a weight w_ij = clamp(1 - gamma*sigmoid((delta_ij-margin)/
            eta), w_min, 1) and ADD log(w_ij) to that negative's logit in the
            DENOMINATOR only (positives untouched).  No learnable params, no
            gradient to the encoder/proto (pure detached re-weighting).  Default
            OFF => the OVLI loss is reproduced byte-for-byte.  Per-epoch
            kill-switch log: relaxed_neg_frac (w<0.95 share) + mean_w (stop if
            frac>0.30 or mean_w<0.75 => negatives broadly weakened = bad).

OVLI details (the load-bearing design)
--------------------------------------
* Tokens: hook model.layer4 (the GeM-input spatial map, 16x8 for 256x128),
  adaptive-avg-pool to a KxK' grid, flatten to K local tokens, then a NEW
  learnable 1x1-conv/linear projection to ovli_dim (256) + per-token L2-norm.
  ** The projection is a new learnable parameter set and IS added to the
     optimizer ** (this is the key structural difference vs OVP, which adds no
     params).  The hook does NOT detach -> gradient flows layer4 -> proj.

* Opposite-view retrieval loss (supervised-contrastive, logsumexp):
  within the batch, for each anchor i in view v, the positives are the same-pid
  samples in the OPPOSITE view (1-v) and the negatives are the opposite-view
  samples of OTHER pids.  Same-view samples are excluded as candidates entirely
  (this is a *cross-view* objective).  Pairwise score:
      score(i,j) = alpha * cos(g_i, g_j)
                 + (1 - alpha) * sym_MaxSim(tok_i, tok_j)
      sym_MaxSim = 0.5 * ( pool_u max_s <u,s> + pool_s max_u <u,s> )   # bidir
  where pool_* is the --ovli_pool dustbin variant over the per-token max scores:
      mean     : average over ALL token-max scores (original; NOT a true dustbin
                 -- low-score non-corresponding tokens still drag the pair down).
      topk     : average of the top-k highest token-max scores (--ovli_topk),
                 i.e. drop the K-k worst-matching tokens -> sparse evidence /
                 dustbin approximation; the headline AG-ReID design.
      thresh   : average of token-max scores above theta (--ovli_thresh), with a
                 single-max fallback so a fully-masked pair never NaNs.
      softtopk : softmax(token-max / tau)-weighted mean (smooth, differentiable
                 top-k surrogate).
  Both MaxSim directions use the same pooling, so sym_MaxSim stays symmetric;
  the eval rerank (--ovli_rerank) reuses the identical pooling (train/test
  symmetry).  --ovli_pool mean reproduces the previous behaviour exactly.
  Multi-positive InfoNCE per anchor:
      L_i = -logsumexp(score(i,pos)/tau) + logsumexp(score(i,cand)/tau)
  averaged over anchors that have >=1 opposite-view positive AND >=1
  opposite-view negative in the batch.  No memory / EMA / prototype.

* lambda warmup (--ovli_warmup, default 10): the H1 lesson from OVP -- linearly
  ramp lambda over the first N epochs so the (randomly-initialised) projection
  cannot inject a sharp early gradient.  Per-epoch log records
  OVLI[lam_eff pos_score neg_score gap] for collapse / over-strong monitoring.

* eval: OVLI is a TRAIN-time loss only; default eval is global-only (unchanged,
  identical to the baseline).  --ovli_rerank additionally reports a
  global + sym_MaxSim rerank at eval time (both numbers printed), so train/test
  stay symmetric and the rerank is opt-in.

Baseline (no flags):
    resnet50(IMAGENET1K_V1) + GeM + BNNeck
    loss = CE(label-smooth 0.1) + batch-hard triplet (margin 0.3)
    PK sampler P=16 x K=4 (bs=64), AdamW lr 3.5e-4, 10-ep warmup + cosine, 60 ep.
    eval every 10 ep: A->G and G->A cross-view mAP / R1 / mINP.

--ovp (OVP-Mem):
    Maintain, per train pid, two EMA prototypes (aerial / ground) of the
    L2-normalized BNNeck feature in a register_buffer of shape
    [num_pid, 2, feat_dim]; EMA momentum 0.2.  Each step:
      1) update the prototypes of the pids/views present in the batch (EMA),
      2) add an InfoNCE loss pulling each sample toward its OWN pid's
         OPPOSITE-view prototype and away from all other pids' opposite-view
         prototypes:  CE( cos(z, P[:, opp_view]) / tau ,  y ).
    total = CE + triplet + lambda_ovp * OVP.   (batch size unchanged, bs=64.)

    Cold-start handling: a sample only contributes to the OVP loss once its own
    opposite-view prototype has been initialized (seen >=1 time); candidate
    columns that are still uninitialized are masked out of the InfoNCE logits so
    a zero prototype can never act as an easy negative/positive.

Run on lab-3090:
    cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
    # OVP-Mem (empirical auxiliary):
    PYTHONUNBUFFERED=1 python3 afd_train.py \
        --data_root /root/work/SOLIDER-REID/data \
        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovp \
        --ovp \
        2>&1 | tee /tmp/cvpb_ovp.log
    # OVLI (headline; late-interaction opposite-view retrieval):
    PYTHONUNBUFFERED=1 python3 afd_train.py \
        --data_root /root/work/SOLIDER-REID/data \
        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovli \
        --ovli --ovli_rerank \
        2>&1 | tee /tmp/cvpb_ovli.log
    # both (complementarity test: OVP prototype + OVLI late-interaction):
    PYTHONUNBUFFERED=1 python3 afd_train.py \
        --data_root /root/work/SOLIDER-REID/data \
        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovp_ovli \
        --ovp --ovli --ovli_rerank \
        2>&1 | tee /tmp/cvpb_ovp_ovli.log
    # baseline reproduction: drop all of --ovp / --ovli
"""
import os
import sys
import time
import math
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# reuse afd_reid building blocks unchanged
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'afd_reid'))
from cargo_dataset import (CARGO, CARGOImageDataset, build_transforms,  # noqa: E402
                           RandomIdentitySampler, filter_by_view)
from agreid_dataset import AGReIDv2  # noqa: E402  -- second benchmark (--dataset agreid)
from agreid_v2_combined import AGReIDV2Combined  # noqa: E402  -- official exp1(A->G)+exp4(G->A) (--dataset agreid_v2)
from afd_model import build_model, weights_init_kaiming  # noqa: E402
# reuse the exact loss / schedule / eval helpers so nothing diverges
from afd_train import (CrossEntropyLabelSmooth, TripletLoss, WarmupCosineLR,  # noqa: E402
                       run_cross_view_eval, print_eval, set_seed)


# --------------------------------------------------------------------------- #
# OVP-Mem: opposite-view prototype memory
# --------------------------------------------------------------------------- #
class OVPMemory(nn.Module):
    """Per-pid, per-view EMA prototype bank + opposite-view InfoNCE loss.

    bank: register_buffer [num_pid, 2, feat_dim]  (view 0 = Aerial, 1 = Ground)
    init: register_buffer [num_pid, 2] uint8 = has this (pid,view) been seen.

    Prototypes are L2-normalized; features used for update/loss are the
    L2-normalized BNNeck features (detached for the EMA update).
    """

    def __init__(self, num_pid, feat_dim, momentum=0.2, tau=0.05):
        super().__init__()
        self.num_pid = num_pid
        self.feat_dim = feat_dim
        self.momentum = momentum          # weight on the *new* batch mean
        self.tau = tau
        self.register_buffer('bank', torch.zeros(num_pid, 2, feat_dim))
        self.register_buffer('inited', torch.zeros(num_pid, 2, dtype=torch.uint8))

    @torch.no_grad()
    def update(self, feats, labels, views):
        """EMA-update prototypes for the (pid,view) groups present in the batch.

        feats:(B,D) L2-normed BN feats (detached). labels:(B,) long.
        views:(B,) long in {0,1}.  Per group, EMA toward the group mean, then
        re-normalize the prototype.
        """
        feats = feats.detach()
        for v in (0, 1):
            vmask = (views == v)
            if not torch.any(vmask):
                continue
            fv = feats[vmask]
            lv = labels[vmask]
            uniq = torch.unique(lv)
            for pid in uniq:
                pid_i = int(pid)
                gmean = fv[lv == pid].mean(dim=0)
                gmean = F.normalize(gmean, dim=0)
                if self.inited[pid_i, v] == 0:
                    self.bank[pid_i, v] = gmean
                    self.inited[pid_i, v] = 1
                else:
                    self.bank[pid_i, v] = ((1 - self.momentum) * self.bank[pid_i, v]
                                           + self.momentum * gmean)
                    self.bank[pid_i, v] = F.normalize(self.bank[pid_i, v], dim=0)

    def loss(self, feats, labels, views):
        """InfoNCE: each sample classified against ALL pids' opposite-view
        prototypes, target = its own pid.  feats:(B,D) L2-normed BN feats
        (NOT detached -> gradient flows to the encoder).

        Returns a scalar loss (0 if no sample has an initialized own-opp proto).
        """
        device = feats.device
        opp = 1 - views                                   # opposite view per sample
        total = feats.new_zeros(())
        count = 0
        for v_opp in (0, 1):
            # samples whose opposite view == v_opp (i.e. samples from view 1-v_opp)
            smask = (opp == v_opp)
            if not torch.any(smask):
                continue
            protos = self.bank[:, v_opp, :]               # (num_pid, D)
            valid = (self.inited[:, v_opp] == 1)          # (num_pid,) candidate cols
            if valid.sum() == 0:
                continue
            z = feats[smask]                              # (b,D)
            y = labels[smask]                             # (b,)
            # only keep samples whose OWN opposite-view proto exists
            own_ok = valid[y]
            if own_ok.sum() == 0:
                continue
            z = z[own_ok]
            y = y[own_ok]
            logits = (z @ protos.t()) / self.tau          # (b, num_pid)
            # mask out uninitialized candidate columns
            logits = logits.masked_fill(~valid.view(1, -1), float('-inf'))
            total = total + F.cross_entropy(logits, y)
            count += 1
        if count == 0:
            return feats.new_zeros(())
        return total / count


# --------------------------------------------------------------------------- #
# OVLI set pooling: learnable permutation-invariant aggregation of K tokens
# --------------------------------------------------------------------------- #
class OVLISetPool(nn.Module):
    """Permutation-invariant pooling of K projected tokens -> 1 vector.

    The headline OVLI score with --ovli_match avg --ovli_pool mean reduces
    EXACTLY to <mean_k tok_i, mean_k tok_j> -- the gram of the per-sample MEAN of
    the K L2-normed tokens -- and that mean-pool aggregation is the current best
    single mechanism (52.37 cross-view mAP).

    KEY DESIGN -- "mean + zero-init residual" (residual=True, default).  A
    standalone *random-init* learnable pooling starts from noise (NetVLAD ep20
    mAP collapsed to 14.66 << 52.37 << even pure global 45.14), because the
    random aggregated vector is meaningless and drags the cross-view cosine down.
    To start LOSSLESS from the 52.37 mean-pool and only *learn a correction*, the
    forward is

        pooled = mean_k(tok)  +  g * residual_module(tok)

    where `g` is a scalar gate `nn.Parameter(zeros(1))` (one per module).  At
    init g == 0, so `pooled == mean_k(tok)` BYTE-FOR-BYTE (== the existing
    --ovli_match avg --ovli_pool mean 52.37 path); the gradient then opens the
    residual.  `mean_k(tok)` is the UN-normalized mean of the K L2-normed tokens
    (identical to what `tok.mean(dim=1)` feeds the avg/mean gram), so the residual
    baseline matches the current best mechanism exactly.  residual=True REQUIRES
    out_dim == dim (the residual is added to the D-dim mean); enforced by assert.

    residual=False reverts to the ORIGINAL standalone pooling (random init -> the
    aggregated vector fully REPLACES the mean) -- kept only so an ablation can
    contrast standalone-vs-residual; it is the collapsing config and NOT default.

    Each `*_residual()` returns a per-sample (B,out_dim) residual that is itself
    permutation-invariant (the K axis is collapsed by a SUM / softmax-weighted-SUM
    / mean / covariance), so `pooled` is permutation-invariant in the tokens
    (verified to <1e-5 in smoke_ovli_setpool.py).  All parameters live in THIS
    module -> list(ovli.parameters()) collects them -> they land in the optimizer
    (the same contract as OVLIHead.proj).  Always runs in fp32 (called under
    autocast(enabled=False)); every reduction is NaN-safe.

      netvlad     : C learnable centers + per-token soft-assignment; residual
                    aggregation -> intra-normalized (C x D) VLAD -> linear to
                    out_dim.  (Arandjelovic et al., NetVLAD.)
      attn        : H learnable query vectors; multi-head attention pooling over
                    the K tokens (softmax weights over k) -> linear.  (Set
                    Transformer PMA / learned-query attention pooling.)
      gated       : a small per-token scalar reliability gate (sigmoid); gated
                    convex average over the K tokens (lightweight token-wise
                    reliability weighting).
      secondorder : low-rank covariance pooling -- reduce tokens to r dims, take
                    the order-invariant token covariance (z^T z / K), signed-sqrt
                    normalize, flatten -> linear (token second-order statistics).
    """

    def __init__(self, mode, dim, out_dim=None, clusters=8, heads=4,
                 lowrank=32, gate_hidden=None, residual=True, eps=1e-6):
        super().__init__()
        self.mode = str(mode)
        self.dim = int(dim)
        self.out_dim = int(out_dim) if out_dim else int(dim)
        self.residual = bool(residual)
        self.eps = float(eps)
        # "mean + zero-init residual" needs the residual to be ADDED to the D-dim
        # mean -> the residual must map to out_dim == dim.  (The configured call
        # site always passes out_dim == proj_dim == dim, so this never trips in
        # practice; the assert just makes the contract explicit.)
        if self.residual:
            assert self.out_dim == self.dim, (
                f"--ovli_setpool_residual True needs out_dim==dim "
                f"(mean+residual added in D-space), got dim={self.dim} "
                f"out_dim={self.out_dim}")
        D = self.dim
        if self.mode == 'netvlad':
            C = int(clusters)
            self.clusters = C
            self.assign = nn.Linear(D, C)                  # per-token soft-assign
            self.centers = nn.Parameter(torch.randn(C, D) * 0.01)
            self.out = nn.Linear(C * D, self.out_dim)
        elif self.mode == 'attn':
            H = int(heads)
            assert D % H == 0, f"--ovli_attn_heads {H} must divide --ovli_dim {D}"
            self.heads = H
            self.head_dim = D // H
            self.query = nn.Parameter(torch.randn(H, self.head_dim) * 0.02)
            self.out = nn.Linear(D, self.out_dim)
        elif self.mode == 'gated':
            h = int(gate_hidden) if gate_hidden else max(8, D // 4)
            self.gate = nn.Sequential(nn.Linear(D, h), nn.GELU(), nn.Linear(h, 1))
            # identity when out_dim == D so the gate stays truly lightweight
            self.out = nn.Linear(D, self.out_dim) if self.out_dim != D else nn.Identity()
        elif self.mode == 'secondorder':
            r = int(lowrank)
            self.lowrank = r
            self.reduce = nn.Linear(D, r, bias=False)
            self.out = nn.Linear(r * r, self.out_dim)
        else:
            raise ValueError(f"unknown --ovli_setpool {self.mode}")
        # kaiming init for the Linear layers (matches OVLIHead.proj convention);
        # the raw Parameters (centers / query) keep their small-random init above.
        self.apply(weights_init_kaiming)
        # Zero-init scalar gate on the residual.  At init `g == 0` -> the residual
        # contributes exactly 0 -> `forward == mean_k(tok)` byte-for-byte (== the
        # 52.37 avg/mean-pool path), REGARDLESS of how `self.out` was initialized.
        # Step-0 gradients: d pooled / d g = residual(tok) (generally NON-zero, so
        # the gate moves off 0); d pooled / d theta = g * d residual / d theta = 0
        # because g == 0.  So at step 0 ONLY the gate receives gradient -- the
        # residual module's own params (centers / query / Linear / gate MLP) have
        # ZERO gradient and do NOT update; they only start training from step 1
        # onward, once g != 0 makes their gradient non-zero.  A scalar gate is used
        # (not relying on zero-init `self.out`) because it makes the "lossless
        # start" provable to bytes and independent of each branch's internal init.
        self.gate_res = nn.Parameter(torch.zeros(1)) if self.residual else None

    # -- per-mode permutation-invariant residual (the learnable correction) ---- #
    def _residual(self, tok):
        """tok:(B,K,D) L2-normed fp32 -> (B,out_dim) residual, permutation-invariant
        in the K (token) axis (every branch collapses K by an order-invariant op).
        """
        B, K, D = tok.shape
        if self.mode == 'netvlad':
            a = torch.softmax(self.assign(tok), dim=2)            # (B,K,C) over C
            res = tok.unsqueeze(2) - self.centers.view(1, 1, -1, D)  # (B,K,C,D)
            vlad = (a.unsqueeze(3) * res).sum(dim=1)              # (B,C,D) sum over K
            vlad = F.normalize(vlad, dim=2)                       # intra-norm /cluster
            vlad = vlad.reshape(B, -1)                            # (B, C*D)
            vlad = F.normalize(vlad, dim=1)                       # global VLAD L2-norm
            return self.out(vlad)                                 # (B, out_dim)
        if self.mode == 'attn':
            H, Dh = self.heads, self.head_dim
            t = tok.reshape(B, K, H, Dh)                          # (B,K,H,Dh)
            scores = (t * self.query.view(1, 1, H, Dh)).sum(-1) / math.sqrt(Dh)
            a = torch.softmax(scores, dim=1)                      # (B,K,H) over K
            out = (a.unsqueeze(-1) * t).sum(dim=1)                # (B,H,Dh) sum over K
            return self.out(out.reshape(B, H * Dh))               # (B, out_dim)
        if self.mode == 'gated':
            g = torch.sigmoid(self.gate(tok))                     # (B,K,1) in (0,1)
            num = (g * tok).sum(dim=1)                            # (B,D) sum over K
            den = g.sum(dim=1).clamp(min=self.eps)                # (B,1) >0 -> NaN-safe
            return self.out(num / den)                            # gated convex mean
        # secondorder: order-invariant low-rank token covariance
        z = self.reduce(tok)                                      # (B,K,r)
        cov = torch.einsum('bkr,bks->brs', z, z) / K             # (B,r,r) sum over K
        cov = torch.sign(cov) * torch.sqrt(cov.abs() + self.eps)  # signed-sqrt, NaN-safe
        return self.out(cov.reshape(B, -1))                       # (B, r*r) -> out_dim

    def forward(self, tok):
        """tok:(B,K,D) L2-normed fp32 -> (B,out_dim) raw aggregated vector.

        residual=True (default): `mean_k(tok) + gate_res * residual(tok)` with the
        gate zero-init, so at step 0 the output == the un-normalized mean of the K
        tokens.  aggregate_tokens then returns this raw vector AS-IS (no final
        L2-norm), so its gram is the UN-normalized `mean @ mean.T` == the 52.37
        avg/mean-pool path BYTE-FOR-BYTE.  residual=False: the original standalone
        pooling REPLACES the mean (random-init), and aggregate_tokens L2-normalizes
        it (cosine gram) -> the collapsing control.  Both branches are
        permutation-invariant in K.
        """
        if not self.residual:
            return self._residual(tok)                            # standalone (control)
        # mean + zero-init residual: lossless start from the 52.37 mean-pool.
        m = tok.mean(dim=1)                                        # (B,D) == avg/mean path
        return m + self.gate_res * self._residual(tok)            # (B,out_dim==D)


# --------------------------------------------------------------------------- #
# OVLI: Opposite-View Late-Interaction Evidence Alignment
# --------------------------------------------------------------------------- #
class OVLIHead(nn.Module):
    """Token projection + opposite-view late-interaction retrieval loss.

    Reuses the maxsim_probe token-extraction recipe (hook model.layer4 -> the
    GeM-input spatial map -> adaptive-avg-pool to a grid -> flatten to K local
    tokens), but adds a NEW learnable 1x1-conv/linear projection to `proj_dim`
    + per-token L2-norm.  ** The projection parameters are owned by this module
    and MUST be added to the optimizer ** (this is the structural difference vs
    OVP-Mem, which has no learnable params).

    There is NO prototype / memory / EMA: the loss is a pure sample-to-sample,
    in-batch, opposite-view supervised-contrastive retrieval objective.

    Forward never holds a backbone reference; the layer4 map is captured by a
    forward hook that does NOT detach, so gradient flows layer4 -> proj.
    """

    def __init__(self, model, in_ch=2048, proj_dim=256, grid=(8, 4),
                 alpha=0.5, tau=0.05, pool='mean', topk=8, thresh=0.0,
                 allview=False, match='maxsim', align='free',
                 setpool='mean', vlad_clusters=8, attn_heads=4, so_rank=32,
                 setpool_residual=True):
        super().__init__()
        self.grid = tuple(grid)               # adaptive pool grid (gh, gw)
        self.alpha = float(alpha)             # weight on global cosine in score
        self.tau = float(tau)
        # MaxSim pooling variant over the per-token max scores (dustbin family).
        #   'mean'    : average over ALL token-max scores  (original behaviour).
        #   'topk'    : average only the top-k highest token-max scores (drop the
        #               low-score, non-corresponding tokens -> dustbin approx).
        #   'thresh'  : average only token-max scores above `thresh` (soft floor
        #               fallback if none pass, so a fully-masked pair never NaNs).
        #   'softtopk': logsumexp-softmax weighted mean over token-max scores
        #               (smooth top-k, temperature reuses self.tau).
        self.pool = str(pool)
        self.topk = int(topk)
        self.thresh = float(thresh)
        assert self.pool in ('mean', 'topk', 'thresh', 'softtopk'), \
            f"unknown --ovli_pool {self.pool}"
        # Ablation: candidate view-masking mode for the contrastive loss.
        #   allview=False (default): OPPOSITE-VIEW-ONLY -- per anchor the
        #     candidate set is restricted to opposite-view samples (the headline
        #     cross-view constraint). Reproduces the original behaviour exactly.
        #   allview=True (--ovli_allview): drop the opposite-view constraint --
        #     candidates = ALL other samples (incl. same view); positives =
        #     same-pid any view (excl. self), negatives = other-pid any view.
        #     A plain all-view token-set supervised-contrastive control isolating
        #     "is the cross-view restriction what helps, or just an extra token
        #     loss?". Only the view mask changes; score/MaxSim/pool/tau/proj are
        #     identical across modes.
        self.allview = bool(allview)
        # Ablation: token-token match reduction (the late-interaction selection).
        #   match='maxsim' (default): for each query token take the MAX similarity
        #     over the other token set (ColBERT/MaxSim late-interaction selection).
        #     Reproduces the original behaviour exactly.
        #   match='avg' (--ovli_match avg): replace that per-token MAX with a MEAN
        #     over the other token set -> the token-token similarities are fully
        #     averaged, degenerating to a near-global soft match. Isolates "is the
        #     max selection what makes late-interaction work, vs a soft average?".
        #     ONLY the inner token reduction changes; the bidirectional structure,
        #     the outer pool over query tokens, alpha mixing and the loss are
        #     identical across modes.
        self.match = str(match)
        assert self.match in ('maxsim', 'avg'), \
            f"unknown --ovli_match {self.match}"
        # Ablation: spatial alignment of the late interaction.
        #   align='free' (default): every query token may match ANY token in the
        #     other set (free/global late interaction). Reproduces the original.
        #   align='ordered' (--ovli_align ordered): AlignedReID-style row-ordered
        #     alignment -- the K tokens form a (gh x gw) spatial grid; a query
        #     token in spatial row r is restricted to match ONLY the other-set
        #     tokens in the SAME row r (the simplified monotonic/diagonal cut =
        #     row-correspondence), instead of a free global max. Isolates "free
        #     partial set matching vs ordered body-region alignment". Needs the
        #     per-token row index (flat token k -> row k // gw).
        self.align = str(align)
        assert self.align in ('free', 'ordered'), \
            f"unknown --ovli_align {self.align}"
        # Precompute a (1,K,1,K) row-equality mask used ONLY by align='ordered':
        # entry (ki,kj) is True iff token ki and token kj fall in the same grid
        # row.  Registered as a NON-persistent buffer so it (a) follows .to(device)
        # with the module and (b) is never written into the checkpoint (it is a
        # pure constant of `grid`, and keeping it out keeps old ckpts loadable).
        _gh, _gw = self.grid
        _K = _gh * _gw
        _rows = torch.arange(_K) // _gw                     # (K,) row of each tok
        _row_eq = _rows.view(_K, 1).eq(_rows.view(1, _K))    # (K,K) bool
        self.register_buffer('_row_mask4', _row_eq.view(1, _K, 1, _K),
                             persistent=False)
        # Set-pooling of the K tokens into a single per-sample vector (the
        # headline aggregation step).
        #   setpool='mean' (default): NO new module -- the sym_MaxSim path below
        #     runs verbatim (byte-identical to the original for EVERY match / pool
        #     / align combo).  --ovli_match avg --ovli_pool mean reduces to the
        #     gram of the per-sample MEAN of the K tokens (the current best mech).
        #   setpool in {netvlad, attn, gated, secondorder}: a learnable, order-
        #     invariant OVLISetPool REPLACES the token-set MaxSim entirely -- the
        #     K tokens are aggregated into one vector and the cross-view score is
        #     the gram of those aggregated vectors (residual mode: UN-normalized
        #     `mean(+residual) @ .T` == the 52.37 avg/mean path; standalone mode:
        #     L2-normed cosine gram).  The match / pool / align / topk / thresh
        #     knobs only govern the MaxSim path and
        #     are BYPASSED in this mode.  Its params live under this module, so
        #     list(ovli.parameters()) -> optimizer (same contract as proj).
        self.setpool = str(setpool)
        assert self.setpool in ('mean', 'netvlad', 'attn', 'gated', 'secondorder'), \
            f"unknown --ovli_setpool {self.setpool}"
        # setpool_residual=True (default): each learnable pool is "mean + zero-init
        # residual", so aggregate_tokens starts BYTE-IDENTICAL to the 52.37 mean-
        # pool and only learns a correction (no random-init collapse).  =False:
        # the original standalone pooling (random-init replaces the mean) -- kept
        # only for the standalone-vs-residual ablation.
        self.setpool_residual = bool(setpool_residual)
        # 1x1 conv == per-token linear projection over channels (new params).
        # Build proj BEFORE the set-pool so its weights consume the SAME slice of
        # the RNG stream regardless of setpool (the set-pool draws randn for its
        # centers/query + kaiming-inits its Linear layers, which would otherwise
        # shift proj's init under a fixed seed).  With proj first, setpool != 'mean'
        # and setpool == 'mean' get byte-identical proj weights at the same seed,
        # so the residual start truly matches the 52.37 avg/mean path to the bit.
        self.proj = nn.Conv2d(in_ch, proj_dim, kernel_size=1, bias=True)
        self.proj.apply(weights_init_kaiming)
        self.setpool_mod = None
        if self.setpool != 'mean':
            self.setpool_mod = OVLISetPool(self.setpool, dim=proj_dim,
                                           out_dim=proj_dim, clusters=vlad_clusters,
                                           heads=attn_heads, lowrank=so_rank,
                                           residual=self.setpool_residual)
        # hook the GeM-input map; store WITHOUT detach so grad can flow.
        self._buf = {}
        self._handle = model.layer4.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        # keep the graph (no .detach()) -> proj/loss can backprop into layer4
        self._buf['map'] = out

    def remove_hook(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # -- token extraction ---------------------------------------------------- #
    def tokens_from_cached_map(self):
        """Project the most-recent layer4 map into L2-normed tokens (B,K,D).

        Reads the map captured by the hook during the just-run forward.  The
        projection runs in fp32 (numerical safety for the cos/MaxSim/logsumexp
        downstream); tokens are L2-normalized per token.
        """
        fmap = self._buf.get('map', None)
        if fmap is None:
            raise RuntimeError("OVLIHead: no cached layer4 map; run model "
                               "forward before tokens_from_cached_map().")
        # pool to the token grid, then project. Cast to fp32 for stability
        # (the cached map may be fp16 under autocast).
        fmap = F.adaptive_avg_pool2d(fmap.float(), self.grid)      # (B,Dc,gh,gw)
        tok = self.proj(fmap)                                       # (B,proj,gh,gw)
        B, C, H, W = tok.shape
        tok = tok.flatten(2).permute(0, 2, 1).contiguous()          # (B,K,proj)
        tok = F.normalize(tok, dim=2)                               # per-token L2
        return tok

    # -- learnable set pooling of the K tokens into one vector --------------- #
    def aggregate_tokens(self, tok):
        """Aggregate (B,K,D) L2-normed tokens -> (B,Dout) via the chosen
        permutation-invariant set pooling (OVLISetPool).

        ONLY called when setpool != 'mean'.  Shared by the train loss and the eval
        rerank so train and test use the IDENTICAL aggregation (train/test
        symmetry).  Runs in fp32.

        ** Normalization convention (load-bearing for the residual kill-switch) **
        The 52.37 avg/mean path computes its cross-view gram as the UN-normalized
        `mean_k(tok) @ mean_k(tok).T` (the per-sample mean of the K L2-normed
        tokens is NOT re-normalized -- see --ovli_match avg --ovli_pool mean, which
        reduces to exactly that gram).  So:

          * setpool_residual=True (default): return the RAW aggregated vector
            `mean_k(tok) + gate_res * residual(tok)` WITHOUT a final L2-norm.  At
            gate_res==0 this is exactly `mean_k(tok)`, hence `a @ a.T` is the
            UN-normalized mean gram == the 52.37 avg/mean-pool gram BYTE-FOR-BYTE
            (verified torch.equal / <1e-6 in smoke_ovli_residual.py).  Re-norming
            here would instead give `<normalize(mean), normalize(mean)>` (diag==1),
            which is a DIFFERENT score and would make the kill-switch start off the
            52.37 path.  The residual is what the gate learns to add.

          * setpool_residual=False (standalone control): the aggregated vector
            REPLACES the mean, so it IS L2-normalized -> the cross-view gram is a
            cosine in [-1,1] (the original standalone convention; this branch is
            the random-init collapse control, not the residual start).
        """
        a = self.setpool_mod(tok)              # (B, Dout) fp32, permutation-invariant
        if self.setpool_residual:
            # raw mean(+residual): UN-normalized gram == 52.37 avg/mean-pool path.
            return a
        return F.normalize(a, dim=1)           # standalone: unit vectors -> cosine gram

    # -- pooling over per-token max scores (mean / topk / thresh / softtopk) -- #
    @staticmethod
    def pool_token_max(tmax, dim, pool='mean', topk=8, thresh=0.0, tau=0.05):
        """Reduce per-token max scores `tmax` along `dim` into a pair score.

        `tmax` holds, for each (pair, token) the best (max) match of that token
        into the other token set; this collapses the surviving token axis `dim`
        into one number per pair, using the chosen dustbin variant.

        * mean     : ordinary average over all token-max scores.
        * topk     : average of the top-min(k, K) highest token-max scores along
                     `dim` (low-score non-corresponding tokens are dropped).
        * thresh   : average of token-max scores > `thresh`; if a pair has none
                     above threshold, fall back to its single max (small but
                     finite) so the result is never NaN.
        * softtopk : softmax(tmax/tau)-weighted mean over token-max scores along
                     `dim` (a smooth, differentiable top-k surrogate).

        Returns the pair-score tensor with axis `dim` removed.
        """
        K = tmax.size(dim)
        if pool == 'mean':
            return tmax.mean(dim=dim)
        if pool == 'topk':
            k = max(1, min(int(topk), K))
            vals = tmax.topk(k, dim=dim).values                    # (...,k,...)
            return vals.mean(dim=dim)
        if pool == 'thresh':
            mask = tmax > thresh                                    # bool, same shape
            masked = torch.where(mask, tmax, tmax.new_zeros(()))
            cnt = mask.sum(dim=dim)                                 # (...) ints
            summed = masked.sum(dim=dim)                            # (...)
            mean_above = summed / cnt.clamp(min=1).to(tmax.dtype)
            # pairs with no token above threshold -> fall back to the single max
            # (finite, never NaN); broadcast cnt==0 back over the reduced axis.
            fallback = tmax.max(dim=dim).values
            no_valid = cnt == 0
            return torch.where(no_valid, fallback, mean_above)
        if pool == 'softtopk':
            w = torch.softmax(tmax / tau, dim=dim)                  # weights sum 1
            return (w * tmax).sum(dim=dim)
        raise ValueError(f"unknown pool {pool}")

    # -- per-query-token reduction over the OTHER token set ------------------ #
    def _reduce_other(self, sim, other_dim):
        """Collapse the 'other' token axis `other_dim` of a (.,Kq,.,Kg) similarity
        tensor into one score per query token, honoring the match / align modes.

        `sim` layout: the two token axes are at dim 1 (query tokens) and dim 3
        (other tokens); `other_dim` is the axis to reduce (3 = query->other,
        1 = other->query).  The (K,K) row mask is symmetric so the same buffer
        serves both directions, keeping sym_MaxSim symmetric.

        * match='maxsim' (default): max over the other tokens (ColBERT selection).
        * match='avg'             : mean over the other tokens (soft global match).
        * align='free'   (default): reduce over ALL other tokens.
        * align='ordered'         : reduce only over other tokens in the SAME grid
                                    row (row-correspondence).  Masked max uses a
                                    finite floor (never -inf) so an empty row (can
                                    only happen for a degenerate gw=0 grid, not the
                                    fixed gh x gw one) is NaN-safe; masked mean
                                    clamps the per-row count to >=1.

        Default (maxsim + free) returns exactly `sim.max(dim=other_dim).values`,
        so the off-mode forward is byte-identical to the original.
        """
        if self.align == 'free':
            if self.match == 'maxsim':
                return sim.max(dim=other_dim).values            # original behaviour
            return sim.mean(dim=other_dim)                       # match == 'avg'
        # align == 'ordered': restrict each query token to its same-row others.
        mask4 = self._row_mask4                                  # (1,K,1,K) bool
        if self.match == 'maxsim':
            floor = sim.new_full((), -1e4)                      # finite, NaN-safe
            masked = torch.where(mask4, sim, floor)
            return masked.max(dim=other_dim).values
        # match == 'avg': masked mean over the same-row other tokens.
        cnt = mask4.sum(dim=other_dim).clamp(min=1).to(sim.dtype)
        summed = torch.where(mask4, sim, sim.new_zeros(())).sum(dim=other_dim)
        return summed / cnt

    # -- symmetric token-set MaxSim (bidirectional, full BxB) ---------------- #
    def sym_maxsim_matrix(self, tok):
        """Bidirectional MaxSim score for every ordered pair in the batch.

        tok:(B,K,D) L2-normed.  Returns (B,B):
            s(i,j) = 0.5*( pool_u max_s <u_i, s_j> + pool_s max_u <u_i, s_j> )
        where pool_* is the configured dustbin pooling (mean/topk/thresh/
        softtopk).  Symmetric in (i,j) by construction (both directions use the
        same pooling).  Computed densely (B=64, K=32 -> a 64x64x32x32 sim tensor
        ~ 4M floats; fine on-GPU).

        setpool != 'mean' REPLACES the whole token-set MaxSim with the learnable
        order-invariant aggregation: msim = gram of the per-sample aggregated
        vectors (still symmetric).  In the default residual mode that gram is the
        UN-normalized `mean(+residual) @ .T`, so at gate_res==0 it is byte-equal to
        the --ovli_match avg --ovli_pool mean (52.37) gram; standalone mode uses
        the L2-normed cosine gram.  setpool == 'mean' (default) leaves the original
        MaxSim path below completely untouched (byte-identical for every match /
        pool / align combo -> off-mode reproduction is structural).
        """
        if self.setpool != 'mean':
            a = self.aggregate_tokens(tok)         # (B,Dout) L2-normed
            return a @ a.t()                       # (B,B) cosine gram, symmetric
        B, K, D = tok.shape
        flat = tok.reshape(B * K, D)                                # (B*K, D)
        sim = (flat @ flat.t()).reshape(B, K, B, K)                 # (B,K,B,K)
        # i-token reduced over j-tokens (dim=3; max/avg, free/row) -> pool i-tokens
        i2j_max = self._reduce_other(sim, other_dim=3)            # (B,K,B) over i-tok
        i2j = self.pool_token_max(i2j_max, dim=1, pool=self.pool,
                                  topk=self.topk, thresh=self.thresh,
                                  tau=self.tau)                     # (B,B)
        # j-token reduced over i-tokens (dim=1; max/avg, free/row) -> pool j-tokens
        j2i_max = self._reduce_other(sim, other_dim=1)            # (B,B,K) over j-tok
        j2i = self.pool_token_max(j2i_max, dim=2, pool=self.pool,
                                  topk=self.topk, thresh=self.thresh,
                                  tau=self.tau)                     # (B,B)
        return 0.5 * (i2j + j2i)

    # -- ACVP: ambiguity-calibrated opposite-view negative relaxation -------- #
    @torch.no_grad()
    def acvp_neg_bias(self, gfeat, labels, views, neg, proto, inited,
                      gamma, wmin, eta, margin):
        """Detached log-weight bias added to the NEGATIVE logits of the OVLI
        contrastive denominator (ACVP).  Returns `(bias, frac, mean_w, n_soft)`:

          bias:(B,B) fp32 additive term = log(w_ij) on negative (anchor i, cand j)
                     entries and 0 everywhere else.  Added to `cand_logits` ONLY,
                     so the numerator (positives) is untouched -- ACVP softens
                     negatives only.
          frac:      scalar = fraction of SOFTENABLE negative pairs with
                     w_ij < 0.95 (kill-switch: frac > 0.30 => ACVP is broadly
                     weakening negatives = bad).
          mean_w:    scalar = mean w_ij over the SOFTENABLE negative pairs
                     (kill-switch: mean_w < 0.75 => negatives broadly softened
                     = bad).
          n_soft:    scalar long = number of softenable negative pairs (ok.sum()),
                     i.e. opposite-view negatives with BOTH prototypes initialised.
                     The caller weights the per-step (frac, mean_w) stats by this
                     count and SKIPS steps with n_soft==0, so cold-start batches
                     (no initialised prototypes) never bias the per-epoch summary.

        EVERYTHING here is under torch.no_grad() and uses the DETACHED prototype
        bank, so ACVP injects no gradient into the encoder/proj/prototypes: it is a
        pure, detached re-weighting of the contrastive denominator (an ambiguity
        SENSOR, not a learned alignment).  No-op-safe: pairs whose prototype lookup
        is uninitialised get w_ij == 1 (log 0) -> they contribute 0 bias.

        delta_ij = cos(z_i, P[y_j, view_j]) - cos(z_i, P[y_i, view_j])
                   (how much closer negative j's opposite-view identity sits to i
                    than i's own opposite-view identity -> ambiguity of j as a
                    negative; z_i, P[.] L2-normed so cos == dot).
        w_ij     = clamp(1 - gamma * sigmoid((delta_ij - margin)/eta), wmin, 1.0)
        bias_ij  = log(w_ij)   (w_ij >= wmin > 0 -> finite, never -inf/NaN).
        """
        B = gfeat.size(0)
        # work in fp32 for the cos/sigmoid/clamp/log numerics (gfeat may be fp32
        # already; .float() is a no-op then).  proto is L2-normed in the bank.
        z = gfeat.float()                                           # (B,D) L2-normed
        P = proto.float()                                          # (num_pid,2,D) detached
        view_j = views.view(1, B).expand(B, B)                      # (B,B) cand view
        y_i = labels.view(B, 1).expand(B, B)                        # (B,B) anchor pid
        y_j = labels.view(1, B).expand(B, B)                        # (B,B) cand pid
        # gather prototypes P[pid, view_j] for the "i's own opp identity" and
        # "j's identity in j's view" terms.  view_j indexes the SAME (cand) view
        # for both, so both prototypes live in the view the negative occupies.
        flatPV = P.reshape(-1, P.size(-1))                          # (num_pid*2, D)
        idx_self = (y_i * 2 + view_j).reshape(-1)                   # P[y_i, view_j]
        idx_neg = (y_j * 2 + view_j).reshape(-1)                   # P[y_j, view_j]
        proto_self = flatPV[idx_self].reshape(B, B, -1)            # (B,B,D)
        proto_neg = flatPV[idx_neg].reshape(B, B, -1)             # (B,B,D)
        zb = z.view(B, 1, P.size(-1))                               # (B,1,D) anchor i
        cos_self = (zb * proto_self).sum(-1)                       # (B,B) cos(z_i,P[y_i,vj])
        cos_neg = (zb * proto_neg).sum(-1)                        # (B,B) cos(z_i,P[y_j,vj])
        delta = cos_neg - cos_self                                 # (B,B) ambiguity
        s = torch.sigmoid((delta - margin) / eta)                  # (B,B) in (0,1)
        w = torch.clamp(1.0 - gamma * s, min=wmin, max=1.0)        # (B,B) in [wmin,1]
        # validity: only softens a negative pair when BOTH prototype lookups are
        # initialised (never-seen prototype is a zero vector -> meaningless cos);
        # uninitialised -> w forced to 1 (bias 0, no softening).
        flat_inited = inited.reshape(-1).bool()                    # (num_pid*2,)
        valid_self = flat_inited[idx_self].reshape(B, B)
        valid_neg = flat_inited[idx_neg].reshape(B, B)
        ok = neg & valid_self & valid_neg                          # (B,B) softenable negs
        w = torch.where(ok, w, w.new_ones(()))                     # uninit/non-neg -> w=1
        bias = torch.log(w)                                        # (B,B) <=0, finite
        # kill-switch stats over softenable negatives only (where ACVP can act).
        # n_soft = #softenable-neg this step; the caller weights frac/mean_w by it
        # and drops n_soft==0 steps so cold-start batches don't skew the summary.
        n_soft = ok.sum()                                          # scalar long
        if ok.any():
            wv = w[ok]
            frac = (wv < 0.95).float().mean()
            mean_w = wv.mean()
        else:
            frac = w.new_zeros(())
            mean_w = w.new_ones(())
        return bias, frac, mean_w, n_soft

    # -- loss ---------------------------------------------------------------- #
    def loss(self, gfeat, tok, labels, views,
             acvp_proto=None, acvp_inited=None, acvp_gamma=0.0,
             acvp_wmin=0.3, acvp_eta=0.05, acvp_margin=0.0):
        """Supervised-contrastive late-interaction retrieval loss.

        gfeat:(B,D) L2-normed global feature (gradient flows -> encoder).
        tok:(B,K,Dp) L2-normed projected tokens (gradient flows -> proj+encoder).
        labels:(B,) long.  views:(B,) long in {0,1}.

        Per anchor i the candidate set depends on the ablation mode:
          * self.allview=False (default, headline): candidates = opposite-view
            samples (view != view_i); positives = opposite-view same-pid samples.
          * self.allview=True (--ovli_allview control): candidates = ALL other
            samples (any view, excl. self); positives = same-pid any view.
        Multi-positive InfoNCE via logsumexp over the same candidate set. Returns
        (loss, pos_score_mean, neg_score_mean) where the score means are
        diagnostics over the contributing (anchor, candidate) pairs.

        ACVP (optional, default OFF -> byte-identical to the pre-ACVP loss):
          When `acvp_proto is not None`, the NEGATIVE logits in the contrastive
          DENOMINATOR get a detached ambiguity log-weight `log(w_ij)` added
          (acvp_neg_bias); positives/numerator are untouched.  `acvp_proto` is the
          DETACHED opposite-view prototype bank [num_pid,2,D] (no grad to encoder/
          proto).  Returns the SAME 3-tuple; the per-epoch ACVP kill-switch stats
          (relaxed_neg_frac, mean_w) are stashed on self._acvp_stats for the caller.
          acvp_proto is None reproduces the original loss EXACTLY (the floor /
          where / logsumexp lines are reached unchanged).
        """
        B = gfeat.size(0)
        device = gfeat.device
        # pairwise hybrid score in fp32 (cos in [-1,1], maxsim in [-1,1])
        gsim = gfeat @ gfeat.t()                                   # (B,B)
        msim = self.sym_maxsim_matrix(tok)                         # (B,B)
        score = self.alpha * gsim + (1.0 - self.alpha) * msim       # (B,B)

        same_view = views.view(-1, 1).eq(views.view(1, -1))         # (B,B)
        same_pid = labels.view(-1, 1).eq(labels.view(1, -1))        # (B,B)
        eye = torch.eye(B, dtype=torch.bool, device=device)

        # Candidate set depends on the ablation mode (only this mask changes;
        # everything downstream is identical across modes).
        #   allview=False (default): opposite-view-only -> identical to original.
        #   allview=True  (control): all-view -> candidates = every other sample.
        if self.allview:
            cand = ~eye                     # all-view, not self
        else:
            cand = (~same_view) & (~eye)    # opposite-view, not self
        pos = cand & same_pid               # positives: same pid (any cand view)
        neg = cand & (~same_pid)            # negatives: other pid (any cand view)

        # anchors that have >=1 positive AND >=1 negative in the candidate set
        valid = (pos.sum(dim=1) > 0) & (neg.sum(dim=1) > 0)         # (B,)
        if valid.sum() == 0:
            z = gfeat.new_zeros(())
            if acvp_proto is not None:
                # keep the stats slot defined even on an empty batch (no
                # softenable negatives -> n_soft=0 => the caller skips this step).
                self._acvp_stats = (z.detach(), gfeat.new_ones(()),
                                    gfeat.new_zeros((), dtype=torch.long))
            return z, z, z

        logits = score / self.tau                                   # (B,B)
        # Use a large finite floor (not -inf) so a fully-masked row can never
        # yield -inf, and so (-inf)-(-inf)=nan can never appear even before the
        # valid-row selection. logsumexp over real candidates dominates this
        # floor by a wide margin (logits ~ [-20,20] at tau=0.05).
        floor = logits.new_full((), -1e4)
        # numerator: logsumexp over positives only; denominator: over all cands
        pos_logits = torch.where(pos, logits, floor)
        cand_logits = torch.where(cand, logits, floor)
        # ACVP: soften the NEGATIVE logits in the denominator with a detached
        # ambiguity log-weight.  acvp_proto is None -> this whole block is skipped
        # and cand_logits is byte-identical to the original (off-mode reproduction).
        if acvp_proto is not None:
            acvp_bias, acvp_frac, acvp_mean_w, acvp_nsoft = self.acvp_neg_bias(
                gfeat, labels, views, neg, acvp_proto, acvp_inited,
                acvp_gamma, acvp_wmin, acvp_eta, acvp_margin)
            # log(w_ij)<=0 added to negative entries ONLY (bias is 0 elsewhere and
            # on uninitialised/non-neg pairs).  Detached -> no grad to proto/feat.
            cand_logits = cand_logits + acvp_bias.to(cand_logits.dtype)
            # stash (frac, mean_w, #softenable-neg) for the caller's per-epoch
            # kill-switch summary (weighted by #softenable-neg, not batch size).
            self._acvp_stats = (acvp_frac.detach(), acvp_mean_w.detach(),
                                acvp_nsoft.detach())
        log_num = torch.logsumexp(pos_logits, dim=1)                # (B,)
        log_den = torch.logsumexp(cand_logits, dim=1)               # (B,)
        per_anchor = -(log_num - log_den)                           # (B,)
        # only anchors with >=1 positive AND >=1 negative in the candidate set
        loss = per_anchor[valid].mean()

        # diagnostics (detached): mean positive / negative pair scores
        with torch.no_grad():
            ps = score[pos].mean() if pos.any() else score.new_zeros(())
            ns = score[neg].mean() if neg.any() else score.new_zeros(())
        return loss, ps, ns


# --------------------------------------------------------------------------- #
# OVLI rerank: eval-time global + sym_MaxSim rerank (opt-in, symmetric w/ train)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ovli_rerank_eval(model, ovli, dataset, args, device):
    """Report A->G / G->A mAP/R1 for (a) global-only and (b) global+MaxSim
    rerank, using the SAME projected tokens + sym MaxSim as the training loss.

    Mirrors run_cross_view_eval but additionally extracts projected tokens via
    the OVLI hook and reranks by score = alpha*cos(global) + (1-alpha)*MaxSim.
    Gallery token sets can be large, so MaxSim is chunked over the gallery axis.
    Returns {tag: {'global': (mAP,R1), 'rerank': (mAP,R1)}}.
    """
    from cargo_dataset import filter_by_view as _fbv

    model.eval()

    @torch.no_grad()
    def extract(samples):
        from afd_train import build_eval_loader as _bel
        loader = _bel(samples, args)
        gfs, tks, pids, cams = [], [], [], []
        view_map = {'Aerial': 0, 'Ground': 1}
        for batch in loader:
            imgs = batch['img'].to(device, non_blocking=True)
            vidx = (torch.tensor([view_map[v] for v in batch['view']],
                                 device=device) if args.use_afd else None)
            gf = model(imgs, view_idx=vidx)              # (b,D) L2-normed BN
            tok = ovli.tokens_from_cached_map()           # (b,K,Dp) L2-normed
            gfs.append(gf.cpu())
            tks.append(tok.cpu())
            pids.append(batch['pid'])
            cams.append(batch['camid'])
        if not gfs:
            return (torch.empty(0), torch.empty(0),
                    np.empty(0, np.int64), np.empty(0, np.int64))
        return (torch.cat(gfs, 0), torch.cat(tks, 0),
                torch.cat(pids, 0).numpy(), torch.cat(cams, 0).numpy())

    # eval rerank uses the SAME dustbin pooling as the train loss so train/test
    # stay symmetric (mean reproduces the original rerank exactly).
    _pool = ovli.pool
    _topk = ovli.topk
    _thresh = ovli.thresh
    _tau = ovli.tau

    @torch.no_grad()
    def maxsim_block(qt, gt):
        """(Nq,Ng) bidirectional MaxSim, chunked over the gallery axis."""
        # setpool != 'mean': the cross-view score is the gram of the learnable
        # aggregated vectors (train/test symmetric via the SAME aggregate_tokens
        # used by the train loss -> residual mode = UN-normalized mean(+residual)
        # gram == 52.37 avg/mean path at gate_res==0; standalone = cosine gram),
        # NOT the token-set MaxSim.  Aggregate query/gallery tokens in sample-row
        # blocks (bounds the netvlad (N,K,C,D) intermediate) on-device, gram on CPU.
        if ovli.setpool != 'mean':
            def _agg_all(t):
                outs = []
                for s in range(0, t.size(0), 256):
                    outs.append(ovli.aggregate_tokens(t[s:s + 256].to(device)).cpu())
                return (torch.cat(outs, 0) if outs
                        else torch.empty(0, ovli.setpool_mod.out_dim))
            return _agg_all(qt) @ _agg_all(gt).t()
        Nq, Kq, C = qt.shape
        Ng, Kg, _ = gt.shape
        qd = qt.to(device).reshape(Nq * Kq, C)
        budget = 80_000_000
        per_g = max(1, Nq * Kq * Kg)
        gblk = max(1, min(Ng, budget // per_g))
        out = torch.empty(Nq, Ng)
        for s in range(0, Ng, gblk):
            e = min(s + gblk, Ng)
            gc = gt[s:e].to(device)
            g = gc.size(0)
            sim = (qd @ gc.reshape(g * Kg, C).t()).reshape(Nq, Kq, g, Kg)
            # same match/align reduction as the train loss (train/test symmetry):
            # ovli._reduce_other honors --ovli_match and --ovli_align identically.
            q2g_max = ovli._reduce_other(sim, other_dim=3)  # (Nq,Kq,g) over q-tok
            q2g = OVLIHead.pool_token_max(q2g_max, dim=1, pool=_pool,
                                          topk=_topk, thresh=_thresh, tau=_tau)
            g2q_max = ovli._reduce_other(sim, other_dim=1)  # (Nq,g,Kg) over g-tok
            g2q = OVLIHead.pool_token_max(g2q_max, dim=2, pool=_pool,
                                          topk=_topk, thresh=_thresh, tau=_tau)
            out[:, s:e] = (0.5 * (q2g + g2q)).cpu()
            del sim, q2g_max, q2g, g2q_max, g2q, gc
        del qd
        if device == 'cuda':
            torch.cuda.empty_cache()
        return out

    from afd_train import eval_market
    results = {}
    splits = {
        'A->G': (_fbv(dataset.query, 'Aerial'), _fbv(dataset.gallery, 'Ground')),
        'G->A': (_fbv(dataset.query, 'Ground'), _fbv(dataset.gallery, 'Aerial')),
    }
    for tag, (q, g) in splits.items():
        qf, qt, qp, qc = extract(q)
        gf, gt, gp, gc = extract(g)
        if qf.numel() == 0 or gf.numel() == 0:
            results[tag] = {'global': (float('nan'), float('nan')),
                            'rerank': (float('nan'), float('nan'))}
            continue
        qf = F.normalize(qf, dim=1)
        gf = F.normalize(gf, dim=1)
        gsim = (qf @ gf.t()).numpy()                      # (Nq,Ng) cosine
        # global-only (rank by cosine distance == -gsim)
        gmap, gcmc, _ = eval_market(qf, qp, qc, gf, gp, gc)
        # rerank: alpha*cos + (1-alpha)*MaxSim, rank by descending hybrid
        msim = maxsim_block(qt, gt).numpy()
        hyb = args.ovli_alpha * gsim + (1.0 - args.ovli_alpha) * msim
        from maxsim_probe import eval_from_distmat
        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
        results[tag] = {
            'global': (gmap * 100, gcmc[0] * 100),
            'rerank': (rmap, rr1),
        }
    return results


def airl_dualbranch_eval(model, dataset, args, device):
    """AIRL dual-branch eval: extract BOTH heads (f_full, f_rec) in ONE forward
    and report f_full-only, f_rec-only, and the SOFT-FUSED cosine ranking
    (cos = w*cos_rec + (1-w)*cos_full, w = args.airl_fuse_w, fixed) for A->G and
    G->A.  This is the single-model analog of the kill-switch #3 two-model score
    fusion: cos_rec replaces the AIRL-model cosine, cos_full replaces the
    baseline-model cosine, and they share ONE backbone forward.

    Mirrors run_cross_view_eval / ovli_rerank_eval exactly for the per-split
    feature extraction and the eval_market / eval_from_distmat ranking, so the
    f_full number reproduces run_cross_view_eval's A<->G mAP bit-for-bit (same
    feature, same ranking) and the fusion is a pure distance-matrix combination.
    Returns {tag: {'full': (mAP,R1), 'rec': (mAP,R1), 'fuse': (mAP,R1)}}.
    """
    from cargo_dataset import filter_by_view as _fbv
    from afd_train import build_eval_loader as _bel
    from maxsim_probe import eval_from_distmat

    model.eval()
    view_map = {'Aerial': 0, 'Ground': 1}

    @torch.no_grad()
    def extract(samples):
        loader = _bel(samples, args)
        ffs, frs, pids, cams = [], [], [], []
        for batch in loader:
            imgs = batch['img'].to(device, non_blocking=True)
            vidx = (torch.tensor([view_map[v] for v in batch['view']],
                                 device=device) if args.use_afd else None)
            # ONE forward -> two L2-normalized features (f_full, f_rec).
            f_full, f_rec = model(imgs, view_idx=vidx, return_dual=True)
            ffs.append(f_full.cpu())
            frs.append(f_rec.cpu())
            pids.append(batch['pid'])
            cams.append(batch['camid'])
        if not ffs:
            empty = (torch.empty(0), torch.empty(0),
                     np.empty(0, np.int64), np.empty(0, np.int64))
            return empty
        return (torch.cat(ffs, 0), torch.cat(frs, 0),
                torch.cat(pids, 0).numpy(), torch.cat(cams, 0).numpy())

    w = args.airl_fuse_w
    results = {}
    splits = {
        'A->G': (_fbv(dataset.query, 'Aerial'), _fbv(dataset.gallery, 'Ground')),
        'G->A': (_fbv(dataset.query, 'Ground'), _fbv(dataset.gallery, 'Aerial')),
    }
    for tag, (q, g) in splits.items():
        q_full, q_rec, qp, qc = extract(q)
        g_full, g_rec, gp, gc = extract(g)
        if q_full.numel() == 0 or g_full.numel() == 0:
            nan2 = (float('nan'), float('nan'))
            results[tag] = {'full': nan2, 'rec': nan2, 'fuse': nan2}
            continue
        # features are already L2-normalized at eval; renormalize defensively so
        # the cosine == the gram of unit vectors (matches eval_market exactly).
        q_full = F.normalize(q_full, dim=1); g_full = F.normalize(g_full, dim=1)
        q_rec = F.normalize(q_rec, dim=1);   g_rec = F.normalize(g_rec, dim=1)
        s_full = (q_full @ g_full.t()).numpy()        # (Nq,Ng) cosine, f_full
        s_rec = (q_rec @ g_rec.t()).numpy()           # (Nq,Ng) cosine, f_rec
        # soft fusion: cos = w*cos_rec + (1-w)*cos_full -> distance = 2 - 2*cos
        # (identical to kill-switch #3 GATE 5; cosine in [-1,1] -> dist in [0,4]).
        dm_full = (2.0 - 2.0 * s_full)
        dm_rec = (2.0 - 2.0 * s_rec)
        dm_fuse = (2.0 - 2.0 * (w * s_rec + (1.0 - w) * s_full))
        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
        results[tag] = {'full': (fmap, fr1), 'rec': (rmap, rr1),
                        'fuse': (zmap, zr1)}
    return results


# --------------------------------------------------------------------------- #
# AIRL: Aerial Identity Recoverability Learning (resolution-degradation
#       consistency).  kill-switch #2.
# --------------------------------------------------------------------------- #
# Motivation (new_angle_AIRL.md, kill-switch #1 PASS): on CARGO the aerial->ground
# error is dominated by the AERIAL crop's low PIXEL BUDGET (small bbox -> identity
# physically under-resolved), NOT only a view-alignment gap.  The zero-training
# bucketed diagnostic showed the lowest aerial-scale bucket collapses by +13~19 mAP
# vs the top bucket -- on the STRONG Swin baseline too, so it is a physical pixel
# problem, not a backbone-headroom artifact (the OVLI failure mode).
#
# AIRL turns that diagnostic into a training signal WITHOUT any cross-view
# contrastive / late-interaction / pooling / prototype machinery (those are the
# OVLI dead zone).  Mechanism = resolution-degradation CONSISTENCY:
#   1. For each GROUND image (high-res) sample a "pixel budget" = a scale ratio
#      drawn to MATCH the aerial bbox scale distribution (small aerial buckets ->
#      heavy degradation), degrade the image to that budget (bilinear down then
#      up back to the original H x W, + optional light avg-pool blur), simulating
#      "if this person were shot from a UAV, how much information would survive".
#   2. Both the original and the degraded image pass the SAME backbone (shared
#      weights, one extra forward); a PREDICTION-CONSISTENCY loss forces the
#      degraded view's identity prediction to agree with the original
#      (KL on logits, or cosine/MSE on the L2-normed BNNeck feature).  Intuition:
#      learn identity evidence that is STABLE under a low pixel budget; suppress
#      reliance on ground-only high-frequency detail.
#   3. total = CE + triplet + airl_lambda_eff * consistency.
#
# Design contract (hard):
#   * NO new learnable parameters -- degradation is an image-space augmentation,
#     consistency is a loss.  The optimizer / param groups are untouched.
#   * --airl OFF (default) => NO degradation, NO extra forward, NO loss term =>
#     the baseline is reproduced BYTE-FOR-BYTE (the whole AIRL block is skipped).
#   * The consistency loss runs in TRUE fp32 (autocast disabled) for KL/cosine
#     numeric safety (finite inputs: logits/features from a finite forward).
#   * AIRL is a TRAIN-time loss only; eval is unchanged (train/test symmetric).
#   * Backbone-agnostic: the degradation is purely in image space, so resnet50 and
#     swin_small are both supported (the second forward just reuses `model`).
def airl_degrade(imgs, min_scale, blur=False, generator=None):
    """Resolution-degrade a NORMALIZED image batch to a sampled pixel budget.

    imgs:(B,C,H,W) the model-input batch (already Resize+Normalized by the
    dataloader; degradation is a linear resample in normalized space, which is a
    faithful resolution/low-pass proxy -- it only removes high-frequency detail
    and never shifts the per-channel statistics the backbone expects).

    Per image a scale ratio s ~ U[min_scale, 1.0] is drawn (the "pixel budget":
    s=1 keeps full resolution, s=min_scale is the heaviest aerial-small-bucket
    degradation).  The image is bilinearly DOWN-sampled to (round(s*H), round(s*W))
    (>=1 px) then bilinearly UP-sampled back to (H, W), so the output keeps the
    original shape but only carries ~s*100% of the spatial detail.  Optionally a
    light 3x3 avg-pool blur (stride 1, reflect pad) is applied AFTER the up-sample
    to mimic UAV optical blur without any PIL/cv2 dependency.

    Runs in fp32 on the input device; antialias=True for a clean low-pass on the
    down step.  Returns (degraded:(B,C,H,W), scales:(B,) the per-image s used) so
    the caller can log deg_scale_mean.  Per-image scales => per-image target sizes,
    so the resample is done one image at a time (B is small, bs<=64).
    """
    B, C, H, W = imgs.shape
    x = imgs.float()
    if generator is not None:
        s = (torch.rand(B, generator=generator, device='cpu')
             .to(x.device) * (1.0 - min_scale) + min_scale)
    else:
        s = torch.rand(B, device=x.device) * (1.0 - min_scale) + min_scale
    out = torch.empty_like(x)
    for i in range(B):
        si = float(s[i])
        th = max(1, int(round(si * H)))
        tw = max(1, int(round(si * W)))
        if th >= H and tw >= W:
            # s rounds to full size -> no spatial detail removed (still pass blur
            # below if requested so the op is uniform).
            yi = x[i:i + 1]
        else:
            down = F.interpolate(x[i:i + 1], size=(th, tw), mode='bilinear',
                                 align_corners=False, antialias=True)
            yi = F.interpolate(down, size=(H, W), mode='bilinear',
                               align_corners=False)
        out[i] = yi[0]
    if blur:
        # 3x3 average blur (stride 1) with reflect padding -> shape preserved,
        # NaN-safe (pure local mean), no learnable params.
        out = F.avg_pool2d(F.pad(out, (1, 1, 1, 1), mode='reflect'),
                           kernel_size=3, stride=1)
    return out, s


def airl_consistency_loss(logits_o, bn_o, logits_d, bn_d, mode='kl', tau=4.0):
    """Prediction-consistency between the ORIGINAL and DEGRADED views (fp32).

    The degraded view must keep its identity prediction CLOSE to the original's,
    so the model learns evidence that survives a low pixel budget.  The ORIGINAL
    side is DETACHED (it is the stable target; gradient flows only through the
    degraded branch -> the model is pulled to make the degraded prediction match
    the clean one, not the reverse).

      mode='kl' (default): symmetric KL would double-count; we use the standard
        distillation direction KL(softmax(logits_o/tau).detach || softmax(
        logits_d/tau)) * tau^2 (temperature-scaled soft-target consistency on the
        ID logits).  fp32 for numeric safety; log_softmax avoids log(0).
      mode='feat': 1 - cos(bn_o.detach, bn_d) on the L2-normed BNNeck feature
        (MSE-equivalent up to scale on unit vectors; bounded in [0,2]).

    Returns a finite non-negative scalar.  A light finite guard (nan_to_num) is
    applied to the scalar: inputs are finite under normal training, so this is a
    no-op there, but it keeps the documented "finite scalar" contract honest if a
    pathological forward ever produced inf/NaN logits.
    """
    if mode == 'feat':
        zo = F.normalize(bn_o.float(), dim=1).detach()
        zd = F.normalize(bn_d.float(), dim=1)
        # 1 - cosine in [0,2]; mean over batch.  (== 0.5*||zo-zd||^2 on unit vecs.)
        out = (1.0 - (zo * zd).sum(dim=1)).mean()
        return torch.nan_to_num(out)
    # mode == 'kl': temperature-scaled soft-target KL (clean = detached target).
    lo = logits_o.float() / tau
    ld = logits_d.float() / tau
    p_o = F.softmax(lo, dim=1).detach()                       # stable target
    log_p_d = F.log_softmax(ld, dim=1)                        # fp32; no log(0)
    # KL(p_o || p_d) = sum p_o (log p_o - log p_d); * tau^2 keeps the gradient
    # magnitude comparable across temperatures (Hinton distillation convention).
    log_p_o = F.log_softmax(lo, dim=1).detach()
    kl = (p_o * (log_p_o - log_p_d)).sum(dim=1).mean()
    return torch.nan_to_num(kl * (tau * tau))


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
    # 2nd benchmark. 'cargo' (default) = byte-identical CARGO path.
    # 'agreid'    = AG-ReID.v2 binary aerial<->ground merge (NON-official, legacy).
    # 'agreid_v2' = AG-ReID.v2 OFFICIAL protocols: A->G == exp1 aerial_to_cctv,
    #               G->A == exp4 cctv_to_aerial, mean of the two (the analogue of
    #               CARGO's A<->G mean). This is the cross-dataset main-table column.
    ap.add_argument('--dataset', default='cargo',
                    choices=['cargo', 'agreid', 'agreid_v2'])
    ap.add_argument('--out_dir', default='./log/cargo/cvpb_ovp')
    ap.add_argument('--img_size', type=int, nargs=2, default=[256, 128])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--warmup_epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=3.5e-4)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--P', type=int, default=16)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--test_batch', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--margin', type=float, default=0.3)
    ap.add_argument('--label_smooth', type=float, default=0.1)
    ap.add_argument('--eval_period', type=int, default=10)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--no_amp', action='store_true', help='disable mixed precision')
    # model switches (keep AFD off by default -> pure BoT baseline + OVP)
    ap.add_argument('--last_stride', type=int, default=1)
    ap.add_argument('--pool', default='gem', choices=['gem', 'avg'])
    # backbone selector. 'resnet50' (default) = the existing BoT baseline
    # (IMAGENET1K_V1 + GeM + BNNeck), byte-for-byte unchanged (52.37 headline).
    # 'swin_small' = SOLIDER Swin-Small (team asset, SOTA push): SOLIDER teacher
    # pretrain + avg-pool + BNNeck, in_planes=768; AFD freq modules are NOT
    # supported on swin (OVP/OVLI are independent and DO work). OVLI hooks the
    # last-stage spatial map (B,768,8,4 @ 256x128) -> same hook contract.
    ap.add_argument('--backbone', default='resnet50',
                    choices=['resnet50', 'swin_small'],
                    help="backbone: resnet50 (default, BoT baseline, byte-identical) "
                         "or swin_small (SOLIDER Swin-Small, in_planes=768)")
    ap.add_argument('--swin_pretrain', default='',
                    help="path to the SOLIDER swin_small.pth teacher checkpoint "
                         "(e.g. <repo>/pretrained/swin_small.pth). Empty -> train "
                         "the Swin from trunc-normal init. Only used with "
                         "--backbone swin_small.")
    ap.add_argument('--swin_semantic_weight', type=float, default=0.2,
                    help="SOLIDER semantic weight for the Swin backbone "
                         "(0.2 = ReID default; <0 disables the semantic embedding). "
                         "Only used with --backbone swin_small.")
    ap.add_argument('--swin_lr_factor', type=float, default=0.1,
                    help="LR multiplier applied to the Swin BACKBONE params only "
                         "(heads/BNNeck/OVLI stay at full --lr). The resnet50-tuned "
                         "peak LR (3.5e-4) diverges the Swin transformer (collapse at "
                         "epoch ~8); 0.1 fine-tunes the backbone gently. Set 1.0 to "
                         "disable the split. Only used with --backbone swin_small.")
    ap.add_argument('--use_afd', action='store_true')
    ap.add_argument('--afd_router', type=int, default=1)
    ap.add_argument('--afd_cvfc', type=int, default=1)
    ap.add_argument('--afd_stage', default='layer1',
                    choices=['stem', 'layer1', 'layer2'])
    ap.add_argument('--router_cond_view', type=int, default=1)
    ap.add_argument('--low_r', type=float, default=0.125)
    ap.add_argument('--mid_r', type=float, default=0.30)
    ap.add_argument('--high_drop_p', type=float, default=0.5)
    ap.add_argument('--w_cvfc', type=float, default=0.5)
    # --- OVP-Mem ---
    ap.add_argument('--ovp', action='store_true',
                    help='enable OVP-Mem opposite-view prototype InfoNCE loss')
    ap.add_argument('--ovp_lambda', type=float, default=0.5,
                    help='weight of the OVP InfoNCE loss')
    ap.add_argument('--ovp_tau', type=float, default=0.05,
                    help='temperature for the OVP InfoNCE logits')
    ap.add_argument('--ovp_momentum', type=float, default=0.2,
                    help='EMA momentum = weight on the new batch mean')
    ap.add_argument('--ovp_warmup', type=int, default=10,
                    help='H1 fix: warmup OVP lambda linearly over this many epochs')
    # --- OVLI (headline: opposite-view late-interaction retrieval) ---
    ap.add_argument('--ovli', action='store_true',
                    help='enable OVLI opposite-view late-interaction retrieval loss')
    ap.add_argument('--ovli_lambda', type=float, default=0.5,
                    help='weight of the OVLI retrieval loss')
    ap.add_argument('--ovli_tau', type=float, default=0.05,
                    help='temperature for the OVLI supervised-contrastive logits')
    ap.add_argument('--ovli_alpha', type=float, default=0.5,
                    help='score = alpha*cos(global) + (1-alpha)*sym_MaxSim(tokens)')
    ap.add_argument('--ovli_dim', type=int, default=256,
                    help='token projection output dim (new learnable params)')
    ap.add_argument('--ovli_grid', type=int, nargs=2, default=[8, 4],
                    help='adaptive-pool token grid (gh gw); K = gh*gw tokens')
    ap.add_argument('--ovli_warmup', type=int, default=10,
                    help='H1 lesson: warmup OVLI lambda linearly over this many epochs')
    ap.add_argument('--ovli_rerank', action='store_true',
                    help='additionally report global+MaxSim rerank at eval time')
    # MaxSim pooling variant (dustbin / sparse evidence routing). 'mean' = the
    # original behaviour (average over ALL token-max scores, back-compatible).
    ap.add_argument('--ovli_pool', default='mean',
                    choices=['mean', 'topk', 'thresh', 'softtopk'],
                    help="MaxSim pooling over per-token max scores: mean (all "
                         "tokens, original), topk (avg of top-k highest -> drop "
                         "non-corresponding tokens = dustbin approx), thresh "
                         "(avg of token-max > theta, fall back to single max), "
                         "softtopk (softmax(tau)-weighted mean = smooth top-k)")
    ap.add_argument('--ovli_topk', type=int, default=8,
                    help='k for --ovli_pool topk (clamped to [1, K] tokens)')
    ap.add_argument('--ovli_thresh', type=float, default=0.0,
                    help='theta for --ovli_pool thresh (token-max score floor)')
    # Ablation control for the headline opposite-view-only claim.
    ap.add_argument('--ovli_allview', action='store_true',
                    help='ABLATION: drop the opposite-view-only constraint in '
                         'the OVLI loss -> candidates become ALL other samples '
                         '(positives = same-pid any view excl. self, negatives '
                         '= other-pid any view) = plain all-view token-set '
                         'supervised-contrastive. Default OFF reproduces the '
                         'headline opposite-view-only behaviour exactly. Tests '
                         'whether the cross-view restriction (not just an extra '
                         'token loss) is what helps. score/MaxSim/pool/tau/'
                         'warmup/proj are unchanged.')
    # Ablation 1: late-interaction token-match reduction (max vs avg).
    ap.add_argument('--ovli_match', default='maxsim', choices=['maxsim', 'avg'],
                    help="ABLATION: token-token match reduction. maxsim (default) "
                         "= for each query token take the MAX similarity over the "
                         "other token set (ColBERT/late-interaction selection, "
                         "original). avg = replace that max with a MEAN over the "
                         "other token set -> the token-token similarities are "
                         "fully averaged = near-global soft match. Isolates "
                         "whether the MAX selection is what makes late interaction "
                         "work. Only the inner token reduction changes; "
                         "bidirectional/pool/alpha/loss are unchanged.")
    # Ablation 2: late-interaction spatial alignment (free vs ordered/AlignedReID).
    ap.add_argument('--ovli_align', default='free', choices=['free', 'ordered'],
                    help="ABLATION: late-interaction spatial alignment. free "
                         "(default) = each query token may match ANY other token "
                         "(free/global late interaction, original). ordered = "
                         "AlignedReID-style row-ordered alignment: a query token "
                         "in grid row r may only match other-set tokens in the "
                         "SAME row r (row-correspondence / simplified monotonic "
                         "cut). Isolates free partial set matching vs ordered "
                         "body-region alignment. Only the inner token reduction "
                         "changes; bidirectional/pool/alpha/loss are unchanged.")
    # Headline aggregation: learnable permutation-invariant SET POOLING of the K
    # projected tokens into one per-sample vector (the cross-view score is then
    # the cosine gram of those vectors).  This REPLACES the fixed "mean over
    # tokens" that --ovli_match avg --ovli_pool mean reduces to (the current best
    # single mechanism, 52.37).  'mean' (default) keeps the existing token-set
    # MaxSim path verbatim (byte-identical); the four learnable modes BYPASS the
    # match/pool/align/topk/thresh MaxSim knobs.
    ap.add_argument('--ovli_setpool', default='mean',
                    choices=['mean', 'netvlad', 'attn', 'gated', 'secondorder'],
                    help="learnable permutation-invariant aggregation of the K "
                         "tokens into one vector. mean (default) = keep the "
                         "token-set MaxSim path unchanged (byte-identical; "
                         "--ovli_match avg --ovli_pool mean = gram of mean-pooled "
                         "tokens = best). netvlad = NetVLAD residual aggregation; "
                         "attn = multi-head learned-query attention pooling; "
                         "gated = per-token sigmoid reliability gate convex mean; "
                         "secondorder = low-rank token covariance pooling. The "
                         "four learnable modes replace the MaxSim entirely (match/"
                         "pool/align/topk/thresh are bypassed) and add new params "
                         "that ARE optimized (assert self-check at startup).")
    ap.add_argument('--ovli_vlad_clusters', type=int, default=8,
                    help='C learnable clusters for --ovli_setpool netvlad')
    ap.add_argument('--ovli_attn_heads', type=int, default=4,
                    help='heads for --ovli_setpool attn (must divide --ovli_dim)')
    ap.add_argument('--ovli_so_rank', type=int, default=32,
                    help='low-rank dim r for --ovli_setpool secondorder (r x r cov)')
    # "mean + zero-init residual" toggle for the learnable set pools.  Default 1
    # (True): each pool = mean_k(tok) + zero-init_gate * residual, so it starts
    # BYTE-IDENTICAL to the 52.37 mean-pool and only learns a correction (fixes
    # the random-init standalone collapse: netvlad standalone ep20 14.66 << 52.37).
    # 0 (False): the original STANDALONE pooling (random init fully replaces the
    # mean) -- kept ONLY for the standalone-vs-residual ablation, expected to
    # collapse.  Ignored when --ovli_setpool mean (no learnable pool at all).
    ap.add_argument('--ovli_setpool_residual', type=int, default=1,
                    choices=[0, 1],
                    help='1 (default): learnable set pool = mean + zero-init '
                         'residual (lossless start from the 52.37 mean-pool, only '
                         'learns a correction). 0: original standalone pooling '
                         '(random init replaces the mean -> the collapsing '
                         'control, ablation only). No effect with --ovli_setpool '
                         'mean.')
    # --- ACVP (Ambiguity-Calibrated opposite-View negative relaxation) ---
    # Detached opposite-view-prototype ambiguity SENSOR that softens unreliable
    # NEGATIVES in the OVLI cross-view contrastive denominator. NO prototype-
    # positive alignment (avoids OVP/CMPC/PDPA overlap): the prototype bank is read
    # detached only; ACVP adds no learnable param and injects no gradient.  Default
    # OFF -> the OVLI loss path is byte-identical to the current (pre-ACVP) one.
    ap.add_argument('--acvp', action='store_true',
                    help='enable ACVP: detached opposite-view-prototype ambiguity '
                         'softening of UNRELIABLE NEGATIVES in the OVLI contrastive '
                         'denominator (no prototype-positive alignment, no new '
                         'learnable params, detached). Requires --ovli. Default OFF '
                         'reproduces the OVLI loss byte-for-byte.')
    ap.add_argument('--acvp_gamma', type=float, default=0.5,
                    help='ACVP max softening strength: w_ij = clamp(1 - gamma*'
                         'sigmoid((delta-margin)/eta), wmin, 1). gamma=0 disables '
                         'softening even if --acvp is set.')
    ap.add_argument('--acvp_wmin', type=float, default=0.3,
                    help='ACVP floor on the negative weight w_ij (>0 so log(w_ij) '
                         'is finite; never fully removes a negative).')
    ap.add_argument('--acvp_eta', type=float, default=0.05,
                    help='ACVP sigmoid temperature on the ambiguity delta.')
    ap.add_argument('--acvp_margin', type=float, default=0.0,
                    help='ACVP ambiguity margin: only delta>margin softens.')
    ap.add_argument('--acvp_warmup', type=int, default=10,
                    help='ACVP linear gamma warmup over this many epochs (ramp 0 -> '
                         'acvp_gamma) so early, noisy prototypes do not mis-soften.')
    # --- AIRL (Aerial Identity Recoverability Learning) -- resolution-degradation
    # consistency.  Default OFF -> the baseline trains byte-for-byte (no degrade,
    # no extra forward, no loss).  NO learnable params (degrade = augmentation,
    # consistency = loss); the optimizer is untouched.  See airl_degrade /
    # airl_consistency_loss above.  Independent of OVP/OVLI/ACVP (can co-run, but
    # the headline AIRL run is --airl alone on the plain baseline).
    ap.add_argument('--airl', action='store_true',
                    help='enable AIRL: per-image resolution degradation (to a '
                         'sampled aerial-scale pixel budget) + original/degraded '
                         'prediction-consistency loss. NO learnable params, TRAIN-'
                         'time only, eval unchanged. Default OFF reproduces the '
                         'baseline byte-for-byte.')
    ap.add_argument('--airl_lambda', type=float, default=0.5,
                    help='weight of the AIRL consistency loss '
                         '(total = CE + triplet + airl_lambda_eff * consistency).')
    ap.add_argument('--airl_min_scale', type=float, default=0.25,
                    help='lowest degradation scale ratio (per-image s ~ U[min_scale,'
                         ' 1]); s*H x s*W is the down-sampled pixel budget before '
                         'up-sampling back. 0.25 ~ the aerial small bucket (aerial '
                         'median bbox ~1/3 ground). Must be in (0,1].')
    ap.add_argument('--airl_consistency', default='kl', choices=['kl', 'feat'],
                    help="consistency target: kl (default) = temperature-scaled "
                         "soft-target KL on the ID logits (clean detached); feat = "
                         "1 - cosine on the L2-normed BNNeck feature.")
    ap.add_argument('--airl_tau', type=float, default=4.0,
                    help='softmax temperature for --airl_consistency kl (Hinton '
                         'distillation; loss scaled by tau^2). Ignored for feat.')
    ap.add_argument('--airl_blur', action='store_true',
                    help='additionally apply a light 3x3 avg-pool blur after the '
                         'up-sample (UAV optical-blur proxy; no extra params).')
    ap.add_argument('--airl_warmup', type=int, default=5,
                    help='linear AIRL lambda warmup over this many epochs (ramp 0 -> '
                         'airl_lambda) so the consistency term opens gently.')
    # --- AIRL dual-branch (resolvability branch): the COMPLETE AIRL mechanism for
    # a single-model, single-forward score fusion.  Adds a SECOND BNNeck head
    # (f_rec) on the shared backbone: f_full keeps full-resolution identity
    # evidence (protects G->A), f_rec gets its own ID-CE PLUS the AIRL
    # ground-degradation consistency (learns low-pixel-budget recoverable
    # evidence, serves A->G).  At eval the two heads' cosine scores are
    # SOFT-fused at the distance-matrix level:
    #     cos = airl_fuse_w * cos(f_rec) + (1 - airl_fuse_w) * cos(f_full)
    # with a SINGLE FIXED global w (a prior, NOT tuned on the test set, NOT a
    # per-query gate) -> this internalises the kill-switch #3 two-model score
    # fusion (+1.46 mean @ w=0.25) into ONE forward (both heads share the
    # backbone).  Framing (pinned to avoid the RAR/MRJL resolution-adaptive /
    # query-routing collision): "observation-limited evidence ceiling under which
    # a clean (f_full) and a recover (f_rec) evidence head DIVERGE, combined by a
    # FIXED-PRIOR soft fusion".  This is deliberately NOT query-budget routing --
    # kill-switch #3 showed hard per-query routing (area / reliability) fails to
    # recover the trade-off (<=+0.41), and the win comes from the fixed-w soft
    # blend; so we claim head divergence + fixed-prior fusion, not dynamic routing.
    # Default OFF -> the second head is never built and training/eval reproduce
    # the single-head baseline byte-for-byte.
    ap.add_argument('--airl_dualbranch', action='store_true',
                    help='enable the AIRL dual-branch (resolvability branch): a '
                         'second BNNeck head f_rec (own ID-CE + AIRL degradation '
                         'consistency) alongside the clean f_full head, soft-fused '
                         'at eval (cos = w*cos_rec + (1-w)*cos_full). One forward, '
                         'two features. Default OFF reproduces the baseline.')
    ap.add_argument('--airl_fuse_w', type=float, default=0.25,
                    help='fixed global fusion weight on the f_rec cosine at eval '
                         '(cos = airl_fuse_w*cos_rec + (1-airl_fuse_w)*cos_full); '
                         '0.25 = the legal default from kill-switch #3 (plateau '
                         'w in [0.25,0.75] all >= +1.46 mean). Must be in [0,1]. '
                         'NOT tuned on test (train/test symmetric). ABLATION-ONLY: '
                         'the headline is FIXED at 0.25; non-default w is for the '
                         'w-sweep ablation only (a warning prints if changed).')
    ap.add_argument('--airl_dualbranch_iso', action='store_true',
                    help='gradient-ISOLATED AIRL dual-branch (rescue of the failed '
                         'fully-shared --airl_dualbranch): f_rec is a BNNeck over an '
                         'INDEPENDENT late Swin stage forked off the shared trunk at '
                         'iso_stage (not the shared global_feat). The degradation-'
                         'CONSISTENCY gradient updates ONLY the rec late stage + '
                         'BNNeck_rec and NEVER flows back into the shared trunk (the '
                         'degraded pass forks off a DETACHED trunk), so f_rec stays a '
                         '"recover expert" and the +0.06 collapse (shared trunk pulled '
                         'toward degradation-robustness) is avoided. The CLEAN f_rec '
                         'ID-CE routing is governed by --airl_iso_trunk_recce: default '
                         '1 (the FIX) REFLOWS it into the trunk (extra identity '
                         'supervision -> strengthens the otherwise-weak f_full); 0 = '
                         'original full-isolation (clean ID-CE also detached). '
                         'swin_small only. Same eval soft-fusion + consistency '
                         'contract as --airl_dualbranch (shares its AIRL hyperparams '
                         '+ --airl_fuse_w). Default OFF reproduces the baseline.')
    ap.add_argument('--airl_iso_stage', type=int, default=3,
                    help='Swin stage index the f_rec branch forks AFTER (the rec '
                         'branch re-runs stages [iso_stage..last] on its own deep-'
                         'copied weights fed by the DETACHED trunk stream at the '
                         'input of this stage). swin_small has 4 stages (0..3); '
                         'iso_stage=3 (default) = share stages 0-2, split ONLY the '
                         'last stage (MGN-style, cheapest); iso_stage=2 = split the '
                         'last two stages (more f_rec divergence capacity, heavier). '
                         'Must be in [1,3]. Only used with --airl_dualbranch_iso.')
    # The trunk-undersupervision FIX (codex consensus).  The original full-detach
    # iso left f_full WEAK (ep20 45.56 < baseline 48.98 < even fully-shared f_rec
    # 47.39): f_rec's clean ID-CE only updated the DETACHED rec tail, so the shared
    # trunk lost the extra identity supervision the fully-shared dual-branch's trunk
    # got from BOTH heads' ID-CE.  --airl_iso_trunk_recce 1 (default) re-routes ONLY
    # the CLEAN rec ID-CE gradient back into the shared trunk (extra identity
    # supervision -> strengthens f_full) while keeping the degradation-CONSISTENCY
    # gradient detached from the trunk (so f_rec stays a specialised recover pole --
    # the isolation that the iso variant exists for).  0 = the ORIGINAL full-isolation
    # iso (clean ID-CE also detached), kept for the controlled ablation.  Only used
    # with --airl_dualbranch_iso.
    ap.add_argument('--airl_iso_trunk_recce', type=int, default=1, choices=[0, 1],
                    help='1 (default, the FIX): route the CLEAN f_rec ID-CE gradient '
                         'back into the shared trunk (extra identity supervision -> '
                         'strengthens the weak f_full); the degradation-consistency '
                         'gradient stays DETACHED from the trunk (f_rec stays '
                         'specialised). 0: original full-isolation iso (clean ID-CE '
                         'also detached from the trunk), ablation only. No effect '
                         'without --airl_dualbranch_iso.')
    args = ap.parse_args()
    args.afd_router = bool(args.afd_router)
    args.afd_cvfc = bool(args.afd_cvfc)
    args.router_cond_view = bool(args.router_cond_view)
    args.ovli_setpool_residual = bool(args.ovli_setpool_residual)
    args.airl_iso_trunk_recce = bool(args.airl_iso_trunk_recce)
    # backbone guard: the AFD frequency modules (router/cvfc) insert at resnet
    # shallow stages that don't exist in Swin -> --use_afd is incompatible with
    # --backbone swin_small (caught in AFDModel too, but fail fast at parse time).
    if args.backbone == 'swin_small' and args.use_afd:
        ap.error("--backbone swin_small does not support --use_afd (AFD modules "
                 "insert at resnet shallow stages). OVP/OVLI work on swin; drop "
                 "--use_afd.")
    # OVP and OVLI are two distinct cross-view mechanisms (prototype-memory
    # InfoNCE vs sample-to-sample late-interaction retrieval).  All three modes
    # are supported and back-compatible:
    #   OVP-only   (--ovp)         : empirical prototype auxiliary
    #   OVLI-only  (--ovli)        : headline late-interaction retrieval
    #   both       (--ovp --ovli)  : complementarity test -- each loss keeps its
    #                                own warmup / lambda / diagnostics and is
    #                                added to the same total; OVP adds no params,
    #                                OVLI's proj is the only extra optimized set.
    # total = CE + triplet + ovp_lam_eff*OVP + ovli_lam_eff*OVLI (terms that are
    # off contribute exactly 0, so OVP-only / OVLI-only reproduce as before).
    # ACVP is a pure calibration ON TOP of the OVLI loss (softens unreliable
    # negatives in the OVLI denominator via a detached opposite-view prototype
    # ambiguity sensor); it has no loss term of its own and requires --ovli.
    if args.acvp and not args.ovli:
        ap.error("--acvp requires --ovli (it calibrates the OVLI contrastive "
                 "negatives; there is no standalone ACVP loss).")
    # ACVP is "opposite-view negative relaxation": it only makes sense when the
    # OVLI candidate set IS opposite-view-only.  Under --ovli_allview the negatives
    # include same-view pairs, which contradicts the mechanism's wording, so we
    # forbid the combination outright (cleaner than silently calibrating all-view).
    if args.acvp and args.ovli_allview:
        ap.error("--acvp is opposite-view negative relaxation and is incompatible "
                 "with --ovli_allview (which adds same-view negatives). Drop one.")
    # ACVP numeric-safety guards: bad CLI values would make w_ij / log(w_ij)
    # produce inf/NaN.  Enforce wmin in (0,1], eta>0, gamma>=0 at parse time so a
    # typo fails fast instead of corrupting the loss mid-training.
    if args.acvp:
        if not (args.acvp_wmin > 0.0 and args.acvp_wmin <= 1.0):
            ap.error("--acvp_wmin must be in (0,1] (w_ij floor; >0 so log(w) is "
                     f"finite, <=1 since w_ij<=1); got {args.acvp_wmin}.")
        if not (args.acvp_eta > 0.0):
            ap.error("--acvp_eta must be > 0 (sigmoid temperature; 0 -> div-by-0); "
                     f"got {args.acvp_eta}.")
        if not (args.acvp_gamma >= 0.0):
            ap.error("--acvp_gamma must be >= 0 (softening strength; <0 would "
                     f"AMPLIFY negatives); got {args.acvp_gamma}.")

    # AIRL numeric-safety guard: min_scale in (0,1] so the down-sampled budget is a
    # real fraction of the input (>0) and never upscales (<=1); a typo fails fast.
    if args.airl and not (args.airl_min_scale > 0.0 and args.airl_min_scale <= 1.0):
        ap.error("--airl_min_scale must be in (0,1] (per-image scale ratio s in "
                 f"[min_scale,1]); got {args.airl_min_scale}.")
    if args.airl and not (args.airl_tau > 0.0):
        ap.error(f"--airl_tau must be > 0 (softmax temperature); got {args.airl_tau}.")

    # AIRL dual-branch guards.  --airl (single-head consistency) and
    # --airl_dualbranch (two-head, consistency on f_rec only) are two DIFFERENT
    # AIRL instantiations of the SAME degrade+consistency primitive; running both
    # would apply consistency twice (to the single head AND to f_rec) and muddy
    # the ablation, so they are mutually exclusive.  The dual-branch shares
    # --airl_lambda / --airl_min_scale / --airl_consistency / --airl_tau /
    # --airl_blur / --airl_warmup (the consistency on f_rec is the SAME function),
    # and they are validated the same way (so a stray bad --airl_min_scale with
    # only --airl_dualbranch still fails fast).
    if args.airl_dualbranch:
        if args.airl:
            ap.error("--airl_dualbranch and --airl are mutually exclusive (both "
                     "apply the AIRL degradation-consistency; dual-branch applies "
                     "it to the f_rec head only). Pick one.")
        if not (args.airl_min_scale > 0.0 and args.airl_min_scale <= 1.0):
            ap.error("--airl_min_scale must be in (0,1] (used by --airl_dualbranch "
                     f"too); got {args.airl_min_scale}.")
        if not (args.airl_tau > 0.0):
            ap.error("--airl_tau must be > 0 (used by --airl_dualbranch too); got "
                     f"{args.airl_tau}.")
        if not (0.0 <= args.airl_fuse_w <= 1.0):
            ap.error("--airl_fuse_w must be in [0,1] (eval fusion weight cos = "
                     f"w*cos_rec + (1-w)*cos_full); got {args.airl_fuse_w}.")
        # w-lock (soft): the headline fixed-prior fusion uses w=0.25.  A non-default
        # w is ABLATION-ONLY (the w-sweep), so warn rather than assert -- the sweep
        # still needs to pass other values -- but make any deviation from the
        # headline visible in the log so a stray w never silently becomes "the
        # result".
        if args.airl_fuse_w != 0.25:
            print(f"[AIRL-DUAL][WARN] --airl_fuse_w={args.airl_fuse_w} != 0.25: the "
                  "headline uses the FIXED prior w=0.25; non-default w is "
                  "ABLATION-ONLY (w-sweep), not the headline result.")
        # The dual-branch is the standalone headline AIRL mechanism; keep its
        # ablation clean by forbidding co-running the cross-view OVP/OVLI losses
        # (they target a different gap and would confound the f_rec specialisation).
        if args.ovp or args.ovli:
            ap.error("--airl_dualbranch is run standalone (headline AIRL); do not "
                     "combine with --ovp/--ovli (separate cross-view mechanisms).")

    # AIRL gradient-isolated dual-branch guards.  This is the RESCUE variant: the
    # SAME degrade+consistency+soft-fusion contract as --airl_dualbranch, but f_rec
    # forks off a DETACHED trunk into an independent late Swin stage (so the
    # consistency gradient cannot pollute the shared trunk).  It therefore:
    #   * shares the AIRL hyperparams (--airl_lambda/min_scale/consistency/tau/blur/
    #     warmup) and --airl_fuse_w, validated identically;
    #   * is mutually exclusive with BOTH --airl (single-head) and --airl_dualbranch
    #     (fully-shared) -- three distinct AIRL instantiations, one at a time;
    #   * is swin_small-only (the fork lives in the Swin stage list);
    #   * runs standalone (no OVP/OVLI), same as --airl_dualbranch.
    if args.airl_dualbranch_iso:
        if args.airl or args.airl_dualbranch:
            ap.error("--airl_dualbranch_iso is mutually exclusive with --airl and "
                     "--airl_dualbranch (three distinct AIRL instantiations; the "
                     "iso variant forks an independent late stage off a detached "
                     "trunk). Pick one.")
        if args.backbone != 'swin_small':
            ap.error("--airl_dualbranch_iso requires --backbone swin_small (the rec "
                     "branch forks an independent Swin late stage).")
        if not (1 <= args.airl_iso_stage <= 3):
            ap.error("--airl_iso_stage must be in [1,3] (swin_small has 4 stages "
                     "0..3; fork after a shared early stage, before the last); got "
                     f"{args.airl_iso_stage}.")
        if not (args.airl_min_scale > 0.0 and args.airl_min_scale <= 1.0):
            ap.error("--airl_min_scale must be in (0,1] (used by "
                     f"--airl_dualbranch_iso too); got {args.airl_min_scale}.")
        if not (args.airl_tau > 0.0):
            ap.error("--airl_tau must be > 0 (used by --airl_dualbranch_iso too); "
                     f"got {args.airl_tau}.")
        if not (0.0 <= args.airl_fuse_w <= 1.0):
            ap.error("--airl_fuse_w must be in [0,1] (eval fusion weight cos = "
                     f"w*cos_rec + (1-w)*cos_full); got {args.airl_fuse_w}.")
        if args.airl_fuse_w != 0.25:
            print(f"[AIRL-ISO][WARN] --airl_fuse_w={args.airl_fuse_w} != 0.25: the "
                  "headline uses the FIXED prior w=0.25; non-default w is "
                  "ABLATION-ONLY (w-sweep), not the headline result.")
        if args.ovp or args.ovli:
            ap.error("--airl_dualbranch_iso is run standalone (headline AIRL); do "
                     "not combine with --ovp/--ovli (separate cross-view "
                     "mechanisms).")

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda'
    batch_size = args.P * args.K

    print("=" * 70)
    print("CVPB-ReID training (afd_reid baseline + OVP-Mem / OVLI)")
    if args.backbone == 'swin_small':
        print(f"  backbone=swin_small (SOLIDER, in_planes=768) "
              f"pretrain={args.swin_pretrain or 'NONE (from scratch)'} "
              f"semantic_weight={args.swin_semantic_weight}")
    else:
        print(f"  backbone=resnet50 (BoT baseline) pool={args.pool} "
              f"last_stride={args.last_stride}")
    print(f"  use_afd={args.use_afd}  ovp={args.ovp} "
          f"(lambda={args.ovp_lambda} tau={args.ovp_tau} mom={args.ovp_momentum})")
    print(f"  ovli={args.ovli} (lambda={args.ovli_lambda} tau={args.ovli_tau} "
          f"alpha={args.ovli_alpha} dim={args.ovli_dim} grid={tuple(args.ovli_grid)} "
          f"warmup={args.ovli_warmup} rerank={args.ovli_rerank} "
          f"pool={args.ovli_pool} topk={args.ovli_topk} thresh={args.ovli_thresh} "
          f"cand={'allview' if args.ovli_allview else 'oppview'} "
          f"match={args.ovli_match} align={args.ovli_align} "
          f"setpool={args.ovli_setpool} "
          f"setpool_residual={args.ovli_setpool_residual})")
    print(f"  acvp={args.acvp} (gamma={args.acvp_gamma} wmin={args.acvp_wmin} "
          f"eta={args.acvp_eta} margin={args.acvp_margin} "
          f"warmup={args.acvp_warmup}) [detached neg-relaxation on OVLI; "
          f"off => OVLI byte-identical]")
    print(f"  airl={args.airl} (lambda={args.airl_lambda} "
          f"min_scale={args.airl_min_scale} consistency={args.airl_consistency} "
          f"tau={args.airl_tau} blur={args.airl_blur} warmup={args.airl_warmup}) "
          f"[resolution-degradation consistency; NO learnable params; train-only; "
          f"off => baseline byte-identical]")
    print(f"  airl_dualbranch={args.airl_dualbranch} (fuse_w={args.airl_fuse_w} "
          f"lambda={args.airl_lambda} min_scale={args.airl_min_scale} "
          f"consistency={args.airl_consistency} tau={args.airl_tau} "
          f"blur={args.airl_blur} warmup={args.airl_warmup}) "
          f"[resolvability branch: 2nd BNNeck head f_rec (own ID-CE + AIRL "
          f"consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; "
          f"1 forward 2 features; off => baseline byte-identical]")
    print(f"  airl_dualbranch_iso={args.airl_dualbranch_iso} "
          f"(iso_stage={args.airl_iso_stage} trunk_recce={args.airl_iso_trunk_recce} "
          f"fuse_w={args.airl_fuse_w} "
          f"lambda={args.airl_lambda} min_scale={args.airl_min_scale} "
          f"consistency={args.airl_consistency} tau={args.airl_tau} "
          f"blur={args.airl_blur} warmup={args.airl_warmup}) "
          f"[GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late "
          f"Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared "
          f"trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the "
          f"CLEAN f_rec ID-CE into the trunk (extra identity supervision -> "
          f"strengthens f_full); trunk_recce=0 = original full-isolation (clean "
          f"ID-CE also detached). f_rec learns the recover pole; same soft-fusion "
          f"eval; off => baseline byte-identical]")
    print(f"  bs={batch_size} (P={args.P} K={args.K}) lr={args.lr} "
          f"epochs={args.epochs} warmup={args.warmup_epochs} amp={not args.no_amp}")
    print(f"  out_dir={args.out_dir}")
    print("=" * 70)

    # data
    if args.dataset == 'cargo':
        dataset = CARGO(root=args.data_root, verbose=True)
    elif args.dataset == 'agreid_v2':
        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
        # so run_cross_view_eval/print_eval report the official per-protocol mAP
        # and their mean with no change to the eval / AIRL-iso code.
        dataset = AGReIDV2Combined(root=args.data_root, verbose=True)
    else:
        dataset = AGReIDv2(root=args.data_root, verbose=True)
    train_tf = build_transforms(is_train=True, img_size=tuple(args.img_size))
    train_set = CARGOImageDataset(dataset.train, train_tf)
    sampler = RandomIdentitySampler(dataset.train, batch_size, args.K)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True)

    # model
    model = build_model(dataset.num_train_pids, args).to(device)

    # losses
    ce = CrossEntropyLabelSmooth(dataset.num_train_pids, args.label_smooth)
    tri = TripletLoss(args.margin)

    # OVP memory (buffers live on the model device; not optimized)
    ovp = None
    if args.ovp:
        ovp = OVPMemory(dataset.num_train_pids, model.in_planes,
                        momentum=args.ovp_momentum, tau=args.ovp_tau).to(device)

    # OVLI head: token projection (NEW learnable params) + hook on model.layer4.
    # The projection MUST be optimized -> add ovli.parameters() to the optimizer.
    ovli = None
    if args.ovli:
        ovli = OVLIHead(model, in_ch=model.in_planes, proj_dim=args.ovli_dim,
                        grid=tuple(args.ovli_grid), alpha=args.ovli_alpha,
                        tau=args.ovli_tau, pool=args.ovli_pool,
                        topk=args.ovli_topk, thresh=args.ovli_thresh,
                        allview=args.ovli_allview,
                        match=args.ovli_match, align=args.ovli_align,
                        setpool=args.ovli_setpool,
                        vlad_clusters=args.ovli_vlad_clusters,
                        attn_heads=args.ovli_attn_heads,
                        so_rank=args.ovli_so_rank,
                        setpool_residual=args.ovli_setpool_residual).to(device)

    # ACVP prototype bank: a DEDICATED, detached opposite-view EMA prototype bank
    # (its own OVPMemory instance, independent of --ovp so the two never share or
    # double-update a buffer).  ACVP only READS this bank (detached) to compute the
    # ambiguity weight that softens unreliable negatives in the OVLI denominator;
    # it never runs an InfoNCE on it (no prototype-positive alignment) and adds no
    # learnable param -> the bank stays out of the optimizer.  Built ONLY when
    # --acvp is set, so the no-ACVP path constructs no bank at all (off-mode is
    # structurally identical to the current code).
    acvp_mem = None
    if args.acvp:
        acvp_mem = OVPMemory(dataset.num_train_pids, model.in_planes,
                             momentum=args.ovp_momentum, tau=args.ovp_tau).to(device)

    # optimizer -- include the OVLI projection params (model has none of them).
    # AdamW(model.parameters()) alone would silently SKIP the OVLI proj, so when
    # --ovli is on we pass list(model.parameters()) + list(ovli.parameters()).
    #
    # Swin backbone fine-tuning LR: the resnet50-tuned peak LR (3.5e-4 AdamW) is
    # SAFE for resnet50 but DIVERGES the ~50M-param SOLIDER Swin transformer --
    # cvpb_swin_ovli trained healthily for 7 epochs (Acc 0.47) then COLLAPSED at
    # epoch 8 the moment the warmup pushed LR past ~2.5e-4: the backbone fell into
    # a constant-output fixed point (last-stage map off-diag cos +0.99 -> all
    # images map to ~one vector -> cross-view mAP 0.03).  No NaN; a genuine
    # optimization collapse.  Transformer ReID is fine-tuned at a much smaller
    # backbone LR than the randomly-initialised heads (the repo's main SOLIDER
    # config likewise uses a gentle schedule).  So for backbone='swin_small' we
    # scale ONLY the Swin backbone params by swin_lr_factor (default 0.1) and keep
    # the BNNeck / classifier / OVLI proj at the full LR (they are random-init and
    # must learn fast).  resnet50 is untouched (no backbone_swin -> single group,
    # byte-identical to before).
    swin_lr_factor = getattr(args, 'swin_lr_factor', 0.1)
    if model.backbone == 'swin_small' and swin_lr_factor != 1.0:
        swin_param_ids = {id(p) for p in model.backbone_swin.parameters()}
        swin_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) in swin_param_ids]
        other_params = [p for p in model.parameters()
                        if p.requires_grad and id(p) not in swin_param_ids]
        if ovli is not None:
            other_params = other_params + [p for p in ovli.parameters()
                                           if p.requires_grad]
        param_groups = [
            {'params': swin_params, 'lr': args.lr * swin_lr_factor},
            {'params': other_params, 'lr': args.lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr,
                                      weight_decay=args.weight_decay)
        print(f"  [swin] backbone LR = {args.lr * swin_lr_factor:.2e} "
              f"(= base {args.lr:.2e} x {swin_lr_factor}); "
              f"heads/BNNeck/OVLI LR = {args.lr:.2e}  "
              f"[{len(swin_params)} backbone tensors, {len(other_params)} head tensors] "
              f"-- prevents the epoch-8 Swin collapse")
    else:
        opt_params = list(model.parameters())
        if ovli is not None:
            opt_params = opt_params + list(ovli.parameters())
        optimizer = torch.optim.AdamW(opt_params, lr=args.lr,
                                      weight_decay=args.weight_decay)
    # self-check: confirm the OVLI projection params actually landed in the
    # optimizer (the key structural requirement vs OVP).
    if ovli is not None:
        opt_ids = {id(p) for grp in optimizer.param_groups for p in grp['params']}
        proj_in = all(id(p) in opt_ids for p in ovli.proj.parameters())
        n_proj = sum(p.numel() for p in ovli.proj.parameters())
        assert proj_in, "OVLI proj params NOT in optimizer!"
        print(f"  [OVLI] projection params in optimizer: {proj_in} "
              f"({n_proj} params, {sum(1 for _ in ovli.proj.parameters())} tensors)")
        # self-check: the learnable set-pool params (NetVLAD centers / attn query
        # / gate MLP / covariance proj) must ALSO land in the optimizer.
        if ovli.setpool_mod is not None:
            sp_params = list(ovli.setpool_mod.parameters())
            sp_in = all(id(p) in opt_ids for p in sp_params)
            n_sp = sum(p.numel() for p in sp_params)
            assert sp_in, "OVLI setpool params NOT in optimizer!"
            # the zero-init residual gate must also be optimized (it is what lets
            # the residual turn on after the lossless mean-pool start).
            gate_in = (ovli.setpool_mod.gate_res is None
                       or id(ovli.setpool_mod.gate_res) in opt_ids)
            assert gate_in, "OVLI setpool residual gate NOT in optimizer!"
            res_msg = ("mean + zero-init residual (lossless start from 52.37 "
                       "mean-pool, gate_res zero-init)"
                       if ovli.setpool_residual else
                       "STANDALONE (random init replaces mean; ablation/collapse)")
            print(f"  [OVLI] setpool='{ovli.setpool}' params in optimizer: {sp_in} "
                  f"({n_sp} params, {len(sp_params)} tensors); "
                  f"MaxSim match/pool/align BYPASSED (set-pool aggregation); "
                  f"mode={res_msg}")
        else:
            print("  [OVLI] setpool='mean' (token-set MaxSim path; "
                  "match/pool/align active)")
        print(f"  [OVLI] candidate view-mask: "
              + ("ALL-VIEW (ablation: opposite-view constraint OFF; positives = "
                 "same-pid any view, negatives = other-pid any view)"
                 if args.ovli_allview else
                 "OPPOSITE-VIEW-ONLY (headline cross-view supervision)"))
        print(f"  [OVLI] token-match: "
              + ("MAXSIM (ColBERT/late-interaction max selection, headline)"
                 if args.ovli_match == 'maxsim' else
                 "AVG (ablation: per-token MEAN over other tokens = soft global)")
              + " | align: "
              + ("FREE (free/global late interaction, headline)"
                 if args.ovli_align == 'free' else
                 "ORDERED (ablation: AlignedReID row-correspondence)"))
        # ACVP self-check: detached prototype sensor, NO learnable params -> it
        # must NOT introduce anything into the optimizer (the structural contract:
        # ACVP is a re-weighting, not a learned alignment).
        if acvp_mem is not None:
            acvp_buf_in_opt = any(id(b) in opt_ids for b in acvp_mem.buffers())
            assert not acvp_buf_in_opt, "ACVP prototype buffers leaked into optimizer!"
            print(f"  [ACVP] ON: detached opposite-view prototype ambiguity "
                  f"softening of OVLI negatives (gamma={args.acvp_gamma} "
                  f"wmin={args.acvp_wmin} eta={args.acvp_eta} "
                  f"margin={args.acvp_margin} warmup={args.acvp_warmup}); "
                  f"NO learnable params (buffers in optimizer: {acvp_buf_in_opt}); "
                  f"no prototype-positive alignment (read-only, detached)")
    # AIRL dual-branch self-check: the SECOND BNNeck head (bottleneck_rec +
    # classifier_rec) lives inside model.parameters(), so it is in the optimizer
    # automatically -- but assert it explicitly (it is the structural requirement:
    # f_rec must actually train, with its OWN params, at the FULL head LR even on
    # Swin where the backbone is at swin_lr_factor x LR).
    if args.airl_dualbranch:
        opt_ids = {id(p) for grp in optimizer.param_groups for p in grp['params']}
        rec_params = (list(model.bottleneck_rec.parameters())
                      + list(model.classifier_rec.parameters()))
        # bottleneck_rec.bias has requires_grad_=False (frozen, like f_full's BN
        # bias) -> AdamW (a no-arg param list) still RECEIVES it but never updates
        # it (zero grad); only assert the TRAINABLE rec params are present.
        rec_trainable = [p for p in rec_params if p.requires_grad]
        rec_in = all(id(p) in opt_ids for p in rec_trainable)
        assert rec_in, "AIRL dual-branch f_rec head params NOT in optimizer!"
        # on Swin, f_rec must be at the FULL head LR (not the backbone factor):
        # both rec params are random-init heads, identical to f_full's BNNeck.
        n_rec = sum(p.numel() for p in rec_trainable)
        print(f"  [AIRL-DUAL] f_rec head (bottleneck_rec + classifier_rec) params "
              f"in optimizer: {rec_in} ({n_rec} params, {len(rec_trainable)} "
              f"trainable tensors); eval soft-fusion cos=w*cos_rec+(1-w)*cos_full "
              f"w={args.airl_fuse_w}")
    # AIRL gradient-isolated dual-branch self-check: BNNeck_rec + classifier_rec are
    # random-init heads OUTSIDE backbone_swin -> FULL-LR group; the INDEPENDENT rec
    # late stage (rec_stages/rec_norm) lives INSIDE backbone_swin -> the scaled Swin
    # LR group (pretrained weights, same as f_full's stages).  Assert both placements
    # so a future param-group refactor cannot silently freeze or mis-LR the rec path.
    if args.airl_dualbranch_iso:
        opt_ids = {id(p) for grp in optimizer.param_groups for p in grp['params']}
        bsw = model.backbone_swin
        rec_head_params = [p for p in (list(model.bottleneck_rec.parameters())
                                       + list(model.classifier_rec.parameters()))
                           if p.requires_grad]
        rec_head_in = all(id(p) in opt_ids for p in rec_head_params)
        assert rec_head_in, "AIRL-ISO f_rec head params NOT in optimizer!"
        # rec late-stage trainable params (rec_stages + rec_norm; semantic-embed is
        # frozen so excluded) must ALL be in the optimizer and trainable.
        rec_stage_params = [p for p in (list(bsw.rec_stages.parameters())
                                        + list(bsw.rec_norm.parameters()))
                            if p.requires_grad]
        rec_stage_in = all(id(p) in opt_ids for p in rec_stage_params)
        assert rec_stage_in, "AIRL-ISO rec late-stage params NOT in optimizer!"
        # the rec late stage must be on the SCALED Swin LR group (it is pretrained
        # backbone weight, byte-identical recipe to f_full's stages).  Find which
        # group each rec-stage param landed in and confirm it is the swin group when
        # the swin split is active.
        if model.backbone == 'swin_small' and swin_lr_factor != 1.0:
            swin_grp_ids = {id(p) for p in param_groups[0]['params']}
            full_grp_ids = {id(p) for p in param_groups[1]['params']}
            rec_stage_in_swin = all(id(p) in swin_grp_ids for p in rec_stage_params)
            rec_head_in_full = all(id(p) in full_grp_ids for p in rec_head_params)
            assert rec_stage_in_swin, ("AIRL-ISO rec late stage NOT in the scaled "
                                       "Swin LR group (it is pretrained backbone "
                                       "weight)!")
            assert rec_head_in_full, ("AIRL-ISO rec BNNeck head NOT in the full-LR "
                                      "group (it is a random-init head)!")
            lr_msg = (f"rec late stage @ Swin LR {args.lr * swin_lr_factor:.2e}, "
                      f"rec BNNeck @ full LR {args.lr:.2e}")
        else:
            lr_msg = f"single LR group @ {args.lr:.2e}"
        n_rh = sum(p.numel() for p in rec_head_params)
        n_rs = sum(p.numel() for p in rec_stage_params)
        recce_msg = ("trunk_recce=1 (clean f_rec ID-CE REFLOWS to trunk; degraded "
                     "consistency stays detached)" if args.airl_iso_trunk_recce
                     else "trunk_recce=0 (clean ID-CE + consistency BOTH detached = "
                          "original full-isolation)")
        print(f"  [AIRL-ISO] iso_stage={args.airl_iso_stage}: rec late stage "
              f"({n_rs} params, {len(rec_stage_params)} tensors) + rec BNNeck head "
              f"({n_rh} params, {len(rec_head_params)} tensors) in optimizer "
              f"[{lr_msg}]; degradation-consistency grad isolated from shared trunk "
              f"(detached degraded pass at stage-{args.airl_iso_stage} input); "
              f"{recce_msg}; eval soft-fusion "
              f"cos=w*cos_rec+(1-w)*cos_full w={args.airl_fuse_w}")
    scheduler = WarmupCosineLR(optimizer, args.warmup_epochs, args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=not args.no_amp)

    view_map = {'Aerial': 0, 'Ground': 1}
    best_map = -1.0
    best_epoch = -1
    n_iter_total = len(train_loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if ovli is not None:
            ovli.train()
        t0 = time.time()
        # H1 fix: warmup OVP lambda over ovp_warmup epochs to avoid cold-start gradient spikes
        ovp_lambda_eff = (args.ovp_lambda * min(1.0, epoch / max(1, args.ovp_warmup))) if args.ovp else 0.0
        # H1 lesson: same linear warmup for OVLI (random proj -> avoid early spike)
        ovli_lambda_eff = (args.ovli_lambda * min(1.0, epoch / max(1, args.ovli_warmup))) if args.ovli else 0.0
        # ACVP: linear gamma warmup (ramp 0 -> acvp_gamma over acvp_warmup epochs)
        # so early, noisy prototypes do not aggressively soften negatives.
        acvp_gamma_eff = (args.acvp_gamma * min(1.0, epoch / max(1, args.acvp_warmup))) if args.acvp else 0.0
        # AIRL: linear lambda warmup (ramp 0 -> airl_lambda over airl_warmup epochs)
        # so the resolution-consistency term opens gently.  Shared by ALL THREE AIRL
        # instantiations (mutually exclusive): the single-head --airl, the fully-
        # shared dual-branch --airl_dualbranch, AND the gradient-isolated dual-branch
        # --airl_dualbranch_iso (same consistency function, same warmup).  MUST list
        # all three: the flags are mutually exclusive, so omitting iso here would
        # leave airl_lambda_eff==0 every epoch on an iso run and silently zero out
        # the f_rec consistency gradient (the whole mechanism being tested).
        airl_lambda_eff = (args.airl_lambda * min(1.0, epoch / max(1, args.airl_warmup))) \
            if (args.airl or args.airl_dualbranch or args.airl_dualbranch_iso) else 0.0
        meters = {'loss': 0.0, 'ce': 0.0, 'tri': 0.0, 'ovp': 0.0,
                  'ovli': 0.0, 'ovli_pos': 0.0, 'ovli_neg': 0.0, 'acc': 0.0,
                  'airl': 0.0, 'airl_scale': 0.0, 'airl_n_ground': 0.0,
                  'ce_rec': 0.0}
        # ACVP kill-switch accumulators (weighted by #softenable-neg per step;
        # steps with 0 softenable negatives are skipped, not counted).
        acvp_frac_sum = 0.0     # sum of relaxed_neg_frac * n_softenable_neg
        acvp_w_sum = 0.0        # sum of mean_w * n_softenable_neg
        acvp_steps = 0          # total #softenable-neg pairs ACVP acted on
        seen = 0

        for it, batch in enumerate(train_loader):
            imgs = batch['img'].to(device, non_blocking=True)
            labels = batch['pid'].to(device, non_blocking=True)
            views = torch.tensor([view_map[v] for v in batch['view']],
                                 device=device)
            vidx = views if args.use_afd else None

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=not args.no_amp):
                out = model(imgs, view_idx=vidx,
                            return_cvfc=(args.use_afd and args.afd_cvfc))
                logits = out['logits']
                gfeat = out['global_feat']
                bn = out['bn_feat']
                loss_ce = ce(logits, labels)
                loss_tri = tri(gfeat, labels)
                loss = loss_ce + loss_tri

                # AIRL dual-branch: the f_rec head needs its OWN identity grounding
                # so it is a valid discriminative space for the eval fusion (a head
                # trained on consistency alone would be unidentified).  Add f_rec's
                # ID cross-entropy (SAME label-smoothing CE as f_full); the global
                # triplet stays on the SHARED global_feat (NOT duplicated for f_rec).
                # The f_rec degradation-consistency is added below (fp32 block).
                loss_ce_rec = torch.zeros((), device=device)
                if args.airl_dualbranch or args.airl_dualbranch_iso:
                    # f_rec ID grounding.  For --airl_dualbranch f_rec reads the
                    # shared global_feat; for --airl_dualbranch_iso it reads the
                    # INDEPENDENT rec late-stage map.  Both expose logits_rec, so
                    # the CE call is identical -- only the gradient destination of
                    # this CLEAN ID-CE differs:
                    #   * --airl_dualbranch       -> the shared trunk (fully shared).
                    #   * --airl_dualbranch_iso, trunk_recce=1 (FIX) -> the shared
                    #     trunk TOO: model.forward ran the iso clean pass with a
                    #     NON-detached fork, so this clean ID-CE reflows into the
                    #     trunk (extra identity supervision -> strengthens f_full)
                    #     while the degradation-consistency below (rec_only, detached)
                    #     stays trunk-isolated.
                    #   * --airl_dualbranch_iso, trunk_recce=0 -> the isolated rec
                    #     stage only (original full-isolation: clean fork detached).
                    loss_ce_rec = ce(out['logits_rec'], labels)
                    loss = loss + loss_ce_rec

                loss_ovp = torch.zeros((), device=device)
                if args.ovp:
                    # OVP loss in fp32 for numerical safety (cosine + softmax)
                    z = F.normalize(bn.float(), dim=1)
                    loss_ovp = ovp.loss(z, labels, views)
                    loss = loss + ovp_lambda_eff * loss_ovp

            # OVLI: compute in TRUE fp32 (autocast disabled) -- the cos/MaxSim/
            # logsumexp at tau=0.05 want fp32, and running the proj here (after
            # the autocast forward already cached the fp16 layer4 map) keeps the
            # projection weights in fp32 while gradient still flows into layer4.
            loss_ovli = torch.zeros((), device=device)
            ovli_pos = torch.zeros((), device=device)
            ovli_neg = torch.zeros((), device=device)
            if args.ovli:
                with torch.amp.autocast('cuda', enabled=False):
                    # global feature for the score: normalized BN feat (matches
                    # the eval ranking space). gradient flows -> encoder.
                    g_ovli = F.normalize(bn.float(), dim=1)
                    tok = ovli.tokens_from_cached_map()          # (B,K,Dp) fp32
                    if args.acvp:
                        # ACVP ON: pass the DETACHED opposite-view prototype bank +
                        # the warmup-ramped gamma so the OVLI denominator softens
                        # unreliable negatives.  acvp_mem.bank/.inited are buffers
                        # (no grad); .detach() makes the no-grad contract explicit.
                        loss_ovli, ovli_pos, ovli_neg = ovli.loss(
                            g_ovli, tok, labels, views,
                            acvp_proto=acvp_mem.bank.detach(),
                            acvp_inited=acvp_mem.inited.detach(),
                            acvp_gamma=acvp_gamma_eff,
                            acvp_wmin=args.acvp_wmin,
                            acvp_eta=args.acvp_eta,
                            acvp_margin=args.acvp_margin)
                    else:
                        # ACVP OFF: byte-identical original 4-arg call -> the loss
                        # body never touches the ACVP branch (acvp_proto is None).
                        loss_ovli, ovli_pos, ovli_neg = ovli.loss(
                            g_ovli, tok, labels, views)
                loss = loss + ovli_lambda_eff * loss_ovli

            # AIRL: resolution-degradation consistency.  ASYMMETRIC by design --
            # degrade ONLY the high-resolution GROUND view (views==1; Aerial==0) to
            # a sampled aerial-scale pixel budget, run ONE extra forward through the
            # SAME model (shared weights), and pull the degraded GROUND prediction
            # toward its own (detached) clean one.  The hypothesis is "recover
            # ground identity at an aerial pixel budget"; degrading the already
            # low-budget aerial samples would just be all-view self-degradation and
            # break that asymmetry, so aerial rows are NOT degraded.  No learnable
            # params; train-time only.  Empty-ground batch -> loss_airl=0 (no extra
            # forward).  OFF (default) -> this whole block is skipped (no degrade,
            # no extra forward, no loss) => the baseline trains byte-for-byte.
            loss_airl = torch.zeros((), device=device)
            airl_scale_mean = torch.zeros((), device=device)
            n_ground = 0
            if args.airl:
                # GROUND subset = views==1 (high-res view to degrade).  Slice the
                # clean inputs/preds to the SAME rows so consistency compares the
                # degraded ground vs its own clean ground prediction.
                g_mask = (views == 1)
                n_ground = int(g_mask.sum())
                # require >=2 ground rows: the degraded batch goes through the
                # train-mode model whose BNNeck BatchNorm1d raises "Expected more
                # than 1 value per channel" on a size-1 batch.  n_ground in {0,1}
                # -> skip AIRL this step (loss_airl stays 0, no extra forward).  The
                # ID-balanced RandomIdentitySampler makes a <2-ground batch a rare
                # cold edge, so the dropped consistency signal is negligible.
                if n_ground >= 2:
                    imgs_g = imgs[g_mask]
                    vidx_g = vidx[g_mask] if vidx is not None else None
                    # degrade in fp32 image space (resolution/low-pass proxy); the
                    # second forward runs under the SAME autocast as the original so
                    # AMP behaviour matches, while the consistency loss is fp32.
                    with torch.no_grad():
                        deg_imgs, deg_scales = airl_degrade(
                            imgs_g, args.airl_min_scale, blur=args.airl_blur)
                        airl_scale_mean = deg_scales.mean()
                    with torch.amp.autocast('cuda', enabled=not args.no_amp):
                        out_d = model(deg_imgs, view_idx=vidx_g,
                                      return_cvfc=(args.use_afd and args.afd_cvfc))
                    with torch.amp.autocast('cuda', enabled=False):
                        # consistency forces the DEGRADED ground prediction
                        # (gradient on) toward the CLEAN ground one (detached target
                        # inside the loss).  Clean side sliced to the ground rows.
                        loss_airl = airl_consistency_loss(
                            logits[g_mask], bn[g_mask],
                            out_d['logits'], out_d['bn_feat'],
                            mode=args.airl_consistency, tau=args.airl_tau)
                    loss = loss + airl_lambda_eff * loss_airl
                # n_ground < 2 -> too few ground rows this batch: loss_airl stays 0,
                # no extra forward, nothing added to loss (does not crash; avoids
                # the size-1 BatchNorm1d error).

            # AIRL dual-branch: the SAME ground-only degradation-consistency, but
            # applied ONLY to the f_rec head (logits_rec / bn_feat_rec).  f_full is
            # left clean in the sense that it receives ZERO consistency GRADIENT
            # (smoke D4) -> it keeps full-resolution discrimination (protects G->A);
            # f_rec is pulled toward its own clean prediction under the low pixel
            # budget (serves A->G).  NOTE: the degraded forward below is a FULL
            # model(deg_imgs) pass (the model has no rec-only path), so f_full's
            # frozen-bias BNNeck running mean/var DO see the degraded ground images
            # for stat tracking only -- exactly as in the --airl single-head path
            # above (same shared degrade+forward primitive), a deliberately accepted
            # minor exposure, NOT a gradient leak; whether it matters is settled
            # empirically by kill-switch #4, and matching --airl keeps the ablation
            # honest.  Identical degrade + >=2-ground guard + fp32 consistency as
            # --airl above; the only difference is the HEAD the consistency reads.
            # Mutually exclusive with --airl, so loss_airl is 0 unless dual-branch.
            if args.airl_dualbranch:
                g_mask = (views == 1)                      # high-res GROUND subset
                n_ground = int(g_mask.sum())
                if n_ground >= 2:
                    imgs_g = imgs[g_mask]
                    vidx_g = vidx[g_mask] if vidx is not None else None
                    with torch.no_grad():
                        deg_imgs, deg_scales = airl_degrade(
                            imgs_g, args.airl_min_scale, blur=args.airl_blur)
                        airl_scale_mean = deg_scales.mean()
                    with torch.amp.autocast('cuda', enabled=not args.no_amp):
                        out_d = model(deg_imgs, view_idx=vidx_g,
                                      return_cvfc=(args.use_afd and args.afd_cvfc))
                    with torch.amp.autocast('cuda', enabled=False):
                        # consistency on the f_rec head ONLY: degraded f_rec
                        # prediction (grad on) -> clean f_rec one (detached target
                        # inside the loss).  Both sides sliced to the ground rows.
                        loss_airl = airl_consistency_loss(
                            out['logits_rec'][g_mask], out['bn_feat_rec'][g_mask],
                            out_d['logits_rec'], out_d['bn_feat_rec'],
                            mode=args.airl_consistency, tau=args.airl_tau)
                    loss = loss + airl_lambda_eff * loss_airl
                # n_ground < 2 -> skip (same size-1 BatchNorm1d guard as --airl).

            # AIRL gradient-isolated dual-branch: the SAME ground-only degradation-
            # consistency on the f_rec head.  The DEGRADED side (out_d) comes from a
            # rec_only=True forward whose rec fork feed is ALWAYS detached from the
            # trunk (model.forward / _forward_swin_split), and the CLEAN side
            # (out['logits_rec'], out['bn_feat_rec']) is the DETACHED target inside
            # airl_consistency_loss.  So the consistency gradient flows ONLY through
            # out_d -> into the rec late stage + BNNeck_rec, and is severed at the
            # detach BEFORE the shared trunk -- the clean trunk + f_full receive ZERO
            # consistency gradient (smoke I4) REGARDLESS of --airl_iso_trunk_recce
            # (which only governs the CLEAN ID-CE pass, added above; the consistency's
            # clean side is detached here, so trunk_recce never opens a consistency
            # path to the trunk).  They keep full-resolution discrimination while
            # f_rec specialises as the recover pole.  The degraded forward uses
            # rec_only=True: it computes ONLY the f_rec head (the rec late stage +
            # BNNeck_rec), so f_full's BNNeck running stats are NOT updated on the
            # degraded images -> f_full stays a TRUE clean expert (no degraded-ground
            # stat leak, unlike the shared --airl_dualbranch which accepts that minor
            # exposure) and the f_full pool+classifier is skipped
            # (cheaper).  Mutually exclusive with --airl / --airl_dualbranch, so this
            # block fires only for the iso variant.
            if args.airl_dualbranch_iso:
                g_mask = (views == 1)                      # high-res GROUND subset
                n_ground = int(g_mask.sum())
                if n_ground >= 2:
                    imgs_g = imgs[g_mask]
                    vidx_g = vidx[g_mask] if vidx is not None else None
                    with torch.no_grad():
                        deg_imgs, deg_scales = airl_degrade(
                            imgs_g, args.airl_min_scale, blur=args.airl_blur)
                        airl_scale_mean = deg_scales.mean()
                    with torch.amp.autocast('cuda', enabled=not args.no_amp):
                        # rec_only -> dict with ONLY logits_rec / bn_feat_rec (f_full
                        # BNNeck not run on degraded images).
                        out_d = model(deg_imgs, view_idx=vidx_g, rec_only=True)
                    with torch.amp.autocast('cuda', enabled=False):
                        # consistency on the ISOLATED f_rec head: degraded f_rec
                        # prediction (grad on, into the rec stage only) -> clean f_rec
                        # one (detached target).  Both sides sliced to ground rows.
                        loss_airl = airl_consistency_loss(
                            out['logits_rec'][g_mask], out['bn_feat_rec'][g_mask],
                            out_d['logits_rec'], out_d['bn_feat_rec'],
                            mode=args.airl_consistency, tau=args.airl_tau)
                    loss = loss + airl_lambda_eff * loss_airl
                # n_ground < 2 -> skip (same size-1 BatchNorm1d guard as --airl).

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # batch size: define right after the step and BEFORE the OVP/ACVP
            # post-step blocks and the meter accumulation below, so no later code
            # (incl. the ACVP stats / meters) can ever hit UnboundLocalError on
            # the first iteration.  (Was previously assigned AFTER the ACVP stats
            # block -> first --acvp batch crashed.)
            bs = imgs.size(0)

            # update prototypes AFTER the optimizer step, with detached feats
            if args.ovp:
                with torch.no_grad():
                    ovp.update(F.normalize(bn.detach().float(), dim=1),
                               labels, views)
            # ACVP: EMA-update its OWN detached prototype bank from the current
            # batch (same detached BNNeck-feature recipe as OVP).  Done AFTER the
            # optimizer step so the bank tracks the just-updated encoder.  ACVP
            # only ever READS this bank (in ovli.loss above) -> no grad path.
            if args.acvp:
                with torch.no_grad():
                    acvp_mem.update(F.normalize(bn.detach().float(), dim=1),
                                    labels, views)
                    stats = getattr(ovli, '_acvp_stats', None)
                    if stats is not None:
                        # weight the per-step (frac, mean_w) by #softenable-neg
                        # (stats[2]=ok.sum()), NOT batch size, and SKIP steps with
                        # no softenable negatives so cold-start batches don't bias
                        # the per-epoch summary toward "looks safe".
                        n_soft = int(stats[2])
                        if n_soft > 0:
                            acvp_frac_sum += float(stats[0]) * n_soft
                            acvp_w_sum += float(stats[1]) * n_soft
                            acvp_steps += n_soft

            seen += bs
            meters['loss'] += loss.item() * bs
            meters['ce'] += loss_ce.item() * bs
            meters['tri'] += loss_tri.item() * bs
            meters['ovp'] += float(loss_ovp) * bs
            meters['ovli'] += float(loss_ovli) * bs
            meters['ovli_pos'] += float(ovli_pos) * bs
            meters['ovli_neg'] += float(ovli_neg) * bs
            meters['airl'] += float(loss_airl) * bs
            meters['airl_scale'] += float(airl_scale_mean) * bs
            meters['airl_n_ground'] += n_ground
            meters['ce_rec'] += float(loss_ce_rec) * bs
            meters['acc'] += (logits.argmax(1) == labels).float().sum().item()

            if (it + 1) % 50 == 0 or (it + 1) == n_iter_total:
                lr = optimizer.param_groups[0]['lr']
                extra = ""
                if args.ovp:
                    extra += f" OVP: {meters['ovp'] / seen:.4f}"
                if args.ovli:
                    extra += f" OVLI: {meters['ovli'] / seen:.4f}"
                if args.airl:
                    extra += f" AIRL: {meters['airl'] / seen:.4f}"
                if args.airl_dualbranch or args.airl_dualbranch_iso:
                    extra += (f" CE_rec: {meters['ce_rec'] / seen:.3f}"
                              f" AIRL_rec: {meters['airl'] / seen:.4f}")
                print(f"Epoch[{epoch}] Iter[{it + 1}/{n_iter_total}] "
                      f"Loss: {meters['loss'] / seen:.3f} "
                      f"CE: {meters['ce'] / seen:.3f} "
                      f"Tri: {meters['tri'] / seen:.3f}{extra} "
                      f"Acc: {meters['acc'] / seen:.3f} LR: {lr:.2e}")

        scheduler.step()
        dt = time.time() - t0
        ovp_msg = ""
        if args.ovp:
            ovp_msg = (f" OVP[lam_eff={ovp_lambda_eff:.3f} "
                       f"inited={int(ovp.inited.sum())}/{2 * ovp.num_pid}]")
        ovli_msg = ""
        if args.ovli:
            pos = meters['ovli_pos'] / seen
            neg = meters['ovli_neg'] / seen
            ovli_msg = (f" OVLI[lam_eff={ovli_lambda_eff:.3f} "
                        f"loss={meters['ovli'] / seen:.4f} "
                        f"pos={pos:.3f} neg={neg:.3f} gap={pos - neg:+.3f}]")
        # ACVP kill-switch log (per-epoch): relaxed_neg_frac = fraction of
        # SOFTENABLE negatives with w_ij<0.95, mean_w = mean softenable-neg weight,
        # both #softenable-neg-weighted across steps (cold-start / no-softenable
        # steps excluded).  Watch for relaxed_neg_frac>0.30 or mean_w<0.75 (=> ACVP
        # is broadly weakening negatives = bad; stop / reduce gamma).
        acvp_msg = ""
        if args.acvp:
            inited_n = int(acvp_mem.inited.sum())
            if acvp_steps > 0:
                a_frac = acvp_frac_sum / acvp_steps
                a_mw = acvp_w_sum / acvp_steps
            else:
                a_frac, a_mw = 0.0, 1.0
            flag = " <KILL?>" if (a_frac > 0.30 or a_mw < 0.75) else ""
            acvp_msg = (f" ACVP[gamma_eff={acvp_gamma_eff:.3f} "
                        f"relaxed_neg_frac={a_frac:.3f} mean_w={a_mw:.3f} "
                        f"proto_inited={inited_n}/{2 * acvp_mem.num_pid}{flag}]")
        # AIRL per-epoch kill-switch log: lam_eff (warmup-ramped weight),
        # consistency_loss (mean degraded<->clean consistency this epoch -- watch
        # it stay finite and TREND DOWN as the model learns budget-stable evidence;
        # exploding => degradation too harsh / lambda too high), deg_scale_mean
        # (mean sampled pixel-budget ratio, ~ (1+min_scale)/2; sanity that the
        # degradation actually fires), n_ground (total GROUND samples degraded this
        # epoch -- AIRL is asymmetric: only views==1 ground rows get degraded; this
        # confirms the asymmetric mask fires and aerial rows are left intact).
        # airl_collapse flag if the consistency loss is ~0 from the start
        # (degradation not biting) / NaN/inf, OR n_ground==0 all epoch (mask wrong /
        # no ground sampled => AIRL silently did nothing).
        airl_msg = ""
        if args.airl:
            a_cons = meters['airl'] / seen
            a_scale = meters['airl_scale'] / seen
            n_g = int(meters['airl_n_ground'])
            bad = (not math.isfinite(a_cons)) or (n_g == 0)
            flag = " <KILL?>" if bad else ""
            airl_msg = (f" AIRL[lam_eff={airl_lambda_eff:.3f} "
                        f"consistency={a_cons:.4f} deg_scale_mean={a_scale:.3f} "
                        f"n_ground={n_g}{flag}]")
        # AIRL dual-branch per-epoch log: f_rec ID-CE (must converge like f_full's
        # CE -> f_rec is a valid identity space), the f_rec degradation-consistency
        # (same trend-down expectation as --airl), deg_scale_mean and n_ground.
        # Collapse flag if ce_rec is non-finite, the consistency is non-finite, or
        # n_ground==0 all epoch (asymmetric mask never fired).
        if args.airl_dualbranch or args.airl_dualbranch_iso:
            a_cons = meters['airl'] / seen
            a_scale = meters['airl_scale'] / seen
            a_cerec = meters['ce_rec'] / seen
            n_g = int(meters['airl_n_ground'])
            bad = (not math.isfinite(a_cons)) or (not math.isfinite(a_cerec)) \
                or (n_g == 0)
            flag = " <KILL?>" if bad else ""
            tag = "AIRL-ISO" if args.airl_dualbranch_iso else "AIRL-DUAL"
            airl_msg = (f" {tag}[lam_eff={airl_lambda_eff:.3f} "
                        f"ce_rec={a_cerec:.3f} consistency={a_cons:.4f} "
                        f"deg_scale_mean={a_scale:.3f} n_ground={n_g}{flag}]")
        print(f"Epoch[{epoch}] done in {dt:.1f}s  "
              f"Loss={meters['loss'] / seen:.3f} "
              f"Acc={meters['acc'] / seen:.3f}{ovp_msg}{ovli_msg}{acvp_msg}{airl_msg}")

        if epoch % args.eval_period == 0 or epoch == args.epochs:
            results = run_cross_view_eval(model, dataset, args, device)
            mean_map = print_eval(epoch, results)
            # opt-in OVLI rerank report (global-only number above is the primary
            # eval and is unchanged; this just adds the global+MaxSim rerank).
            if args.ovli and args.ovli_rerank:
                rr = ovli_rerank_eval(model, ovli, dataset, args, device)
                print(f"  ---- OVLI rerank (alpha={args.ovli_alpha}) "
                      f"@ epoch {epoch} ----")
                for tag in ('A->G', 'G->A'):
                    gm, gr = rr[tag]['global']
                    rm, rrk = rr[tag]['rerank']
                    print(f"    [{tag}] global mAP={gm:.2f} R1={gr:.2f}  ->  "
                          f"rerank mAP={rm:.2f} R1={rrk:.2f}")
                rmean = (rr['A->G']['rerank'][0] + rr['G->A']['rerank'][0]) / 2
                print(f"    [mean] rerank mAP={rmean:.2f}")
            # AIRL dual-branch: report f_full-only, f_rec-only, and the SOFT-FUSED
            # mean (cos = w*cos_rec + (1-w)*cos_full).  The run_cross_view_eval
            # number above is the f_full-only head (model() returns f_full at eval);
            # the HEADLINE = the fused mean, which is what model-selection uses (the
            # whole point of the dual-branch is the fusion, not f_full alone).
            if args.airl_dualbranch or args.airl_dualbranch_iso:
                # airl_dualbranch_eval calls model(return_dual=True) which yields
                # (f_full, f_rec) for BOTH the shared and the iso variant (the iso
                # forward's want_iso path returns the same tuple), so the soft-fusion
                # eval is shared verbatim.
                dual = airl_dualbranch_eval(model, dataset, args, device)
                _ev_tag = ("AIRL-ISO dual-branch" if args.airl_dualbranch_iso
                           else "AIRL dual-branch")
                print(f"  ---- {_ev_tag} (fuse_w={args.airl_fuse_w}) "
                      f"@ epoch {epoch} ----")
                for tag in ('A->G', 'G->A'):
                    print(f"    [{tag}] full mAP={dual[tag]['full'][0]:.2f} "
                          f"R1={dual[tag]['full'][1]:.2f} | "
                          f"rec mAP={dual[tag]['rec'][0]:.2f} "
                          f"R1={dual[tag]['rec'][1]:.2f} | "
                          f"FUSE mAP={dual[tag]['fuse'][0]:.2f} "
                          f"R1={dual[tag]['fuse'][1]:.2f}")
                full_mean = (dual['A->G']['full'][0] + dual['G->A']['full'][0]) / 2
                rec_mean = (dual['A->G']['rec'][0] + dual['G->A']['rec'][0]) / 2
                fuse_mean = (dual['A->G']['fuse'][0] + dual['G->A']['fuse'][0]) / 2
                print(f"    [mean] full={full_mean:.2f} rec={rec_mean:.2f} "
                      f"FUSE={fuse_mean:.2f}  <- model-selection uses FUSE")
                # override model-selection metric with the fused mean
                mean_map = fuse_mean
            if mean_map > best_map:
                best_map = mean_map
                best_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(args.out_dir, 'model_best.pth'))
                if ovli is not None:
                    torch.save(ovli.state_dict(),
                               os.path.join(args.out_dir, 'ovli_best.pth'))
                print(f"    * new best mean mAP={best_map:.2f} (epoch {epoch}) saved")

    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_final.pth'))
    if ovli is not None:
        torch.save(ovli.state_dict(), os.path.join(args.out_dir, 'ovli_final.pth'))
        ovli.remove_hook()
    print("=" * 70)
    print(f"Training finished. Best mean A<->G mAP={best_map:.2f} @ epoch {best_epoch}")
    print(f"Checkpoints in {args.out_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
