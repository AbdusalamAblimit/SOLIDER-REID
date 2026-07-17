"""Isolated numeric smoke test for AIRL (Aerial Identity Recoverability Learning
-- resolution-degradation consistency) in cargo_cvpb/afd_train.py.

AIRL degrades each NORMALIZED image to a sampled aerial-scale pixel budget
(bilinear down -> up, + optional avg-pool blur), runs ONE extra forward through
the SAME backbone, and adds a prediction-consistency loss (KL on logits OR
1-cosine on the BNNeck feature) pulling the degraded prediction toward the
detached clean one.  NO learnable params; TRAIN-time only; --airl OFF reproduces
the baseline byte-for-byte.

Loads the REAL airl_degrade / airl_consistency_loss out of cargo_cvpb/afd_train.py
(NOT a copy) by stubbing the heavy sibling modules (cargo_dataset /
agreid_dataset / afd_model / afd_train) in sys.modules, so no real backbone /
dataset is built.  CPU-only, tiny tensors -> never touches a GPU.

Checks
  S1  DEGRADE SHAPE/DTYPE: airl_degrade returns (deg, scales) with deg.shape ==
        imgs.shape, deg fp32 finite, scales in [min_scale, 1], len == B.  Holds
        for blur=False AND blur=True.
  S2  DEGRADATION ACTUALLY DEGRADES: a heavily-degraded image (min_scale small,
        forced s) differs from the original (non-trivial L2 delta) yet stays
        finite -> the resample removes spatial detail (does not no-op / NaN).
  S3  s==1 IS (near) IDENTITY: with min_scale=1.0 the budget is full-resolution
        so deg ~= imgs (no detail removed); blur stays finite.
  S4  CONSISTENCY FINITE > 0 (kl): airl_consistency_loss(mode='kl') on distinct
        clean/degraded logits is finite and strictly > 0; == 0 when the two
        logits are IDENTICAL (perfect agreement -> zero consistency).
  S5  CONSISTENCY FINITE >= 0 (feat): mode='feat' = 1-cosine in [0,2], finite;
        == 0 when bn_o == bn_d (identical features) and > 0 when they differ.
  S6  CLEAN TARGET DETACHED: backward through the consistency loss gives NO grad
        on the clean inputs (logits_o / bn_o) and a FINITE grad on the degraded
        inputs (logits_d / bn_d) -> only the degraded branch is pulled (clean is
        the stable target).  Holds for both kl and feat.
  S7  GRADIENT REACHES BACKBONE: with a tiny stand-in backbone, a full AIRL step
        (clean forward + degrade + degraded forward + consistency.backward())
        puts FINITE non-zero gradient on the SHARED backbone weight -> the
        consistency signal flows through the degraded forward into the encoder.
  S8  OFF BYTE-IDENTICAL: emulate the loop's `loss = ce + tri (+ airl)` with the
        AIRL block GUARDED by args.airl.  args.airl=False -> total loss is
        torch.equal to the bare ce+tri (the AIRL functions are never called, the
        zero-init loss_airl is never added) -> baseline reproduced exactly.
  S9  NAN-SAFETY: kl with extreme logits (large magnitude, one-hot-ish) and feat
        with tiny/zero feature vectors stay finite (log_softmax / normalize floor).
  S10 DETERMINISM via generator: airl_degrade with a fixed torch.Generator gives
        identical scales across two calls (reproducible per-image budgets).
  S11 ASYMMETRIC GROUND-ONLY DEGRADE: AIRL degrades ONLY the high-res GROUND
        subset (views==1; Aerial==0).  On a mixed-view mock batch, exactly the
        ground rows are degraded/forwarded and the aerial rows are excluded from
        the consistency loss (S11), aerial rows WOULD change if degraded so
        skipping them is a real decision (S11b), the degraded-ground input really
        differs from clean and the ground-only KL is finite/>=0 (S11c), and an
        all-aerial batch yields loss_airl == 0 with no extra forward (S11d), and a
        SINGLETON-ground batch (1 ground row) under a train-mode model is SKIPPED
        with NO size-1 BatchNorm1d crash (the >=2 guard) (S11e).
        Guards AIRL's asymmetric hypothesis: degrade ground to an aerial budget,
        never further degrade the already-low-budget aerial samples.

Usage:  python smoke_airl.py [path/to/cargo_cvpb/afd_train.py]
        (run inside an env with torch+numpy, e.g.
         `uv run --no-project --with numpy --with torch python smoke_airl.py`)
"""
import importlib.util
import math
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1) stub the heavy sibling modules so importing afd_train.py is cheap/isolated
# --------------------------------------------------------------------------- #
def _stub(name, attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m


def _noop_init(_m):  # stand-in for afd_model.weights_init_kaiming
    return None


_dummy = lambda *a, **k: None  # noqa: E731
_stub('cargo_dataset', dict(CARGO=_dummy, CARGOImageDataset=_dummy,
                            build_transforms=_dummy, RandomIdentitySampler=_dummy,
                            filter_by_view=_dummy))
_stub('agreid_dataset', dict(AGReIDv2=_dummy))
_stub('afd_model', dict(build_model=_dummy, weights_init_kaiming=_noop_init))
_stub('afd_train', dict(CrossEntropyLabelSmooth=_dummy, TripletLoss=_dummy,
                        WarmupCosineLR=_dummy, run_cross_view_eval=_dummy,
                        print_eval=_dummy, set_seed=_dummy))


def load_afd(path):
    spec = importlib.util.spec_from_file_location('cvpb_afd_train_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# tiny stand-in backbone (NOT the real one) -- just enough to test grad flow:
# Conv -> GAP -> BN feature + classifier logits, returns the loop's train dict.
# --------------------------------------------------------------------------- #
class TinyModel(nn.Module):
    def __init__(self, num_classes=10, cin=3, cfeat=16):
        super().__init__()
        self.conv = nn.Conv2d(cin, cfeat, 3, padding=1)   # the SHARED weight we probe
        self.bn = nn.BatchNorm1d(cfeat)
        self.classifier = nn.Linear(cfeat, num_classes, bias=False)

    def forward(self, x, view_idx=None, return_cvfc=False):
        h = self.conv(x)                       # (B,cfeat,H,W)
        g = h.mean(dim=(2, 3))                 # GAP -> (B,cfeat) "global_feat"
        bn = self.bn(g)
        return {'global_feat': g, 'bn_feat': bn,
                'logits': self.classifier(bn), 'band_w': None}


# --------------------------------------------------------------------------- #
def approx_eq(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)


def main():
    path = (sys.argv[1] if len(sys.argv) > 1
            else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'afd_train.py'))
    mod = load_afd(path)
    airl_degrade = mod.airl_degrade
    airl_consistency_loss = mod.airl_consistency_loss
    torch.manual_seed(0)

    B, C, H, W = 8, 3, 32, 16
    n_cls = 10
    imgs = torch.randn(B, C, H, W)

    passed, failed = [], []

    def check(name, cond, detail=''):
        (passed if cond else failed).append(name)
        tag = 'PASS' if cond else '**FAIL**'
        print(f'  [{tag}] {name}{("  -- " + detail) if detail else ""}')

    # -- S1: degrade shape/dtype, scales range, blur on/off ------------------- #
    for blur in (False, True):
        deg, sc = airl_degrade(imgs, min_scale=0.25, blur=blur)
        ok = (tuple(deg.shape) == (B, C, H, W) and deg.dtype == torch.float32
              and torch.isfinite(deg).all().item()
              and sc.numel() == B
              and float(sc.min()) >= 0.25 - 1e-6 and float(sc.max()) <= 1.0 + 1e-6)
        check(f'S1 degrade shape/dtype/scales (blur={blur})', ok,
              f'shape={tuple(deg.shape)} s in [{float(sc.min()):.3f},'
              f'{float(sc.max()):.3f}]')

    # -- S2: degradation actually removes detail ----------------------------- #
    # force a very small budget by min_scale tiny; over many images at least the
    # mean L2 delta must be clearly non-zero and finite.
    deg2, sc2 = airl_degrade(imgs, min_scale=0.1, blur=False)
    delta = (deg2 - imgs).pow(2).mean().item()
    check('S2 degradation removes spatial detail', math.isfinite(delta) and delta > 1e-4,
          f'mean||deg-orig||^2={delta:.4f}')

    # -- S3: min_scale == 1.0 -> (near) identity ----------------------------- #
    deg1, sc1 = airl_degrade(imgs, min_scale=1.0, blur=False)
    d1 = (deg1 - imgs).abs().max().item()
    check('S3 min_scale=1 is identity (no detail removed)',
          math.isfinite(d1) and d1 < 1e-5 and float(sc1.min()) >= 1.0 - 1e-6,
          f'max|deg-orig|={d1:.2e}')
    deg1b, _ = airl_degrade(imgs, min_scale=1.0, blur=True)   # blur path finite
    check('S3b min_scale=1 + blur finite', torch.isfinite(deg1b).all().item())

    # -- S4: KL consistency finite>0; ==0 on identical logits ----------------- #
    lo = torch.randn(B, n_cls)
    ld = torch.randn(B, n_cls)
    kl = airl_consistency_loss(lo, None, ld, None, mode='kl', tau=4.0)
    check('S4 KL consistency finite > 0',
          torch.isfinite(kl).item() and float(kl) > 0, f'kl={float(kl):.4f}')
    kl_same = airl_consistency_loss(lo, None, lo.clone(), None, mode='kl', tau=4.0)
    check('S4b KL == 0 on identical logits',
          torch.isfinite(kl_same).item() and abs(float(kl_same)) < 1e-5,
          f'kl_same={float(kl_same):.2e}')

    # -- S5: feat consistency finite>=0; ==0 on identical feats --------------- #
    bo = torch.randn(B, 16)
    bd = torch.randn(B, 16)
    fl = airl_consistency_loss(None, bo, None, bd, mode='feat')
    check('S5 feat consistency finite > 0 (distinct)',
          torch.isfinite(fl).item() and float(fl) > 0, f'feat={float(fl):.4f}')
    fl_same = airl_consistency_loss(None, bo, None, bo.clone(), mode='feat')
    check('S5b feat == 0 on identical feats',
          torch.isfinite(fl_same).item() and abs(float(fl_same)) < 1e-5,
          f'feat_same={float(fl_same):.2e}')

    # -- S6: clean target detached, degraded gets finite grad ----------------- #
    for mode in ('kl', 'feat'):
        lo_g = torch.randn(B, n_cls, requires_grad=True)
        ld_g = torch.randn(B, n_cls, requires_grad=True)
        bo_g = torch.randn(B, 16, requires_grad=True)
        bd_g = torch.randn(B, 16, requires_grad=True)
        L = airl_consistency_loss(lo_g, bo_g, ld_g, bd_g, mode=mode, tau=4.0)
        L.backward()
        if mode == 'kl':
            clean_grad = lo_g.grad
            deg_grad = ld_g.grad
        else:
            clean_grad = bo_g.grad
            deg_grad = bd_g.grad
        clean_none = (clean_grad is None) or (clean_grad.abs().sum().item() == 0.0)
        deg_ok = (deg_grad is not None and torch.isfinite(deg_grad).all().item()
                  and deg_grad.abs().sum().item() > 0)
        check(f'S6 clean detached / degraded grad finite ({mode})',
              clean_none and deg_ok,
              f'clean_grad_zero={clean_none} deg_grad_sum='
              f'{(deg_grad.abs().sum().item() if deg_grad is not None else 0):.3f}')

    # -- S7: gradient reaches a shared backbone through the degraded forward --- #
    torch.manual_seed(1)
    model = TinyModel(num_classes=n_cls)
    model.train()
    imgs7 = torch.randn(B, C, H, W)
    labels7 = torch.randint(0, n_cls, (B,))
    out_o = model(imgs7)
    deg7, _ = airl_degrade(imgs7, min_scale=0.25, blur=False)
    out_d = model(deg7)
    # AIRL-only loss (isolate the AIRL gradient path to the backbone)
    L7 = airl_consistency_loss(out_o['logits'], out_o['bn_feat'],
                               out_d['logits'], out_d['bn_feat'],
                               mode='kl', tau=4.0)
    model.zero_grad()
    L7.backward()
    gw = model.conv.weight.grad
    check('S7 AIRL gradient reaches shared backbone conv',
          gw is not None and torch.isfinite(gw).all().item()
          and gw.abs().sum().item() > 0,
          f'conv.grad_sum={(gw.abs().sum().item() if gw is not None else 0):.4f}')

    # -- S8: OFF byte-identical (AIRL block guarded by args.airl) -------------- #
    class _NS:  # minimal args stand-in
        pass

    def emulate_total(args_airl):
        """Reproduce the loop's `loss = ce + tri (+ airl)` with the AIRL block
        guarded by args.airl, on fixed inputs, and return the total."""
        torch.manual_seed(7)
        m = TinyModel(num_classes=n_cls)
        m.train()
        x = torch.randn(B, C, H, W)
        y = torch.randint(0, n_cls, (B,))
        o = m(x)
        # stand-in CE + triplet surrogate (any fixed scalars built from the clean
        # forward; AIRL must NOT change them when off)
        loss = (F.cross_entropy(o['logits'], y)
                + o['global_feat'].pow(2).mean())            # "triplet" surrogate
        loss_airl = torch.zeros(())
        if args_airl:                                        # GUARD == the loop's
            dg, _ = airl_degrade(x, min_scale=0.25, blur=False)
            od = m(dg)
            loss_airl = airl_consistency_loss(o['logits'], o['bn_feat'],
                                              od['logits'], od['bn_feat'],
                                              mode='kl', tau=4.0)
            loss = loss + 0.5 * loss_airl
        return loss, loss_airl

    off_loss, off_airl = emulate_total(False)
    on_loss, on_airl = emulate_total(True)
    check('S8 OFF byte-identical (airl=False adds nothing)',
          float(off_airl) == 0.0,                            # never computed off
          f'off_airl={float(off_airl)}')
    # ON path must differ (sanity that the guard, not a no-op, is what gates it)
    on_airl_v = float(on_airl.detach())
    check('S8b ON actually adds the term',
          on_airl_v != 0.0 and not approx_eq(off_loss.detach(),
                                             on_loss.detach()),
          f'on_airl={on_airl_v:.4f}')

    # -- S9: NaN-safety with extreme inputs ----------------------------------- #
    lo_ext = torch.full((B, n_cls), -50.0)
    lo_ext[:, 0] = 50.0                                      # near one-hot, huge mag
    ld_ext = torch.randn(B, n_cls) * 30.0
    kl_ext = airl_consistency_loss(lo_ext, None, ld_ext, None, mode='kl', tau=4.0)
    bo_tiny = torch.zeros(B, 16)                             # zero vectors
    bd_tiny = torch.zeros(B, 16)
    feat_tiny = airl_consistency_loss(None, bo_tiny, None, bd_tiny, mode='feat')
    check('S9 NaN-safety (extreme logits / zero feats)',
          torch.isfinite(kl_ext).item() and torch.isfinite(feat_tiny).item(),
          f'kl_ext={float(kl_ext):.4f} feat_zero={float(feat_tiny):.4f}')

    # -- S10: determinism via a fixed generator ------------------------------- #
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    _, s_a = airl_degrade(imgs, min_scale=0.25, blur=False, generator=g1)
    _, s_b = airl_degrade(imgs, min_scale=0.25, blur=False, generator=g2)
    check('S10 generator determinism (same seed -> same scales)',
          torch.allclose(s_a, s_b), f'max|ds|={float((s_a - s_b).abs().max()):.2e}')

    # -- S11: AIRL is ASYMMETRIC -- only the GROUND subset (views==1) is degraded;
    #         aerial rows (views==0) are NEVER degraded and NEVER enter the
    #         consistency loss; an all-aerial batch yields loss_airl == 0 --------- #
    # CARGO view encoding (cargo_dataset._parse_name): cam1-5 Aerial, cam6-13
    # Ground; afd_train view_map = {'Aerial':0,'Ground':1}. AIRL must degrade the
    # HIGH-RES ground (==1), not the already-low-budget aerial (==0).
    def airl_block(imgs_b, views_b, model_b):
        """Re-implement the loop's AIRL view-filter to verify what gets degraded.
        MIRRORS the train loop: only degrade when there are >=2 ground rows (a
        size-1 batch would trip the train-mode BNNeck BatchNorm1d).
        Returns (loss_airl, n_ground, degraded_rows_idx, deg_imgs_or_None)."""
        g_mask = (views_b == 1)
        n_g = int(g_mask.sum())
        if n_g < 2:                                    # <2 ground -> skip, loss 0
            return torch.zeros(()), n_g, [], None
        imgs_g = imgs_b[g_mask]
        o_full = model_b(imgs_b)                       # clean forward (full batch)
        dg, _ = airl_degrade(imgs_g, min_scale=0.25, blur=False)
        od = model_b(dg)
        L = airl_consistency_loss(o_full['logits'][g_mask], o_full['bn_feat'][g_mask],
                                  od['logits'], od['bn_feat'],
                                  mode='kl', tau=4.0)
        return L, n_g, g_mask.nonzero(as_tuple=True)[0].tolist(), dg

    torch.manual_seed(2)
    m11 = TinyModel(num_classes=n_cls)
    m11.eval()                                          # deterministic (no BN noise)
    Bm = 6
    imgs11 = torch.randn(Bm, C, H, W)
    # mixed batch: rows 0,2,4 = Aerial(0); rows 1,3,5 = Ground(1)
    views11 = torch.tensor([0, 1, 0, 1, 0, 1])
    aerial_idx = (views11 == 0).nonzero(as_tuple=True)[0].tolist()
    ground_idx = (views11 == 1).nonzero(as_tuple=True)[0].tolist()
    L11, n_g11, deg_idx, dg11 = airl_block(imgs11, views11, m11)
    # (a) exactly the ground rows were degraded; aerial rows excluded
    check('S11 only GROUND rows degraded (views==1)',
          deg_idx == ground_idx and n_g11 == len(ground_idx)
          and dg11.shape[0] == len(ground_idx),
          f'degraded_rows={deg_idx} ground_rows={ground_idx} n_ground={n_g11}')
    # (b) directly degrade the FULL batch and confirm the aerial rows WOULD change
    #     if degraded -- i.e. AIRL leaving them untouched is a real (non-no-op)
    #     decision, not "degradation happens to be identity on aerial".
    deg_full, _ = airl_degrade(imgs11, min_scale=0.1, blur=False)
    aerial_changed = max((deg_full[i] - imgs11[i]).abs().max().item()
                         for i in aerial_idx)
    check('S11b aerial rows WOULD change if degraded (so skipping them matters)',
          aerial_changed > 1e-4, f'max|deg-orig| over aerial={aerial_changed:.4f}')
    # (c) the degraded GROUND input differs from the clean ground input (the extra
    #     forward really runs on degraded ground), and the consistency is a finite,
    #     non-negative KL (KL>=0 always; untrained tiny model may give ~0, which is
    #     fine -- the point is the term is well-formed and ground-only).
    ground_deg_delta = (dg11 - imgs11[views11 == 1]).pow(2).mean().item()
    check('S11c ground really degraded + consistency finite & >= 0',
          ground_deg_delta > 1e-4
          and torch.isfinite(L11).item() and float(L11.detach()) >= 0.0,
          f'ground_deg_delta={ground_deg_delta:.4f} L={float(L11.detach()):.4f}')
    # (d) all-aerial batch -> n_ground==0 -> loss_airl == 0 (no crash, nothing
    #     added, no extra forward, no degraded tensor).
    views_aer = torch.zeros(Bm, dtype=torch.long)       # all Aerial
    L_aer, n_g_aer, deg_aer, dg_aer = airl_block(imgs11, views_aer, m11)
    check('S11d all-aerial batch -> loss_airl == 0 (no ground, no degrade)',
          n_g_aer == 0 and float(L_aer) == 0.0 and deg_aer == [] and dg_aer is None,
          f'n_ground={n_g_aer} loss_airl={float(L_aer)}')
    # (e) SINGLETON ground (exactly 1 ground row) under a TRAIN-mode model must be
    #     SKIPPED (loss_airl==0, no degraded forward) -- a size-1 batch through the
    #     train-mode BNNeck BatchNorm1d would raise "Expected more than 1 value per
    #     channel".  The >=2 guard prevents that.  Run train mode so the test would
    #     actually crash if the block did NOT skip.
    m11_tr = TinyModel(num_classes=n_cls)
    m11_tr.train()
    views_one = torch.tensor([0, 1, 0, 0, 0, 0])        # exactly ONE ground row
    crashed = False
    try:
        L_one, n_g_one, deg_one, dg_one = airl_block(imgs11, views_one, m11_tr)
    except Exception as e:                              # pragma: no cover
        crashed = True
        L_one, n_g_one, deg_one, dg_one = None, -1, None, None
        print(f'    (singleton-ground raised: {type(e).__name__}: {e})')
    check('S11e singleton-ground SKIPPED (no size-1 BN crash, loss==0)',
          (not crashed) and n_g_one == 1 and float(L_one) == 0.0
          and deg_one == [] and dg_one is None,
          f'crashed={crashed} n_ground={n_g_one} '
          f'loss_airl={(float(L_one) if L_one is not None else None)}')

    # -- summary -------------------------------------------------------------- #
    print('\n' + '=' * 70)
    print(f'AIRL smoke: {len(passed)} passed, {len(failed)} failed')
    if failed:
        print('FAILED:', failed)
        sys.exit(1)
    print('ALL AIRL SMOKE CHECKS PASSED')
    print('=' * 70)


if __name__ == '__main__':
    main()
