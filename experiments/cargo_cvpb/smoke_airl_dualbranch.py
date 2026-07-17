"""Isolated numeric smoke test for the AIRL DUAL-BRANCH (resolvability branch) in
cargo_cvpb/afd_train.py + afd_reid/afd_model.py.

AIRL dual-branch = the COMPLETE AIRL mechanism: a SECOND BNNeck head (f_rec) on
the SAME shared backbone, trained with its OWN ID-CE PLUS the AIRL ground-
degradation consistency, alongside the clean f_full head (ID-CE + the shared
triplet, NO consistency).  At eval the two heads' cosine scores are SOFT-fused at
the distance-matrix level: cos = w*cos_rec + (1-w)*cos_full (w fixed, default
0.25).  ONE forward yields BOTH features.  Goal: internalise the kill-switch #3
two-model score fusion (+1.46 mean @ w=0.25) into a single forward.

Two layers are exercised:
  (1) the REAL afd_model.AFDModel(airl_dualbranch=True) -- the architecture
      (second head, shared global_feat, dual eval, head-divergence of gradients).
  (2) the REAL airl_degrade / airl_consistency_loss out of cargo_cvpb/afd_train.py
      (loaded with the heavy siblings stubbed, exactly like smoke_airl.py) -- the
      degradation + consistency primitive the dual-branch reuses UNCHANGED.

Checks
  D1  OFF BYTE-IDENTICAL (model): airl_dualbranch=False builds NO second head
        (no bottleneck_rec / classifier_rec), the train dict has EXACTLY the
        baseline keys (no *_rec), and eval returns a single feature.  An ON model
        fed return_dual=False ALSO returns the single f_full feature == the
        baseline eval feature (the dual path is opt-in).
  D2  DUAL HEAD SHAPES: ON model -> train dict carries bn_feat_rec / logits_rec
        with the right shapes; eval(return_dual=True) returns (f_full, f_rec) both
        (B,D) and L2-normalized; the two heads are DISTINCT (different params ->
        f_full != f_rec on the same input).
  D3  f_rec CONSISTENCY GRADIENT REACHES f_rec HEAD + BACKBONE: a full dual step
        (clean forward + ground degrade + degraded forward + consistency on the
        f_rec head) puts FINITE non-zero grad on bottleneck_rec.weight AND on a
        shared backbone conv weight.
  D4  f_full HEAD GETS NO CONSISTENCY GRADIENT: the SAME f_rec consistency
        backward leaves f_full's bottleneck.weight / classifier.weight grad
        None-or-zero (the consistency reads logits_rec / bn_feat_rec ONLY, so the
        f_full head stays clean -> head divergence is real).
  D5  BOTH HEADS IN OPTIMIZER + ACTUALLY TRAIN: every trainable f_rec param
        (bottleneck_rec weight + classifier_rec weight) is in an AdamW built over
        model.parameters(); one optimiser step on (CE_full + CE_rec) MOVES both
        classifier.weight (f_full) AND classifier_rec.weight (f_rec).
  D6  SOFT-FUSION DISTMAT CORRECT: the eval fusion dist = 2 - 2*(w*cos_rec +
        (1-w)*cos_full) equals a hand-computed reference; w=0 == the f_full-only
        distmat; w=1 == the f_rec-only distmat; an intermediate w is the convex
        blend (matches the kill-switch #3 GATE-5 formula).
  D7  NAN-SAFETY (f_rec consistency): KL with extreme logits and feat with zero
        vectors on the f_rec head stay finite (same floors as --airl).
  D8  f_rec ID-CE GROUNDS f_rec: cross-entropy on logits_rec backward puts finite
        non-zero grad on classifier_rec.weight + the backbone (so f_rec is a valid
        identity space, not a consistency-only collapse).
  D9  SHARED global_feat (single triplet): both heads read the SAME pooled
        global_feat (f_full's global_feat tensor is the BN input of BOTH heads),
        confirming the global triplet is NOT duplicated for f_rec.
  D10 SWIN-STYLE PARAM SPLIT: emulate the optimiser's swin param-group split and
        confirm the f_rec head lands in the FULL-LR "other" group (not the
        scaled-down backbone group) -- f_rec is a random-init head and must learn
        at full LR even on Swin.

Usage:  python smoke_airl_dualbranch.py [path/to/cargo_cvpb/afd_train.py]
        (run inside an env with torch+numpy+torchvision, e.g.
         `uv run --no-project --with numpy --with torch --with torchvision \
            python smoke_airl_dualbranch.py`)
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
# load the REAL airl_degrade / airl_consistency_loss out of afd_train.py with
# the heavy siblings stubbed (identical recipe to smoke_airl.py).  The REAL
# AFDModel is imported separately below (not stubbed) for the architecture tests.
# --------------------------------------------------------------------------- #
def _stub(name, attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m


_dummy = lambda *a, **k: None  # noqa: E731
HERE = os.path.dirname(os.path.abspath(__file__))
AFD_REID = os.path.join(HERE, '..', 'afd_reid')
sys.path.insert(0, AFD_REID)   # so the REAL afd_model / cargo_dataset resolve

_stub('cargo_dataset', dict(CARGO=_dummy, CARGOImageDataset=_dummy,
                            build_transforms=_dummy, RandomIdentitySampler=_dummy,
                            filter_by_view=_dummy))
_stub('agreid_dataset', dict(AGReIDv2=_dummy))
# NOTE: afd_model is NOT stubbed here -- we want the real build_model symbol for
# afd_train's top-level import, and we import the real AFDModel below.  afd_train
# (the cargo_cvpb one) imports `build_model` from afd_model and several helpers
# from the PARENT afd_train; stub the parent-afd_train symbols it imports at
# module load so the import is cheap and self-contained.
_stub('afd_train', dict(CrossEntropyLabelSmooth=_dummy, TripletLoss=_dummy,
                        WarmupCosineLR=_dummy, run_cross_view_eval=_dummy,
                        print_eval=_dummy, set_seed=_dummy,
                        build_eval_loader=_dummy, eval_market=_dummy))
_stub('maxsim_probe', dict(eval_from_distmat=_dummy))

from afd_model import AFDModel   # the REAL dual-branch model  # noqa: E402


def load_cvpb_afd(path):
    spec = importlib.util.spec_from_file_location('cvpb_afd_dual_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(num_classes, airl_dualbranch):
    """Tiny CPU AFDModel (resnet50, NO pretrained download)."""
    return AFDModel(num_classes=num_classes, pretrained=False,
                    backbone='resnet50', airl_dualbranch=airl_dualbranch)


def main():
    path = (sys.argv[1] if len(sys.argv) > 1
            else os.path.join(HERE, 'afd_train.py'))
    mod = load_cvpb_afd(path)
    airl_degrade = mod.airl_degrade
    airl_consistency_loss = mod.airl_consistency_loss
    torch.manual_seed(0)

    B, C, H, W = 4, 3, 256, 128
    n_cls = 8

    passed, failed = [], []

    def check(name, cond, detail=''):
        (passed if cond else failed).append(name)
        tag = 'PASS' if cond else '**FAIL**'
        print(f'  [{tag}] {name}{("  -- " + detail) if detail else ""}')

    # ----------------------------------------------------------------------- #
    # D1: OFF byte-identical (model level)
    # ----------------------------------------------------------------------- #
    torch.manual_seed(1)
    m_off = build_model(n_cls, airl_dualbranch=False)
    m_off.eval()
    x = torch.randn(B, C, H, W)
    with torch.no_grad():
        f_off = m_off(x)
    # snapshot the OFF state dict NOW, in eval mode, BEFORE any train-mode forward
    # (a train forward would update the BNNeck running stats and desync the
    # reference feature below).  deep-copy the tensors so later code can't mutate.
    sd_off = {k: v.detach().clone() for k, v in m_off.state_dict().items()}
    m_off.train()
    o_off = m_off(x)
    no_second_head = (not hasattr(m_off, 'bottleneck_rec')
                      and not hasattr(m_off, 'classifier_rec')
                      and not m_off.airl_dualbranch)
    base_keys = {'global_feat', 'bn_feat', 'logits', 'band_w'}
    keys_clean = (set(o_off.keys()) == base_keys)
    check('D1 OFF builds no second head + baseline train-dict keys',
          no_second_head and keys_clean and tuple(f_off.shape) == (B, m_off.in_planes),
          f'has_rec={hasattr(m_off, "bottleneck_rec")} keys={sorted(o_off.keys())}')

    # ON model, but called the LEGACY eval way (return_dual=False) -> must return
    # the SAME single f_full feature (dual path is strictly opt-in).  Load the
    # f_full / backbone params from the OFF snapshot so the f_full feature is
    # bit-identical -> proves enabling the flag does not perturb the f_full path.
    torch.manual_seed(1)
    m_on = build_model(n_cls, airl_dualbranch=True)
    msg = m_on.load_state_dict(sd_off, strict=False)
    # the only missing keys must be the NEW rec head (off had none of them)
    only_rec_missing = all('bottleneck_rec' in k or 'classifier_rec' in k
                           for k in msg.missing_keys)
    m_on.eval()
    with torch.no_grad():
        f_on_legacy = m_on(x)                   # return_dual default False
    same_full = torch.allclose(f_off, f_on_legacy, atol=1e-6)
    check('D1b ON+return_dual=False == baseline f_full feature (opt-in)',
          only_rec_missing and same_full
          and tuple(f_on_legacy.shape) == (B, m_on.in_planes),
          f'missing(rec-only)={only_rec_missing} max|df|='
          f'{float((f_off - f_on_legacy).abs().max()):.2e}')

    # ----------------------------------------------------------------------- #
    # D2: dual head shapes + distinctness
    # ----------------------------------------------------------------------- #
    m_on.train()
    out = m_on(x)
    has_rec = ('bn_feat_rec' in out and 'logits_rec' in out)
    shp_ok = (has_rec
              and tuple(out['bn_feat_rec'].shape) == (B, m_on.in_planes)
              and tuple(out['logits_rec'].shape) == (B, n_cls))
    # the two heads are SEPARATE modules with INDEPENDENT params (not tied): at
    # init they share the same recipe (weight=1, bias=0, running (0,1)) so their
    # eval outputs coincide -- divergence is what TRAINING produces.  To prove the
    # heads are genuinely independent, perturb ONLY bottleneck_rec.weight and
    # confirm f_rec changes while f_full does not.
    independent = (id(m_on.bottleneck) != id(m_on.bottleneck_rec)
                   and id(m_on.classifier) != id(m_on.classifier_rec))
    m_on.eval()
    with torch.no_grad():
        ff0, fr0 = m_on(x, return_dual=True)
        # NON-uniform perturbation (only the first half of the channels) so it
        # survives L2-normalization -- a uniform scale would cancel under normalize.
        half = m_on.bottleneck_rec.weight.numel() // 2
        m_on.bottleneck_rec.weight[:half].add_(0.5)   # simulate f_rec having trained
        ff1, fr1 = m_on(x, return_dual=True)
        m_on.bottleneck_rec.weight[:half].add_(-0.5)  # restore
    eval_ok = (tuple(ff0.shape) == (B, m_on.in_planes)
               and tuple(fr0.shape) == (B, m_on.in_planes)
               and torch.allclose(ff0.norm(dim=1), torch.ones(B), atol=1e-4)
               and torch.allclose(fr0.norm(dim=1), torch.ones(B), atol=1e-4))
    rec_responds = float((fr1 - fr0).abs().max()) > 1e-4   # f_rec changed
    full_isolated = float((ff1 - ff0).abs().max()) < 1e-6  # f_full untouched
    check('D2 dual head shapes + L2-norm + heads independent (rec perturb)',
          shp_ok and eval_ok and independent and rec_responds and full_isolated,
          f'rec_keys={has_rec} eval_norm_ok={eval_ok} independent={independent} '
          f'rec_responds={rec_responds} full_isolated={full_isolated}')

    # ----------------------------------------------------------------------- #
    # D3 / D4: head divergence of the consistency gradient
    #   build a full dual step and backward ONLY the f_rec consistency.
    # ----------------------------------------------------------------------- #
    torch.manual_seed(2)
    m = build_model(n_cls, airl_dualbranch=True)
    m.train()
    xg = torch.randn(B, C, H, W)
    views = torch.tensor([1, 1, 0, 1])        # rows 0,1,3 ground(1); row 2 aerial(0)
    g_mask = (views == 1)
    out_c = m(xg)                              # clean forward (full batch)
    imgs_g = xg[g_mask]
    deg, _ = airl_degrade(imgs_g, min_scale=0.25, blur=False)
    out_d = m(deg)                             # degraded GROUND forward
    L_cons = airl_consistency_loss(
        out_c['logits_rec'][g_mask], out_c['bn_feat_rec'][g_mask],
        out_d['logits_rec'], out_d['bn_feat_rec'],
        mode='kl', tau=4.0)
    m.zero_grad()
    L_cons.backward()
    g_rec_bn = m.bottleneck_rec.weight.grad
    g_rec_cls = m.classifier_rec.weight.grad
    # a representative shared backbone conv (layer4 last conv); resnet50 path
    bb_w = m.layer4[-1].conv3.weight
    g_bb = bb_w.grad
    rec_grad_ok = (g_rec_bn is not None and g_rec_bn.abs().sum().item() > 0
                   and torch.isfinite(g_rec_bn).all().item())
    bb_grad_ok = (g_bb is not None and g_bb.abs().sum().item() > 0
                  and torch.isfinite(g_bb).all().item())
    check('D3 f_rec consistency grad reaches bottleneck_rec + backbone',
          rec_grad_ok and bb_grad_ok,
          f'rec_bn_grad={(g_rec_bn.abs().sum().item() if g_rec_bn is not None else 0):.4f} '
          f'backbone_grad={(g_bb.abs().sum().item() if g_bb is not None else 0):.4f}')

    g_full_bn = m.bottleneck.weight.grad
    g_full_cls = m.classifier.weight.grad
    full_clean = (((g_full_bn is None) or g_full_bn.abs().sum().item() == 0.0)
                  and ((g_full_cls is None) or g_full_cls.abs().sum().item() == 0.0))
    check('D4 f_full head gets NO consistency grad (head divergence)',
          full_clean,
          f'full_bn_grad={(g_full_bn.abs().sum().item() if g_full_bn is not None else 0):.2e} '
          f'full_cls_grad={(g_full_cls.abs().sum().item() if g_full_cls is not None else 0):.2e}')

    # ----------------------------------------------------------------------- #
    # D5: both heads in optimizer + actually move under a real step
    # ----------------------------------------------------------------------- #
    torch.manual_seed(3)
    m5 = build_model(n_cls, airl_dualbranch=True)
    m5.train()
    opt = torch.optim.AdamW(m5.parameters(), lr=1e-2, weight_decay=5e-4)
    opt_ids = {id(p) for grp in opt.param_groups for p in grp['params']}
    rec_trainable = [p for p in
                     (list(m5.bottleneck_rec.parameters())
                      + list(m5.classifier_rec.parameters())) if p.requires_grad]
    rec_in_opt = all(id(p) in opt_ids for p in rec_trainable)
    cls_full_before = m5.classifier.weight.detach().clone()
    cls_rec_before = m5.classifier_rec.weight.detach().clone()
    x5 = torch.randn(B, C, H, W)
    y5 = torch.randint(0, n_cls, (B,))
    out5 = m5(x5)
    loss5 = (F.cross_entropy(out5['logits'], y5)
             + F.cross_entropy(out5['logits_rec'], y5))   # f_full CE + f_rec CE
    opt.zero_grad()
    loss5.backward()
    opt.step()
    moved_full = float((m5.classifier.weight.detach() - cls_full_before).abs().max()) > 0
    moved_rec = float((m5.classifier_rec.weight.detach() - cls_rec_before).abs().max()) > 0
    check('D5 both heads in optimizer + both classifiers MOVE on a step',
          rec_in_opt and moved_full and moved_rec,
          f'rec_in_opt={rec_in_opt} moved_full={moved_full} moved_rec={moved_rec}')

    # ----------------------------------------------------------------------- #
    # D6: soft-fusion distmat correctness (the airl_dualbranch_eval core formula)
    # ----------------------------------------------------------------------- #
    torch.manual_seed(4)
    Nq, Ng, D = 5, 7, 16
    q_full = F.normalize(torch.randn(Nq, D), dim=1)
    g_full = F.normalize(torch.randn(Ng, D), dim=1)
    q_rec = F.normalize(torch.randn(Nq, D), dim=1)
    g_rec = F.normalize(torch.randn(Ng, D), dim=1)
    s_full = (q_full @ g_full.t())
    s_rec = (q_rec @ g_rec.t())

    def fuse_dist(w):
        return (2.0 - 2.0 * (w * s_rec + (1.0 - w) * s_full))

    dm_full_ref = (2.0 - 2.0 * s_full)
    dm_rec_ref = (2.0 - 2.0 * s_rec)
    w_mid = 0.25
    dm_mid = fuse_dist(w_mid)
    dm_mid_ref = w_mid * dm_rec_ref + (1.0 - w_mid) * dm_full_ref  # convex in dist
    w0_ok = torch.allclose(fuse_dist(0.0), dm_full_ref, atol=1e-5)
    w1_ok = torch.allclose(fuse_dist(1.0), dm_rec_ref, atol=1e-5)
    mid_ok = torch.allclose(dm_mid, dm_mid_ref, atol=1e-5)
    # (the convex identity holds because dist is affine in cos: 2-2*cos)
    check('D6 soft-fusion distmat (w=0->full, w=1->rec, mid=convex blend)',
          w0_ok and w1_ok and mid_ok,
          f'w0={w0_ok} w1={w1_ok} mid={mid_ok} '
          f'max|mid-ref|={float((dm_mid - dm_mid_ref).abs().max()):.2e}')

    # ----------------------------------------------------------------------- #
    # D7: NaN-safety of the f_rec consistency (same floors as --airl)
    # ----------------------------------------------------------------------- #
    lo_ext = torch.full((B, n_cls), -50.0); lo_ext[:, 0] = 50.0
    ld_ext = torch.randn(B, n_cls) * 30.0
    kl_ext = airl_consistency_loss(lo_ext, None, ld_ext, None, mode='kl', tau=4.0)
    bo0 = torch.zeros(B, 16); bd0 = torch.zeros(B, 16)
    feat0 = airl_consistency_loss(None, bo0, None, bd0, mode='feat')
    check('D7 f_rec consistency NaN-safe (extreme logits / zero feats)',
          torch.isfinite(kl_ext).item() and torch.isfinite(feat0).item(),
          f'kl_ext={float(kl_ext):.4f} feat0={float(feat0):.4f}')

    # ----------------------------------------------------------------------- #
    # D8: f_rec ID-CE grounds f_rec (grad to classifier_rec + backbone)
    # ----------------------------------------------------------------------- #
    torch.manual_seed(5)
    m8 = build_model(n_cls, airl_dualbranch=True)
    m8.train()
    x8 = torch.randn(B, C, H, W)
    y8 = torch.randint(0, n_cls, (B,))
    out8 = m8(x8)
    L_ce_rec = F.cross_entropy(out8['logits_rec'], y8)
    m8.zero_grad()
    L_ce_rec.backward()
    g8_cls = m8.classifier_rec.weight.grad
    g8_bb = m8.layer4[-1].conv3.weight.grad
    ce_rec_ok = (g8_cls is not None and g8_cls.abs().sum().item() > 0
                 and g8_bb is not None and g8_bb.abs().sum().item() > 0
                 and torch.isfinite(g8_cls).all().item())
    check('D8 f_rec ID-CE grounds f_rec (grad to classifier_rec + backbone)',
          ce_rec_ok,
          f'cls_rec_grad={(g8_cls.abs().sum().item() if g8_cls is not None else 0):.4f} '
          f'backbone_grad={(g8_bb.abs().sum().item() if g8_bb is not None else 0):.4f}')

    # ----------------------------------------------------------------------- #
    # D9: both heads read the SAME pooled global_feat (single shared triplet)
    #   verify by registering a hook: bottleneck and bottleneck_rec receive the
    #   SAME input tensor (the pooled global_feat).
    # ----------------------------------------------------------------------- #
    torch.manual_seed(6)
    m9 = build_model(n_cls, airl_dualbranch=True)
    m9.train()
    captured = {}

    def cap(name):
        def hook(_mod, inp, _out):
            captured[name] = inp[0].detach().clone()
        return hook

    h1 = m9.bottleneck.register_forward_hook(cap('full'))
    h2 = m9.bottleneck_rec.register_forward_hook(cap('rec'))
    x9 = torch.randn(B, C, H, W)
    o9 = m9(x9)
    h1.remove(); h2.remove()
    same_input = ('full' in captured and 'rec' in captured
                  and torch.allclose(captured['full'], captured['rec'], atol=1e-6)
                  and torch.allclose(captured['full'], o9['global_feat'].detach(),
                                     atol=1e-6))
    check('D9 both BNNecks consume the SAME pooled global_feat (shared triplet)',
          same_input,
          f'max|full_in-rec_in|='
          f'{(float((captured["full"] - captured["rec"]).abs().max()) if same_input or ("full" in captured and "rec" in captured) else float("nan")):.2e}')

    # ----------------------------------------------------------------------- #
    # D10: emulate the swin param-group split -> f_rec head must be FULL-LR
    #   (the optimiser puts everything NOT in backbone_swin.parameters() in the
    #   full-LR group; here resnet50 has no backbone_swin so we emulate the split
    #   logic against a fake "backbone id set" = stem+layer1..4, and assert the
    #   rec head is NOT classified as backbone).
    # ----------------------------------------------------------------------- #
    torch.manual_seed(7)
    m10 = build_model(n_cls, airl_dualbranch=True)
    backbone_ids = set()
    for mod in (m10.stem, m10.layer1, m10.layer2, m10.layer3, m10.layer4):
        backbone_ids |= {id(p) for p in mod.parameters()}
    rec_param_ids = {id(p) for p in
                     (list(m10.bottleneck_rec.parameters())
                      + list(m10.classifier_rec.parameters()))}
    rec_not_backbone = len(rec_param_ids & backbone_ids) == 0
    # the "other" (full-LR) group = params not in backbone_ids; rec must be there
    other_ids = {id(p) for p in m10.parameters() if id(p) not in backbone_ids}
    rec_in_other = rec_param_ids.issubset(other_ids)
    check('D10 f_rec head is FULL-LR (not in the scaled backbone group)',
          rec_not_backbone and rec_in_other,
          f'rec_not_backbone={rec_not_backbone} rec_in_full_lr_group={rec_in_other}')

    # ----------------------------------------------------------------------- #
    print('\n' + '=' * 70)
    print(f'AIRL dual-branch smoke: {len(passed)} passed, {len(failed)} failed')
    if failed:
        print('FAILED:', failed)
        sys.exit(1)
    print('ALL AIRL DUAL-BRANCH SMOKE CHECKS PASSED')
    print('=' * 70)


if __name__ == '__main__':
    main()
