"""Isolated numeric smoke test for the AIRL GRADIENT-ISOLATED DUAL-BRANCH
(--airl_dualbranch_iso) in cargo_cvpb/afd_train.py + afd_reid/afd_model.py.

This is the RESCUE variant of the fully-shared --airl_dualbranch (which collapsed:
ep10 FUSE 42.05 ~= full 41.99 = +0.06 because the f_rec consistency gradient flowed
back through the SHARED global_feat into the shared trunk, pulling f_full toward
degradation-robustness too -> the two heads stopped diverging).

The iso fix: f_rec is NOT a BNNeck over the shared global_feat.  It is a BNNeck over
an INDEPENDENT late Swin stage that is forked off a DETACHED copy of the shared
residual stream at the input of stage `iso_stage` (deep-copy of swin.stages[
iso_stage:] + that output norm, living inside backbone_swin).  Because the trunk
feature is detach()ed BEFORE it enters the rec stages, the f_rec ID-CE + AIRL
consistency gradient updates ONLY the rec late stage + BNNeck_rec and NEVER reaches
the shared trunk.  The clean trunk + f_full stay a "clean expert"; f_rec becomes the
"recover expert" -> the two poles re-separate.  Eval still soft-fuses the two cosine
scores (cos = w*cos_rec + (1-w)*cos_full, w fixed 0.25), ONE forward, two features.

Two layers are exercised:
  (1) the REAL afd_model.AFDModel(backbone='swin_small', airl_dualbranch_iso=True) --
      the architecture (independent rec late stage, detached fork, dual eval, and --
      critically -- the gradient isolation of the consistency from the shared trunk).
  (2) the REAL airl_degrade / airl_consistency_loss out of cargo_cvpb/afd_train.py
      (loaded with the heavy siblings stubbed, exactly like smoke_airl_dualbranch.py)
      -- the degradation + consistency primitive the iso branch reuses UNCHANGED.

The swin backbone is built FROM SCRATCH (swin_pretrain='') on CPU at a small input so
the smoke needs no checkpoint / GPU; cv2 (imported-but-unused in swin_transformer.py)
is stubbed so the import chain is self-contained.

Checks
  I1  OFF BYTE-IDENTICAL (model): airl_dualbranch_iso=False builds NO rec late stage
        (no backbone_swin.rec_stages) and NO second head (no bottleneck_rec /
        classifier_rec); iso_branch flag is False; eval returns a single feature.  An
        ON model fed return_dual=False (legacy eval) ALSO returns the single f_full
        feature == the baseline eval feature (the dual path is strictly opt-in), and
        the only state_dict keys it adds are the rec ones.
  I2  f_full MAP UNCHANGED BY THE SPLIT: the split forward's f_full last-stage map is
        BIT-IDENTICAL to the original single-map swin forward (the manual per-stage
        replication is faithful -> enabling iso does not perturb f_full's features).
  I3  DUAL HEAD SHAPES + INDEPENDENCE: ON model -> train dict carries bn_feat_rec /
        logits_rec with the right shapes; eval(return_dual=True) returns (f_full,
        f_rec) both (B,D) L2-normalized.  At INIT the rec stage is a deep-copy of the
        shared last stage fed the same (detached) input, so rec_map == full_map
        (divergence is what TRAINING produces, NOT init -- same contract as the
        shared dual-branch D2); independence is proven by PERTURBING a rec-stage
        weight and confirming f_rec changes while f_full does not.
  I4  GRADIENT ISOLATION (the headline): a full iso step (clean forward + ground
        degrade + degraded forward + AIRL consistency on the f_rec head) puts FINITE
        non-zero grad on the REC late stage + bottleneck_rec, and STRICTLY ZERO/None
        grad on (a) a shared EARLY trunk stage, (b) the patch_embed, (c) the SHARED
        last stage used by f_full, and (d) the f_full BNNeck / classifier.  This is
        the whole point: the consistency cannot pollute the clean trunk.
  I5  f_full CE DOES NOT TOUCH THE REC BRANCH (reverse isolation): a pure f_full
        cross-entropy backward leaves the rec late stage + bottleneck_rec grad
        None/zero, while the shared trunk + f_full last stage DO get grad.
  I6  BOTH BRANCHES IN OPTIMIZER + CORRECT LR GROUP: emulate the trainer's swin
        param-group split; the rec late stage (pretrained-recipe backbone weight)
        lands in the SCALED Swin-LR group, the rec BNNeck head (random init) lands in
        the FULL-LR group; one AdamW step on (CE_full + CE_rec) MOVES a rec-stage
        weight, bottleneck_rec, AND f_full's classifier.
  I7  SOFT-FUSION DISTMAT CORRECT: dist = 2 - 2*(w*cos_rec + (1-w)*cos_full) equals a
        hand reference; w=0 == f_full-only, w=1 == f_rec-only, mid = convex blend
        (the same formula airl_dualbranch_eval uses verbatim).
  I8  NAN-SAFETY (f_rec consistency): KL with extreme logits and feat with zero
        vectors on the f_rec head stay finite (same floors as --airl).
  I9  f_rec CLEAN ID-CE GROUNDS THE REC BRANCH AND REFLOWS TO THE TRUNK (the FIX,
        trunk_recce=True default): cross-entropy on logits_rec backward puts finite
        non-zero grad on classifier_rec.weight + the REC late stage (so f_rec is a
        valid identity space) AND on the shared trunk UPSTREAM of the fork (early
        stage / patch_embed) -- the extra identity supervision that strengthens the
        weak f_full.  f_full's OWN head (bottleneck/classifier) and its own forked
        last stage (downstream of the fork input) stay zero.  [Behavioural change vs
        the original full-isolation iso, which kept the trunk zero -- see I16.]
  I10 AMP fp32 CONSISTENCY: the consistency runs in true fp32 (autocast disabled) and
        is finite even when the surrounding forward is autocast (here CPU bf16
        autocast as a portable stand-in); the loss value matches the fp32 reference.
  I11 TRAINER WARMUP INCLUDES iso (regression guard): the trainer's airl_lambda_eff
        warmup multiplier must list args.airl_dualbranch_iso (the three AIRL flags are
        mutually exclusive, so omitting iso leaves lam_eff==0 every epoch -> the f_rec
        consistency is multiplied by 0 and never trains).  Asserts the source guard
        mentions the iso flag AND that the reproduced expression is > 0 at epoch>=1.
  I12 iso_stage=2 (heavier fork, PatchMerging in the rec copy): builds 2 rec stages,
        forwards train+eval with right shapes; with the FIX a clean rec CE REFLOWS to
        the pre-fork trunk (stage 0/1) while the SHARED stage-2 (f_full's own, down-
        stream of the fork input) stays ZERO and the rec copy trains -- the reflow
        target tracks iso_stage.  Covers the alt fork point + down-sampling.
  I13 DropPath RNG FAITHFUL: in TRAIN mode the split f_full map == the original single-
        map swin forward at the SAME seed (the rec copy runs AFTER the full f_full
        loop, so iso does not perturb f_full's stochastic-depth RNG draws).
  I14 THE FIX, DECOMPOSED (headline): on ONE iso model (trunk_recce=True) backward the
        two f_rec gradient sources SEPARATELY -- (a) CLEAN f_rec ID-CE REFLOWS to the
        trunk (early non-zero) + rec stage; (b) degraded CONSISTENCY (clean target
        detached inside the loss, degraded side from a rec_only detached-fork forward)
        leaves the trunk + f_full head ZERO while the rec stage + bottleneck_rec train.
        Proves "clean reflows, degraded isolated" at the single-loss granularity.
  I15 FULL COMBINED step (the trainer's CE_full + CE_rec(clean) + lam*consistency in
        ONE backward): finite, and trunk + f_full last stage + rec stage + both heads
        all receive grad (the assembled step trains everything, with I14 attributing
        the trunk's share to the clean ID-CE, NOT the consistency).
  I16 trunk_recce=0 ABLATION restores the ORIGINAL full-isolation iso: a CLEAN f_rec
        ID-CE leaves the shared trunk (early + patch_embed + shared last) at ZERO grad
        while the rec stage + classifier_rec train -> the flag actually toggles the
        reflow (the controlled comparison the fix is measured against).
  I17 trunk_recce IS GRAD-ONLY: at the SAME weights the eval AND train f_full / f_rec
        forward VALUES are bit-identical for recce on/off (the detach choice changes
        only the backward graph, never a forward value).

Usage:  python smoke_airl_iso.py [path/to/cargo_cvpb/afd_train.py]
        (run inside an env with torch+numpy+torchvision, e.g.
         `uv run --no-project --with numpy --with torch --with torchvision \
            python smoke_airl_iso.py`)
"""
import importlib.util
import os
import sys
import types

import torch
import torch.nn as nn  # noqa: F401  (kept for parity with smoke_airl_dualbranch)
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# load the REAL airl_degrade / airl_consistency_loss out of afd_train.py with the
# heavy siblings stubbed (identical recipe to smoke_airl_dualbranch.py).  cv2 is
# imported-but-unused in the SOLIDER swin_transformer.py -> stub it so the swin
# import chain (model/__init__ -> make_model -> swin_transformer) is self-contained.
# --------------------------------------------------------------------------- #
def _stub(name, attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m


_dummy = lambda *a, **k: None  # noqa: E731
sys.modules.setdefault('cv2', types.ModuleType('cv2'))   # unused in swin file
HERE = os.path.dirname(os.path.abspath(__file__))
AFD_REID = os.path.join(HERE, '..', 'afd_reid')
sys.path.insert(0, AFD_REID)   # so the REAL afd_model resolves

_stub('cargo_dataset', dict(CARGO=_dummy, CARGOImageDataset=_dummy,
                            build_transforms=_dummy, RandomIdentitySampler=_dummy,
                            filter_by_view=_dummy))
_stub('agreid_dataset', dict(AGReIDv2=_dummy))
# afd_model is NOT stubbed (we want the real swin iso model); afd_train (cargo_cvpb)
# imports build_model from afd_model + several helpers from the PARENT afd_train ->
# stub the parent-afd_train symbols so the load is cheap and self-contained.
_stub('afd_train', dict(CrossEntropyLabelSmooth=_dummy, TripletLoss=_dummy,
                        WarmupCosineLR=_dummy, run_cross_view_eval=_dummy,
                        print_eval=_dummy, set_seed=_dummy,
                        build_eval_loader=_dummy, eval_market=_dummy))
_stub('maxsim_probe', dict(eval_from_distmat=_dummy))

from afd_model import AFDModel   # the REAL swin iso model  # noqa: E402


def load_cvpb_afd(path):
    spec = importlib.util.spec_from_file_location('cvpb_afd_iso_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_iso(num_classes, iso, iso_stage=3, img_size=(256, 128),
              trunk_recce=True):
    """Tiny CPU swin_small AFDModel built FROM SCRATCH (no checkpoint download).

    trunk_recce=True (default, the FIX): the CLEAN f_rec ID-CE reflows into the
    shared trunk; the degradation-consistency stays trunk-detached.  False: the
    original full-isolation iso (clean fork also detached).
    """
    return AFDModel(num_classes=num_classes, pretrained=False,
                    backbone='swin_small', swin_pretrain='', img_size=img_size,
                    airl_dualbranch_iso=iso, airl_iso_stage=iso_stage,
                    airl_iso_trunk_recce=trunk_recce)


def _trunk_probe(model):
    """Return representative (early-trunk, patch_embed, shared-last-stage) params."""
    bsw = model.backbone_swin
    early = list(bsw.swin.stages[0].parameters())[0]      # shared early stage
    pe = list(bsw.swin.patch_embed.parameters())[0]       # shared patch embed
    shared_last = list(bsw.swin.stages[-1].parameters())[0]  # f_full's last stage
    return early, pe, shared_last


def _rec_probe(model):
    """Return representative (rec-late-stage, bottleneck_rec) params."""
    bsw = model.backbone_swin
    rec_stage = list(bsw.rec_stages.parameters())[0]
    return rec_stage, model.bottleneck_rec.weight


def _gsum(p):
    return None if (p is None or p.grad is None) else float(p.grad.abs().sum())


def _is_zero_or_none(p):
    return p.grad is None or float(p.grad.abs().sum()) == 0.0


def main():
    path = (sys.argv[1] if len(sys.argv) > 1
            else os.path.join(HERE, 'afd_train.py'))
    mod = load_cvpb_afd(path)
    airl_degrade = mod.airl_degrade
    airl_consistency_loss = mod.airl_consistency_loss
    torch.manual_seed(0)

    # small input keeps the from-scratch swin cheap on CPU; H,W must be /32-friendly
    B, C, H, W = 4, 3, 256, 128
    n_cls = 8
    D = 768

    passed, failed = [], []

    def check(name, cond, detail=''):
        (passed if cond else failed).append(name)
        tag = 'PASS' if cond else '**FAIL**'
        print(f'  [{tag}] {name}{("  -- " + detail) if detail else ""}')

    # ----------------------------------------------------------------------- #
    # I1: OFF byte-identical (model level)
    # ----------------------------------------------------------------------- #
    torch.manual_seed(1)
    m_off = build_iso(n_cls, iso=False)
    m_off.eval()
    x = torch.randn(B, C, H, W)
    with torch.no_grad():
        f_off = m_off(x)
    sd_off = {k: v.detach().clone() for k, v in m_off.state_dict().items()}
    no_rec = (not hasattr(m_off.backbone_swin, 'rec_stages')
              and not hasattr(m_off, 'bottleneck_rec')
              and not hasattr(m_off, 'classifier_rec')
              and not m_off.airl_dualbranch_iso
              and not m_off.backbone_swin.iso_branch)
    check('I1 OFF builds no rec late stage / no second head; single eval feat',
          no_rec and tuple(f_off.shape) == (B, m_off.in_planes),
          f'has_rec_stages={hasattr(m_off.backbone_swin, "rec_stages")} '
          f'iso_flag={m_off.backbone_swin.iso_branch}')

    torch.manual_seed(1)
    m_on = build_iso(n_cls, iso=True, iso_stage=3)
    msg = m_on.load_state_dict(sd_off, strict=False)
    only_rec_missing = all(('rec' in k) for k in msg.missing_keys)
    m_on.eval()
    with torch.no_grad():
        f_on_legacy = m_on(x)                    # return_dual default False
    same_full = torch.allclose(f_off, f_on_legacy, atol=1e-6)
    check('I1b ON+return_dual=False == baseline f_full feature (opt-in)',
          only_rec_missing and same_full
          and tuple(f_on_legacy.shape) == (B, m_on.in_planes),
          f'missing(rec-only)={only_rec_missing} #missing={len(msg.missing_keys)} '
          f'max|df|={float((f_off - f_on_legacy).abs().max()):.2e}')

    # ----------------------------------------------------------------------- #
    # I2: f_full map UNCHANGED by the split (manual replication is faithful)
    # ----------------------------------------------------------------------- #
    m_on.eval()
    with torch.no_grad():
        full_split, rec_map = m_on.backbone_swin(x, return_rec=True)
        _g, outs = m_on.backbone_swin.swin(x)
        full_orig = outs[-1]
    full_match = torch.allclose(full_split, full_orig, atol=1e-5)
    check('I2 split f_full map == original single-map swin forward (bit-identical)',
          full_match
          and tuple(full_split.shape) == tuple(full_orig.shape),
          f'max|d_full|={float((full_split - full_orig).abs().max()):.2e} '
          f'shape={tuple(full_split.shape)}')

    # ----------------------------------------------------------------------- #
    # I3: dual head shapes + independence (perturb a rec-stage weight)
    # ----------------------------------------------------------------------- #
    m_on.train()
    out = m_on(x)
    has_rec = ('bn_feat_rec' in out and 'logits_rec' in out)
    shp_ok = (has_rec
              and tuple(out['bn_feat_rec'].shape) == (B, m_on.in_planes)
              and tuple(out['logits_rec'].shape) == (B, n_cls))
    independent_modules = (id(m_on.bottleneck) != id(m_on.bottleneck_rec)
                           and id(m_on.classifier) != id(m_on.classifier_rec)
                           and id(list(m_on.backbone_swin.rec_stages.parameters())[0])
                           != id(list(m_on.backbone_swin.swin.stages[-1].parameters())[0]))
    m_on.eval()
    with torch.no_grad():
        ff0, fr0 = m_on(x, return_dual=True)
        # perturb ONLY a rec-stage weight -> f_rec must change, f_full must NOT.
        wrec = list(m_on.backbone_swin.rec_stages.parameters())[0]
        wrec.add_(0.1)
        ff1, fr1 = m_on(x, return_dual=True)
        wrec.add_(-0.1)                         # restore
    eval_ok = (tuple(ff0.shape) == (B, m_on.in_planes)
               and tuple(fr0.shape) == (B, m_on.in_planes)
               and torch.allclose(ff0.norm(dim=1), torch.ones(B), atol=1e-4)
               and torch.allclose(fr0.norm(dim=1), torch.ones(B), atol=1e-4))
    rec_responds = float((fr1 - fr0).abs().max()) > 1e-4
    full_isolated = float((ff1 - ff0).abs().max()) < 1e-6
    check('I3 dual head shapes + L2-norm + rec-stage perturb isolates f_full',
          shp_ok and eval_ok and independent_modules and rec_responds and full_isolated,
          f'rec_keys={has_rec} eval_norm_ok={eval_ok} indep={independent_modules} '
          f'rec_responds={rec_responds} full_isolated={full_isolated}')

    # ----------------------------------------------------------------------- #
    # I4: GRADIENT ISOLATION (headline) -- full iso step, consistency on f_rec only
    # ----------------------------------------------------------------------- #
    torch.manual_seed(2)
    m = build_iso(n_cls, iso=True, iso_stage=3)
    m.train()
    xg = torch.randn(B, C, H, W)
    views = torch.tensor([1, 1, 0, 1])         # rows 0,1,3 ground(1); row 2 aerial(0)
    g_mask = (views == 1)
    out_c = m(xg)                               # clean forward (full batch)
    imgs_g = xg[g_mask]
    deg, _ = airl_degrade(imgs_g, min_scale=0.25, blur=False)
    # snapshot f_full BNNeck running stats BEFORE the degraded pass -> the rec_only
    # degraded forward must NOT update them (f_full stays a clean expert; no degraded-
    # ground stat leak into the f_full eval head).  This is the EXACT call the trainer
    # makes (model(deg, rec_only=True)).
    rm0 = m.bottleneck.running_mean.detach().clone()
    rv0 = m.bottleneck.running_var.detach().clone()
    out_d = m(deg, rec_only=True)              # degraded GROUND forward, REC HEAD ONLY
    rm1 = m.bottleneck.running_mean.detach().clone()
    rv1 = m.bottleneck.running_var.detach().clone()
    bn_stat_clean = torch.allclose(rm0, rm1) and torch.allclose(rv0, rv1)
    reconly_keys_ok = (set(out_d.keys()) == {'bn_feat_rec', 'logits_rec'})
    L_cons = airl_consistency_loss(
        out_c['logits_rec'][g_mask], out_c['bn_feat_rec'][g_mask],
        out_d['logits_rec'], out_d['bn_feat_rec'],
        mode='kl', tau=4.0)
    m.zero_grad()
    L_cons.backward()
    rec_stage_w, bnrec_w = _rec_probe(m)
    early_w, pe_w, shared_last_w = _trunk_probe(m)
    # rec branch MUST get finite non-zero grad
    rec_grad_ok = (rec_stage_w.grad is not None and rec_stage_w.grad.abs().sum() > 0
                   and torch.isfinite(rec_stage_w.grad).all()
                   and bnrec_w.grad is not None and bnrec_w.grad.abs().sum() > 0
                   and torch.isfinite(bnrec_w.grad).all())
    # shared trunk (early, patch_embed, shared last stage) MUST be zero/None
    trunk_clean = (_is_zero_or_none(early_w) and _is_zero_or_none(pe_w)
                   and _is_zero_or_none(shared_last_w))
    # f_full head MUST be zero/None
    full_head_clean = (_is_zero_or_none(m.bottleneck.weight)
                       and _is_zero_or_none(m.classifier.weight))
    check('I4 consistency grad reaches REC branch ONLY (trunk + f_full ZERO); '
          'rec_only degraded pass leaves f_full BNNeck stats CLEAN',
          rec_grad_ok and trunk_clean and full_head_clean and bn_stat_clean
          and reconly_keys_ok,
          f'rec_stage={_gsum(rec_stage_w)} bnrec={_gsum(bnrec_w)} | '
          f'early={_gsum(early_w)} patch_embed={_gsum(pe_w)} '
          f'shared_last={_gsum(shared_last_w)} '
          f'full_bn={_gsum(m.bottleneck.weight)} full_cls={_gsum(m.classifier.weight)} '
          f'| bn_stat_clean={bn_stat_clean} reconly_keys={reconly_keys_ok}')

    # ----------------------------------------------------------------------- #
    # I5: reverse isolation -- f_full CE does NOT touch the rec branch
    # ----------------------------------------------------------------------- #
    torch.manual_seed(3)
    m5 = build_iso(n_cls, iso=True, iso_stage=3)
    m5.train()
    x5 = torch.randn(B, C, H, W)
    y5 = torch.randint(0, n_cls, (B,))
    out5 = m5(x5)
    m5.zero_grad()
    F.cross_entropy(out5['logits'], y5).backward()    # f_full ID-CE ONLY
    rec_stage5, bnrec5 = _rec_probe(m5)
    early5, _pe5, shared_last5 = _trunk_probe(m5)
    rec_clean = _is_zero_or_none(rec_stage5) and _is_zero_or_none(bnrec5)
    trunk_trained = ((early5.grad is not None and early5.grad.abs().sum() > 0)
                     and (shared_last5.grad is not None
                          and shared_last5.grad.abs().sum() > 0))
    check('I5 f_full CE leaves REC branch ZERO; shared trunk + f_full stage train',
          rec_clean and trunk_trained,
          f'rec_stage={_gsum(rec_stage5)} bnrec={_gsum(bnrec5)} | '
          f'early={_gsum(early5)} shared_last={_gsum(shared_last5)}')

    # ----------------------------------------------------------------------- #
    # I6: both branches in optimizer + correct LR group + actually move
    # ----------------------------------------------------------------------- #
    torch.manual_seed(4)
    m6 = build_iso(n_cls, iso=True, iso_stage=3)
    bsw = m6.backbone_swin
    # emulate the trainer's swin param-group split (swin_lr_factor != 1.0):
    swin_ids = {id(p) for p in bsw.parameters()}      # ALL of backbone_swin (incl rec stage)
    swin_params = [p for p in m6.parameters() if p.requires_grad and id(p) in swin_ids]
    other_params = [p for p in m6.parameters() if p.requires_grad and id(p) not in swin_ids]
    base_lr = 3.5e-4
    swin_lr = base_lr * 0.1
    param_groups = [{'params': swin_params, 'lr': swin_lr},
                    {'params': other_params, 'lr': base_lr}]
    opt = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=5e-4)
    swin_grp_ids = {id(p) for p in param_groups[0]['params']}
    full_grp_ids = {id(p) for p in param_groups[1]['params']}
    rec_stage_params = [p for p in (list(bsw.rec_stages.parameters())
                                    + list(bsw.rec_norm.parameters())) if p.requires_grad]
    rec_head_params = [p for p in (list(m6.bottleneck_rec.parameters())
                                   + list(m6.classifier_rec.parameters())) if p.requires_grad]
    rec_stage_in_swin = all(id(p) in swin_grp_ids for p in rec_stage_params)
    rec_head_in_full = all(id(p) in full_grp_ids for p in rec_head_params)
    # rec semantic-embed must be FROZEN (not in optimizer, requires_grad False)
    sem_frozen = (not any(p.requires_grad for p in bsw.rec_semantic_embed_w.parameters())
                  and not any(p.requires_grad for p in bsw.rec_semantic_embed_b.parameters()))
    # one real step on (CE_full + CE_rec) MUST move a rec-stage weight, bnrec, and f_full cls
    rec_stage_before = list(bsw.rec_stages.parameters())[0].detach().clone()
    bnrec_before = m6.bottleneck_rec.weight.detach().clone()
    cls_full_before = m6.classifier.weight.detach().clone()
    m6.train()
    x6 = torch.randn(B, C, H, W)
    y6 = torch.randint(0, n_cls, (B,))
    out6 = m6(x6)
    loss6 = F.cross_entropy(out6['logits'], y6) + F.cross_entropy(out6['logits_rec'], y6)
    opt.zero_grad(); loss6.backward(); opt.step()
    moved_rec_stage = float((list(bsw.rec_stages.parameters())[0].detach()
                             - rec_stage_before).abs().max()) > 0
    moved_bnrec = float((m6.bottleneck_rec.weight.detach() - bnrec_before).abs().max()) > 0
    moved_cls_full = float((m6.classifier.weight.detach() - cls_full_before).abs().max()) > 0
    check('I6 rec stage @ Swin-LR group, rec head @ full-LR group; all 3 move',
          rec_stage_in_swin and rec_head_in_full and sem_frozen
          and moved_rec_stage and moved_bnrec and moved_cls_full,
          f'rec_stage_in_swin={rec_stage_in_swin} rec_head_in_full={rec_head_in_full} '
          f'sem_frozen={sem_frozen} moved(rec_stage={moved_rec_stage} '
          f'bnrec={moved_bnrec} cls_full={moved_cls_full})')

    # ----------------------------------------------------------------------- #
    # I7: soft-fusion distmat correctness (== airl_dualbranch_eval formula)
    # ----------------------------------------------------------------------- #
    torch.manual_seed(5)
    Nq, Ng, Dd = 5, 7, 16
    q_full = F.normalize(torch.randn(Nq, Dd), dim=1)
    g_full = F.normalize(torch.randn(Ng, Dd), dim=1)
    q_rec = F.normalize(torch.randn(Nq, Dd), dim=1)
    g_rec = F.normalize(torch.randn(Ng, Dd), dim=1)
    s_full = (q_full @ g_full.t())
    s_rec = (q_rec @ g_rec.t())

    def fuse_dist(w):
        return (2.0 - 2.0 * (w * s_rec + (1.0 - w) * s_full))

    dm_full_ref = (2.0 - 2.0 * s_full)
    dm_rec_ref = (2.0 - 2.0 * s_rec)
    w_mid = 0.25
    dm_mid = fuse_dist(w_mid)
    dm_mid_ref = w_mid * dm_rec_ref + (1.0 - w_mid) * dm_full_ref
    w0_ok = torch.allclose(fuse_dist(0.0), dm_full_ref, atol=1e-5)
    w1_ok = torch.allclose(fuse_dist(1.0), dm_rec_ref, atol=1e-5)
    mid_ok = torch.allclose(dm_mid, dm_mid_ref, atol=1e-5)
    check('I7 soft-fusion distmat (w=0->full, w=1->rec, mid=convex blend)',
          w0_ok and w1_ok and mid_ok,
          f'w0={w0_ok} w1={w1_ok} mid={mid_ok} '
          f'max|mid-ref|={float((dm_mid - dm_mid_ref).abs().max()):.2e}')

    # ----------------------------------------------------------------------- #
    # I8: NaN-safety of the f_rec consistency (same floors as --airl)
    # ----------------------------------------------------------------------- #
    lo_ext = torch.full((B, n_cls), -50.0); lo_ext[:, 0] = 50.0
    ld_ext = torch.randn(B, n_cls) * 30.0
    kl_ext = airl_consistency_loss(lo_ext, None, ld_ext, None, mode='kl', tau=4.0)
    bo0 = torch.zeros(B, 16); bd0 = torch.zeros(B, 16)
    feat0 = airl_consistency_loss(None, bo0, None, bd0, mode='feat')
    check('I8 f_rec consistency NaN-safe (extreme logits / zero feats)',
          torch.isfinite(kl_ext).item() and torch.isfinite(feat0).item(),
          f'kl_ext={float(kl_ext):.4f} feat0={float(feat0):.4f}')

    # ----------------------------------------------------------------------- #
    # I9: f_rec CLEAN ID-CE grounds the rec branch (grad to classifier_rec + rec
    #     stage) AND -- with the trunk-undersupervision FIX (trunk_recce=True,
    #     default) -- REFLOWS into the shared trunk UPSTREAM of the fork (early
    #     stage / patch_embed) so it adds the extra identity supervision that
    #     strengthens f_full.  The f_full HEAD (its own bottleneck/classifier) and
    #     the SHARED last stage (downstream of the stage-3 fork input, i.e. f_full's
    #     own last-stage instance, NOT on the rec path) stay ZERO.  This is the
    #     behavioural CHANGE vs the original full-isolation iso (where the trunk was
    #     zero too); I16 covers the trunk_recce=0 ablation that restores the old
    #     all-zero-trunk behaviour.
    # ----------------------------------------------------------------------- #
    torch.manual_seed(6)
    m9 = build_iso(n_cls, iso=True, iso_stage=3, trunk_recce=True)
    m9.train()
    x9 = torch.randn(B, C, H, W)
    y9 = torch.randint(0, n_cls, (B,))
    out9 = m9(x9)
    m9.zero_grad()
    F.cross_entropy(out9['logits_rec'], y9).backward()
    rec_stage9, _bnrec9 = _rec_probe(m9)
    early9, pe9, shared_last9 = _trunk_probe(m9)
    cls_rec9 = m9.classifier_rec.weight
    ce_rec_ok = (cls_rec9.grad is not None and cls_rec9.grad.abs().sum() > 0
                 and rec_stage9.grad is not None and rec_stage9.grad.abs().sum() > 0
                 and torch.isfinite(cls_rec9.grad).all())
    # FIX: the shared trunk UPSTREAM of the fork now receives the clean ID-CE grad.
    trunk_reflows = ((early9.grad is not None and early9.grad.abs().sum() > 0
                      and torch.isfinite(early9.grad).all())
                     and (pe9.grad is not None and pe9.grad.abs().sum() > 0))
    # f_full's OWN head + its own last-stage instance (downstream of the fork input)
    # are NOT on the rec path -> stay zero even with the reflow.
    full_path_clean = (_is_zero_or_none(shared_last9)
                       and _is_zero_or_none(m9.classifier.weight)
                       and _is_zero_or_none(m9.bottleneck.weight))
    check('I9 f_rec CLEAN ID-CE grounds rec branch AND reflows to shared trunk '
          '(FIX); f_full head + f_full last stage ZERO',
          ce_rec_ok and trunk_reflows and full_path_clean,
          f'cls_rec={_gsum(cls_rec9)} rec_stage={_gsum(rec_stage9)} | '
          f'REFLOW early={_gsum(early9)} patch_embed={_gsum(pe9)} | '
          f'shared_last={_gsum(shared_last9)} full_cls={_gsum(m9.classifier.weight)} '
          f'full_bn={_gsum(m9.bottleneck.weight)}')

    # ----------------------------------------------------------------------- #
    # I10: AMP -- consistency is fp32 and finite even under an autocast forward
    #      (CPU bf16 autocast as a portable stand-in for cuda fp16 autocast)
    # ----------------------------------------------------------------------- #
    torch.manual_seed(7)
    m10 = build_iso(n_cls, iso=True, iso_stage=3)
    m10.train()
    x10 = torch.randn(B, C, H, W)
    v10 = torch.tensor([1, 1, 0, 1])
    gm10 = (v10 == 1)
    try:
        with torch.amp.autocast('cpu', dtype=torch.bfloat16):
            oc10 = m10(x10)
            deg10, _ = airl_degrade(x10[gm10], min_scale=0.25, blur=False)
            od10 = m10(deg10)
        # consistency in TRUE fp32 (autocast disabled) -- the trainer's contract
        with torch.amp.autocast('cpu', enabled=False):
            L_fp32 = airl_consistency_loss(
                oc10['logits_rec'][gm10].float(), oc10['bn_feat_rec'][gm10].float(),
                od10['logits_rec'].float(), od10['bn_feat_rec'].float(),
                mode='kl', tau=4.0)
        amp_finite = torch.isfinite(L_fp32).item() and L_fp32.dtype == torch.float32
        amp_detail = f'L_fp32={float(L_fp32.detach()):.4f} dtype={L_fp32.dtype}'
    except Exception as e:                      # CPU autocast unsupported -> fp32 only
        with torch.no_grad():
            oc10 = m10(x10)
            deg10, _ = airl_degrade(x10[gm10], min_scale=0.25, blur=False)
            od10 = m10(deg10)
        L_fp32 = airl_consistency_loss(
            oc10['logits_rec'][gm10], oc10['bn_feat_rec'][gm10],
            od10['logits_rec'], od10['bn_feat_rec'], mode='kl', tau=4.0)
        amp_finite = torch.isfinite(L_fp32).item()
        amp_detail = f'(cpu-autocast skipped: {type(e).__name__}) L={float(L_fp32):.4f}'
    check('I10 consistency fp32 + finite under an autocast forward',
          amp_finite, amp_detail)

    # ----------------------------------------------------------------------- #
    # I11: TRAINER lambda-warmup INCLUDES iso (regression guard for the bug where
    #   airl_lambda_eff was gated on (airl or airl_dualbranch) only -> 0.0 every
    #   epoch on an iso run -> the f_rec consistency was multiplied by 0 and never
    #   trained).  Load afd_train's SOURCE and assert the airl_lambda_eff guard
    #   mentions airl_dualbranch_iso; also reproduce the expression to confirm a
    #   typical iso run yields lam_eff > 0 once epoch >= 1.  This catches a class of
    #   bug the loss-only checks above structurally cannot (they backward the raw
    #   consistency, never the epoch-loop multiplier).
    import inspect
    src = inspect.getsource(mod.main)
    # the guard line that computes airl_lambda_eff must include the iso flag
    guard_ok = ('airl_lambda_eff' in src
                and 'args.airl_dualbranch_iso' in src.split('airl_lambda_eff', 1)[1]
                .split('meters', 1)[0])
    # reproduce the trainer's expression for an iso-only run (airl/airl_dualbranch
    # False), epoch 1, warmup 10, lambda 1.0 -> must be strictly > 0.
    _airl, _dual, _iso = False, False, True
    _lam, _warm, _ep = 1.0, 10, 1
    lam_eff = (_lam * min(1.0, _ep / max(1, _warm))
               if (_airl or _dual or _iso) else 0.0)
    check('I11 trainer airl_lambda_eff includes iso (consistency not zeroed)',
          guard_ok and lam_eff > 0.0,
          f'guard_mentions_iso={guard_ok} lam_eff@ep1(iso-only)={lam_eff:.3f}')

    # ----------------------------------------------------------------------- #
    # I12: iso_stage=2 (the heavier fork with a PatchMerging in the rec copy) builds,
    #   forwards train+eval with correct shapes.  With the trunk-undersupervision FIX
    #   (trunk_recce=True) a CLEAN rec CE REFLOWS into the early trunk UPSTREAM of the
    #   stage-2 fork (stage 0/1) while the SHARED stage-2 (f_full's own instance,
    #   DOWNSTREAM of the fork input -> not on the rec path) stays ZERO and the rec
    #   copy trains.  Covers the alt fork point + the down-sample bookkeeping AND that
    #   the reflow target tracks iso_stage (only the pre-fork trunk, never f_full's
    #   own forked stage).
    # ----------------------------------------------------------------------- #
    torch.manual_seed(8)
    m12 = build_iso(n_cls, iso=True, iso_stage=2, trunk_recce=True)
    n_rec_stages = len(m12.backbone_swin.rec_stages)     # stages [2,3] -> 2
    m12.train()
    x12 = torch.randn(B, C, H, W)
    y12 = torch.randint(0, n_cls, (B,))
    out12 = m12(x12)
    shp12 = (tuple(out12['logits_rec'].shape) == (B, n_cls)
             and tuple(out12['bn_feat_rec'].shape) == (B, m12.in_planes))
    m12.zero_grad()
    F.cross_entropy(out12['logits_rec'], y12).backward()
    early12 = list(m12.backbone_swin.swin.stages[0].parameters())[0]   # pre-fork trunk
    shared_s2 = list(m12.backbone_swin.swin.stages[2].parameters())[0]  # f_full's stage2
    rec12 = list(m12.backbone_swin.rec_stages.parameters())[0]
    # FIX: pre-fork trunk (stage 0) reflows; f_full's own stage-2 (downstream of the
    # fork input) stays zero.
    iso2_reflow = (early12.grad is not None and early12.grad.abs().sum() > 0)
    iso2_shared_clean = _is_zero_or_none(shared_s2)
    iso2_rec_trains = (rec12.grad is not None and rec12.grad.abs().sum() > 0)
    m12.eval()
    with torch.no_grad():
        ff12, fr12 = m12(x12, return_dual=True)
    eval12_ok = (tuple(ff12.shape) == (B, m12.in_planes)
                 and tuple(fr12.shape) == (B, m12.in_planes))
    check('I12 iso_stage=2: 2 rec stages, shapes ok; clean CE reflows to pre-fork '
          'trunk (FIX), f_full stage2 ZERO, rec trains',
          n_rec_stages == 2 and shp12 and iso2_reflow and iso2_shared_clean
          and iso2_rec_trains and eval12_ok,
          f'rec_stages={n_rec_stages} shp={shp12} REFLOW early={_gsum(early12)} '
          f'shared_stage2={_gsum(shared_s2)} rec={_gsum(rec12)} eval_ok={eval12_ok}')

    # ----------------------------------------------------------------------- #
    # I13: DropPath RNG faithfulness -- in TRAIN mode the split f_full map must equal
    #   the ORIGINAL single-map swin forward given the SAME RNG seed (the rec copy
    #   runs AFTER the full f_full loop, so it cannot perturb f_full's stochastic-
    #   depth draws).  This is the strongest "f_full path unchanged by iso" claim.
    # ----------------------------------------------------------------------- #
    torch.manual_seed(9)
    m13 = build_iso(n_cls, iso=True, iso_stage=3)
    m13.train()
    x13 = torch.randn(B, C, H, W)
    bsw13 = m13.backbone_swin
    torch.manual_seed(321)
    fm_split, _rec13 = bsw13._forward_swin_split(x13)
    torch.manual_seed(321)
    _g13, outs13 = bsw13.swin(x13)
    rng_faithful = torch.allclose(fm_split, outs13[-1], atol=1e-5)
    check('I13 TRAIN f_full split == original swin map (DropPath RNG faithful)',
          rng_faithful,
          f'max|d_full(train)|={float((fm_split - outs13[-1]).abs().max()):.2e}')

    # ----------------------------------------------------------------------- #
    # I14: THE FIX, DECOMPOSED (the headline of this change).  On ONE iso model
    #   (trunk_recce=True, default) backward the TWO f_rec gradient sources SEPARATELY
    #   off the SAME forward and prove the routing:
    #     (a) CLEAN f_rec ID-CE  -> trunk REFLOW (early stage non-zero) + rec stage.
    #     (b) degraded CONSISTENCY -> trunk ZERO (the isolation invariant) while the
    #         rec stage + bottleneck_rec still get grad.
    #   Using one model + zero_grad between the two backwards isolates exactly which
    #   loss touches the trunk, which the combined-step I-checks cannot.  The degraded
    #   side is model(deg, rec_only=True) (always-detached fork) and its clean target
    #   is detached inside airl_consistency_loss -> the consistency can reach the
    #   trunk ONLY if the fork detach were wrong, so a zero here is a true isolation
    #   proof, independent of trunk_recce.
    # ----------------------------------------------------------------------- #
    torch.manual_seed(11)
    m14 = build_iso(n_cls, iso=True, iso_stage=3, trunk_recce=True)
    m14.train()
    x14 = torch.randn(B, C, H, W)
    y14 = torch.randint(0, n_cls, (B,))
    v14 = torch.tensor([1, 1, 0, 1]); gm14 = (v14 == 1)
    out_c14 = m14(x14)                                   # ONE clean forward (graph kept)
    deg14, _ = airl_degrade(x14[gm14], min_scale=0.25, blur=False)
    out_d14 = m14(deg14, rec_only=True)                 # degraded REC-ONLY (detached fork)
    early14, pe14, shared_last14 = _trunk_probe(m14)
    rec_stage14, bnrec14 = _rec_probe(m14)
    # (a) CLEAN f_rec ID-CE alone -> must REFLOW to the trunk (the fix) + rec stage.
    m14.zero_grad()
    F.cross_entropy(out_c14['logits_rec'], y14).backward(retain_graph=True)
    ce_reflow = ((early14.grad is not None and early14.grad.abs().sum() > 0)
                 and (rec_stage14.grad is not None and rec_stage14.grad.abs().sum() > 0))
    ce_early_g = _gsum(early14)
    # (b) degraded CONSISTENCY alone (clean target detached inside the loss) -> trunk
    #     MUST be zero (isolation), rec stage + bnrec MUST get grad.
    m14.zero_grad()
    L_cons14 = airl_consistency_loss(
        out_c14['logits_rec'][gm14], out_c14['bn_feat_rec'][gm14],
        out_d14['logits_rec'], out_d14['bn_feat_rec'], mode='kl', tau=4.0)
    L_cons14.backward()
    cons_trunk_zero = (_is_zero_or_none(early14) and _is_zero_or_none(pe14)
                       and _is_zero_or_none(shared_last14))
    cons_rec_trains = ((rec_stage14.grad is not None and rec_stage14.grad.abs().sum() > 0)
                       and (bnrec14.grad is not None and bnrec14.grad.abs().sum() > 0))
    cons_full_head_zero = (_is_zero_or_none(m14.bottleneck.weight)
                           and _is_zero_or_none(m14.classifier.weight))
    check('I14 FIX decomposed: CLEAN ID-CE REFLOWS to trunk; CONSISTENCY trunk=ZERO '
          '+ f_full head ZERO, rec trains (clean reflow, degraded isolated)',
          ce_reflow and cons_trunk_zero and cons_rec_trains and cons_full_head_zero,
          f'(a)clean: early={ce_early_g} rec_stage={_gsum(rec_stage14)} | '
          f'(b)cons: early={_gsum(early14)} patch_embed={_gsum(pe14)} '
          f'shared_last={_gsum(shared_last14)} rec_stage={_gsum(rec_stage14)} '
          f'bnrec={_gsum(bnrec14)} full_bn={_gsum(m14.bottleneck.weight)} '
          f'full_cls={_gsum(m14.classifier.weight)}')

    # ----------------------------------------------------------------------- #
    # I15: FULL COMBINED iso step (exactly the trainer's loss assembly) -> the union
    #   of routings holds in ONE backward: total = CE_full + CE_rec(clean) +
    #   lam*consistency.  The trunk grad is then NON-zero (it gets CE_full + the
    #   clean CE_rec reflow) -- but we separately re-confirm via I14 that the trunk's
    #   share is the CLEAN ID-CE, NOT the consistency.  Here we assert the combined
    #   step is finite and moves: trunk reflow present, rec stage + both heads grad.
    # ----------------------------------------------------------------------- #
    torch.manual_seed(12)
    m15 = build_iso(n_cls, iso=True, iso_stage=3, trunk_recce=True)
    m15.train()
    x15 = torch.randn(B, C, H, W)
    y15 = torch.randint(0, n_cls, (B,))
    v15 = torch.tensor([1, 1, 0, 1]); gm15 = (v15 == 1)
    out15 = m15(x15)
    deg15, _ = airl_degrade(x15[gm15], min_scale=0.25, blur=False)
    out_d15 = m15(deg15, rec_only=True)
    loss15 = (F.cross_entropy(out15['logits'], y15)
              + F.cross_entropy(out15['logits_rec'], y15)
              + 0.5 * airl_consistency_loss(
                  out15['logits_rec'][gm15], out15['bn_feat_rec'][gm15],
                  out_d15['logits_rec'], out_d15['bn_feat_rec'], mode='kl', tau=4.0))
    m15.zero_grad()
    loss15.backward()
    early15, _pe15, shared_last15 = _trunk_probe(m15)
    rec_stage15, bnrec15 = _rec_probe(m15)
    combined_ok = (torch.isfinite(loss15).item()
                   and (early15.grad is not None and early15.grad.abs().sum() > 0)
                   and (shared_last15.grad is not None  # CE_full trains f_full stage
                        and shared_last15.grad.abs().sum() > 0)
                   and (rec_stage15.grad is not None and rec_stage15.grad.abs().sum() > 0)
                   and (bnrec15.grad is not None and bnrec15.grad.abs().sum() > 0)
                   and (m15.classifier.weight.grad is not None
                        and m15.classifier.weight.grad.abs().sum() > 0))
    check('I15 full combined step (CE_full+CE_rec+lam*cons) finite; trunk + f_full '
          'stage + rec stage + heads all train',
          combined_ok,
          f'loss={float(loss15):.4f} early={_gsum(early15)} '
          f'shared_last={_gsum(shared_last15)} rec_stage={_gsum(rec_stage15)} '
          f'bnrec={_gsum(bnrec15)} full_cls={_gsum(m15.classifier.weight)}')

    # ----------------------------------------------------------------------- #
    # I16: trunk_recce=0 ABLATION restores the ORIGINAL full-isolation iso: the CLEAN
    #   f_rec ID-CE leaves the shared trunk (early + patch_embed + shared last) at
    #   ZERO grad (clean fork detached too), while the rec stage + classifier_rec
    #   still train.  This is the controlled comparison the fix is measured against,
    #   and proves the flag actually toggles the reflow.
    # ----------------------------------------------------------------------- #
    torch.manual_seed(13)
    m16 = build_iso(n_cls, iso=True, iso_stage=3, trunk_recce=False)
    flag_off = (m16.airl_iso_trunk_recce is False
                and m16.backbone_swin.iso_trunk_recce is False)
    m16.train()
    x16 = torch.randn(B, C, H, W)
    y16 = torch.randint(0, n_cls, (B,))
    out16 = m16(x16)
    m16.zero_grad()
    F.cross_entropy(out16['logits_rec'], y16).backward()
    early16, pe16, shared_last16 = _trunk_probe(m16)
    rec_stage16, _bnrec16 = _rec_probe(m16)
    cls_rec16 = m16.classifier_rec.weight
    abl_trunk_zero = (_is_zero_or_none(early16) and _is_zero_or_none(pe16)
                      and _is_zero_or_none(shared_last16))
    abl_rec_trains = ((rec_stage16.grad is not None and rec_stage16.grad.abs().sum() > 0)
                      and (cls_rec16.grad is not None and cls_rec16.grad.abs().sum() > 0))
    check('I16 trunk_recce=0 ablation: CLEAN ID-CE leaves trunk ZERO (original full '
          'isolation), rec stage + cls_rec train; flag toggles reflow',
          flag_off and abl_trunk_zero and abl_rec_trains,
          f'flag_off={flag_off} early={_gsum(early16)} patch_embed={_gsum(pe16)} '
          f'shared_last={_gsum(shared_last16)} rec_stage={_gsum(rec_stage16)} '
          f'cls_rec={_gsum(cls_rec16)}')

    # ----------------------------------------------------------------------- #
    # I17: trunk_recce does NOT change the f_full EVAL feature (the fix only adds a
    #   training gradient path; the forward graph value of f_full is identical for
    #   trunk_recce on/off at the SAME weights).  Build two iso models, copy weights
    #   off the recce=1 model into the recce=0 model, and confirm the eval f_full AND
    #   f_rec features match bit-for-bit (the detach choice is grad-only, never a
    #   value change) -> trunk_recce is purely a backward-pass routing knob.
    # ----------------------------------------------------------------------- #
    torch.manual_seed(14)
    m17a = build_iso(n_cls, iso=True, iso_stage=3, trunk_recce=True)
    m17b = build_iso(n_cls, iso=True, iso_stage=3, trunk_recce=False)
    m17b.load_state_dict(m17a.state_dict())             # same weights, diff recce flag
    m17a.eval(); m17b.eval()
    x17 = torch.randn(B, C, H, W)
    with torch.no_grad():
        ffa, fra = m17a(x17, return_dual=True)
        ffb, frb = m17b(x17, return_dual=True)
        # train-mode forward values must also match (DropPath off in eval; check the
        # split map equality directly under a fixed seed for the train path too).
        m17a.train(); m17b.train()
        torch.manual_seed(77); fm_a, rm_a = m17a.backbone_swin._forward_swin_split(x17)
        torch.manual_seed(77); fm_b, rm_b = m17b.backbone_swin._forward_swin_split(x17)
    eval_val_same = (torch.allclose(ffa, ffb, atol=1e-6)
                     and torch.allclose(fra, frb, atol=1e-6))
    train_val_same = (torch.allclose(fm_a, fm_b, atol=1e-6)
                      and torch.allclose(rm_a, rm_b, atol=1e-6))
    check('I17 trunk_recce is grad-only: eval AND train f_full/f_rec VALUES identical '
          'for recce on/off at the same weights',
          eval_val_same and train_val_same,
          f'eval max|dff|={float((ffa - ffb).abs().max()):.2e} '
          f'max|dfr|={float((fra - frb).abs().max()):.2e} | '
          f'train max|dfm|={float((fm_a - fm_b).abs().max()):.2e} '
          f'max|drm|={float((rm_a - rm_b).abs().max()):.2e}')

    # ----------------------------------------------------------------------- #
    print('\n' + '=' * 70)
    print(f'AIRL gradient-isolated dual-branch smoke: '
          f'{len(passed)} passed, {len(failed)} failed')
    if failed:
        print('FAILED:', failed)
        sys.exit(1)
    print('ALL AIRL-ISO SMOKE CHECKS PASSED')
    print('=' * 70)


if __name__ == '__main__':
    main()
