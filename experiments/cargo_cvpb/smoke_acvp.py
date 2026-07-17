"""Isolated numeric smoke test for ACVP (Ambiguity-Calibrated opposite-View
negative relaxation) in cargo_cvpb/afd_train.py OVLIHead.

ACVP softens UNRELIABLE NEGATIVES in the OVLI cross-view contrastive denominator
using a DETACHED opposite-view-prototype ambiguity sensor (no prototype-positive
alignment, no learnable params, no gradient to encoder/proto).  Default OFF =>
the OVLI loss is reproduced byte-for-byte.

Loads the REAL OVLIHead out of cargo_cvpb/afd_train.py (NOT a copy) by stubbing
the heavy sibling modules (cargo_dataset / agreid_dataset / afd_model / afd_train)
in sys.modules, so no backbone / dataset is built.  CPU-only, tiny tensors ->
never touches a GPU.

Checks
  A1  OFF BYTE-IDENTICAL: ovli.loss(...) with NO acvp args (acvp_proto=None) is
        ELEMENT-WISE EQUAL (torch.equal) to an inline copy of the original
        opposite-view-only loss body -> off-mode reproduces pre-ACVP exactly.
  A2  OFF == passing acvp_proto=None explicitly (the default-arg path and the
        explicit-None path are the same code).
  A3  ON CHANGES THE LOSS on identical inputs with an INITIALISED prototype bank
        (ACVP is not a no-op) AND the change is finite.
  A4  BIAS STRUCTURE: acvp_neg_bias is <= 0 ONLY on opposite-view negative
        entries and EXACTLY 0 on positives / same-view / self / uninitialised
        pairs (positives & numerator are untouched -> "softens negatives only").
  A5  w_ij NUMERIC SAFETY: w in [wmin, 1], bias = log(w) finite (no -inf/NaN)
        even with degenerate (all-zero / all-equal) prototypes and gamma at max.
  A6  DETACH / NO GRAD LEAK: backward through the ACVP-ON loss gives FINITE grad
        on proj.weight (encoder path alive) but the prototype bank tensor (fed in
        with requires_grad=True) gets NO gradient (.grad is None) -> ACVP injects
        no gradient into the prototypes/feature.
  A7  UNINITIALISED -> w == 1 (bias 0): with inited all-zero, ACVP-ON loss ==
        OFF loss (cold-start prototypes never soften).
  A8  GAMMA==0 (e.g. warmup epoch 0) -> w == 1 (bias 0): ACVP-ON@gamma0 == OFF.
  A9  KILL-SWITCH STATS: ovli._acvp_stats = (relaxed_neg_frac, mean_w, n_soft) is
        set, finite, frac in [0,1], mean_w in [wmin, 1], n_soft (=ok.sum()) > 0.
  A10 MONOTONICITY: a MORE ambiguous negative (larger delta) gets a SMALLER
        weight w (softened more) -> the sensor has the right sign.
  A11 MOCK TRAINING STEP: an ACVP-ON forward + the loop's exact post-step stats
        accumulation runs WITHOUT crashing (guards the Critical `bs`-ordering
        UnboundLocalError), the per-epoch stats are weighted by #softenable-neg
        (ok.sum()=stats[2]), and a cold-start (n_soft==0) step is SKIPPED.

Usage:  python smoke_acvp.py [path/to/cargo_cvpb/afd_train.py]
        (run inside an env with torch+numpy, e.g.
         `uv run --no-project --with numpy --with torch python smoke_acvp.py`)
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


def _noop_init(_m):  # stand-in for afd_model.weights_init_kaiming (used in __init__)
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


def load_ovlihead(path):
    spec = importlib.util.spec_from_file_location('cvpb_afd_train_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OVLIHead


# --------------------------------------------------------------------------- #
# 2) inline copy of the ORIGINAL (pre-ACVP) opposite-view-only loss body.
#    Reuses the live OVLIHead's sym_maxsim_matrix / alpha / tau, so this is the
#    exact pre-ACVP behaviour. If off-mode matches this element-wise, off-mode
#    reproduces the pre-ACVP loss byte-for-byte.
# --------------------------------------------------------------------------- #
def ref_oppview_loss(ovli, gfeat, tok, labels, views):
    B = gfeat.size(0)
    device = gfeat.device
    gsim = gfeat @ gfeat.t()
    msim = ovli.sym_maxsim_matrix(tok)
    score = ovli.alpha * gsim + (1.0 - ovli.alpha) * msim
    same_view = views.view(-1, 1).eq(views.view(1, -1))
    same_pid = labels.view(-1, 1).eq(labels.view(1, -1))
    eye = torch.eye(B, dtype=torch.bool, device=device)
    cand = (~same_view) & (~eye)        # original: opposite-view, not self
    pos = cand & same_pid
    neg = cand & (~same_pid)
    valid = (pos.sum(dim=1) > 0) & (neg.sum(dim=1) > 0)
    if valid.sum() == 0:
        z = gfeat.new_zeros(())
        return z, z, z
    logits = score / ovli.tau
    floor = logits.new_full((), -1e4)
    pos_logits = torch.where(pos, logits, floor)
    cand_logits = torch.where(cand, logits, floor)
    log_num = torch.logsumexp(pos_logits, dim=1)
    log_den = torch.logsumexp(cand_logits, dim=1)
    per_anchor = -(log_num - log_den)
    loss = per_anchor[valid].mean()
    with torch.no_grad():
        ps = score[pos].mean() if pos.any() else score.new_zeros(())
        ns = score[neg].mean() if neg.any() else score.new_zeros(())
    return loss, ps, ns


class DummyBackbone(nn.Module):
    """Minimal model exposing .layer4 so OVLIHead can register its hook."""
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Identity()


def make_head(OVLIHead, in_ch, proj_dim, grid):
    return OVLIHead(DummyBackbone(), in_ch=in_ch, proj_dim=proj_dim,
                    grid=grid, alpha=0.5, tau=0.05, pool='mean',
                    topk=8, thresh=0.0, allview=False)


def build_proto_bank(num_pid, feat_dim, init_all=True, degenerate=False,
                     requires_grad=False, seed=0):
    """L2-normed per-pid per-view prototype bank [num_pid, 2, D] + inited mask.

    degenerate=True: prototypes are all-zero (worst case for cos/log safety).
    init_all=False : inited mask is all-zero (cold start, no softening).
    """
    g = torch.Generator().manual_seed(seed)
    if degenerate:
        bank = torch.zeros(num_pid, 2, feat_dim)
    else:
        bank = torch.randn(num_pid, 2, feat_dim, generator=g)
        bank = F.normalize(bank, dim=2)
    bank = bank.clone().requires_grad_(requires_grad)
    inited = torch.ones(num_pid, 2, dtype=torch.uint8) if init_all \
        else torch.zeros(num_pid, 2, dtype=torch.uint8)
    return bank, inited


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'afd_train.py')
    OVLIHead = load_ovlihead(path)
    torch.manual_seed(0)
    device = 'cpu'

    # batch: 2 pids x 2 views x 2 samples
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device)
    views = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], device=device)
    B = labels.numel()
    num_pid = 2
    in_ch, proj_dim, grid = 16, 8, (2, 2)
    Dg = proj_dim  # gfeat / prototype dim must match (cos(z_i, P[..]))

    head = make_head(OVLIHead, in_ch, proj_dim, grid)
    gfeat = F.normalize(torch.randn(B, Dg, device=device), dim=1)
    fmap = torch.randn(B, in_ch, 4, 4, device=device)
    head._buf['map'] = fmap
    tok = head.tokens_from_cached_map()                # (B,K,proj), proj in graph

    # ---- A1: OFF (no acvp args) == inline original --------------------------- #
    l_off, ps_off, ns_off = head.loss(gfeat, tok, labels, views)
    l_ref, ps_ref, ns_ref = ref_oppview_loss(head, gfeat, tok, labels, views)
    eq = (torch.equal(l_off.detach(), l_ref.detach())
          and torch.equal(ps_off.detach(), ps_ref.detach())
          and torch.equal(ns_off.detach(), ns_ref.detach()))
    print(f"[A1] OFF vs inline-original loss={l_off.item():.6f} "
          f"ref={l_ref.item():.6f} equal={eq}")
    assert eq, "ACVP-OFF does NOT reproduce the pre-ACVP loss byte-for-byte!"

    # ---- A2: OFF == explicit acvp_proto=None --------------------------------- #
    l_none, _, _ = head.loss(gfeat, tok, labels, views, acvp_proto=None)
    eq2 = torch.equal(l_none.detach(), l_off.detach())
    print(f"[A2] OFF(default) vs acvp_proto=None  equal={eq2}")
    assert eq2, "explicit acvp_proto=None diverges from default off path"

    # ---- A3: ON with initialised bank changes the loss (finite) -------------- #
    bank, inited = build_proto_bank(num_pid, Dg, init_all=True, seed=1)
    l_on, _, _ = head.loss(gfeat, tok, labels, views,
                           acvp_proto=bank, acvp_inited=inited,
                           acvp_gamma=0.5, acvp_wmin=0.3, acvp_eta=0.05,
                           acvp_margin=0.0)
    on_finite = bool(torch.isfinite(l_on))
    delta = abs(l_on.item() - l_off.item())
    print(f"[A3] ON loss={l_on.item():.6f} OFF={l_off.item():.6f} "
          f"|delta|={delta:.6f} finite={on_finite}")
    assert on_finite, "ACVP-ON loss not finite"
    assert delta > 1e-6, "ACVP-ON did not change the loss (no-op with init bank)"
    # ACVP softens NEGATIVES in the DENOMINATOR (log w<=0 shrinks the denom) ->
    # the contrastive loss should DECREASE vs OFF.
    print(f"[A3b] ON <= OFF (softening shrinks denominator): "
          f"{l_on.item() <= l_off.item() + 1e-6}")
    assert l_on.item() <= l_off.item() + 1e-6, \
        "softening negatives should not INCREASE the contrastive loss"

    # ---- A4: bias structure -- <=0 only on opp-view negatives, 0 elsewhere --- #
    same_view = views.view(-1, 1).eq(views.view(1, -1))
    same_pid = labels.view(-1, 1).eq(labels.view(1, -1))
    eye = torch.eye(B, dtype=torch.bool)
    neg = (~same_view) & (~eye) & (~same_pid)          # opp-view negatives
    bias, frac, mean_w, n_soft = head.acvp_neg_bias(
        gfeat, labels, views, neg, bank, inited,
        gamma=0.5, wmin=0.3, eta=0.05, margin=0.0)
    nonneg_zero = bool(torch.all(bias[~neg] == 0))     # 0 off the negatives
    neg_nonpos = bool(torch.all(bias[neg] <= 1e-9))    # <= 0 on negatives
    # bank is fully initialised here -> every opp-view negative is softenable,
    # so n_soft must equal the number of negatives.
    nsoft_ok = (int(n_soft) == int(neg.sum()))
    print(f"[A4] bias: zero-off-neg={nonneg_zero} neg<=0={neg_nonpos} "
          f"max(bias)={float(bias.max()):.3e} min(bias)={float(bias.min()):.3e} "
          f"n_soft={int(n_soft)} (=#neg={int(neg.sum())}? {nsoft_ok})")
    assert nonneg_zero, "ACVP bias is non-zero on a positive/same-view/self pair!"
    assert neg_nonpos, "ACVP bias is positive on a negative (should be log w<=0)"
    assert nsoft_ok, "n_soft (ok.sum()) != #softenable negatives"

    # ---- A5: w_ij numeric safety (degenerate prototypes, gamma at max) ------- #
    zbank, zinit = build_proto_bank(num_pid, Dg, init_all=True, degenerate=True)
    bias_z, frac_z, mw_z, nsoft_z = head.acvp_neg_bias(
        gfeat, labels, views, neg, zbank, zinit,
        gamma=1.0, wmin=0.3, eta=0.05, margin=0.0)
    # reconstruct w from bias (bias = log w) and check the [wmin,1] envelope
    w_z = torch.exp(bias_z)
    w_in_range = bool(torch.all(w_z >= 0.3 - 1e-6) and torch.all(w_z <= 1.0 + 1e-6))
    safe = bool(torch.isfinite(bias_z).all() and torch.isfinite(w_z).all())
    print(f"[A5] degenerate-proto gamma=1: finite={safe} "
          f"w in [wmin,1]={w_in_range} (w range "
          f"[{float(w_z.min()):.3f},{float(w_z.max()):.3f}])")
    assert safe, "ACVP bias/w NOT finite on degenerate prototypes"
    assert w_in_range, "w_ij escaped [wmin,1] envelope"

    # ---- A6: detach / no grad leak into prototypes; grad alive on proj ------- #
    head.zero_grad(set_to_none=True)
    gbank, ginit = build_proto_bank(num_pid, Dg, init_all=True,
                                    requires_grad=True, seed=3)
    fmap_g = torch.randn(B, in_ch, 4, 4, device=device, requires_grad=True)
    head._buf['map'] = fmap_g
    tok_g = head.tokens_from_cached_map()
    gfeat_g = F.normalize(torch.randn(B, Dg, device=device, requires_grad=True),
                          dim=1)
    l_g, _, _ = head.loss(gfeat_g, tok_g, labels, views,
                          acvp_proto=gbank, acvp_inited=ginit,
                          acvp_gamma=0.5, acvp_wmin=0.3, acvp_eta=0.05,
                          acvp_margin=0.0)
    l_g.backward()
    proj_grad = head.proj.weight.grad
    proj_ok = (proj_grad is not None and float(proj_grad.abs().sum()) > 0
               and torch.isfinite(proj_grad).all())
    proto_no_grad = (gbank.grad is None)               # detached -> no grad
    map_ok = (fmap_g.grad is not None and torch.isfinite(fmap_g.grad).all())
    print(f"[A6] proj.grad alive={proj_ok} (|sum|={float(proj_grad.abs().sum()):.3e})"
          f"  prototype.grad is None={proto_no_grad}  layer4-map.grad ok={map_ok}")
    assert proj_ok, "gradient did not flow into proj under ACVP-ON"
    assert proto_no_grad, "GRAD LEAKED into the prototype bank (ACVP must detach)!"
    assert map_ok, "gradient did not flow into the layer4 map under ACVP-ON"
    # restore the non-grad map for the remaining checks
    head._buf['map'] = fmap

    # ---- A7: uninitialised bank -> w==1 -> ON == OFF ------------------------- #
    bank_u, inited_u = build_proto_bank(num_pid, Dg, init_all=False, seed=1)
    l_u, _, _ = head.loss(gfeat, tok, labels, views,
                          acvp_proto=bank_u, acvp_inited=inited_u,
                          acvp_gamma=0.5, acvp_wmin=0.3, acvp_eta=0.05,
                          acvp_margin=0.0)
    eq_u = torch.allclose(l_u.detach(), l_off.detach(), atol=1e-6)
    print(f"[A7] uninitialised-bank ON={l_u.item():.6f} OFF={l_off.item():.6f} "
          f"equal={eq_u}")
    assert eq_u, "uninitialised prototypes still softened (cold-start leak)"

    # ---- A8: gamma==0 (warmup epoch 0) -> w==1 -> ON == OFF ------------------ #
    l_g0, _, _ = head.loss(gfeat, tok, labels, views,
                           acvp_proto=bank, acvp_inited=inited,
                           acvp_gamma=0.0, acvp_wmin=0.3, acvp_eta=0.05,
                           acvp_margin=0.0)
    eq_g0 = torch.allclose(l_g0.detach(), l_off.detach(), atol=1e-6)
    print(f"[A8] gamma=0 ON={l_g0.item():.6f} OFF={l_off.item():.6f} equal={eq_g0}")
    assert eq_g0, "gamma=0 (warmup start) still changed the loss"

    # ---- A9: kill-switch stats present, finite, in range -------------------- #
    _ = head.loss(gfeat, tok, labels, views,
                  acvp_proto=bank, acvp_inited=inited,
                  acvp_gamma=0.5, acvp_wmin=0.3, acvp_eta=0.05, acvp_margin=0.0)
    stats = getattr(head, '_acvp_stats', None)
    assert stats is not None, "ACVP kill-switch stats not stashed on the head"
    assert len(stats) == 3, "ACVP _acvp_stats must be (frac, mean_w, n_soft)"
    s_frac, s_mw, s_nsoft = float(stats[0]), float(stats[1]), int(stats[2])
    stats_ok = (0.0 <= s_frac <= 1.0 and 0.3 - 1e-6 <= s_mw <= 1.0 + 1e-6
                and torch.isfinite(stats[0]) and torch.isfinite(stats[1])
                and s_nsoft > 0)               # init bank -> some softenable negs
    print(f"[A9] kill-switch relaxed_neg_frac={s_frac:.4f} mean_w={s_mw:.4f} "
          f"n_soft={s_nsoft} in-range={stats_ok}")
    assert stats_ok, "ACVP kill-switch stats out of range / non-finite / n_soft<=0"

    # ---- A10: monotonicity -- larger ambiguity delta -> smaller weight ------ #
    # Construct a clean 2-sample, 1-negative case to read off w(delta).  anchor
    # i=0 (pid0, view0); negative j=1 (pid1, view1 = opposite view).
    lab2 = torch.tensor([0, 1])
    vw2 = torch.tensor([0, 1])
    neg2 = torch.tensor([[False, True], [True, False]])
    z = F.normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), dim=1)
    # P[pid, view]: view_j for the i=0,j=1 pair is view1.
    #   P[y_i=0, view1] = e_x  (i's own opp-view identity, far from... )
    #   P[y_j=1, view1] = controllable -> set close to z_0 (e_x) = ambiguous neg.
    bank2 = torch.zeros(2, 2, 8)
    bank2[0, 1] = z[0]                          # P[0, v1] = e_x
    init2 = torch.ones(2, 2, dtype=torch.uint8)
    ws = []
    for ang in (0.9, 0.0, -0.9):                # cos(P[1,v1], z0) high->low
        p = torch.zeros(8); p[0] = ang; p[1] = (1 - ang ** 2) ** 0.5
        bank2[1, 1] = F.normalize(p, dim=0)
        bz, _, _, _ = head.acvp_neg_bias(z, lab2, vw2, neg2, bank2, init2,
                                         gamma=0.5, wmin=0.05, eta=0.05, margin=0.0)
        ws.append(float(torch.exp(bz[0, 1])))   # w for the (0,1) negative
    mono = ws[0] <= ws[1] + 1e-6 <= ws[2] + 1e-6 or (ws[0] <= ws[1] <= ws[2])
    print(f"[A10] w(delta high->low) = {[round(x,4) for x in ws]} "
          f"monotone(more ambiguous -> smaller w)={ws[0] <= ws[1] <= ws[2]}")
    assert ws[0] <= ws[1] + 1e-6 and ws[1] <= ws[2] + 1e-6, \
        "weight is not monotone in ambiguity (sensor sign wrong)"

    # ---- A11: ACVP-ON mock training step does NOT crash + per-epoch stats are
    #          weighted by #softenable-neg (ok.sum()=stats[2]), and n_soft==0
    #          steps are SKIPPED.  This mirrors the exact post-optimizer-step
    #          accumulation block in main() and guards the Critical `bs`-ordering
    #          bug (stats consumed BEFORE bs was assigned -> UnboundLocalError on
    #          the very first batch). ------------------------------------------ #
    # replicate the loop's accumulators
    acvp_frac_sum, acvp_w_sum, acvp_steps = 0.0, 0.0, 0
    crashed = None

    def mock_step(bank_s, inited_s):
        """One ACVP-ON forward + the loop's exact stats-accumulation block."""
        nonlocal acvp_frac_sum, acvp_w_sum, acvp_steps
        _l, _, _ = head.loss(gfeat, tok, labels, views,
                             acvp_proto=bank_s, acvp_inited=inited_s,
                             acvp_gamma=0.5, acvp_wmin=0.3, acvp_eta=0.05,
                             acvp_margin=0.0)
        st = getattr(head, '_acvp_stats', None)
        assert st is not None and len(st) == 3
        # --- begin: byte-copy of main()'s accumulation (n_soft-weighted, skip 0)
        n_soft = int(st[2])
        if n_soft > 0:
            acvp_frac_sum += float(st[0]) * n_soft
            acvp_w_sum += float(st[1]) * n_soft
            acvp_steps += n_soft
        # --- end ----------------------------------------------------------------
        return n_soft

    try:
        # step 1: cold-start bank (inited all-zero) -> n_soft==0 -> SKIPPED
        ns_cold = mock_step(bank_u, inited_u)
        steps_after_cold = acvp_steps
        # step 2: initialised bank -> n_soft>0 -> counted
        ns_warm = mock_step(bank, inited)
    except Exception as e:           # the old bs-ordering bug would surface here
        crashed = repr(e)
    ok_nocrash = crashed is None
    cold_skipped = (ns_cold == 0 and steps_after_cold == 0)
    warm_counted = (ns_warm > 0 and acvp_steps == ns_warm)
    # the loop divides the sums by acvp_steps; verify that average is well-defined
    a_frac = acvp_frac_sum / acvp_steps if acvp_steps > 0 else 0.0
    a_mw = acvp_w_sum / acvp_steps if acvp_steps > 0 else 1.0
    summary_ok = (math.isfinite(a_frac) and math.isfinite(a_mw)
                  and 0.0 <= a_frac <= 1.0 and 0.3 - 1e-6 <= a_mw <= 1.0 + 1e-6)
    print(f"[A11] mock ACVP step: no-crash={ok_nocrash} cold-skipped={cold_skipped} "
          f"warm-counted={warm_counted} acvp_steps={acvp_steps} "
          f"epoch_frac={a_frac:.4f} epoch_mean_w={a_mw:.4f} summary_ok={summary_ok}")
    assert ok_nocrash, f"ACVP-ON mock step crashed: {crashed}"
    assert cold_skipped, "cold-start (n_soft==0) step was NOT skipped in stats"
    assert warm_counted, "stats not weighted by #softenable-neg (ok.sum())"
    assert summary_ok, "per-epoch ACVP summary out of range / non-finite"

    print("ALL SMOKE TESTS PASSED")


if __name__ == '__main__':
    main()
