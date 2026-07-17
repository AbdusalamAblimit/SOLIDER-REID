"""Isolated numeric smoke test for the "mean + zero-init residual" set-pooling.

Covers the fix to cargo_cvpb/afd_train.py OVLISetPool:
  --ovli_setpool {netvlad, attn, gated, secondorder} --ovli_setpool_residual 1
which makes each learnable pool start LOSSLESS from the 52.37 mean-pool

    pooled = mean_k(tok) + gate_res * residual(tok)        gate_res zero-init

** What "52.37 path" means here (the codex High fix) **
The real 52.37 cross-view score is `--ovli_match avg --ovli_pool mean`, whose
sym_maxsim_matrix reduces to the UN-normalized gram `mean_k(tok) @ mean_k(tok).T`
(the per-sample mean of the K L2-normed tokens is NOT re-normalized).  So the
residual pool must, at gate_res==0, produce that SAME un-normalized gram --
NOT `<F.normalize(mean), F.normalize(mean)>` (which has diag==1 and is a
different score).  The fix makes aggregate_tokens RETURN THE RAW vector in
residual mode (no final L2-norm), so `a @ a.T` == the avg/mean-pool gram
BYTE-FOR-BYTE (torch.equal).  This is the load-bearing kill-switch fairness
guarantee: the residual variant starts EXACTLY on the 52.37 path.

This also fixes the standalone-random-init collapse (netvlad standalone ep20 mAP
14.66 << 52.37 << even pure global 45.14): with the residual start the learnable
pool only LEARNS a correction off the 52.37 mean-pool.

Loads the REAL OVLIHead + OVLISetPool out of cargo_cvpb/afd_train.py (not a copy)
by stubbing the heavy sibling modules (cargo_dataset / agreid_dataset / afd_model
/ afd_train) in sys.modules, so no backbone / dataset is built.  CPU-only, tiny
tensors -> never touches a GPU.

Checks
  R1  LOSSLESS ZERO-INIT START vs the REAL 52.37 avg path (the load-bearing
        guarantee).  For every learnable mode, with residual=True at fresh
        (zero-gate) init, build a TWIN setpool=mean head with --ovli_match avg
        --ovli_pool mean (the literal 52.37 code path) and assert
            residual.sym_maxsim_matrix(tok) == avg_head.sym_maxsim_matrix(tok)
        to torch.equal / <1e-6, AND that this equals the inline un-normalized
        `mean_tok @ mean_tok.T`.  Also: gate_res is exactly 0; forward(tok) ==
        mean_k(tok); aggregate_tokens(tok) == mean_k(tok) (RAW, not normalized).
  R2  sym_maxsim_matrix at init == the un-normalized mean gram (the cross-view
        SCORE the train loss / eval rerank see) to 1e-6, symmetric & finite ->
        the whole OVLI score starts on the 52.37 avg path.
  R3  FULL LOSS at init == the avg-path reference loss (msim = un-normalized mean
        gram) to 1e-6 -> training step 0 == the 52.37 avg/mean-pool step 0.
  R4  RESIDUAL TURNS ON: after manually setting gate_res != 0 (and perturbing the
        residual module), aggregate_tokens MOVES off the mean (not frozen at the
        mean forever); gradient reaches gate_res AND the residual params.
  R5  PERMUTATION INVARIANCE holds in residual mode (mean + residual both only
        collapse the K axis): shuffling K leaves aggregate/msim unchanged <1e-5.
  R6  GRADIENT FLOW at init: loss finite; backward gives finite grad on proj
        (encoder path) and on gate_res (so the residual can switch on); the
        layer4 map receives gradient.  Step-0 residual-module params have ZERO
        grad (g==0) -> only the gate moves first.
  R7  NaN-SAFETY in residual mode: degenerate tokens (all-zero, all-equal) keep
        aggregate/msim/loss finite; rectangular eval shapes (Nq != Ng) finite.
  R8  STANDALONE FALLBACK (residual=False): gate_res is None, forward == the
        original standalone pooling (no mean added) and aggregate_tokens DOES
        L2-normalize it (cosine gram) -> the ablation control is preserved and
        DIFFERS from the residual start.
  R9  RNG PARITY: a setpool!=mean head and a setpool=mean head built under the
        SAME seed get byte-identical proj weights (proj is now constructed BEFORE
        the set-pool, so the set-pool's RNG draws don't shift proj's init).

Usage:  python smoke_ovli_residual.py [path/to/cargo_cvpb/afd_train.py]
        (run inside an env with torch+numpy, e.g.
         `uv run --no-project --with numpy --with torch python smoke_ovli_residual.py`)
"""
import importlib.util
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


def load_mod(path):
    spec = importlib.util.spec_from_file_location('cvpb_afd_train_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 2) inline reference: loss whose msim is the cosine gram of the L2-normed mean
#    token (the 52.37 path's aggregated-cosine equivalent).  Independent of the
#    residual branch under test.
# --------------------------------------------------------------------------- #
def mean_gram(tok):
    """Cross-view SCORE the residual pool must start at: the UN-normalized gram of
    the per-sample mean of the K L2-normed tokens -- `mean_tok @ mean_tok.T`.

    This is EXACTLY what `--ovli_match avg --ovli_pool mean` (the 52.37 path)
    reduces its sym_maxsim_matrix to (verified independently in R1 against the
    real avg-mode head).  Note: the mean is NOT re-normalized (no F.normalize),
    so the diagonal is ||mean_tok||^2 (< 1), not 1 -- re-norming would give a
    DIFFERENT (cosine) score and is precisely the bug this test guards against."""
    m = tok.mean(dim=1)                            # (B,D) un-normalized mean
    return m @ m.t()                               # (B,B) un-normalized gram


def avg_path_gram(OVLIHead, in_ch, proj_dim, grid, tok):
    """The LITERAL 52.37 path: a setpool='mean', match='avg', pool='mean' head's
    sym_maxsim_matrix on the GIVEN (already-projected) tokens.  setpool=mean's
    sym_maxsim_matrix operates on `tok` directly (it does NOT re-project), so
    feeding it the residual head's own projected tokens isolates the AGGREGATION:
    any difference would be the residual vs the avg reduction, not the proj.
    This is the ground-truth score the residual start must byte-match."""
    head = OVLIHead(DummyBackbone(), in_ch=in_ch, proj_dim=proj_dim, grid=grid,
                    alpha=0.5, tau=0.05, pool='mean', topk=8, thresh=0.0,
                    allview=False, match='avg', align='free', setpool='mean',
                    setpool_residual=True)
    return head.sym_maxsim_matrix(tok)             # (B,B) un-normalized mean gram


def reference_loss(ovli, gfeat, tok, labels, views):
    """Exact copy of OVLIHead.loss body, but msim = mean_gram(tok) (the un-norm
    avg/mean-pool score), so it is independent of the setpool branch under test."""
    B = gfeat.size(0)
    gsim = gfeat @ gfeat.t()
    msim = mean_gram(tok)
    score = ovli.alpha * gsim + (1.0 - ovli.alpha) * msim
    same_view = views.view(-1, 1).eq(views.view(1, -1))
    same_pid = labels.view(-1, 1).eq(labels.view(1, -1))
    eye = torch.eye(B, dtype=torch.bool, device=gfeat.device)
    cand = ~eye if ovli.allview else (~same_view) & (~eye)
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


def make_head(OVLIHead, in_ch, proj_dim, grid, setpool='netvlad',
              setpool_residual=True, match='avg', vlad_clusters=8, attn_heads=4,
              so_rank=32):
    return OVLIHead(DummyBackbone(), in_ch=in_ch, proj_dim=proj_dim, grid=grid,
                    alpha=0.5, tau=0.05, pool='mean', topk=8, thresh=0.0,
                    allview=False, match=match, align='free', setpool=setpool,
                    vlad_clusters=vlad_clusters, attn_heads=attn_heads,
                    so_rank=so_rank, setpool_residual=setpool_residual)


def tokens(head, B, in_ch, hw=(16, 8), requires_grad=False):
    """Feed a synthetic layer4 map through the head's real proj -> L2 tokens."""
    fmap = torch.randn(B, in_ch, hw[0], hw[1], requires_grad=requires_grad)
    head._buf['map'] = fmap
    return head.tokens_from_cached_map(), fmap


LEARNABLE = ('netvlad', 'attn', 'gated', 'secondorder')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'afd_train.py')
    mod = load_mod(path)
    OVLIHead, OVLISetPool = mod.OVLIHead, mod.OVLISetPool
    torch.manual_seed(0)

    # opposite-view PK batch: 4 pids x {aerial(0), ground(1)}
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    views = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    B = labels.numel()
    in_ch, proj_dim, grid = 16, 256, (8, 4)            # K = 32 tokens; dim==out_dim
    K = grid[0] * grid[1]
    Dg = 12
    gfeat = F.normalize(torch.randn(B, Dg), dim=1)

    # =====================================================================  R1
    # LOSSLESS zero-init start vs the REAL 52.37 avg path: the residual head's
    # sym_maxsim_matrix == a setpool=mean / match=avg / pool=mean head's
    # sym_maxsim_matrix (the literal 52.37 code) -- to torch.equal / <1e-6.
    print("[R1] lossless zero-init start == REAL 52.37 avg/mean path (torch.equal):")
    for mode in LEARNABLE:
        head = make_head(OVLIHead, in_ch, proj_dim, grid, setpool=mode,
                         setpool_residual=True)
        assert head.setpool_mod is not None and head.setpool_residual
        g0 = head.setpool_mod.gate_res
        assert g0 is not None and float(g0.detach().abs().max()) == 0.0, \
            f"{mode}: gate_res must be exactly 0 at init"
        tok, _ = tokens(head, B, in_ch)
        m = tok.mean(dim=1).detach()
        # forward (raw, no final L2-norm) must equal the un-normalized mean
        fwd = head.setpool_mod(tok).detach()
        d_fwd = float((fwd - m).abs().max())
        # aggregate_tokens (residual mode -> RAW, no L2-norm) must equal mean_k(tok)
        agg = head.aggregate_tokens(tok).detach()
        d_agg = float((agg - m).abs().max())
        m_res = head.sym_maxsim_matrix(tok).detach()
        # (a) BIT-EXACT vs the canonical un-normalized mean gram `mean @ mean.T`:
        #     residual aggregate is `mean + 0*residual == mean`, so a@a.T is
        #     torch.equal to mean@mean.T (same op, same order) -> diag == ||mean||^2
        #     (NOT 1: this is the un-normalized convention the 52.37 path uses).
        m_direct = m @ m.t()
        eq_direct = torch.equal(m_res, m_direct)
        d_direct = float((m_res - m_direct).abs().max())
        # (b) *** the load-bearing check ***: residual sym_maxsim_matrix == the
        #     LITERAL 52.37 code path (a setpool=mean / match=avg / pool=mean head's
        #     sym_maxsim_matrix on the SAME projected tokens).  The two differ only
        #     in float reduction order (a@a.T vs the 4D mean-reduce), so this is
        #     torch.equal OR < 1e-6 (the task's acceptance criterion).
        m_avg = avg_path_gram(OVLIHead, in_ch, proj_dim, grid, tok).detach()
        eq_avg = torch.equal(m_res, m_avg)
        d_avg = float((m_res - m_avg).abs().max())
        print(f"     {mode:11s} gate=0 |fwd-mean|={d_fwd:.1e} |agg-mean|={d_agg:.1e} "
              f"| ==mean@mean.T: equal={eq_direct}({d_direct:.1e}) "
              f"| ==avg_path: equal={eq_avg} max|diff|={d_avg:.1e}")
        assert d_fwd <= 1e-6, f"{mode}: forward != mean_k(tok) at init"
        assert d_agg <= 1e-6, f"{mode}: aggregate_tokens != raw mean_k(tok) at init"
        assert eq_direct, \
            f"{mode}: residual sym_maxsim NOT bit-equal to un-normalized mean@mean.T"
        assert eq_avg or d_avg <= 1e-6, \
            f"{mode}: residual sym_maxsim != the 52.37 avg/mean path (>1e-6, NOT clean!)"

    # =====================================================================  R2
    # the cross-view SCORE (sym_maxsim_matrix) starts at the UN-normalized mean gram.
    print("[R2] sym_maxsim_matrix at init == un-normalized mean gram:")
    for mode in LEARNABLE:
        head = make_head(OVLIHead, in_ch, proj_dim, grid, setpool=mode)
        tok, _ = tokens(head, B, in_ch)
        m_new = head.sym_maxsim_matrix(tok).detach()
        m_ref = mean_gram(tok).detach()
        d = float((m_new - m_ref).abs().max())
        sym = torch.allclose(m_new, m_new.t(), atol=1e-6)
        fin = bool(torch.isfinite(m_new).all())
        print(f"     {mode:11s} |msim - mean_gram|={d:.2e} symmetric={sym} finite={fin}")
        assert d <= 1e-6, f"{mode}: msim != un-normalized mean gram at init"
        assert sym and fin, f"{mode}: msim not symmetric/finite"

    # =====================================================================  R3
    # FULL LOSS at init == the inline avg/mean-pool reference loss.
    print("[R3] full loss at init == avg/mean-pool reference loss:")
    for mode in LEARNABLE:
        head = make_head(OVLIHead, in_ch, proj_dim, grid, setpool=mode)
        tok, _ = tokens(head, B, in_ch)
        l_new, p_new, n_new = head.loss(gfeat, tok, labels, views)
        l_ref, p_ref, n_ref = reference_loss(head, gfeat, tok, labels, views)
        dl = float((l_new - l_ref).abs().max())
        dp = float((p_new - p_ref).abs().max())
        dn = float((n_new - n_ref).abs().max())
        print(f"     {mode:11s} loss={l_new.item():.6f} ref={l_ref.item():.6f} "
              f"|dl|={dl:.2e} |dpos|={dp:.2e} |dneg|={dn:.2e}")
        assert dl <= 1e-6 and dp <= 1e-6 and dn <= 1e-6, \
            f"{mode}: loss != mean-pool reference at init"

    # =====================================================================  R4
    # residual TURNS ON: open the gate + perturb the residual -> aggregate MOVES.
    print("[R4] residual turns on (gate_res != 0 -> aggregate moves off mean):")
    for mode in LEARNABLE:
        head = make_head(OVLIHead, in_ch, proj_dim, grid, setpool=mode)
        tok, _ = tokens(head, B, in_ch)
        agg0 = head.aggregate_tokens(tok).clone()
        with torch.no_grad():
            head.setpool_mod.gate_res.fill_(0.5)            # open the gate
            # nudge the residual's output projection so residual(tok) != 0
            for p in head.setpool_mod.out.parameters():
                p.add_(0.1 * torch.randn_like(p))
        agg1 = head.aggregate_tokens(tok)
        moved = float((agg1 - agg0).abs().max())
        fin = bool(torch.isfinite(agg1).all())
        print(f"     {mode:11s} |aggregate(after) - aggregate(init)|={moved:.3e} finite={fin}")
        assert moved > 1e-4, f"{mode}: residual never moves the output (frozen at mean)"
        assert fin, f"{mode}: aggregate non-finite after opening the gate"

    # =====================================================================  R5
    # permutation invariance in residual mode (mean + residual both K-collapsing).
    print("[R5] permutation invariance (residual mode):")
    tok_base, _ = tokens(make_head(OVLIHead, in_ch, proj_dim, grid), B, in_ch)
    perm = torch.randperm(K)
    assert not torch.equal(perm, torch.arange(K)), "need a real shuffle"
    tok_shuf = tok_base[:, perm, :].contiguous()
    for mode in LEARNABLE:
        head = make_head(OVLIHead, in_ch, proj_dim, grid, setpool=mode)
        # open the gate so we test invariance of the FULL mean+residual, not just mean
        with torch.no_grad():
            head.setpool_mod.gate_res.fill_(0.7)
            for p in head.setpool_mod.out.parameters():
                p.add_(0.1 * torch.randn_like(p))
        a1 = head.aggregate_tokens(tok_base)
        a2 = head.aggregate_tokens(tok_shuf)
        m1 = head.sym_maxsim_matrix(tok_base)
        m2 = head.sym_maxsim_matrix(tok_shuf)
        da = float((a1 - a2).abs().max())
        dm = float((m1 - m2).abs().max())
        print(f"     {mode:11s} agg|diff|={da:.2e} msim|diff|={dm:.2e}")
        assert da < 1e-5 and dm < 1e-5, f"{mode}: NOT permutation invariant"

    # =====================================================================  R6
    # gradient flow at init: proj (encoder), gate_res (so residual can switch on),
    # and the layer4 map all receive finite gradient; loss finite.
    print("[R6] gradient flow at init (proj / gate_res / layer4 map):")
    for mode in LEARNABLE:
        head = make_head(OVLIHead, in_ch, proj_dim, grid, setpool=mode)
        tok_g, fmap_g = tokens(head, B, in_ch, requires_grad=True)
        l, _, _ = head.loss(gfeat, tok_g, labels, views)
        head.zero_grad(set_to_none=True)
        l.backward()
        gp = head.proj.weight.grad
        gg = head.setpool_mod.gate_res.grad
        proj_ok = gp is not None and float(gp.abs().sum()) > 0 and torch.isfinite(gp).all()
        # d loss / d gate_res = sum over the residual contribution; non-zero in
        # general because residual(tok) != 0 even at init (only g==0 zeros the
        # FORWARD contribution, not the gradient wrt g).
        gate_ok = gg is not None and torch.isfinite(gg).all()
        fmap_ok = fmap_g.grad is not None and float(fmap_g.grad.abs().sum()) > 0
        # Low-finding guarantee: at step 0 (g==0) the residual MODULE's own params
        # get ZERO gradient (d pooled/d theta = g * d residual/d theta = 0), so they
        # do NOT train until the gate moves off 0.  Check the residual out-proj.
        res_grads = [p.grad for p in head.setpool_mod.out.parameters()
                     if p.grad is not None]
        res_zero = all(float(g.abs().sum()) == 0.0 for g in res_grads) \
            if res_grads else True
        print(f"     {mode:11s} loss={l.item():.5f} finite={bool(torch.isfinite(l))} "
              f"proj.grad|sum|={float(gp.abs().sum()):.3e} "
              f"gate_res.grad={float(gg) if gg is not None else None} "
              f"map_grad={fmap_ok} residual_param_grad=0:{res_zero}")
        assert bool(torch.isfinite(l)) and proj_ok, f"{mode}: proj grad bad"
        assert gate_ok, f"{mode}: gate_res got no/NaN grad (residual can't switch on)"
        assert fmap_ok, f"{mode}: gradient did not reach the layer4 map (encoder)"
        assert res_zero, \
            f"{mode}: residual-module params got NON-zero grad at step 0 (g==0 should zero them)"

    # =====================================================================  R7
    # NaN-safety in residual mode: degenerate tokens + rectangular eval shapes.
    print("[R7] NaN-safety (residual mode):")
    for mode in LEARNABLE:
        head = make_head(OVLIHead, in_ch, proj_dim, grid, setpool=mode)
        with torch.no_grad():                               # exercise the live residual
            head.setpool_mod.gate_res.fill_(0.5)
        for name, raw in (('all-zero', torch.zeros(B, K, proj_dim)),
                          ('all-equal', torch.ones(B, K, proj_dim))):
            tk = F.normalize(raw, dim=2)                    # all-zero -> stays zero
            a = head.aggregate_tokens(tk)
            M = head.sym_maxsim_matrix(tk)
            lo, _, _ = head.loss(F.normalize(torch.zeros(B, Dg) + 1e-6, dim=1),
                                 tk, labels, views)
            fin = bool(torch.isfinite(a).all() and torch.isfinite(M).all()
                       and torch.isfinite(lo))
            print(f"     {mode:11s} degenerate {name:9s}: finite={fin} (loss={float(lo):.4f})")
            assert fin, f"{mode}: NaN/Inf on {name} tokens"
        Nq, Ng = 3, 5
        qa = head.aggregate_tokens(F.normalize(torch.randn(Nq, K, proj_dim), dim=2))
        ga = head.aggregate_tokens(F.normalize(torch.randn(Ng, K, proj_dim), dim=2))
        gram = qa @ ga.t()
        fin_e = bool(torch.isfinite(gram).all())
        print(f"     {mode:11s} eval-shape Nq={Nq} Ng={Ng}: gram={tuple(gram.shape)} finite={fin_e}")
        assert gram.shape == (Nq, Ng) and fin_e, f"{mode}: eval gram wrong/non-finite"

    # =====================================================================  R8
    # STANDALONE fallback (residual=False): gate_res None, forward == original
    # standalone pooling (no mean added); aggregate_tokens L2-normalizes it
    # (cosine gram) -> control preserved, differs from the residual (raw-mean)
    # start.
    print("[R8] standalone fallback (residual=False) preserved & distinct:")
    for mode in LEARNABLE:
        head_s = make_head(OVLIHead, in_ch, proj_dim, grid, setpool=mode,
                           setpool_residual=False)
        assert head_s.setpool_mod.gate_res is None and not head_s.setpool_residual
        tok, _ = tokens(head_s, B, in_ch)
        fwd_s = head_s.setpool_mod(tok)                     # standalone pooled vector
        d_from_mean = float((fwd_s - tok.mean(dim=1)).abs().max())
        # standalone aggregate_tokens IS L2-normalized (cosine gram convention)
        agg_s = head_s.aggregate_tokens(tok)
        is_unit = torch.allclose(agg_s.norm(dim=1),
                                 torch.ones(agg_s.size(0)), atol=1e-5)
        # a residual-mode head on the SAME tokens starts at the RAW mean (unnorm)
        head_r = make_head(OVLIHead, in_ch, proj_dim, grid, setpool=mode,
                           setpool_residual=True)
        head_r._buf['map'] = head_s._buf['map']
        tok_r = head_r.tokens_from_cached_map()
        agg_r = head_r.aggregate_tokens(tok_r)              # raw mean (gate==0)
        d_r_from_mean = float((agg_r - tok_r.mean(dim=1)).abs().max())
        diff_sr = float((agg_s - agg_r).abs().max())
        fin = bool(torch.isfinite(fwd_s).all())
        print(f"     {mode:11s} standalone: gate_res=None finite={fin} unit={is_unit} "
              f"|standalone-mean|={d_from_mean:.3e} |residual-rawmean|={d_r_from_mean:.1e} "
              f"|standalone-residual_start|={diff_sr:.3e}")
        assert fin, f"{mode}: standalone non-finite"
        assert is_unit, f"{mode}: standalone aggregate_tokens not L2-normalized"
        assert d_from_mean > 1e-4, f"{mode}: standalone unexpectedly == mean"
        assert d_r_from_mean <= 1e-6, f"{mode}: residual start != raw mean"
        assert diff_sr > 1e-4, f"{mode}: standalone == residual start (no difference)"

    # =====================================================================  R9
    # RNG PARITY: proj is built BEFORE the set-pool, so a setpool!=mean head and a
    # setpool=mean head built under the SAME seed get byte-identical proj weights
    # (the set-pool's randn/kaiming draws no longer shift proj's init).
    print("[R9] proj RNG parity (setpool!=mean vs setpool=mean, same seed):")
    for mode in LEARNABLE:
        torch.manual_seed(1234)
        h_mean = make_head(OVLIHead, in_ch, proj_dim, grid, setpool='mean')
        torch.manual_seed(1234)
        h_res = make_head(OVLIHead, in_ch, proj_dim, grid, setpool=mode,
                          setpool_residual=True)
        dw = float((h_mean.proj.weight - h_res.proj.weight).abs().max())
        db = float((h_mean.proj.bias - h_res.proj.bias).abs().max())
        eqw = torch.equal(h_mean.proj.weight, h_res.proj.weight)
        eqb = torch.equal(h_mean.proj.bias, h_res.proj.bias)
        print(f"     {mode:11s} proj.weight equal={eqw} (|dw|={dw:.1e}) "
              f"proj.bias equal={eqb} (|db|={db:.1e})")
        assert eqw and eqb, \
            f"{mode}: proj weights differ from setpool=mean under same seed (RNG parity broken)"

    print("\nALL RESIDUAL SMOKE TESTS PASSED")


if __name__ == '__main__':
    main()
