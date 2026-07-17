"""Isolated numeric smoke test for the two OVLIHead ablation switches.

Covers the two reviewer-defense ablations added to cargo_cvpb/afd_train.py:
  --ovli_match {maxsim, avg}   : late-interaction token-match reduction.
  --ovli_align {free, ordered} : free vs AlignedReID row-ordered alignment.

Loads the REAL OVLIHead out of cargo_cvpb/afd_train.py (not a copy) by stubbing
the heavy sibling modules (cargo_dataset / afd_model / afd_train) in sys.modules,
so no backbone / dataset is built. CPU-only, tiny tensors -> never touches a GPU.

Checks
  T1  OFF-MODE BYTE IDENTITY (match=maxsim, align=free):
        T1a  _reduce_other default == sim.max(dim) element-wise.
        T1b  sym_maxsim_matrix == an inline copy of the ORIGINAL free+max formula.
        T1c  loss / pos-score / neg-score == an inline copy of the ORIGINAL loss.
  T2  OPTIMIZER / CHECKPOINT ISOLATION:
        the new _row_mask4 is a NON-persistent buffer -> NOT a parameter (the
        optimizer self-check still sees exactly the 2 proj tensors) and NOT in
        state_dict (old OVLI checkpoints stay loadable).
  T3  align='ordered' (AlignedReID row-correspondence):
        T3a  decisive synthetic test: a query token's match is restricted to its
             own grid row (free picks the big cross-row sim, ordered the in-row).
        T3b  full sym_maxsim_matrix == a row-restricted reference, stays symmetric.
        T3c  finite loss, gradient flows into proj, and the value DIFFERS from
             default (the flag is not a no-op).
  T4  match='avg' (soft global match):
        T4a  sym_maxsim_matrix ~= <mean_query_token, mean_gallery_token> (the
             documented "degenerates to a near-global soft match" semantics).
        T4b  finite loss, gradient flows into proj, value DIFFERS from default.
  T5  SYMMETRY + NaN-SAFETY:
        sym_maxsim symmetric (and diag~=1 for the two maxsim modes) in all four
        mode combos; ordered & ordered+avg stay finite on degenerate (all-equal,
        all-zero) tokens; the rectangular eval-shape (Nq != Ng) reduction used by
        the rerank path is finite and matches the row-restricted reference.

Usage:  python smoke_ovli_ablations.py [path/to/cargo_cvpb/afd_train.py]
        (run inside an env with torch+numpy, e.g.
         `uv run --no-project --with numpy --with torch python smoke_ovli_ablations.py`)
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


_dummy = lambda *a, **k: None  # noqa: E731  (placeholder for names never called here)
_stub('cargo_dataset', dict(CARGO=_dummy, CARGOImageDataset=_dummy,
                            build_transforms=_dummy, RandomIdentitySampler=_dummy,
                            filter_by_view=_dummy))
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
# 2) inline references replicating the PRE-ABLATION behaviour (free + max).
#    These reuse OVLIHead.pool_token_max (an unchanged static method) so ONLY
#    the inner token reduction is what we compare against.
# --------------------------------------------------------------------------- #
def original_sym_maxsim(OVLIHead, tok, pool='mean', topk=8, thresh=0.0, tau=0.05):
    """Exact copy of the ORIGINAL sym_maxsim_matrix inner logic (free + max)."""
    B, K, D = tok.shape
    flat = tok.reshape(B * K, D)
    sim = (flat @ flat.t()).reshape(B, K, B, K)
    i2j_max = sim.max(dim=3).values
    i2j = OVLIHead.pool_token_max(i2j_max, dim=1, pool=pool, topk=topk,
                                  thresh=thresh, tau=tau)
    j2i_max = sim.max(dim=1).values
    j2i = OVLIHead.pool_token_max(j2i_max, dim=2, pool=pool, topk=topk,
                                  thresh=thresh, tau=tau)
    return 0.5 * (i2j + j2i)


def original_loss(ovli, OVLIHead, gfeat, tok, labels, views):
    """Exact copy of the ORIGINAL OVLIHead.loss body, msim via the free+max
    reference above (so it is independent of the module under test)."""
    B = gfeat.size(0)
    device = gfeat.device
    gsim = gfeat @ gfeat.t()
    msim = original_sym_maxsim(OVLIHead, tok, pool=ovli.pool, topk=ovli.topk,
                               thresh=ovli.thresh, tau=ovli.tau)
    score = ovli.alpha * gsim + (1.0 - ovli.alpha) * msim
    same_view = views.view(-1, 1).eq(views.view(1, -1))
    same_pid = labels.view(-1, 1).eq(labels.view(1, -1))
    eye = torch.eye(B, dtype=torch.bool, device=device)
    if ovli.allview:
        cand = ~eye
    else:
        cand = (~same_view) & (~eye)
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


def ref_ordered_sym_maxsim(OVLIHead, tok, gh, gw, pool='mean', topk=8,
                           thresh=0.0, tau=0.05):
    """Row-restricted reference: each query token may only match same-row others."""
    B, K, D = tok.shape
    rows = torch.arange(K) // gw
    row_eq = rows.view(K, 1).eq(rows.view(1, K))           # (K,K)
    flat = tok.reshape(B * K, D)
    sim = (flat @ flat.t()).reshape(B, K, B, K)
    mask = row_eq.view(1, K, 1, K)
    floor = sim.new_full((), -1e4)
    masked = torch.where(mask, sim, floor)
    i2j_max = masked.max(dim=3).values
    i2j = OVLIHead.pool_token_max(i2j_max, dim=1, pool=pool, topk=topk,
                                  thresh=thresh, tau=tau)
    j2i_max = masked.max(dim=1).values
    j2i = OVLIHead.pool_token_max(j2i_max, dim=2, pool=pool, topk=topk,
                                  thresh=thresh, tau=tau)
    return 0.5 * (i2j + j2i)


class DummyBackbone(nn.Module):
    """Minimal model exposing .layer4 so OVLIHead can register its hook."""
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Identity()


def make_head(OVLIHead, in_ch, proj_dim, grid, match='maxsim', align='free',
              pool='mean'):
    return OVLIHead(DummyBackbone(), in_ch=in_ch, proj_dim=proj_dim, grid=grid,
                    alpha=0.5, tau=0.05, pool=pool, topk=8, thresh=0.0,
                    allview=False, match=match, align=align)


def tokens(head, B, in_ch, hw=(16, 8), requires_grad=False):
    """Feed a synthetic layer4 map through the head's real proj -> L2 tokens."""
    fmap = torch.randn(B, in_ch, hw[0], hw[1], requires_grad=requires_grad)
    head._buf['map'] = fmap
    return head.tokens_from_cached_map(), fmap


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'afd_train.py')
    OVLIHead = load_ovlihead(path)
    torch.manual_seed(0)

    # opposite-view PK batch: 4 pids x {aerial(0), ground(1)} -> opp-view positives
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    views = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    B = labels.numel()
    in_ch, proj_dim, grid = 16, 8, (8, 4)             # K = 32 tokens (8 rows x 4)
    gh, gw = grid
    Dg = 12

    # =====================================================================  T1
    head_def = make_head(OVLIHead, in_ch, proj_dim, grid)   # maxsim + free
    assert head_def.match == 'maxsim' and head_def.align == 'free'
    tok, _ = tokens(head_def, B, in_ch)

    # T1a: _reduce_other default == sim.max(dim) element-wise
    flat = tok.reshape(B * 32, proj_dim)
    sim = (flat @ flat.t()).reshape(B, 32, B, 32)
    d3 = torch.equal(head_def._reduce_other(sim, other_dim=3), sim.max(dim=3).values)
    d1 = torch.equal(head_def._reduce_other(sim, other_dim=1), sim.max(dim=1).values)
    print(f"[T1a] _reduce_other default == sim.max  (dim3={d3}, dim1={d1})")
    assert d3 and d1, "default _reduce_other is NOT a plain max -> off-mode changed!"

    # T1b: sym_maxsim_matrix == inline original free+max formula
    m_def = head_def.sym_maxsim_matrix(tok)
    m_ref = original_sym_maxsim(OVLIHead, tok, pool='mean', tau=0.05)
    eq_sym = torch.equal(m_def, m_ref)
    print(f"[T1b] sym_maxsim default == original formula: equal={eq_sym} "
          f"(max|diff|={float((m_def - m_ref).detach().abs().max()):.3e})")
    assert eq_sym, "off-mode sym_maxsim diverged from the original"

    # T1c: full loss == inline original loss (loss/pos/neg)
    gfeat = F.normalize(torch.randn(B, Dg), dim=1)
    l_def, p_def, n_def = head_def.loss(gfeat, tok, labels, views)
    l_ref, p_ref, n_ref = original_loss(head_def, OVLIHead, gfeat, tok, labels, views)
    eqs = (torch.equal(l_def.detach(), l_ref.detach()),
           torch.equal(p_def.detach(), p_ref.detach()),
           torch.equal(n_def.detach(), n_ref.detach()))
    print(f"[T1c] loss default == original: loss={l_def.item():.6f} "
          f"ref={l_ref.item():.6f} equal(loss/ps/ns)={eqs}")
    assert all(eqs), "off-mode loss diverged from the original"

    # =====================================================================  T2
    names_p = [n for n, _ in head_def.named_parameters()]
    names_b = [n for n, _ in head_def.named_buffers()]
    n_params = len(list(head_def.parameters()))
    in_state = '_row_mask4' in head_def.state_dict()
    print(f"[T2] params={names_p} (#={n_params})  buffers={names_b}  "
          f"_row_mask4 in state_dict={in_state}")
    assert names_p == ['proj.weight', 'proj.bias'] and n_params == 2, \
        "row mask leaked into parameters -> optimizer self-check would change"
    assert '_row_mask4' in names_b, "_row_mask4 should be a (registered) buffer"
    assert not in_state, "_row_mask4 must be NON-persistent (keep old ckpts loadable)"

    # =====================================================================  T3
    # T3a: decisive synthetic row-restriction (grid 2x2 -> rows [0,0,1,1])
    head_o22 = make_head(OVLIHead, in_ch, proj_dim, (2, 2), align='ordered')
    head_f22 = make_head(OVLIHead, in_ch, proj_dim, (2, 2), align='free')
    s = torch.full((1, 4, 1, 4), -1.0)
    # query token 0 is in row0 (cols {0,1}); a BIG sim sits in row1 (col2)
    s[0, 0, 0, 0] = 0.10
    s[0, 0, 0, 1] = 0.20        # best WITHIN row0
    s[0, 0, 0, 2] = 0.90        # best overall, but CROSS-row (row1)
    s[0, 0, 0, 3] = 0.80
    free_v = float(head_f22._reduce_other(s, other_dim=3)[0, 0, 0])
    ord_v = float(head_o22._reduce_other(s, other_dim=3)[0, 0, 0])
    print(f"[T3a] synthetic row-restriction: free max={free_v:.2f} (cross-row 0.90) "
          f"ordered max={ord_v:.2f} (in-row 0.20)")
    assert abs(free_v - 0.90) < 1e-6, "free should pick the global (cross-row) max"
    assert abs(ord_v - 0.20) < 1e-6, "ordered should pick the in-row max only"

    # T3b: full ordered sym_maxsim == row-restricted reference, and symmetric
    head_ord = make_head(OVLIHead, in_ch, proj_dim, grid, align='ordered')
    tok_o, _ = tokens(head_ord, B, in_ch)
    m_ord = head_ord.sym_maxsim_matrix(tok_o)
    m_ord_ref = ref_ordered_sym_maxsim(OVLIHead, tok_o, gh, gw, pool='mean', tau=0.05)
    eq_ord = torch.equal(m_ord, m_ord_ref)
    sym_ord = torch.allclose(m_ord, m_ord.t(), atol=1e-6)
    print(f"[T3b] ordered sym_maxsim == row-restricted ref: equal={eq_ord} "
          f"symmetric={sym_ord} "
          f"(max|diff|={float((m_ord - m_ord_ref).detach().abs().max()):.3e})")
    assert eq_ord and sym_ord, "ordered sym_maxsim wrong or not symmetric"

    # T3c: ordered loss finite, grad flows into proj, differs from default
    head_ord2 = make_head(OVLIHead, in_ch, proj_dim, grid, align='ordered')
    tok_og, fmap_og = tokens(head_ord2, B, in_ch, requires_grad=True)
    l_ord, _, _ = head_ord2.loss(gfeat, tok_og, labels, views)
    head_ord2.zero_grad(set_to_none=True)
    l_ord.backward()
    gw_ord = head_ord2.proj.weight.grad
    grad_ok = (gw_ord is not None and float(gw_ord.abs().sum()) > 0
               and torch.isfinite(gw_ord).all())
    # same proj weights as default? no (fresh init) -> compare on SAME tokens via toggle
    head_def.align = 'ordered'
    l_def_ord, _, _ = head_def.loss(gfeat, tok, labels, views)
    head_def.align = 'free'
    delta = abs(l_def_ord.item() - l_def.item())
    print(f"[T3c] ordered loss={l_ord.item():.6f} finite={bool(torch.isfinite(l_ord))} "
          f"proj.grad|sum|={float(gw_ord.abs().sum()):.4e} grad_ok={grad_ok}; "
          f"toggle on same inputs free={l_def.item():.6f} ordered={l_def_ord.item():.6f} "
          f"|delta|={delta:.6f}")
    assert bool(torch.isfinite(l_ord)) and grad_ok, "ordered loss/grad bad"
    assert delta > 1e-6, "align=ordered is a no-op on identical inputs"

    # =====================================================================  T4
    # T4a: avg sym_maxsim ~= <mean_query_token, mean_gallery_token> (soft global)
    head_avg = make_head(OVLIHead, in_ch, proj_dim, grid, match='avg')
    tok_a, _ = tokens(head_avg, B, in_ch)
    m_avg = head_avg.sym_maxsim_matrix(tok_a)
    mean_tok = tok_a.mean(dim=1)                            # (B,D) mean token
    m_global = mean_tok @ mean_tok.t()                     # (B,B) soft-global ref
    close = torch.allclose(m_avg, m_global, atol=1e-5)
    sym_avg = torch.allclose(m_avg, m_avg.t(), atol=1e-6)
    print(f"[T4a] avg sym_maxsim ~= <mean_q,mean_g>: close={close} symmetric={sym_avg} "
          f"(max|diff|={float((m_avg - m_global).detach().abs().max()):.3e})")
    assert close and sym_avg, "match=avg should reduce to the mean-token soft global"

    # T4b: avg loss finite, grad flows into proj, differs from default
    head_avg2 = make_head(OVLIHead, in_ch, proj_dim, grid, match='avg')
    tok_ag, _ = tokens(head_avg2, B, in_ch, requires_grad=True)
    l_avg, _, _ = head_avg2.loss(gfeat, tok_ag, labels, views)
    head_avg2.zero_grad(set_to_none=True)
    l_avg.backward()
    gw_avg = head_avg2.proj.weight.grad
    grad_ok_a = (gw_avg is not None and float(gw_avg.abs().sum()) > 0
                 and torch.isfinite(gw_avg).all())
    head_def.match = 'avg'
    l_def_avg, _, _ = head_def.loss(gfeat, tok, labels, views)
    head_def.match = 'maxsim'
    delta_a = abs(l_def_avg.item() - l_def.item())
    print(f"[T4b] avg loss={l_avg.item():.6f} finite={bool(torch.isfinite(l_avg))} "
          f"proj.grad|sum|={float(gw_avg.abs().sum()):.4e} grad_ok={grad_ok_a}; "
          f"toggle on same inputs maxsim={l_def.item():.6f} avg={l_def_avg.item():.6f} "
          f"|delta|={delta_a:.6f}")
    assert bool(torch.isfinite(l_avg)) and grad_ok_a, "avg loss/grad bad"
    assert delta_a > 1e-6, "match=avg is a no-op on identical inputs"

    # =====================================================================  T5
    # symmetry + diag for all four combos
    combos = [('maxsim', 'free'), ('maxsim', 'ordered'), ('avg', 'free'),
              ('avg', 'ordered')]
    for mt, al in combos:
        h = make_head(OVLIHead, in_ch, proj_dim, grid, match=mt, align=al)
        tk, _ = tokens(h, B, in_ch)
        M = h.sym_maxsim_matrix(tk)
        sym = torch.allclose(M, M.t(), atol=1e-6)
        finite = bool(torch.isfinite(M).all())
        diag = float(M.diag().mean().detach())
        # maxsim self-match always includes the identical token -> diag == 1
        diag_ok = (abs(diag - 1.0) < 1e-5) if mt == 'maxsim' else True
        print(f"[T5] combo match={mt:6s} align={al:7s}  symmetric={sym} "
              f"finite={finite} diag_mean={diag:.4f} diag_ok={diag_ok}")
        assert sym and finite and diag_ok, f"combo {mt}/{al} broke symmetry/finite/diag"

    # NaN-safety on degenerate tokens (all-equal, all-zero) for ordered & ordered+avg
    for mt in ('maxsim', 'avg'):
        h = make_head(OVLIHead, in_ch, proj_dim, grid, match=mt, align='ordered')
        for name, raw in (('all-equal', torch.ones(B, 32, proj_dim)),
                          ('all-zero', torch.zeros(B, 32, proj_dim))):
            tk = F.normalize(raw, dim=2)        # all-zero -> normalize keeps zeros
            M = h.sym_maxsim_matrix(tk)
            lo, _, _ = h.loss(F.normalize(torch.zeros(B, Dg) + 1e-6, dim=1),
                              tk, labels, views)
            fin = bool(torch.isfinite(M).all()) and bool(torch.isfinite(lo))
            print(f"[T5] ordered/{mt:6s} degenerate {name:9s}: "
                  f"sym/loss finite={fin} (loss={float(lo):.4f})")
            assert fin, f"ordered/{mt} produced NaN/Inf on {name} tokens"

    # rectangular eval-shape (Nq != Ng) reduction used by the rerank path
    Nq, Ng, K = 3, 5, 32
    qt = F.normalize(torch.randn(Nq, K, proj_dim), dim=2).reshape(Nq * K, proj_dim)
    gt = F.normalize(torch.randn(Ng, K, proj_dim), dim=2)
    sim_e = (qt @ gt.reshape(Ng * K, proj_dim).t()).reshape(Nq, K, Ng, K)
    q2g = head_ord._reduce_other(sim_e, other_dim=3)       # (Nq,K,Ng)
    g2q = head_ord._reduce_other(sim_e, other_dim=1)       # (Nq,Ng,K)
    # row-restricted reference for the q->g direction
    rows = torch.arange(K) // gw
    row_eq = rows.view(K, 1).eq(rows.view(1, K)).view(1, K, 1, K)
    ref_q2g = torch.where(row_eq, sim_e, sim_e.new_full((), -1e4)).max(dim=3).values
    eq_e = torch.equal(q2g, ref_q2g)
    fin_e = bool(torch.isfinite(q2g).all() and torch.isfinite(g2q).all())
    print(f"[T5] eval-shape Nq={Nq} Ng={Ng}: q2g={tuple(q2g.shape)} "
          f"g2q={tuple(g2q.shape)} ordered==ref={eq_e} finite={fin_e}")
    assert q2g.shape == (Nq, K, Ng) and g2q.shape == (Nq, Ng, K), "eval reduce shape"
    assert eq_e and fin_e, "eval-path ordered reduction wrong/non-finite"

    print("\nALL SMOKE TESTS PASSED")


if __name__ == '__main__':
    main()
