"""Isolated numeric smoke test for the --ovli_setpool learnable set-pooling.

Covers the new switch added to cargo_cvpb/afd_train.py:
  --ovli_setpool {mean, netvlad, attn, gated, secondorder}
which replaces the "mean over the K projected tokens" aggregation (the headline
--ovli_match avg --ovli_pool mean config, which reduces EXACTLY to the gram of
the per-sample mean of the K L2-normed tokens = the current best mechanism) with
a learnable, permutation-invariant set pooling.

Loads the REAL OVLIHead + OVLISetPool out of cargo_cvpb/afd_train.py (not a copy)
by stubbing the heavy sibling modules (cargo_dataset / agreid_dataset / afd_model
/ afd_train) in sys.modules, so no backbone / dataset is built.  CPU-only, tiny
tensors -> never touches a GPU.

Checks
  T1  OFF-MODE BYTE IDENTITY (setpool='mean'):
        T1a  setpool=mean,match=maxsim  sym_maxsim_matrix == inline ORIGINAL
             free+max formula (torch.equal) -> the default default is untouched.
        T1b  setpool=mean,match=avg     sym_maxsim_matrix == inline ORIGINAL
             free+avg formula (torch.equal) AND ~= <mean_q,mean_g> (the 52.37
             aggregation the learnable pools replace).
        T1c  full loss (setpool=mean, both match modes) == inline ORIGINAL loss
             (loss / pos-score / neg-score equal to <=1e-6).
        T1d  setpool=mean builds NO extra module (setpool_mod is None) and adds
             NO parameters beyond proj (optimizer/checkpoint unchanged).
  T2  PERMUTATION INVARIANCE (the load-bearing property):
        for every learnable mode, shuffling the K tokens leaves aggregate_tokens
        AND sym_maxsim_matrix unchanged to <1e-5; the gram is symmetric & finite.
  T3  OPTIMIZER MEMBERSHIP:
        a mirror of the afd_train self-check -- every setpool_mod parameter id is
        in AdamW(list(model.params)+list(ovli.params)); proj params too.
  T4  GRADIENT FLOW:
        for every learnable mode the loss is finite and backward populates a
        NON-zero, finite gradient on BOTH proj.weight (encoder path) and the
        setpool_mod params (the new learnable pooling is actually trained).
  T5  NOT A NO-OP:
        on IDENTICAL tokens, swapping setpool mean->learnable changes msim/loss.
  T6  NaN-SAFETY:
        degenerate tokens (all-zero, all-equal) keep aggregate/msim/loss finite
        for every mode; rectangular eval shapes (Nq != Ng) gram is finite.
  T7  PARAMETER COUNTS:
        report the real (dim=256) parameter count + breakdown of each learnable
        set pool (paper-table material).

Usage:  python smoke_ovli_setpool.py [path/to/cargo_cvpb/afd_train.py]
        (run inside an env with torch+numpy, e.g.
         `uv run --no-project --with numpy --with torch python smoke_ovli_setpool.py`)
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
# 2) inline references replicating the ORIGINAL (pre-setpool) behaviour.
#    These reuse OVLIHead.pool_token_max (an unchanged static method) so the
#    reference is independent of the setpool branch under test.
# --------------------------------------------------------------------------- #
def original_sym_maxsim(OVLIHead, tok, match='maxsim', pool='mean', topk=8,
                        thresh=0.0, tau=0.05):
    """Exact copy of the ORIGINAL sym_maxsim_matrix inner logic (free align)."""
    B, K, D = tok.shape
    flat = tok.reshape(B * K, D)
    sim = (flat @ flat.t()).reshape(B, K, B, K)
    if match == 'maxsim':
        i2j_max = sim.max(dim=3).values
        j2i_max = sim.max(dim=1).values
    else:  # avg
        i2j_max = sim.mean(dim=3)
        j2i_max = sim.mean(dim=1)
    i2j = OVLIHead.pool_token_max(i2j_max, dim=1, pool=pool, topk=topk,
                                  thresh=thresh, tau=tau)
    j2i = OVLIHead.pool_token_max(j2i_max, dim=2, pool=pool, topk=topk,
                                  thresh=thresh, tau=tau)
    return 0.5 * (i2j + j2i)


def original_loss(ovli, OVLIHead, gfeat, tok, labels, views):
    """Exact copy of the ORIGINAL OVLIHead.loss body, msim via the reference
    above (so it is independent of the module under test)."""
    B = gfeat.size(0)
    device = gfeat.device
    gsim = gfeat @ gfeat.t()
    msim = original_sym_maxsim(OVLIHead, tok, match=ovli.match, pool=ovli.pool,
                               topk=ovli.topk, thresh=ovli.thresh, tau=ovli.tau)
    score = ovli.alpha * gsim + (1.0 - ovli.alpha) * msim
    same_view = views.view(-1, 1).eq(views.view(1, -1))
    same_pid = labels.view(-1, 1).eq(labels.view(1, -1))
    eye = torch.eye(B, dtype=torch.bool, device=device)
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


def make_head(OVLIHead, in_ch, proj_dim, grid, setpool='mean', match='maxsim',
              vlad_clusters=8, attn_heads=4, so_rank=32, setpool_residual=True):
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

    # opposite-view PK batch: 4 pids x {aerial(0), ground(1)} -> opp-view positives
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    views = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    B = labels.numel()
    in_ch, proj_dim, grid = 16, 8, (8, 4)             # K = 32 tokens (8 rows x 4)
    K = grid[0] * grid[1]
    Dg = 12
    gfeat = F.normalize(torch.randn(B, Dg), dim=1)

    # =====================================================================  T1
    # off-mode byte identity: setpool='mean' must reproduce the original exactly
    # for BOTH match modes (the default default match=maxsim, and the headline
    # match=avg whose mean-pool the learnable variants replace).
    for match in ('maxsim', 'avg'):
        head = make_head(OVLIHead, in_ch, proj_dim, grid, setpool='mean', match=match)
        assert head.setpool == 'mean' and head.setpool_mod is None
        tok, _ = tokens(head, B, in_ch)
        m_new = head.sym_maxsim_matrix(tok)
        m_ref = original_sym_maxsim(OVLIHead, tok, match=match, pool='mean', tau=0.05)
        eq_sym = torch.equal(m_new, m_ref)
        maxd = float((m_new - m_ref).detach().abs().max())
        print(f"[T1{'a' if match == 'maxsim' else 'b'}] setpool=mean match={match:6s}"
              f"  sym_maxsim == original: equal={eq_sym} max|diff|={maxd:.2e}")
        assert eq_sym and maxd <= 1e-6, f"off-mode (match={match}) diverged from original"
        # T1b extra: avg reduces to the gram of the (un-normalized) mean token
        if match == 'avg':
            mean_tok = tok.mean(dim=1)                   # (B,D) the aggregation replaced
            m_global = mean_tok @ mean_tok.t()
            close = torch.allclose(m_new, m_global, atol=1e-5)
            print(f"[T1b] match=avg == <mean_q,mean_g> (the 52.37 mean-pool): "
                  f"close={close} max|diff|={float((m_new - m_global).detach().abs().max()):.2e}")
            assert close, "match=avg should equal the gram of mean-pooled tokens"
        # T1c: full loss equals the inline original loss
        l_new, p_new, n_new = head.loss(gfeat, tok, labels, views)
        l_ref, p_ref, n_ref = original_loss(head, OVLIHead, gfeat, tok, labels, views)
        eqs = (torch.equal(l_new.detach(), l_ref.detach()),
               torch.equal(p_new.detach(), p_ref.detach()),
               torch.equal(n_new.detach(), n_ref.detach()))
        print(f"[T1c] setpool=mean match={match:6s} loss == original: "
              f"loss={l_new.item():.6f} ref={l_ref.item():.6f} equal(l/ps/ns)={eqs}")
        assert all(eqs), f"off-mode loss (match={match}) diverged from original"

    # T1d: mean adds no module / no extra params (optimizer & ckpt unchanged)
    head_mean = make_head(OVLIHead, in_ch, proj_dim, grid, setpool='mean')
    names_p = [n for n, _ in head_mean.named_parameters()]
    print(f"[T1d] setpool=mean params={names_p} (setpool_mod={head_mean.setpool_mod})")
    assert names_p == ['proj.weight', 'proj.bias'], \
        "setpool=mean must NOT add params (optimizer self-check unchanged)"
    assert head_mean.setpool_mod is None

    # =====================================================================  T2
    # permutation invariance: shuffling the K tokens must not change the output.
    tok_base, _ = tokens(make_head(OVLIHead, in_ch, proj_dim, grid), B, in_ch)
    perm = torch.randperm(K)
    assert not torch.equal(perm, torch.arange(K)), "need a real shuffle"
    tok_shuf = tok_base[:, perm, :].contiguous()
    for mode in LEARNABLE:
        head = make_head(OVLIHead, in_ch, proj_dim, grid, setpool=mode)
        a1 = head.aggregate_tokens(tok_base)
        a2 = head.aggregate_tokens(tok_shuf)
        m1 = head.sym_maxsim_matrix(tok_base)
        m2 = head.sym_maxsim_matrix(tok_shuf)
        da = float((a1 - a2).abs().max())
        dm = float((m1 - m2).abs().max())
        sym = torch.allclose(m1, m1.t(), atol=1e-6)
        fin = bool(torch.isfinite(m1).all() and torch.isfinite(a1).all())
        print(f"[T2] {mode:11s} perm-invariant: agg max|diff|={da:.2e} "
              f"msim max|diff|={dm:.2e} symmetric={sym} finite={fin}")
        assert da < 1e-5 and dm < 1e-5, f"{mode} is NOT permutation invariant"
        assert sym and fin, f"{mode} gram not symmetric/finite"

    # =====================================================================  T3
    # optimizer membership: mirror the afd_train self-check for each mode.
    for mode in LEARNABLE:
        model = DummyBackbone()
        head = make_head(OVLIHead, in_ch, proj_dim, grid, setpool=mode)
        opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                                lr=1e-3)
        opt_ids = {id(p) for grp in opt.param_groups for p in grp['params']}
        proj_in = all(id(p) in opt_ids for p in head.proj.parameters())
        sp_params = list(head.setpool_mod.parameters())
        sp_in = all(id(p) in opt_ids for p in sp_params)
        print(f"[T3] {mode:11s} in optimizer: proj={proj_in} setpool={sp_in} "
              f"({len(sp_params)} tensors)")
        assert proj_in and sp_in, f"{mode} params NOT all in optimizer"

    # =====================================================================  T4
    # gradient flow into BOTH proj (encoder path) and the setpool params.
    for mode in LEARNABLE:
        head = make_head(OVLIHead, in_ch, proj_dim, grid, setpool=mode)
        tok_g, fmap_g = tokens(head, B, in_ch, requires_grad=True)
        l, _, _ = head.loss(gfeat, tok_g, labels, views)
        head.zero_grad(set_to_none=True)
        l.backward()
        gp = head.proj.weight.grad
        proj_ok = (gp is not None and float(gp.abs().sum()) > 0 and torch.isfinite(gp).all())
        sp_grads = [p.grad for p in head.setpool_mod.parameters() if p.requires_grad]
        sp_ok = all(g is not None and torch.isfinite(g).all() for g in sp_grads)
        sp_nonzero = sum(float(g.abs().sum()) for g in sp_grads if g is not None)
        fmap_ok = fmap_g.grad is not None and float(fmap_g.grad.abs().sum()) > 0
        print(f"[T4] {mode:11s} loss={l.item():.5f} finite={bool(torch.isfinite(l))} "
              f"proj.grad|sum|={float(gp.abs().sum()):.3e} "
              f"setpool.grad|sum|={sp_nonzero:.3e} (proj_ok={proj_ok} sp_ok={sp_ok} "
              f"map_grad={fmap_ok})")
        assert bool(torch.isfinite(l)) and proj_ok, f"{mode} proj grad bad"
        assert sp_ok and sp_nonzero > 0, f"{mode} setpool params got no/NaN grad"
        assert fmap_ok, f"{mode} gradient did not reach the layer4 map (encoder)"

    # =====================================================================  T5
    # not a no-op: on IDENTICAL tokens, mean vs a learnable pool gives a different
    # score.  NOTE: a RESIDUAL pool (setpool_residual=True) starts EXACTLY at the
    # un-normalized mean gram BY DESIGN (the lossless 52.37 start, asserted in
    # smoke_ovli_residual.py R1) -- so it is intentionally a no-op at init.  The
    # config that genuinely differs from mean-pool is the STANDALONE pool
    # (setpool_residual=False), whose random-init aggregation replaces the mean;
    # T5 checks THAT to prove the learnable aggregation is not vacuous.
    tok_id, _ = tokens(make_head(OVLIHead, in_ch, proj_dim, grid), B, in_ch)
    head_avg = make_head(OVLIHead, in_ch, proj_dim, grid, setpool='mean', match='avg')
    m_mean = head_avg.sym_maxsim_matrix(tok_id)
    l_mean, _, _ = head_avg.loss(gfeat, tok_id, labels, views)
    for mode in LEARNABLE:
        head = make_head(OVLIHead, in_ch, proj_dim, grid, setpool=mode,
                         setpool_residual=False)        # standalone: differs from mean
        m_sp = head.sym_maxsim_matrix(tok_id)
        l_sp, _, _ = head.loss(gfeat, tok_id, labels, views)
        dmsim = float((m_sp - m_mean).abs().max())
        dl = abs(l_sp.item() - l_mean.item())
        print(f"[T5] {mode:11s} standalone vs mean-pool on SAME tokens: "
              f"msim max|diff|={dmsim:.3e} |loss diff|={dl:.4f}")
        assert dmsim > 1e-5 and dl > 1e-6, f"{mode} standalone is a no-op vs mean-pool"
    # T5b: a RESIDUAL pool at init IS a no-op vs the un-normalized mean gram (the
    # intended lossless start) -- the complement of the standalone check above.
    for mode in LEARNABLE:
        head = make_head(OVLIHead, in_ch, proj_dim, grid, setpool=mode,
                         setpool_residual=True)
        m_sp = head.sym_maxsim_matrix(tok_id).detach()
        dmsim = float((m_sp - m_mean.detach()).abs().max())
        print(f"[T5b] {mode:11s} residual init == mean-pool (lossless start): "
              f"msim max|diff|={dmsim:.3e}")
        assert dmsim <= 1e-6, f"{mode} residual init should match mean-pool (got {dmsim:.2e})"

    # =====================================================================  T6
    # NaN-safety on degenerate tokens + rectangular eval shapes.
    for mode in LEARNABLE:
        head = make_head(OVLIHead, in_ch, proj_dim, grid, setpool=mode)
        for name, raw in (('all-zero', torch.zeros(B, K, proj_dim)),
                          ('all-equal', torch.ones(B, K, proj_dim))):
            tk = F.normalize(raw, dim=2)            # all-zero -> stays zero
            a = head.aggregate_tokens(tk)
            M = head.sym_maxsim_matrix(tk)
            lo, _, _ = head.loss(F.normalize(torch.zeros(B, Dg) + 1e-6, dim=1),
                                 tk, labels, views)
            fin = bool(torch.isfinite(a).all() and torch.isfinite(M).all()
                       and torch.isfinite(lo))
            print(f"[T6] {mode:11s} degenerate {name:9s}: agg/msim/loss finite={fin} "
                  f"(loss={float(lo):.4f})")
            assert fin, f"{mode} produced NaN/Inf on {name} tokens"
        # rectangular eval shape (Nq != Ng), as in the rerank gram path
        Nq, Ng = 3, 5
        qa = head.aggregate_tokens(F.normalize(torch.randn(Nq, K, proj_dim), dim=2))
        ga = head.aggregate_tokens(F.normalize(torch.randn(Ng, K, proj_dim), dim=2))
        gram = qa @ ga.t()
        fin_e = bool(torch.isfinite(gram).all())
        print(f"[T6] {mode:11s} eval-shape Nq={Nq} Ng={Ng}: gram={tuple(gram.shape)} "
              f"finite={fin_e}")
        assert gram.shape == (Nq, Ng) and fin_e, f"{mode} eval gram wrong/non-finite"

    # =====================================================================  T7
    # parameter counts at the REAL projection dim (paper-table material).
    print("\n[T7] learnable set-pool parameter counts (dim=256, out_dim=256):")
    D = 256
    for mode in LEARNABLE:
        sp = OVLISetPool(mode, dim=D, out_dim=D, clusters=8, heads=4, lowrank=32)
        total = sum(p.numel() for p in sp.parameters())
        breakdown = ", ".join(f"{n}={p.numel()}" for n, p in sp.named_parameters())
        print(f"       {mode:11s}: {total:>8d} params  [{breakdown}]")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == '__main__':
    main()
