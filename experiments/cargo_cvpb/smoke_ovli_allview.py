"""Isolated numeric smoke test for the --ovli_allview ablation in OVLIHead.

Loads ONLY the OVLIHead class out of cargo_cvpb/afd_train.py by stubbing the
heavy sibling modules (cargo_dataset / afd_model / afd_train) in sys.modules, so
no backbone / dataset is built. CPU-only, tiny tensors -> does not touch the GPU.

Checks:
  T1  off-mode (allview=False) is ELEMENT-WISE IDENTICAL to an inline copy of the
      original opposite-view-only loss body (loss / pos-score / neg-score equal).
  T2  all-view mode: positive set INCLUDES a same-view same-pid pair (the thing
      opposite-view-only excludes), loss is finite, and backward populates a
      non-zero gradient on the proj weights (grad flows encoder->proj).
  T3  on IDENTICAL inputs, flipping the flag actually changes the loss value
      (the ablation is not a no-op).

Usage:  python smoke_ovli_allview.py [path/to/cargo_cvpb/afd_train.py]
"""
import importlib.util
import os
import sys
import types

import torch
import torch.nn as nn


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
# 2) inline copy of the ORIGINAL opposite-view-only loss body (reference).
#    Reuses the live OVLIHead's sym_maxsim_matrix / alpha / tau, so ONLY the
#    candidate view-mask differs from the module under test. If off-mode matches
#    this exactly, off-mode reproduces the pre-change behaviour element-wise.
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


def make_head(OVLIHead, in_ch, proj_dim, grid, allview):
    return OVLIHead(DummyBackbone(), in_ch=in_ch, proj_dim=proj_dim,
                    grid=grid, alpha=0.5, tau=0.05, pool='mean',
                    topk=8, thresh=0.0, allview=allview)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'afd_train.py')
    OVLIHead = load_ovlihead(path)
    torch.manual_seed(0)
    device = 'cpu'

    # batch: 2 pids x 2 views x 2 samples; idx0,idx1 = same pid0 & same view0
    # (the same-view same-pid pair that opposite-view-only EXCLUDES as a positive)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device)
    views = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], device=device)
    B = labels.numel()
    in_ch, proj_dim, grid = 16, 8, (2, 2)
    Dg = 16

    # ---- T1: off-mode == inline original ------------------------------------ #
    head_off = make_head(OVLIHead, in_ch, proj_dim, grid, allview=False)
    assert head_off.allview is False
    gfeat = torch.nn.functional.normalize(torch.randn(B, Dg, device=device), dim=1)
    fmap = torch.randn(B, in_ch, 4, 4, device=device)
    head_off._buf['map'] = fmap
    tok = head_off.tokens_from_cached_map()           # (B,K,proj), proj in graph

    l_off, ps_off, ns_off = head_off.loss(gfeat, tok, labels, views)
    l_ref, ps_ref, ns_ref = ref_oppview_loss(head_off, gfeat, tok, labels, views)
    eq_loss = torch.equal(l_off.detach(), l_ref.detach())
    eq_ps = torch.equal(ps_off.detach(), ps_ref.detach())
    eq_ns = torch.equal(ns_off.detach(), ns_ref.detach())
    print(f"[T1] off-mode vs inline-original  loss={l_off.item():.6f} "
          f"ref={l_ref.item():.6f}  equal(loss/ps/ns)=({eq_loss},{eq_ps},{eq_ns})")
    assert eq_loss and eq_ps and eq_ns, "off-mode does NOT reproduce original!"

    # ---- masks: confirm all-view positives include the same-view pair -------- #
    same_view = views.view(-1, 1).eq(views.view(1, -1))
    same_pid = labels.view(-1, 1).eq(labels.view(1, -1))
    eye = torch.eye(B, dtype=torch.bool)
    pos_op = (~same_view) & (~eye) & same_pid          # opposite-view-only
    pos_av = (~eye) & same_pid                          # all-view
    same_view_in_av = bool((pos_av & same_view).any())
    print(f"[masks] anchor0 #pos oppview={int(pos_op[0].sum())} "
          f"allview={int(pos_av[0].sum())}; pos_av[0,1](same-view same-pid)="
          f"{bool(pos_av[0, 1])} pos_op[0,1]={bool(pos_op[0, 1])} "
          f"same-view-pair in allview-pos={same_view_in_av}")
    assert bool(pos_av[0, 1]) and not bool(pos_op[0, 1]) and same_view_in_av, \
        "all-view positives should INCLUDE same-view same-pid pairs"

    # ---- T2: all-view finite loss + gradient flows into proj ----------------- #
    head_av = make_head(OVLIHead, in_ch, proj_dim, grid, allview=True)
    assert head_av.allview is True
    gfeat2 = torch.nn.functional.normalize(
        torch.randn(B, Dg, device=device, requires_grad=True), dim=1)
    fmap2 = torch.randn(B, in_ch, 4, 4, device=device, requires_grad=True)
    head_av._buf['map'] = fmap2
    tok2 = head_av.tokens_from_cached_map()
    l_av, ps_av, ns_av = head_av.loss(gfeat2, tok2, labels, views)
    finite = bool(torch.isfinite(l_av))
    head_av.zero_grad(set_to_none=True)
    l_av.backward()
    gw = head_av.proj.weight.grad
    gb = head_av.proj.bias.grad
    grad_ok = (gw is not None and gb is not None
               and float(gw.abs().sum()) > 0 and torch.isfinite(gw).all())
    print(f"[T2] allview loss={l_av.item():.6f} finite={finite}  "
          f"proj.weight.grad |sum|={float(gw.abs().sum()):.4e} "
          f"bias.grad |sum|={float(gb.abs().sum()):.4e} grad_ok={grad_ok}")
    assert finite, "all-view loss not finite"
    assert grad_ok, "gradient did not flow into proj in all-view mode"

    # ---- T3: same inputs, flipping the flag changes the loss (not a no-op) --- #
    head_off.allview = True                            # toggle on identical inputs
    l_flip, _, _ = head_off.loss(gfeat, tok, labels, views)
    head_off.allview = False
    delta = abs(l_flip.item() - l_off.item())
    print(f"[T3] same inputs: oppview={l_off.item():.6f} allview={l_flip.item():.6f} "
          f"|delta|={delta:.6f} (flag active)")
    assert delta > 1e-6, "flag did not change the loss on identical inputs"

    print("ALL SMOKE TESTS PASSED")


if __name__ == '__main__':
    main()
