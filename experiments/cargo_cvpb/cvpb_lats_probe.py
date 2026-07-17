#!/usr/bin/env python3
"""LATS probe -- Lattice-Aware Token Sidecar (cheaper stripe-pooled version), frozen backbone.

The ONLY LM-ReID train-side mechanism that was analytically killed but NEVER measured. Kill
reasoning was "same class as LS-MRT (frozen-adaptation), so no headroom" -- but LS-MRT reprojects
the POST-pool final feat (+0.028), while LATS uses PRE-pool spatial stripe info that global-
average-pooling throws away. That extrapolation is exactly the kind of "total-account" overconfidence
the OA-SD episode warned against, so MEASURE it.

Frozen backbone. Per lattice variant: global feat (out[0]) + 6 horizontal-stripe-pooled token feats
(out[1][-1] featmap). Train a light side-branch (per-stripe attention -> residual projection) so
z_k = norm(global_k + alpha * stripe_residual_k). Eval K=9 decision marginalization with z_k vs
uniform global marginalization.

PASS if h16 stripe-LATS marg >= uniform-global marg + 0.3 mAP AND K-variant cosine doesn't rise.
If ~0, frozen-adaptation is dead BY MEASUREMENT (not extrapolation)."""
import sys, os, numpy as np, argparse
from collections import defaultdict
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='log/market1501/exp359_abl_noLMloss/transformer_40.pth')
ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
ap.add_argument('--h', type=int, default=16)
ap.add_argument('--K', type=int, default=9)
ap.add_argument('--n_stripe', type=int, default=6)
ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--alpha_init', type=float, default=0.1)
ap.add_argument('--tau_c', type=float, default=0.1)
ap.add_argument('--P', type=int, default=16)
ap.add_argument('--Kins', type=int, default=4)
ap.add_argument('--train_cap', type=int, default=4000)
ap.add_argument('--cache_gallery', default='/tmp/g_lats.npz')
cli = ap.parse_args()
sys.argv = ['ks', '--ckpt', cli.ckpt, '--config', cli.config, '--data_root', 'data', '--K', str(cli.K)]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpb_lattice_killswitch as ks
import torch, torch.nn.functional as F
from datasets.bases import read_image
RNG = np.random.RandomState(42); H, K, NS = cli.h, cli.K, cli.n_stripe
ext = ks.FrozenExtractor(); REPO = ks._repo; DEV = 'cuda'


def items(s):
    return ks.list_split(os.path.join(REPO, 'data', 'market1501', s))


def feats_and_stripes(pil_list):
    """(global [N,D] L2, stripes [N,NS,D] per-stripe L2) from out[0] + out[1][-1] featmap."""
    torch = ext.torch; B = ks.cli.batch; gs, ss = [], []
    for s in range(0, len(pil_list), B):
        arr = np.stack([ks.pil_to_tensor_np(im) for im in pil_list[s:s + B]], 0)
        t = torch.from_numpy(arr).cuda(non_blocking=True)
        cam = torch.zeros(t.shape[0], dtype=torch.long, device=t.device)
        view = torch.zeros(t.shape[0], dtype=torch.long, device=t.device)
        with torch.no_grad():
            out = ext.model(t, cam_label=cam, view_label=view, pose_dict=None)
            g = F.normalize(out[0].float(), dim=1)
            fm = out[1][-1].float()                                     # [B,C,Hf,Wf]
            st = F.adaptive_avg_pool2d(fm, (NS, 1)).squeeze(-1).transpose(1, 2)  # [B,NS,C]
            st = F.normalize(st, dim=2)
        gs.append(g.cpu().numpy().astype(np.float32)); ss.append(st.cpu().numpy().astype(np.float32))
    return np.concatenate(gs, 0), np.concatenate(ss, 0)


def variant_feats(its, cap=0):
    if cap:
        its = its[:cap]
    pid = np.array([x[1] for x in its]); cam = np.array([x[2] for x in its])
    Gl, Sl = [], []
    for it in its:
        pils = ks.make_lattice_variants(ks._to_target_aspect(read_image(it[0])), H, K, RNG)
        g, s = feats_and_stripes(pils)
        Gl.append(g); Sl.append(s)
    return np.stack(Gl).astype(np.float32), np.stack(Sl).astype(np.float32), pid, cam  # [N,K,D],[N,K,NS,D]


class Sidecar(torch.nn.Module):
    def __init__(self, D, alpha):
        super().__init__()
        self.att = torch.nn.Linear(D, 1)
        self.proj = torch.nn.Linear(D, D)
        self.alpha = torch.nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, g, s):                                  # g [.,D], s [.,NS,D]
        w = torch.softmax(self.att(s).squeeze(-1), dim=-1)    # [.,NS]
        res = (w.unsqueeze(-1) * self.proj(s)).sum(-2)        # [.,D]
        return F.normalize(g + self.alpha * res, dim=-1)


def marg(z, zg, tau=0.1):  # logmeanexp_k sim(z[.,K,D], zg[.,D]) -> [Nq,Ng]
    sim = np.einsum('ikd,jd->ijk', z, zg)
    return tau * np.log(np.exp(sim / tau).mean(2) + 1e-12)


# ---- 1. TRAIN: cache (global,stripes) variant feats, fit Sidecar with set-retrieval SupCon ----
print(f"[LATS] extract TRAIN (h={H},K={K},stripes={NS}) ...", flush=True)
Gt, St, ytr, _ = variant_feats(items('bounding_box_train'), cap=cli.train_cap)
N, _, D = Gt.shape
Gt_t = torch.tensor(Gt, device=DEV); St_t = torch.tensor(St, device=DEV); yt = torch.tensor(ytr, device=DEV)
id2idx = defaultdict(list)
for i, y in enumerate(ytr):
    id2idx[int(y)].append(i)
ids = [y for y in id2idx if len(id2idx[y]) >= 2]
net = Sidecar(D, cli.alpha_init).to(DEV)
opt = torch.optim.Adam(net.parameters(), lr=cli.lr)
iters = max(1, N // (cli.P * cli.Kins))
for ep in range(cli.epochs):
    last = 0.0
    for _ in range(iters):
        bids = RNG.choice(ids, min(cli.P, len(ids)), replace=False); bidx = []
        for y in bids:
            pool = id2idx[int(y)]
            bidx.extend(RNG.choice(pool, cli.Kins, replace=len(pool) < cli.Kins))
        bidx = np.array(bidx)
        zb = net(Gt_t[bidx].reshape(-1, D), St_t[bidx].reshape(-1, NS, D)).reshape(len(bidx), K, D)  # [b,K,D]
        zg = zb[:, 0]                                          # canonical slot-0 gallery
        S = 0.1 * torch.logsumexp(torch.einsum('ikd,jd->ijk', zb, zg) / 0.1
                                  - float(np.log(K)), dim=2)   # [b,b] logmeanexp set score
        yb = yt[bidx]
        pos = (yb[:, None] == yb[None, :]).float(); pos.fill_diagonal_(0)
        logits = S / cli.tau_c - 1e9 * torch.eye(len(bidx), device=DEV)
        loss = (-(pos * torch.log_softmax(logits, 1)).sum(1) / pos.sum(1).clamp_min(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); last = loss.item()
print(f"[LATS] Sidecar fit {cli.epochs}ep, loss={last:.4f}, alpha={net.alpha.item():.3f}", flush=True)

# ---- 2. TEST: stripe-LATS z marg vs uniform global marg ----
print("[LATS] extract QUERY ...", flush=True)
Gq, Sq, q_pid, q_cam = variant_feats(items('query'))
gits = items('bounding_box_test')
g_pid = np.array([x[1] for x in gits]); g_cam = np.array([x[2] for x in gits])
print("[LATS] extract GALLERY (HR single) ...", flush=True)
Gg, Sg = feats_and_stripes([ks._to_target_aspect(read_image(x[0])) for x in gits])  # [Ng,D],[Ng,NS,D]
with torch.no_grad():
    zq = net(torch.tensor(Gq, device=DEV).reshape(-1, D),
             torch.tensor(Sq, device=DEV).reshape(-1, NS, D)).reshape(len(q_pid), K, D).cpu().numpy()
    zg = net(torch.tensor(Gg, device=DEV), torch.tensor(Sg, device=DEV)).cpu().numpy()
uq = Gq / (np.linalg.norm(Gq, axis=2, keepdims=True) + 1e-12)     # uniform: global only, no sidecar
ug = Gg / (np.linalg.norm(Gg, axis=1, keepdims=True) + 1e-12)
r_lats = ks.eval_map(-marg(zq, zg), q_pid, q_cam, g_pid, g_cam)
r_unif = ks.eval_map(-marg(uq, ug), q_pid, q_cam, g_pid, g_cam)
cos_l = float(np.mean([(zq[i] @ zq[i].T)[np.triu_indices(K, 1)].mean() for i in range(len(q_pid))]))
cos_u = float(np.mean([(uq[i] @ uq[i].T)[np.triu_indices(K, 1)].mean() for i in range(len(q_pid))]))
d = r_lats['mAP'] - r_unif['mAP']
print(f"\n[LATS RESULT h={H}]  uniform-global marg={r_unif['mAP']:.3f}  stripe-LATS marg={r_lats['mAP']:.3f}  "
      f"(LATS-uniform={d:+.3f})")
print(f"[LATS DIAG] K-cos uniform={cos_u:.4f} LATS={cos_l:.4f} (rise={cos_l-cos_u:+.4f}); alpha={net.alpha.item():.3f}")
print(f"[LATS VERDICT] {'PASS (token stripe sidecar helps marginalization)' if d >= 0.3 and cos_l-cos_u <= 0.02 else 'FAIL (frozen-adaptation dead BY MEASUREMENT)'}")
print("[done] LATS probe complete.")
