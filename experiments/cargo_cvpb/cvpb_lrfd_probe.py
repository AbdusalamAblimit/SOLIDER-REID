#!/usr/bin/env python3
"""LRFD probe — Lattice-Residual Feature Disentangle, frozen backbone (cheap).

机理（区别 LCRS）: LCRS 塑造变体（残差互补）→ K-cos 升塌缩 DEAD −4.964。LRFD **不塑造变体**,
而是 disentangle: z_id 纯身份子空间（推理用） + r_lat 吸 lattice nuisance（必须能预测 lattice
axis, 推理丢）。"分离 lattice nuisance" 非 "塑造变体"。codex train_time_pipeline 7/10 但排队没跑,
codex 代码审查 2026-06-27 确认 LRFD 是剩下唯一值得 cheap measure（不 extrapolate）的。

  z_id  = norm(P_id(g_k))         推理用, per-variant set-retrieval SupCon 身份
  r_lat = P_lat(g_k)              lattice nuisance sink, 预测 lattice axis(CE), 推理丢
  orth  : z_id ⊥ r_lat            (disentangle, 身份子空间不含 lattice 信息)
  test  : z_id K=9 logmeanexp marginalization(丢 r_lat) vs uniform(no-P)

PASS: z_id K=9 gain >= +0.5 over uniform  AND  r_lat lattice-axis pred acc > 60%（nuisance 真学到
      lattice, 否则 disentangle 没发生）  AND  K-cos 不升.
DEAD: gain≈0（frozen 墙: z_id 去 lattice ≈ uniform marg 也去 lattice noise, 无 headroom）/ 塌缩.
复用 cvpb_lattice_killswitch / cvpb_lcrs framework.

Run on lab-3090-d:
  cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 \
    /root/miniconda3/envs/solider-reid/bin/python experiments/cargo_cvpb/cvpb_lrfd_probe.py \
    --ckpt log/market1501/exp359_abl_noLMloss/transformer_40.pth --h 16 --K 9 2>&1 | tee /tmp/cvpb_lrfd.log
"""
import sys, os, numpy as np, argparse
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='log/market1501/exp359_abl_noLMloss/transformer_40.pth')
ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
ap.add_argument('--h', type=int, default=16)
ap.add_argument('--K', type=int, default=9)
ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--tau_l', type=float, default=0.1)
ap.add_argument('--tau_c', type=float, default=0.1)
ap.add_argument('--P', type=int, default=16)               # ids per batch
ap.add_argument('--Kins', type=int, default=4)             # instances per id
ap.add_argument('--lambda_lat', type=float, default=0.3)   # lattice-axis CE weight (nuisance sink 学 lattice)
ap.add_argument('--lambda_orth', type=float, default=0.1)  # z_id ⊥ r_lat disentangle weight
ap.add_argument('--lambda_id', type=float, default=0.1)    # ||P_id - I||^2 keep near identity
ap.add_argument('--n_axis', type=int, default=3)           # phase/bbox/kernel lattice axes
ap.add_argument('--train_cap', type=int, default=0)
ap.add_argument('--cache_gallery', default='/tmp/g_lrfd.npz')
cli = ap.parse_args()
sys.argv = ['ks', '--ckpt', cli.ckpt, '--config', cli.config, '--data_root', 'data',
            '--K', str(cli.K), '--reuse_gallery', '--cache_gallery', cli.cache_gallery]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpb_lattice_killswitch as ks
import torch, torch.nn.functional as F
from datasets.bases import read_image

RNG = np.random.RandomState(42); H, K = cli.h, cli.K
ext = ks.FrozenExtractor(); REPO = ks._repo
DEV = 'cuda'


def items(s):
    return ks.list_split(os.path.join(REPO, 'data', 'market1501', s))


def variant_feats(its, cap=0, batch=256):
    if cap:
        its = its[:cap]
    pids = np.array([it[1] for it in its]); cams = np.array([it[2] for it in its])
    out = []                                                 # streaming 分批提, 避 OOM (LCRS bug4 教训)
    for b0 in range(0, len(its), batch):
        chunk = its[b0:b0 + batch]
        flat = []
        for it in chunk:
            hr = ks._to_target_aspect(read_image(it[0]))
            flat.extend(ks.make_lattice_variants(hr, H, K, RNG))
        out.append(ext.feats_from_pil(flat).reshape(len(chunk), K, -1).astype(np.float32))
    return np.concatenate(out, 0), pids, cams


class LRFD(torch.nn.Module):
    """disentangle: z_id=norm(P_id(g)), r_lat=P_lat(g)；lat_cls 让 r_lat 预测 lattice axis。"""
    def __init__(self, D, n_axis):
        super().__init__()
        self.id_head = torch.nn.Linear(D, D, bias=False)
        self.id_head.weight.data.copy_(torch.eye(D))         # init eye (start = frozen feat)
        self.lat_head = torch.nn.Linear(D, D, bias=False)
        self.lat_head.weight.data.zero_()                    # init 0 (no nuisance at start)
        self.lat_cls = torch.nn.Linear(D, n_axis)            # r_lat -> which lattice axis
        self.n_axis = n_axis

    def forward(self, g):                                    # g [...,K,D]
        z_id = F.normalize(self.id_head(g), dim=-1)
        r_lat = self.lat_head(g)
        return z_id, r_lat


def main():
    print(f"[LRFD] extract TRAIN variant feats (h={H}, K={K}) ...", flush=True)
    ft, ytr, _ = variant_feats(items('bounding_box_train'), cap=cli.train_cap)
    N, _, D = ft.shape
    ft_t = torch.tensor(ft, device=DEV); yt = torch.tensor(ytr, device=DEV)
    id2idx = defaultdict(list)
    for i, y in enumerate(ytr):
        id2idx[int(y)].append(i)
    ids = [y for y in id2idx if len(id2idx[y]) >= 2]
    axis_lab = torch.tensor([k % cli.n_axis for k in range(K)], device=DEV)  # [K] each variant's axis

    net = LRFD(D, cli.n_axis).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=cli.lr)
    eyeD = torch.eye(D, device=DEV)
    iters = max(1, N // (cli.P * cli.Kins))
    for ep in range(cli.epochs):
        last = 0.0
        for _ in range(iters):
            bids = RNG.choice(ids, min(cli.P, len(ids)), replace=False)
            bidx = []
            for y in bids:
                pool = id2idx[int(y)]
                bidx.extend(RNG.choice(pool, cli.Kins, replace=len(pool) < cli.Kins))
            bidx = np.array(bidx)
            z_id, r_lat = net(ft_t[bidx])                    # [b,K,D] each
            zg = z_id[:, 0]                                   # canonical-0 in-batch gallery
            yb = yt[bidx]
            pos = (yb[:, None] == yb[None, :]).float(); pos.fill_diagonal_(0)
            npos = pos.sum(1).clamp_min(1.0)
            # per-variant SupCon on z_id (身份判别, 避 classifier 塌缩)
            l_id = 0.0
            for k in range(K):
                simk = (z_id[:, k] @ zg.t()) / cli.tau_c
                simk.fill_diagonal_(-1e9)
                logpk = F.log_softmax(simk, dim=1)
                l_id = l_id - ((pos * logpk).sum(1) / npos)
            l_id = (l_id / K).mean()
            # r_lat 预测 lattice axis (nuisance sink 学 lattice, 否则 disentangle 没发生)
            lat_logits = net.lat_cls(r_lat)                  # [b,K,n_axis]
            l_lat = F.cross_entropy(lat_logits.reshape(-1, cli.n_axis),
                                    axis_lab.repeat(len(bidx)))
            # disentangle: z_id ⊥ r_lat (身份子空间不含 lattice)
            rl = F.normalize(r_lat, dim=-1)
            l_orth = (z_id * rl).sum(-1).pow(2).mean()
            l_reg = (net.id_head.weight - eyeD).pow(2).mean()
            loss = l_id + cli.lambda_lat * l_lat + cli.lambda_orth * l_orth + cli.lambda_id * l_reg
            opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss.item())
        if ep % 5 == 0 or ep == cli.epochs - 1:
            print(f"  ep{ep} loss={last:.4f}", flush=True)

    # ---- EVAL: z_id K=9 marginalization (丢 r_lat) vs uniform(no-P) ----
    print("[LRFD] extract QUERY/GALLERY variant feats ...", flush=True)
    qf, yq, cq = variant_feats(items('query'))
    gf, yg, cg = variant_feats(items('bounding_box_test'))
    with torch.no_grad():
        zq, _ = net(torch.tensor(qf, device=DEV)); zq = zq.cpu().numpy()
        zg_all, _ = net(torch.tensor(gf, device=DEV)); zg = zg_all[:, 0].cpu().numpy()  # gallery canonical-0
        gf0 = F.normalize(torch.tensor(gf[:, 0], device=DEV), dim=-1).cpu().numpy()      # no-P gallery-0
        # r_lat lattice-axis pred acc (nuisance 真学到 lattice 吗)
        _, rq = net(torch.tensor(qf[:min(1000, len(qf))], device=DEV))
        lat_pred = net.lat_cls(rq).argmax(-1).cpu().numpy()                              # [n,K]
        axis_gt = np.array([k % cli.n_axis for k in range(K)])[None, :].repeat(lat_pred.shape[0], 0)
        lat_acc = float((lat_pred == axis_gt).mean())

    def setscore(zq_, zg_):
        sim = np.einsum('ikd,jd->ijk', zq_, zg_)
        return cli.tau_l * np.log(np.exp(sim / cli.tau_l).mean(2) + 1e-12)

    r_P = ks.eval_map(-setscore(zq, zg), yq, cq, yg, cg)
    r_u = ks.eval_map(-setscore(qf / (np.linalg.norm(qf, axis=-1, keepdims=True) + 1e-9), gf0),
                      yq, cq, yg, cg)
    mAP_P = r_P['mAP']; mAP_u = r_u['mAP']
    kcos_u = float(np.mean([np.mean(np.einsum('kd,ld->kl',
                  qf[i] / (np.linalg.norm(qf[i], axis=-1, keepdims=True) + 1e-9),
                  qf[i] / (np.linalg.norm(qf[i], axis=-1, keepdims=True) + 1e-9))) for i in range(min(500, len(qf)))]))
    kcos_P = float(np.mean([np.mean(zq[i] @ zq[i].T) for i in range(min(500, len(zq)))]))
    print(f"\n[LRFD RESULT] h={H} K={K}")
    print(f"  uniform-lattice-marg (no-P): mAP={mAP_u:.3f}  K-cos={kcos_u:.4f}")
    print(f"  z_id (drop r_lat) marg     : mAP={mAP_P:.3f}  K-cos={kcos_P:.4f}")
    print(f"  r_lat lattice-axis pred acc: {lat_acc:.3f} (chance={1.0/cli.n_axis:.3f}; >0.6 = nuisance 真学到 lattice)")
    print(f"  gain = {mAP_P - mAP_u:+.3f} mAP  (PASS >= +0.5 AND lat_acc>0.6 AND K-cos not rise)")
    verdict = 'PASS' if (mAP_P - mAP_u >= 0.5 and lat_acc > 0.6 and kcos_P <= kcos_u + 0.01) else 'DEAD'
    print(f"  [verdict] {verdict}")
    print("[done]")


if __name__ == '__main__':
    main()
