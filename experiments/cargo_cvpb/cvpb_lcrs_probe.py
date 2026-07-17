#!/usr/bin/env python3
"""LCRS probe — Lattice-Complementary Residual Subspaces, frozen backbone (cheap).

用户直觉（2026-06-27 二次质疑后挖出）: 训练时多糊法 **不压成一样**（invariance-collapse 即
consistency 已实测 −1.73 死），而是让每个糊法 variant 都识别身份 + 残差子空间 **互补**
（decorrelation, "准确后分工"非硬推开）。codex 当时给 7/10 但停在"排队"没跑（codex 代码审查
2026-06-27 确认 LCRS 没训过, "训练端穷尽"结论不可信）。

  z_k = norm( P_shared(g_k) + alpha * P_axis[k](g_k) )
    P_shared : 共有身份证据（D->D linear, init eye）
    P_axis   : 每个 lattice axis（phase/bbox/kernel）一个残差子空间 head（D->D, init 0）
  身份 loss: per-variant set-retrieval SupCon（q-g 相似度真负 gallery, 避 L_marg train-ID
             classifier posterior 塌缩死因）— 每个 z_k 单独都要能检索身份。
  decorr   : 不同 axis 残差 P_axis(g) 互相 decorrelate（不重复信息）, 只在该样本 set-score 正确后启用。
  test     : K=9 z_k logmeanexp marginalization（同 test-time, 拿更富 union）。

PASS: K=9 marg gain >= +0.5 over uniform-lattice-marg(no-P)  AND  individual variant mAP 不掉 >0.8
      AND K-cosine 不升(不塌缩).   DEAD: 撞 frozen-已最优墙(像 LS-MRT +0.028) / variant 掉太多.
frozen backbone + cached K=9 feats => cheap, 复用 cvpb_lattice_killswitch / lsmrt framework.

Run on lab-3090-d:
  cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 \
    /root/miniconda3/envs/solider-reid/bin/python experiments/cargo_cvpb/cvpb_lcrs_probe.py \
    --ckpt log/market1501/exp359_abl_noLMloss/transformer_40.pth --h 16 --K 9 2>&1 | tee /tmp/cvpb_lcrs.log
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
ap.add_argument('--tau_l', type=float, default=0.1)        # lattice marginalization temp
ap.add_argument('--tau_c', type=float, default=0.1)        # contrastive temp
ap.add_argument('--P', type=int, default=16)               # ids per batch
ap.add_argument('--Kins', type=int, default=4)             # instances per id
ap.add_argument('--alpha', type=float, default=0.3)        # residual scale (small, residual not dominate)
ap.add_argument('--lambda_dec', type=float, default=0.1)   # decorrelation weight
ap.add_argument('--lambda_id', type=float, default=0.1)    # ||P_shared - I||^2 keep near identity
ap.add_argument('--n_axis', type=int, default=3)           # phase/bbox/kernel residual subspaces
ap.add_argument('--train_cap', type=int, default=0)
ap.add_argument('--cache_gallery', default='/tmp/g_lcrs.npz')
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
    out = []                                                 # (codex bug4 fix) streaming 分批提, 避一次 materialize 全 K 张 PIL → OOM
    for b0 in range(0, len(its), batch):
        chunk = its[b0:b0 + batch]
        flat = []
        for it in chunk:
            hr = ks._to_target_aspect(read_image(it[0]))
            flat.extend(ks.make_lattice_variants(hr, H, K, RNG))
        out.append(ext.feats_from_pil(flat).reshape(len(chunk), K, -1).astype(np.float32))
    return np.concatenate(out, 0), pids, cams


class LCRS(torch.nn.Module):
    """z_k = norm(P_shared(g_k) + alpha * P_axis[k%n_axis](g_k))."""
    def __init__(self, D, n_axis, alpha):
        super().__init__()
        self.shared = torch.nn.Linear(D, D, bias=False)
        self.shared.weight.data.copy_(torch.eye(D))
        self.axis = torch.nn.ModuleList([torch.nn.Linear(D, D, bias=False) for _ in range(n_axis)])
        for a in self.axis:
            a.weight.data.zero_()                              # residual init 0 (start = shared only)
        self.alpha, self.n_axis = alpha, n_axis

    def forward(self, g):                                      # g [..., K, D]
        Kk = g.shape[-2]
        sh = self.shared(g)                                    # [...,K,D]
        res = torch.stack([self.axis[k % self.n_axis](g[..., k, :]) for k in range(Kk)], dim=-2)
        return F.normalize(sh + self.alpha * res, dim=-1), res  # z [...,K,D], residual [...,K,D]


def main():
    print(f"[LCRS] extract TRAIN variant feats (h={H}, K={K}) ...", flush=True)
    ft, ytr, _ = variant_feats(items('bounding_box_train'), cap=cli.train_cap)
    N, _, D = ft.shape
    ft_t = torch.tensor(ft, device=DEV); yt = torch.tensor(ytr, device=DEV)
    id2idx = defaultdict(list)
    for i, y in enumerate(ytr):
        id2idx[int(y)].append(i)
    ids = [y for y in id2idx if len(id2idx[y]) >= 2]

    net = LCRS(D, cli.n_axis, cli.alpha).to(DEV)
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
            z, res = net(ft_t[bidx])                            # z [b,K,D], res [b,K,D]
            zg = z[:, 0]                                         # canonical-0 in-batch gallery
            yb = yt[bidx]
            pos = (yb[:, None] == yb[None, :]).float(); pos.fill_diagonal_(0)
            npos = pos.sum(1).clamp_min(1.0)
            # (codex bug1 fix) per-variant SupCon: 每个 z_k 单独检索 zg, 强制每 variant 都判别（非 logmeanexp 后）
            l_id = 0.0
            for k in range(K):
                simk = (z[:, k] @ zg.t()) / cli.tau_c           # [b,b]
                simk.fill_diagonal_(-1e9)                       # 不匹配 self canonical
                logpk = F.log_softmax(simk, dim=1)
                l_id = l_id - ((pos * logpk).sum(1) / npos)
            l_id = (l_id / K).mean()
            # (codex bug2 fix) K=9 marg top1 同 pid 才算 correct（去对角线 self-match）
            with torch.no_grad():
                sim = torch.einsum('ikd,jd->ijk', z, zg) / cli.tau_l
                S = cli.tau_l * torch.logsumexp(sim - float(np.log(K)), dim=2)
                S.fill_diagonal_(-1e9)
                correct = (yb[S.argmax(1)] == yb) | (pos.sum(1) == 0)
            # (codex bug3 fix) axis-level decorr: 同样本不同 axis 残差互补（非全局 Gram 含身份/样本），只 correct
            l_dec = 0.0; npair = 0
            for a in range(cli.n_axis):
                for b2 in range(a + 1, cli.n_axis):
                    ra = F.normalize(res[:, a], dim=1); rb = F.normalize(res[:, b2], dim=1)
                    cab = (ra * rb).sum(1)                       # [b] 同样本 axis a vs b cos
                    l_dec = l_dec + (cab[correct].pow(2).mean() if correct.any() else (cab.pow(2).mean() * 0.0))
                    npair += 1
            l_dec = l_dec / max(1, npair)
            l_reg = (net.shared.weight - eyeD).pow(2).mean()
            loss = l_id + cli.lambda_dec * l_dec + cli.lambda_id * l_reg
            opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss.item())
        if ep % 5 == 0 or ep == cli.epochs - 1:
            print(f"  ep{ep} loss={last:.4f}", flush=True)

    # ---- EVAL: K=9 marginalization with LCRS heads vs uniform(no-P) ----
    print("[LCRS] extract QUERY/GALLERY variant feats ...", flush=True)
    qf, yq, cq = variant_feats(items('query'))
    gf, yg, cg = variant_feats(items('bounding_box_test'))
    with torch.no_grad():
        zq, _ = net(torch.tensor(qf, device=DEV)); zq = zq.cpu().numpy()
        zg_all, _ = net(torch.tensor(gf, device=DEV)); zg = zg_all[:, 0].cpu().numpy()  # gallery canonical-0
        gf0 = F.normalize(torch.tensor(gf[:, 0], device=DEV), dim=-1).cpu().numpy()      # no-P gallery-0

    def setscore(zq_, zg_):
        sim = np.einsum('ikd,jd->ijk', zq_, zg_)
        return cli.tau_l * np.log(np.exp(sim / cli.tau_l).mean(2) + 1e-12)

    r_P = ks.eval_map(-setscore(zq, zg), yq, cq, yg, cg)
    r_u = ks.eval_map(-setscore(qf / (np.linalg.norm(qf, axis=-1, keepdims=True) + 1e-9), gf0),
                      yq, cq, yg, cg)
    mAP_P, r1_P = r_P['mAP'], r_P.get('R1', r_P.get('rank1', 0.0))
    mAP_u, r1_u = r_u['mAP'], r_u.get('R1', r_u.get('rank1', 0.0))
    kcos_u = float(np.mean([np.mean(np.einsum('kd,ld->kl',
                  qf[i] / (np.linalg.norm(qf[i], axis=-1, keepdims=True) + 1e-9),
                  qf[i] / (np.linalg.norm(qf[i], axis=-1, keepdims=True) + 1e-9))) for i in range(min(500, len(qf)))]))
    kcos_P = float(np.mean([np.mean(zq[i] @ zq[i].T) for i in range(min(500, len(zq)))]))
    print(f"\n[LCRS RESULT] h={H} K={K}")
    print(f"  uniform-lattice-marg (no-P): mAP={mAP_u:.3f} R1={r1_u:.3f}  K-cos={kcos_u:.4f}")
    print(f"  LCRS heads marg            : mAP={mAP_P:.3f} R1={r1_P:.3f}  K-cos={kcos_P:.4f}")
    print(f"  gain = {mAP_P - mAP_u:+.3f} mAP  (PASS >= +0.5; K-cos must NOT rise: {kcos_P:.4f} vs {kcos_u:.4f})")
    verdict = 'PASS' if (mAP_P - mAP_u >= 0.5 and kcos_P <= kcos_u + 0.01) else 'DEAD'
    print(f"  [verdict] {verdict}")
    print("[done]")


if __name__ == '__main__':
    main()
