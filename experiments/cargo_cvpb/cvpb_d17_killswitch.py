#!/usr/bin/env python3
"""d17-KS1: Rank-Leverage Concentration (RLC) diagnostic, controlled for #false-in-topk.

Observation (d17): ReID top-k ranking can be hijacked by a few local regions -- a single
stripe/patch carries most of the query-gallery similarity for the top-k.  RLC[q] measures how
CONCENTRATED that leverage is on one stripe.  Hypothesis: high RLC -> fragile ranking -> more
AP-error.  KILL-SWITCH (codex + session lesson): RLC must explain AP-error BEYOND the trivial
proxy #false-in-topk (Hubness/evidence both died here).  GO if partial corr(RLC, AP_err | #false)
>= +0.18; KILL if <= +0.10 or only nonzero where #false is high.

Reuses cvpb_lattice_killswitch's FrozenExtractor + eval helpers (HR ReID, no LR/lattice)."""
import sys, os, numpy as np, argparse
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth')
ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
ap.add_argument('--dataset', default='market1501')
ap.add_argument('--n_stripe', type=int, default=6, help='horizontal stripes to mask (leverage probe)')
ap.add_argument('--topk', type=int, default=10)
ap.add_argument('--batch', type=int, default=128)
ap.add_argument('--smoke', type=int, default=0)
cli = ap.parse_args()
sys.argv = ['ks', '--ckpt', cli.ckpt, '--config', cli.config, '--dataset', cli.dataset,
            '--batch', str(cli.batch)]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpb_lattice_killswitch as ks
from datasets.bases import read_image
from PIL import Image

ext = ks.FrozenExtractor(); REPO = ks._repo


def items(split_market, split_msmt=None):
    if cli.dataset == 'msmt17':
        return ks.msmt17_split(os.path.join(REPO, 'data'), split_msmt)
    return ks.list_split(os.path.join(REPO, 'data', cli.dataset, split_market))


def stripe_masked(pil, s, n):
    """Blacken horizontal stripe s of n on a copy of the target-aspect HR image."""
    im = ks._to_target_aspect(pil).copy()
    W, H = im.size
    y0, y1 = int(H * s / n), int(H * (s + 1) / n)
    im.paste((0, 0, 0), (0, y0, W, y1))
    return im


# ---- 1. full query + gallery feats ----
q_it = items('query', 'list_query.txt'); g_it = items('bounding_box_test', 'list_gallery.txt')
if cli.smoke:
    q_it = q_it[:cli.smoke]
q_pid = np.array([x[1] for x in q_it]); q_cam = np.array([x[2] for x in q_it])
g_pid = np.array([x[1] for x in g_it]); g_cam = np.array([x[2] for x in g_it])
print(f"[d17] q {len(q_it)} g {len(g_it)} | extracting full feats ...", flush=True)
qf = ext.feats_from_pil([ks._to_target_aspect(read_image(x[0])) for x in q_it])   # [Nq,D] L2
gf = ext.feats_from_pil([ks._to_target_aspect(read_image(x[0])) for x in g_it])   # [Ng,D] L2
dist = 1.0 - qf @ gf.T
r = ks.eval_map(dist, q_pid, q_cam, g_pid, g_cam)
print(f"[d17] baseline mAP={r['mAP']:.3f} R1={r['r1']:.3f}", flush=True)

# ---- 2. per-query AP-error + #false-in-topk (trivial proxy to control) ----
ap_err = 1.0 - ks.per_query_ap(dist, q_pid, q_cam, g_pid, g_cam)                  # [Nq]
nfalse = ks.n_false_in_topk(dist, q_pid, q_cam, g_pid, g_cam, k=cli.topk).astype(np.float32)

# ---- 3. RLC: per-query, mask each stripe -> drop in top-k similarity; concentration on one stripe ----
print(f"[d17] stripe-mask leverage ({cli.n_stripe} stripes) ...", flush=True)
idx = np.argsort(dist, axis=1)[:, :cli.topk]                                      # [Nq,topk] gallery idx
N = cli.n_stripe
lev = np.zeros((len(q_it), N), dtype=np.float32)
for s in range(N):
    qf_s = ext.feats_from_pil([stripe_masked(read_image(x[0]), s, N) for x in q_it])  # [Nq,D]
    for i in range(len(q_it)):
        full_sim = qf[i] @ gf[idx[i]].T                                          # [topk]
        mask_sim = qf_s[i] @ gf[idx[i]].T
        lev[i, s] = float(np.mean(full_sim - mask_sim))                          # avg top-k sim drop
    print(f"   stripe {s+1}/{N} done", flush=True)
lev = np.clip(lev, 0, None)                                                       # only positive leverage
rlc = lev.max(1) / (lev.sum(1) + 1e-8)                                            # [Nq] concentration in [1/N,1]

# ---- 4. KS1: does RLC explain AP-error BEYOND #false-in-topk? ----
sp_raw, _ = ks.spearman(rlc, ap_err)
sp_partial, _ = ks.partial_spearman(rlc, ap_err, nfalse[:, None])                # control #false
sp_false, _ = ks.spearman(nfalse, ap_err)
print(f"\n[d17-KS1 RESULT]")
print(f"   RLC range [{rlc.min():.3f},{rlc.max():.3f}] mean {rlc.mean():.3f} (uniform={1.0/N:.3f})")
print(f"   Spearman(RLC, AP_err)            = {sp_raw:+.3f}")
print(f"   Spearman(#false, AP_err)         = {sp_false:+.3f}  (trivial proxy)")
print(f"   partial(RLC, AP_err | #false)    = {sp_partial:+.3f}  <-- KILL-SWITCH")
verdict = 'GO (RLC explains failure beyond #false)' if sp_partial >= 0.18 else (
    'KILL (RLC absorbed by trivial #false)' if sp_partial <= 0.10 else 'WEAK (0.10-0.18, marginal)')
print(f"   VERDICT: {verdict}  [GO>=+0.18 / KILL<=+0.10]")
print("[done] d17-KS1 complete.")
