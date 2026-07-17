#!/usr/bin/env python3
"""Hubness FAILURE-CASE characterization (ZERO-TRAINING).

Reuse the cached frozen exp255 global features (/tmp/hub_oduke_feats.npz) to:
  1. recompute H_k(g) = gallery NEGATIVE in-degree (same logic as the kill-switch),
     for k in {10,20}.
  2. pick top-N HUB gallery (highest H_k at k=10) + N camera-matched LOW-H_k controls.
  3. compute QUANTIFIABLE per-image stats (brightness, contrast, aspect ratio, box
     size, foreground/center-mass proxy via background-edge fraction, color hue, dark
     fraction, edge density as a blur proxy) for the selected images AND for the whole
     gallery (reference means).
  4. copy the selected images to a staging dir for retrieval + dump a JSON/CSV of stats.

NOTHING is trained. frozen features + numpy + PIL only.
"""
import os, sys, json, shutil, argparse
import numpy as np
from PIL import Image, ImageFilter, ImageStat

ap = argparse.ArgumentParser()
ap.add_argument('--cache_feat', default='/tmp/hub_oduke_feats.npz')
ap.add_argument('--gallery_dir', default='/root/work/SOLIDER-REID/data/occluded_duke/bounding_box_test')
ap.add_argument('--out_dir', default='/tmp/hub_chars')
ap.add_argument('--n', type=int, default=30, help='#hub and #control to select')
ap.add_argument('--k_main', type=int, default=10)
ap.add_argument('--k_aux', type=int, default=20)
ap.add_argument('--seed', type=int, default=42)
cli = ap.parse_args()
RNG = np.random.RandomState(cli.seed)


# ------------------------------------------------------------------ H_k core
def topk_per_query(sim, k):
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    order = np.argsort(-sim[rows, idx], axis=1)
    return idx[rows, order]


def compute_Hneg(sim, q_pid, g_pid, k):
    tk = topk_per_query(sim, k)
    H = np.zeros(sim.shape[1], dtype=np.int64)
    for col in range(k):
        gj = tk[:, col]
        sel = (g_pid[gj] != q_pid)
        np.add.at(H, gj[sel], 1)
    return H, tk


# ----------------------------------------------------- per-image statistics
def image_stats(path):
    """Return a dict of quantifiable stats for one image. Robust to failures."""
    try:
        im = Image.open(path).convert('RGB')
    except Exception as e:
        return None
    W, H = im.size
    arr = np.asarray(im, dtype=np.float32)            # (H,W,3)
    gray = arr.mean(axis=2)                            # (H,W)
    # basic photometric
    brightness = float(gray.mean())                    # 0-255
    contrast = float(gray.std())                       # global std (contrast proxy)
    dark_frac = float((gray < 50).mean())              # fraction of very dark pixels
    bright_frac = float((gray > 200).mean())
    aspect = float(H) / float(W) if W > 0 else 0.0     # taller=bigger (person crops ~2:1)
    box_area = int(W * H)                               # detection box size proxy
    # color: mean RGB + saturation (color-fulness)
    r, g, b = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
    mx = arr.max(axis=2); mn = arr.min(axis=2)
    sat = float(((mx - mn) / (mx + 1e-6)).mean())      # mean saturation 0-1
    colorfulness = float(np.sqrt((r - g) ** 2 + ((r + g) / 2 - b) ** 2))  # Hasler-Susstrunk-lite
    # edge density (blur proxy): higher = sharper / more texture
    edges = im.convert('L').filter(ImageFilter.FIND_EDGES)
    edge_arr = np.asarray(edges, dtype=np.float32)
    edge_density = float(edge_arr.mean())
    edge_strong_frac = float((edge_arr > 40).mean())
    # foreground/background proxy: person crops center the subject. Compare the
    # CENTER box (middle 50% w x 60% h, person torso/legs) vs the BORDER ring
    # (left/right 15% columns = usually background). If border is brighter/edgier than
    # center -> more background dominance. We report center-vs-border brightness ratio
    # and a "border edge fraction" (busy background = high).
    cw0, cw1 = int(W * 0.25), int(W * 0.75)
    ch0, ch1 = int(H * 0.20), int(H * 0.80)
    center = gray[ch0:ch1, cw0:cw1]
    bw = max(1, int(W * 0.15))
    border_cols = np.concatenate([gray[:, :bw], gray[:, W - bw:]], axis=1)
    center_mean = float(center.mean()) if center.size else brightness
    border_mean = float(border_cols.mean()) if border_cols.size else brightness
    # center-mass: where is the visual energy (edges)? fraction of total edge energy
    # that falls inside the center box. Low -> subject not centered / background-heavy.
    tot_edge = edge_arr.sum() + 1e-6
    center_edge = edge_arr[ch0:ch1, cw0:cw1].sum()
    center_edge_frac = float(center_edge / tot_edge)
    border_edge = np.concatenate([edge_arr[:, :bw], edge_arr[:, W - bw:]], axis=1).sum()
    border_edge_frac = float(border_edge / tot_edge)
    return dict(
        W=W, H=H, aspect=aspect, box_area=box_area,
        brightness=brightness, contrast=contrast,
        dark_frac=dark_frac, bright_frac=bright_frac,
        meanR=float(r), meanG=float(g), meanB=float(b),
        sat=sat, colorfulness=colorfulness,
        edge_density=edge_density, edge_strong_frac=edge_strong_frac,
        center_mean=center_mean, border_mean=border_mean,
        center_over_border_bright=float(center_mean / (border_mean + 1e-6)),
        center_edge_frac=center_edge_frac, border_edge_frac=border_edge_frac,
    )


def summarize(stats_list, keys):
    out = {}
    for kk in keys:
        vals = np.array([s[kk] for s in stats_list if s is not None and kk in s], float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[kk] = (float(vals.mean()), float(vals.std()), float(np.median(vals)))
    return out


def main():
    z = np.load(cli.cache_feat, allow_pickle=True)
    qf = z['q_feat'].astype(np.float32); gf = z['g_feat'].astype(np.float32)
    qf /= (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gf /= (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    q_pid, g_pid = z['q_pid'], z['g_pid']
    q_cam, g_cam = z['q_cam'], z['g_cam']
    g_name = z['g_name']
    Nq, Ng = qf.shape[0], gf.shape[0]
    print(f"[data] Nq={Nq} Ng={Ng} dim={qf.shape[1]}")

    sim = qf @ gf.T
    Hk, _ = compute_Hneg(sim, q_pid, g_pid, cli.k_main)
    Hk20, _ = compute_Hneg(sim, q_pid, g_pid, cli.k_aux)
    print(f"[Hk] k={cli.k_main}: max={Hk.max()} mean={Hk.mean():.3f} "
          f"#>0={int((Hk>0).sum())} ({100*(Hk>0).mean():.1f}%)")
    print(f"[Hk] k={cli.k_aux}: max={Hk20.max()} mean={Hk20.mean():.3f}")

    # ---- select TOP-N hubs (highest Hk at k_main; break ties by Hk20) ----
    rank_key = Hk.astype(np.float64) + 1e-6 * Hk20.astype(np.float64)
    order_h = np.argsort(-rank_key)
    hub_idx = order_h[:cli.n]
    hub_cams = g_cam[hub_idx]
    print(f"[hub] selected top-{cli.n}: H_k range {Hk[hub_idx].min()}..{Hk[hub_idx].max()}")
    # camera distribution of hubs (to match controls)
    cam_counts = {int(c): int((hub_cams == c).sum()) for c in np.unique(hub_cams)}
    print(f"[hub] camera distribution: {cam_counts}")

    # ---- select N camera-MATCHED LOW-H_k controls (H_k == 0) ----
    # pool of zero-hub gallery, sample to match hub camera histogram.
    zero_pool = np.where(Hk == 0)[0]
    print(f"[ctrl] zero-H_k pool size = {len(zero_pool)}")
    ctrl_idx = []
    for c, cnt in cam_counts.items():
        cand = zero_pool[g_cam[zero_pool] == c]
        if len(cand) >= cnt:
            pick = RNG.choice(cand, size=cnt, replace=False)
        else:
            pick = cand  # take all available
            # top up from global zero pool if a camera is short
        ctrl_idx.extend(pick.tolist())
    ctrl_idx = np.array(ctrl_idx, dtype=np.int64)
    # top up to N if short
    if len(ctrl_idx) < cli.n:
        remain = np.setdiff1d(zero_pool, ctrl_idx)
        extra = RNG.choice(remain, size=cli.n - len(ctrl_idx), replace=False)
        ctrl_idx = np.concatenate([ctrl_idx, extra])
    ctrl_idx = ctrl_idx[:cli.n]
    print(f"[ctrl] selected {len(ctrl_idx)} controls; "
          f"camera dist: {dict((int(c), int((g_cam[ctrl_idx]==c).sum())) for c in np.unique(g_cam[ctrl_idx]))}")

    # ---- copy images + compute stats ----
    os.makedirs(cli.out_dir, exist_ok=True)
    hub_dir = os.path.join(cli.out_dir, 'hub'); os.makedirs(hub_dir, exist_ok=True)
    ctrl_dir = os.path.join(cli.out_dir, 'ctrl'); os.makedirs(ctrl_dir, exist_ok=True)

    def process(indices, dst_dir, tag):
        stats = []
        meta = []
        for rank, gi in enumerate(indices):
            fn = str(g_name[gi])
            src = os.path.join(cli.gallery_dir, fn)
            # rank-prefixed name so the grid is ordered by H_k
            dst = os.path.join(dst_dir, f"{rank:02d}_H{int(Hk[gi])}_{fn}")
            try:
                shutil.copy(src, dst)
            except Exception as e:
                print(f"  [warn] copy failed {src}: {e}")
            s = image_stats(src)
            stats.append(s)
            meta.append(dict(rank=rank, gidx=int(gi), name=fn, cam=int(g_cam[gi]),
                             pid=int(g_pid[gi]), Hk=int(Hk[gi]), Hk20=int(Hk20[gi]),
                             stats=s))
        return stats, meta

    hub_stats, hub_meta = process(hub_idx, hub_dir, 'hub')
    ctrl_stats, ctrl_meta = process(ctrl_idx, ctrl_dir, 'ctrl')

    # ---- whole-gallery reference stats (sample for speed) ----
    samp = RNG.choice(Ng, size=min(800, Ng), replace=False)
    ref_stats = []
    for gi in samp:
        s = image_stats(os.path.join(cli.gallery_dir, str(g_name[gi])))
        if s is not None:
            ref_stats.append(s)

    keys = ['aspect', 'box_area', 'W', 'H', 'brightness', 'contrast', 'dark_frac',
            'bright_frac', 'sat', 'colorfulness', 'edge_density', 'edge_strong_frac',
            'center_over_border_bright', 'center_edge_frac', 'border_edge_frac',
            'meanR', 'meanG', 'meanB']
    hub_sum = summarize(hub_stats, keys)
    ctrl_sum = summarize(ctrl_stats, keys)
    ref_sum = summarize(ref_stats, keys)

    print("\n" + "=" * 96)
    print(f"{'stat':<28}{'HUB mean(std)':>22}{'CTRL mean(std)':>22}{'GALLERY-ref mean':>20}")
    print("=" * 96)
    for kk in keys:
        h = hub_sum.get(kk); c = ctrl_sum.get(kk); r = ref_sum.get(kk)
        hs = f"{h[0]:.3f}({h[1]:.3f})" if h else "-"
        cs = f"{c[0]:.3f}({c[1]:.3f})" if c else "-"
        rs = f"{r[0]:.3f}" if r else "-"
        # simple effect-size flag: |hub-ctrl| / pooled std
        flag = ''
        if h and c:
            pooled = (h[1] + c[1]) / 2 + 1e-6
            d = abs(h[0] - c[0]) / pooled
            if d >= 0.8: flag = '  <== LARGE diff'
            elif d >= 0.5: flag = '  <- moderate'
        print(f"{kk:<28}{hs:>22}{cs:>22}{rs:>20}{flag}")

    # dump everything
    dump = dict(
        hub_meta=hub_meta, ctrl_meta=ctrl_meta,
        hub_summary=hub_sum, ctrl_summary=ctrl_sum, ref_summary=ref_sum,
        hub_cam_counts=cam_counts,
        Hk_max=int(Hk.max()), Ng=int(Ng), Nq=int(Nq),
    )
    with open(os.path.join(cli.out_dir, 'characterize.json'), 'w') as f:
        json.dump(dump, f, indent=2, default=float)
    print(f"\n[done] images in {hub_dir} and {ctrl_dir}; stats in "
          f"{os.path.join(cli.out_dir, 'characterize.json')}")
    print(f"[hub names] {[str(g_name[i]) for i in hub_idx]}")


if __name__ == '__main__':
    main()
