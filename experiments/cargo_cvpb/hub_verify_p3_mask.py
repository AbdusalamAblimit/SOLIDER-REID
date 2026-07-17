#!/usr/bin/env python3
"""P3 mechanism causal test (occluded_duke, ZERO-TRAINING, frozen exp255).

Hypothesis (from HUBNESS_ANALYSIS sec 7.5): hubs are gallery crops where the model
over-encodes a NON-IDENTITY scene factor (bright orange car / brick plaza). If true,
removing the BACKGROUND (keep only the person via the pose heatmap) should DESTROY the
shared scene factor -> the hub's negative in-degree H_k should DROP. Control: removing
the PERSON (keep background) should NOT drop H_k (and may raise it) if the scene factor
is what attracts cross-id queries.

Method (frozen inference, no training, no backward):
  1. cached features -> H_k(g) -> pick top-N hub gallery + N camera-matched low-H_k ctrl.
  2. for each selected gallery image, build a PERSON mask at original image resolution
     from the pose .npz heatmap (sum 17 channels, placed on full-image canvas via the
     SAME _place_heatmap used in training; threshold + dilate). Then re-extract the
     frozen GLOBAL feature under 3 conditions:
        orig    : unmodified image
        bg_mask : keep person pixels, fill BACKGROUND with the dataset pixel-mean (gray)
        pp_mask : keep background, fill PERSON pixels with gray  (person-masked)
     pose_dict for the model forward is built identically to the eval dataloader
     (POSE_TEST_FEAT='global' -> part branches inert, returns the single global vector).
  3. Recompute H_k for ONLY the selected gallery items: replace their feature column in
     the full gallery feature matrix with the re-extracted (masked) feature, recompute
     each query's top-k over the FULL gallery, and count how many DIFFERENT-id queries
     still rank that item in top-k. Report mean H_k: orig vs bg_mask vs pp_mask, for the
     HUB set and the CONTROL set.

  Background-mask sanity: re-extracting with NO mask must reproduce the cached feature
  (cosine ~1.0). We print this so any preprocessing mismatch is caught.

Run: /root/miniconda3/envs/solider-reid/bin/python experiments/cargo_cvpb/hub_verify_p3_mask.py \
        --config configs/occluded_duke/pose_psg_lgpa_gcn512_2stage_small.yml \
        --ckpt   log/occluded_duke/exp255_small_gcn512_2stage/transformer_120.pth \
        --cache_feat /tmp/hub_oduke_feats.npz 2>&1 | tee /tmp/hub_p3_oduke.log
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, _repo)

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/occluded_duke/pose_psg_lgpa_gcn512_2stage_small.yml')
ap.add_argument('--ckpt', default='log/occluded_duke/exp255_small_gcn512_2stage/transformer_120.pth')
ap.add_argument('--cache_feat', default='/tmp/hub_oduke_feats.npz')
ap.add_argument('--gallery_dir', default='data/occluded_duke/bounding_box_test')
ap.add_argument('--pose_dir', default='data/occluded_duke/pose_data/gallery')
ap.add_argument('--n', type=int, default=30)
ap.add_argument('--k_main', type=int, default=10)
ap.add_argument('--mask_thr', type=float, default=0.10, help='person-mask heatmap threshold (frac of max)')
ap.add_argument('--dilate', type=int, default=6, help='dilation (px) of person mask at orig res')
ap.add_argument('--seed', type=int, default=42)
cli = ap.parse_args()
RNG = np.random.RandomState(cli.seed)


# --------------------------------------------------------- H_k from cached feats
def topk_per_query(sim, k):
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    return idx[rows, np.argsort(-sim[rows, idx], axis=1)]

def compute_Hneg(sim, q_pid, g_pid, k):
    tk = topk_per_query(sim, k)
    H = np.zeros(sim.shape[1], dtype=np.int64)
    for c in range(k):
        gj = tk[:, c]; sel = g_pid[gj] != q_pid; np.add.at(H, gj[sel], 1)
    return H


# --------------------------------------------------- person mask from pose npz
def _place_canvas(npz_path, img_h, img_w):
    """EXACT replica of PoseImageDataset._place_heatmap: project the (17,64,48) bbox-local
    heatmap onto the full image canvas at (img_h,img_w). Returns (17,img_h,img_w)."""
    with np.load(npz_path) as d:
        hm = torch.from_numpy(d['heatmap'].astype(np.float32))      # (17,64,48)
        crop_bounds = d['crop_bounds'].astype(np.float32)
    cx1, cy1, cx2, cy2 = crop_bounds
    crop_w_int = max(int(round(cx2 - cx1)), 1); crop_h_int = max(int(round(cy2 - cy1)), 1)
    hm_r = F.interpolate(hm.unsqueeze(0), size=(crop_h_int, crop_w_int),
                         mode='bilinear', align_corners=False).squeeze(0)
    src_x1 = max(0, int(round(-cx1))); src_y1 = max(0, int(round(-cy1)))
    dst_x1 = max(0, int(round(cx1))); dst_y1 = max(0, int(round(cy1)))
    dst_x2 = min(img_w, int(round(cx2))); dst_y2 = min(img_h, int(round(cy2)))
    canvas = torch.zeros(17, img_h, img_w)
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return canvas
    copy_w = min(dst_x2 - dst_x1, crop_w_int - src_x1)
    copy_h = min(dst_y2 - dst_y1, crop_h_int - src_y1)
    if copy_w <= 0 or copy_h <= 0:
        return canvas
    canvas[:, dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] = \
        hm_r[:, src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w]
    return canvas

def place_heatmap_sum(npz_path, img_h, img_w):
    """Person occupancy map = sum of 17 placed heatmap channels -> (img_h,img_w)."""
    return _place_canvas(npz_path, img_h, img_w).sum(0).numpy()


def dilate_mask_np(mask_bool, px):
    """Pure-numpy square dilation (no scipy dependency) via sliding-window max."""
    if px <= 0:
        return mask_bool
    from numpy.lib.stride_tricks import sliding_window_view
    k = 2 * px + 1
    pad = np.pad(mask_bool.astype(np.float32), px, mode='constant')
    win = sliding_window_view(pad, (k, k))     # (H,W,k,k)
    return win.max(axis=(-1, -2)) > 0.5


def build_person_mask(npz_path, img_h, img_w):
    occ = place_heatmap_sum(npz_path, img_h, img_w)
    mx = occ.max()
    if mx <= 0:
        return None
    m = occ >= (cli.mask_thr * mx)
    m = dilate_mask_np(m, cli.dilate)
    return m


# --------------------------------------------------- model + masked extraction
def main():
    from config import cfg
    from datasets import make_dataloader
    from model import make_model
    t0 = time.time()

    cfg.merge_from_file(os.path.join(_repo, cli.config))
    cfg.merge_from_list([
        'TEST.WEIGHT', os.path.join(_repo, cli.ckpt),
        'MODEL.POSE_TEST_FEAT', 'global',
        'TEST.NECK_FEAT', 'after',
        'TEST.FEAT_NORM', 'yes',
        'TEST.IMS_PER_BATCH', 64,
    ])
    cfg.freeze()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

    # ---- cached feats -> H_k -> select hub + ctrl ----
    z = np.load(cli.cache_feat, allow_pickle=True)
    qf = z['q_feat'].astype(np.float32); gf = z['g_feat'].astype(np.float32)
    q_pid, g_pid = z['q_pid'], z['g_pid']; q_cam, g_cam = z['q_cam'], z['g_cam']
    g_name = z['g_name']
    qf /= (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gf /= (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    Nq, Ng = qf.shape[0], gf.shape[0]; km = cli.k_main
    sim = qf @ gf.T
    Hk = compute_Hneg(sim, q_pid, g_pid, km)
    order_h = np.argsort(-Hk)
    hub_idx = order_h[:cli.n]
    hub_cams = g_cam[hub_idx]
    cam_counts = {int(c): int((hub_cams == c).sum()) for c in np.unique(hub_cams)}
    print(f"[data] Nq={Nq} Ng={Ng} dim={qf.shape[1]}; hub H_k range "
          f"{Hk[hub_idx].min()}..{Hk[hub_idx].max()}; hub cam dist {cam_counts}", flush=True)
    zero_pool = np.where(Hk == 0)[0]
    ctrl_idx = []
    for c, cnt in cam_counts.items():
        cand = zero_pool[g_cam[zero_pool] == c]
        pick = RNG.choice(cand, size=min(cnt, len(cand)), replace=False)
        ctrl_idx.extend(pick.tolist())
    ctrl_idx = np.array(ctrl_idx[:cli.n], dtype=np.int64)
    print(f"[ctrl] {len(ctrl_idx)} camera-matched zero-H_k controls", flush=True)

    sel_idx = np.concatenate([hub_idx, ctrl_idx])

    # ---- build model ----
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = \
        make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(cfg.TEST.WEIGHT)
    model = model.cuda().eval()
    print(f"[model] loaded {cli.ckpt}", flush=True)

    H_img, W_img = cfg.INPUT.SIZE_TEST           # (384,128)
    hm_h, hm_w = cfg.MODEL.POSE_HEATMAP_SIZE      # (96,32)
    mean = torch.tensor(cfg.INPUT.PIXEL_MEAN).view(3, 1, 1)
    std = torch.tensor(cfg.INPUT.PIXEL_STD).view(3, 1, 1)
    fill_val = np.array([m * 255.0 for m in cfg.INPUT.PIXEL_MEAN], dtype=np.float32)  # gray fill in 0-255

    # pose index for gallery (to locate the target npz for each gallery image)
    pose_index_path = os.path.join(_repo, cli.pose_dir, 'index.json')
    pose_index = {}
    if os.path.exists(pose_index_path):
        with open(pose_index_path) as f:
            pose_index = json.load(f)

    def npz_for(fname):
        """Find the target-person npz for a gallery image file name."""
        entry = pose_index.get(fname)
        if entry is not None and entry.get('persons'):
            ti = entry.get('target_person_idx', 0)
            npz_name = entry['persons'][ti] if ti < len(entry['persons']) else entry['persons'][0]
            p = npz_name if os.path.isabs(npz_name) else os.path.join(_repo, cli.pose_dir, npz_name)
            if os.path.exists(p):
                return p
        # fallback: guess by stripping .jpg + _p0.npz
        base = os.path.splitext(fname)[0]
        cand = os.path.join(_repo, cli.pose_dir, base + '_p0.npz')
        return cand if os.path.exists(cand) else None

    def load_pose_dict(npz_path):
        """Build the eval pose_dict (single person, downsampled heatmap to hm_h x hm_w)."""
        max_p = 6
        out_hm = torch.zeros(max_p, 17, hm_h, hm_w)
        out_kp = torch.zeros(max_p, 17, 2); out_sc = torch.zeros(max_p, 17)
        out_vis = torch.zeros(max_p, 17); out_vb = torch.zeros(max_p, 17)
        out_mask = torch.zeros(max_p)
        if npz_path is not None and os.path.exists(npz_path):
            with np.load(npz_path) as d:
                hm = torch.from_numpy(d['heatmap'].astype(np.float32))   # (17,64,48)
                # the dataloader places heatmap on the full image then resizes to (target),
                # then downsamples to (hm_h,hm_w). For the scene heatmap used by PSG the exact
                # spatial placement matters only mildly; we replicate place->resize for fidelity.
            # place on full-res target then to hm size
            occ_full = []
            # reuse place: but we need per-channel placed canvas at (H_img,W_img)
            placed = _place_to(npz_path, H_img, W_img)        # (17,H_img,W_img)
            hm_ds = F.interpolate(placed.unsqueeze(0), size=(hm_h, hm_w),
                                  mode='bilinear', align_corners=False).squeeze(0)
            out_hm[0] = hm_ds
            with np.load(npz_path) as d:
                out_kp[0] = torch.from_numpy(d['keypoints'].astype(np.float32))
                out_sc[0] = torch.from_numpy(d['scores'].astype(np.float32))
                if 'visibility' in d.files:
                    out_vis[0] = torch.from_numpy(d['visibility'].astype(np.float32))
                if 'visibility_binary' in d.files:
                    out_vb[0] = torch.from_numpy(d['visibility_binary'].astype(np.float32))
            out_mask[0] = 1.0
            np_ = 1
        else:
            np_ = 0
        pd = {'heatmaps': out_hm.unsqueeze(0).cuda(),
              'keypoints': out_kp.unsqueeze(0).cuda(),
              'scores': out_sc.unsqueeze(0).cuda(),
              'visibility': out_vis.unsqueeze(0).cuda(),
              'visibility_binary': out_vb.unsqueeze(0).cuda(),
              'person_mask': out_mask.unsqueeze(0).cuda(),
              'num_persons': np_}
        return pd

    def _place_to(npz_path, img_h, img_w):
        return _place_canvas(npz_path, img_h, img_w)

    def img_to_tensor(pil):
        pil = pil.resize((W_img, H_img), Image.BILINEAR)
        arr = torch.from_numpy(np.asarray(pil, np.float32) / 255.0).permute(2, 0, 1)
        return ((arr - mean) / std)

    @torch.no_grad()
    def extract(fname, mode):
        """mode in {'orig','bg_mask','pp_mask'}. Returns (D,) L2-normed numpy feat."""
        path = os.path.join(_repo, cli.gallery_dir, fname)
        pil = Image.open(path).convert('RGB')
        W0, H0 = pil.size
        npz_path = npz_for(fname)
        if mode != 'orig':
            arr = np.asarray(pil, np.float32).copy()       # (H0,W0,3) orig res
            pmask = build_person_mask(npz_path, H0, W0) if npz_path else None
            if pmask is None:
                # no pose -> cannot mask; return None to skip
                return None, npz_path
            if mode == 'bg_mask':
                arr[~pmask] = fill_val                       # keep person, gray background
            else:  # pp_mask
                arr[pmask] = fill_val                        # keep background, gray person
            pil = Image.fromarray(arr.astype(np.uint8))
        x = img_to_tensor(pil).unsqueeze(0).cuda()
        pd = load_pose_dict(npz_path)
        out = model(x, cam_label=torch.zeros(1, dtype=torch.long).cuda(),
                    view_label=torch.zeros(1, dtype=torch.long).cuda(), pose_dict=pd)
        feat = out[0] if isinstance(out, (tuple, list)) else out
        feat = F.normalize(feat, p=2, dim=1)
        return feat.squeeze(0).cpu().numpy().astype(np.float32), npz_path

    # ---- sanity: re-extract 'orig' for a few hub items, compare to cached ----
    print("\n[sanity] re-extracted 'orig' vs cached cosine (want ~1.0):", flush=True)
    n_ok = 0; cos_list = []
    for gi in hub_idx[:5]:
        f, _ = extract(str(g_name[gi]), 'orig')
        if f is not None:
            c = float(f @ gf[gi]); cos_list.append(c)
            print(f"    {str(g_name[gi]):28s} cos={c:.4f}", flush=True)
    if cos_list:
        print(f"    mean re-extract cosine = {np.mean(cos_list):.4f} "
              f"({'OK' if np.mean(cos_list) > 0.97 else 'MISMATCH -> preprocessing differs'})", flush=True)

    # ---- extract masked features for all selected items ----
    feats = {'orig': {}, 'bg_mask': {}, 'pp_mask': {}}
    skipped = []
    for gi in sel_idx:
        fname = str(g_name[gi])
        for mode in ('orig', 'bg_mask', 'pp_mask'):
            f, npzp = extract(fname, mode)
            if f is None:
                skipped.append((int(gi), mode))
            else:
                feats[mode][int(gi)] = f
    print(f"\n[extract] done ({time.time()-t0:.0f}s); skipped (no pose mask) = {len(skipped)}", flush=True)

    # ---- recompute H_k for selected items under each condition ----
    # For a given condition, replace ONLY the selected gallery columns' features and
    # recompute how many DIFFERENT-id queries rank each selected item in their top-k.
    # (All OTHER gallery features stay at their original cached value.)
    def recompute_Hk_for(items, cond):
        gf_mod = gf.copy()
        valid_items = []
        for gi in items:
            if int(gi) in feats[cond]:
                gf_mod[gi] = feats[cond][int(gi)]
                valid_items.append(int(gi))
        sim_mod = qf @ gf_mod.T                 # (Nq,Ng)
        tk = topk_per_query(sim_mod, km)        # (Nq,km)
        H = np.zeros(Ng, dtype=np.int64)
        for c in range(km):
            gj = tk[:, c]; sel = g_pid[gj] != q_pid; np.add.at(H, gj[sel], 1)
        return H, valid_items

    print("\n" + "=" * 84)
    print("P3 RESULT — mean gallery NEGATIVE in-degree H_k under image masking")
    print("=" * 84)
    for tag, items in [('HUB', hub_idx), ('CTRL', ctrl_idx)]:
        row = {}
        for cond in ('orig', 'bg_mask', 'pp_mask'):
            H, vi = recompute_Hk_for(items, cond)
            vals = np.array([H[gi] for gi in vi], float)
            row[cond] = (vals.mean() if len(vals) else float('nan'), len(vi))
        o = row['orig'][0]; b = row['bg_mask'][0]; p = row['pp_mask'][0]
        n = row['orig'][1]
        print(f"  {tag} (n={n}):  orig H_k = {o:6.2f}   "
              f"bg_masked(keep person) = {b:6.2f} ({100*(b-o)/max(o,1e-9):+.0f}%)   "
              f"person_masked(keep bg) = {p:6.2f} ({100*(p-o)/max(o,1e-9):+.0f}%)")
    print("\n  EXPECT if scene-factor mechanism is real:")
    print("    HUB bg_masked << HUB orig  (removing background DESTROYS the cross-id attraction)")
    print("    HUB person_masked >= HUB orig (keeping background preserves/raises the attraction)")
    print(f"\n[done] P3 in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
