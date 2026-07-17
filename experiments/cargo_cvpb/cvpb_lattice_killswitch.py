#!/usr/bin/env python3
"""Lattice-Marginalized ReID — ZERO-TRAINING kill-switch  (Market, frozen SOLIDER).

HYPOTHESIS UNDER TEST (d_8.txt 机会1):
    Low-resolution ReID failure is NOT "blur / missing detail" but SAMPLING-LATTICE
    UNCERTAINTY. A single person at h=16-32px is a *family* of alias/crop-lattice
    observations (sub-pixel sampling phase, +/-1 LR-pixel bbox quantization, antialias
    kernel, slight detector crop error). The model has only ever seen ONE member of that
    family. Identity matching should MARGINALIZE over the family, not treat one
    deterministic LR image as ground truth.

ZERO-TRAINING (no backward, frozen ckpt + numpy/PIL).  Standard CR-ReID setting:
    HR gallery (enrolled), LR query (far/small).  Gallery features extracted ONCE at HR.

WHAT WE MEASURE per LR height h in {16,24,32,48}:
    (A) same-image phase feature variance: feed K lattice variants of the SAME hr query,
        measure mean pairwise (1-cos) of their frozen features.  (does the lattice move
        the embedding at all?)
    (B) rank volatility: top1 agreement + top10 Jaccard ACROSS the K lattice variants
        (do retrieved IDs flip between phases?).
    (C) does phase variance EXPLAIN LR false matches?  per-query Spearman(phase-var, AP-error)
        AND -- decisive, Hubness-§7.6 lesson -- partial-Spearman CONTROLLING the trivial
        proxy #false-in-topk (and LR severity).  If phase-var is just a proxy for
        "#wrong-in-topk", it has no independent value.
    (D) ensemble mAP: K-phase feature-mean / MaxSim vs a SINGLE deterministic bicubic LR.

  ******  THE LIFE/DEATH CONTROLS  ******
    (C1) vs ORDINARY TTA: the SAME K, the SAME fusion (mean / MaxSim), but the K views are
         ordinary test-time augmentation (pad+RandomCrop + hflip) of ONE bicubic LR image
         -- NOT lattice variants.  phase-lattice ensemble MUST clearly beat ordinary-TTA
         ensemble, else it is just TTA renamed.
    (C2) vs #false-in-topk: phase-variance explaining failure MUST survive partialling out
         #false-in-topk (Hubness lesson: a trivial proxy must not silently do the work).

VERDICT:
    GO   if  h<=32: rank volatility clearly nonzero  AND  phase-ensemble >= +2 mAP over single
         LR  AND that gain CLEARLY exceeds ordinary-TTA  AND phase-var explains failure
         INDEPENDENTLY of #false-in-topk.
    DEAD if  phase variance tiny  /  ensemble ~ single LR  /  ensemble ~ ordinary TTA  /
         phase-var absorbed by #false-in-topk.

Run on lab-3090-d:
    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 \
      /root/miniconda3/envs/solider-reid/bin/python \
      experiments/cargo_cvpb/cvpb_lattice_killswitch.py \
      --config configs/market/pose_psg_lgpa_gcn_base.yml \
      --ckpt   log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth \
      --K 9  2>&1 | tee /tmp/cvpb_lattice_market.log
    # smoke first:  --smoke 150 --heights 32   (fast)
"""
import os, sys, time, argparse
import numpy as np
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, _repo)

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
ap.add_argument('--ckpt', default='log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth')
ap.add_argument('--data_root', default='data')
ap.add_argument('--heights', type=int, nargs='+', default=[16, 24, 32, 48],
                help='LR person heights to test')
ap.add_argument('--K', type=int, default=9, help='#lattice (phase) variants per LR query')
ap.add_argument('--smoke', type=int, default=0, help='cap #query for a fast smoke run')
ap.add_argument('--batch', type=int, default=128)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--cache_gallery', default='/tmp/lattice_gallery_hr.npz')
ap.add_argument('--reuse_gallery', action='store_true')
ap.add_argument('--lattice_axis', type=int, default=-1, help='LM-S4: restrict lattice variants to ONE axis (0=phase,1=bbox,2=zoom); -1=all (round-robin)')
ap.add_argument('--strong_tta', action='store_true', help='LM-S2 defense: richer ordinary-TTA (resize-jitter+color) so lattice must beat a STRONG baseline')
ap.add_argument('--jitter_mode', default='lattice', choices=['lattice', 'detector'], help='push-7.0: lattice=uniform +-1 LR-px theoretical sampling lattice; detector=continuous Gaussian center+scale jitter calibrated to detector localization error (tests if marginalization holds under realistic detector bbox uncertainty, NOT just synthetic lattice). Market has no source frame so this is a literature-informed proxy, not real detector boxes.')
ap.add_argument('--jitter_sigma', type=float, default=0.5, help='detector jitter translation sigma in LR-px (scale sigma=0.2*this). smaller=closer to precise lattice. sweep to map marginalization gain vs detector-error magnitude.')
ap.add_argument('--dataset', default='market1501', choices=['market1501', 'msmt17'], help='cross-dataset push-7.0 kill-switch②: market1501 (dir split) or msmt17 (list-file split). MSMT17 needs its own ckpt+config (num_class differs).')
ap.add_argument('--semantic_weight', type=float, default=-1.0, help='override MODEL.SEMANTIC_WEIGHT to match ckpt training (MSMT17 swin ckpt trained sw=0.6 but config has 0.2 -> SOLIDER backbone feature mismatch). -1=use config.')
ap.add_argument('--adaptive_k', action='store_true', help='supporting: per-query phase-volatility selects K (high-vol query marginalize over K, low-vol use K=1). Reduces avg compute keeping most marginalization gain -> rebut "K=9 too expensive".')
cli = ap.parse_args()
np.random.seed(cli.seed)
RNG = np.random.RandomState(cli.seed)

SIZE_TEST = (384, 128)       # (H, W) the model input
PIXEL_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
PIXEL_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# PIL resample kernels for the "different antialias kernel" lattice axis
_KERNELS = {
    'bicubic': Image.BICUBIC,
    'bilinear': Image.BILINEAR,
    'lanczos': Image.LANCZOS,
    'box': Image.BOX,
    'hamming': Image.HAMMING,
    'nearest': Image.NEAREST,
}


# =========================================================================== #
# dataset list (parse Market dirs directly; no dataloader needed)
# =========================================================================== #
import re, glob
_PAT = re.compile(r'([-\d]+)_c(\d)')


def list_split(dir_path):
    items = []
    for p in sorted(glob.glob(os.path.join(dir_path, '*.jpg'))):
        pid, cam = map(int, _PAT.search(p).groups())
        if pid == -1:
            continue
        items.append((p, pid, cam - 1))
    return items


def msmt17_split(data_root, list_file):
    """MSMT17 list-file split: each line 'relpath pid'; cam parsed from filename
    (pid_seq_CAM_time_...), images under <data_root>/MSMT17/test/<relpath>."""
    items = []
    base = os.path.join(data_root, 'MSMT17')
    with open(os.path.join(base, list_file)) as f:
        for line in f:
            rel, pid = line.strip().split(' ')
            cam = int(os.path.basename(rel).split('_')[2]) - 1
            items.append((os.path.join(base, 'test', rel), int(pid), cam))
    return items


# =========================================================================== #
# LR + lattice variant generation  (all in PIL space, from the ORIGINAL image)
# =========================================================================== #
def _to_target_aspect(img):
    """Resize the original crop to the model's 384x128 (3:1) HR canvas with BICUBIC.
    This is the 'HR' reference everything is degraded from (gallery also uses this)."""
    return img.resize((SIZE_TEST[1], SIZE_TEST[0]), Image.BICUBIC)


def make_lr(hr_img, h, kernel='bicubic'):
    """Deterministic synthetic LR: HR(384x128) --down--> (h, w) --up--> 384x128.
    w preserves the 3:1 canvas aspect: w = round(h/3).  Returns a 384x128 PIL image
    (degrade-then-restore-size, the standard CR-ReID synthetic LR convention)."""
    w = max(1, int(round(h * SIZE_TEST[1] / SIZE_TEST[0])))   # h*128/384 = h/3
    k = _KERNELS[kernel]
    small = hr_img.resize((w, h), k)
    return small.resize((SIZE_TEST[1], SIZE_TEST[0]), Image.BICUBIC)


def make_lattice_variants(hr_img, h, K, rng, fixed_axis=None, jitter_mode='lattice'):
    """K plausible PHASE/CROP/KERNEL variants of the SAME hr image at height h.
    LM-S4 factor ablation: fixed_axis (0=phase,1=bbox,2=zoom) restricts ALL variants to ONE
    lattice axis, isolating which axis drives the test-time gain (cleanest story=phase).

    Each variant perturbs the SAMPLING LATTICE relative to the scene by a SUB-LR-pixel
    amount, then forms the LR image.  The depicted person is (almost) the same extent;
    only WHICH hr pixels land on each LR sample point changes.  Axes:
      - sub-pixel phase shift  (fractional HR translate before downsample)
      - +/-1 LR-pixel bbox crop shift / expand (integer LR-pixel = h/.. HR pixels)
      - antialias kernel choice

    variant 0 is ALWAYS the canonical deterministic bicubic LR (no perturbation) so the
    single-LR baseline == variants[0].
    Returns list of K PIL images (each 384x128)."""
    W_hr, H_hr = hr_img.size                      # 128, 384
    # how many HR pixels correspond to 1 LR pixel at this height
    hr_per_lr_y = H_hr / float(h)                  # 384/h
    hr_per_lr_x = W_hr / float(max(1, round(h / 3.0)))  # 128/(h/3) ~ 3
    variants = [make_lr(hr_img, h, 'bicubic')]     # 0: canonical
    kernels_cycle = ['bicubic', 'bilinear', 'lanczos', 'box', 'hamming']
    for j in range(1, K):
        # --- pick a lattice perturbation type round-robin so the K cover all axes ---
        mode = fixed_axis if fixed_axis is not None else (j % 3)
        kern = kernels_cycle[j % len(kernels_cycle)]
        if jitter_mode == 'detector':
            # detector-like localization jitter: continuous Gaussian center-shift + scale error,
            # calibrated to typical detector bbox localization error (sigma ~0.5 LR-px translate,
            # ~10% scale). NOT real detector boxes (Market has no source frame) -- a literature-
            # informed proxy for deployment uncertainty, replacing the uniform +-1 LR-px lattice.
            dx = rng.normal(0, cli.jitter_sigma) * hr_per_lr_x
            dy = rng.normal(0, cli.jitter_sigma) * hr_per_lr_y
            sc = float(np.clip(1.0 + rng.normal(0, cli.jitter_sigma * 0.2), 0.7, 1.3))
            cw, ch = W_hr / sc, H_hr / sc
            cx, cy = W_hr / 2.0 + dx, H_hr / 2.0 + dy
            l, u = int(round(cx - cw / 2)), int(round(cy - ch / 2))
            r, b = int(round(cx + cw / 2)), int(round(cy + ch / 2))
            pad = max(0, -l, -u, r - W_hr, b - H_hr) + 1
            canvas = Image.new('RGB', (W_hr + 2 * pad, H_hr + 2 * pad), (0, 0, 0))
            canvas.paste(hr_img, (pad, pad))
            cropped = canvas.crop((l + pad, u + pad, r + pad, b + pad)).resize(
                (W_hr, H_hr), Image.BICUBIC)
            variants.append(make_lr(cropped, h, kern)); continue
        if mode == 0:
            # sub-pixel phase: fractional shift of up to +/-0.5 LR pixel (in HR px)
            dx = rng.uniform(-0.5, 0.5) * hr_per_lr_x
            dy = rng.uniform(-0.5, 0.5) * hr_per_lr_y
            shifted = hr_img.transform(
                (W_hr, H_hr), Image.AFFINE, (1, 0, dx, 0, 1, dy),
                resample=Image.BICUBIC)
            v = make_lr(shifted, h, kern)
        elif mode == 1:
            # +/-1 LR-pixel bbox crop shift: crop the HR by an integer # of LR pixels on
            # each side then resize back to the HR canvas (== shifting the bbox window).
            sx = int(round(rng.choice([-1, 0, 1]) * hr_per_lr_x))
            sy = int(round(rng.choice([-1, 0, 1]) * hr_per_lr_y))
            left = max(0, sx); upper = max(0, sy)
            right = W_hr + min(0, sx); lower = H_hr + min(0, sy)
            if right - left < 4 or lower - upper < 4:
                left, upper, right, lower = 0, 0, W_hr, H_hr
            cropped = hr_img.crop((left, upper, right, lower)).resize(
                (W_hr, H_hr), Image.BICUBIC)
            v = make_lr(cropped, h, kern)
        else:
            # bbox expand / contract by 1 LR pixel (zoom in/out a touch) + kernel swap
            ez = rng.choice([-1, 1]) * 0.5 * hr_per_lr_y   # expand/contract in HR px
            box = (-ez, -ez * (W_hr / H_hr), W_hr + ez, H_hr + ez * (W_hr / H_hr)) \
                if ez > 0 else (abs(ez), abs(ez) * (W_hr / H_hr),
                                W_hr - abs(ez), H_hr - abs(ez) * (W_hr / H_hr))
            # PIL crop on a fractional/negative box: emulate via paste on padded canvas
            l, u, r, b = box
            l, u, r, b = int(round(l)), int(round(u)), int(round(r)), int(round(b))
            pad = max(0, -l, -u, r - W_hr, b - H_hr) + 1
            canvas = Image.new('RGB', (W_hr + 2 * pad, H_hr + 2 * pad), (0, 0, 0))
            canvas.paste(hr_img, (pad, pad))
            cropped = canvas.crop((l + pad, u + pad, r + pad, b + pad)).resize(
                (W_hr, H_hr), Image.BICUBIC)
            v = make_lr(cropped, h, kern)
        variants.append(v)
    return variants


def make_tta_variants(lr_img, K, rng, pad=10, strong=False):
    """ORDINARY TTA control: K views of ONE bicubic LR image via pad+RandomCrop (+ hflip).
    NO lattice/phase semantics -- the standard cheap test-time augmentation.  variant 0 ==
    the un-augmented LR so the single-LR baseline is shared with the lattice path.
    strong=True (LM-S2 defense): ALSO add resize-jitter + brightness/contrast = a RICHER TTA,
    so the lattice ensemble must beat a STRONG (not just crop+flip) ordinary baseline."""
    from PIL import ImageEnhance
    W, H = lr_img.size
    out = [lr_img]                                  # 0: identity (== single LR)
    for j in range(1, K):
        canvas = Image.new('RGB', (W + 2 * pad, H + 2 * pad), (0, 0, 0))
        canvas.paste(lr_img, (pad, pad))
        cx = rng.randint(0, 2 * pad + 1)
        cy = rng.randint(0, 2 * pad + 1)
        crop = canvas.crop((cx, cy, cx + W, cy + H))
        if rng.rand() < 0.5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        if strong:
            if rng.rand() < 0.5:
                s = rng.uniform(0.8, 1.0)
                crop = crop.resize((max(1, int(W * s)), max(1, int(H * s))), Image.BILINEAR).resize((W, H), Image.BILINEAR)
            if rng.rand() < 0.5:
                crop = ImageEnhance.Brightness(crop).enhance(rng.uniform(0.8, 1.2))
            if rng.rand() < 0.5:
                crop = ImageEnhance.Contrast(crop).enhance(rng.uniform(0.8, 1.2))
        out.append(crop)
    return out


def pil_to_tensor_np(img):
    """PIL 384x128 -> normalized CHW float32 ndarray (matches dataset _image_to_tensor)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC
    arr = (arr - PIXEL_MEAN) / PIXEL_STD
    return arr.transpose(2, 0, 1)                            # CHW


# =========================================================================== #
# frozen model (pose DISABLED at test: pose_dict=None -> plain SOLIDER global feat)
# =========================================================================== #
class FrozenExtractor:
    def __init__(self):
        import torch
        from config import cfg
        from model import make_model
        from datasets.market1501 import Market1501
        from datasets.msmt17 import MSMT17
        self.torch = torch
        cfg.merge_from_file(os.path.join(_repo, cli.config))
        _overrides = [
            'TEST.WEIGHT', os.path.join(_repo, cli.ckpt),
            'MODEL.POSE_TEST_FEAT', 'global',     # single clean global vector
            'TEST.NECK_FEAT', 'after',            # trained BN-neck eval feature
            'TEST.FEAT_NORM', 'yes',
        ]
        if cli.semantic_weight >= 0:               # match ckpt training (SOLIDER semantic-aware backbone)
            _overrides += ['MODEL.SEMANTIC_WEIGHT', str(cli.semantic_weight)]
        cfg.merge_from_list(_overrides)  # leave PRETRAIN_* as in config; load_param(ckpt) overwrites all weights anyway
        cfg.freeze()
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
        # need num_classes/camera_num/view_num to build the head; read from Market meta.
        ds = {'market1501': Market1501, 'msmt17': MSMT17}[cli.dataset](
            root=os.path.join(_repo, cli.data_root), verbose=False)
        model = make_model(cfg, num_class=ds.num_train_pids,
                           camera_num=ds.num_train_cams, view_num=1,
                           semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
        model.load_param(os.path.join(_repo, cli.ckpt))
        self.model = model.cuda().eval()
        self.cam_dummy = None
        print(f"[model] loaded {cli.ckpt}; POSE_TEST_FEAT=global; pose DISABLED at test "
              f"(pose_dict=None -> plain backbone global feat). num_cls={ds.num_train_pids}",
              flush=True)

    def feats_from_pil(self, pil_list):
        """Batched frozen features for a list of 384x128 PIL images. L2-normalized."""
        torch = self.torch
        feats = []
        B = cli.batch
        for s in range(0, len(pil_list), B):
            chunk = pil_list[s:s + B]
            arr = np.stack([pil_to_tensor_np(im) for im in chunk], 0)
            t = torch.from_numpy(arr).cuda(non_blocking=True)
            cam = torch.zeros(t.shape[0], dtype=torch.long, device=t.device)
            view = torch.zeros(t.shape[0], dtype=torch.long, device=t.device)
            with torch.no_grad():
                out = self.model(t, cam_label=cam, view_label=view, pose_dict=None)
            feat = out[0] if isinstance(out, (tuple, list)) else out
            assert torch.is_tensor(feat) and feat.dim() == 2, \
                f"expected single global vector, got {type(feat)} {getattr(feat,'shape',None)}"
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            feats.append(feat.cpu().numpy().astype(np.float32))
        return np.concatenate(feats, 0)


# =========================================================================== #
# eval helpers (Market protocol: drop same pid&cam junk)
# =========================================================================== #
def eval_map(dist, q_pid, q_cam, g_pid, g_cam, max_rank=10):
    num_q = dist.shape[0]
    idx = np.argsort(dist, axis=1)
    all_AP, all_cmc, nv = [], [], 0
    for i in range(num_q):
        order = idx[i]
        keep = ~((g_pid[order] == q_pid[i]) & (g_cam[order] == q_cam[i]))
        m = (g_pid[order][keep] == q_pid[i]).astype(np.int32)
        if not m.any():
            continue
        nv += 1
        cmc = m.cumsum(); cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        tmp = m.cumsum(); prec = tmp / (np.arange(len(m)) + 1.0)
        all_AP.append((prec * m).sum() / m.sum())
    all_cmc = np.asarray(all_cmc).mean(0)
    return dict(mAP=float(np.mean(all_AP)) * 100, r1=float(all_cmc[0]) * 100,
                r5=float(all_cmc[4]) * 100, r10=float(all_cmc[9]) * 100, nq=nv)


def per_query_ap(dist, q_pid, q_cam, g_pid, g_cam):
    num_q = dist.shape[0]
    idx = np.argsort(dist, axis=1)
    aps = np.full(num_q, -1.0)
    for i in range(num_q):
        order = idx[i]
        keep = ~((g_pid[order] == q_pid[i]) & (g_cam[order] == q_cam[i]))
        m = (g_pid[order][keep] == q_pid[i]).astype(np.int32)
        if not m.any():
            continue
        prec = m.cumsum() / (np.arange(len(m)) + 1.0)
        aps[i] = (prec * m).sum() / m.sum()
    return aps


def n_false_in_topk(dist, q_pid, q_cam, g_pid, g_cam, k=10):
    """#wrong-identity gallery within the LR top-k (after junk removal). The TRIVIAL
    failure proxy from Hubness §7.6 that the signal must survive."""
    num_q = dist.shape[0]
    idx = np.argsort(dist, axis=1)
    nf = np.zeros(num_q)
    for i in range(num_q):
        order = idx[i]
        keep = ~((g_pid[order] == q_pid[i]) & (g_cam[order] == q_cam[i]))
        order_k = order[keep][:k]
        nf[i] = int((g_pid[order_k] != q_pid[i]).sum())
    return nf


# --- stats (no scipy) ---
def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 3:
        return float('nan'), 0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return (float((rx * ry).sum() / d) if d > 0 else float('nan')), len(x)


def partial_spearman(x, y, Z):
    """partial Spearman of (x,y) controlling covariate(s) Z (rank-residual correlation)."""
    x = np.asarray(x, float); y = np.asarray(y, float); Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(1)
    x, y, Z = x[ok], y[ok], Z[ok]
    if len(x) < 5:
        return float('nan'), 0
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    rx, ry = rk(x), rk(y)
    Zr = np.column_stack([np.ones(len(x))] + [rk(Z[:, j]) for j in range(Z.shape[1])])
    res = lambda r: r - Zr @ np.linalg.lstsq(Zr, r, rcond=None)[0]
    ex, ey = res(rx), res(ry)
    d = np.sqrt((ex**2).sum() * (ey**2).sum())
    return (float((ex * ey).sum() / d) if d > 0 else float('nan')), len(x)


# =========================================================================== #
# MAIN
# =========================================================================== #
def main():
    print("#" * 88)
    print("# LATTICE-MARGINALIZED ReID KILL-SWITCH  (Market, frozen exp260b, pose-OFF global feat)")
    print("#" * 88)

    if cli.dataset == 'msmt17':
        q_items = msmt17_split(os.path.join(_repo, cli.data_root), 'list_query.txt')
        g_items = msmt17_split(os.path.join(_repo, cli.data_root), 'list_gallery.txt')
    else:
        q_items = list_split(os.path.join(_repo, cli.data_root, 'market1501', 'query'))
        g_items = list_split(os.path.join(_repo, cli.data_root, 'market1501', 'bounding_box_test'))
    if cli.smoke > 0:
        q_items = q_items[:cli.smoke]
    print(f"[data] #query={len(q_items)}  #gallery={len(g_items)}  K={cli.K}  heights={cli.heights}")

    ext = FrozenExtractor()

    # ---- HR query reference images (for lattice generation) ----
    t0 = time.time()
    q_pid = np.array([it[1] for it in q_items]); q_cam = np.array([it[2] for it in q_items])
    g_pid = np.array([it[1] for it in g_items]); g_cam = np.array([it[2] for it in g_items])
    from datasets.bases import read_image
    hr_q = [_to_target_aspect(read_image(it[0])) for it in q_items]

    # ---- HR gallery features (once) ----
    if cli.reuse_gallery and os.path.exists(cli.cache_gallery) and cli.smoke == 0:
        z = np.load(cli.cache_gallery, allow_pickle=True)
        gf = z['gf']; assert len(gf) == len(g_items)
        print(f"[gallery] reuse HR feats {cli.cache_gallery}  ({gf.shape})")
    else:
        hr_g = [_to_target_aspect(read_image(it[0])) for it in g_items]
        gf = ext.feats_from_pil(hr_g)
        if cli.smoke == 0:
            np.savez(cli.cache_gallery, gf=gf)
        del hr_g
        print(f"[gallery] HR feats {gf.shape}  ({time.time()-t0:.0f}s)")

    # ---- sanity: HR query vs HR gallery (upper bound; no LR) ----
    hr_qf = ext.feats_from_pil(hr_q)
    dist_hr = 1.0 - hr_qf @ gf.T
    r_hr = eval_map(dist_hr, q_pid, q_cam, g_pid, g_cam)
    print(f"\n[SANITY] HR-query vs HR-gallery  mAP={r_hr['mAP']:.2f} R1={r_hr['r1']:.2f} "
          f"R10={r_hr['r10']:.2f}  nq={r_hr['nq']}   (upper bound)")

    summary = {}
    for h in cli.heights:
        print("\n" + "=" * 88)
        print(f"  LR HEIGHT h = {h}px      (K={cli.K} variants/query)")
        print("=" * 88)

        # ============ build feats for every variant ==================
        # STREAMING over query chunks so we never hold all K*Nq PIL images in RAM at once
        # (materializing 2*K*Nq decoded 384x128 images = ~18GB at K=9,Nq=3368 -> OOM).
        # phase-lattice variants -> f_lat [Nq,K,D]; ordinary-TTA -> f_tta [Nq,K,D].
        Nq = len(hr_q)
        f_lat_chunks, f_tta_chunks = [], []
        CHUNK = 256                                  # queries per chunk -> 256*9 = 2304 PILs peak
        for cs in range(0, Nq, CHUNK):
            ce = min(cs + CHUNK, Nq)
            _ax = cli.lattice_axis if cli.lattice_axis >= 0 else None
            lat_pils = [make_lattice_variants(hr_q[i], h, cli.K, RNG, fixed_axis=_ax, jitter_mode=cli.jitter_mode) for i in range(cs, ce)]
            single_lr = [v[0] for v in lat_pils]
            tta_pils = [make_tta_variants(single_lr[k], cli.K, RNG, strong=cli.strong_tta) for k in range(len(lat_pils))]
            flat_lat = [im for vs in lat_pils for im in vs]
            flat_tta = [im for vs in tta_pils for im in vs]
            f_lat_chunks.append(ext.feats_from_pil(flat_lat).reshape(ce - cs, cli.K, -1))
            f_tta_chunks.append(ext.feats_from_pil(flat_tta).reshape(ce - cs, cli.K, -1))
            del lat_pils, single_lr, tta_pils, flat_lat, flat_tta
        f_lat = np.concatenate(f_lat_chunks, 0)      # [Nq,K,D]
        f_tta = np.concatenate(f_tta_chunks, 0)
        del f_lat_chunks, f_tta_chunks
        f_single = f_lat[:, 0, :]                                     # == canonical LR feat

        # ---------- (A) same-image phase feature variance ----------
        # mean over queries of mean pairwise (1-cos) among the K lattice variants.
        def mean_pairwise_dist(F):       # F: [Nq,K,D] L2-normed
            G = F @ np.transpose(F, (0, 2, 1))      # [Nq,K,K] cos
            iu = np.triu_indices(F.shape[1], k=1)
            pd = 1.0 - G[:, iu[0], iu[1]]            # [Nq, n_pairs]
            return pd.mean(1)                        # [Nq]
        phase_var = mean_pairwise_dist(f_lat)        # per-query lattice spread
        tta_var = mean_pairwise_dist(f_tta)
        # also: drift of canonical LR from HR (how far one LR is from the true HR feat)
        lr_hr_drift = 1.0 - (f_single * hr_qf).sum(1)
        print(f"  (A) same-image phase feature variance (mean pairwise 1-cos over K):")
        print(f"      lattice phase-var  mean={phase_var.mean():.4f}  median={np.median(phase_var):.4f}  "
              f"p90={np.quantile(phase_var,0.9):.4f}")
        print(f"      ordinary TTA  var  mean={tta_var.mean():.4f}  (reference)")
        print(f"      single-LR -> HR feat drift  mean={lr_hr_drift.mean():.4f}  "
              f"(absolute LR distortion)")

        # ---------- (B) rank volatility across phases ----------
        # for each variant get top-10 gallery (raw kNN, no junk removal needed for volatility).
        # argpartition (not full argsort) keeps the transient allocation small.
        sims = f_lat @ gf.T                          # [Nq,K,Ng]  (~1.9GB f32)
        part = np.argpartition(-sims, kth=10, axis=2)[:, :, :10]      # top-10 unordered
        rows = np.arange(Nq)[:, None, None]; kk = np.arange(cli.K)[None, :, None]
        ord10 = np.argsort(-sims[rows, kk, part], axis=2)             # order within 10
        top10 = np.take_along_axis(part, ord10, axis=2)               # [Nq,K,10] sorted
        del sims, part, ord10                        # free the big arrays before ensemble
        top1 = top10[:, :, 0]                         # [Nq,K]
        # top1 agreement: fraction of variants whose top1 == canonical-variant top1
        top1_agree = (top1 == top1[:, [0]]).mean(1)   # [Nq]  (1.0 = perfectly stable)
        # top10 Jaccard between canonical variant and each other, averaged
        def jacc(a, b):
            sa, sb = set(a.tolist()), set(b.tolist())
            return len(sa & sb) / float(len(sa | sb))
        jac10 = np.array([np.mean([jacc(top10[i, 0], top10[i, j]) for j in range(1, cli.K)])
                          for i in range(len(hr_q))])
        # ID-level top1 flip: does the IDENTITY of rank-1 change across phases?
        top1_pid = g_pid[top1]                        # [Nq,K]
        id_flip = np.array([len(np.unique(top1_pid[i])) for i in range(len(hr_q))])  # #distinct top1 IDs
        print(f"  (B) rank volatility across the K phases:")
        print(f"      top1 stays==canonical : mean={top1_agree.mean():.3f}  "
              f"(1.0=stable; lower=more volatile)")
        print(f"      top10 Jaccard(canon,j): mean={jac10.mean():.3f}  "
              f"(1.0=identical sets)")
        print(f"      #distinct rank-1 IDs over K phases: mean={id_flip.mean():.2f}  "
              f"(>1 => the retrieved identity FLIPS with sampling phase)  "
              f"frac queries with >=2 = {100*(id_flip>=2).mean():.1f}%")

        # ---------- (D) ensemble mAP ----------
        # single LR baseline
        d_single = 1.0 - f_single @ gf.T
        r_single = eval_map(d_single, q_pid, q_cam, g_pid, g_cam)
        # phase-lattice ENSEMBLE: feature-mean (renormed) and MaxSim
        f_lat_mean = f_lat.mean(1)
        f_lat_mean /= (np.linalg.norm(f_lat_mean, axis=1, keepdims=True) + 1e-12)
        d_lat_mean = 1.0 - f_lat_mean @ gf.T
        r_lat_mean = eval_map(d_lat_mean, q_pid, q_cam, g_pid, g_cam)
        # MaxSim: per (q,g) take the BEST sim over the K query variants
        sim_lat_full = f_lat @ gf.T                  # [Nq,K,Ng]
        sim_lat_max = sim_lat_full.max(1)            # [Nq,Ng]
        r_lat_max = eval_map(1.0 - sim_lat_max, q_pid, q_cam, g_pid, g_cam)
        # logsumexp: soft decision-level marginalization, LM-ReID s=tau*log[1/K sum_k exp(cos/tau)]
        # (interpolates embedding-mean tau->inf and MaxSim tau->0). stabilized by subtracting max.
        _tau_lse = 0.1
        _s = sim_lat_full / _tau_lse                 # [Nq,K,Ng]
        _smax = _s.max(1, keepdims=True)             # [Nq,1,Ng]
        sim_lat_lse = _tau_lse * (_smax[:, 0] + np.log(np.exp(_s - _smax).mean(1) + 1e-12))  # [Nq,Ng]
        r_lat_lse = eval_map(1.0 - sim_lat_lse, q_pid, q_cam, g_pid, g_cam)
        # ---- adaptive-K (supporting): per-query phase volatility -> spend K only where it helps ----
        # high-volatility queries (lattice-sensitive) marginalize over K; low-vol use K=1 (single).
        r_adapt, avg_k = None, float(cli.K)
        if cli.adaptive_k:
            fl = f_lat / (np.linalg.norm(f_lat, axis=2, keepdims=True) + 1e-12)   # [Nq,K,D] L2
            pvol = 1.0 - np.einsum('ikd,ild->ikl', fl, fl).mean((1, 2))           # [Nq] 1-mean pairwise cos
            use_marg = pvol > np.median(pvol)                                     # high vol -> marginalize K
            sim_adapt = np.where(use_marg[:, None], sim_lat_max, f_single @ gf.T)  # [Nq,Ng]
            r_adapt = eval_map(1.0 - sim_adapt, q_pid, q_cam, g_pid, g_cam)
            avg_k = float(use_marg.mean()) * cli.K + (1 - float(use_marg.mean())) * 1.0
        # ---- LPA ORACLE headroom (train-time mechanism A kill-switch) ----
        # per query, pick the SINGLE variant that best separates the true ID (oracle uses labels).
        # = upper bound for a LEARNED per-variant weight (LPA posterior). If ~= uniform mean-feat,
        # the LPA weighting is DEAD before we build the head (no headroom to learn).
        oracle_sim = np.empty((len(q_pid), len(g_pid)), dtype=np.float32)
        for i in range(len(q_pid)):
            keep = ~((g_pid == q_pid[i]) & (g_cam == q_cam[i]))
            pos = (g_pid == q_pid[i]) & keep
            if not pos.any():
                oracle_sim[i] = sim_lat_full[i].mean(0); continue
            neg = (g_pid != q_pid[i]) & keep
            mar = sim_lat_full[i][:, pos].max(1) - sim_lat_full[i][:, neg].max(1)   # [K]
            oracle_sim[i] = sim_lat_full[i][int(mar.argmax())]
        r_oracle = eval_map(1.0 - oracle_sim, q_pid, q_cam, g_pid, g_cam)
        # ordinary-TTA ENSEMBLE (the life/death control), SAME fusions
        f_tta_mean = f_tta.mean(1)
        f_tta_mean /= (np.linalg.norm(f_tta_mean, axis=1, keepdims=True) + 1e-12)
        r_tta_mean = eval_map(1.0 - f_tta_mean @ gf.T, q_pid, q_cam, g_pid, g_cam)
        sim_tta_max = (f_tta @ gf.T).max(1)
        r_tta_max = eval_map(1.0 - sim_tta_max, q_pid, q_cam, g_pid, g_cam)

        print(f"  (D) ENSEMBLE mAP (K={cli.K}):")
        print(f"      single bicubic LR            : mAP={r_single['mAP']:.3f}  R1={r_single['r1']:.3f}")
        print(f"      phase-lattice  mean-feat     : mAP={r_lat_mean['mAP']:.3f}  R1={r_lat_mean['r1']:.3f}  "
              f"(d{r_lat_mean['mAP']-r_single['mAP']:+.3f})")
        print(f"      phase-lattice  MaxSim        : mAP={r_lat_max['mAP']:.3f}  R1={r_lat_max['r1']:.3f}  "
              f"(d{r_lat_max['mAP']-r_single['mAP']:+.3f})")
        print(f"      phase-lattice  logsumexp     : mAP={r_lat_lse['mAP']:.3f}  R1={r_lat_lse['r1']:.3f}  "
              f"(d{r_lat_lse['mAP']-r_single['mAP']:+.3f})")
        if r_adapt is not None:
            print(f"      ADAPTIVE-K (median-vol thr)  : mAP={r_adapt['mAP']:.3f}  avg_K={avg_k:.2f}  "
                  f"({avg_k/cli.K*100:.0f}% of K{cli.K} compute; fixed-K MaxSim {r_lat_max['mAP']:.3f} / single {r_single['mAP']:.3f})")
        print(f"      LPA-ORACLE per-q best variant: mAP={r_oracle['mAP']:.3f}  R1={r_oracle['r1']:.3f}  "
              f"(headroom over mean-feat={r_oracle['mAP']-r_lat_mean['mAP']:+.3f}; >~1.0 => LPA weighting worth building)")
        print(f"      ----  LIFE/DEATH CONTROL (ordinary TTA, same K & fusion)  ----")
        print(f"      ordinary-TTA   mean-feat     : mAP={r_tta_mean['mAP']:.3f}  R1={r_tta_mean['r1']:.3f}  "
              f"(d{r_tta_mean['mAP']-r_single['mAP']:+.3f})")
        print(f"      ordinary-TTA   MaxSim        : mAP={r_tta_max['mAP']:.3f}  R1={r_tta_max['r1']:.3f}  "
              f"(d{r_tta_max['mAP']-r_single['mAP']:+.3f})")
        best_lat = max(r_lat_mean['mAP'], r_lat_max['mAP'])
        best_tta = max(r_tta_mean['mAP'], r_tta_max['mAP'])
        print(f"      >> phase-lattice best gain = {best_lat-r_single['mAP']:+.3f}   "
              f"ordinary-TTA best gain = {best_tta-r_single['mAP']:+.3f}   "
              f"LATTICE-MINUS-TTA = {best_lat-best_tta:+.3f}  (must be clearly >0 to live)")

        # ---------- (C) does phase variance EXPLAIN failure? ----------
        ap_single = per_query_ap(d_single, q_pid, q_cam, g_pid, g_cam)
        err = 1.0 - ap_single
        nfalse = n_false_in_topk(d_single, q_pid, q_cam, g_pid, g_cam, k=10)
        valid = ap_single >= 0
        r_pv, _ = spearman(err[valid], phase_var[valid])
        r_nf, _ = spearman(err[valid], nfalse[valid])
        r_drift, _ = spearman(err[valid], lr_hr_drift[valid])
        # decisive: partial out the trivial proxy #false-in-topk (Hubness §7.6)
        pr_pv_nf, npart = partial_spearman(err[valid], phase_var[valid], nfalse[valid])
        pr_nf_pv, _ = partial_spearman(err[valid], nfalse[valid], phase_var[valid])
        rho_pv_nf, _ = spearman(phase_var[valid], nfalse[valid])
        # also control per-image LR severity (single-LR->HR drift) jointly with #false,
        # so phase-var cannot be a proxy for "this crop degrades badly under LR".
        cov2 = np.column_stack([nfalse[valid], lr_hr_drift[valid]])
        pr_pv_both, _ = partial_spearman(err[valid], phase_var[valid], cov2)
        print(f"  (C) does phase-variance explain LR failure?  (decisive vs #false-in-topk)")
        print(f"      rho(AP-err, phase-var)            = {r_pv:+.4f}")
        print(f"      rho(AP-err, #false-in-topk)       = {r_nf:+.4f}   [TRIVIAL proxy]")
        print(f"      rho(AP-err, single-LR->HR drift)  = {r_drift:+.4f}   [LR severity proxy]")
        print(f"      Spearman(phase-var, #false)       = {rho_pv_nf:+.4f}")
        print(f"      partial rho(AP-err, phase-var | #false)        = {pr_pv_nf:+.4f}  (n={npart})  "
              f"<== phase-var INDEPENDENT signal")
        print(f"      partial rho(AP-err, phase-var | #false+drift)  = {pr_pv_both:+.4f}  "
              f"<== also controlling LR severity")
        print(f"      partial rho(AP-err, #false | phase-var)        = {pr_nf_pv:+.4f}  (reverse)")

        summary[h] = dict(
            phase_var=phase_var.mean(), tta_var=tta_var.mean(),
            top1_agree=top1_agree.mean(), jac10=jac10.mean(), id_flip=id_flip.mean(),
            frac_flip=100 * (id_flip >= 2).mean(),
            mAP_single=r_single['mAP'], R1_single=r_single['r1'],
            mAP_lat_mean=r_lat_mean['mAP'], mAP_lat_max=r_lat_max['mAP'],
            mAP_tta_mean=r_tta_mean['mAP'], mAP_tta_max=r_tta_max['mAP'],
            best_lat_gain=best_lat - r_single['mAP'], best_tta_gain=best_tta - r_single['mAP'],
            lat_minus_tta=best_lat - best_tta,
            rho_pv=r_pv, rho_nf=r_nf, partial_pv_nf=pr_pv_nf, partial_pv_both=pr_pv_both,
            rho_drift=r_drift,
        )

    # =================== FINAL TABLE / VERDICT ===================
    print("\n" + "#" * 88)
    print("SUMMARY TABLE")
    print("#" * 88)
    hdr = ("  h | phase-var | TTAvar | top1stab jac10 idFlip flip% | "
           "single  lat-mean lat-max  tta-max | LATgain TTAgain LAT-TTA | "
           "rho_pv rho_nf  pv|nf")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for h in cli.heights:
        s = summary[h]
        print(f" {h:3d}| {s['phase_var']:8.4f} | {s['tta_var']:6.4f} | "
              f"{s['top1_agree']:6.3f} {s['jac10']:5.3f} {s['id_flip']:5.2f} {s['frac_flip']:5.1f}| "
              f"{s['mAP_single']:6.2f} {s['mAP_lat_mean']:7.2f} {s['mAP_lat_max']:7.2f} {s['mAP_tta_max']:7.2f}| "
              f"{s['best_lat_gain']:+6.2f} {s['best_tta_gain']:+6.2f} {s['lat_minus_tta']:+6.2f} | "
              f"{s['rho_pv']:+.3f} {s['rho_nf']:+.3f} {s['partial_pv_nf']:+.3f}")

    print("\n  VERDICT GUIDE (h<=32 is the regime of interest):")
    print("   GO  : rank volatility clearly nonzero (idFlip>1 / top1stab<1) AND lattice gain >= +2 mAP")
    print("         AND LAT-TTA clearly >0 AND partial rho(AP-err, phase-var | #false) clearly >0")
    print("   DEAD: phase-var ~ TTAvar / lattice gain ~ 0 / LAT-TTA ~ 0 / pv|nf ~ 0 (absorbed by #false)")
    # auto-call
    go_flags = []
    for h in cli.heights:
        if h > 32:
            continue
        s = summary[h]
        cond = (s['id_flip'] > 1.05 and s['best_lat_gain'] >= 2.0 and
                s['lat_minus_tta'] > 0.5 and s['partial_pv_nf'] > 0.10)
        go_flags.append(cond)
        print(f"   h={h}: volatility={'Y' if (s['id_flip']>1.05) else 'n'} "
              f"lat_gain>=2={'Y' if s['best_lat_gain']>=2.0 else 'n'} "
              f"LAT-TTA>0.5={'Y' if s['lat_minus_tta']>0.5 else 'n'} "
              f"pv|nf>0.10={'Y' if s['partial_pv_nf']>0.10 else 'n'}  ==> "
              f"{'GO' if cond else 'fail-this-h'}")
    print(f"\n  >>> AUTO VERDICT: {'GO (>=1 height passes all gates)' if any(go_flags) else 'DEAD (no height clears every gate)'}")
    print("\n[done] lattice kill-switch complete.")


if __name__ == '__main__':
    main()
