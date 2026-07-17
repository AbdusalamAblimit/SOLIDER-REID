#!/usr/bin/env python3
"""Candidate B — "Physically directed uncertainty containment" ZERO-TRAINING kill-switch.

Re-frame under test (B_CONTAINMENT_DESIGN.md):
    Aerial-ground ReID is NOT symmetric matching but *physically directed
    uncertainty containment*. The aerial (low-altitude, top-down, pixel-budget
    limited) observation is an UNDER-DETERMINED projection -> WIDE candidate
    identity distribution. The ground (high-res) observation is CERTAIN -> NARROW
    distribution. The correct ground evidence should fall *inside* the aerial
    uncertainty envelope.

Per image we build a diagonal Gaussian N(mu, diag(sigma^2)):
    mu     = frozen Swin global (BN, pre-L2-normalize) feature.
    sigma^2 = TTA / augmentation variance over K augmented views (hflip +
              RandomResizedCrop(scale 0.8-1.0) + light ColorJitter), variance per
              dim. sigma ONLY comes from the image itself, NEVER from ID label.

Directional containment score (A->G: query aerial a, gallery ground g):
    correct direction = ground NARROW falls into aerial WIDE envelope
                      = D = KL(N_g || N_a), retrieve by D ASCENDING.
    KL(N_g||N_a) = 0.5 * sum_d[ ln(s2_a/s2_g) + (s2_g + (mu_g-mu_a)^2)/s2_a - 1 ]
    the mean term (mu_g-mu_a)^2 is divided by the AERIAL variance s2_a (large ->
    tolerant) -- this is the source of the asymmetry.

NOTHING is trained. Frozen Swin + torch.no_grad + numpy. swin_fix256 ckpt (mAP 67.33).

Run on lab-3090-d:
    cd /root/work/SOLIDER-REID/experiments/cargo_cvpb && \
    PYTHONUNBUFFERED=1 python3 cvpb_containment_killswitch.py \
      --ckpt /root/work/SOLIDER-REID/log/cargo/cvpb_swin_fix256/model_best.pth \
      --data_root /root/work/SOLIDER-REID/data \
      --swin_pretrain /root/work/SOLIDER-REID/pretrained/swin_small.pth \
      --K 16  2>&1 | tee /tmp/cvpb_containment.log
    # smoke first: add  --smoke 200
"""
import os, re, sys, time, argparse
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)                                    # cargo_cvpb: afd_train
sys.path.insert(0, os.path.join(_here, '..', 'afd_reid'))    # afd_reid: afd_model, cargo_dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='/root/work/SOLIDER-REID/log/cargo/cvpb_swin_fix256/model_best.pth')
ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
ap.add_argument('--swin_pretrain', default='/root/work/SOLIDER-REID/pretrained/swin_small.pth')
ap.add_argument('--img_size', type=int, nargs=2, default=[256, 128])
ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--test_batch', type=int, default=128)
ap.add_argument('--K', type=int, default=16, help='# TTA augmented views for sigma')
ap.add_argument('--sigma_floor', type=float, default=1e-4)
ap.add_argument('--smoke', type=int, default=0, help='if >0, cap #query to this for a smoke run')
ap.add_argument('--seed', type=int, default=42)
cli = ap.parse_args()
np.random.seed(cli.seed); torch.manual_seed(cli.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from afd_model import build_model
from cargo_dataset import CARGO, filter_by_view, CARGOImageDataset

MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225)
IMG_H, IMG_W = cli.img_size


# --------------------------------------------------------------------------- #
# model: frozen Swin, extract BN feature (pre-L2-normalize) as mu
# --------------------------------------------------------------------------- #
args = argparse.Namespace(
    backbone='swin_small', swin_pretrain=cli.swin_pretrain, swin_semantic_weight=0.2,
    img_size=cli.img_size, use_afd=False, pool='gem', last_stride=1,
    workers=cli.workers, test_batch=cli.test_batch,
    airl_dualbranch=False, airl_dualbranch_iso=False,
)
ds = CARGO(root=cli.data_root, verbose=True)
model = build_model(ds.num_train_pids, args).to(device)
ck = torch.load(cli.ckpt, map_location='cpu')
state = ck.get('model', ck.get('state_dict', ck)) if isinstance(ck, dict) else ck
miss = model.load_state_dict(state, strict=False)
print(f"[load] missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}")
# sanity: the only acceptable 'missing' are the classifier (eval-unused). loudly flag others.
bad_missing = [k for k in miss.missing_keys if 'classifier' not in k]
if bad_missing:
    print(f"[WARN] non-classifier missing keys: {bad_missing[:8]} ...")
model.eval()


@torch.no_grad()
def _bn_feat(img_batch):
    """frozen Swin -> pooled global -> BNNeck -> bn_feat (B,768), pre-L2-normalize.

    This is exactly the model's eval feature BEFORE F.normalize, so L2-normalizing
    it reproduces the trained cosine eval (mAP 67.33 line).
    """
    feat_map = model.backbone_swin(img_batch.to(device, non_blocking=True))  # (B,768,H,W)
    _g, bn = model._embed(feat_map)   # _embed = pool(avg) -> bottleneck (BN)
    return bn.detach().cpu()          # NOT normalized


# --------------------------------------------------------------------------- #
# transforms: deterministic (for mu) + stochastic TTA (for sigma)
# --------------------------------------------------------------------------- #
_normalize = T.Normalize(mean=MEAN, std=STD)
# deterministic eval transform == cargo_dataset.build_transforms(is_train=False)
_tf_det = T.Compose([
    T.Resize((IMG_H, IMG_W), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(), _normalize,
])
# stochastic TTA: hflip + RandomResizedCrop(scale 0.8-1.0) + light ColorJitter.
# sigma comes ONLY from the image (augmentations), never from ID label.
_tf_tta = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomResizedCrop((IMG_H, IMG_W), scale=(0.8, 1.0), ratio=(0.4, 0.6),
                        interpolation=T.InterpolationMode.BICUBIC),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
    T.ToTensor(), _normalize,
])


def _load_img(path):
    got = None
    for _ in range(5):
        try:
            got = Image.open(path).convert('RGB'); break
        except (IOError, OSError):
            continue
    if got is None:
        got = Image.new('RGB', (IMG_W, IMG_H))
    return got


class _MuTTADataset(torch.utils.data.Dataset):
    """Returns (det_img, K stacked TTA imgs) for one sample."""
    def __init__(self, samples, K, degrade=None):
        self.samples = samples; self.K = K; self.degrade = degrade
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img = _load_img(self.samples[idx]['img_path'])
        if self.degrade is not None:
            img = self.degrade(img)             # synthetic degradation (PIL->PIL)
        det = _tf_det(img)
        tta = torch.stack([_tf_tta(img) for _ in range(self.K)], 0)  # (K,3,H,W)
        return det, tta


@torch.no_grad()
def extract_mu_sigma(samples, K, degrade=None, tag=''):
    """Return mu (N,768) and sigma2 (N,768): mu from deterministic feat, sigma2 the
    per-dim variance of bn_feat over K TTA views. Diagonal Gaussian per image."""
    dl = torch.utils.data.DataLoader(
        _MuTTADataset(samples, K, degrade=degrade),
        batch_size=max(1, cli.test_batch // max(1, K)),   # K views inflate effective batch
        shuffle=False, num_workers=cli.workers, pin_memory=True)
    mus, s2s = [], []
    t0 = time.time()
    for bi, (det, tta) in enumerate(dl):
        b = det.size(0)
        mu = _bn_feat(det)                                  # (b,768)
        kk = tta.view(b * K, *tta.shape[2:])                # (b*K,3,H,W)
        ft = _bn_feat(kk).view(b, K, -1)                    # (b,K,768)
        s2 = ft.var(dim=1, unbiased=True)                   # (b,768) per-dim variance
        mus.append(mu); s2s.append(s2)
        if bi % 20 == 0:
            print(f"    [{tag}] batch {bi}/{len(dl)}  ({time.time()-t0:.0f}s)", flush=True)
    mu = torch.cat(mus, 0).numpy().astype(np.float64)
    s2 = torch.cat(s2s, 0).numpy().astype(np.float64)
    s2 = np.maximum(s2, cli.sigma_floor)                    # floor to avoid /0
    return mu, s2


# --------------------------------------------------------------------------- #
# scoring functions  (all return a DISTANCE matrix Nq x Ng; retrieve by ascending)
# --------------------------------------------------------------------------- #
def cosine_dist(mu_q, mu_g):
    qn = mu_q / (np.linalg.norm(mu_q, axis=1, keepdims=True) + 1e-12)
    gn = mu_g / (np.linalg.norm(mu_g, axis=1, keepdims=True) + 1e-12)
    return 1.0 - (qn @ gn.T)                                 # (Nq,Ng)


def _kl_diag(mu_a, s2_a, mu_b, s2_b, block=256):
    """KL( N_b || N_a ) for every (a in rows, b in cols), broadcast in blocks.

    KL(N_b||N_a) = 0.5 * sum_d[ ln(s2_a/s2_b) + (s2_b + (mu_b-mu_a)^2)/s2_a - 1 ]
    Returns matrix of shape (Na, Nb) -- rows index 'a' (the WIDE/aerial side when
    used as KL(g||a) with a=aerial), cols index 'b'.
    """
    Na, D = mu_a.shape; Nb = mu_b.shape[0]
    out = np.empty((Na, Nb), dtype=np.float64)
    log_s2a = np.log(s2_a)                                   # (Na,D)
    sum_log_s2b = np.log(s2_b).sum(axis=1)                   # (Nb,)
    inv_s2a = 1.0 / s2_a                                     # (Na,D)
    for i in range(0, Na, block):
        a_mu = mu_a[i:i+block]                               # (ba,D)
        a_inv = inv_s2a[i:i+block]                           # (ba,D)
        a_logsum = log_s2a[i:i+block].sum(axis=1)            # (ba,)
        # ln(s2_a/s2_b) summed over d = sum_log_s2a - sum_log_s2b
        ln_term = a_logsum[:, None] - sum_log_s2b[None, :]   # (ba,Nb)
        # trace term: sum_d s2_b / s2_a  = a_inv (ba,D) . s2_b^T (D,Nb)
        tr_term = a_inv @ s2_b.T                             # (ba,Nb)
        # mean term: sum_d (mu_b-mu_a)^2 / s2_a
        #   = sum_d mu_b^2/s2_a - 2 sum_d mu_a*mu_b/s2_a + sum_d mu_a^2/s2_a
        m1 = a_inv @ (mu_b**2).T                             # (ba,Nb)
        m2 = (a_inv * a_mu) @ mu_b.T                         # (ba,Nb)
        m3 = (a_inv * a_mu * a_mu).sum(axis=1)               # (ba,)
        mean_term = m1 - 2.0 * m2 + m3[:, None]              # (ba,Nb)
        out[i:i+block] = 0.5 * (ln_term + tr_term + mean_term - D)
    return out


def kl_g_into_a(mu_a, s2_a, mu_g, s2_g):
    """CORRECT direction for A->G: D = KL(N_g || N_a), shape (Nq=Na, Ng).
    rows = aerial (wide envelope, the 'a' in _kl_diag), cols = ground (narrow)."""
    return _kl_diag(mu_a, s2_a, mu_g, s2_g)                  # (Na, Ng) = KL(g||a)


def kl_a_into_g(mu_a, s2_a, mu_g, s2_g):
    """REVERSED direction: D = KL(N_a || N_g), shape (Nq=Na, Ng).
    Must be MUCH worse than correct if the re-frame holds."""
    return _kl_diag(mu_g, s2_g, mu_a, s2_a).T               # _kl_diag rows=g -> .T to (Na,Ng)


def sym_kl(mu_a, s2_a, mu_g, s2_g):
    return 0.5 * (kl_g_into_a(mu_a, s2_a, mu_g, s2_g) + kl_a_into_g(mu_a, s2_a, mu_g, s2_g))


def js_div(mu_a, s2_a, mu_g, s2_g, block=24):
    """Jensen-Shannon between two diagonal Gaussians via mixture M=0.5(A+G).
    Closed-form JS is intractable; use the standard Gaussian-mixture approximation
    JS ~ 0.5*KL(A||M)+0.5*KL(G||M) where M is the moment-matched Gaussian of the
    0.5/0.5 mixture (mean = 0.5(mu_a+mu_g), var = 0.5(s2_a+s2_g)+0.25(mu_a-mu_g)^2).
    Returns (Na,Ng)."""
    Na = mu_a.shape[0]; Ng = mu_g.shape[0]; D = mu_a.shape[1]
    # keep each (block,Ng,D) float64 temp <= ~0.6GB (several live at once)
    block = max(1, min(block, int(0.6e9 / max(1, Ng * D * 8))))
    out = np.empty((Na, Ng), dtype=np.float64)
    for i in range(0, Na, block):
        a_mu = mu_a[i:i+block][:, None, :]                  # (ba,1,D)
        a_s2 = s2_a[i:i+block][:, None, :]
        g_mu = mu_g[None, :, :]                             # (1,Ng,D)
        g_s2 = s2_g[None, :, :]
        m_mu = 0.5 * (a_mu + g_mu)
        m_s2 = 0.5 * (a_s2 + g_s2) + 0.25 * (a_mu - g_mu) ** 2
        # KL(A||M) and KL(G||M), summed over d
        def _kl(p_mu, p_s2, q_mu, q_s2):
            return 0.5 * (np.log(q_s2 / p_s2) + (p_s2 + (p_mu - q_mu) ** 2) / q_s2 - 1.0).sum(-1)
        out[i:i+block] = 0.5 * _kl(a_mu, a_s2, m_mu, m_s2) + 0.5 * _kl(g_mu, g_s2, m_mu, m_s2)
    return out


def bhattacharyya(mu_a, s2_a, mu_g, s2_g, block=24):
    """Bhattacharyya distance between two diagonal Gaussians (symmetric). (Na,Ng).
    D_B = 1/8 (mu_a-mu_g)^T S^-1 (mu_a-mu_g) + 1/2 ln( |S| / sqrt(|s2_a||s2_g|) ),
    S = 0.5(s2_a+s2_g) diagonal."""
    Na = mu_a.shape[0]; Ng = mu_g.shape[0]; D = mu_a.shape[1]
    block = max(1, min(block, int(0.6e9 / max(1, Ng * D * 8))))
    out = np.empty((Na, Ng), dtype=np.float64)
    log_s2a = np.log(s2_a).sum(1); log_s2g = np.log(s2_g).sum(1)
    for i in range(0, Na, block):
        a_mu = mu_a[i:i+block][:, None, :]
        a_s2 = s2_a[i:i+block][:, None, :]
        g_mu = mu_g[None, :, :]; g_s2 = s2_g[None, :, :]
        S = 0.5 * (a_s2 + g_s2)
        term1 = 0.125 * (((a_mu - g_mu) ** 2) / S).sum(-1)
        logS = np.log(S).sum(-1)                            # (ba,Ng)
        term2 = 0.5 * (logS - 0.5 * (log_s2a[i:i+block][:, None] + log_s2g[None, :]))
        out[i:i+block] = term1 + term2
    return out


def equal_var_maha(mu_a, s2_a, mu_g, s2_g):
    """Equal-variance Mahalanobis: set ALL sigma to a single global constant
    (mean of all variances), leaving only the distance FORM. -> reduces to scaled
    Euclidean on mu. Should behave ~like Euclidean/cosine (no containment)."""
    s2_const = float(np.concatenate([s2_a.mean(0), s2_g.mean(0)]).mean())
    # Mahalanobis^2 with isotropic S = s2_const*I : ||mu_a-mu_g||^2 / s2_const
    qn = mu_a; gn = mu_g
    q2 = (qn ** 2).sum(1)[:, None]; g2 = (gn ** 2).sum(1)[None, :]
    d2 = q2 + g2 - 2.0 * (qn @ gn.T)
    return np.maximum(d2, 0.0) / s2_const


# --------------------------------------------------------------------------- #
# evaluation: CARGO mAP / Rank-1 / mINP with same-pid&cam junk removal
# --------------------------------------------------------------------------- #
def eval_dist(distmat, q_pids, q_cams, g_pids, g_cams, max_rank=20):
    """fast-reid / market style: remove gallery items with same (pid,cam) as query."""
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)                   # ascending distance
    matches = (g_pids[indices] == q_pids[:, None]).astype(np.int32)
    all_AP, all_cmc, all_INP = [], [], []
    num_valid_q = 0
    for i in range(num_q):
        order = indices[i]
        remove = (g_pids[order] == q_pids[i]) & (g_cams[order] == q_cams[i])
        keep = ~remove
        m = matches[i][keep]
        if not np.any(m):
            continue
        num_valid_q += 1
        cmc = m.cumsum()
        pos_idx = np.where(m == 1)[0]
        max_pos = pos_idx[-1]
        inp = cmc[max_pos] / (max_pos + 1.0)
        all_INP.append(inp)
        cmc_clip = cmc.copy(); cmc_clip[cmc_clip > 1] = 1
        all_cmc.append(cmc_clip[:max_rank])
        num_rel = m.sum()
        tmp = m.cumsum()
        prec = tmp / (np.arange(len(m)) + 1.0)
        AP = (prec * m).sum() / num_rel
        all_AP.append(AP)
    if num_valid_q == 0:
        return dict(mAP=0.0, r1=0.0, r5=0.0, mINP=0.0, nq=0)
    all_cmc = np.asarray(all_cmc).mean(0)
    return dict(mAP=float(np.mean(all_AP)) * 100,
                r1=float(all_cmc[0]) * 100,
                r5=float(all_cmc[4]) * 100 if len(all_cmc) > 4 else float('nan'),
                mINP=float(np.mean(all_INP)) * 100, nq=num_valid_q)


def per_query_ap(distmat, q_pids, q_cams, g_pids, g_cams):
    """per-query AP (with same pid&cam junk removal), -1 if no valid pos. For buckets."""
    num_q = distmat.shape[0]
    indices = np.argsort(distmat, axis=1)
    aps = np.full(num_q, -1.0)
    for i in range(num_q):
        order = indices[i]
        remove = (g_pids[order] == q_pids[i]) & (g_cams[order] == q_cams[i])
        keep = ~remove
        m = (g_pids[order][keep] == q_pids[i]).astype(np.int32)
        if not np.any(m):
            continue
        prec = m.cumsum() / (np.arange(len(m)) + 1.0)
        aps[i] = (prec * m).sum() / m.sum()
    return aps


def area_of(path, _cache={}):
    if path not in _cache:
        try:
            w, h = Image.open(path).size
            _cache[path] = float(h * w)
        except Exception:
            _cache[path] = -1.0
    return _cache[path]


def spearman(x, y):
    """Spearman rho via ranks (no scipy dependency)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float('nan')


# =========================================================================== #
# MAIN
# =========================================================================== #
def main():
    K = cli.K
    q_aer = filter_by_view(ds.query, 'Aerial')
    g_grd = filter_by_view(ds.gallery, 'Ground')
    q_grd = filter_by_view(ds.query, 'Ground')
    g_aer = filter_by_view(ds.gallery, 'Aerial')
    if cli.smoke > 0:
        q_aer = q_aer[:cli.smoke]; q_grd = q_grd[:cli.smoke]
        # cap gallery too so smoke is fast but still A vs G cross-view
        g_grd = g_grd[:min(len(g_grd), 4000)]
        g_aer = g_aer[:min(len(g_aer), 4000)]
        print(f"[SMOKE] q_aer={len(q_aer)} g_grd={len(g_grd)} q_grd={len(q_grd)} g_aer={len(g_aer)}")

    print(f"\n=== sizes: q_aer={len(q_aer)} g_grd={len(g_grd)} | q_grd={len(q_grd)} g_aer={len(g_aer)} ===")
    print(f"=== extracting mu/sigma with K={K} TTA views (frozen, no_grad) ===")

    t0 = time.time()
    mu_qa, s2_qa = extract_mu_sigma(q_aer, K, tag='q_aer')
    mu_gg, s2_gg = extract_mu_sigma(g_grd, K, tag='g_grd')
    mu_qg, s2_qg = extract_mu_sigma(q_grd, K, tag='q_grd')
    mu_ga, s2_ga = extract_mu_sigma(g_aer, K, tag='g_aer')
    print(f"[extract done] {time.time()-t0:.0f}s")

    qa_pid = np.array([s['pid'] for s in q_aer]); qa_cam = np.array([s['camid'] for s in q_aer])
    gg_pid = np.array([s['pid'] for s in g_grd]); gg_cam = np.array([s['camid'] for s in g_grd])
    qg_pid = np.array([s['pid'] for s in q_grd]); qg_cam = np.array([s['camid'] for s in q_grd])
    ga_pid = np.array([s['pid'] for s in g_aer]); ga_cam = np.array([s['camid'] for s in g_aer])

    # ---- helper to run a named scorer for both directions ------------------ #
    def report(name, dm_AG, dm_GA):
        rA = eval_dist(dm_AG, qa_pid, qa_cam, gg_pid, gg_cam)
        rG = eval_dist(dm_GA, qg_pid, qg_cam, ga_pid, ga_cam)
        print(f"  {name:28s} | A->G mAP {rA['mAP']:6.2f} R1 {rA['r1']:6.2f} mINP {rA['mINP']:5.2f}"
              f"  || G->A mAP {rG['mAP']:6.2f} R1 {rG['r1']:6.2f} mINP {rG['mINP']:5.2f}")
        return rA, rG

    # ================= MAIN COMPARISON (5 scorers x 2 directions) =========== #
    print("\n########## MAIN COMPARISON (mu/sigma shared, same eval pipeline) ##########")
    # 1. cosine (strongest symmetric baseline)
    report("1.cosine(mu)", cosine_dist(mu_qa, mu_gg), cosine_dist(mu_qg, mu_ga))
    # 2a. sym-KL
    report("2a.sym-KL", sym_kl(mu_qa, s2_qa, mu_gg, s2_gg), sym_kl(mu_qg, s2_qg, mu_ga, s2_ga))
    # 2b. JS
    report("2b.JS", js_div(mu_qa, s2_qa, mu_gg, s2_gg), js_div(mu_qg, s2_qg, mu_ga, s2_ga))
    # 2c. Bhattacharyya
    report("2c.Bhattacharyya", bhattacharyya(mu_qa, s2_qa, mu_gg, s2_gg),
           bhattacharyya(mu_qg, s2_qg, mu_ga, s2_ga))
    # 3. CORRECT direction: KL(g||a) for A->G ; for G->A the 'wide envelope' is the
    #    GALLERY aerial, so D = KL(N_query_ground || N_gallery_aerial) per item.
    dm_AG_correct = kl_g_into_a(mu_qa, s2_qa, mu_gg, s2_gg)             # KL(g||a), a=aerial query
    #    G->A: query=ground(narrow), gallery=aerial(wide) -> KL(N_q_ground || N_g_aerial)
    dm_GA_correct = _kl_diag(mu_ga, s2_ga, mu_qg, s2_qg).T             # rows=gallery aerial -> .T to (Nq_grd,Ng_aer)
    report("3.KL(g||a) CORRECT", dm_AG_correct, dm_GA_correct)
    # 4. REVERSED direction
    dm_AG_rev = kl_a_into_g(mu_qa, s2_qa, mu_gg, s2_gg)
    dm_GA_rev = _kl_diag(mu_qg, s2_qg, mu_ga, s2_ga)                   # KL(N_g_aerial || N_q_ground), reversed
    report("4.KL(a||g) REVERSED", dm_AG_rev, dm_GA_rev)
    # 5. equal-variance Mahalanobis
    report("5.equal-var Maha", equal_var_maha(mu_qa, s2_qa, mu_gg, s2_gg),
           equal_var_maha(mu_qg, s2_qg, mu_ga, s2_ga))

    # =================== 8 DESTRUCTIVE CONTROLS (A->G mAP) ================== #
    print("\n########## 8 DESTRUCTIVE CONTROLS (A->G; each must DROP vs CORRECT) ##########")
    base = eval_dist(dm_AG_correct, qa_pid, qa_cam, gg_pid, gg_cam)['mAP']
    cos_base = eval_dist(cosine_dist(mu_qa, mu_gg), qa_pid, qa_cam, gg_pid, gg_cam)['mAP']
    print(f"  reference: CORRECT KL(g||a) A->G mAP = {base:.2f} ; cosine = {cos_base:.2f}")

    def ag_map(dm):
        return eval_dist(dm, qa_pid, qa_cam, gg_pid, gg_cam)['mAP']

    # C1. direction destruction (reversed) -- already have dm_AG_rev
    print(f"  C1 direction-destroy (reversed KL(a||g)) : mAP {ag_map(dm_AG_rev):6.2f}  (vs {base:.2f})")
    # C2. symmetrization destruction (sym-KL / JS)
    print(f"  C2 symmetrize (sym-KL)                   : mAP {ag_map(sym_kl(mu_qa,s2_qa,mu_gg,s2_gg)):6.2f}")
    print(f"  C2 symmetrize (JS)                       : mAP {ag_map(js_div(mu_qa,s2_qa,mu_gg,s2_gg)):6.2f}")
    # C3. view-level variance: replace ALL aerial sigma by the per-dim aerial MEAN
    # variance vector, ground by ground mean (= a single view-level constant per view
    # -> destroys image-level sigma, keeps only the view prior).
    s2_qa_vm = np.repeat(s2_qa.mean(0, keepdims=True), len(s2_qa), 0)
    s2_gg_vm = np.repeat(s2_gg.mean(0, keepdims=True), len(s2_gg), 0)
    print(f"  C3 view-mean sigma (per-view constant)   : mAP {ag_map(kl_g_into_a(mu_qa,s2_qa_vm,mu_gg,s2_gg_vm)):6.2f}")
    # C4. within-view sigma permutation (shuffle sigma rows within each view)
    rng = np.random.RandomState(cli.seed)
    pa = rng.permutation(len(s2_qa)); pg = rng.permutation(len(s2_gg))
    print(f"  C4 within-view sigma permute             : mAP {ag_map(kl_g_into_a(mu_qa,s2_qa[pa],mu_gg,s2_gg[pg])):6.2f}")
    # C5. hardness-matched permutation: bucket by feature-norm, permute sigma WITHIN buckets
    def hardness_permute(mu, s2, nb=10):
        norm = np.linalg.norm(mu, axis=1)
        order = np.argsort(norm)
        out = s2.copy()
        bins = np.array_split(order, nb)
        for b in bins:
            out[b] = s2[b][rng.permutation(len(b))]
        return out
    s2_qa_h = hardness_permute(mu_qa, s2_qa); s2_gg_h = hardness_permute(mu_gg, s2_gg)
    print(f"  C5 hardness-matched permute (norm-bucket): mAP {ag_map(kl_g_into_a(mu_qa,s2_qa_h,mu_gg,s2_gg_h)):6.2f}")
    # C6. per-image dimension shuffle of sigma
    def dim_shuffle(s2):
        out = np.empty_like(s2)
        for i in range(len(s2)):
            out[i] = s2[i][rng.permutation(s2.shape[1])]
        return out
    print(f"  C6 per-image dim-shuffle sigma           : mAP {ag_map(kl_g_into_a(mu_qa,dim_shuffle(s2_qa),mu_gg,dim_shuffle(s2_gg))):6.2f}")
    # C7. variance-only / norm-only baselines (score by trace(sigma) or feature-norm alone)
    #     variance-only: distance = |trace(s2_g) - trace(s2_a)| broadcast; or rank gallery by trace(s2_g).
    tr_qa = s2_qa.sum(1); tr_gg = s2_gg.sum(1)
    dm_var_only = np.abs(tr_qa[:, None] - tr_gg[None, :])
    print(f"  C7 variance-only (|trace diff|)          : mAP {ag_map(dm_var_only):6.2f}")
    nrm_qa = np.linalg.norm(mu_qa, axis=1); nrm_gg = np.linalg.norm(mu_gg, axis=1)
    dm_norm_only = np.abs(nrm_qa[:, None] - nrm_gg[None, :])
    print(f"  C7 norm-only (|norm diff|)               : mAP {ag_map(dm_norm_only):6.2f}")
    # C8 is the bucket concentration (reported below).

    # =================== SYNTHETIC DEGRADATION POSITIVE CONTROL ============= #
    print("\n########## SYNTHETIC DEGRADATION (ground sigma must MONOTONICALLY rise) ##########")
    # re-extract ground-query sigma under downsample x4 and gaussian blur
    def deg_downsample(factor):
        def f(img):
            w, h = img.size
            small = img.resize((max(1, w // factor), max(1, h // factor)),
                               Image.BILINEAR)
            return small.resize((w, h), Image.BILINEAR)
        return f
    def deg_blur(radius):
        from PIL import ImageFilter
        def f(img):
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return f
    # use a subset of ground gallery for speed
    sub = g_grd[:min(len(g_grd), 600)]
    _, s2_clean = extract_mu_sigma(sub, K, tag='deg_clean')
    _, s2_ds2 = extract_mu_sigma(sub, K, degrade=deg_downsample(2), tag='deg_ds2')
    _, s2_ds4 = extract_mu_sigma(sub, K, degrade=deg_downsample(4), tag='deg_ds4')
    _, s2_blur = extract_mu_sigma(sub, K, degrade=deg_blur(3.0), tag='deg_blur')
    print(f"  trace(sigma) ground   clean={s2_clean.sum(1).mean():.4f}  "
          f"down2={s2_ds2.sum(1).mean():.4f}  down4={s2_ds4.sum(1).mean():.4f}  "
          f"blur={s2_blur.sum(1).mean():.4f}")
    mono = (s2_clean.sum(1).mean() <= s2_ds2.sum(1).mean() <= s2_ds4.sum(1).mean())
    print(f"  monotonic up (clean<=down2<=down4)? {mono}")

    # =================== BUCKETS (C8: gain concentration) ================== #
    print("\n########## BUCKETS by aerial-query bbox area (A->G; cosine vs CORRECT) ##########")
    aps_cos = per_query_ap(cosine_dist(mu_qa, mu_gg), qa_pid, qa_cam, gg_pid, gg_cam)
    aps_cont = per_query_ap(dm_AG_correct, qa_pid, qa_cam, gg_pid, gg_cam)
    q_area = np.array([area_of(s['img_path']) for s in q_aer])
    valid = aps_cos >= 0
    qa_idx = np.where(valid)[0]
    if len(qa_idx) >= 4:
        qtiles = np.quantile(q_area[valid], [0.25, 0.5, 0.75])
        edges = [-np.inf] + list(qtiles) + [np.inf]
        print(f"  bbox-area quartile edges: {qtiles}")
        for bi in range(4):
            sel = valid & (q_area > edges[bi]) & (q_area <= edges[bi + 1])
            if sel.sum() == 0:
                continue
            mc = 100 * 0  # placeholder
            cos_m = aps_cos[sel].mean() * 100
            cont_m = aps_cont[sel].mean() * 100
            print(f"  bucket{bi} area<= {edges[bi+1]:>10.0f} (n={sel.sum():3d}): "
                  f"cosine {cos_m:6.2f}  CORRECT {cont_m:6.2f}  delta {cont_m-cos_m:+6.2f}")
    print("  (expectation: SMALLEST area bucket -> largest containment gain)")

    # =================== DIAGNOSTICS ======================================= #
    print("\n########## DIAGNOSTICS ##########")
    tr_qa_m = s2_qa.sum(1).mean(); tr_gg_m = s2_gg.sum(1).mean()
    tr_ga_m = s2_ga.sum(1).mean(); tr_qg_m = s2_qg.sum(1).mean()
    print(f"  trace(sigma) AERIAL  q={tr_qa_m:.4f} g={tr_ga_m:.4f}  |  "
          f"GROUND q={tr_qg_m:.4f} g={tr_gg_m:.4f}")
    print(f"  -> aerial>ground (query side)? {tr_qa_m > tr_qg_m}  (gallery side)? {tr_ga_m > tr_gg_m}")
    rho = spearman(q_area, s2_qa.sum(1))
    print(f"  Spearman(aerial-query bbox-area, trace sigma) = {rho:+.3f}  (expect NEGATIVE)")

    # coverage calibration: same-ID ground positive vs hard-negative, fraction inside
    # the aerial query's Gaussian envelope at 50/80/95% per-dim coverage.
    print("\n  -- coverage calibration (aerial query envelope; ground positive vs hard-neg) --")
    # z-score of each candidate's mu under the aerial query Gaussian, per dim,
    # then fraction of dims within k-sigma; threshold k for 50/80/95% chi-square.
    # Simpler & robust: per-pair, average per-dim coverage prob, compare pos vs neg.
    from math import erf, sqrt
    def frac_within(mu_a, s2_a, mu_c, ks):
        """fraction of dims with |mu_c-mu_a|/sigma_a <= k, for each k in ks. (avg over dims)"""
        z = np.abs(mu_c - mu_a) / np.sqrt(s2_a)
        return {k: float((z <= k).mean()) for k in ks}
    ks = {0.674: '50%', 1.282: '80%', 1.960: '95%'}
    # pick first 200 aerial queries that have a positive in ground gallery
    pos_cov = {k: [] for k in ks}; neg_cov = {k: [] for k in ks}
    cnt = 0
    for i in range(len(q_aer)):
        pos_mask = (gg_pid == qa_pid[i]) & (gg_cam != qa_cam[i])
        neg_mask = (gg_pid != qa_pid[i])
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        # nearest positive (by cosine) and a hard negative (most similar wrong)
        cd = 1 - (mu_gg @ mu_qa[i]) / (np.linalg.norm(mu_gg, axis=1) * np.linalg.norm(mu_qa[i]) + 1e-12)
        pj = np.where(pos_mask)[0][np.argmin(cd[pos_mask])]
        nj = np.where(neg_mask)[0][np.argmin(cd[neg_mask])]
        fp = frac_within(mu_qa[i], s2_qa[i], mu_gg[pj], list(ks))
        fn = frac_within(mu_qa[i], s2_qa[i], mu_gg[nj], list(ks))
        for k in ks:
            pos_cov[k].append(fp[k]); neg_cov[k].append(fn[k])
        cnt += 1
        if cnt >= 300:
            break
    for k, lbl in ks.items():
        pm = np.mean(pos_cov[k]) if pos_cov[k] else float('nan')
        nm = np.mean(neg_cov[k]) if neg_cov[k] else float('nan')
        print(f"  coverage@{lbl} (k={k:.3f}): positive {pm:.3f}  hard-neg {nm:.3f}  "
              f"(positive should be HIGHER) [n={cnt}]")

    print("\n[done] kill-switch complete. See B_CONTAINMENT_DESIGN.md sec.4 pass criteria.")


if __name__ == '__main__':
    main()
