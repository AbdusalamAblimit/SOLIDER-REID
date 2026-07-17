#!/usr/bin/env python3
"""LM-ReID (Lattice-Marginalized ReID) — TRAINING  (exp359, Market, fine-tune exp260b).

HYPOTHESIS: a lattice-marginalized embedding (trained to be INVARIANT to the sampling
lattice — sub-pixel phase / +/-1 LR-pixel bbox / antialias kernel) BEATS the zero-training
frozen K-phase ensemble (the kill-switch's +4.23 mAP @ h=16). If the trained model clears
frozen-ensemble +0.8~2.0 @ h=16 -> it is a METHOD; if it only ~= the frozen ensemble ->
it is an ensemble trick (honest fail, report as such).

Design: experiments/exp359_lm_reid/design.md  (method-design codex, CCF-B 7/10).
Loss:   L = L_id + lam_marg*L_marg + lam_cons*L_cons + lam_adv*L_adv.
        L_id   = mean_l [ CE(cls^l, y) + Triplet(gf^l, y) ]            (per-variant ReID)
        L_marg = -log[ mean_l softmax(cls^l)[y] ] + Triplet(mean_l gf^l, y)  (marginal lik.)
        L_cons = mean_l (1-cos(z^l, sg(z_mu))) + beta*mean_l KL(p^l || sg(p_mu))  (lattice inv.)
        L_adv  = GRL: a discriminator that predicts the lattice-variant index from z is
                 reversed, so z carries NO predictable lattice label (weak, warmup-gated).

EVAL is done SEPARATELY (apples-to-apples, byte-identical to the GO kill-switch):
    cvpb_lattice_killswitch.py --ckpt <this output>/transformer_<ep>.pth
    -> compare the fine-tuned single / lat-mean / lat-max vs the FROZEN lat-MaxSim 46.87.

Backbone fine-tune uses pose_dict=None (POSE DISABLED, identical to the frozen baseline's
PSG-off global feat); all pose-conditional branches are skipped inside the model forward.

Run (lab-3090):
    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 \
      /root/miniconda3/envs/solider-reid/bin/python \
      experiments/cargo_cvpb/cvpb_lm_reid_train.py \
      --epochs 40 --out log/market1501/exp359_lm_reid 2>&1 | tee /tmp/exp359_lm_reid.log
    # smoke first:  --epochs 1 --smoke_ids 32 --workers 4
"""
import os, sys, time, argparse, random, math
import numpy as np
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, _repo)

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
ap.add_argument('--ckpt', default='log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth')
ap.add_argument('--data_root', default='data')
ap.add_argument('--out', default='log/market1501/exp359_lm_reid')
ap.add_argument('--heights', type=int, nargs='+', default=[16, 24, 32])
ap.add_argument('--height_p', type=float, nargs='+', default=[0.5, 0.3, 0.2],
                help='sampling prob per height (severe-biased)')
ap.add_argument('--M', type=int, default=3, help='#lattice variants/img/iter (1 canonical + M-1 random-axis)')
ap.add_argument('--aug_mode', default='lattice', choices=['lattice', 'ordinary'], help='lattice=sampling-lattice variants; ordinary=random crop/flip/color CONTROL (codex 命门: 是否 lattice-specific)')
ap.add_argument('--P', type=int, default=16, help='#ids per batch')
ap.add_argument('--Kins', type=int, default=4, help='#instances per id per batch (P*Kins=BS=64)')
ap.add_argument('--epochs', type=int, default=40)
ap.add_argument('--lr', type=float, default=3.5e-3)
ap.add_argument('--weight_decay', type=float, default=1e-4)
ap.add_argument('--warmup', type=int, default=5)
ap.add_argument('--lam_marg', type=float, default=1.0)
ap.add_argument('--lam_cons', type=float, default=0.2)
ap.add_argument('--loss_mode', default='std', choices=['std', 'hard', 'lsrc'], help='hard=Hard-Lattice ERM (dead); lsrc=Lattice-Set Retrieval Contrastive (bag-to-bag set-supcon + neg-tail suppression, shapes backbone for decision marginalization)')
ap.add_argument('--lam_lsrc', type=float, default=1.0, help='LSRC set-supcon weight')
ap.add_argument('--lam_negtail', type=float, default=1.0, help='LSRC negative lattice-pair tail suppression weight')
ap.add_argument('--lsrc_tau', type=float, default=0.1, help='LSRC lattice logsumexp temp')
ap.add_argument('--lsrc_tau_c', type=float, default=0.1, help='LSRC contrastive temp')
ap.add_argument('--lsrc_margin', type=float, default=0.4, help='LSRC neg-tail margin')
ap.add_argument('--lam_coverage', type=float, default=0.0, help='LSRC pos-coverage weight (top-M positive lattice-pair must win, codex step2; 0=off)')
ap.add_argument('--cov_m', type=int, default=3, help='LSRC pos-coverage top-M lattice pairs')
ap.add_argument('--lsrc_asym', action='store_true', help='codex High-fix: asymmetric set score (query LR M-variants vs gallery single canonical slot-0), aligns train to test q(LR-K)xg(HR-1)')
ap.add_argument('--gamma_hard', type=float, default=0.5, help='LM-S5 weight on the worst-case logsumexp term')
ap.add_argument('--tau_hard', type=float, default=1.0, help='LM-S5 temperature for the worst-case logsumexp (lower=harder)')
ap.add_argument('--beta_kl', type=float, default=0.5)
ap.add_argument('--lam_adv', type=float, default=0.0, help='0 disables L_adv (weak aux)')
ap.add_argument('--adv_start', type=int, default=10)
ap.add_argument('--margin', type=float, default=0.3)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--smoke_ids', type=int, default=0, help='cap #train ids for a fast smoke')
ap.add_argument('--smoke_iters', type=int, default=0, help='cap iters/epoch for smoke')
ap.add_argument('--save_every', type=int, default=20)
cli = ap.parse_args()

# Codex Critical-fix: loss_mode=='lsrc' isolates the train-side mechanism, but lam_marg(1.0)/
# lam_cons(0.2) defaults are non-zero and would silently contaminate the LSRC ablation with the
# already-falsified L_marg/L_cons. Force them to 0 so the isolation can't be forgotten on the CLI.
if cli.loss_mode == 'lsrc':
    if cli.lam_marg != 0.0 or cli.lam_cons != 0.0:
        print(f'[lsrc] force lam_marg={cli.lam_marg}->0 lam_cons={cli.lam_cons}->0 (isolate LSRC, codex Critical)')
    cli.lam_marg = 0.0; cli.lam_cons = 0.0
assert cli.cov_m >= 1, 'cov_m must be >=1 else topk(...,0).mean() is NaN (codex Low)'

random.seed(cli.seed); np.random.seed(cli.seed)
SIZE_TEST = (384, 128)                       # (H, W) model input / HR canvas
PIXEL_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
PIXEL_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_KERNELS = {'bicubic': Image.BICUBIC, 'bilinear': Image.BILINEAR, 'lanczos': Image.LANCZOS,
            'box': Image.BOX, 'hamming': Image.HAMMING, 'nearest': Image.NEAREST}

# =========================================================================== #
# data list (Market train; relabel pids to 0..N-1)
# =========================================================================== #
import re, glob
_PAT = re.compile(r'([-\d]+)_c(\d)')


def list_train(dir_path):
    raw = []
    pids = set()
    for p in sorted(glob.glob(os.path.join(dir_path, '*.jpg'))):
        pid, cam = map(int, _PAT.search(p).groups())
        if pid == -1:
            continue
        raw.append([p, pid, cam - 1]); pids.add(pid)
    pid2lbl = {pid: i for i, pid in enumerate(sorted(pids))}
    items = [[p, pid2lbl[pid], cam] for (p, pid, cam) in raw]
    return items, len(pids)


# =========================================================================== #
# LR + lattice variant generation (COPIED verbatim from cvpb_lattice_killswitch.py
# so the training-time degradation is byte-identical to the GO kill-switch eval).
# =========================================================================== #
def _to_target_aspect(img):
    return img.resize((SIZE_TEST[1], SIZE_TEST[0]), Image.BICUBIC)


def make_lr(hr_img, h, kernel='bicubic'):
    w = max(1, int(round(h * SIZE_TEST[1] / SIZE_TEST[0])))
    small = hr_img.resize((w, h), _KERNELS[kernel])
    return small.resize((SIZE_TEST[1], SIZE_TEST[0]), Image.BICUBIC)


def make_lattice_variants(hr_img, h, K, rng, rand_mode=False):
    """K plausible PHASE/CROP/KERNEL variants of the SAME hr image at height h.
    variant 0 = canonical deterministic bicubic LR (single-LR baseline)."""
    W_hr, H_hr = hr_img.size
    hr_per_lr_y = H_hr / float(h)
    hr_per_lr_x = W_hr / float(max(1, round(h / 3.0)))
    variants = [make_lr(hr_img, h, 'bicubic')]
    axes = [0]                          # 0=canonical; non-canonical: 1=phase, 2=bbox, 3=zoom
    kernels_cycle = ['bicubic', 'bilinear', 'lanczos', 'box', 'hamming']
    for j in range(1, K):
        # rand_mode (TRAINING): random axis+kernel per variant so even small M covers ALL
        # lattice axes across epochs; eval (kill-switch) keeps deterministic round-robin j%3.
        mode = int(rng.randint(0, 3)) if rand_mode else (j % 3)
        kern = (kernels_cycle[int(rng.randint(0, len(kernels_cycle)))] if rand_mode
                else kernels_cycle[j % len(kernels_cycle)])
        if mode == 0:
            dx = rng.uniform(-0.5, 0.5) * hr_per_lr_x
            dy = rng.uniform(-0.5, 0.5) * hr_per_lr_y
            shifted = hr_img.transform((W_hr, H_hr), Image.AFFINE, (1, 0, dx, 0, 1, dy),
                                       resample=Image.BICUBIC)
            v = make_lr(shifted, h, kern)
        elif mode == 1:
            sx = int(round(rng.choice([-1, 0, 1]) * hr_per_lr_x))
            sy = int(round(rng.choice([-1, 0, 1]) * hr_per_lr_y))
            left = max(0, sx); upper = max(0, sy)
            right = W_hr + min(0, sx); lower = H_hr + min(0, sy)
            if right - left < 4 or lower - upper < 4:
                left, upper, right, lower = 0, 0, W_hr, H_hr
            cropped = hr_img.crop((left, upper, right, lower)).resize((W_hr, H_hr), Image.BICUBIC)
            v = make_lr(cropped, h, kern)
        else:
            ez = rng.choice([-1, 1]) * 0.5 * hr_per_lr_y
            box = (-ez, -ez * (W_hr / H_hr), W_hr + ez, H_hr + ez * (W_hr / H_hr)) if ez > 0 \
                else (abs(ez), abs(ez) * (W_hr / H_hr), W_hr - abs(ez), H_hr - abs(ez) * (W_hr / H_hr))
            l, u, r, b = (int(round(v_)) for v_ in box)
            pad = max(0, -l, -u, r - W_hr, b - H_hr) + 1
            canvas = Image.new('RGB', (W_hr + 2 * pad, H_hr + 2 * pad), (0, 0, 0))
            canvas.paste(hr_img, (pad, pad))
            cropped = canvas.crop((l + pad, u + pad, r + pad, b + pad)).resize((W_hr, H_hr), Image.BICUBIC)
            v = make_lr(cropped, h, kern)
        variants.append(v)
        axes.append(mode + 1)           # 1=phase, 2=bbox, 3=zoom
    return variants, axes


def make_ordinary_variants(hr_img, h, K, rng):
    """CONTROL (codex 命门对照): K ORDINARY-augmentation variants of the canonical LR — NOT
    lattice. variant 0 = canonical bicubic LR; 1+ = random pad-crop / hflip / resize-jitter /
    color jitter. Same #views/h/compute as the lattice path → isolates whether the M=3 gain is
    lattice-SPECIFIC or just 'more augmentation'."""
    from PIL import ImageEnhance
    canon = make_lr(hr_img, h, 'bicubic')
    W, H = canon.size
    variants = [canon]; axes = [0]
    for j in range(1, K):
        v = canon
        pad = int(rng.randint(4, 13))
        canvas = Image.new('RGB', (W + 2 * pad, H + 2 * pad), (0, 0, 0))
        canvas.paste(v, (pad, pad))
        cx = int(rng.randint(0, 2 * pad + 1)); cy = int(rng.randint(0, 2 * pad + 1))
        v = canvas.crop((cx, cy, cx + W, cy + H))
        if rng.rand() < 0.5:
            v = v.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.rand() < 0.5:
            s = rng.uniform(0.8, 1.0)
            v = v.resize((max(1, int(W * s)), max(1, int(H * s))), Image.BILINEAR).resize((W, H), Image.BILINEAR)
        if rng.rand() < 0.5:
            v = ImageEnhance.Brightness(v).enhance(rng.uniform(0.8, 1.2))
        if rng.rand() < 0.5:
            v = ImageEnhance.Contrast(v).enhance(rng.uniform(0.8, 1.2))
        variants.append(v); axes.append(0)
    return variants, axes


def pil_to_tensor_np(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - PIXEL_MEAN) / PIXEL_STD
    return arr.transpose(2, 0, 1)


# =========================================================================== #
# PK dataset: each item = ONE person image -> M lattice LR variants (random height).
# A PK batch sampler draws P ids x Kins instances; collate stacks [B, M, C, H, W].
# =========================================================================== #
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from datasets.bases import read_image


class LatticeTrainSet(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def set_epoch(self, ep):
        self.epoch = ep

    def __getitem__(self, idx):
        p, lbl, cam = self.items[idx]
        # rng MIXED WITH EPOCH so each image gets FRESH random lattice variants every epoch
        # (Codex High-1: seed+idx alone froze the variants identically across all epochs).
        seed = (cli.seed * 1000003 + getattr(self, 'epoch', 0) * 9973 + idx * 2654435761) % (2**32)
        rng = np.random.RandomState(seed)
        h = int(rng.choice(cli.heights, p=np.array(cli.height_p) / np.sum(cli.height_p)))
        hr = _to_target_aspect(read_image(p))
        # rand_mode=True: random axis per non-canonical variant (Codex High-2). aug_mode=ordinary
        # is the codex 命门 control: same M views/h but random crop/flip/color, NOT lattice.
        if getattr(cli, 'aug_mode', 'lattice') == 'ordinary':
            vs, axes = make_ordinary_variants(hr, h, cli.M, rng)
        else:
            vs, axes = make_lattice_variants(hr, h, cli.M, rng, rand_mode=True)
        t = np.stack([pil_to_tensor_np(v) for v in vs], 0)   # [M,3,H,W]
        return torch.from_numpy(t), int(lbl), torch.tensor(axes, dtype=torch.long)


class PKSampler(Sampler):
    """Yield flat indices in P-id x Kins-instance blocks (one 'batch' = P*Kins)."""
    def __init__(self, items, P, Kins, num_iters=None):
        self.P, self.Kins = P, Kins
        self.by_pid = {}
        for i, (_, lbl, _) in enumerate(items):
            self.by_pid.setdefault(lbl, []).append(i)
        self.pids = list(self.by_pid.keys())
        self.length = (len(items) // (P * Kins)) if num_iters is None else num_iters
        self._n_items = len(items)

    def __len__(self):
        return self.length * self.P * self.Kins

    def __iter__(self):
        flat = []
        for _ in range(self.length):
            chosen = random.sample(self.pids, self.P) if len(self.pids) >= self.P \
                else [random.choice(self.pids) for _ in range(self.P)]
            for pid in chosen:
                pool = self.by_pid[pid]
                if len(pool) >= self.Kins:
                    flat.extend(random.sample(pool, self.Kins))
                else:
                    flat.extend([random.choice(pool) for _ in range(self.Kins)])
        return iter(flat)


def collate(batch):
    ts = torch.stack([b[0] for b in batch], 0)               # [B,M,3,H,W]
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    axes = torch.stack([b[2] for b in batch], 0)             # [B,M] lattice-axis labels for L_adv
    return ts, ys, axes


# =========================================================================== #
# losses
# =========================================================================== #
def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = (x ** 2).sum(1, keepdim=True).expand(m, n)
    yy = (y ** 2).sum(1, keepdim=True).expand(n, m).t()
    d = xx + yy - 2 * x @ y.t()
    return d.clamp(min=1e-12).sqrt()


def batch_hard_triplet(feat, labels, margin):
    """standard batch-hard triplet (hardest pos / hardest neg) on euclidean feat."""
    d = euclidean_dist(feat, feat)
    N = labels.size(0)
    is_pos = labels[:, None].eq(labels[None, :])
    is_neg = ~is_pos
    # hardest positive: max same-id dist (diagonal is 0/same-id, included is fine)
    d_ap = (d * is_pos.float()).max(1)[0]
    # hardest negative: min diff-id dist
    d_an = (d + is_pos.float() * 1e9).min(1)[0]
    y = torch.ones_like(d_ap)
    return torch.nn.functional.margin_ranking_loss(d_an, d_ap, y, margin=margin)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


# =========================================================================== #
# model (TRAINABLE; pose disabled -> plain backbone global feat + classifier)
# =========================================================================== #
def build_trainable_model():
    from config import cfg
    from model import make_model
    from datasets.market1501 import Market1501
    cfg.merge_from_file(os.path.join(_repo, cli.config))
    cfg.merge_from_list([
        'MODEL.POSE_TEST_FEAT', 'global',
        'TEST.NECK_FEAT', 'after',
        'TEST.FEAT_NORM', 'yes',
    ])
    cfg.freeze()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    ds = Market1501(root=os.path.join(_repo, cli.data_root), verbose=False)
    model = make_model(cfg, num_class=ds.num_train_pids, camera_num=ds.num_train_cams,
                       view_num=1, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(os.path.join(_repo, cli.ckpt))
    print(f"[model] loaded {cli.ckpt}; num_cls={ds.num_train_pids}; pose DISABLED (pose_dict=None)",
          flush=True)
    return model.cuda(), ds.num_train_pids


def main():
    print("#" * 88)
    print("# LM-ReID TRAINING (exp359) — fine-tune exp260b for lattice-marginalized embedding")
    print("#" * 88)
    t_items, n_cls = list_train(os.path.join(_repo, cli.data_root, 'market1501', 'bounding_box_train'))
    if cli.smoke_ids > 0:
        keep = set(sorted({it[1] for it in t_items})[:cli.smoke_ids])
        t_items = [it for it in t_items if it[1] in keep]
        # relabel again to contiguous
        remap = {l: i for i, l in enumerate(sorted({it[1] for it in t_items}))}
        t_items = [[p, remap[l], c] for (p, l, c) in t_items]
        n_cls = len(remap)
    print(f"[data] #train_img={len(t_items)} #ids(for sampler)={len({it[1] for it in t_items})} "
          f"n_cls={n_cls}  M={cli.M} P={cli.P} Kins={cli.Kins} (BS={cli.P*cli.Kins}) heights={cli.heights}")

    model, _ = build_trainable_model()
    in_planes = model.bottleneck.weight.shape[0]
    print(f"[model] embedding dim (bottleneck) = {in_planes}")

    # lattice discriminator for L_adv (predict which of M variant slots): tiny MLP
    disc = torch.nn.Sequential(torch.nn.Linear(in_planes, 256), torch.nn.ReLU(inplace=True),
                               torch.nn.Linear(256, 4)).cuda()   # 4 axes: 0=canon,1=phase,2=bbox,3=zoom

    params = [p for p in model.parameters() if p.requires_grad] + list(disc.parameters())
    opt = torch.optim.SGD(params, lr=cli.lr, momentum=0.9, weight_decay=cli.weight_decay, nesterov=True)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler()    # AMP: B*M=128 Swin-base fwd+bwd would be tight in fp32

    ds = LatticeTrainSet(t_items)
    iters_per_epoch = (len(t_items) // (cli.P * cli.Kins))
    if cli.smoke_iters > 0:
        iters_per_epoch = min(iters_per_epoch, cli.smoke_iters)
    sampler = PKSampler(t_items, cli.P, cli.Kins, num_iters=iters_per_epoch)
    loader = DataLoader(ds, batch_size=cli.P * cli.Kins, sampler=sampler, num_workers=cli.workers,
                        collate_fn=collate, pin_memory=True, drop_last=True)

    os.makedirs(os.path.join(_repo, cli.out), exist_ok=True)

    def lr_at(ep):
        if ep < cli.warmup:
            return cli.lr * (ep + 1) / max(1, cli.warmup)
        prog = (ep - cli.warmup) / max(1, cli.epochs - cli.warmup)
        return 0.5 * cli.lr * (1 + math.cos(math.pi * prog))

    print(f"[train] iters/epoch={iters_per_epoch}  epochs={cli.epochs}  lr={cli.lr}", flush=True)
    for ep in range(cli.epochs):
        model.train()
        ds.set_epoch(ep)                  # FRESH random lattice variants each epoch (Codex High-1)
        for g in opt.param_groups:
            g['lr'] = lr_at(ep)
        adv_lamb = cli.lam_adv if (cli.lam_adv > 0 and ep >= cli.adv_start) else 0.0
        agg = {k: 0.0 for k in ('L', 'id', 'marg', 'cons', 'adv', 'acc')}
        t0 = time.time()
        for it, (xb, yb, axb) in enumerate(loader):
            B, M = xb.shape[0], xb.shape[1]
            x = xb.view(B * M, *xb.shape[2:]).cuda(non_blocking=True)
            y = yb.cuda(non_blocking=True)
            y_rep = y.repeat_interleave(M)                    # [B*M]
            cam0 = torch.zeros(B * M, dtype=torch.long, device=x.device)
            with torch.cuda.amp.autocast():
                out = model(x, label=y_rep, cam_label=cam0, view_label=cam0, pose_dict=None)
            # pose-OFF training return is (cls_score, global_feat, featmaps, None); be robust to the
            # list form ([cls_score]+heads, [global_feat]+heads) in case a pose branch ever fires.
            # losses computed in fp32 (.float()) for numerical safety (log / KL underflow in fp16).
            cls = (out[0][0] if isinstance(out[0], (list, tuple)) else out[0]).float()   # [B*M, n_cls]
            gf = (out[1][0] if isinstance(out[1], (list, tuple)) else out[1]).float()    # [B*M, D]
            D = gf.shape[1]

            # ---- reshape to [B, M, .] ----
            cls_bm = cls.view(B, M, -1)
            gf_bm = gf.view(B, M, D)

            # ---- L_id (CE on all variants + PER-SLOT batch-hard triplet) ----
            # Codex Medium-4: triplet over flat B*M treats same-image variants as positives;
            # do triplet WITHIN each lattice slot (clean PK structure) and average over slots.
            L_ce = ce(cls, y_rep)
            L_tri = torch.stack([batch_hard_triplet(gf_bm[:, m], y, cli.margin)
                                 for m in range(M)]).mean()
            L_id = L_ce + L_tri
            # ---- LM-S5 Hard-Lattice ERM (codex): worst-case lattice variant emphasis ----
            # add gamma*tau*logsumexp_m(CE^m/tau): optimize the HARDEST lattice variant per
            # identity (robustness, NOT embedding collapse). Pair with --aug_mode ordinary for the
            # Hard-ordinary control; lattice must beat Hard-ordinary by >=+0.8 mAP to live.
            if cli.loss_mode == 'hard':
                ce_none = torch.nn.functional.cross_entropy(
                    cls_bm.reshape(B * M, -1), y_rep, reduction='none').view(B, M)   # [B,M]
                L_id = L_id + cli.gamma_hard * cli.tau_hard * torch.logsumexp(
                    ce_none / cli.tau_hard, dim=1).mean()

            # ---- LSRC (Lattice-Set Retrieval Contrastive): bag-to-bag set score shaped for
            # decision marginalization. set-supcon aligns training to test-time logsumexp marg;
            # neg-tail suppresses the false-high lattice-pair on negatives (the marg math
            # bottleneck). NOT consistency (no pull-to-mean), NOT L_marg (retrieval decision, not
            # classifier posterior -> explains why frozen reweight had no headroom). ----
            if cli.loss_mode == 'lsrc':
                zb = torch.nn.functional.normalize(gf_bm, dim=-1)               # [B,M,D]
                if cli.lsrc_asym:
                    # codex High-fix: asymmetric set score matches test q(LR K-variants) x g(HR/canonical single).
                    # gallery side uses ONLY slot-0 -> no gallery-side oracle, no test-absent worst-pair on negs.
                    zg = zb[:, 0]                                               # [B,D] canonical gallery
                    sim_src = torch.einsum('ikd,jd->ijk', zb, zg)              # [B,B,M] query-set vs gallery-single
                else:
                    sim_src = torch.einsum('ikd,jld->ijkl', zb, zb).reshape(B, B, -1)  # [B,B,M*M] symmetric bag-to-bag
                S = cli.lsrc_tau * torch.logsumexp(sim_src / cli.lsrc_tau, dim=2)  # [B,B] set score
                pos = (y[:, None] == y[None, :]).float(); pos.fill_diagonal_(0)
                logits = S / cli.lsrc_tau_c - 1e9 * torch.eye(B, device=x.device)
                logp = torch.log_softmax(logits, dim=1)
                L_setsupcon = (-(pos * logp).sum(1) / pos.sum(1).clamp_min(1)).mean()
                neg = (y[:, None] != y[None, :]).float()
                max_neg = sim_src.max(2)[0]                                     # [B,B] max lattice-pair cos
                L_negtail = (neg * torch.nn.functional.softplus(
                    (max_neg - cli.lsrc_margin) / 0.1)).sum() / neg.sum().clamp_min(1)
                L_cov = torch.zeros((), device=x.device)
                if cli.lam_coverage > 0:    # codex step2: positives win via top-M lattice pairs, not one fixed
                    topm = torch.topk(sim_src, min(cli.cov_m, sim_src.shape[2]), dim=2)[0].mean(2)  # [B,B]
                    L_cov = -(pos * topm).sum() / pos.sum().clamp_min(1)
                L_id = L_id + cli.lam_lsrc * (L_setsupcon + cli.lam_negtail * L_negtail
                                              + cli.lam_coverage * L_cov)

            # ---- L_marg (marginal likelihood + triplet on mean feat) ----
            p_bm = torch.softmax(cls_bm, dim=-1)             # [B,M,C]
            p_mean = p_bm.mean(1)                            # [B,C]
            ll = torch.log(p_mean.gather(1, y[:, None]).clamp_min(1e-8)).squeeze(1)  # [B]
            gf_mean = gf_bm.mean(1)                          # [B,D]
            L_marg = -ll.mean() + batch_hard_triplet(gf_mean, y, cli.margin)

            # ---- L_cons (lattice invariance: pull each variant to the mean) ----
            z = torch.nn.functional.normalize(gf_bm, dim=-1)            # [B,M,D]
            z_mu = torch.nn.functional.normalize(gf_bm.mean(1), dim=-1).detach()  # [B,D]
            cos_term = (1.0 - (z * z_mu[:, None, :]).sum(-1)).mean()
            # KL(p^l || sg(p_mu)) — forward KL (design): pull each variant's prediction to the mean.
            logp_l = torch.log_softmax(cls_bm, dim=-1)                  # [B,M,C] = log p^l
            log_pmu = torch.log(p_mean.detach().clamp_min(1e-8))[:, None, :]   # [B,1,C] = log sg(p_mu)
            kl_term = (p_bm * (logp_l - log_pmu)).sum(-1).mean()        # p_bm = softmax(cls_bm) = p^l
            L_cons = cos_term + cli.beta_kl * kl_term

            # ---- L_adv (GRL: remove predictable lattice-AXIS label from z) ----
            # Codex Medium-3: GRL already scales the reversed grad by adv_lamb -> do NOT also
            # weight L_adv by adv_lamb in the total (was lambda^2). Predict the REAL axis (axb),
            # not the slot index (now a random axis per slot), so the adversary label is meaningful.
            if adv_lamb > 0:
                axis_lbl = axb.to(x.device).reshape(B * M)             # [B*M] real lattice axis
                zr = GradReverse.apply(torch.nn.functional.normalize(gf, dim=-1), adv_lamb)
                L_adv = ce(disc(zr), axis_lbl)
            else:
                L_adv = torch.zeros((), device=x.device)

            loss = L_id + cli.lam_marg * L_marg + cli.lam_cons * L_cons + L_adv
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                acc = (cls.argmax(1) == y_rep).float().mean().item()
            agg['L'] += loss.item(); agg['id'] += L_id.item(); agg['marg'] += L_marg.item()
            agg['cons'] += L_cons.item(); agg['adv'] += float(L_adv.item()); agg['acc'] += acc
            if (it + 1) % 50 == 0 or cli.smoke_iters > 0:
                n = it + 1
                print(f"  ep{ep} it{n}/{iters_per_epoch} L={agg['L']/n:.3f} id={agg['id']/n:.3f} "
                      f"marg={agg['marg']/n:.3f} cons={agg['cons']/n:.3f} adv={agg['adv']/n:.3f} "
                      f"acc={agg['acc']/n:.3f} lr={lr_at(ep):.2e}", flush=True)
        n = max(1, it + 1)
        print(f"[epoch {ep}] L={agg['L']/n:.3f} id={agg['id']/n:.3f} marg={agg['marg']/n:.3f} "
              f"cons={agg['cons']/n:.3f} adv={agg['adv']/n:.3f} acc={agg['acc']/n:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if (ep + 1) % cli.save_every == 0 or (ep + 1) == cli.epochs:
            sp = os.path.join(_repo, cli.out, f'transformer_{ep+1}.pth')
            torch.save(model.state_dict(), sp)
            print(f"[save] {sp}", flush=True)
    print("[done] LM-ReID training complete. EVAL via cvpb_lattice_killswitch.py --ckpt <out>.")


if __name__ == '__main__':
    main()
