# encoding: utf-8
"""
AFD-ReID training loop on CARGO.

Baseline (default, --use_afd off):
    resnet50(IMAGENET1K_V1) + GeM + BNNeck
    loss = CE(label smoothing 0.1) + batch-hard triplet (margin 0.3)
    PK sampler P=16 x K=4 (bs=64), AdamW lr 3.5e-4, 10-ep warmup + cosine, 60 epochs.
    eval every 10 epochs: A->G and G->A cross-view mAP / Rank-1 / mINP.

AFD (--use_afd) adds the Frequency Reliability Router + Cross-View Frequency
Counterfactual losses on top; everything else identical for a clean ablation.

Run on lab-3090:
    cd /root/work/SOLIDER-REID/experiments/afd_reid
    PYTHONUNBUFFERED=1 python afd_train.py \
        --data_root /root/work/SOLIDER-REID/data \
        --out_dir   /root/work/SOLIDER-REID/log/cargo/afd_baseline \
        2>&1 | tee /tmp/afd_baseline.log
    # AFD variant: add  --use_afd
"""
import os
import sys
import time
import math
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cargo_dataset import (CARGO, CARGOImageDataset, build_transforms,
                           RandomIdentitySampler, filter_by_view)
from afd_model import build_model


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        return (-targets * log_probs).mean(0).sum()


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * (x @ y.t())
    return dist.clamp(min=1e-12).sqrt()


class TripletLoss(nn.Module):
    """Batch-hard triplet (Hermans et al.) with margin."""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        n = feats.size(0)
        dist = euclidean_dist(feats, feats)
        mask = labels.expand(n, n).eq(labels.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            pos = dist[i][mask[i]]
            neg = dist[i][mask[i] == 0]
            dist_ap.append(pos.max().unsqueeze(0))
            dist_an.append(neg.min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


# --------------------------------------------------------------------------- #
# LR schedule: linear warmup -> cosine
# --------------------------------------------------------------------------- #
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_factor=0.01,
                 last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_factor = warmup_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            alpha = e / max(1, self.warmup_epochs)
            factor = self.warmup_factor * (1 - alpha) + alpha
        else:
            prog = (e - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * prog))
        return [base_lr * factor for base_lr in self.base_lrs]


# --------------------------------------------------------------------------- #
# evaluation: market-style mAP / CMC / mINP with same-pid&cam junk removal
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_features(model, loader, device, use_afd):
    model.eval()
    feats, pids, camids = [], [], []
    view_map = {'Aerial': 0, 'Ground': 1}
    for batch in loader:
        imgs = batch['img'].to(device, non_blocking=True)
        vidx = torch.tensor([view_map[v] for v in batch['view']],
                            device=device) if use_afd else None
        f = model(imgs, view_idx=vidx)
        feats.append(f.cpu())
        pids.append(batch['pid'])
        camids.append(batch['camid'])
    if not feats:   # empty split (degenerate direction)
        return torch.empty(0), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    feats = torch.cat(feats, 0)
    pids = torch.cat(pids, 0).numpy()
    camids = torch.cat(camids, 0).numpy()
    return feats, pids, camids


def eval_market(qf, q_pids, q_camids, gf, g_pids, g_camids, max_rank=50):
    """Standard market1501 eval. Removes gallery items with same (pid,camid) as query.

    Returns (mAP, cmc[max_rank], mINP).
    """
    if qf.numel() == 0 or gf.numel() == 0:
        # degenerate direction (one view missing) -> report NaN, don't crash
        return float('nan'), np.full(max_rank, float('nan'), dtype=np.float32), float('nan')
    qf = F.normalize(qf, dim=1)
    gf = F.normalize(gf, dim=1)
    distmat = (2 - 2 * qf @ gf.t()).numpy()   # cosine distance
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc, all_AP, all_INP = [], [], []
    num_valid_q = 0
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        # junk: same pid AND same camid
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue   # this query has no ground-truth in (filtered) gallery
        cmc = raw_cmc.cumsum()

        # mINP
        pos_idx = np.where(raw_cmc == 1)[0]
        max_pos_idx = pos_idx[-1]
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc_clip = cmc.copy()
        cmc_clip[cmc_clip > 1] = 1
        all_cmc.append(cmc_clip[:max_rank])
        num_valid_q += 1

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    if num_valid_q == 0:
        return float('nan'), np.full(max_rank, float('nan'), dtype=np.float32), float('nan')
    all_cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q
    mAP = float(np.mean(all_AP))
    mINP = float(np.mean(all_INP))
    return mAP, all_cmc, mINP


def build_eval_loader(samples, args):
    tf = build_transforms(is_train=False, img_size=tuple(args.img_size))
    ds = CARGOImageDataset(samples, tf)
    return DataLoader(ds, batch_size=args.test_batch, shuffle=False,
                      num_workers=args.workers, pin_memory=True)


def run_cross_view_eval(model, dataset, args, device):
    """Evaluate both cross-view directions: A->G and G->A. Returns dict of metrics."""
    q_aerial = filter_by_view(dataset.query, 'Aerial')
    q_ground = filter_by_view(dataset.query, 'Ground')
    g_aerial = filter_by_view(dataset.gallery, 'Aerial')
    g_ground = filter_by_view(dataset.gallery, 'Ground')

    results = {}
    # A->G : aerial query vs ground gallery
    for tag, q, g in (('A->G', q_aerial, g_ground),
                      ('G->A', q_ground, g_aerial)):
        ql = build_eval_loader(q, args)
        gl = build_eval_loader(g, args)
        qf, qp, qc = extract_features(model, ql, device, args.use_afd)
        gf, gp, gc = extract_features(model, gl, device, args.use_afd)
        mAP, cmc, mINP = eval_market(qf, qp, qc, gf, gp, gc)
        results[tag] = {'mAP': mAP * 100, 'R1': cmc[0] * 100,
                        'R5': cmc[4] * 100 if len(cmc) > 4 else float('nan'),
                        'mINP': mINP * 100}
    return results


def print_eval(epoch, results):
    print(f"  ---- A<->G cross-view eval @ epoch {epoch} ----")
    for tag in ('A->G', 'G->A'):
        r = results[tag]
        print(f"    [{tag}] mAP={r['mAP']:.2f}  R1={r['R1']:.2f}  "
              f"R5={r['R5']:.2f}  mINP={r['mINP']:.2f}")
    mean_map = (results['A->G']['mAP'] + results['G->A']['mAP']) / 2
    mean_r1 = (results['A->G']['R1'] + results['G->A']['R1']) / 2
    print(f"    [mean] mAP={mean_map:.2f}  R1={mean_r1:.2f}")
    return mean_map


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
    ap.add_argument('--out_dir', default='./log/cargo/afd_baseline')
    ap.add_argument('--img_size', type=int, nargs=2, default=[256, 128])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--warmup_epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=3.5e-4)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--P', type=int, default=16)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--test_batch', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--margin', type=float, default=0.3)
    ap.add_argument('--label_smooth', type=float, default=0.1)
    ap.add_argument('--eval_period', type=int, default=10)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--no_amp', action='store_true', help='disable mixed precision')
    # model / AFD switches
    ap.add_argument('--last_stride', type=int, default=1)
    ap.add_argument('--pool', default='gem', choices=['gem', 'avg'])
    ap.add_argument('--use_afd', action='store_true')
    ap.add_argument('--afd_router', type=int, default=1)
    ap.add_argument('--afd_cvfc', type=int, default=1)
    ap.add_argument('--afd_stage', default='layer1',
                    choices=['stem', 'layer1', 'layer2'])
    ap.add_argument('--router_cond_view', type=int, default=1)
    ap.add_argument('--low_r', type=float, default=0.125)
    ap.add_argument('--mid_r', type=float, default=0.30)
    ap.add_argument('--high_drop_p', type=float, default=0.5)
    # AFD loss weights
    ap.add_argument('--w_cvfc', type=float, default=0.5,
                    help='weight of cross-view counterfactual consistency loss')
    args = ap.parse_args()
    # normalize int-bool flags
    args.afd_router = bool(args.afd_router)
    args.afd_cvfc = bool(args.afd_cvfc)
    args.router_cond_view = bool(args.router_cond_view)

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda'
    batch_size = args.P * args.K

    print("=" * 70)
    print("AFD-ReID training")
    print(f"  use_afd={args.use_afd} router={args.afd_router} cvfc={args.afd_cvfc} "
          f"stage={args.afd_stage} cond_view={args.router_cond_view}")
    print(f"  bs={batch_size} (P={args.P} K={args.K}) lr={args.lr} "
          f"epochs={args.epochs} warmup={args.warmup_epochs} amp={not args.no_amp}")
    print(f"  out_dir={args.out_dir}")
    print("=" * 70)

    # data
    dataset = CARGO(root=args.data_root, verbose=True)
    train_tf = build_transforms(is_train=True, img_size=tuple(args.img_size))
    train_set = CARGOImageDataset(dataset.train, train_tf)
    sampler = RandomIdentitySampler(dataset.train, batch_size, args.K)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True)

    # model
    model = build_model(dataset.num_train_pids, args).to(device)

    # losses
    ce = CrossEntropyLabelSmooth(dataset.num_train_pids, args.label_smooth)
    tri = TripletLoss(args.margin)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = WarmupCosineLR(optimizer, args.warmup_epochs, args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=not args.no_amp)

    view_map = {'Aerial': 0, 'Ground': 1}
    best_map = -1.0
    best_epoch = -1
    n_iter_total = len(train_loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        meters = {'loss': 0.0, 'ce': 0.0, 'tri': 0.0, 'cvfc': 0.0, 'acc': 0.0}
        seen = 0

        for it, batch in enumerate(train_loader):
            imgs = batch['img'].to(device, non_blocking=True)
            labels = batch['pid'].to(device, non_blocking=True)
            vidx = torch.tensor([view_map[v] for v in batch['view']],
                                device=device) if args.use_afd else None

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=not args.no_amp):
                out = model(imgs, view_idx=vidx,
                            return_cvfc=(args.use_afd and args.afd_cvfc))
                logits = out['logits']
                gfeat = out['global_feat']
                loss_ce = ce(logits, labels)
                loss_tri = tri(gfeat, labels)
                loss = loss_ce + loss_tri

                loss_cvfc = torch.zeros((), device=device)
                if args.use_afd and args.afd_cvfc and 'cf_lowpass_bn' in out:
                    bn = out['bn_feat'].detach()  # anchor on the real feature
                    # consistency: real <-> low-pass counterfactual,
                    #              real <-> high-band-dropout counterfactual
                    lp = out['cf_lowpass_bn']
                    hd = out['cf_highdrop_bn']
                    loss_cvfc = (F.mse_loss(F.normalize(lp, dim=1),
                                            F.normalize(bn, dim=1))
                                 + F.mse_loss(F.normalize(hd, dim=1),
                                              F.normalize(bn, dim=1)))
                    loss = loss + args.w_cvfc * loss_cvfc

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = imgs.size(0)
            seen += bs
            meters['loss'] += loss.item() * bs
            meters['ce'] += loss_ce.item() * bs
            meters['tri'] += loss_tri.item() * bs
            meters['cvfc'] += float(loss_cvfc) * bs
            meters['acc'] += (logits.argmax(1) == labels).float().sum().item()

            if (it + 1) % 50 == 0 or (it + 1) == n_iter_total:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch[{epoch}] Iter[{it + 1}/{n_iter_total}] "
                      f"Loss: {meters['loss'] / seen:.3f} "
                      f"CE: {meters['ce'] / seen:.3f} "
                      f"Tri: {meters['tri'] / seen:.3f} "
                      f"CVFC: {meters['cvfc'] / seen:.4f} "
                      f"Acc: {meters['acc'] / seen:.3f} LR: {lr:.2e}")

        scheduler.step()
        dt = time.time() - t0
        print(f"Epoch[{epoch}] done in {dt:.1f}s  "
              f"Loss={meters['loss'] / seen:.3f} Acc={meters['acc'] / seen:.3f}")

        # periodic eval + last epoch
        if epoch % args.eval_period == 0 or epoch == args.epochs:
            results = run_cross_view_eval(model, dataset, args, device)
            mean_map = print_eval(epoch, results)
            if mean_map > best_map:
                best_map = mean_map
                best_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(args.out_dir, 'model_best.pth'))
                print(f"    * new best mean mAP={best_map:.2f} (epoch {epoch}) saved")

    # always save final
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_final.pth'))
    print("=" * 70)
    print(f"Training finished. Best mean A<->G mAP={best_map:.2f} @ epoch {best_epoch}")
    print(f"Checkpoints in {args.out_dir} (model_best.pth / model_final.pth)")
    print("=" * 70)


if __name__ == '__main__':
    main()
