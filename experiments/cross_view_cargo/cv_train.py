# encoding: utf-8
"""
CVCL training: address *cross-view positive scarcity* on CARGO (aerial<->ground).

Reuses the afd_reid baseline pipeline (losses, LR schedule, cross-view eval) and
adds two switchable mechanisms, both gated by --cv so the default reproduces the
afd_train.py baseline exactly:

  1. VC-PK sampler (ViewBalancedIdentitySampler):
       per pid, when K=4, prefer 2 aerial + 2 ground instances. If the pid is not
       dual-view (only one view available) it falls back to the standard random
       K-instance draw. Batch size is unchanged (P*K = 64).

  2. CV-triplet loss (batch-hard, restricted to OPPOSITE view):
       for each anchor, positive = hardest (farthest) same-id OPPOSITE-view sample
       in batch, negative = hardest (nearest) diff-id OPPOSITE-view sample in batch.
       Anchors with no opposite-view positive AND/OR no opposite-view negative in the
       batch are skipped (masked out). The VC-PK sampler exists precisely to make
       these in-batch opposite-view pairs available.
       total = CE + standard batch-hard triplet + lambda * CV-triplet   (lambda=0.5)

Everything else (model = resnet50 BoT use_afd=False, AdamW, cosine LR, eval every
10 ep A<->G) is identical to afd_train.py for a clean single-variable ablation.

Run on lab-3090 (single GPU):
    cd /root/work/SOLIDER-REID/experiments/cross_view_cargo
    # baseline reproduction (no cv):
    PYTHONUNBUFFERED=1 python cv_train.py \
        --data_root /root/work/SOLIDER-REID/data \
        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvcl_baseline \
        2>&1 | tee /tmp/cvcl_baseline.log
    # full method (VC-PK sampler + CV-triplet):
    PYTHONUNBUFFERED=1 python cv_train.py --cv \
        --data_root /root/work/SOLIDER-REID/data \
        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvcl_both \
        2>&1 | tee /tmp/cvcl_both.log
    # ablations:
    #   sampler only : --cv --cv_lambda 0      (VC-PK on, CV-triplet weight 0)
    #   triplet only : --cv --cv_no_sampler    (CV-triplet on, standard sampler)
"""
import os
import sys
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# reuse afd_reid pieces (losses / schedule / eval / model / dataset)
_HERE = os.path.dirname(os.path.abspath(__file__))
_AFD = os.path.normpath(os.path.join(_HERE, '..', 'afd_reid'))
sys.path.insert(0, _AFD)
from cargo_dataset import (CARGO, CARGOImageDataset, build_transforms,  # noqa: E402
                          RandomIdentitySampler)
from afd_model import build_model  # noqa: E402
from afd_train import (CrossEntropyLabelSmooth, TripletLoss, euclidean_dist,  # noqa: E402
                       WarmupCosineLR, run_cross_view_eval, print_eval, set_seed)

VIEW2IDX = {'Aerial': 0, 'Ground': 1}


# --------------------------------------------------------------------------- #
# VC-PK sampler: per pid prefer K/2 aerial + K/2 ground
# --------------------------------------------------------------------------- #
class ViewBalancedIdentitySampler(torch.utils.data.Sampler):
    """P identities x K instances, but each pid's K instances prefer a 50/50
    aerial/ground split so cross-view positive pairs land in the same batch.

    Fallbacks:
      - a pid with only one view available -> standard random K draw (replace if short)
      - a pid that is dual-view but short on one side -> fill the deficit from the
        other view (still keeps both views present whenever possible)
    """

    def __init__(self, samples, batch_size, num_instances):
        assert batch_size % num_instances == 0
        self.samples = samples
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances

        # global indices grouped by (pid, view)
        self.idx_by_pid_view = defaultdict(lambda: {0: [], 1: []})
        for idx, s in enumerate(samples):
            self.idx_by_pid_view[s['pid']][VIEW2IDX[s['view']]].append(idx)
        self.pids = list(self.idx_by_pid_view.keys())

        # length estimate identical in spirit to RandomIdentitySampler
        self.length = 0
        for pid in self.pids:
            n = len(self.idx_by_pid_view[pid][0]) + len(self.idx_by_pid_view[pid][1])
            n = max(n, self.num_instances)
            self.length += n - n % self.num_instances

    def _draw_pid_chunks(self, pid):
        """Return a list of K-length index chunks for one pid, each chunk
        view-balanced when the pid is dual-view."""
        a = list(self.idx_by_pid_view[pid][0])
        g = list(self.idx_by_pid_view[pid][1])
        random.shuffle(a)
        random.shuffle(g)
        K = self.num_instances
        half = K // 2

        chunks = []
        if a and g:
            # dual-view: build as many balanced chunks as the data supports
            n_chunks = max(1, (len(a) + len(g)) // K)
            ai = gi = 0
            for _ in range(n_chunks):
                want_a, want_g = half, K - half
                chunk = []
                # take from each side with wrap-around (replace) if a side runs short
                for _k in range(want_a):
                    if not a:
                        break
                    chunk.append(a[ai % len(a)]); ai += 1
                for _k in range(want_g):
                    if not g:
                        break
                    chunk.append(g[gi % len(g)]); gi += 1
                # top up to K from the combined pool if rounding left it short
                pool = a + g
                while len(chunk) < K:
                    chunk.append(pool[random.randrange(len(pool))])
                random.shuffle(chunk)
                chunks.append(chunk[:K])
        else:
            # single-view pid: standard random K-instance behavior
            pool = a + g
            if len(pool) < K:
                pool = np.random.choice(pool, size=K, replace=True).tolist()
            random.shuffle(pool)
            for b0 in range(0, len(pool) - K + 1, K):
                chunks.append(pool[b0:b0 + K])
            if not chunks:
                chunks.append(pool[:K])
        return chunks

    def __iter__(self):
        batch_chunks = {pid: self._draw_pid_chunks(pid) for pid in self.pids}
        avail = [pid for pid in self.pids if batch_chunks[pid]]
        final = []
        while len(avail) >= self.num_pids_per_batch:
            selected = random.sample(avail, self.num_pids_per_batch)
            for pid in selected:
                final.extend(batch_chunks[pid].pop(0))
                if not batch_chunks[pid]:
                    avail.remove(pid)
        self.length = len(final)
        return iter(final)

    def __len__(self):
        return self.length


# --------------------------------------------------------------------------- #
# CV-triplet: batch-hard triplet restricted to OPPOSITE-view pairs
# --------------------------------------------------------------------------- #
class CrossViewTripletLoss(nn.Module):
    """Batch-hard triplet but positives/negatives are limited to the opposite view.

    positive = farthest same-id opposite-view sample in batch (hardest positive)
    negative = nearest diff-id opposite-view sample in batch (hardest negative)
    Anchors lacking an opposite-view positive OR an opposite-view negative are
    skipped. Returns (loss, n_valid_anchors / batch_size) for logging coverage.
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels, views):
        n = feats.size(0)
        dist = euclidean_dist(feats, feats)               # (n,n), fp32 distances
        labels = labels.view(n)
        views = views.view(n)

        same_id = labels.unsqueeze(1).eq(labels.unsqueeze(0))      # (n,n) bool
        opp_view = views.unsqueeze(1).ne(views.unsqueeze(0))
        eye = torch.eye(n, dtype=torch.bool, device=feats.device)

        pos_mask = same_id & opp_view & (~eye)            # opposite-view positives
        neg_mask = (~same_id) & opp_view                  # opposite-view negatives

        has_pos = pos_mask.any(dim=1)
        has_neg = neg_mask.any(dim=1)
        valid = has_pos & has_neg
        coverage = valid.float().mean().item()
        if valid.sum() == 0:
            return feats.new_zeros(()), coverage

        # hardest positive = max dist among opposite-view positives
        neg_inf = torch.finfo(dist.dtype).min
        pos_filled = torch.where(pos_mask, dist, dist.new_full((), neg_inf))
        dist_ap = pos_filled.max(dim=1).values

        # hardest negative = min dist among opposite-view negatives
        pos_inf = torch.finfo(dist.dtype).max
        neg_filled = torch.where(neg_mask, dist, dist.new_full((), pos_inf))
        dist_an = neg_filled.min(dim=1).values

        dist_ap = dist_ap[valid]
        dist_an = dist_an[valid]
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss, coverage


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
    ap.add_argument('--out_dir', default='./log/cargo/cvcl_baseline')
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
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--last_stride', type=int, default=1)
    ap.add_argument('--pool', default='gem', choices=['gem', 'avg'])
    # --- CVCL switches ---
    ap.add_argument('--cv', action='store_true',
                    help='enable CVCL: VC-PK sampler + CV-triplet')
    ap.add_argument('--cv_lambda', type=float, default=0.5,
                    help='weight of the cross-view triplet loss')
    ap.add_argument('--cv_no_sampler', action='store_true',
                    help='(with --cv) keep the standard sampler, CV-triplet only')
    ap.add_argument('--cv_margin', type=float, default=0.3,
                    help='margin for the cross-view triplet (defaults to --margin)')
    args = ap.parse_args()
    args.use_afd = False           # CVCL operates on the plain BoT baseline
    if args.cv_margin is None:
        args.cv_margin = args.margin

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda'
    batch_size = args.P * args.K
    use_vc_sampler = args.cv and (not args.cv_no_sampler)
    use_cv_triplet = args.cv and (args.cv_lambda > 0)

    print("=" * 72)
    print("CVCL training (cross-view positive scarcity)")
    print(f"  cv={args.cv}  VC-PK sampler={use_vc_sampler}  "
          f"CV-triplet={use_cv_triplet} (lambda={args.cv_lambda})")
    print(f"  bs={batch_size} (P={args.P} K={args.K}) lr={args.lr} "
          f"epochs={args.epochs} warmup={args.warmup_epochs} amp={not args.no_amp}")
    print(f"  out_dir={args.out_dir}")
    print("=" * 72)

    # data
    dataset = CARGO(root=args.data_root, verbose=True)
    train_tf = build_transforms(is_train=True, img_size=tuple(args.img_size))
    train_set = CARGOImageDataset(dataset.train, train_tf)
    if use_vc_sampler:
        sampler = ViewBalancedIdentitySampler(dataset.train, batch_size, args.K)
        print("  [sampler] ViewBalancedIdentitySampler (prefer K/2 aerial + K/2 ground)")
    else:
        sampler = RandomIdentitySampler(dataset.train, batch_size, args.K)
        print("  [sampler] RandomIdentitySampler (baseline)")
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True)

    # model (plain BoT baseline)
    model = build_model(dataset.num_train_pids, args).to(device)

    # losses
    ce = CrossEntropyLabelSmooth(dataset.num_train_pids, args.label_smooth)
    tri = TripletLoss(args.margin)
    cv_tri = CrossViewTripletLoss(args.cv_margin) if use_cv_triplet else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = WarmupCosineLR(optimizer, args.warmup_epochs, args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=not args.no_amp)

    best_map, best_epoch = -1.0, -1
    n_iter_total = len(train_loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        meters = {'loss': 0.0, 'ce': 0.0, 'tri': 0.0, 'cvtri': 0.0, 'acc': 0.0, 'cov': 0.0}
        seen = 0
        cov_seen = 0

        for it, batch in enumerate(train_loader):
            imgs = batch['img'].to(device, non_blocking=True)
            labels = batch['pid'].to(device, non_blocking=True)
            views = torch.tensor([VIEW2IDX[v] for v in batch['view']], device=device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=not args.no_amp):
                out = model(imgs, view_idx=None, return_cvfc=False)
                logits = out['logits']
                gfeat = out['global_feat']
                loss_ce = ce(logits, labels)
                loss_tri = tri(gfeat, labels)
                loss = loss_ce + loss_tri

                loss_cv = torch.zeros((), device=device)
                cov = 0.0
                if cv_tri is not None:
                    # compute the cross-view triplet in fp32 for stable hard mining
                    loss_cv, cov = cv_tri(gfeat.float(), labels, views)
                    loss = loss + args.cv_lambda * loss_cv

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = imgs.size(0)
            seen += bs
            meters['loss'] += loss.item() * bs
            meters['ce'] += loss_ce.item() * bs
            meters['tri'] += loss_tri.item() * bs
            meters['cvtri'] += float(loss_cv) * bs
            meters['acc'] += (logits.argmax(1) == labels).float().sum().item()
            if cv_tri is not None:
                meters['cov'] += cov * bs
                cov_seen += bs

            if (it + 1) % 50 == 0 or (it + 1) == n_iter_total:
                lr = optimizer.param_groups[0]['lr']
                cov_str = (f"CVcov: {meters['cov'] / max(1, cov_seen):.2f} "
                           if cv_tri is not None else "")
                print(f"Epoch[{epoch}] Iter[{it + 1}/{n_iter_total}] "
                      f"Loss: {meters['loss'] / seen:.3f} "
                      f"CE: {meters['ce'] / seen:.3f} "
                      f"Tri: {meters['tri'] / seen:.3f} "
                      f"CVtri: {meters['cvtri'] / seen:.3f} "
                      f"{cov_str}"
                      f"Acc: {meters['acc'] / seen:.3f} LR: {lr:.2e}")

        scheduler.step()
        dt = time.time() - t0
        cov_final = (f" CVcov={meters['cov'] / max(1, cov_seen):.2f}"
                     if cv_tri is not None else "")
        print(f"Epoch[{epoch}] done in {dt:.1f}s  "
              f"Loss={meters['loss'] / seen:.3f} Acc={meters['acc'] / seen:.3f}{cov_final}")

        if epoch % args.eval_period == 0 or epoch == args.epochs:
            results = run_cross_view_eval(model, dataset, args, device)
            mean_map = print_eval(epoch, results)
            if mean_map > best_map:
                best_map = mean_map
                best_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(args.out_dir, 'model_best.pth'))
                print(f"    * new best mean mAP={best_map:.2f} (epoch {epoch}) saved")

    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_final.pth'))
    print("=" * 72)
    print(f"Training finished. Best mean A<->G mAP={best_map:.2f} @ epoch {best_epoch}")
    print(f"Checkpoints in {args.out_dir}")
    print("=" * 72)


if __name__ == '__main__':
    main()
