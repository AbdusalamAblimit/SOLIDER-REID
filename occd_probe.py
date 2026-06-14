#!/usr/bin/env python3
# Eval-only probe for shared synthetic occluder shortcuts.
# Tiny:
# python occd_probe.py --config_file configs/occluded_posetrack/swin_tiny.yml --num_ids 256 --num_occ 32 TEST.WEIGHT /path/to/tiny.pth MODEL.PRETRAIN_PATH /path/to/tiny_pretrain.pth DATASETS.ROOT_DIR /path/to/data MODEL.DEVICE_ID 0
# Small:
# python occd_probe.py --config_file configs/occluded_posetrack/swin_tiny.yml --num_ids 256 --num_occ 32 TEST.WEIGHT /path/to/small.pth MODEL.PRETRAIN_PATH /path/to/small_pretrain.pth MODEL.TRANSFORMER_TYPE swin_small_patch4_window7_224 DATASETS.ROOT_DIR /path/to/data MODEL.DEVICE_ID 0

import argparse
import json
import os
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from config import cfg
from datasets.bases import ImageDataset
from datasets.make_dataloader import make_dataloader
from model import make_model
from model.occ_shortcut import build_fixed_occluder_pool, paste_occluder_batch
class PatchRng:
    def __init__(self, pool_size, seed, fixed=None, seq=None):
        self.pool_size, self.fixed, self.seq = int(pool_size), fixed, seq or []
        self.pos, self.base = 0, random.Random(int(seed))
    def random(self): return 0.0
    def randint(self, a, b): return self.base.randint(a, b)
    def randrange(self, *args):
        if len(args) == 1 and int(args[0]) == self.pool_size:
            if self.fixed is not None:
                return int(self.fixed)
            value = self.seq[self.pos % len(self.seq)]
            self.pos += 1
            return int(value)
        return self.base.randrange(*args)
def parse_args():
    parser = argparse.ArgumentParser(description="Shared-occluder shortcut probe")
    parser.add_argument("--config_file", required=True)
    for name, default, typ in [
        ("num_ids", 256, int), ("num_occ", 32, int), ("pool_size", 64, int),
        ("batch_size", 0, int), ("same_pairs", 512, int), ("seed", 20260607, int),
        ("sanity_margin", 0.05, float),
    ]:
        parser.add_argument("--" + name, default=default, type=typ)
    parser.add_argument("--out", default="", help="optional JSON output path")
    args, opts = parser.parse_known_args()
    return args, opts[1:] if opts and opts[0] == "--" else opts
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
def group_by_pid(raw_items):
    by_pid = defaultdict(list)
    for idx, (_, pid, _, _) in enumerate(raw_items):
        by_pid[int(pid)].append(idx)
    return by_pid
def sample_one_per_pid(raw_items, max_ids, seed):
    by_pid, rng = group_by_pid(raw_items), random.Random(seed)
    pids = sorted(by_pid); rng.shuffle(pids)
    return [rng.choice(by_pid[pid]) for pid in pids[:max_ids]]
def sample_same_id_pairs(raw_items, max_pairs, seed):
    by_pid, rng = group_by_pid(raw_items), random.Random(seed)
    valid = [pid for pid, idxs in by_pid.items() if len(idxs) >= 2]; rng.shuffle(valid)
    pairs, seen, attempts = [], set(), 0
    while valid and len(pairs) < max_pairs and attempts < max_pairs * 50:
        a, b = rng.sample(by_pid[valid[attempts % len(valid)]], 2)
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key); pairs.append(key)
        attempts += 1
    return pairs
def load_samples(image_set, indices):
    imgs, pids, camids, views = [], [], [], []
    for idx in indices:
        img, pid, camid, viewid, _ = image_set[idx]
        imgs.append(img); pids.append(int(pid)); camids.append(int(camid)); views.append(int(viewid))
    return torch.stack(imgs), pids, torch.tensor(camids), torch.tensor(views)
def model_features(model, images, camids, views, batch_size, device):
    feats = []; model.eval()
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            end = min(start + batch_size, images.shape[0])
            out = model(images[start:end].to(device), cam_label=camids[start:end].to(device),
                        view_label=views[start:end].to(device))
            feats.append((out[0] if isinstance(out, (tuple, list)) else out).detach())
    return F.normalize(torch.cat(feats), p=2, dim=1)
def mean_different_id_cos(feats, pids):
    pid_t = torch.tensor(pids, device=feats.device); sim = feats @ feats.t()
    upper = torch.triu(torch.ones_like(sim, dtype=torch.bool), 1)
    vals = sim[upper & (pid_t[:, None] != pid_t[None, :])]
    if vals.numel() == 0:
        raise RuntimeError("no different-id pairs were sampled")
    return float(vals.mean().item()), int(vals.numel())
def independent_seq(n, pool_size, seed):
    rng, seq = random.Random(seed), []
    while len(seq) < n:
        block = list(range(pool_size)); rng.shuffle(block); seq.extend(block)
    return seq[:n]
def main():
    args, opts = parse_args(); seed_everything(args.seed)
    if args.pool_size < 2 or args.num_occ < 1:
        raise ValueError("--pool_size must be >= 2 and --num_occ must be >= 1")
    cfg.merge_from_file(args.config_file); cfg.merge_from_list(opts); cfg.freeze()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this probe is intended for a GPU server")
    _, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num,
                       semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    if cfg.TEST.WEIGHT != "":
        model.load_param(cfg.TEST.WEIGHT)
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs for inference".format(torch.cuda.device_count())); model = torch.nn.DataParallel(model)
    device = torch.device("cuda"); model.to(device)
    val_set = val_loader.dataset
    probe_indices = sample_one_per_pid(val_set.dataset[:num_query], args.num_ids, args.seed + 1)
    if not probe_indices:
        raise RuntimeError("query set is empty")
    images, pids, camids, views = load_samples(val_set, probe_indices)
    batch_size = args.batch_size if args.batch_size > 0 else int(cfg.TEST.IMS_PER_BATCH)
    clean_feats = model_features(model, images, camids, views, batch_size, device)
    mean_clean_neg, clean_pairs = mean_different_id_cos(clean_feats, pids)
    same_pairs = sample_same_id_pairs(val_set.dataset, args.same_pairs, args.seed + 2)
    if not same_pairs:
        raise RuntimeError("no same-id pairs found in query+gallery for sanity validation")
    same_unique = sorted({idx for pair in same_pairs for idx in pair})
    same_imgs, _, same_camids, same_views = load_samples(val_set, same_unique)
    same_feats = model_features(model, same_imgs, same_camids, same_views, batch_size, device)
    pos = {idx: i for i, idx in enumerate(same_unique)}
    mean_sameid = float(torch.stack([(same_feats[pos[a]] * same_feats[pos[b]]).sum()
                                     for a, b in same_pairs]).mean().item())
    pool_transform = T.Compose([T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3), T.ToTensor(),
                                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
    pool_set = ImageDataset(train_loader_normal.dataset.dataset, pool_transform)
    pool = build_fixed_occluder_pool(pool_set, pool_size=args.pool_size, seed=args.seed)
    actual_num_occ, shared_means = min(args.num_occ, len(pool)), []
    for j, occ_idx in enumerate(range(actual_num_occ)):
        rng = PatchRng(len(pool), args.seed + 1000 + j, fixed=occ_idx)
        occ_images, _ = paste_occluder_batch(images, pool, aug_prob=1.0, rng=rng)
        occ_feats = model_features(model, occ_images, camids, views, batch_size, device)
        shared_means.append(mean_different_id_cos(occ_feats, pids)[0])
    mean_shared_neg = float(np.mean(shared_means))
    seq = independent_seq(images.shape[0], len(pool), args.seed + 3000)
    indep_images, _ = paste_occluder_batch(
        images, pool, aug_prob=1.0, rng=PatchRng(len(pool), args.seed + 4000, seq=seq))
    indep_feats = model_features(model, indep_images, camids, views, batch_size, device)
    mean_indep_neg, indep_pairs = mean_different_id_cos(indep_feats, pids)
    infl_vs_clean = mean_shared_neg - mean_clean_neg
    infl_vs_indep = mean_shared_neg - mean_indep_neg
    same_gap = mean_sameid - mean_clean_neg; sanity_ok = same_gap > args.sanity_margin
    checkpoint = cfg.TEST.WEIGHT if cfg.TEST.WEIGHT != "" else cfg.MODEL.PRETRAIN_PATH
    result = dict(backbone=str(cfg.MODEL.TRANSFORMER_TYPE), checkpoint=str(checkpoint),
                  dataset=str(cfg.DATASETS.NAMES), root=str(cfg.DATASETS.ROOT_DIR), n_ids=len(pids),
                  n_occ=actual_num_occ, pool_size=len(pool), feature_dim=int(clean_feats.shape[1]),
                  mean_clean_neg=mean_clean_neg, mean_shared_neg=mean_shared_neg,
                  mean_indep_neg=mean_indep_neg, inflation_vs_clean=infl_vs_clean,
                  inflation_vs_indep=infl_vs_indep, sameid=mean_sameid,
                  sameid_gap_vs_clean=same_gap, sameid_count=len(same_pairs),
                  clean_neg_count=clean_pairs, indep_neg_count=indep_pairs, sanity_ok=sanity_ok)
    lines = [
        ("backbone={}", cfg.MODEL.TRANSFORMER_TYPE), ("checkpoint={}", checkpoint),
        ("dataset={} root={} query_images={} sampled_ids={} feature_dim={}",
         cfg.DATASETS.NAMES, cfg.DATASETS.ROOT_DIR, num_query, len(pids), int(clean_feats.shape[1])),
        ("pool_size={} num_occ={} batch_size={}", len(pool), actual_num_occ, batch_size),
        ("mean_clean_neg={:.6f} count={}", mean_clean_neg, clean_pairs),
        ("mean_shared_neg={:.6f} count_per_occ={} occ_mean_std={:.6f}",
         mean_shared_neg, clean_pairs, float(np.std(shared_means))),
        ("mean_indep_neg={:.6f} count={}", mean_indep_neg, indep_pairs),
        ("sameid_mean={:.6f} count={} gap_vs_clean={:.6f} sanity_margin={:.6f} sanity_ok={}",
         mean_sameid, len(same_pairs), same_gap, args.sanity_margin, sanity_ok),
        ("inflation_vs_clean={:.6f}", infl_vs_clean), ("inflation_vs_indep={:.6f}", infl_vs_indep),
    ]
    for fmt, *vals in lines:
        print(fmt.format(*vals))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
    if not sanity_ok:
        raise AssertionError("same-id mean similarity is not clearly higher than clean different-id similarity")
    print("SUMMARY backbone={} clean_neg={:.6f} shared_neg={:.6f} indep_neg={:.6f} "
          "infl_vs_clean={:.6f} infl_vs_indep={:.6f} sameid={:.6f} n_ids={} n_occ={}".format(
              cfg.MODEL.TRANSFORMER_TYPE, mean_clean_neg, mean_shared_neg, mean_indep_neg,
              infl_vs_clean, infl_vs_indep, mean_sameid, len(pids), actual_num_occ))
if __name__ == "__main__":
    main()
