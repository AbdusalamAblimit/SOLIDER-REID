#!/usr/bin/env python3
# Eval-only person-identity leakage probe. No training.
# Occ-PoseTrack Tiny:
# CUDA_VISIBLE_DEVICES=0 python leak_probe.py --config_file configs/occluded_posetrack/swin_tiny.yml --num_ids 256 --out json TEST.WEIGHT /hy-tmp/reid-clean/log/occluded_posetrack/exp001h/checkpoint_120.pth MODEL.PRETRAIN_CHOICE self MODEL.PRETRAIN_PATH /hy-tmp/reid-clean/weights/solider_swin_tiny_tea.pth DATASETS.ROOT_DIR /hy-tmp/reid-clean/data TEST.IMS_PER_BATCH 64
# Occ-PoseTrack Small MULTIHYP:
# CUDA_VISIBLE_DEVICES=0 python leak_probe.py --config_file configs/occluded_posetrack/swin_small_multihyp.yml --num_ids 256 --out json TEST.WEIGHT /hy-tmp/reid-clean/log/occluded_posetrack/occ_small_mh/checkpoint_120.pth MODEL.PRETRAIN_CHOICE self MODEL.PRETRAIN_PATH /hy-tmp/reid-clean/weights/solider_swin_small_tea.pth DATASETS.ROOT_DIR /hy-tmp/reid-clean/data TEST.IMS_PER_BATCH 64
import argparse, json, os, random
from collections import defaultdict
import numpy as np, torch
import torch.nn.functional as F
from config import cfg
from datasets.make_dataloader import make_dataloader
from model import make_model
def parse_args():
    p = argparse.ArgumentParser(description="Non-target person identity leakage probe")
    p.add_argument("--config_file", required=True)
    for name, default, typ in [("num_ids", 256, int), ("clean_m", 3, int), ("rand_ids", 8, int),
                               ("batch_size", 0, int), ("seed", 20260607, int)]:
        p.add_argument("--" + name, default=default, type=typ)
    p.add_argument("--out", default="text", help="text, json, or a JSON file path")
    args, opts = p.parse_known_args()
    return args, opts[1:] if opts and opts[0] == "--" else opts
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
def model_features(model, images, camids, views, batch_size, device):
    feats = []; model.eval()
    with torch.no_grad():
        for s in range(0, images.shape[0], batch_size):
            e = min(s + batch_size, images.shape[0])
            out = model(images[s:e].to(device), cam_label=camids[s:e].to(device), view_label=views[s:e].to(device))
            feats.append((out[0] if isinstance(out, (tuple, list)) else out).detach())
    return F.normalize(torch.cat(feats), p=2, dim=1)
def extract_features(model, loader, device, batch_size):
    feats, pids = [], []
    for img, pid, _, camids_t, views_t, _ in loader:
        feats.append(model_features(model, img, camids_t, views_t, batch_size, device).cpu())
        pids.extend(int(x) for x in pid)
    return torch.cat(feats), np.asarray(pids)
def clean_centroids(gf, g_pids, clean_m):
    by_pid = defaultdict(list)
    for i, pid in enumerate(g_pids):
        by_pid[int(pid)].append(i)
    pids, cents, counts = [], [], {}
    for pid in sorted(by_pid):
        idx = torch.tensor(by_pid[pid], dtype=torch.long); f = gf[idx]
        mean = F.normalize(f.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
        top = idx[torch.argsort(f @ mean, descending=True)[:min(clean_m, len(idx))]]
        cents.append(F.normalize(gf[top].mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0))
        pids.append(pid); counts[pid] = int(len(top))
    if len(cents) < 3:
        raise RuntimeError("at least three gallery identities are required")
    return torch.stack(cents), np.asarray(pids), counts
def group_indices(raw_items, offset=0):
    by_pid = defaultdict(list)
    for i, item in enumerate(raw_items):
        by_pid[int(item[1])].append(offset + i)
    return by_pid
def sample_targets(query_items, centroid_pids, max_ids, seed):
    by_pid, valid, rng = group_indices(query_items), set(int(x) for x in centroid_pids), random.Random(seed)
    pids = [pid for pid in sorted(by_pid) if pid in valid]
    rng.shuffle(pids); pids = pids[:max_ids]
    idx = [rng.choice(by_pid[pid]) for pid in pids]
    if len(idx) < 2:
        raise RuntimeError("need at least two query identities with gallery centroids")
    return idx, pids
def derange_pids(pids, seed):
    order = list(pids); random.Random(seed).shuffle(order)
    return dict(zip(order, order[1:] + order[:1]))
def choose_sources(b_pids, query_by_pid, gallery_by_pid, seed):
    rng, src, from_gallery = random.Random(seed), [], []
    for pid in b_pids:
        pool = gallery_by_pid.get(int(pid), []); flag = bool(pool)
        if not pool:
            pool = query_by_pid.get(int(pid), [])
        if not pool:
            raise RuntimeError("no source image found for pid {}".format(pid))
        src.append(rng.choice(pool)); from_gallery.append(flag)
    return src, from_gallery
def load_samples(image_set, indices):
    imgs, pids, camids, views = [], [], [], []
    for idx in indices:
        img, pid, camid, viewid, _ = image_set[idx]
        imgs.append(img); pids.append(int(pid)); camids.append(int(camid)); views.append(int(viewid))
    return torch.stack(imgs), pids, torch.tensor(camids), torch.tensor(views)
def sample_patch_hw(height, width, rng, min_scale=0.3, max_scale=0.5):
    sh, sw = rng.uniform(min_scale, max_scale), rng.uniform(min_scale, max_scale)
    return max(1, min(height, int(round(height * sh)))), max(1, min(width, int(round(width * sw))))
def paste_person_crops(targets, sources, seed):
    if targets.shape != sources.shape:
        raise ValueError("target and source tensors must have the same shape")
    out, rng = targets.clone(), random.Random(seed)
    _, _, h, w = targets.shape; areas = []
    for i in range(targets.shape[0]):
        ph, pw = sample_patch_hw(h, w, rng)
        st, sl = rng.randint(0, h - ph), rng.randint(0, w - pw)
        dt, dl = rng.randint(0, h - ph), rng.randint(0, w - pw)
        out[i, :, dt:dt + ph, dl:dl + pw] = sources[i, :, st:st + ph, sl:sl + pw]
        areas.append((ph * pw) / float(h * w))
    return out, float(np.mean(areas))
def random_leak(sim_clean, sim_plus, a_pids, b_pids, centroid_pids, col, rand_ids, seed):
    rng, vals, all_pids = random.Random(seed), [], [int(x) for x in centroid_pids]
    for i, (a, b) in enumerate(zip(a_pids, b_pids)):
        cand = [pid for pid in all_pids if pid != int(a) and pid != int(b)]
        if not cand:
            raise RuntimeError("no random non-pasted identity is available")
        chosen = rng.sample(cand, min(rand_ids, len(cand)))
        cidx = torch.tensor([col[pid] for pid in chosen], dtype=torch.long)
        vals.append((sim_plus[i, cidx] - sim_clean[i, cidx]).mean())
    return float(torch.stack(vals).mean().item())
def main():
    args, opts = parse_args(); seed_everything(args.seed)
    cfg.merge_from_file(args.config_file); cfg.merge_from_list(opts); cfg.freeze()
    if cfg.TEST.WEIGHT == "":
        raise ValueError("TEST.WEIGHT must be set")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.MODEL.DEVICE_ID))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this probe is intended for a GPU server")
    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num,
                       semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(cfg.TEST.WEIGHT)
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs for inference".format(torch.cuda.device_count())); model = torch.nn.DataParallel(model)
    device = torch.device("cuda"); model.to(device)
    batch_size = args.batch_size if args.batch_size > 0 else int(cfg.TEST.IMS_PER_BATCH)
    feats, pids = extract_features(model, val_loader, device, batch_size)
    qf, gf, g_pids = feats[:num_query], feats[num_query:], pids[num_query:]
    centroids, centroid_pids, clean_counts = clean_centroids(gf, g_pids, args.clean_m)
    val_set = val_loader.dataset; raw = val_set.dataset; query_raw, gallery_raw = raw[:num_query], raw[num_query:]
    target_idx, a_pids = sample_targets(query_raw, centroid_pids, args.num_ids, args.seed + 1)
    pair = derange_pids(a_pids, args.seed + 2); b_pids = [pair[pid] for pid in a_pids]
    q_by_pid, g_by_pid = group_indices(query_raw), group_indices(gallery_raw, offset=num_query)
    source_idx, source_is_gallery = choose_sources(b_pids, q_by_pid, g_by_pid, args.seed + 3)
    target_imgs, _, target_camids, target_views = load_samples(val_set, target_idx)
    source_imgs, _, _, _ = load_samples(val_set, source_idx)
    composite, mean_patch_area = paste_person_crops(target_imgs, source_imgs, args.seed + 4)
    plus = model_features(model, composite, target_camids, target_views, batch_size, device).cpu()
    clean = qf[torch.tensor(target_idx, dtype=torch.long)]
    col = {int(pid): i for i, pid in enumerate(centroid_pids)}
    a_cols = torch.tensor([col[int(pid)] for pid in a_pids], dtype=torch.long)
    b_cols = torch.tensor([col[int(pid)] for pid in b_pids], dtype=torch.long)
    sim_clean, sim_plus = clean @ centroids.t(), plus @ centroids.t()
    row = torch.arange(len(a_pids))
    leak_b = float((sim_plus[row, b_cols] - sim_clean[row, b_cols]).mean().item())
    leak_rand = random_leak(sim_clean, sim_plus, a_pids, b_pids, centroid_pids, col, args.rand_ids, args.seed + 5)
    net_leak = leak_b - leak_rand
    drop_self = float((sim_clean[row, a_cols] - sim_plus[row, a_cols]).mean().item())
    self_sim = float(sim_clean[row, a_cols].mean().item())
    crossid_sim = float(sim_clean[row, b_cols].mean().item())
    result = dict(dataset=str(cfg.DATASETS.NAMES), backbone=str(cfg.MODEL.TRANSFORMER_TYPE),
                  checkpoint=str(cfg.TEST.WEIGHT), n=len(a_pids), num_query=int(num_query),
                  feature_dim=int(clean.shape[1]), clean_m=args.clean_m, rand_ids=args.rand_ids,
                  source_gallery_frac=float(np.mean(source_is_gallery)), mean_patch_area=mean_patch_area,
                  mean_clean_count=float(np.mean(list(clean_counts.values()))), leak_B=leak_b,
                  leak_rand=leak_rand, net_leak=net_leak, drop_self=drop_self,
                  self_sim=self_sim, crossid_sim=crossid_sim)
    print("dataset={} backbone={} checkpoint={}".format(result["dataset"], result["backbone"], result["checkpoint"]))
    print("targets={} query_images={} gallery_images={} feature_dim={} clean_M={} rand_ids={}".format(
        len(a_pids), num_query, len(g_pids), result["feature_dim"], args.clean_m, args.rand_ids))
    print("source_gallery_frac={:.3f} mean_patch_area={:.4f} mean_clean_count={:.2f}".format(
        result["source_gallery_frac"], mean_patch_area, result["mean_clean_count"]))
    print("sanity self_sim={:.6f} crossid_sim={:.6f}".format(self_sim, crossid_sim))
    summary = ("SUMMARY backbone={} dataset={} leak_B={:.6f} leak_rand={:.6f} net_leak={:.6f} "
               "drop_self={:.6f} self_sim={:.6f} crossid_sim={:.6f} n={}").format(
        result["backbone"], result["dataset"], leak_b, leak_rand, net_leak,
        drop_self, self_sim, crossid_sim, len(a_pids))
    print(summary); result["summary"] = summary
    if args.out == "json":
        print("JSON_RESULT {}".format(json.dumps(result, sort_keys=True)))
    elif args.out != "text":
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
if __name__ == "__main__":
    main()
