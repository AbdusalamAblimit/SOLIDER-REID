#!/usr/bin/env python3
# Frozen-backbone donor-leakage counterfactual consistency gate. Example, Occ-PoseTrack Small MULTIHYP:
# CUDA_VISIBLE_DEVICES=0 python leak_gate.py --config_file configs/occluded_posetrack/swin_small_multihyp.yml --expect_map 77.4 --out json TEST.WEIGHT /hy-tmp/reid-clean/log/occluded_posetrack/occ_small_mh/checkpoint_120.pth MODEL.PRETRAIN_CHOICE self MODEL.PRETRAIN_PATH /hy-tmp/reid-clean/weights/solider_swin_small_tea.pth DATASETS.ROOT_DIR /hy-tmp/reid-clean/data TEST.IMS_PER_BATCH 64
# The same gate can be run on Occluded-Duke by passing the Duke config, root, checkpoint, and --expect_map.
import argparse, json, os, random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from config import cfg
from datasets.make_dataloader import make_dataloader
from model import make_model
def parse_args():
    p = argparse.ArgumentParser(description="Frozen-head donor leakage gate")
    p.add_argument("--config_file", required=True)
    p.add_argument("--expect_map", type=float, required=True)
    p.add_argument("--expect_tol", type=float, default=1.0)
    p.add_argument("--out", default="text", help="text, json, or a JSON file path")
    for name, default, typ in [
        ("seed", 20260607, int), ("batch_size", 0, int), ("head_batch", 512, int),
        ("epochs", 30, int), ("lr", 1e-3, float), ("weight_decay", 1e-4, float),
        ("clean_m", 3, int), ("num_val_ids", 256, int), ("rand_ids", 8, int),
        ("margin", 0.0, float), ("clean_w", 1.0, float), ("consist_w", 0.3, float),
        ("leak_w", 0.3, float), ("max_train", 0, int),
    ]:
        p.add_argument("--" + name, default=default, type=typ)
    args, opts = p.parse_known_args()
    return args, opts[1:] if opts and opts[0] == "--" else opts
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
def first_feat(out):
    return out[0] if isinstance(out, (tuple, list)) else out
def model_features(model, images, camids, views, batch_size, device):
    feats = []; model.eval()
    with torch.no_grad():
        for s in range(0, images.shape[0], batch_size):
            e = min(s + batch_size, images.shape[0])
            out = model(images[s:e].to(device), cam_label=camids[s:e].to(device), view_label=views[s:e].to(device))
            feats.append(first_feat(out).detach())
    return F.normalize(torch.cat(feats), p=2, dim=1)
def extract_features(model, loader, device, batch_size):
    feats, pids, camids = [], [], []
    for img, pid, camid, camids_t, views_t, _ in loader:
        feats.append(model_features(model, img, camids_t, views_t, batch_size, device).cpu())
        pids.extend(int(x) for x in pid); camids.extend(int(x) for x in camid)
    return torch.cat(feats), np.asarray(pids), np.asarray(camids)
def group_indices(raw_items, offset=0):
    by_pid = defaultdict(list)
    for i, item in enumerate(raw_items):
        by_pid[int(item[1])].append(offset + i)
    return by_pid
def deranged_pid_map(pids, seed):
    order = list(pids)
    if len(order) < 2:
        raise RuntimeError("need at least two identities")
    rng = random.Random(seed); rng.shuffle(order)
    return dict(zip(order, order[1:] + order[:1]))
def train_donor_indices(raw_items, max_train, seed):
    by_pid = group_indices(raw_items); pids = sorted(by_pid)
    pair = deranged_pid_map(pids, seed + 1)
    rng = random.Random(seed + 2); ptr = defaultdict(int)
    indices = list(range(len(raw_items)))
    if max_train > 0:
        rng.shuffle(indices); indices = sorted(indices[:max_train])
    donors, b_pids = [], []
    for idx in indices:
        pid = int(raw_items[idx][1]); b = int(pair[pid]); pool = list(by_pid[b])
        rng.shuffle(pool)
        donors.append(pool[ptr[b] % len(pool)]); ptr[b] += 1; b_pids.append(b)
    return indices, donors, b_pids
def sample_targets(query_items, centroid_pids, max_ids, seed):
    by_pid, valid, rng = group_indices(query_items), set(int(x) for x in centroid_pids), random.Random(seed)
    pids = [pid for pid in sorted(by_pid) if pid in valid]; rng.shuffle(pids)
    pids = pids[:max_ids] if max_ids > 0 else pids
    idx = [rng.choice(by_pid[pid]) for pid in pids]
    if len(idx) < 2:
        raise RuntimeError("need at least two query identities with gallery centroids")
    return idx, pids
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
def composite_features(model, image_set, targets, donors, batch_size, device, seed):
    feats, areas = [], []
    for s in range(0, len(targets), batch_size):
        e = min(s + batch_size, len(targets))
        t_img, _, camids, views = load_samples(image_set, targets[s:e])
        d_img, _, _, _ = load_samples(image_set, donors[s:e])
        comp, area = paste_person_crops(t_img, d_img, seed + s)
        feats.append(model_features(model, comp, camids, views, batch_size, device).cpu())
        areas.append(area * (e - s))
    return torch.cat(feats), float(sum(areas) / max(1, len(targets)))
def clean_centroids(feats, pids, clean_m):
    by_pid = defaultdict(list)
    for i, pid in enumerate(pids):
        by_pid[int(pid)].append(i)
    pid_list, means = [], []
    for pid in sorted(by_pid):
        idx = torch.tensor(by_pid[pid], dtype=torch.long)
        means.append(F.normalize(feats[idx].mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0))
        pid_list.append(pid)
    means = torch.stack(means); pids_np = np.asarray(pid_list)
    cents = []
    for j, pid in enumerate(pid_list):
        idx = torch.tensor(by_pid[pid], dtype=torch.long); f = feats[idx]
        sim = f @ means.t(); own = sim[:, j].clone(); sim[:, j] = -float("inf")
        margin = own - sim.max(dim=1).values
        top = idx[torch.argsort(margin, descending=True)[:min(clean_m, len(idx))]]
        cents.append(F.normalize(feats[top].mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0))
    if len(cents) < 3:
        raise RuntimeError("at least three identities are required for centroids")
    return torch.stack(cents), pids_np
class IdentityResidualHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        return F.normalize(x + self.proj(x), p=2, dim=1)
def train_head(args, f_clean, f_comp, pids, donor_pids, centroids, centroid_pids, device):
    dim = f_clean.shape[1]; pid_to_label = {pid: i for i, pid in enumerate(sorted(set(map(int, pids))))}
    labels = torch.tensor([pid_to_label[int(p)] for p in pids], dtype=torch.long)
    col = {int(pid): i for i, pid in enumerate(centroid_pids)}
    donor_cols = torch.tensor([col[int(p)] for p in donor_pids], dtype=torch.long)
    ds = TensorDataset(f_clean.float(), f_comp.float(), labels, donor_cols)
    g = torch.Generator().manual_seed(args.seed + 123)
    loader = DataLoader(ds, batch_size=args.head_batch, shuffle=True, generator=g, drop_last=False)
    head, clf = IdentityResidualHead(dim).to(device), nn.Linear(dim, len(pid_to_label), bias=False).to(device)
    opt = torch.optim.Adam(list(head.parameters()) + list(clf.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    centroids_d = centroids.float().to(device)
    last = {}
    for epoch in range(1, args.epochs + 1):
        sums = defaultdict(float); n = 0
        for clean, comp, label, bcol in loader:
            clean = clean.to(device); comp = comp.to(device); label = label.to(device); bcol = bcol.to(device)
            zc, zp = head(clean), head(comp)
            with torch.no_grad():
                cb = head(centroids_d[bcol]).detach()
            clean_loss = F.cross_entropy(clf(zc), label)
            consist_loss = (1.0 - (zp * zc.detach()).sum(dim=1)).mean()
            leak_gap = (zp * cb).sum(dim=1) - (zc * cb).sum(dim=1) - args.margin
            leak_loss = F.relu(leak_gap).mean()
            loss = args.clean_w * clean_loss + args.consist_w * consist_loss + args.leak_w * leak_loss
            opt.zero_grad(); loss.backward(); opt.step()
            bs = clean.shape[0]; n += bs
            for k, v in [("loss", loss), ("ce", clean_loss), ("consist", consist_loss), ("leak", leak_loss)]:
                sums[k] += float(v.detach().cpu()) * bs
        last = {k: v / max(1, n) for k, v in sums.items()}
        if epoch in {1, args.epochs} or epoch % max(1, args.epochs // 5) == 0:
            print("HEAD_EPOCH epoch={} loss={:.6f} ce={:.6f} consist={:.6f} leak={:.6f}".format(
                epoch, last["loss"], last["ce"], last["consist"], last["leak"]))
    return head.eval(), last
def market_ap(distmat, q_pids, g_pids, q_camids, g_camids):
    order = np.argsort(distmat, axis=1)
    matches = (g_pids[order] == q_pids[:, None]).astype(np.int32)
    aps = np.full(len(q_pids), np.nan, dtype=np.float64); cmcs = []; valid = np.zeros(len(q_pids), dtype=bool)
    for i, q_pid in enumerate(q_pids):
        keep = ~((g_pids[order[i]] == q_pid) & (g_camids[order[i]] == q_camids[i]))
        cmc0 = matches[i][keep]
        if not np.any(cmc0):
            continue
        cmc = cmc0.cumsum(); cmc[cmc > 1] = 1; cmcs.append(cmc[:50]); valid[i] = True
        prec = cmc0.cumsum() / (np.arange(len(cmc0)) + 1.0)
        aps[i] = float((prec * cmc0).sum() / cmc0.sum())
    if not valid.any():
        raise RuntimeError("all query identities are absent from gallery after Market removal")
    max_len = max(len(x) for x in cmcs)
    cmc = np.asarray([np.pad(x, (0, max_len - len(x)), constant_values=x[-1]) for x in cmcs]).mean(axis=0)
    return float(np.nanmean(aps)), cmc
def eval_map(qf, gf, q_pids, g_pids, q_camids, g_camids):
    dist = (2.0 - 2.0 * (qf @ gf.t()).numpy()).astype(np.float32)
    mAP, cmc = market_ap(dist, q_pids, g_pids, q_camids, g_camids)
    return 100.0 * mAP, 100.0 * float(cmc[0])
@torch.no_grad()
def apply_head(head, feats, device, batch_size):
    out = []
    for s in range(0, feats.shape[0], batch_size):
        out.append(head(feats[s:s + batch_size].to(device)).cpu())
    return torch.cat(out)
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
def net_leak(clean, plus, centroids, a_pids, b_pids, centroid_pids, rand_ids, seed):
    col = {int(pid): i for i, pid in enumerate(centroid_pids)}
    b_cols = torch.tensor([col[int(pid)] for pid in b_pids], dtype=torch.long)
    sim_clean, sim_plus = clean @ centroids.t(), plus @ centroids.t()
    row = torch.arange(len(a_pids))
    leak_b = float((sim_plus[row, b_cols] - sim_clean[row, b_cols]).mean().item())
    leak_rand = random_leak(sim_clean, sim_plus, a_pids, b_pids, centroid_pids, col, rand_ids, seed)
    return leak_b - leak_rand, leak_b, leak_rand
def main():
    args, opts = parse_args(); seed_everything(args.seed)
    cfg.merge_from_file(args.config_file); cfg.merge_from_list(opts); cfg.freeze()
    if cfg.TEST.WEIGHT == "":
        raise ValueError("TEST.WEIGHT must be set")
    if bool(cfg.TEST.RE_RANKING):
        raise ValueError("TEST.RE_RANKING must be False for this pre-registered gate")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.MODEL.DEVICE_ID))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this gate is intended for a GPU server")
    _, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num,
                       semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(cfg.TEST.WEIGHT)
    for p in model.parameters():
        p.requires_grad_(False)
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs for frozen feature extraction".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    device = torch.device("cuda"); model.to(device).eval()
    bs = args.batch_size if args.batch_size > 0 else int(cfg.TEST.IMS_PER_BATCH)
    print("CACHE train clean features")
    train_f, train_pids, _ = extract_features(model, train_loader_normal, device, bs)
    train_set = train_loader_normal.dataset; train_raw = train_set.dataset
    tgt_idx, donor_idx, donor_pids = train_donor_indices(train_raw, args.max_train, args.seed + 10)
    clean_train = train_f[torch.tensor(tgt_idx, dtype=torch.long)] if args.max_train > 0 else train_f
    train_pids_used = train_pids[tgt_idx] if args.max_train > 0 else train_pids
    print("CACHE train composite features n={}".format(len(tgt_idx)))
    comp_train, train_area = composite_features(model, train_set, tgt_idx, donor_idx, bs, device, args.seed + 20)
    train_cent, train_cent_pids = clean_centroids(train_f, train_pids, args.clean_m)
    head, last_loss = train_head(args, clean_train, comp_train, train_pids_used, donor_pids,
                                 train_cent, train_cent_pids, device)
    print("CACHE val clean features")
    val_f, val_pids, val_camids = extract_features(model, val_loader, device, bs)
    qf, gf = val_f[:num_query], val_f[num_query:]
    q_pids, g_pids = val_pids[:num_query], val_pids[num_query:]
    q_camids, g_camids = val_camids[:num_query], val_camids[num_query:]
    base_map, base_r1 = eval_map(qf, gf, q_pids, g_pids, q_camids, g_camids)
    expect = args.expect_map if args.expect_map > 1.0 else 100.0 * args.expect_map
    diff = abs(base_map - expect)
    print("AP_GATE baseline_clean_mAP={:.2f} expect={:.2f} diff={:.2f} tol={:.2f} {}".format(
        base_map, expect, diff, args.expect_tol, "PASS" if diff <= args.expect_tol else "FAIL"))
    if diff > args.expect_tol:
        raise AssertionError("baseline mAP is outside --expect_map tolerance")
    h_val = apply_head(head, val_f, device, bs)
    hqf, hgf = h_val[:num_query], h_val[num_query:]
    head_map, head_r1 = eval_map(hqf, hgf, q_pids, g_pids, q_camids, g_camids)
    val_cent, val_cent_pids = clean_centroids(gf, g_pids, args.clean_m)
    val_cent_h = apply_head(head, val_cent, device, bs)
    val_set = val_loader.dataset; raw = val_set.dataset; query_raw, gallery_raw = raw[:num_query], raw[num_query:]
    target_idx, a_pids = sample_targets(query_raw, val_cent_pids, args.num_val_ids, args.seed + 30)
    pair = deranged_pid_map(a_pids, args.seed + 31); b_pids = [pair[pid] for pid in a_pids]
    q_by_pid, g_by_pid = group_indices(query_raw), group_indices(gallery_raw, offset=num_query)
    source_idx, source_is_gallery = choose_sources(b_pids, q_by_pid, g_by_pid, args.seed + 32)
    print("CACHE val composite features n={}".format(len(target_idx)))
    plus_val, val_area = composite_features(model, val_set, target_idx, source_idx, bs, device, args.seed + 40)
    clean_val = qf[torch.tensor(target_idx, dtype=torch.long)]
    plus_val_h, clean_val_h = apply_head(head, plus_val, device, bs), apply_head(head, clean_val, device, bs)
    base_net, base_b, base_rand = net_leak(clean_val, plus_val, val_cent, a_pids, b_pids,
                                           val_cent_pids, args.rand_ids, args.seed + 50)
    head_net, head_b, head_rand = net_leak(clean_val_h, plus_val_h, val_cent_h, a_pids, b_pids,
                                           val_cent_pids, args.rand_ids, args.seed + 50)
    drop = base_map - head_map
    leak_ok, clean_ok = head_net <= 0.030, drop < 0.2
    result = dict(dataset=str(cfg.DATASETS.NAMES), backbone=str(cfg.MODEL.TRANSFORMER_TYPE),
                  checkpoint=str(cfg.TEST.WEIGHT), feature_dim=int(val_f.shape[1]),
                  train_n=int(len(tgt_idx)), val_triples=int(len(target_idx)),
                  train_patch_area=train_area, val_patch_area=val_area,
                  source_gallery_frac=float(np.mean(source_is_gallery)),
                  clean_m=args.clean_m, rand_ids=args.rand_ids, last_loss=last_loss,
                  baseline_clean_mAP=base_map, baseline_rank1=base_r1,
                  head_clean_mAP=head_map, head_rank1=head_r1, clean_mAP_drop=drop,
                  baseline_net_leak=base_net, baseline_leak_B=base_b, baseline_leak_rand=base_rand,
                  head_net_leak=head_net, head_leak_B=head_b, head_leak_rand=head_rand,
                  leak_reduction_abs=base_net - head_net, leak_pass=leak_ok,
                  clean_pass=clean_ok, pass_=bool(leak_ok and clean_ok))
    print("SUMMARY baseline_clean_mAP={:.2f} head_clean_mAP={:.2f} clean_mAP_drop={:.2f} "
          "baseline_net_leak={:.6f} head_net_leak={:.6f} leak_reduction_abs={:.6f}".format(
              base_map, head_map, drop, base_net, head_net, base_net - head_net))
    print("GATE leak_pass={} clean_pass={} PASS={}".format(leak_ok, clean_ok, bool(leak_ok and clean_ok)))
    print("DETAIL baseline_rank1={:.2f} head_rank1={:.2f} train_area={:.4f} val_area={:.4f}".format(
        base_r1, head_r1, train_area, val_area))
    if args.out == "json":
        print("JSON_RESULT {}".format(json.dumps(result, sort_keys=True)))
    elif args.out != "text":
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
if __name__ == "__main__":
    main()
