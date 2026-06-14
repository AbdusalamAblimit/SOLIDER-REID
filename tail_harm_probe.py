#!/usr/bin/env python3
# Eval-only tail harm probe. No training.
# Occ-Duke Tiny baseline with keypoints:
# python tail_harm_probe.py --config_file configs/occluded_duke/swin_tiny.yml --expect_map 56.4 --kp_dir /path/to/kp TEST.WEIGHT /path/to/tiny_last.pth MODEL.PRETRAIN_CHOICE imagenet MODEL.PRETRAIN_PATH /path/to/solider_swin_tiny_tea.pth DATASETS.ROOT_DIR /path/to/data MODEL.DEVICE_ID 0
# Occ-PoseTrack Small baseline without keypoints:
# python tail_harm_probe.py --config_file configs/occluded_posetrack/swin_tiny.yml --expect_map 78.0 TEST.WEIGHT /path/to/small_last.pth MODEL.PRETRAIN_CHOICE imagenet MODEL.PRETRAIN_PATH /path/to/solider_swin_small_tea.pth MODEL.TRANSFORMER_TYPE swin_small_patch4_window7_224 DATASETS.ROOT_DIR /path/to/data MODEL.DEVICE_ID 0
import argparse, json, os, random
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from config import cfg
from datasets.make_dataloader import make_dataloader
from model import make_model
def parse_args():
    p = argparse.ArgumentParser(description="Evidence-insufficient tail harm probe")
    p.add_argument("--config_file", required=True)
    p.add_argument("--clean_m", type=int, default=3)
    p.add_argument("--expect_map", type=float, required=True)
    p.add_argument("--expect_tol", type=float, default=0.5, help="mAP tolerance in percentage points")
    p.add_argument("--kp_dir", default="")
    p.add_argument("--kp_conf", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260607)
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
def extract_features(model, loader, device):
    feats, flip_feats, pids, camids, paths = [], [], [], [], []
    bs = int(cfg.TEST.IMS_PER_BATCH)
    for img, pid, camid, camids_t, views_t, imgpath in loader:
        feat = model_features(model, img, camids_t, views_t, bs, device)
        flip = model_features(model, torch.flip(img, dims=[3]), camids_t, views_t, bs, device)
        feats.append(feat.cpu()); flip_feats.append(flip.cpu())
        pids.extend(int(x) for x in pid); camids.extend(int(x) for x in camid); paths.extend(imgpath)
    return torch.cat(feats), torch.cat(flip_feats), np.asarray(pids), np.asarray(camids), list(paths)
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
    return aps, float(np.nanmean(aps)), cmc, valid
def validate_ap(qf, gf, q_pids, g_pids, q_camids, g_camids, expect_map, expect_tol):
    dist = (2.0 - 2.0 * (qf @ gf.t()).numpy()).astype(np.float32)
    aps, mAP, cmc, valid = market_ap(dist, q_pids, g_pids, q_camids, g_camids)
    expect = expect_map / 100.0 if expect_map > 1.0 else expect_map
    diff, tol = abs(mAP - expect), expect_tol / 100.0
    print("AP_GATE mAP={:.2f} expect={:.2f} diff={:.2f} tol={:.2f} {}".format(
        100 * mAP, 100 * expect, 100 * diff, expect_tol, "PASS" if diff <= tol else "FAIL"))
    if diff > tol:
        raise AssertionError("mAP reproduction failed; feature extraction or AP protocol is not trusted")
    return aps, mAP, cmc, valid
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
    if len(cents) < 2:
        raise RuntimeError("at least two gallery identities are required for cross-id centroid margins")
    return torch.stack(cents), np.asarray(pids), counts
def centroid_margins(feats, pids, centroids, centroid_pids):
    col = {int(pid): i for i, pid in enumerate(centroid_pids)}
    missing = sorted({int(pid) for pid in pids if int(pid) not in col})
    if missing:
        raise RuntimeError("missing gallery clean centroid for ids: {}".format(missing[:10]))
    own_col = torch.tensor([col[int(pid)] for pid in pids], dtype=torch.long)
    sim = feats @ centroids.t(); row = torch.arange(len(pids))
    own = sim[row, own_col].clone(); sim[row, own_col] = -float("inf")
    cross = sim.max(dim=1).values
    return (own - cross).numpy()
def to_kpts(x):
    if isinstance(x, list) and x and all(isinstance(v, dict) for v in x):
        return np.asarray([[v.get("x", 0.0), v.get("y", 0.0),
                            v.get("conf", v.get("confidence", v.get("score", v.get("visibility", 0.0))))]
                           for v in x], dtype=float)
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 1 and arr.size >= 3 and arr.size % 3 == 0:
        arr = arr.reshape(-1, 3)
    return arr[:, :3] if arr.ndim == 2 and arr.shape[1] >= 3 else None
def kp_candidates(obj):
    if isinstance(obj, dict):
        out = []
        for k in ("target", "target_skeleton", "skeleton", "keypoints", "pose_keypoints_2d"):
            if k in obj:
                arr = to_kpts(obj[k]); out.extend([arr] if arr is not None else kp_candidates(obj[k]))
        for k in ("people", "persons", "instances", "predictions"):
            if k in obj:
                for item in obj[k] if isinstance(obj[k], list) else [obj[k]]:
                    out.extend(kp_candidates(item))
        return out
    if isinstance(obj, list):
        arr = to_kpts(obj)
        if arr is not None:
            return [arr]
        out = []
        for item in obj:
            out.extend(kp_candidates(item))
        return out
    return []
def load_visibility(paths, kp_dir, kp_conf):
    vals = []
    for path in paths:
        kp_path = os.path.join(kp_dir, os.path.basename(path) + "_keypoints.json")
        if not os.path.exists(kp_path):
            raise FileNotFoundError(kp_path)
        with open(kp_path, "r") as f:
            cand = kp_candidates(json.load(f))
        if not cand:
            raise ValueError("no keypoints found in {}".format(kp_path))
        arr = max(cand, key=lambda a: int((a[:, 2] > kp_conf).sum()))
        vals.append(float((arr[:, 2] > kp_conf).sum()) / max(1, arr.shape[0]))
    return np.asarray(vals, dtype=np.float64)
def sigmoid_z(raw, name):
    mu, std = float(np.mean(raw)), float(np.std(raw))
    std = std if std >= 1e-12 else 1.0
    letter = dict(margin="a", consist="b", vis="c").get(name, name)
    print("{}={:.6f} {}_mu={:.6f} {}_std={:.6f}".format(letter, 1.0 / std, name, mu, name, std))
    return 1.0 / (1.0 + np.exp(-np.clip((raw - mu) / std, -60.0, 60.0))), dict(mu=mu, std=std, scale=1.0 / std)
def reliability(margin, consist, paths, args):
    rm, cm = sigmoid_z(margin, "margin"); rc, cc = sigmoid_z(consist, "consist")
    rel, constants = rm * rc, dict(margin=cm, consist=cc)
    if args.kp_dir:
        rv, cv = sigmoid_z(load_visibility(paths, args.kp_dir, args.kp_conf), "vis")
        rel *= rv; constants["vis"] = cv; print("vis_enabled=True kp_conf={:.2f}".format(args.kp_conf))
    else:
        print("vis_enabled=False")
    return rel, constants
def bottom(values, frac):
    n = max(1, int(np.ceil(len(values) * frac))); mask = np.zeros(len(values), dtype=bool)
    mask[np.argsort(values)[:n]] = True
    return mask
def top(values, frac):
    n = max(1, int(np.ceil(len(values) * frac))); mask = np.zeros(len(values), dtype=bool)
    mask[np.argsort(-values)[:n]] = True
    return mask
def c1_tail_loss(aps, q_rel, valid):
    v = np.where(valid)[0]; low = v[np.argsort(q_rel[v])[:max(1, int(np.ceil(0.20 * len(v))))]]
    total = float((1.0 - aps[v]).sum())
    captured = float((1.0 - aps[low]).sum() / total) if total > 1e-12 else 0.0
    return captured, captured / 0.20, len(low), (captured >= 0.30 and captured / 0.20 >= 1.5)
def c2_hard_negative(qf, gf, q_pids, g_pids, q_rel, g_rel):
    high = np.where(top(q_rel, 0.50))[0]; low_g = bottom(g_rel, 0.30); low_t = torch.from_numpy(low_g)
    total, low_count, g_pid_arr = 0, 0, np.asarray(g_pids)
    for s in range(0, len(high), 512):
        idx = high[s:s + 512]; sim = qf[idx] @ gf.t()
        same = torch.from_numpy(np.asarray(q_pids[idx])[:, None] == g_pid_arr[None, :])
        sim[same] = -float("inf")
        vals, nn = torch.topk(sim, k=min(5, gf.shape[0]), dim=1)
        valid = torch.isfinite(vals)
        low_count += int((low_t[nn] & valid).sum()); total += int(valid.sum())
    frac = low_count / max(1, total); base = float(low_g.mean()); enrich = frac / max(base, 1e-12)
    return frac, base, enrich, len(high), enrich >= 1.5
def c3_weak_positive(q_margin, q_rel):
    low = bottom(q_rel, 0.20); frac = float((q_margin[low] > 0.0).mean())
    return frac, int(low.sum()), frac >= 0.65
def main():
    args, opts = parse_args(); seed_everything(args.seed)
    cfg.merge_from_file(args.config_file); cfg.merge_from_list(opts); cfg.freeze()
    if cfg.TEST.WEIGHT == "":
        raise ValueError("TEST.WEIGHT must be set")
    if bool(getattr(cfg, "MULTIHYP", None) is not None and cfg.MULTIHYP.ENABLED):
        raise ValueError("tail_harm_probe expects a single embedding config; set MULTIHYP.ENABLED False")
    if bool(cfg.TEST.RE_RANKING):
        raise ValueError("TEST.RE_RANKING must be False")
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
    feats, flip_feats, pids, camids, paths = extract_features(model, val_loader, device)
    qf, gf = feats[:num_query], feats[num_query:]
    q_pids, g_pids = pids[:num_query], pids[num_query:]
    q_camids, g_camids = camids[:num_query], camids[num_query:]
    aps, mAP, cmc, valid = validate_ap(qf, gf, q_pids, g_pids, q_camids, g_camids,
                                       args.expect_map, args.expect_tol)
    centroids, centroid_pids, clean_counts = clean_centroids(gf, g_pids, args.clean_m)
    margin = centroid_margins(feats, pids, centroids, centroid_pids)
    consist = (feats * flip_feats).sum(dim=1).numpy()
    rel, constants = reliability(margin, consist, paths, args)
    q_rel, g_rel, q_margin = rel[:num_query], rel[num_query:], margin[:num_query]
    c1_cap, c1_lift, c1_n, c1_ok = c1_tail_loss(aps, q_rel, valid)
    c2_frac, c2_base, c2_enrich, c2_n, c2_ok = c2_hard_negative(qf, gf, q_pids, g_pids, q_rel, g_rel)
    c3_frac, c3_n, c3_ok = c3_weak_positive(q_margin, q_rel)
    print("clean_exemplar_M={} gallery_ids={} mean_clean_count={:.2f}".format(
        args.clean_m, len(clean_counts), float(np.mean(list(clean_counts.values())))))
    print("C1 {} captured={:.3f} lift={:.3f} low_queries={} total_valid_queries={}".format(
        "PASS" if c1_ok else "FAIL", c1_cap, c1_lift, c1_n, int(valid.sum())))
    print("C2 {} low_neighbor_frac={:.3f} base_rate={:.3f} enrich={:.3f} high_queries={}".format(
        "PASS" if c2_ok else "FAIL", c2_frac, c2_base, c2_enrich, c2_n))
    print("C3 {} frac={:.3f} low_queries={}".format("PASS" if c3_ok else "FAIL", c3_frac, c3_n))
    summary = ("SUMMARY dataset={} backbone={} mAP={:.2f} C1={}(captured={:.3f},lift={:.3f}) "
               "C2={}(enrich={:.3f}) C3={}(frac={:.3f})").format(
        cfg.DATASETS.NAMES, cfg.MODEL.TRANSFORMER_TYPE, 100 * mAP,
        "PASS" if c1_ok else "FAIL", c1_cap, c1_lift,
        "PASS" if c2_ok else "FAIL", c2_enrich, "PASS" if c3_ok else "FAIL", c3_frac)
    print(summary)
    result = dict(dataset=str(cfg.DATASETS.NAMES), backbone=str(cfg.MODEL.TRANSFORMER_TYPE),
                  checkpoint=str(cfg.TEST.WEIGHT), mAP=100 * mAP, rank1=100 * float(cmc[0]),
                  clean_m=args.clean_m, constants=constants, summary=summary,
                  C1=dict(pass_=c1_ok, captured=c1_cap, lift=c1_lift),
                  C2=dict(pass_=c2_ok, frac=c2_frac, base_rate=c2_base, enrich=c2_enrich),
                  C3=dict(pass_=c3_ok, frac=c3_frac))
    if args.out == "json":
        print("JSON_RESULT {}".format(json.dumps(result, sort_keys=True)))
    elif args.out != "text":
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
if __name__ == "__main__":
    main()
