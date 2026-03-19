#!/usr/bin/env python
"""
Oracle Support Bank diagnostic.

Estimate the upper bound of support-complete per-keypoint prototypes by using
ground-truth same-ID samples from the evaluation split. This is an oracle
diagnostic only; it is not a valid test-time method.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import cfg
from datasets import make_dataloader
from model import make_model
from processor.processor import _pose_to_device
from utils.metrics import eval_func


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--weight', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--vis_thr', type=float, default=0.3)
    parser.add_argument('--dup_iou', type=float, default=0.55)
    parser.add_argument('--dup_kp_dist', type=float, default=0.12)
    return parser.parse_args()


def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


def kp_to_bbox(kp, scores, score_thr):
    vis = scores > score_thr
    if vis.sum() < 2:
        vis = scores > 0
    if vis.sum() == 0:
        return None
    pts = kp[vis]
    return np.asarray([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], dtype=np.float32)


def normalized_kp_distance(kp1, sc1, kp2, sc2, score_thr):
    vis = (sc1 > score_thr) & (sc2 > score_thr)
    if vis.sum() < 4:
        return np.inf
    pts1 = kp1[vis]
    pts2 = kp2[vis]
    box1 = kp_to_bbox(kp1, sc1, score_thr)
    box2 = kp_to_bbox(kp2, sc2, score_thr)
    if box1 is None or box2 is None:
        return np.inf
    union_box = np.asarray([
        min(box1[0], box2[0]), min(box1[1], box2[1]),
        max(box1[2], box2[2]), max(box1[3], box2[3])], dtype=np.float32)
    scale = np.sqrt(max((union_box[2] - union_box[0]) * (union_box[3] - union_box[1]), 1.0))
    return float(np.linalg.norm(pts1 - pts2, axis=1).mean() / scale)


def is_duplicate_person(kp1, sc1, kp2, sc2, score_thr, dup_iou_thr, dup_kp_dist_thr):
    box1 = kp_to_bbox(kp1, sc1, score_thr)
    box2 = kp_to_bbox(kp2, sc2, score_thr)
    if box1 is None or box2 is None:
        return False
    if bbox_iou(box1, box2) < dup_iou_thr:
        return False
    return normalized_kp_distance(kp1, sc1, kp2, sc2, score_thr) < dup_kp_dist_thr


def detect_duplicate_flag(keypoints, scores, num_persons, score_thr, dup_iou_thr, dup_kp_dist_thr):
    if num_persons <= 1:
        return False
    for i in range(num_persons):
        for j in range(i + 1, num_persons):
            if is_duplicate_person(keypoints[i], scores[i], keypoints[j], scores[j],
                                   score_thr, dup_iou_thr, dup_kp_dist_thr):
                return True
    return False


def eval_per_query(distmat, q_pids, g_pids, q_camids, g_camids):
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    all_ap, all_r1, valid = [], [], []
    for q_idx in range(distmat.shape[0]):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue
        valid.append(q_idx)
        all_r1.append(bool(orig_cmc[0]))
        num_rel = orig_cmc.sum()
        tmp = orig_cmc.cumsum()
        precision = (tmp / np.arange(1, len(tmp) + 1)) * orig_cmc
        all_ap.append(float(precision.sum() / num_rel))
    return all_ap, all_r1, valid


def summarize_subsets(distmat, q_pids, g_pids, q_camids, g_camids, subsets):
    all_ap, all_r1, valid = eval_per_query(distmat, q_pids, g_pids, q_camids, g_camids)
    idx_to_pos = {idx: pos for pos, idx in enumerate(valid)}
    summary = {}
    for name, subset_indices in subsets.items():
        valid_subset = [idx for idx in subset_indices if idx in idx_to_pos]
        if not valid_subset:
            continue
        subset_ap = [all_ap[idx_to_pos[idx]] for idx in valid_subset]
        subset_r1 = [all_r1[idx_to_pos[idx]] for idx in valid_subset]
        summary[name] = {
            'num_query': len(subset_indices),
            'num_valid': len(valid_subset),
            'mAP': float(np.mean(subset_ap) * 100.0),
            'R1': float(np.mean(subset_r1) * 100.0),
        }
    return summary


def build_subsets(num_persons, duplicate_flags, target_visible_counts):
    return {
        'all': list(range(len(num_persons))),
        'single': [i for i, n in enumerate(num_persons) if n == 1],
        'multi': [i for i, n in enumerate(num_persons) if n >= 2],
        'n=2': [i for i, n in enumerate(num_persons) if n == 2],
        'n=3': [i for i, n in enumerate(num_persons) if n == 3],
        'n>=4': [i for i, n in enumerate(num_persons) if n >= 4],
        'duplicate-suspect multi': [
            i for i, (n, dup) in enumerate(zip(num_persons, duplicate_flags))
            if n >= 2 and dup],
        'clean multi': [
            i for i, (n, dup) in enumerate(zip(num_persons, duplicate_flags))
            if n >= 2 and not dup],
        'target_vis<=8': [i for i, n in enumerate(target_visible_counts) if n <= 8],
        'target_vis<=5': [i for i, n in enumerate(target_visible_counts) if n <= 5],
    }


def common_visible_distmat_chunked(q_kp, g_kp, q_w, g_w, chunk_size=1024, device='cuda'):
    q_kp = q_kp.to(device)
    q_w = q_w.to(device)
    q_t = q_kp.transpose(1, 0)
    q_sq = q_t.pow(2).sum(dim=-1)
    q_w_t = q_w.transpose(1, 0)
    out = []
    for start in range(0, g_kp.shape[0], chunk_size):
        end = min(start + chunk_size, g_kp.shape[0])
        g_chunk = g_kp[start:end].to(device)
        gw_chunk = g_w[start:end].to(device)
        g_t = g_chunk.transpose(1, 0)
        g_sq = g_t.pow(2).sum(dim=-1)
        dot = torch.matmul(q_t, g_t.transpose(2, 1))
        kp_dist = (q_sq.unsqueeze(2) - 2 * dot + g_sq.unsqueeze(1)).clamp_min_(0.0).sqrt_()
        weights = torch.sqrt(q_w_t.unsqueeze(2) * gw_chunk.transpose(1, 0).unsqueeze(1))
        weight_sum = weights.sum(dim=0)
        masked = (kp_dist * weights).sum(dim=0) / weight_sum.clamp(min=1e-12)
        masked = torch.where(weight_sum > 0, masked, torch.full_like(masked, kp_dist.max().detach()))
        out.append(masked.cpu())
    return torch.cat(out, dim=1).numpy()


def extract_features(cfg_obj, model, val_loader, vis_thr, dup_iou_thr, dup_kp_dist_thr):
    model.eval()
    global_feats = []
    kp_feats = []
    kp_weights = []
    duplicate_flags = []
    num_persons_all = []
    target_visible_counts = []
    all_pids = []
    all_camids = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            img, pid, camid, camids, target_view, imgpath, pose_dict = batch
            pose_dict = _pose_to_device(pose_dict, 'cuda')
            img = img.cuda()
            camids = camids.cuda()
            target_view = target_view.cuda()

            test_feat, _ = model(img, cam_label=camids, view_label=target_view, pose_dict=pose_dict)
            global_feats.append(F.normalize(test_feat['global_feat'], dim=1).cpu())
            kp_feats.append(F.normalize(test_feat['kp_feats'], dim=2).cpu())
            kp_weights.append(test_feat['kp_weights'].float().cpu())

            batch_num_persons = pose_dict['num_persons'].detach().cpu().long().numpy()
            keypoints = pose_dict['keypoints'].detach().cpu().numpy()
            scores = pose_dict['scores'].detach().cpu().numpy()
            target_visible = (test_feat['kp_weights'].detach().cpu() > vis_thr).sum(dim=1).numpy()

            for b_idx in range(img.shape[0]):
                n_person = int(batch_num_persons[b_idx])
                duplicate_flags.append(
                    detect_duplicate_flag(keypoints[b_idx], scores[b_idx], n_person,
                                          vis_thr, dup_iou_thr, dup_kp_dist_thr))
                num_persons_all.append(n_person)
                target_visible_counts.append(int(target_visible[b_idx]))

            all_pids.extend(np.asarray(pid))
            all_camids.extend(np.asarray(camid))
            if (batch_idx + 1) % 20 == 0:
                print(f'  extracted batch {batch_idx + 1}/{len(val_loader)}', flush=True)

    return (
        torch.cat(global_feats, dim=0),
        torch.cat(kp_feats, dim=0).float(),
        torch.cat(kp_weights, dim=0).float(),
        np.asarray(duplicate_flags, dtype=np.bool_),
        np.asarray(num_persons_all, dtype=np.int64),
        np.asarray(target_visible_counts, dtype=np.int64),
        np.asarray(all_pids),
        np.asarray(all_camids),
    )


def build_oracle_recovery(kp_feats, kp_weights, pids, vis_thr, promote_weights):
    recovered_feats = kp_feats.clone()
    recovered_weights = kp_weights.clone()
    pid_to_indices = defaultdict(list)
    for idx, pid in enumerate(pids.tolist()):
        pid_to_indices[int(pid)].append(idx)

    samples_recovered = 0
    keypoints_recovered = 0
    support_count_total = 0

    for pid, indices in pid_to_indices.items():
        if len(indices) <= 1:
            continue
        idx_t = torch.tensor(indices, dtype=torch.long)
        feats_pid = kp_feats.index_select(0, idx_t)
        weights_pid = kp_weights.index_select(0, idx_t)
        vis_mask = weights_pid > vis_thr
        weighted_feats = feats_pid * weights_pid.unsqueeze(-1) * vis_mask.unsqueeze(-1).float()
        sum_feat = weighted_feats.sum(dim=0)
        sum_w = (weights_pid * vis_mask.float()).sum(dim=0)
        count = vis_mask.sum(dim=0)

        for local_idx, sample_idx in enumerate(indices):
            sample_vis = vis_mask[local_idx]
            local_sum_feat = sum_feat.clone()
            local_sum_w = sum_w.clone()
            local_count = count.clone()

            if sample_vis.any():
                vis_ids = sample_vis.nonzero(as_tuple=True)[0]
                local_sum_feat[vis_ids] -= weighted_feats[local_idx, vis_ids]
                local_sum_w[vis_ids] -= weights_pid[local_idx, vis_ids]
                local_count[vis_ids] -= 1

            recover_mask = (kp_weights[sample_idx] <= vis_thr) & (local_count > 0) & (local_sum_w > 1e-12)
            if not recover_mask.any():
                continue

            rec_ids = recover_mask.nonzero(as_tuple=True)[0]
            proto = local_sum_feat[rec_ids] / local_sum_w[rec_ids].unsqueeze(-1)
            proto = F.normalize(proto, dim=1)
            recovered_feats[sample_idx, rec_ids] = proto

            if promote_weights:
                proto_weight = local_sum_w[rec_ids] / local_count[rec_ids].clamp_min(1).float()
                recovered_weights[sample_idx, rec_ids] = proto_weight

            samples_recovered += 1
            keypoints_recovered += int(rec_ids.numel())
            support_count_total += int(local_count[rec_ids].sum().item())

    stats = {
        'samples_recovered': int(samples_recovered),
        'keypoints_recovered': int(keypoints_recovered),
        'avg_support_count': float(support_count_total / max(keypoints_recovered, 1)),
    }
    return recovered_feats, recovered_weights, stats


def print_metrics(tag, distmat, q_pids, g_pids, q_camids, g_camids, subsets):
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
    subset_summary = summarize_subsets(distmat, q_pids, g_pids, q_camids, g_camids, subsets)
    print(f'\n=== {tag} ===')
    print(f'  overall: mAP={mAP * 100:.2f}%, R1={cmc[0] * 100:.2f}%')
    for subset_name in ['single', 'multi', 'n=2', 'n=3', 'n>=4',
                        'duplicate-suspect multi', 'clean multi', 'target_vis<=8', 'target_vis<=5']:
        if subset_name not in subset_summary:
            continue
        item = subset_summary[subset_name]
        print(f"  {subset_name:<24s} n={item['num_valid']:4d}  mAP={item['mAP']:.2f}%  R1={item['R1']:.2f}%")
    return {'overall': {'mAP': float(mAP * 100.0), 'R1': float(cmc[0] * 100.0)}, 'subsets': subset_summary}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg.defrost()
    cfg.MODEL.POSE_ADDITIVE_ADAPTER = False
    cfg.MODEL.POSE_PAA_TARGET_ONLY = False
    cfg.MODEL.POSE_PAA_SCENE_TARGET = False
    cfg.MODEL.POSE_TDPC = False
    cfg.MODEL.POSE_ATTN_MASK = False
    cfg.MODEL.POSE_KP_RPE = False
    cfg.MODEL.POSE_COND_LORA = False
    cfg.MODEL.POSE_PAA_PART_STRUCTURED = False
    cfg.MODEL.POSE_PAA_ROUTED = False
    cfg.merge_from_file(args.config)
    cfg.TEST.WEIGHT = args.weight
    cfg.TEST.RE_RANKING = False
    cfg.MODEL.POSE_TEST_FEAT = 'cvk_hybrid'
    cfg.freeze()

    (_, _, val_loader, num_query, num_classes, camera_num, view_num) = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(args.weight)
    model = model.cuda().eval()

    print('Extracting global + target keypoint features...')
    global_feats, kp_feats, kp_weights, duplicate_flags, num_persons, target_visible_counts, pids, camids = extract_features(
        cfg, model, val_loader, args.vis_thr, args.dup_iou, args.dup_kp_dist)

    q_global = global_feats[:num_query]
    g_global = global_feats[num_query:]
    q_kp = kp_feats[:num_query]
    g_kp = kp_feats[num_query:]
    q_w = kp_weights[:num_query]
    g_w = kp_weights[num_query:]
    q_dup = duplicate_flags[:num_query]
    q_num = num_persons[:num_query]
    q_visible = target_visible_counts[:num_query]
    q_pids = pids[:num_query]
    g_pids = pids[num_query:]
    q_camids = camids[:num_query]
    g_camids = camids[num_query:]
    subsets = build_subsets(q_num, q_dup, q_visible)

    print('Building oracle support bank...')
    recovered_feat_only, _, stats_feat_only = build_oracle_recovery(
        kp_feats, kp_weights, pids, args.vis_thr, promote_weights=False)
    recovered_feat_weight, recovered_weights, stats_feat_weight = build_oracle_recovery(
        kp_feats, kp_weights, pids, args.vis_thr, promote_weights=True)

    q_rec_feat_only = recovered_feat_only[:num_query]
    g_rec_feat_only = recovered_feat_only[num_query:]
    q_rec_feat_weight = recovered_feat_weight[:num_query]
    g_rec_feat_weight = recovered_feat_weight[num_query:]
    q_rec_w = recovered_weights[:num_query]
    g_rec_w = recovered_weights[num_query:]

    print('Computing distance matrices...')
    global_dist = torch.cdist(q_global, g_global).cpu().numpy()
    base_kp_dist = common_visible_distmat_chunked(q_kp, g_kp, q_w, g_w)
    feat_only_kp_dist = common_visible_distmat_chunked(q_rec_feat_only, g_rec_feat_only, q_w, g_w)
    feat_weight_kp_dist = common_visible_distmat_chunked(q_rec_feat_weight, g_rec_feat_weight, q_rec_w, g_rec_w)

    results = {}
    results['base_cvk_hybrid'] = print_metrics(
        'base_cvk_hybrid', (global_dist + base_kp_dist) / 2.0,
        q_pids, g_pids, q_camids, g_camids, subsets)
    results['oracle_feat_only_cvk'] = print_metrics(
        'oracle_feat_only_cvk', (global_dist + feat_only_kp_dist) / 2.0,
        q_pids, g_pids, q_camids, g_camids, subsets)
    results['oracle_feat_weight_cvk'] = print_metrics(
        'oracle_feat_weight_cvk', (global_dist + feat_weight_kp_dist) / 2.0,
        q_pids, g_pids, q_camids, g_camids, subsets)

    summary = {
        'meta': {
            'config': args.config,
            'weight': args.weight,
            'vis_thr': args.vis_thr,
            'dup_iou': args.dup_iou,
            'dup_kp_dist': args.dup_kp_dist,
            'query_multi': int((q_num >= 2).sum()),
            'query_duplicate_suspect': int(q_dup.sum()),
            'query_target_vis<=8': int((q_visible <= 8).sum()),
            'query_target_vis<=5': int((q_visible <= 5).sum()),
        },
        'oracle_stats': {
            'feat_only': stats_feat_only,
            'feat_weight': stats_feat_weight,
        },
        'results': results,
    }
    out_path = os.path.join(args.output_dir, 'summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved summary to {out_path}')


if __name__ == '__main__':
    main()
