#!/usr/bin/env python
"""
DACCM: Duplicate-Aware Counterfactual Common-Support Matching

Training-free reranking on top of cvk_hybrid-style target-target matching.
Unlike exp107 (pooled skeleton hypothesis), this script reasons directly in
per-keypoint / common-support space.
"""

import argparse
import json
import os
import sys

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
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--gap_cap', type=float, default=0.1)
    parser.add_argument('--score_thr', type=float, default=0.3)
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


def build_keep_indices(keypoints, scores, num_persons, score_thr, dup_iou_thr, dup_kp_dist_thr):
    if num_persons <= 0:
        return [], False
    duplicate_suspect = False
    for i in range(num_persons):
        for j in range(i + 1, num_persons):
            if is_duplicate_person(keypoints[i], scores[i], keypoints[j], scores[j],
                                   score_thr, dup_iou_thr, dup_kp_dist_thr):
                duplicate_suspect = True
                break
        if duplicate_suspect:
            break
    kept = [0]
    distractors = list(range(1, num_persons))
    distractors.sort(key=lambda idx: float(scores[idx].mean()), reverse=True)
    for idx in distractors:
        dup = False
        for kept_idx in kept:
            if is_duplicate_person(keypoints[idx], scores[idx], keypoints[kept_idx], scores[kept_idx],
                                   score_thr, dup_iou_thr, dup_kp_dist_thr):
                dup = True
                break
        if not dup:
            kept.append(idx)
    return kept, duplicate_suspect


def reorder_pose_dict(pose_dict, person_idx):
    order = [person_idx] + [i for i in range(pose_dict['heatmaps'].shape[1]) if i != person_idx]
    order_t = torch.tensor(order, device=pose_dict['heatmaps'].device)
    out = {}
    for key, value in pose_dict.items():
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == len(order):
            out[key] = value.index_select(1, order_t)
        else:
            out[key] = value
    return out


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


def build_subsets(num_persons, duplicate_flags):
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
        fallback = kp_dist.max().detach() + 1.0
        masked = torch.where(weight_sum > 0, masked, torch.full_like(masked, fallback))
        out.append(masked.cpu())
    return torch.cat(out, dim=1).numpy()


def pair_kp_distance_single(q_feat, g_feat, q_w, g_w):
    dist = torch.norm(q_feat.float() - g_feat.float(), dim=1)
    weights = torch.sqrt(q_w.float() * g_w.float())
    weight_sum = weights.sum()
    if float(weight_sum.item()) <= 1e-12:
        return None
    return float((dist * weights).sum().item() / weight_sum.item())


def extract_features(cfg_obj, model, val_loader, score_thr, dup_iou_thr, dup_kp_dist_thr):
    model.eval()
    max_persons = 6
    global_feats = []
    person_kp_feats = []
    person_kp_weights = []
    keep_indices = []
    duplicate_flags = []
    num_persons_all = []
    all_pids = []
    all_camids = []
    _m = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            img, pid, camid, camids, target_view, imgpath, pose_dict = batch
            pose_dict = _pose_to_device(pose_dict, 'cuda')
            img = img.cuda()
            camids = camids.cuda()
            target_view = target_view.cuda()

            test_feat, featmaps = model(img, cam_label=camids, view_label=target_view, pose_dict=pose_dict)
            global_feats.append(test_feat['global_feat'].cpu())
            featmap = featmaps[-1]
            batch_num_persons = pose_dict['num_persons'].detach().cpu().long()
            batch_feats = torch.zeros(img.shape[0], max_persons, 17, 768, dtype=torch.float16, device='cpu')
            batch_weights = torch.zeros(img.shape[0], max_persons, 17, dtype=torch.float16, device='cpu')

            for p_idx in range(max_persons):
                active = (batch_num_persons > p_idx).nonzero(as_tuple=True)[0]
                if len(active) == 0:
                    continue
                sub_pose = {}
                for key, value in pose_dict.items():
                    if torch.is_tensor(value):
                        sub_pose[key] = value[active]
                    else:
                        sub_pose[key] = value
                if p_idx > 0:
                    sub_pose = reorder_pose_dict(sub_pose, p_idx)
                _, _, aux = _m.skeleton_head(featmap[active], sub_pose, return_cls=False)
                batch_feats[active, p_idx] = F.normalize(aux['kp_feats'], dim=2).half().cpu()
                batch_weights[active, p_idx] = aux['kp_weights'].half().cpu()

            kp_cpu = pose_dict['keypoints'].detach().cpu().numpy()
            sc_cpu = pose_dict['scores'].detach().cpu().numpy()
            for b_idx in range(img.shape[0]):
                n_person = int(batch_num_persons[b_idx].item())
                kept, dup_flag = build_keep_indices(kp_cpu[b_idx], sc_cpu[b_idx], n_person,
                                                    score_thr, dup_iou_thr, dup_kp_dist_thr)
                person_kp_feats.append(batch_feats[b_idx, :n_person].clone())
                person_kp_weights.append(batch_weights[b_idx, :n_person].clone())
                keep_indices.append(kept)
                duplicate_flags.append(bool(dup_flag))
                num_persons_all.append(n_person)
            all_pids.extend(np.asarray(pid))
            all_camids.extend(np.asarray(camid))
            if (batch_idx + 1) % 20 == 0:
                print(f'  extracted batch {batch_idx + 1}/{len(val_loader)}', flush=True)

    return (
        F.normalize(torch.cat(global_feats, dim=0), dim=1),
        person_kp_feats,
        person_kp_weights,
        keep_indices,
        np.asarray(duplicate_flags, dtype=np.bool_),
        np.asarray(num_persons_all, dtype=np.int64),
        np.asarray(all_pids),
        np.asarray(all_camids),
    )


def print_metrics(tag, distmat, q_pids, g_pids, q_camids, g_camids, subsets):
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
    subset_summary = summarize_subsets(distmat, q_pids, g_pids, q_camids, g_camids, subsets)
    print(f'\n=== {tag} ===')
    print(f'  overall: mAP={mAP * 100:.2f}%, R1={cmc[0] * 100:.2f}%')
    for subset_name in ['single', 'multi', 'n=2', 'n=3', 'n>=4', 'duplicate-suspect multi', 'clean multi']:
        if subset_name not in subset_summary:
            continue
        item = subset_summary[subset_name]
        print(f"  {subset_name:<24s} n={item['num_valid']:4d}  mAP={item['mAP']:.2f}%  R1={item['R1']:.2f}%")
    return {'overall': {'mAP': float(mAP * 100.0), 'R1': float(cmc[0] * 100.0)}, 'subsets': subset_summary}


def rerank_penalty(base_distmat, base_kp_dist, q_feats_list, g_feats_list, q_w_list, g_w_list,
                   q_keep, g_keep, topk, alpha, gap_cap, use_dedup):
    reranked = base_distmat.copy()
    rank = np.argsort(base_distmat, axis=1)
    topk = min(topk, base_distmat.shape[1])
    for q_idx in range(base_distmat.shape[0]):
        q_ids = q_keep[q_idx] if use_dedup else list(range(q_feats_list[q_idx].shape[0]))
        q_ids = [idx for idx in q_ids if idx < q_feats_list[q_idx].shape[0]]
        for g_idx in rank[q_idx, :topk]:
            g_ids = g_keep[g_idx] if use_dedup else list(range(g_feats_list[g_idx].shape[0]))
            g_ids = [idx for idx in g_ids if idx < g_feats_list[g_idx].shape[0]]
            d_tt = float(base_kp_dist[q_idx, g_idx])
            conf = []
            if len(g_ids) > 1:
                for idx in g_ids[1:]:
                    d = pair_kp_distance_single(q_feats_list[q_idx][0], g_feats_list[g_idx][idx],
                                                q_w_list[q_idx][0], g_w_list[g_idx][idx])
                    if d is not None:
                        conf.append(d)
            if len(q_ids) > 1:
                for idx in q_ids[1:]:
                    d = pair_kp_distance_single(q_feats_list[q_idx][idx], g_feats_list[g_idx][0],
                                                q_w_list[q_idx][idx], g_w_list[g_idx][0])
                    if d is not None:
                        conf.append(d)
            if not conf:
                continue
            support_gap = float(np.clip(min(conf) - d_tt, -gap_cap, gap_cap))
            penalty = max(0.0, -support_gap)
            reranked[q_idx, g_idx] = base_distmat[q_idx, g_idx] + alpha * penalty
    return reranked


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

    print('Extracting global + per-person keypoint features...')
    global_feats, person_kp_feats, person_kp_weights, keep_indices, duplicate_flags, num_persons, pids, camids = extract_features(
        cfg, model, val_loader, args.score_thr, args.dup_iou, args.dup_kp_dist)

    q_global = global_feats[:num_query]
    g_global = global_feats[num_query:]
    q_person_feats = person_kp_feats[:num_query]
    g_person_feats = person_kp_feats[num_query:]
    q_person_w = person_kp_weights[:num_query]
    g_person_w = person_kp_weights[num_query:]
    q_keep = keep_indices[:num_query]
    g_keep = keep_indices[num_query:]
    q_dup = duplicate_flags[:num_query]
    q_num = num_persons[:num_query]
    q_pids = pids[:num_query]
    g_pids = pids[num_query:]
    q_camids = camids[:num_query]
    g_camids = camids[num_query:]
    subsets = build_subsets(q_num, q_dup)

    print('Computing base cvk_hybrid distance matrix...')
    global_dist = torch.cdist(q_global, g_global).cpu().numpy()
    q_target_feats = torch.stack([x[0].float() for x in q_person_feats], dim=0)
    g_target_feats = torch.stack([x[0].float() for x in g_person_feats], dim=0)
    q_target_w = torch.stack([x[0].float() for x in q_person_w], dim=0)
    g_target_w = torch.stack([x[0].float() for x in g_person_w], dim=0)
    kp_dist = common_visible_distmat_chunked(q_target_feats, g_target_feats, q_target_w, g_target_w)
    base_dist = (global_dist + kp_dist) / 2.0

    results = {}
    results['base_cvk_hybrid'] = print_metrics('base_cvk_hybrid', base_dist, q_pids, g_pids, q_camids, g_camids, subsets)

    print('\nReranking with raw_daccm_penalty...')
    raw_dist = rerank_penalty(base_dist, kp_dist, q_person_feats, g_person_feats, q_person_w, g_person_w,
                              q_keep, g_keep, args.topk, args.alpha, args.gap_cap, use_dedup=False)
    results['raw_daccm_penalty'] = print_metrics('raw_daccm_penalty', raw_dist, q_pids, g_pids, q_camids, g_camids, subsets)

    print('\nReranking with daccm_penalty...')
    dedup_dist = rerank_penalty(base_dist, kp_dist, q_person_feats, g_person_feats, q_person_w, g_person_w,
                                q_keep, g_keep, args.topk, args.alpha, args.gap_cap, use_dedup=True)
    results['daccm_penalty'] = print_metrics('daccm_penalty', dedup_dist, q_pids, g_pids, q_camids, g_camids, subsets)

    meta = {
        'config': args.config,
        'weight': args.weight,
        'topk': args.topk,
        'alpha': args.alpha,
        'gap_cap': args.gap_cap,
        'score_thr': args.score_thr,
        'dup_iou': args.dup_iou,
        'dup_kp_dist': args.dup_kp_dist,
        'query_duplicate_suspect': int(q_dup.sum()),
        'query_multi': int((q_num >= 2).sum()),
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump({'meta': meta, 'results': results}, f, indent=2)
    print(f"\nSaved summary to {os.path.join(args.output_dir, 'summary.json')}")


if __name__ == '__main__':
    main()
