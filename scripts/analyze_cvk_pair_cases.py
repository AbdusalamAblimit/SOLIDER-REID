#!/usr/bin/env python
"""Compare query-level ranking changes between equal_concat and cvk_hybrid."""
import argparse
import csv
import json
import os
import sys
from copy import deepcopy

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from datasets import make_dataloader
from model import make_model
from processor.processor import _pose_to_device
from utils.metrics import R1_mAP_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Override OUTPUT_DIR for analysis artifacts",
    )
    parser.add_argument(
        "--topk",
        default=20,
        type=int,
        help="Number of improved/degraded queries to keep in markdown summary",
    )
    return parser.parse_args()


def _build_cfg(base_cfg, mode, output_dir, global_weight=1.0, kp_weight=1.0):
    cfg_local = base_cfg.clone()
    cfg_local.defrost()
    cfg_local.MODEL.POSE_TEST_FEAT = mode
    cfg_local.TEST.CVK_GLOBAL_WEIGHT = float(global_weight)
    cfg_local.TEST.CVK_KP_WEIGHT = float(kp_weight)
    cfg_local.OUTPUT_DIR = output_dir
    cfg_local.freeze()
    return cfg_local


def _run_mode(cfg_local, val_loader, num_query, num_classes, camera_num, view_num):
    model = make_model(
        cfg_local,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=cfg_local.MODEL.SEMANTIC_WEIGHT,
    )
    if cfg_local.TEST.WEIGHT:
        model.load_param(cfg_local.TEST.WEIGHT)

    device = "cuda"
    use_pose = cfg_local.MODEL.POSE_ENABLED
    evaluator = R1_mAP_eval(
        num_query,
        max_rank=50,
        feat_norm=cfg_local.TEST.FEAT_NORM,
        reranking=cfg_local.TEST.RE_RANKING,
        cfg=cfg_local,
    )
    evaluator.reset()
    model.to(device)
    model.eval()

    img_paths = []
    for batch_data in val_loader:
        with torch.no_grad():
            if use_pose:
                img, pid, camid, camids, target_view, batch_imgpaths, pose_dict = batch_data
                pose_dict = _pose_to_device(pose_dict, device)
            else:
                img, pid, camid, camids, target_view, batch_imgpaths = batch_data
                pose_dict = None
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            if use_pose:
                feat, _ = model(
                    img,
                    cam_label=camids,
                    view_label=target_view,
                    pose_dict=pose_dict,
                )
            else:
                feat, _ = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_paths.extend(batch_imgpaths)

    cmc, mAP, distmat, pids, camids, _, _ = evaluator.compute()
    torch.cuda.empty_cache()

    pids = np.asarray(pids)
    camids = np.asarray(camids)
    q_meta = {
        "paths": list(img_paths[:num_query]),
        "pids": pids[:num_query],
        "camids": camids[:num_query],
    }
    g_meta = {
        "paths": list(img_paths[num_query:]),
        "pids": pids[num_query:],
        "camids": camids[num_query:],
    }
    metrics = {
        "mAP": float(mAP),
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
    }
    return {
        "metrics": metrics,
        "distmat": distmat,
        "q_meta": q_meta,
        "g_meta": g_meta,
    }


def _per_query_stats(distmat, q_meta, g_meta):
    q_pids = q_meta["pids"]
    q_camids = q_meta["camids"]
    g_pids = g_meta["pids"]
    g_camids = g_meta["camids"]
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    rows = []
    for q_idx in range(distmat.shape[0]):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        kept_order = order[keep]
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        num_rel = int(orig_cmc.sum())
        tmp_cmc = orig_cmc.cumsum()
        denom = np.arange(1, tmp_cmc.shape[0] + 1, dtype=np.float64)
        precision = (tmp_cmc / denom) * orig_cmc
        ap = float(precision.sum() / num_rel)
        first_correct_rank = int(np.where(orig_cmc == 1)[0][0] + 1)
        top1_gallery_idx = int(kept_order[0])
        rows.append(
            {
                "query_idx": q_idx,
                "query_path": q_meta["paths"][q_idx],
                "query_pid": int(q_pid),
                "query_camid": int(q_camid),
                "ap": ap,
                "first_correct_rank": first_correct_rank,
                "top1_correct": bool(orig_cmc[0]),
                "top1_gallery_idx": top1_gallery_idx,
                "top1_gallery_path": g_meta["paths"][top1_gallery_idx],
                "top1_gallery_pid": int(g_pids[top1_gallery_idx]),
                "top1_gallery_camid": int(g_camids[top1_gallery_idx]),
            }
        )
    return rows


def _merge_rows(eq_rows, cvk_rows):
    eq_map = {row["query_idx"]: row for row in eq_rows}
    cvk_map = {row["query_idx"]: row for row in cvk_rows}
    merged = []
    for q_idx in sorted(eq_map.keys()):
        eq_row = eq_map[q_idx]
        cvk_row = cvk_map[q_idx]
        if (not eq_row["top1_correct"]) and cvk_row["top1_correct"]:
            status = "top1_fixed"
        elif eq_row["top1_correct"] and (not cvk_row["top1_correct"]):
            status = "top1_degraded"
        elif eq_row["top1_correct"] and cvk_row["top1_correct"]:
            status = "both_top1_correct"
        else:
            status = "both_top1_wrong"

        merged.append(
            {
                "query_idx": q_idx,
                "query_path": eq_row["query_path"],
                "query_pid": eq_row["query_pid"],
                "query_camid": eq_row["query_camid"],
                "eq_ap": eq_row["ap"],
                "cvk_ap": cvk_row["ap"],
                "delta_ap": cvk_row["ap"] - eq_row["ap"],
                "eq_first_correct_rank": eq_row["first_correct_rank"],
                "cvk_first_correct_rank": cvk_row["first_correct_rank"],
                "rank_gain": eq_row["first_correct_rank"] - cvk_row["first_correct_rank"],
                "eq_top1_correct": eq_row["top1_correct"],
                "cvk_top1_correct": cvk_row["top1_correct"],
                "status": status,
                "eq_top1_gallery_path": eq_row["top1_gallery_path"],
                "eq_top1_gallery_pid": eq_row["top1_gallery_pid"],
                "eq_top1_gallery_camid": eq_row["top1_gallery_camid"],
                "cvk_top1_gallery_path": cvk_row["top1_gallery_path"],
                "cvk_top1_gallery_pid": cvk_row["top1_gallery_pid"],
                "cvk_top1_gallery_camid": cvk_row["top1_gallery_camid"],
            }
        )
    return merged


def _write_query_csv(output_dir, rows):
    csv_path = os.path.join(output_dir, "query_deltas.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _write_summary(output_dir, eq_metrics, cvk_metrics, rows, topk):
    delta_aps = np.asarray([row["delta_ap"] for row in rows], dtype=np.float64)
    top1_fixed = [row for row in rows if row["status"] == "top1_fixed"]
    top1_degraded = [row for row in rows if row["status"] == "top1_degraded"]
    improved = sorted(rows, key=lambda x: x["delta_ap"], reverse=True)
    degraded = sorted(rows, key=lambda x: x["delta_ap"])

    summary = {
        "equal_concat": eq_metrics,
        "cvk_hybrid": cvk_metrics,
        "delta": {
            "mAP": cvk_metrics["mAP"] - eq_metrics["mAP"],
            "rank1": cvk_metrics["rank1"] - eq_metrics["rank1"],
            "rank5": cvk_metrics["rank5"] - eq_metrics["rank5"],
            "rank10": cvk_metrics["rank10"] - eq_metrics["rank10"],
        },
        "query_stats": {
            "num_queries": len(rows),
            "mean_delta_ap": float(delta_aps.mean()),
            "median_delta_ap": float(np.median(delta_aps)),
            "positive_delta_ap": int((delta_aps > 0).sum()),
            "negative_delta_ap": int((delta_aps < 0).sum()),
            "zero_delta_ap": int((delta_aps == 0).sum()),
            "top1_fixed": len(top1_fixed),
            "top1_degraded": len(top1_degraded),
            "both_top1_correct": int(sum(row["status"] == "both_top1_correct" for row in rows)),
            "both_top1_wrong": int(sum(row["status"] == "both_top1_wrong" for row in rows)),
        },
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# exp042 Pair-Case Summary\n\n")
        f.write("## Aggregate\n")
        f.write(
            f"- `equal_concat`: {eq_metrics['mAP']:.1%} / {eq_metrics['rank1']:.1%}\n"
        )
        f.write(
            f"- `cvk_hybrid`: {cvk_metrics['mAP']:.1%} / {cvk_metrics['rank1']:.1%}\n"
        )
        f.write(
            f"- delta: mAP {summary['delta']['mAP']:+.1%}, "
            f"R1 {summary['delta']['rank1']:+.1%}\n\n"
        )
        f.write("## Query Stats\n")
        for key, value in summary["query_stats"].items():
            f.write(f"- `{key}`: {value}\n")
        f.write("\n## Top Improved Queries\n")
        for row in improved[:topk]:
            f.write(
                f"- q{row['query_idx']}: delta_ap={row['delta_ap']:+.4f}, "
                f"status={row['status']}, eq_rank={row['eq_first_correct_rank']}, "
                f"cvk_rank={row['cvk_first_correct_rank']}, path={row['query_path']}\n"
            )
        f.write("\n## Top Degraded Queries\n")
        for row in degraded[:topk]:
            f.write(
                f"- q{row['query_idx']}: delta_ap={row['delta_ap']:+.4f}, "
                f"status={row['status']}, eq_rank={row['eq_first_correct_rank']}, "
                f"cvk_rank={row['cvk_first_correct_rank']}, path={row['query_path']}\n"
            )
    return summary_path, md_path


def main():
    args = parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    output_dir = args.output_dir or cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    eq_cfg = _build_cfg(cfg, "equal_concat", output_dir)
    cvk_cfg = _build_cfg(cfg, "cvk_hybrid", output_dir, global_weight=1.0, kp_weight=1.0)

    print("=== Running equal_concat ===")
    eq_out = _run_mode(eq_cfg, val_loader, num_query, num_classes, camera_num, view_num)
    print("=== Running cvk_hybrid ===")
    cvk_out = _run_mode(cvk_cfg, val_loader, num_query, num_classes, camera_num, view_num)

    eq_rows = _per_query_stats(eq_out["distmat"], eq_out["q_meta"], eq_out["g_meta"])
    cvk_rows = _per_query_stats(cvk_out["distmat"], cvk_out["q_meta"], cvk_out["g_meta"])
    merged_rows = _merge_rows(eq_rows, cvk_rows)

    csv_path = _write_query_csv(output_dir, merged_rows)
    summary_path, md_path = _write_summary(
        output_dir,
        eq_out["metrics"],
        cvk_out["metrics"],
        merged_rows,
        args.topk,
    )

    print(f"Saved query delta CSV to {csv_path}")
    print(f"Saved summary JSON to {summary_path}")
    print(f"Saved summary Markdown to {md_path}")


if __name__ == "__main__":
    main()
