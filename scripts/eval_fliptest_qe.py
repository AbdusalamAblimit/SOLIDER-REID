"""
Flip-Test + Query Expansion evaluation.
Applies TTA (horizontal flip) and QE post-processing on top of MaxSim.

Usage:
    python scripts/eval_fliptest_qe.py \
        --config_file configs/occluded_duke/pose_psg_lgpa_detach.yml \
        --weight log/occluded_duke/exp255_small_gcn512_2stage/transformer_120.pth \
        [MODEL overrides...]
"""
import argparse
import sys
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from utils.metrics import eval_func
from processor.processor import do_inference_collect_structured


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--weight', required=True)
    parser.add_argument('--flip', action='store_true', default=True,
                        help='Enable flip-test TTA')
    parser.add_argument('--qe_k', type=int, default=5,
                        help='QE top-K neighbors (0=disabled)')
    parser.add_argument('--qe_alpha', type=float, default=0.5,
                        help='QE blending weight')
    parser.add_argument('--dba_k', type=int, default=0,
                        help='DBA top-K neighbors (0=disabled)')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def l2_normalize(x):
    return F.normalize(torch.from_numpy(x), dim=1).numpy()


def query_expansion(qf, gf, distmat, k=5, alpha=0.5):
    """Alpha Query Expansion: expand query with top-K gallery neighbors."""
    print(f"  QE: K={k}, alpha={alpha}")
    nq = qf.shape[0]
    qf_new = qf.copy()
    for i in range(nq):
        top_idx = np.argsort(distmat[i])[:k]
        neighbors = gf[top_idx]
        qf_new[i] = qf[i] + alpha * neighbors.mean(axis=0)
    qf_new = l2_normalize(qf_new)
    return qf_new


def database_augmentation(gf, k=5):
    """DBA: augment gallery with its own top-K neighbors."""
    print(f"  DBA: K={k}")
    ng = gf.shape[0]
    # Gallery-gallery distance
    gf_t = torch.from_numpy(gf)
    gg_dist = 1 - torch.mm(gf_t, gf_t.t()).numpy()
    gf_new = gf.copy()
    for i in range(ng):
        top_idx = np.argsort(gg_dist[i])[1:k+1]  # skip self
        neighbors = gf[top_idx]
        gf_new[i] = gf[i] + neighbors.mean(axis=0)
    gf_new = l2_normalize(gf_new)
    return gf_new


def main():
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.merge_from_list(['TEST.WEIGHT', args.weight])
    cfg.freeze()

    # Build model + dataloader
    from model.make_model import make_model
    from datasets.make_dataloader import make_dataloader

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=0,
                       semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(cfg.TEST.WEIGHT)
    model = model.cuda().eval()

    num_query_val = num_query

    print("=" * 60)
    print("Extracting features (original)...")
    feats_orig, pids, camids = do_inference_collect_structured(
        cfg, model, val_loader, num_query_val)

    if args.flip:
        print("Extracting features (flipped)...")
        # For flip-test, we need to flip images and heatmaps
        # Simple approach: extract with horizontal flip transform
        from datasets.pose_dataset import make_flip_val_loader
        try:
            flip_loader = make_flip_val_loader(cfg, val_loader)
            feats_flip, _, _ = do_inference_collect_structured(
                cfg, model, flip_loader, num_query_val)
            # Average original + flipped
            for key in feats_orig:
                if isinstance(feats_orig[key], np.ndarray):
                    feats_orig[key] = l2_normalize(
                        (feats_orig[key] + feats_flip[key]) / 2)
            print("  Flip-test: averaged original + flipped features")
        except Exception as e:
            print(f"  Flip-test failed: {e}. Using original features only.")

    # Extract query/gallery
    global_feats = feats_orig.get('global', feats_orig.get('feat'))
    qg = global_feats[:num_query_val]
    gg = global_feats[num_query_val:]
    qp = pids[:num_query_val]
    gp = pids[num_query_val:]
    qc = camids[:num_query_val]
    gc = camids[num_query_val:]

    # 1. Baseline cosine
    print("\n" + "=" * 60)
    print("1. Global Cosine (baseline)")
    distmat = 1 - np.dot(qg, gg.T)
    cmc, mAP = eval_func(distmat, qp, gp, qc, gc)
    print(f"   mAP={mAP*100:.1f}%  R1={cmc[0]*100:.1f}%")

    # 2. QE
    if args.qe_k > 0:
        print("\n" + "=" * 60)
        print(f"2. QE (K={args.qe_k}, alpha={args.qe_alpha})")
        qg_qe = query_expansion(qg, gg, distmat, k=args.qe_k, alpha=args.qe_alpha)

        # DBA
        if args.dba_k > 0:
            gg_dba = database_augmentation(gg, k=args.dba_k)
        else:
            gg_dba = gg

        distmat_qe = 1 - np.dot(qg_qe, gg_dba.T)
        cmc_qe, mAP_qe = eval_func(distmat_qe, qp, gp, qc, gc)
        print(f"   mAP={mAP_qe*100:.1f}%  R1={cmc_qe[0]*100:.1f}%  "
              f"(Δ mAP{(mAP_qe-mAP)*100:+.1f}%)")

    print("\nDone.")


if __name__ == '__main__':
    main()
