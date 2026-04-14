"""
Flip-test TTA + MaxSim evaluation.
Extracts features twice (original + flipped), averages, then runs MaxSim.

Usage:
    python scripts/eval_fliptest_maxsim.py \
        --config_file configs/occluded_duke/pose_psg_lgpa_gcn512_2stage_small.yml \
        --weight log/occluded_duke/exp255_small_gcn512_2stage/transformer_120.pth \
        DATALOADER.NUM_WORKERS 2
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from datasets.make_dataloader import make_dataloader
from datasets.pose_dataset import FLIP_PAIRS
from model.make_model import make_model
from utils.metrics import eval_func


def flip_batch(imgs, pose_dict):
    """Flip images and pose data for TTA."""
    imgs_flip = imgs.flip(-1)  # horizontal flip (W dimension)

    if pose_dict is None:
        return imgs_flip, None

    pd = {}
    for k, v in pose_dict.items():
        if isinstance(v, torch.Tensor):
            pd[k] = v.clone()
        else:
            pd[k] = v

    # Flip heatmaps: (B, max_persons, 17, H, W) -> flip W
    if 'heatmaps' in pd:
        hm = pd['heatmaps'].flip(-1)
        # Swap left-right channels
        for l, r in FLIP_PAIRS:
            hm[:, :, [l, r]] = hm[:, :, [r, l]]
        pd['heatmaps'] = hm

    # Flip keypoints x coordinates
    if 'keypoints' in pd:
        kp = pd['keypoints'].clone()
        # keypoints shape: (B, max_persons, 17, 2)
        # x = img_width - 1 - x
        img_w = imgs.shape[-1]  # W dimension
        kp[..., 0] = img_w - 1 - kp[..., 0]
        # Swap left-right keypoints
        for l, r in FLIP_PAIRS:
            kp[:, :, [l, r]] = kp[:, :, [r, l]]
        pd['keypoints'] = kp

    # Swap left-right scores
    if 'scores' in pd:
        sc = pd['scores'].clone()
        for l, r in FLIP_PAIRS:
            sc[:, :, [l, r]] = sc[:, :, [r, l]]
        pd['scores'] = sc

    return imgs_flip, pd


def extract_features_flip(model, loader, device='cuda', do_flip=True):
    """Extract features with optional flip-test TTA."""
    model.eval()
    all_global_orig, all_parts_orig, all_vis_orig = [], [], []
    all_global_flip, all_parts_flip, all_vis_flip = [], [], []
    all_pids, all_camids = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            if len(batch) == 7:
                imgs, pid, camid, _, _, _, pose_dict = batch
            else:
                imgs, pid, camid = batch[0], batch[1], batch[2]
                pose_dict = batch[-1] if isinstance(batch[-1], dict) else None

            img = imgs.to(device)
            if pose_dict and isinstance(pose_dict, dict):
                pose_dict = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in pose_dict.items()
                }

            # Original forward
            feat, _ = model(img, pose_dict=pose_dict)

            if isinstance(feat, dict):
                g = F.normalize(feat['global_feat'], p=2, dim=1)
                kp = F.normalize(feat['kp_feats'], p=2, dim=2)
                w = feat['kp_weights']
            else:
                # Auto-detect feature dim: try 1024 (Base), 768 (Small/Tiny)
                C = 1024 if feat.shape[1] % 1024 == 0 else 768
                n_feats = feat.shape[1] // C
                g = F.normalize(feat[:, :C], p=2, dim=1)
                if n_feats > 1:
                    kp = F.normalize(feat[:, C:].reshape(-1, n_feats - 1, C), p=2, dim=2)
                    w = torch.ones(img.shape[0], n_feats - 1, device=device)
                else:
                    kp = None
                    w = None

            all_global_orig.append(g.cpu())
            if kp is not None:
                all_parts_orig.append(kp.cpu())
                all_vis_orig.append(w.cpu())

            # Flipped forward
            if do_flip:
                img_flip, pd_flip = flip_batch(imgs.to(device), pose_dict)
                feat_f, _ = model(img_flip, pose_dict=pd_flip)

                if isinstance(feat_f, dict):
                    gf = F.normalize(feat_f['global_feat'], p=2, dim=1)
                    kpf = F.normalize(feat_f['kp_feats'], p=2, dim=2)
                    wf = feat_f['kp_weights']
                else:
                    gf = F.normalize(feat_f[:, :C], p=2, dim=1)
                    if n_feats > 1:
                        kpf = F.normalize(feat_f[:, C:].reshape(-1, n_feats - 1, C), p=2, dim=2)
                        wf = torch.ones(img.shape[0], n_feats - 1, device=device)
                    else:
                        kpf = None
                        wf = None

                all_global_flip.append(gf.cpu())
                if kpf is not None:
                    all_parts_flip.append(kpf.cpu())
                    all_vis_flip.append(wf.cpu())

            all_pids.append(pid if isinstance(pid, torch.Tensor) else torch.tensor(pid))
            all_camids.append(camid if isinstance(camid, torch.Tensor) else torch.tensor(camid))

    # Concatenate
    g_orig = torch.cat(all_global_orig)
    pids = torch.cat(all_pids).numpy()
    camids = torch.cat(all_camids).numpy()

    p_orig = torch.cat(all_parts_orig) if all_parts_orig else None
    v_orig = torch.cat(all_vis_orig) if all_vis_orig else None

    if do_flip:
        g_flip = torch.cat(all_global_flip)
        p_flip = torch.cat(all_parts_flip) if all_parts_flip else None
        v_flip = torch.cat(all_vis_flip) if all_vis_flip else None

        # Average original + flipped, then re-normalize
        g_avg = F.normalize((g_orig + g_flip) / 2, p=2, dim=1)
        if p_orig is not None and p_flip is not None:
            p_avg = F.normalize((p_orig + p_flip) / 2, p=2, dim=2)
            v_avg = (v_orig + v_flip) / 2
        else:
            p_avg, v_avg = p_orig, v_orig
    else:
        g_avg = g_orig
        p_avg = p_orig
        v_avg = v_orig

    return g_avg, p_avg, v_avg, pids, camids


def compute_maxsim_distmat(q_global, g_global, q_parts, g_parts, q_vis, g_vis,
                            global_weight=1.0, batch_size=1000):
    """MaxSim hybrid distance (same as eval_pot.py)."""
    nq, ng = q_global.shape[0], g_global.shape[0]
    gd = 1 - torch.mm(q_global, g_global.t())

    distmat = np.zeros((nq, ng), dtype=np.float32)
    for qi in tqdm(range(0, nq, batch_size), desc="MaxSim"):
        qe = min(qi + batch_size, nq)
        q_batch = q_parts[qi:qe]
        q_w = q_vis[qi:qe]
        for gi in range(0, ng, batch_size):
            ge = min(gi + batch_size, ng)
            sim_block = torch.einsum('bkc,gpc->bgkp', q_batch, g_parts[gi:ge])
            maxsim = sim_block.max(dim=3)[0]
            weighted = (maxsim * q_w.unsqueeze(1)).sum(dim=2) / q_w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            distmat[qi:qe, gi:ge] = (1 - weighted).numpy()

    gd_np = gd.numpy()
    alpha = global_weight / (global_weight + 1.0)
    return alpha * gd_np + (1 - alpha) * distmat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--weight', required=True)
    parser.add_argument('--no_flip', action='store_true', help='Disable flip-test')
    args, remaining = parser.parse_known_args()

    cfg.merge_from_file(args.config_file)
    if remaining:
        cfg.merge_from_list(remaining)
    cfg.TEST.WEIGHT = args.weight
    cfg.freeze()

    _, _, val_loader, num_query, num_classes, _, _ = make_dataloader(cfg)
    model = make_model(cfg, num_classes, 8, 0, cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(args.weight)
    model.cuda()

    do_flip = not args.no_flip
    print(f"Flip-test: {'ON' if do_flip else 'OFF'}")

    global_feats, part_feats, vis_weights, pids, camids = \
        extract_features_flip(model, val_loader, do_flip=do_flip)

    del model
    torch.cuda.empty_cache()
    print("GPU released.")

    nq = num_query
    qg, gg = global_feats[:nq], global_feats[nq:]
    qp, gp = pids[:nq], pids[nq:]
    qc, gc = camids[:nq], camids[nq:]

    # 1. Global cosine
    dist_global = (1 - torch.mm(qg, gg.t())).numpy()
    cmc, mAP = eval_func(dist_global, qp, gp, qc, gc)
    flip_tag = "+flip" if do_flip else ""
    print(f"\nGlobal cosine{flip_tag}: mAP={mAP*100:.1f}%  R1={cmc[0]*100:.1f}%")

    if part_feats is not None:
        qparts = part_feats[:nq]
        gparts = part_feats[nq:]
        qvis = vis_weights[:nq]
        gvis = vis_weights[nq:]

        # 2. MaxSim hybrid
        dist_maxsim = compute_maxsim_distmat(qg, gg, qparts, gparts, qvis, gvis, global_weight=1.0)
        cmc_ms, mAP_ms = eval_func(dist_maxsim, qp, gp, qc, gc)
        print(f"MaxSim hybrid{flip_tag}: mAP={mAP_ms*100:.1f}%  R1={cmc_ms[0]*100:.1f}%")

    print("\nDone.")


if __name__ == '__main__':
    main()
