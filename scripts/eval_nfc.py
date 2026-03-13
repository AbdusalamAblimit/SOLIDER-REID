#!/usr/bin/env python
"""Quick NFC evaluation on existing exp030a checkpoint.

Usage:
    python scripts/eval_nfc.py --config configs/occluded_duke/pose_psg_gcn.yml \
        --k1 2 --k2 2 --mode equal_concat
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import cfg
from datasets.make_dataloader import make_dataloader
from model import make_model
from processor.processor import _pose_to_device
from utils.metrics import R1_mAP_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--k1', type=int, default=2)
    parser.add_argument('--k2', type=int, default=2)
    parser.add_argument('--mode', default='equal_concat',
                        choices=['global', 'equal_concat', 'concat_scaled'])
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    # Override NFC settings
    cfg.TEST.NFC = True
    cfg.TEST.NFC_K1 = args.k1
    cfg.TEST.NFC_K2 = args.k2
    # Set test mode
    cfg.MODEL.POSE_TEST_FEAT = args.mode
    cfg.freeze()

    # Determine checkpoint
    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = os.path.join(cfg.OUTPUT_DIR,
                                  cfg.MODEL.NAME + '_120.pth')
    print(f'Config: {args.config}')
    print(f'Checkpoint: {checkpoint}')
    print(f'Mode: {args.mode}')
    print(f'NFC: k1={args.k1}, k2={args.k2}')

    # Build dataloader
    _, _, val_loader, num_query, num_classes, _, _ = make_dataloader(cfg)

    # Build model
    model = make_model(cfg, num_class=num_classes, camera_num=0,
                       view_num=0, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.cuda().eval()

    # Build evaluator with NFC enabled
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=True,
                            reranking=False, cfg=cfg)
    evaluator.reset()

    use_pose = cfg.MODEL.POSE_ENABLED

    # Extract features
    print('Extracting features...')
    with torch.no_grad():
        for batch_data in val_loader:
            if use_pose:
                img, vid, camid, camids, target_view, _, pose_dict = batch_data
                pose_dict = _pose_to_device(pose_dict, 'cuda')
            else:
                img, vid, camid, camids, target_view, _ = batch_data
                pose_dict = None
            img = img.cuda()
            camids = camids.cuda()
            target_view = target_view.cuda()
            if use_pose:
                feat, _ = model(img, cam_label=camids, view_label=target_view,
                                pose_dict=pose_dict)
            else:
                feat, _ = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, vid, camid))

    # Compute with NFC
    print('Computing metrics with NFC...')
    cmc, mAP, _, _, _, _, _ = evaluator.compute()

    print(f'\n=== Results ({args.mode} + NFC k1={args.k1} k2={args.k2}) ===')
    print(f'mAP: {mAP:.4f} ({mAP*100:.1f}%)')
    for r in [1, 5, 10]:
        print(f'Rank-{r}: {cmc[r-1]:.4f} ({cmc[r-1]*100:.1f}%)')


if __name__ == '__main__':
    main()
