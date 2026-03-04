"""
Offline keypoint extraction using ViTPose + VisPredictHead.

Extracts 17 COCO keypoints + visibility scores for all Occluded-Duke images.
Saves per-split JSON files: {img_name: {"keypoints": [[x,y,conf],...], "visibility": [v1,...,v17]}}

Usage:
    python tools/extract_keypoints.py \
        --pose-config pose/config_vispredict.py \
        --pose-checkpoint pretrained/best_coco_AP_epoch_210.pth \
        --data-root data/occluded_duke \
        --output-dir data/occluded_duke/pose \
        --batch-size 64 \
        --device cuda:0
"""

import argparse
import json
import os
import os.path as osp
import glob
import numpy as np
import torch
from tqdm import tqdm

from mmpose.apis import init_model
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description='Extract keypoints with ViTPose')
    parser.add_argument('--pose-config', type=str,
                        default='pose/config_vispredict.py')
    parser.add_argument('--pose-checkpoint', type=str,
                        default='pretrained/best_coco_AP_epoch_210.pth')
    parser.add_argument('--data-root', type=str,
                        default='data/occluded_duke')
    parser.add_argument('--output-dir', type=str,
                        default='data/occluded_duke/pose')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def extract_for_split(model, img_dir, output_path, batch_size, device):
    """Extract keypoints for all images in a directory."""
    img_paths = sorted(glob.glob(osp.join(img_dir, '*.jpg')) +
                       glob.glob(osp.join(img_dir, '*.png')))
    print(f"Processing {len(img_paths)} images from {img_dir}")

    if len(img_paths) == 0:
        print(f"  No images found, skipping.")
        return

    results = {}

    # Process in batches
    from mmpose.apis import inference_topdown

    for i in tqdm(range(0, len(img_paths), batch_size), desc=osp.basename(img_dir)):
        batch_paths = img_paths[i:i + batch_size]

        for img_path in batch_paths:
            img_name = osp.basename(img_path)
            img = mmcv.imread(img_path)
            h, w = img.shape[:2]

            # Create bounding box covering full image (person crops)
            bboxes = [[0, 0, w, h]]

            # Run inference
            pose_results = inference_topdown(model, img_path, bboxes, bbox_format='xyxy')

            if len(pose_results) > 0:
                result = pose_results[0]
                # keypoints: (17, 2) x,y coordinates
                # keypoint_scores: (17,) confidence
                kpts = result.pred_instances.keypoints[0]  # (17, 2)
                kpt_scores = result.pred_instances.keypoint_scores[0]  # (17,)

                # Get visibility from VisPredictHead if available
                if hasattr(result.pred_instances, 'keypoints_visible'):
                    vis = result.pred_instances.keypoints_visible[0]  # (17,)
                else:
                    vis = kpt_scores  # fallback to confidence scores

                # Combine into (17, 3): x, y, confidence
                keypoints_with_conf = np.concatenate([
                    kpts, kpt_scores.reshape(-1, 1)
                ], axis=1).tolist()  # (17, 3)

                results[img_name] = {
                    'keypoints': keypoints_with_conf,
                    'visibility': vis.tolist(),
                    'img_hw': [h, w]
                }
            else:
                # No detection - fill with zeros
                results[img_name] = {
                    'keypoints': [[0.0, 0.0, 0.0]] * 17,
                    'visibility': [0.0] * 17,
                    'img_hw': [h, w]
                }

    # Save results
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} results to {output_path}")


def main():
    args = parse_args()

    # Initialize model
    print(f"Loading ViTPose model from {args.pose_checkpoint}")
    model = init_model(args.pose_config, args.pose_checkpoint, device=args.device)
    model.eval()

    # Process each split
    splits = {
        'train': osp.join(args.data_root, 'bounding_box_train'),
        'query': osp.join(args.data_root, 'query'),
        'gallery': osp.join(args.data_root, 'bounding_box_test'),
    }

    for split_name, img_dir in splits.items():
        output_path = osp.join(args.output_dir, f'{split_name}_keypoints.json')
        if osp.exists(output_path):
            print(f"  {output_path} already exists, skipping.")
            continue
        extract_for_split(model, img_dir, output_path, args.batch_size, args.device)

    print("Done! All keypoints extracted.")


if __name__ == '__main__':
    main()
