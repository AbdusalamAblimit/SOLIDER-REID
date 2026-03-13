#!/usr/bin/env python
"""Compute target person assignment for multi-person ReID images.

For each image with multiple detected persons, scores each person's
likelihood of being the ReID target and saves the assignment.

Targetness is computed as a weighted combination of:
  1. center_dist: how close the bbox center is to the image center
  2. area_ratio: bbox area relative to image area (larger = more likely target)
  3. containment: fraction of bbox within image bounds
  4. mean_score: average keypoint confidence

Output: updates index.json with target_person_idx, target_score, target_margin
        and per-person targetness scores.
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm


def compute_targetness(bbox, img_h, img_w, scores):
    """Compute targetness score for a single person detection.

    Args:
        bbox: [x1, y1, x2, y2] detection bbox in image pixels
        img_h, img_w: image dimensions
        scores: (17,) keypoint confidence scores

    Returns:
        targetness: float, overall targetness score
        components: dict with individual factor values
    """
    center_x, center_y = img_w / 2.0, img_h / 2.0

    # 1. Center distance (normalized)
    bx = (bbox[0] + bbox[2]) / 2.0
    by = (bbox[1] + bbox[3]) / 2.0
    # Normalized by half-diagonal so dist=1 means corner
    dist = np.sqrt(((bx - center_x) / img_w) ** 2 +
                   ((by - center_y) / img_h) ** 2)
    center_score = float(max(0.0, 1.0 - 2.0 * dist))

    # 2. Area ratio (bbox area / image area)
    area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    img_area = float(img_w * img_h)
    area_ratio = min(area / max(img_area, 1.0), 1.5)
    area_score = float(min(area_ratio / 0.8, 1.0))

    # 3. Containment (fraction of bbox inside image)
    clipped_x1 = max(0.0, bbox[0])
    clipped_y1 = max(0.0, bbox[1])
    clipped_x2 = min(float(img_w), bbox[2])
    clipped_y2 = min(float(img_h), bbox[3])
    clipped_area = max(0.0, clipped_x2 - clipped_x1) * \
                   max(0.0, clipped_y2 - clipped_y1)
    containment = float(clipped_area / max(area, 1e-6))

    # 4. Mean keypoint score
    mean_score = float(scores.mean())

    # Weighted combination
    targetness = (0.35 * center_score +
                  0.30 * area_score +
                  0.20 * containment +
                  0.15 * mean_score)

    components = {
        'center_score': round(center_score, 4),
        'area_score': round(area_score, 4),
        'containment': round(containment, 4),
        'mean_score': round(mean_score, 4),
    }

    return float(targetness), components


def process_split(pose_dir, split):
    """Process one split and update its index.json with target assignments."""
    split_dir = os.path.join(pose_dir, split)
    index_path = os.path.join(split_dir, 'index.json')

    if not os.path.exists(index_path):
        print(f"  WARNING: {index_path} not found, skipping")
        return

    with open(index_path, 'r') as f:
        index = json.load(f)

    stats = {
        'total': len(index),
        'single': 0,
        'multi': 0,
        'p0_is_target': 0,
        'p0_not_target': 0,
    }

    for fname, entry in tqdm(index.items(), desc=f'{split}'):
        n_persons = entry['num_persons']
        img_h, img_w = entry['image_size']

        if n_persons <= 1:
            # Single person: trivially person 0
            entry['target_person_idx'] = 0
            entry['target_score'] = 1.0
            entry['target_margin'] = 1.0
            entry['person_targetness'] = [1.0]
            stats['single'] += 1
            continue

        stats['multi'] += 1

        # Compute targetness for each person
        targetness_list = []
        for i, npz_name in enumerate(entry['persons']):
            if os.path.isabs(npz_name):
                npz_path = npz_name
            else:
                npz_path = os.path.join(split_dir, npz_name)

            if not os.path.exists(npz_path):
                targetness_list.append(0.0)
                continue

            with np.load(npz_path) as data:
                bbox = data['bbox'].astype(np.float64)
                scores = data['scores'].astype(np.float64)

            t, _ = compute_targetness(bbox, img_h, img_w, scores)
            targetness_list.append(round(t, 4))

        # Find best target
        best_idx = int(np.argmax(targetness_list))
        best_score = targetness_list[best_idx]

        # Compute margin (difference to second best)
        sorted_scores = sorted(targetness_list, reverse=True)
        if len(sorted_scores) >= 2:
            margin = sorted_scores[0] - sorted_scores[1]
        else:
            margin = 1.0

        entry['target_person_idx'] = best_idx
        entry['target_score'] = round(best_score, 4)
        entry['target_margin'] = round(margin, 4)
        entry['person_targetness'] = targetness_list

        if best_idx == 0:
            stats['p0_is_target'] += 1
        else:
            stats['p0_not_target'] += 1

    # Save updated index
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\n  {split} stats:")
    print(f"    Total: {stats['total']}")
    print(f"    Single-person: {stats['single']}")
    print(f"    Multi-person: {stats['multi']}")
    if stats['multi'] > 0:
        print(f"    Person 0 is target: {stats['p0_is_target']} "
              f"({100*stats['p0_is_target']/stats['multi']:.1f}%)")
        print(f"    Person 0 NOT target: {stats['p0_not_target']} "
              f"({100*stats['p0_not_target']/stats['multi']:.1f}%)")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose-dir', default='data/occluded_duke/pose_data')
    parser.add_argument('--splits', nargs='+',
                        default=['train', 'query', 'gallery'])
    args = parser.parse_args()

    print("Computing target person assignments...")
    for split in args.splits:
        print(f"\n=== Processing {split} ===")
        process_split(args.pose_dir, split)

    print("\nDone!")


if __name__ == '__main__':
    main()
