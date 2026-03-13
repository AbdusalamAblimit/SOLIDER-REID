#!/usr/bin/env python
"""Visualize target person assignment for multi-person ReID images.

Draws all detected persons' bounding boxes and keypoints on each image,
highlighting the selected target person. Saves visualizations to a directory.
"""

import os
import json
import argparse
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# COCO skeleton for drawing
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6),                            # shoulders
    (5, 7), (7, 9),                    # left arm
    (6, 8), (8, 10),                   # right arm
    (5, 11), (6, 12),                  # torso
    (11, 12),                          # hips
    (11, 13), (13, 15),               # left leg
    (12, 14), (14, 16),               # right leg
]

# Person colors: target = green, others = red/yellow/blue/...
PERSON_COLORS = [
    (255, 0, 0),     # red
    (255, 165, 0),   # orange
    (0, 0, 255),     # blue
    (255, 0, 255),   # magenta
    (128, 128, 0),   # olive
    (0, 128, 128),   # teal
]
TARGET_COLOR = (0, 255, 0)  # green for target


DATASET_CONFIGS = {
    'occluded_duke': {
        'split_dirs': {
            'train': 'bounding_box_train',
            'query': 'query',
            'gallery': 'bounding_box_test',
        },
    },
}


def draw_person(draw, bbox, keypoints, scores, color, label, is_target=False,
                line_width=2):
    """Draw one person's bbox and skeleton on the image."""
    x1, y1, x2, y2 = bbox
    lw = line_width + 1 if is_target else line_width

    # Draw bbox
    for offset in range(lw):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline=color)

    # Label
    draw.text((x1, max(0, y1 - 12)), label, fill=color)

    # Draw skeleton
    for (i, j) in COCO_SKELETON:
        if scores[i] > 0.3 and scores[j] > 0.3:
            x_i, y_i = keypoints[i]
            x_j, y_j = keypoints[j]
            draw.line([(x_i, y_i), (x_j, y_j)], fill=color, width=lw)

    # Draw keypoints
    r = 3 if is_target else 2
    for k in range(17):
        if scores[k] > 0.3:
            x, y = keypoints[k]
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def visualize_image(img_path, entry, pose_dir, output_path):
    """Create visualization for one image."""
    img = Image.open(img_path).convert('RGB')
    # Scale up for visibility
    scale = max(1, 400 // max(img.size))
    if scale > 1:
        img = img.resize((img.size[0] * scale, img.size[1] * scale),
                         Image.NEAREST)

    draw = ImageDraw.Draw(img)
    img_h, img_w = entry['image_size']
    target_idx = entry.get('target_person_idx', 0)
    targetness_list = entry.get('person_targetness', [])

    for i, npz_name in enumerate(entry['persons']):
        if os.path.isabs(npz_name):
            npz_path = npz_name
        else:
            npz_path = os.path.join(pose_dir, npz_name)
        if not os.path.exists(npz_path):
            continue

        with np.load(npz_path) as data:
            bbox = data['bbox'].astype(np.float64)
            kp = data['keypoints'].astype(np.float64)
            scores = data['scores'].astype(np.float64)

        # Scale coordinates
        bbox_scaled = bbox * scale
        kp_scaled = kp * scale

        is_target = (i == target_idx)
        color = TARGET_COLOR if is_target else PERSON_COLORS[
            i % len(PERSON_COLORS)]

        t_str = f"{targetness_list[i]:.3f}" if i < len(targetness_list) else "?"
        label = f"p{i} t={t_str}"
        if is_target:
            label = f"★ TARGET p{i} t={t_str}"

        draw_person(draw, bbox_scaled, kp_scaled, scores, color, label,
                    is_target=is_target, line_width=1 if scale <= 2 else 2)

    # Add summary text at bottom
    margin = entry.get('target_margin', 0)
    summary = (f"n={entry['num_persons']}  "
               f"target=p{target_idx}  "
               f"margin={margin:.3f}")
    draw.text((2, img.size[1] - 12), summary, fill=(255, 255, 255))

    img.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/occluded_duke')
    parser.add_argument('--pose-dir', default='data/occluded_duke/pose_data')
    parser.add_argument('--output-dir',
                        default='experiments/exp033/visualizations')
    parser.add_argument('--split', default='train')
    parser.add_argument('--num-samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--only-disagreements', action='store_true',
                        help='Only visualize where target != person 0')
    args = parser.parse_args()

    random.seed(args.seed)

    split_dir = os.path.join(args.pose_dir, args.split)
    index_path = os.path.join(split_dir, 'index.json')

    with open(index_path) as f:
        index = json.load(f)

    # Get dataset image directory
    cfg = DATASET_CONFIGS.get('occluded_duke', {})
    img_dir_name = cfg['split_dirs'].get(args.split, args.split)
    img_base = os.path.join(args.data_root, img_dir_name)

    # Filter multi-person images
    multi = {k: v for k, v in index.items() if v['num_persons'] > 1}
    print(f"Multi-person images in {args.split}: {len(multi)}")

    if args.only_disagreements:
        # Only show cases where target != person 0
        candidates = {k: v for k, v in multi.items()
                      if v.get('target_person_idx', 0) != 0}
        print(f"Disagreements (target != p0): {len(candidates)}")
    else:
        candidates = multi

    # Sample
    keys = list(candidates.keys())
    if len(keys) > args.num_samples:
        keys = random.sample(keys, args.num_samples)
    keys.sort()

    os.makedirs(args.output_dir, exist_ok=True)

    # Also save all disagreements regardless of sample
    disagree_dir = os.path.join(args.output_dir, 'disagreements')
    os.makedirs(disagree_dir, exist_ok=True)

    print(f"Generating {len(keys)} visualizations...")
    for fname in keys:
        entry = candidates[fname]
        img_path = os.path.join(img_base, fname)
        if not os.path.exists(img_path):
            continue

        stem = fname.rsplit('.', 1)[0]
        out_path = os.path.join(args.output_dir, f"{stem}_target.png")
        visualize_image(img_path, entry, split_dir, out_path)

    # Additionally save ALL disagreements
    all_disagree = {k: v for k, v in multi.items()
                    if v.get('target_person_idx', 0) != 0}
    print(f"\nGenerating {len(all_disagree)} disagreement visualizations...")
    for fname, entry in all_disagree.items():
        img_path = os.path.join(img_base, fname)
        if not os.path.exists(img_path):
            continue
        stem = fname.rsplit('.', 1)[0]
        out_path = os.path.join(disagree_dir, f"{stem}_disagree.png")
        visualize_image(img_path, entry, split_dir, out_path)

    print(f"\nSaved to {args.output_dir}")
    print(f"Disagreements saved to {disagree_dir}")


if __name__ == '__main__':
    main()
