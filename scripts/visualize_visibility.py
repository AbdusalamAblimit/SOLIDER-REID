#!/usr/bin/env python
"""Visualize multi-person keypoints with per-keypoint visibility text.

For each selected image:
  1. Draw all persons' keypoints and skeletons on the original image.
  2. Add text panels below the image, one panel per person.
  3. Each panel lists all 17 COCO keypoints with visibility score and
     thresholded binary label from the pose_data `.npz`.

This script expects visibility fields to already exist in `pose_data/*.npz`.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from extract_pose import detect_dataset, get_image_list


SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

KEYPOINT_NAMES = [
    'nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear',
    'l_sho', 'r_sho', 'l_elb', 'r_elb', 'l_wri', 'r_wri',
    'l_hip', 'r_hip', 'l_kne', 'r_kne', 'l_ank', 'r_ank',
]

PERSON_COLORS = [
    '#e63946',
    '#1d3557',
    '#2a9d8f',
    '#f4a261',
    '#6a4c93',
    '#ff006e',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize multi-person keypoints with visibility labels')
    parser.add_argument('--data-root', default='data/occluded_duke')
    parser.add_argument('--pose-dir', default='data/occluded_duke/pose_data')
    parser.add_argument('--output-dir', default='viz_visibility_multi')
    parser.add_argument('--splits', nargs='+',
                        default=['train', 'query', 'gallery'])
    parser.add_argument('--num-images', type=int, default=100)
    parser.add_argument('--min-persons', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--score-thr', type=float, default=0.1)
    return parser.parse_args()


def load_index(split_dir: Path):
    index_path = split_dir / 'index.json'
    with open(index_path, 'r') as f:
        return json.load(f)


def build_image_map(data_root: str, split: str, dataset_type: str):
    return {filename: img_path for img_path, filename in get_image_list(
        data_root, split, dataset_type=dataset_type)}


def load_person_payloads(split_dir: Path, person_files):
    persons = []
    for person_file in person_files:
        npz_path = split_dir / person_file
        with np.load(npz_path) as data:
            visibility = data['visibility'].astype(np.float32)
            if 'visibility_binary' in data.files:
                visibility_binary = data['visibility_binary'].astype(np.int32)
            else:
                visibility_binary = (visibility >= 0.5).astype(np.int32)
            persons.append({
                'keypoints': data['keypoints'].astype(np.float32),
                'scores': data['scores'].astype(np.float32),
                'visibility': visibility,
                'visibility_binary': visibility_binary,
            })
    return persons


def draw_person(ax, person, color, score_thr):
    keypoints = person['keypoints']
    scores = person['scores']

    for j1, j2 in SKELETON:
        if scores[j1] > score_thr and scores[j2] > score_thr:
            ax.plot(
                [keypoints[j1, 0], keypoints[j2, 0]],
                [keypoints[j1, 1], keypoints[j2, 1]],
                color=color,
                linewidth=1.8,
                alpha=0.85,
            )

    for j in range(17):
        if scores[j] > score_thr:
            ax.scatter(
                keypoints[j, 0], keypoints[j, 1],
                s=24, c=color, edgecolors='white', linewidths=0.6, zorder=3)
        else:
            ax.scatter(
                keypoints[j, 0], keypoints[j, 1],
                s=12, facecolors='none', edgecolors='#777777',
                linewidths=0.8, alpha=0.45, zorder=2)

    valid = keypoints[scores > score_thr]
    if len(valid) > 0:
        cx = float(valid[:, 0].mean())
        cy = max(float(valid[:, 1].min()) - 8.0, 10.0)
        return cx, cy
    return float(keypoints[:, 0].mean()), float(np.clip(keypoints[:, 1].min(), 10.0, None))


def add_person_panel(ax, person_idx, person, color):
    ax.axis('off')
    ax.set_facecolor('#f7f7f7')
    ax.set_title(f'P{person_idx}', loc='left', fontsize=11,
                 fontweight='bold', color=color, pad=6)

    lines = []
    for name, vis, vis_bin in zip(
            KEYPOINT_NAMES,
            person['visibility'],
            person['visibility_binary']):
        lines.append(f'{name:>6}: {vis:0.3f} [{int(vis_bin)}]')

    ax.text(
        0.02, 0.98,
        '\n'.join(lines),
        va='top',
        ha='left',
        fontsize=9,
        family='monospace',
        color='#111111',
        transform=ax.transAxes,
    )


def render_one(image_path, persons, out_path, title, score_thr):
    image = np.array(Image.open(image_path).convert('RGB'))
    num_persons = len(persons)
    panel_cols = min(3, max(1, num_persons))
    panel_rows = int(math.ceil(num_persons / panel_cols))

    fig_height = 8.0 + panel_rows * 4.2
    fig = plt.figure(figsize=(13, fig_height))
    grid = fig.add_gridspec(
        1 + panel_rows,
        panel_cols,
        height_ratios=[8.0] + [4.2] * panel_rows)

    ax_img = fig.add_subplot(grid[0, :])
    ax_img.imshow(image)
    ax_img.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax_img.axis('off')

    for idx, person in enumerate(persons):
        color = PERSON_COLORS[idx % len(PERSON_COLORS)]
        lx, ly = draw_person(ax_img, person, color, score_thr)
        ax_img.text(
            lx, ly, f'P{idx}',
            color='white',
            fontsize=8,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.22', facecolor=color, edgecolor='none', alpha=0.95),
        )

    panel_axes = []
    for row in range(panel_rows):
        for col in range(panel_cols):
            panel_axes.append(fig.add_subplot(grid[row + 1, col]))

    for idx, person in enumerate(persons):
        add_person_panel(
            panel_axes[idx], idx, person, PERSON_COLORS[idx % len(PERSON_COLORS)])

    for ax in panel_axes[num_persons:]:
        ax.axis('off')

    fig.tight_layout(pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def collect_candidates(args):
    dataset_type = detect_dataset(args.data_root)
    candidates = []
    for split in args.splits:
        split_dir = Path(args.pose_dir) / split
        if not split_dir.exists():
            raise FileNotFoundError(f'Missing pose split dir: {split_dir}')

        index = load_index(split_dir)
        image_map = build_image_map(args.data_root, split, dataset_type)
        for filename, meta in index.items():
            num_persons = int(meta.get('num_persons', len(meta.get('persons', []))))
            if num_persons < args.min_persons:
                continue
            img_path = image_map.get(filename)
            if img_path is None:
                continue
            candidates.append({
                'split': split,
                'filename': filename,
                'img_path': img_path,
                'split_dir': split_dir,
                'num_persons': num_persons,
                'person_files': meta['persons'],
            })
    return candidates


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidates(args)
    print(f'Found {len(candidates)} images with >= {args.min_persons} persons')
    if not candidates:
        return

    if len(candidates) <= args.num_images:
        selected = list(candidates)
    else:
        selected = random.sample(candidates, args.num_images)
    selected.sort(key=lambda item: (item['split'], item['filename']))

    manifest = []
    for idx, item in enumerate(selected, start=1):
        print(f'[{idx:03d}/{len(selected):03d}] {item["split"]}/{item["filename"]}')
        persons = load_person_payloads(item['split_dir'], item['person_files'])
        title = f'{item["split"]}/{item["filename"]}  persons={len(persons)}'
        stem = Path(item['filename']).stem
        out_name = f'{item["split"]}_{stem}.png'
        out_path = output_dir / out_name
        render_one(item['img_path'], persons, out_path, title, args.score_thr)
        manifest.append({
            'split': item['split'],
            'filename': item['filename'],
            'num_persons': len(persons),
            'output': out_name,
        })

    with open(output_dir / 'manifest.json', 'w') as f:
        json.dump({
            'num_images': len(manifest),
            'min_persons': args.min_persons,
            'splits': args.splits,
            'seed': args.seed,
            'items': manifest,
        }, f, indent=2)

    print(f'Saved {len(manifest)} visualizations to {output_dir}')


if __name__ == '__main__':
    main()
