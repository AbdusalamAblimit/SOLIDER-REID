#!/usr/bin/env python
"""Extract keypoint visibility for existing pose_data person crops.

This script reuses the per-person bbox stored in `pose_data/*.npz`, runs the
custom ViTPose checkpoint with `VisPredictHead`, and writes the following
fields back into each `.npz`:

    visibility:        (17,) float32   visibility probability in [0, 1]
    visibility_binary: (17,) uint8     thresholded visibility labels

It does not re-run person detection and does not modify `index.json`.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mmpose.apis import inference_topdown, init_model

from extract_pose import detect_dataset, get_image_list


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract visibility scores for existing pose_data files')
    parser.add_argument('--data-root', default='data/occluded_duke')
    parser.add_argument('--pose-data-root', default='data/occluded_duke/pose_data')
    parser.add_argument('--pose-config', default='pose/config_vispredict.py')
    parser.add_argument('--pose-checkpoint',
                        default='pretrained/best_coco_AP_epoch_210.pth')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--splits', nargs='+',
                        default=['train', 'query', 'gallery'])
    parser.add_argument('--binary-thr', type=float, default=0.5)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def resolve_config(config_path: str, checkpoint_path: str) -> str:
    """Use the provided config if present, otherwise fall back to checkpoint meta."""
    if config_path and os.path.exists(config_path):
        return config_path

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    cfg_text = ckpt.get('meta', {}).get('cfg')
    if not cfg_text:
        raise FileNotFoundError(
            f'Config file {config_path!r} not found and checkpoint meta has no cfg')

    tmp = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False)
    tmp.write(cfg_text)
    tmp.flush()
    tmp.close()
    return tmp.name


def build_image_map(data_root: str, split: str, dataset_type: str):
    return {filename: img_path for img_path, filename in get_image_list(
        data_root, split, dataset_type=dataset_type)}


def load_npz_dict(npz_path: Path):
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def save_npz_dict(npz_path: Path, arrays):
    np.savez_compressed(npz_path, **arrays)


def summarize_split(summary, split, split_dir: Path):
    out = {
        'split': split,
        'num_images': summary['num_images'],
        'num_person_files': summary['num_person_files'],
        'binary_threshold': summary['binary_threshold'],
        'mean_visibility': (summary['vis_sum'] / max(summary['num_person_files'], 1)).tolist(),
        'visible_rate': (summary['bin_sum'] / max(summary['num_person_files'], 1)).tolist(),
        'checkpoint': summary['checkpoint'],
    }
    out_path = split_dir / 'visibility_summary.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)


def main():
    args = parse_args()
    dataset_type = detect_dataset(args.data_root)
    config_path = resolve_config(args.pose_config, args.pose_checkpoint)

    print(f'Loading visibility pose model from {config_path}')
    pose_model = init_model(
        config_path,
        args.pose_checkpoint,
        device=args.device,
        cfg_options={'model.test_cfg.flip_test': False},
    )

    for split in args.splits:
        split_dir = Path(args.pose_data_root) / split
        index_path = split_dir / 'index.json'
        if not index_path.exists():
            print(f'Skip split {split}: missing {index_path}')
            continue

        with open(index_path, 'r') as f:
            index = json.load(f)

        image_map = build_image_map(args.data_root, split, dataset_type)
        summary = {
            'num_images': 0,
            'num_person_files': 0,
            'vis_sum': np.zeros(17, dtype=np.float64),
            'bin_sum': np.zeros(17, dtype=np.float64),
            'binary_threshold': args.binary_thr,
            'checkpoint': args.pose_checkpoint,
        }

        print(f'\n=== Processing split: {split} ({len(index)} images) ===')
        for filename, meta in tqdm(index.items(), desc=f'vis:{split}'):
            img_path = image_map.get(filename)
            if img_path is None:
                raise FileNotFoundError(
                    f'Image {filename!r} from {index_path} not found under {args.data_root}')

            person_files = meta.get('persons', [])
            if not person_files:
                continue

            bboxes = []
            npz_payloads = []
            process_flags = []
            for person_file in person_files:
                npz_path = split_dir / person_file
                arrays = load_npz_dict(npz_path)
                should_process = args.overwrite or 'visibility' not in arrays
                process_flags.append(should_process)
                npz_payloads.append((npz_path, arrays))
                bboxes.append(arrays['bbox'].astype(np.float32))

            if not any(process_flags):
                summary['num_images'] += 1
                summary['num_person_files'] += len(person_files)
                for _, arrays in npz_payloads:
                    vis = arrays['visibility'].astype(np.float64)
                    vis_bin = arrays['visibility_binary'].astype(np.float64)
                    summary['vis_sum'] += vis
                    summary['bin_sum'] += vis_bin
                continue

            results = inference_topdown(
                pose_model,
                img_path,
                np.stack(bboxes, axis=0),
                bbox_format='xyxy',
            )

            if len(results) != len(person_files):
                raise RuntimeError(
                    f'{filename}: expected {len(person_files)} persons, got {len(results)}')

            for (npz_path, arrays), result in zip(npz_payloads, results):
                vis = result.pred_instances.keypoints_visible
                vis = np.asarray(vis, dtype=np.float32)
                if vis.ndim == 2:
                    vis = vis[0]
                vis_bin = (vis >= args.binary_thr).astype(np.uint8)

                arrays['visibility'] = vis.astype(np.float32)
                arrays['visibility_binary'] = vis_bin
                save_npz_dict(npz_path, arrays)

                summary['vis_sum'] += vis.astype(np.float64)
                summary['bin_sum'] += vis_bin.astype(np.float64)

            summary['num_images'] += 1
            summary['num_person_files'] += len(person_files)

        summarize_split(summary, split, split_dir)
        mean_vis = summary['vis_sum'] / max(summary['num_person_files'], 1)
        visible_rate = summary['bin_sum'] / max(summary['num_person_files'], 1)
        print(f'{split}: persons={summary["num_person_files"]}, '
              f'mean vis={mean_vis.mean():.4f}, visible rate={visible_rate.mean():.4f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
