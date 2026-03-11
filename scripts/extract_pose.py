#!/usr/bin/env python
"""Extract per-person pose heatmaps and keypoints using RTMDet-s + ViTPose-Huge.

For each image:
  1. RTMDet-s detects all persons (with NMS)
  2. ViTPose-Huge estimates pose per person, raw output heatmap captured via hook
  3. Each person saved as a separate .npz file

Output layout:
    {output_dir}/{split}/{filename}_p{i}.npz   — per-person pose data
    {output_dir}/{split}/index.json            — filename -> person files mapping

Each .npz contains:
    heatmap:     (17, 64, 48) float16  — model's raw output heatmap (bbox-local)
    keypoints:   (17, 2)     float32   — keypoint coords in original image pixels
    scores:      (17,)       float32   — per-keypoint confidence
    bbox:        (4,)        float32   — [x1, y1, x2, y2] detection bbox in image pixels
    crop_bounds: (4,)        float32   — [x1, y1, x2, y2] actual ViTPose crop region
"""

import os
import json
import argparse
import subprocess
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model, inference_topdown
from mmpose.structures.bbox import bbox_xyxy2cs
from mmengine.registry import DefaultScope

PERSON_LABEL = 0  # COCO person class

# Download URLs for pretrained models
PRETRAINED_URLS = {
    'rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth':
        'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth',
    'td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth':
        'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth',
}


def ensure_pretrained(args):
    """Download pretrained model files if they don't exist."""
    os.makedirs(os.path.dirname(args.det_config) or 'pretrained', exist_ok=True)

    files_to_check = [
        (args.det_config, 'RTMDet-s config'),
        (args.det_checkpoint, 'RTMDet-s checkpoint'),
        (args.pose_config, 'ViTPose-Huge config'),
        (args.pose_checkpoint, 'ViTPose-Huge checkpoint'),
    ]

    for fpath, desc in files_to_check:
        if os.path.exists(fpath):
            continue

        fname = os.path.basename(fpath)

        # Try download URL for checkpoint files
        if fname in PRETRAINED_URLS:
            url = PRETRAINED_URLS[fname]
            print(f"Downloading {desc}: {fname} ...")
            subprocess.run(
                ['wget', '-q', '--show-progress', '-O', fpath, url],
                check=True)
            print(f"  Saved to {fpath}")

        # For config .py files, use mim to dump config
        elif fname.endswith('.py'):
            # Try to find from mim
            if 'rtmdet' in fname:
                pkg, model = 'mmdet', 'rtmdet_s_8xb32-300e_coco'
            elif 'ViTPose' in fname:
                pkg, model = 'mmpose', 'td-hm_ViTPose-huge_8xb64-210e_coco-256x192'
            else:
                raise FileNotFoundError(
                    f"Missing {desc}: {fpath}. Please provide it manually.")
            print(f"Downloading {desc} via mim: {model} ...")
            dest_dir = os.path.dirname(fpath) or '.'
            subprocess.run(
                ['mim', 'download', pkg, '--config', model, '--dest', dest_dir],
                check=True)
            # mim may download both config and checkpoint; rename if needed
            if not os.path.exists(fpath):
                # Try to find downloaded config
                for f in os.listdir(dest_dir):
                    if f.endswith('.py') and model.split('_')[0] in f:
                        os.rename(os.path.join(dest_dir, f), fpath)
                        break
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Failed to download {desc}. Please download manually to {fpath}")
            print(f"  Saved to {fpath}")
        else:
            raise FileNotFoundError(
                f"Missing {desc}: {fpath}. Please download manually.")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract per-person pose heatmaps with RTMDet + ViTPose')
    parser.add_argument('--data-root', default='data/occluded_duke')
    parser.add_argument('--output-dir', default='data/occluded_duke/pose_data')
    parser.add_argument('--det-config',
                        default='pretrained/rtmdet_s_8xb32-300e_coco.py')
    parser.add_argument('--det-checkpoint',
                        default='pretrained/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth')
    parser.add_argument('--pose-config',
                        default='pretrained/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py')
    parser.add_argument('--pose-checkpoint',
                        default='pretrained/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth')
    parser.add_argument('--det-score-thr', type=float, default=0.3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-persons', type=int, default=6)
    parser.add_argument('--splits', nargs='+',
                        default=['train', 'query', 'gallery'])
    parser.add_argument('--padding', type=float, default=1.25,
                        help='BBox padding factor used by mmpose (default 1.25)')
    return parser.parse_args()


# Dataset-specific split configurations
DATASET_CONFIGS = {
    'occluded_duke': {
        'split_dirs': {
            'train': 'bounding_box_train',
            'query': 'query',
            'gallery': 'bounding_box_test',
        },
        'split_lists': {
            'train': 'train.list',
            'query': 'query.list',
            'gallery': 'gallery.list',
        },
    },
    'market1501': {
        'split_dirs': {
            'train': 'bounding_box_train',
            'query': 'query',
            'gallery': 'bounding_box_test',
        },
        'split_lists': {},  # no list files, scan directory
    },
    'msmt17': {
        'split_dirs': {
            'train': 'train',
            'query': 'test',
            'gallery': 'test',
        },
        'split_lists': {
            'train': ['list_train.txt', 'list_val.txt'],  # multiple list files
            'query': ['list_query.txt'],
            'gallery': ['list_gallery.txt'],
        },
        'list_has_pid': True,  # list format: "subdir/filename.jpg pid"
    },
}


def detect_dataset(data_root):
    """Auto-detect dataset type from data_root path."""
    root_lower = data_root.lower()
    if 'msmt' in root_lower:
        return 'msmt17'
    elif 'market' in root_lower:
        return 'market1501'
    else:
        return 'occluded_duke'


def get_image_list(data_root, split, dataset_type=None):
    """Get image list from .list file or by scanning directory."""
    if dataset_type is None:
        dataset_type = detect_dataset(data_root)

    cfg = DATASET_CONFIGS.get(dataset_type, DATASET_CONFIGS['occluded_duke'])
    img_dir = os.path.join(data_root, cfg['split_dirs'][split])
    list_files = cfg.get('split_lists', {}).get(split, [])
    has_pid = cfg.get('list_has_pid', False)

    # Normalize to list
    if isinstance(list_files, str):
        list_files = [list_files]

    filenames = []
    for lf in list_files:
        list_path = os.path.join(data_root, lf)
        if os.path.exists(list_path):
            with open(list_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if has_pid:
                        # Format: "subdir/filename.jpg pid"
                        rel_path = line.split()[0]
                    else:
                        rel_path = line
                    filenames.append(rel_path)

    if not filenames:
        # Fallback: scan directory for image files
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        filenames = sorted(
            fn for fn in os.listdir(img_dir)
            if os.path.splitext(fn)[1].lower() in exts and not fn.startswith('.')
        )
        print(f"  No .list file found, scanned {img_dir}: {len(filenames)} images")

    # Return (full_path, basename) pairs
    # Use basename as key in index.json (unique across splits)
    return [(os.path.join(img_dir, fn), os.path.basename(fn)) for fn in filenames]


def detect_persons(det_model, img_path, score_thr=0.3, max_persons=6):
    """Detect persons with RTMDet. Returns bboxes sorted by area (largest first)."""
    with DefaultScope.overwrite_default_scope('mmdet'):
        result = inference_detector(det_model, img_path)
    pred = result.pred_instances
    mask = (pred.labels == PERSON_LABEL) & (pred.scores >= score_thr)
    bboxes = pred.bboxes[mask].cpu().numpy().astype(np.float32)
    scores = pred.scores[mask].cpu().numpy().astype(np.float32)

    if len(bboxes) == 0:
        # Fallback: use full image as person bbox
        img = Image.open(img_path)
        w, h = img.size
        bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
        scores = np.array([1.0], dtype=np.float32)

    # Sort by area descending — largest person = primary (index 0)
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    order = areas.argsort()[::-1]
    bboxes = bboxes[order][:max_persons]
    scores = scores[order][:max_persons]
    return bboxes, scores


def compute_crop_bounds(bbox, padding=1.25, input_size=(192, 256)):
    """Compute the actual image-space crop region used by ViTPose.

    ViTPose applies bbox_xyxy2cs (with padding) then fixes aspect ratio
    to match the model input. The heatmap (17, 64, 48) covers exactly
    this crop region.

    Args:
        bbox: [x1, y1, x2, y2]
        padding: bbox padding factor (default 1.25)
        input_size: (W, H) model input size

    Returns:
        crop_bounds: [x1, y1, x2, y2] in image pixel space
    """
    center, scale = bbox_xyxy2cs(
        np.array(bbox, dtype=np.float32), padding=padding)

    input_w, input_h = input_size
    aspect = input_w / input_h
    sw, sh = float(scale[0]), float(scale[1])
    if sw > sh * aspect:
        sh = sw / aspect
    else:
        sw = sh * aspect

    return np.array([center[0] - sw / 2, center[1] - sh / 2,
                     center[0] + sw / 2, center[1] + sh / 2],
                    dtype=np.float32)


def main():
    args = parse_args()

    # Auto-download pretrained models if missing
    ensure_pretrained(args)

    print("Loading RTMDet-s for person detection...")
    det_model = init_detector(args.det_config, args.det_checkpoint,
                              device=args.device)

    print("Loading ViTPose-Huge for pose estimation...")
    pose_model = init_model(
        args.pose_config, args.pose_checkpoint, device=args.device,
        cfg_options={'model.test_cfg.flip_test': False})

    # Hook on head.final_layer to capture raw heatmaps
    heatmap_store = []
    hook = pose_model.head.final_layer.register_forward_hook(
        lambda m, inp, out: heatmap_store.append(out.detach().cpu()))

    for split in args.splits:
        print(f"\n=== Processing split: {split} ===")
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        image_list = get_image_list(args.data_root, split)
        index = {}

        for img_path, filename in tqdm(image_list, desc=split):
            img = Image.open(img_path)
            img_w, img_h = img.size

            # 1. Detect persons
            bboxes, det_scores = detect_persons(
                det_model, img_path, args.det_score_thr, args.max_persons)

            # 2. Run ViTPose on each person
            heatmap_store.clear()
            results = inference_topdown(
                pose_model, img_path, bboxes, bbox_format='xyxy')

            if len(heatmap_store) == 0:
                print(f"  WARNING: no heatmap captured for {filename}, skipping")
                continue

            # heatmap_store[-1]: (N_persons, 17, 64, 48)
            captured = heatmap_store[-1]

            # 3. Save per-person npz
            stem = filename.rsplit('.', 1)[0]
            person_files = []

            for i, result in enumerate(results):
                kp = result.pred_instances.keypoints
                kp_scores = result.pred_instances.keypoint_scores
                # Handle shape: (1, 17, 2) or (17, 2)
                if kp.ndim == 3:
                    kp = kp[0]
                if kp_scores.ndim == 2:
                    kp_scores = kp_scores[0]

                hm = captured[i].numpy()  # (17, 64, 48)
                bbox = bboxes[i]
                crop_bounds = compute_crop_bounds(
                    bbox, padding=args.padding,
                    input_size=(192, 256))

                npz_name = f"{stem}_p{i}.npz"
                npz_path = os.path.join(split_dir, npz_name)
                np.savez_compressed(
                    npz_path,
                    heatmap=hm.astype(np.float16),
                    keypoints=kp.astype(np.float32),
                    scores=kp_scores.astype(np.float32),
                    bbox=bbox.astype(np.float32),
                    crop_bounds=crop_bounds,
                )
                person_files.append(npz_name)

            index[filename] = {
                'num_persons': len(person_files),
                'image_size': [img_h, img_w],
                'persons': person_files,
            }

        # Save index
        index_path = os.path.join(split_dir, 'index.json')
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        print(f"Saved index: {index_path} ({len(index)} images)")

    hook.remove()
    print("\nDone!")


if __name__ == '__main__':
    main()
