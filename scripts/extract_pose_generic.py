"""Extract offline pose (keypoints + visibility) for any ReID dataset.

Supports: market1501, msmt17, occluded_duke
Saves per-split results as .npz files in the dataset directory.

Usage:
  conda run -n solider-reid python scripts/extract_pose_generic.py --dataset market1501
  conda run -n solider-reid python scripts/extract_pose_generic.py --dataset msmt17
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.backbones.pose_predictor import MMPoseTopDownPredictor


DATASET_SPLITS = {
    'market1501': {
        'train': 'bounding_box_train',
        'gallery': 'bounding_box_test',
        'query': 'query',
    },
    'msmt17': {
        'train': 'bounding_box_train',
        'gallery': 'bounding_box_test',
        'query': 'query',
    },
    'occluded_duke': {
        'train': 'bounding_box_train',
        'gallery': 'bounding_box_test',
        'query': 'query',
    },
}


def extract_keypoints_from_heatmaps(heatmaps):
    """Extract (x, y) keypoint coordinates from heatmaps."""
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    idx = flat.argmax(dim=-1)
    y = (idx // W).float()
    x = (idx % W).float()
    x_scale = 128.0 / W
    y_scale = 384.0 / H
    x = x * x_scale
    y = y * y_scale
    keypoints = torch.stack([x, y], dim=-1)
    return keypoints.cpu().numpy().astype(np.int16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASET_SPLITS.keys()))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_root', default='data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading ViTPose model...")
    predictor = MMPoseTopDownPredictor(
        cfg_path='pose/config_vispredict.py',
        ckpt_path='pretrained/best_coco_AP_epoch_210.pth',
        device=device
    )
    predictor.eval()

    transform = transforms.Compose([
        transforms.Resize((384, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    data_dir = os.path.join(args.data_root, args.dataset)
    splits = DATASET_SPLITS[args.dataset]

    for split_name, img_subdir in splits.items():
        print(f"\n--- Processing {split_name} ---")
        full_dir = os.path.join(data_dir, img_subdir)

        # Get all jpg images, sorted for reproducibility
        img_files = sorted([f for f in os.listdir(full_dir)
                           if f.endswith('.jpg') and not f.startswith('-1')])  # skip junk only

        print(f"Found {len(img_files)} images in {full_dir}")

        all_heatmaps = []
        all_visibility = []
        all_keypoints = []
        valid_filenames = []

        batch_imgs = []
        batch_names = []

        for i, fname in enumerate(tqdm(img_files, desc=split_name)):
            img_path = os.path.join(full_dir, fname)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                batch_imgs.append(img_tensor)
                batch_names.append(fname)
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                continue

            if len(batch_imgs) == args.batch_size or i == len(img_files) - 1:
                if not batch_imgs:
                    continue
                batch = torch.stack(batch_imgs).to(device)

                with torch.no_grad(), torch.cuda.amp.autocast():
                    heatmaps, visibility = predictor(batch)

                kpts = extract_keypoints_from_heatmaps(heatmaps)
                all_visibility.append(visibility.cpu().float().numpy())
                all_keypoints.append(kpts)
                valid_filenames.extend(batch_names)

                batch_imgs = []
                batch_names = []

        all_visibility = np.concatenate(all_visibility, axis=0)
        all_keypoints = np.concatenate(all_keypoints, axis=0)

        out_path = os.path.join(data_dir, f'pose_{split_name}.npz')
        np.savez_compressed(
            out_path,
            filenames=np.array(valid_filenames),
            visibility=all_visibility,
            keypoints=all_keypoints,
        )

        print(f"Saved {split_name}: {len(valid_filenames)} images -> {out_path}")
        print(f"  visibility: {all_visibility.shape}, keypoints: {all_keypoints.shape}")
        vis_mean = all_visibility.mean(axis=0)
        print(f"  Mean visibility per keypoint: {vis_mean.round(3)}")


if __name__ == '__main__':
    main()
