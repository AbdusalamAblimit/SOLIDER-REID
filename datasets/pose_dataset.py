"""Pose-aware image dataset that loads per-person keypoints and heatmaps.

Handles geometric augmentations (flip, crop) jointly for images and pose data.
"""
import os
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from .bases import read_image

# COCO left-right keypoint swap pairs
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


class PoseImageDataset(Dataset):
    """ImageDataset with multi-person pose data loading.

    Handles geometric augmentations (flip, pad+crop) jointly for images and heatmaps.
    Returns (img, pid, camid, trackid, img_path, pose_data_dict).
    """

    MAX_PERSONS = 6

    def __init__(self, dataset, transform=None, pose_data=None,
                 heatmap_dir=None, target_size=None,
                 is_train=False, flip_prob=0.0, pad=0, crop_size=None):
        """
        Args:
            dataset: list of (img_path, pid, camid, trackid)
            transform: pixel-level transforms (ToTensor, Normalize, RandomErasing)
            pose_data: list of per-image dicts (PKL) or legacy dict (NPZ)
            heatmap_dir: directory or list of dirs containing .npy heatmap files
            target_size: (H, W) to resize heatmaps
            is_train: whether this is training (enables augmentation)
            flip_prob: probability of horizontal flip
            pad: padding pixels before random crop
            crop_size: (H, W) for random crop after padding
        """
        self.dataset = dataset
        self.transform = transform
        self.target_size = target_size
        # Determine final heatmap size
        if target_size is not None:
            self.hm_h, self.hm_w = target_size
        elif crop_size is not None:
            self.hm_h, self.hm_w = crop_size
        else:
            self.hm_h, self.hm_w = 64, 48
        self.is_train = is_train
        self.flip_prob = flip_prob
        self.pad = pad
        self.crop_size = crop_size

        # Normalize heatmap dirs to list
        if heatmap_dir is None:
            self.heatmap_dirs = []
        elif isinstance(heatmap_dir, (list, tuple)):
            self.heatmap_dirs = list(heatmap_dir)
        else:
            self.heatmap_dirs = [heatmap_dir]

        # Build filename -> pose data lookup
        self.pose_lookup = {}
        if pose_data is not None:
            if isinstance(pose_data, dict) and 'filenames' in pose_data:
                # Legacy NPZ format — bbox omitted to use full-image default
                for idx, fname in enumerate(pose_data['filenames']):
                    self.pose_lookup[fname] = [{
                        'keypoints': pose_data['keypoints'][idx],
                        'kp_scores': pose_data['scores'][idx],
                        'heatmap_file': fname.replace('.jpg', '_p0.npy'),
                    }]
            elif isinstance(pose_data, list):
                for entry in pose_data:
                    self.pose_lookup[entry['filename']] = entry['persons']
            elif isinstance(pose_data, dict):
                self.pose_lookup = pose_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)  # PIL Image
        orig_w, orig_h = img.size  # original image dimensions

        fname = os.path.basename(img_path)
        persons = self.pose_lookup.get(fname, [])

        # Load all heatmaps and map to full-image coordinate space
        person_heatmaps = []
        for p in persons[:self.MAX_PERSONS]:
            hm = self._load_heatmap_raw(p.get('heatmap_file'))
            bbox = p.get('bbox', np.array([0, 0, orig_w, orig_h], dtype=np.float32))
            # Place bbox-local heatmap into full-image coordinates
            hm_fullimg = self._place_heatmap_in_fullimg(hm, bbox, orig_h, orig_w)
            person_heatmaps.append(hm_fullimg)

        # --- Geometric augmentations (applied jointly to image + heatmaps) ---
        # 1. Resize to target size (always)
        if self.crop_size:
            img = img.resize((self.crop_size[1], self.crop_size[0]), Image.BICUBIC)
        # Heatmaps resize handled later

        # 2. Random horizontal flip
        flipped = False
        if self.is_train and random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped = True

        # 3. Padding + random crop
        crop_offset_x, crop_offset_y = 0, 0
        if self.is_train and self.pad > 0 and self.crop_size:
            # Pad image
            w, h = img.size
            padded = Image.new(img.mode, (w + 2 * self.pad, h + 2 * self.pad), (0, 0, 0))
            padded.paste(img, (self.pad, self.pad))
            # Random crop
            pw, ph = padded.size
            crop_offset_x = random.randint(0, pw - self.crop_size[1])
            crop_offset_y = random.randint(0, ph - self.crop_size[0])
            img = padded.crop((crop_offset_x, crop_offset_y,
                               crop_offset_x + self.crop_size[1],
                               crop_offset_y + self.crop_size[0]))

        # --- Pixel-level transforms (ToTensor, Normalize, RandomErasing) ---
        if self.transform is not None:
            img = self.transform(img)

        # --- Apply same geometric transforms to pose data ---
        n_persons = min(len(persons), self.MAX_PERSONS)
        all_kp = torch.zeros(self.MAX_PERSONS, 17, 2)
        all_scores = torch.zeros(self.MAX_PERSONS, 17)
        all_hm = torch.zeros(self.MAX_PERSONS, 17, self.hm_h, self.hm_w)
        all_bboxes = torch.zeros(self.MAX_PERSONS, 4)
        person_mask = torch.zeros(self.MAX_PERSONS)

        for i in range(n_persons):
            p = persons[i]
            kp = np.array(p['keypoints'], dtype=np.float32).copy()
            scores = np.array(p['kp_scores'], dtype=np.float32).copy()
            hm = person_heatmaps[i]  # (17, 64, 48) raw

            # Resize heatmap to crop_size first (same as image)
            if self.crop_size is not None:
                hm = F.interpolate(
                    hm.unsqueeze(0), size=(self.crop_size[0], self.crop_size[1]),
                    mode='bilinear', align_corners=False
                ).squeeze(0)  # (17, crop_H, crop_W)

            # Apply flip to keypoints and heatmap
            if flipped:
                kp[:, 0] = 1.0 - kp[:, 0]
                for l, r in FLIP_PAIRS:
                    kp[[l, r]] = kp[[r, l]]
                    scores[[l, r]] = scores[[r, l]]
                hm = hm.flip(-1)
                for l, r in FLIP_PAIRS:
                    hm[[l, r]] = hm[[r, l]]

            # Final resize to target size for model (e.g. 12x4)
            if self.target_size is not None:
                hm = F.interpolate(
                    hm.unsqueeze(0), size=self.target_size,
                    mode='bilinear', align_corners=False
                ).squeeze(0)

            all_kp[i] = torch.from_numpy(kp)
            all_scores[i] = torch.from_numpy(scores)
            all_hm[i] = hm
            bbox = p.get('bbox', np.array([0, 0, 1, 1], dtype=np.float32))
            all_bboxes[i] = torch.from_numpy(np.array(bbox, dtype=np.float32))
            person_mask[i] = 1.0

        # Primary person = index 0
        pose_dict = {
            'primary_keypoints': all_kp[0],
            'primary_scores': all_scores[0],
            'primary_heatmap': all_hm[0],
            'num_persons': n_persons,
            'all_keypoints': all_kp,
            'all_scores': all_scores,
            'all_heatmaps': all_hm,
            'all_bboxes': all_bboxes,
            'person_mask': person_mask,
        }

        return img, pid, camid, trackid, img_path, pose_dict

    def _load_heatmap_raw(self, hm_filename):
        """Load raw heatmap .npy file without resizing."""
        if hm_filename is None:
            return torch.zeros(17, 64, 48)

        for hdir in self.heatmap_dirs:
            npy_path = os.path.join(hdir, hm_filename)
            if os.path.exists(npy_path):
                heatmap = np.load(npy_path)
                return torch.from_numpy(heatmap).float()

        return torch.zeros(17, 64, 48)

    @staticmethod
    def _place_heatmap_in_fullimg(heatmap, bbox, img_h, img_w):
        """Place bbox-local heatmap back into full-image coordinate space.

        Args:
            heatmap: (17, hm_h, hm_w) in bbox-local coordinates
            bbox: (4,) [x1, y1, x2, y2] in pixel coordinates
            img_h, img_w: full image dimensions

        Returns:
            (17, img_h, img_w) heatmap in full-image coordinates
        """
        x1, y1, x2, y2 = bbox
        bw = max(int(x2 - x1), 1)
        bh = max(int(y2 - y1), 1)
        ix1 = max(0, int(x1))
        iy1 = max(0, int(y1))
        ix2 = min(img_w, int(x2))
        iy2 = min(img_h, int(y2))
        actual_w = ix2 - ix1
        actual_h = iy2 - iy1

        if actual_w <= 0 or actual_h <= 0:
            return torch.zeros(17, img_h, img_w)

        # Resize heatmap to bbox size
        hm_resized = F.interpolate(
            heatmap.unsqueeze(0), size=(bh, bw),
            mode='bilinear', align_corners=False
        ).squeeze(0)  # (17, bh, bw)

        # Paste into full-image canvas
        canvas = torch.zeros(17, img_h, img_w)
        # Handle edge clipping
        src_y1 = iy1 - int(y1) if y1 < 0 else 0
        src_x1 = ix1 - int(x1) if x1 < 0 else 0
        canvas[:, iy1:iy2, ix1:ix2] = hm_resized[:, src_y1:src_y1+actual_h,
                                                       src_x1:src_x1+actual_w]
        return canvas


def load_pose_data(pose_dir, split):
    """Load pose data from PKL or fallback to NPZ."""
    pkl_path = os.path.join(pose_dir, f'pose_data_{split}.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded pose data (PKL): {pkl_path} ({len(data)} images)")
        return data

    npz_path = os.path.join(pose_dir, f'pose_data_{split}.npz')
    if not os.path.exists(npz_path):
        npz_path = os.path.join(pose_dir, f'pose_rtmpose_{split}.npz')
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        print(f"Loaded pose data (NPZ): {npz_path}")
        return {
            'filenames': data['filenames'],
            'keypoints': data['keypoints'],
            'scores': data['scores'],
        }

    print(f"Warning: No pose data found for split '{split}' in {pose_dir}")
    return None
