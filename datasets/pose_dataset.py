"""Pose-aware image dataset with joint geometric augmentation.

Loads per-person pose data (.npz) and applies geometric augmentations
(resize, flip, pad+crop, random erasing) jointly to images, heatmaps,
and keypoints so they stay aligned.

Coordinate convention:
  - Keypoints are always in PIXEL coordinates of the current image state.
  - Heatmaps are kept at image resolution during augmentation, then
    optionally downsampled at the end for memory efficiency.
"""

import os
import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from .bases import read_image

# COCO 17-keypoint left-right swap pairs (0-indexed)
FLIP_PAIRS = [
    (1, 2), (3, 4),       # eyes, ears
    (5, 6), (7, 8),       # shoulders, elbows
    (9, 10), (11, 12),    # wrists, hips
    (13, 14), (15, 16),   # knees, ankles
]

# Body part groups for Pose-Guided Erasing (COCO 17 keypoints)
PGE_BODY_PARTS = [
    [0, 1, 2, 3, 4],       # head (nose, eyes, ears)
    [5, 7, 9],              # left arm (left shoulder, elbow, wrist)
    [6, 8, 10],             # right arm (right shoulder, elbow, wrist)
    [5, 6, 11, 12],         # torso (shoulders + hips)
    [13, 14, 15, 16],       # legs (knees, ankles)
]

MAX_PERSONS = 6


class PoseImageDataset(Dataset):
    """Dataset that loads images with per-person pose data and applies
    joint geometric augmentation.

    Returns: (img_tensor, pid, camid, trackid, img_path, pose_dict)
    """

    def __init__(self, dataset, pose_dir,
                 img_size=(384, 128),
                 is_train=False,
                 flip_prob=0.5,
                 pad=10,
                 re_prob=0.5,
                 pixel_mean=(0.5, 0.5, 0.5),
                 pixel_std=(0.5, 0.5, 0.5),
                 heatmap_size=None,
                 max_persons=MAX_PERSONS,
                 pose_guided_erasing=False):
        """
        Args:
            dataset: list of (img_path, pid, camid, trackid)
            pose_dir: directory containing index.json and .npz files for
                      this split (e.g. data/occluded_duke/pose_data/train/)
            img_size: (H, W) target image size after augmentation
            is_train: if True, enable augmentation
            flip_prob: horizontal flip probability
            pad: padding pixels for pad+crop augmentation
            re_prob: random erasing probability
            pixel_mean/pixel_std: image normalization params
            heatmap_size: (H, W) final heatmap size; None = same as img_size
            max_persons: max persons to return per image
            pose_guided_erasing: if True, use pose-guided erasing instead of RE
        """
        self.dataset = dataset
        self.img_size = img_size
        self.is_train = is_train
        self.flip_prob = flip_prob if is_train else 0.0
        self.pad = pad if is_train else 0
        self.re_prob = re_prob if is_train else 0.0
        self.pixel_mean = torch.tensor(pixel_mean, dtype=torch.float32).view(3, 1, 1)
        self.pixel_std = torch.tensor(pixel_std, dtype=torch.float32).view(3, 1, 1)
        self.heatmap_size = heatmap_size or img_size
        self.max_persons = min(max_persons, MAX_PERSONS)
        self.pose_guided_erasing = pose_guided_erasing and is_train

        # Load index
        index_path = os.path.join(pose_dir, 'index.json')
        self.pose_dir = pose_dir
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                self.index = json.load(f)
            print(f"  Loaded pose index: {index_path} ({len(self.index)} entries)")
        else:
            self.index = {}
            print(f"  WARNING: pose index not found at {index_path}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, pid, camid, trackid = self.dataset[idx]
        img = read_image(img_path)  # PIL RGB
        orig_w, orig_h = img.size

        fname = os.path.basename(img_path)

        # ---- Load all persons' pose data ----
        persons = self._load_persons(fname, orig_h, orig_w)
        # persons: list of dict {heatmap: (17,H,W) tensor, kp: (17,2) ndarray,
        #                        scores: (17,) ndarray,
        #                        visibility: (17,) ndarray,
        #                        visibility_binary: (17,) ndarray}
        n_persons = len(persons)

        # ---- Joint Augmentation ----
        target_h, target_w = self.img_size

        # 1) Resize image and pose data to target size
        img, persons = self._joint_resize(img, persons, orig_h, orig_w,
                                          target_h, target_w)

        # 2) Random horizontal flip
        flipped = False
        if self.is_train and random.random() < self.flip_prob:
            flipped = True
            img, persons = self._joint_flip(img, persons, target_w)

        # 3) Pad + random crop
        crop_x, crop_y = 0, 0
        if self.is_train and self.pad > 0:
            img, persons, crop_x, crop_y = self._joint_pad_crop(
                img, persons, target_h, target_w, self.pad)

        # 4) Convert image to tensor + normalize
        img_tensor = self._image_to_tensor(img)

        # 5) Erasing: Pose-Guided Erasing (PGE) or Random Erasing (RE)
        erase_box = None
        erased_channels = None
        if self.is_train and random.random() < self.re_prob:
            if self.pose_guided_erasing and persons:
                img_tensor, erase_box, erased_channels = \
                    self._pose_guided_erase(img_tensor, persons)
            else:
                img_tensor, erase_box = self._random_erase(img_tensor)

        if erase_box is not None:
            ex1, ey1, ex2, ey2 = erase_box
            for p in persons:
                kp = p['kp']
                in_box = ((kp[:, 0] >= ex1) & (kp[:, 0] < ex2) &
                          (kp[:, 1] >= ey1) & (kp[:, 1] < ey2))
                p['scores'][in_box] = 0.0

        # For PGE, zero heatmap channels only within the erased spatial region
        if erased_channels is not None and erase_box is not None:
            ex1, ey1, ex2, ey2 = erase_box
            for p in persons:
                for ch in erased_channels:
                    p['heatmap'][ch, ey1:ey2, ex1:ex2] = 0.0

        # ---- Assemble output tensors ----
        hm_h, hm_w = self.heatmap_size
        out_heatmaps = torch.zeros(self.max_persons, 17, hm_h, hm_w)
        out_keypoints = torch.zeros(self.max_persons, 17, 2)
        out_scores = torch.zeros(self.max_persons, 17)
        out_visibility = torch.zeros(self.max_persons, 17)
        out_visibility_binary = torch.zeros(self.max_persons, 17)
        out_mask = torch.zeros(self.max_persons)

        for i in range(min(n_persons, self.max_persons)):
            hm = persons[i]['heatmap']
            if (hm_h, hm_w) != (target_h, target_w):
                hm = F.interpolate(
                    hm.unsqueeze(0), size=(hm_h, hm_w),
                    mode='bilinear', align_corners=False).squeeze(0)
            out_heatmaps[i] = hm
            out_keypoints[i] = torch.from_numpy(persons[i]['kp'].copy())
            out_scores[i] = torch.from_numpy(persons[i]['scores'].copy())
            out_visibility[i] = torch.from_numpy(
                persons[i]['visibility'].copy())
            out_visibility_binary[i] = torch.from_numpy(
                persons[i]['visibility_binary'].copy())
            out_mask[i] = 1.0

        pose_dict = {
            'heatmaps': out_heatmaps,
            'keypoints': out_keypoints,
            'scores': out_scores,
            'visibility': out_visibility,
            'visibility_binary': out_visibility_binary,
            'person_mask': out_mask,
            'num_persons': min(n_persons, self.max_persons),
        }

        return img_tensor, pid, camid, trackid, img_path, pose_dict

    # ------------------------------------------------------------------
    #  Loading
    # ------------------------------------------------------------------

    def _load_persons(self, filename, img_h, img_w):
        """Load per-person npz files and project heatmaps to full-image space.

        Returns list of dicts with keys: heatmap (17,img_h,img_w), kp (17,2),
        scores (17,), visibility (17,), visibility_binary (17,).
        Target person is always placed at index 0.
        """
        entry = self.index.get(filename)
        if entry is None:
            return []

        # Reorder persons so target is first
        person_files = list(entry['persons'][:self.max_persons])
        target_idx = entry.get('target_person_idx', 0)
        if 0 < target_idx < len(person_files):
            # Move target to front, keep others in original order
            target_file = person_files.pop(target_idx)
            person_files.insert(0, target_file)

        persons = []
        for npz_name in person_files:
            # Support both relative (normal) and absolute (merged val) paths
            if os.path.isabs(npz_name):
                npz_path = npz_name
            else:
                npz_path = os.path.join(self.pose_dir, npz_name)
            if not os.path.exists(npz_path):
                continue
            with np.load(npz_path) as data:
                hm_raw = torch.from_numpy(
                    data['heatmap'].astype(np.float32))   # (17, 64, 48)
                kp = data['keypoints'].astype(np.float32)  # (17, 2) image pixels
                scores = data['scores'].astype(np.float32)  # (17,)
                crop_bounds = data['crop_bounds'].astype(np.float32)  # (4,)
                if 'visibility' in data.files:
                    visibility = data['visibility'].astype(np.float32)
                else:
                    # Backward compatibility for legacy pose_data without
                    # explicit visibility extraction.
                    visibility = np.clip(scores, 0.0, 1.0).astype(np.float32)
                if 'visibility_binary' in data.files:
                    visibility_binary = data['visibility_binary'].astype(
                        np.float32)
                else:
                    visibility_binary = (visibility >= 0.5).astype(np.float32)

            # Project bbox-local heatmap onto full-image canvas
            hm_full = self._place_heatmap(hm_raw, crop_bounds, img_h, img_w)

            persons.append({
                'heatmap': hm_full,
                'kp': kp.copy(),
                'scores': scores.copy(),
                'visibility': visibility.copy(),
                'visibility_binary': visibility_binary.copy(),
            })

        return persons

    @staticmethod
    def _place_heatmap(heatmap, crop_bounds, img_h, img_w):
        """Resize heatmap to crop_bounds region and place on full-image canvas.

        Args:
            heatmap: (17, hm_h, hm_w) in crop-local space
            crop_bounds: [x1, y1, x2, y2] actual crop region in image pixels
                         (may extend beyond image bounds)
            img_h, img_w: full image dimensions

        Returns:
            (17, img_h, img_w) heatmap in full-image space
        """
        cx1, cy1, cx2, cy2 = crop_bounds
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1

        # Resize heatmap to crop dimensions
        crop_w_int = max(int(round(crop_w)), 1)
        crop_h_int = max(int(round(crop_h)), 1)
        hm_resized = F.interpolate(
            heatmap.unsqueeze(0), size=(crop_h_int, crop_w_int),
            mode='bilinear', align_corners=False).squeeze(0)

        # Compute overlap between crop region and image
        # Source region in resized heatmap
        src_x1 = max(0, int(round(-cx1)))
        src_y1 = max(0, int(round(-cy1)))
        # Destination region in image
        dst_x1 = max(0, int(round(cx1)))
        dst_y1 = max(0, int(round(cy1)))
        dst_x2 = min(img_w, int(round(cx2)))
        dst_y2 = min(img_h, int(round(cy2)))

        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return torch.zeros(17, img_h, img_w)

        # Clamp copy size to actual available data on both sides
        copy_w = min(dst_x2 - dst_x1, crop_w_int - src_x1)
        copy_h = min(dst_y2 - dst_y1, crop_h_int - src_y1)
        if copy_w <= 0 or copy_h <= 0:
            return torch.zeros(17, img_h, img_w)

        canvas = torch.zeros(17, img_h, img_w)
        canvas[:, dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] = \
            hm_resized[:, src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w]
        return canvas

    # ------------------------------------------------------------------
    #  Joint augmentations
    # ------------------------------------------------------------------

    @staticmethod
    def _joint_resize(img, persons, orig_h, orig_w, target_h, target_w):
        """Resize image and all pose data to (target_h, target_w)."""
        img = img.resize((target_w, target_h), Image.BICUBIC)

        sx = target_w / orig_w
        sy = target_h / orig_h

        for p in persons:
            p['heatmap'] = F.interpolate(
                p['heatmap'].unsqueeze(0), size=(target_h, target_w),
                mode='bilinear', align_corners=False).squeeze(0)
            p['kp'][:, 0] *= sx
            p['kp'][:, 1] *= sy

        return img, persons

    @staticmethod
    def _joint_flip(img, persons, width):
        """Horizontal flip image + heatmaps + keypoints."""
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

        for p in persons:
            # Flip heatmap spatially
            p['heatmap'] = p['heatmap'].flip(-1)
            # Swap left-right channels
            for l, r in FLIP_PAIRS:
                p['heatmap'][[l, r]] = p['heatmap'][[r, l]]

            # Mirror x coordinates
            p['kp'][:, 0] = width - 1 - p['kp'][:, 0]
            # Swap left-right keypoints
            for l, r in FLIP_PAIRS:
                p['kp'][[l, r]] = p['kp'][[r, l]]
                p['scores'][[l, r]] = p['scores'][[r, l]]

        return img, persons

    @staticmethod
    def _joint_pad_crop(img, persons, target_h, target_w, pad):
        """Pad image+heatmaps with zeros, then random crop back to target size.

        Returns: (cropped_img, persons, crop_x, crop_y)
        """
        # Pad image
        padded = Image.new(img.mode,
                           (target_w + 2 * pad, target_h + 2 * pad),
                           (0, 0, 0))
        padded.paste(img, (pad, pad))

        # Random crop offsets
        crop_x = random.randint(0, 2 * pad)
        crop_y = random.randint(0, 2 * pad)
        img_cropped = padded.crop((crop_x, crop_y,
                                   crop_x + target_w, crop_y + target_h))

        for p in persons:
            # Pad heatmap
            p['heatmap'] = F.pad(p['heatmap'],
                                 (pad, pad, pad, pad), value=0)
            # Crop heatmap with same offsets
            p['heatmap'] = p['heatmap'][:,
                                        crop_y:crop_y + target_h,
                                        crop_x:crop_x + target_w]

            # Offset keypoints: pad shifts, then crop shifts back
            p['kp'][:, 0] += pad - crop_x
            p['kp'][:, 1] += pad - crop_y

            # Mark out-of-bounds keypoints
            oob = ((p['kp'][:, 0] < 0) | (p['kp'][:, 0] >= target_w) |
                   (p['kp'][:, 1] < 0) | (p['kp'][:, 1] >= target_h))
            p['scores'][oob] = 0.0
            p['kp'][:, 0] = np.clip(p['kp'][:, 0], 0, target_w - 1)
            p['kp'][:, 1] = np.clip(p['kp'][:, 1], 0, target_h - 1)

        return img_cropped, persons, crop_x, crop_y

    def _image_to_tensor(self, img):
        """PIL Image -> normalized float tensor (3, H, W)."""
        arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # (3, H, W)
        return (tensor - self.pixel_mean) / self.pixel_std

    @staticmethod
    def _random_erase(img_tensor, sl=0.02, sh=0.4, r1=0.3):
        """Random erasing on image tensor. Returns (tensor, erase_box_or_None).

        erase_box is (x1, y1, x2, y2) in pixel coords if erasing happened.
        """
        _, h, w = img_tensor.shape
        area = h * w

        for _ in range(100):
            target_area = random.uniform(sl, sh) * area
            aspect = random.uniform(r1, 1.0 / r1)

            eh = int(round(math.sqrt(target_area * aspect)))
            ew = int(round(math.sqrt(target_area / aspect)))

            if ew < w and eh < h:
                ey = random.randint(0, h - eh)
                ex = random.randint(0, w - ew)
                img_tensor[:, ey:ey + eh, ex:ex + ew] = torch.empty(
                    3, eh, ew).normal_()
                return img_tensor, (ex, ey, ex + ew, ey + eh)

        return img_tensor, None

    def _pose_guided_erase(self, img_tensor, persons):
        """Pose-guided erasing: erase a body part region based on pose.

        Selects a random body part group, computes the bounding box from
        keypoints of person 0, and erases that region with random noise.

        Returns:
            (img_tensor, erase_box_or_None, erased_channels_or_None)
        """
        _, h, w = img_tensor.shape
        p0 = persons[0]
        kp = p0['kp']       # (17, 2) in pixel coords
        scores = p0['scores']  # (17,)

        # Randomly select a body part group
        group = random.choice(PGE_BODY_PARTS)

        # Get valid keypoints for this group (score > 0.3)
        group_indices = np.array(group)
        valid = scores[group_indices] > 0.3
        if valid.sum() < 2:
            # Not enough keypoints detected, fall back to random erase
            result, box = self._random_erase(img_tensor)
            return result, box, None

        valid_kps = kp[group_indices][valid]  # (n_valid, 2)

        # Compute bounding box with margin
        margin_x = int(w * 0.15)  # ~19px for 128w
        margin_y = int(h * 0.08)  # ~31px for 384h
        x1 = max(0, int(valid_kps[:, 0].min()) - margin_x)
        y1 = max(0, int(valid_kps[:, 1].min()) - margin_y)
        x2 = min(w, int(valid_kps[:, 0].max()) + margin_x)
        y2 = min(h, int(valid_kps[:, 1].max()) + margin_y)

        box_h = y2 - y1
        box_w = x2 - x1
        if box_h < 5 or box_w < 5:
            result, box = self._random_erase(img_tensor)
            return result, box, None

        # Fill erased region with random noise
        img_tensor[:, y1:y2, x1:x2] = torch.empty(3, box_h, box_w).normal_()

        return img_tensor, (x1, y1, x2, y2), group


# ------------------------------------------------------------------
#  Collate functions
# ------------------------------------------------------------------

def _collate_pose_dicts(pose_dicts):
    """Stack a list of pose_dict into batched tensors."""
    batched = {}
    for key in pose_dicts[0]:
        if key == 'num_persons':
            batched[key] = torch.tensor(
                [d[key] for d in pose_dicts], dtype=torch.int64)
        else:
            batched[key] = torch.stack(
                [d[key] for d in pose_dicts], dim=0)
    return batched


def pose_train_collate_fn(batch):
    imgs, pids, camids, viewids, _, pose_dicts = zip(*batch)
    return (torch.stack(imgs, dim=0),
            torch.tensor(pids, dtype=torch.int64),
            torch.tensor(camids, dtype=torch.int64),
            torch.tensor(viewids, dtype=torch.int64),
            _collate_pose_dicts(pose_dicts))


def pose_val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths, pose_dicts = zip(*batch)
    return (torch.stack(imgs, dim=0),
            pids,
            camids,
            torch.tensor(camids, dtype=torch.int64),
            torch.tensor(viewids, dtype=torch.int64),
            img_paths,
            _collate_pose_dicts(pose_dicts))
