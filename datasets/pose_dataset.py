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

PCVT_BODY_PARTS = {
    'head': [0, 1, 2, 3, 4],
    'left_arm': [5, 7, 9],
    'right_arm': [6, 8, 10],
    'torso': [5, 6, 11, 12],
    'left_leg': [11, 13, 15],
    'right_leg': [12, 14, 16],
}

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
                 pose_guided_erasing=False,
                 occluders=None,
                 roa_prob=0.5):
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
            occluders: list of RGBA patches for Realistic Occlusion Augmentation
            roa_prob: probability of applying ROA per image
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
        self.occluders = occluders if is_train else None
        self.roa_prob = roa_prob if is_train else 0.0
        self.pose_aware_roa = False  # set by make_dataloader if configured
        self.parallel_aug = False   # set by make_dataloader if PARALLEL_AUG enabled
        self.lower_body_occ = False      # PLBOA
        self.lower_body_occ_prob = 0.5
        self.lower_body_occ_ratio = 0.5
        self.lower_body_occ_mode = 'lower'  # 'lower' or 'gradient'
        self.upper_body_occ = False     # PGMPOA: additionally occlude upper body parts
        self.upper_body_occ_prob = 0.3  # probability of upper-body occlusion (on top of PLBOA)
        self.pcvt = False
        self.pcvt_resp_thr = 0.10
        self.pcvt_active_thr = 0.30
        self.pcvt_min_parts = 2
        self.pcvt_fill_value = 0.0
        self.pcvt_random = False  # Random block masking control

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

        pcvt_meta = None
        self._oa_sd_mode = getattr(self, '_oa_sd_mode', False)

        # OA-SD: save clean image BEFORE PLBOA for teacher view
        img_clean_for_oa_sd = None
        if self._oa_sd_mode and self.is_train:
            img_clean_for_oa_sd = img.copy()

        # PLBOA/PGMPOA: apply BEFORE branching so all views (parallel/pcvt/standard) share it
        if self.is_train:
            if self.lower_body_occ and persons and random.random() < self.lower_body_occ_prob:
                img = self._apply_lower_body_occlusion(img, persons)
            if self.upper_body_occ and persons and random.random() < self.upper_body_occ_prob:
                img = self._apply_upper_body_part_occlusion(img, persons)

        # ---- Parallel Augmentation (3 views) or Standard (1 view) ----
        if self.pcvt and self.is_train:
            base_tensor = self._image_to_tensor(img)

            # Keep the baseline full-view regularization on the anchor branch.
            img_full_tensor = base_tensor.clone()
            if random.random() < self.re_prob:
                img_full_tensor, _ = self._random_erase(img_full_tensor)

            img_a_tensor, img_b_tensor, pcvt_meta = self._make_pcvt_views(
                base_tensor, persons)
            img_tensor = (img_full_tensor, img_a_tensor, img_b_tensor)

        elif self.parallel_aug and self.is_train:
            # Parallel mode: create 3 image variants from shared base
            # view_full: standard RE
            # view_roa: ROA occlusion
            # view_heavy: forced RE (100% probability)

            # View 1: Full (standard pipeline with normal RE)
            img_full_tensor = self._image_to_tensor(img)
            if random.random() < self.re_prob:
                img_full_tensor, _ = self._random_erase(img_full_tensor)
                # Note: do NOT update persons for erase in parallel mode
                # because pose_dict is shared across all 3 views

            # View 2: ROA (paste occlusion objects)
            img_roa = img.copy()
            if self.occluders:
                img_roa_np = np.array(img_roa)
                from .occlusion_augmentation import occlude_with_objects
                img_roa_np = occlude_with_objects(img_roa_np, self.occluders, n=1,
                                                  min_overlap=0.2, max_overlap=0.5)
                img_roa = Image.fromarray(img_roa_np)
            img_roa_tensor = self._image_to_tensor(img_roa)

            # View 3: Heavy (forced random erasing, 100% probability)
            img_heavy_tensor = self._image_to_tensor(img)
            img_heavy_tensor, _ = self._random_erase(img_heavy_tensor)

            img_tensor = (img_full_tensor, img_roa_tensor, img_heavy_tensor)
        else:
            # Standard single-view pipeline
            # 3.5) Realistic Occlusion Augmentation (ROA): paste VOC objects
            if self.occluders and random.random() < self.roa_prob:
                img_np = np.array(img)  # PIL → numpy (H, W, 3)
                if self.pose_aware_roa and persons:
                    from .occlusion_augmentation import pose_aware_occlude
                    p0 = persons[0]
                    img_np = pose_aware_occlude(
                        img_np, self.occluders,
                        keypoints=p0['kp'], scores=p0['scores'],
                        n=1, min_overlap=0.2, max_overlap=0.5)
                else:
                    from .occlusion_augmentation import occlude_with_objects
                    img_np = occlude_with_objects(img_np, self.occluders, n=1,
                                                  min_overlap=0.2, max_overlap=0.5)
                img = Image.fromarray(img_np)  # numpy → PIL

            # (PLBOA/PGMPOA already applied above, before branching)

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
                self._update_persons_for_erase(persons, erase_box, erased_channels)

            # OA-SD: create teacher view from clean (pre-PLBOA) image
            if img_clean_for_oa_sd is not None:
                img_clean_tensor = self._image_to_tensor(img_clean_for_oa_sd)
                # Teacher view gets mild RE (same as student)
                if random.random() < self.re_prob:
                    img_clean_tensor, _ = self._random_erase(img_clean_tensor)
                img_tensor = (img_tensor, img_clean_tensor)  # (student_occluded, teacher_clean)
            else:
                img_tensor = (img_tensor,)  # wrap in tuple for uniform interface

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
        if pcvt_meta is not None:
            pose_dict.update(pcvt_meta)

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
                p['visibility'][[l, r]] = p['visibility'][[r, l]]
                p['visibility_binary'][[l, r]] = p['visibility_binary'][[r, l]]

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
            p['visibility'][oob] = 0.0
            p['visibility_binary'][oob] = 0.0
            p['kp'][:, 0] = np.clip(p['kp'][:, 0], 0, target_w - 1)
            p['kp'][:, 1] = np.clip(p['kp'][:, 1], 0, target_h - 1)

        return img_cropped, persons, crop_x, crop_y

    def _image_to_tensor(self, img):
        """PIL Image -> normalized float tensor (3, H, W)."""
        arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # (3, H, W)
        return (tensor - self.pixel_mean) / self.pixel_std

    def _make_pcvt_views(self, base_tensor, persons):
        """Create two pose-defined complementary masked views.

        Returns:
            view_a, view_b: (3, H, W) normalized tensors
            meta: dict of scalar tensors for logging
        """
        _, H, W = base_tensor.shape

        # Random block masking control (exp150)
        if self.pcvt_random:
            return self._make_random_block_views(base_tensor, H, W)

        if not persons:
            view_a, _ = self._random_erase(base_tensor.clone())
            view_b, _ = self._random_erase(base_tensor.clone())
            meta = {
                'pcvt_cov_a': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_cov_b': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_cov_u': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_ovr': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_mga': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_mgb': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_gca': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_gcb': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_fb': torch.tensor(1.0, dtype=torch.float32),
            }
            return view_a, view_b, meta

        hm = persons[0]['heatmap']  # (17, H, W)
        visible_parts = []
        for part_name, kp_indices in PCVT_BODY_PARTS.items():
            part_hm = hm[kp_indices]
            response = part_hm.max(dim=0)[0]
            peak = float(response.max().item())
            if peak < self.pcvt_resp_thr:
                continue
            active = response > (peak * self.pcvt_active_thr)
            if not active.any():
                continue
            masked_response = torch.where(active, response, torch.zeros_like(response))
            visible_parts.append((part_name, masked_response))

        fallback = 0.0
        if len(visible_parts) < self.pcvt_min_parts:
            fallback = 1.0
            view_a, box_a = self._random_erase(base_tensor.clone())
            view_b, box_b = self._random_erase(base_tensor.clone())
            area_a = 0.0
            area_b = 0.0
            if box_a is not None:
                x1, y1, x2, y2 = box_a
                area_a = max(0, x2 - x1) * max(0, y2 - y1) / float(H * W)
            if box_b is not None:
                x1, y1, x2, y2 = box_b
                area_b = max(0, x2 - x1) * max(0, y2 - y1) / float(H * W)
            meta = {
                'pcvt_cov_a': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_cov_b': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_cov_u': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_ovr': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_mga': torch.tensor(area_a, dtype=torch.float32),
                'pcvt_mgb': torch.tensor(area_b, dtype=torch.float32),
                'pcvt_gca': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_gcb': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_fb': torch.tensor(fallback, dtype=torch.float32),
            }
            return view_a, view_b, meta

        # Greedy balance by visible support area.
        part_stack = torch.stack([resp for _, resp in visible_parts], dim=0)
        visible_union = part_stack.max(dim=0)[0] > 0
        owner = part_stack.argmax(dim=0)

        part_areas = []
        for idx, (name, _) in enumerate(visible_parts):
            owned_mask = visible_union & (owner == idx)
            area = float(owned_mask.float().sum().item())
            if area <= 0:
                continue
            part_areas.append((name, owned_mask, area))
        if len(part_areas) < self.pcvt_min_parts:
            fallback = 1.0
            view_a, box_a = self._random_erase(base_tensor.clone())
            view_b, box_b = self._random_erase(base_tensor.clone())
            area_a = 0.0
            area_b = 0.0
            if box_a is not None:
                x1, y1, x2, y2 = box_a
                area_a = max(0, x2 - x1) * max(0, y2 - y1) / float(H * W)
            if box_b is not None:
                x1, y1, x2, y2 = box_b
                area_b = max(0, x2 - x1) * max(0, y2 - y1) / float(H * W)
            meta = {
                'pcvt_cov_a': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_cov_b': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_cov_u': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_ovr': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_mga': torch.tensor(area_a, dtype=torch.float32),
                'pcvt_mgb': torch.tensor(area_b, dtype=torch.float32),
                'pcvt_gca': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_gcb': torch.tensor(0.0, dtype=torch.float32),
                'pcvt_fb': torch.tensor(fallback, dtype=torch.float32),
            }
            return view_a, view_b, meta
        part_areas.sort(key=lambda x: x[2], reverse=True)
        bucket_a, bucket_b = [], []
        area_a = 0.0
        area_b = 0.0
        for name, mask, area in part_areas:
            if area_a <= area_b:
                bucket_a.append(mask)
                area_a += area
            else:
                bucket_b.append(mask)
                area_b += area

        drop_a = torch.zeros(H, W, dtype=torch.bool)
        drop_b = torch.zeros(H, W, dtype=torch.bool)
        for mask in bucket_a:
            drop_a |= mask
        for mask in bucket_b:
            drop_b |= mask

        view_a = base_tensor.clone()
        view_b = base_tensor.clone()
        view_a[:, drop_a] = self.pcvt_fill_value
        view_b[:, drop_b] = self.pcvt_fill_value

        visible_area = visible_union.float().sum().clamp(min=1.0)
        keep_a = visible_union & (~drop_a)
        keep_b = visible_union & (~drop_b)
        cov_a = float(keep_a.float().sum().item() / visible_area.item())
        cov_b = float(keep_b.float().sum().item() / visible_area.item())
        cov_u = float((keep_a | keep_b).float().sum().item() / visible_area.item())
        overlap = float((drop_a & drop_b & visible_union).float().sum().item() / visible_area.item())
        mga = float(drop_a.float().mean().item())
        mgb = float(drop_b.float().mean().item())

        meta = {
            'pcvt_cov_a': torch.tensor(cov_a, dtype=torch.float32),
            'pcvt_cov_b': torch.tensor(cov_b, dtype=torch.float32),
            'pcvt_cov_u': torch.tensor(cov_u, dtype=torch.float32),
            'pcvt_ovr': torch.tensor(overlap, dtype=torch.float32),
            'pcvt_mga': torch.tensor(mga, dtype=torch.float32),
            'pcvt_mgb': torch.tensor(mgb, dtype=torch.float32),
            'pcvt_gca': torch.tensor(float(len(bucket_a)), dtype=torch.float32),
            'pcvt_gcb': torch.tensor(float(len(bucket_b)), dtype=torch.float32),
            'pcvt_fb': torch.tensor(fallback, dtype=torch.float32),
        }
        return view_a, view_b, meta

    def _make_random_block_views(self, base_tensor, H, W):
        """Random block masking control for PCVT (exp150).

        Divides image into 8x4 grid (32 blocks), randomly assigns half to A, half to B.
        Guarantees ~50/50 coverage and perfect complementarity (A ∩ B = ∅).
        """
        block_h = H // 8  # 32 pixels per block vertically
        block_w = W // 4  # 32 pixels per block horizontally
        n_blocks = 8 * 4  # 32 blocks total

        # Random permutation, first half → A, second half → B
        perm = torch.randperm(n_blocks)
        set_a = set(perm[:n_blocks // 2].tolist())

        drop_a = torch.zeros(H, W, dtype=torch.bool)
        drop_b = torch.zeros(H, W, dtype=torch.bool)

        for idx in range(n_blocks):
            row = idx // 4
            col = idx % 4
            y1 = row * block_h
            y2 = min(y1 + block_h, H)
            x1 = col * block_w
            x2 = min(x1 + block_w, W)
            if idx in set_a:
                drop_a[y1:y2, x1:x2] = True
            else:
                drop_b[y1:y2, x1:x2] = True

        view_a = base_tensor.clone()
        view_b = base_tensor.clone()
        view_a[:, drop_a] = self.pcvt_fill_value
        view_b[:, drop_b] = self.pcvt_fill_value

        mga = float(drop_a.float().mean().item())
        mgb = float(drop_b.float().mean().item())

        meta = {
            'pcvt_cov_a': torch.tensor(0.5, dtype=torch.float32),
            'pcvt_cov_b': torch.tensor(0.5, dtype=torch.float32),
            'pcvt_cov_u': torch.tensor(1.0, dtype=torch.float32),
            'pcvt_ovr': torch.tensor(0.0, dtype=torch.float32),
            'pcvt_mga': torch.tensor(mga, dtype=torch.float32),
            'pcvt_mgb': torch.tensor(mgb, dtype=torch.float32),
            'pcvt_gca': torch.tensor(float(n_blocks // 2), dtype=torch.float32),
            'pcvt_gcb': torch.tensor(float(n_blocks - n_blocks // 2), dtype=torch.float32),
            'pcvt_fb': torch.tensor(0.0, dtype=torch.float32),
        }
        return view_a, view_b, meta

    def _apply_lower_body_occlusion(self, img, persons):
        """Pose-guided occlusion augmentation with real objects.

        Modes:
        - 'lower': Only occlude below hip (targeting 24.4% query lower-body occ)
        - 'gradient': Bottom-heavy gradient — lower body high prob, upper low prob
        """
        import numpy as np
        p0 = persons[0]
        kp = p0['kp']
        scores = p0['scores']

        w, h = img.size

        if self.lower_body_occ_mode == 'body_random':
            # Body-random mode: paste occluder at random position WITHIN body bbox
            valid_kp = kp[scores > 0.3]
            if len(valid_kp) < 3:
                return img
            body_y1 = max(0, int(valid_kp[:, 1].min()) - 5)
            body_y2 = min(h, int(valid_kp[:, 1].max()) + 5)
            body_x1 = max(0, int(valid_kp[:, 0].min()) - 5)
            body_x2 = min(w, int(valid_kp[:, 0].max()) + 5)
            body_h = body_y2 - body_y1
            body_w = body_x2 - body_x1
            if body_h < 20 or body_w < 10:
                return img
            # Random occlusion region within body bbox (30-60% of body area)
            occ_h = int(body_h * random.uniform(0.3, 0.6))
            occ_w = int(body_w * random.uniform(0.5, 1.0))
            occ_start = random.randint(body_y1, max(body_y1, body_y2 - occ_h))
            occ_x = random.randint(body_x1, max(body_x1, body_x2 - occ_w))
            # Use occ_start and following code to paste occluder
            occ_region_h = occ_h
            # Override h to be occ_start + occ_h for the paste region
            h_orig = h
            h = occ_start + occ_h

        elif self.lower_body_occ_mode == 'gradient':
            # Gradient mode: occlusion start sampled with bottom-heavy distribution
            # Find head and foot y-coordinates
            head_indices = [0, 1, 2, 3, 4]  # nose, eyes, ears
            foot_indices = [15, 16]  # ankles
            head_valid = scores[head_indices] > 0.3
            foot_valid = scores[foot_indices] > 0.3

            head_y = int(kp[head_indices][head_valid][:, 1].min()) if head_valid.any() else 0
            foot_y = int(kp[foot_indices][foot_valid][:, 1].max()) if foot_valid.any() else h

            body_h = max(foot_y - head_y, 20)
            # Sample occ_start with quadratic bias toward bottom
            # u ~ Uniform(0,1), then occ_start = head_y + body_h * u^2
            # This gives P(start > midpoint) = 1 - sqrt(0.5) ≈ 0.29 (bottom-heavy)
            u = random.random()
            occ_start = int(head_y + body_h * (u ** 2))
            occ_start = max(0, min(occ_start, h - 5))
        else:
            # Lower mode: only occlude below hip
            hip_indices = [11, 12]
            hip_valid = scores[hip_indices] > 0.3
            if not hip_valid.any():
                return img

            hip_ys = kp[hip_indices][hip_valid][:, 1]
            hip_y = int(hip_ys.mean())

            if hip_y >= h - 10:
                return img

            base_ratio = self.lower_body_occ_ratio
            occ_ratio = random.uniform(max(0.1, base_ratio - 0.2), min(1.0, base_ratio + 0.2))
            occ_start = int(hip_y + (h - hip_y) * (1.0 - occ_ratio))
            occ_start = max(0, min(occ_start, h - 1))

        occ_region_h = h - occ_start
        if occ_region_h < 10:
            return img

        img_np = np.array(img)

        # Paste a real VOC occluder if available, otherwise use solid fill
        if self.occluders:
            occluder = random.choice(self.occluders)
            occ_h, occ_w_orig = occluder.shape[:2]

            # Scale occluder to cover the lower body region
            target_w = int(w * random.uniform(0.6, 1.0))
            target_h = occ_region_h
            if occ_h > 0 and occ_w_orig > 0:
                import cv2
                occ_resized = cv2.resize(occluder, (target_w, target_h))
                occ_x = random.randint(0, max(0, w - target_w))

                # Alpha blend
                occ_rgb = occ_resized[:, :, :3]
                occ_alpha = occ_resized[:, :, 3:4].astype(np.float32) / 255.0

                y1 = occ_start
                y2 = min(y1 + target_h, h)
                x1 = occ_x
                x2 = min(x1 + target_w, w)
                actual_h = y2 - y1
                actual_w = x2 - x1

                if actual_h > 0 and actual_w > 0:
                    region = img_np[y1:y2, x1:x2].astype(np.float32)
                    patch = occ_rgb[:actual_h, :actual_w].astype(np.float32)
                    alpha = occ_alpha[:actual_h, :actual_w]
                    img_np[y1:y2, x1:x2] = (alpha * patch + (1.0 - alpha) * region).astype(np.uint8)
        else:
            # Fallback: gray fill
            occ_x = 0
            target_w = w
            gray_val = random.randint(60, 180)
            img_np[occ_start:h, :] = gray_val

        img = Image.fromarray(img_np)

        # Update person metadata for occluded keypoints
        occ_x_end = occ_x + target_w if self.occluders else w
        for p in persons:
            kp_p = p['kp']
            in_occ = ((kp_p[:, 1] >= occ_start) &
                      (kp_p[:, 0] >= occ_x) &
                      (kp_p[:, 0] < occ_x_end))
            p['scores'][in_occ] = 0.0
            p['visibility'][in_occ] = 0.0
            p['visibility_binary'][in_occ] = 0.0
            p['heatmap'][:, occ_start:h, occ_x:occ_x_end] = 0.0

        return img

    # Upper-body part groups for PGMPOA: (name, keypoint_indices, padding_ratio)
    _UPPER_BODY_PARTS = [
        ('head', [0, 1, 2, 3, 4], 0.3),         # nose, eyes, ears
        ('left_arm', [5, 7, 9], 0.2),            # left shoulder, elbow, wrist
        ('right_arm', [6, 8, 10], 0.2),          # right shoulder, elbow, wrist
    ]

    def _apply_upper_body_part_occlusion(self, img, persons):
        """PGMPOA: Occlude a random upper-body part with a real occluder.

        Randomly selects one of {head, left_arm, right_arm}, computes its
        bounding box from keypoints, and pastes an occluder patch over it.
        This complements PLBOA (lower-body) to create diverse occlusion patterns.
        """
        import numpy as np

        p0 = persons[0]
        kp = p0['kp']
        scores = p0['scores']
        w, h = img.size

        # Randomly select an upper-body part
        part_name, kp_indices, pad_ratio = random.choice(self._UPPER_BODY_PARTS)

        # Check that at least 2 keypoints of this part are visible
        part_scores = scores[kp_indices]
        visible = part_scores > 0.3
        if visible.sum() < 2:
            return img

        # Compute bounding box from visible keypoints
        part_kp = kp[kp_indices][visible]
        x_min = max(0, int(part_kp[:, 0].min()))
        x_max = min(w, int(part_kp[:, 0].max()))
        y_min = max(0, int(part_kp[:, 1].min()))
        y_max = min(h, int(part_kp[:, 1].max()))

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        if bbox_w < 5 or bbox_h < 5:
            return img

        # Pad the bbox to cover surrounding area (parts extend beyond keypoints)
        pad_x = int(bbox_w * pad_ratio)
        pad_y = int(bbox_h * pad_ratio)
        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + pad_y)

        occ_w = x2 - x1
        occ_h = y2 - y1
        if occ_w < 5 or occ_h < 5:
            return img

        img_np = np.array(img)

        # Paste occluder
        if self.occluders:
            import cv2
            occluder = random.choice(self.occluders)
            occ_orig_h, occ_orig_w = occluder.shape[:2]
            if occ_orig_h > 0 and occ_orig_w > 0:
                occ_resized = cv2.resize(occluder, (occ_w, occ_h))
                occ_rgb = occ_resized[:, :, :3]
                occ_alpha = occ_resized[:, :, 3:4].astype(np.float32) / 255.0

                region = img_np[y1:y2, x1:x2].astype(np.float32)
                patch = occ_rgb[:occ_h, :occ_w].astype(np.float32)
                alpha = occ_alpha[:occ_h, :occ_w]
                img_np[y1:y2, x1:x2] = (alpha * patch + (1.0 - alpha) * region).astype(np.uint8)
        else:
            # Fallback: gray fill
            gray_val = random.randint(60, 180)
            img_np[y1:y2, x1:x2] = gray_val

        img = Image.fromarray(img_np)

        # Update person metadata for occluded keypoints
        for p in persons:
            kp_p = p['kp']
            in_occ = ((kp_p[:, 1] >= y1) & (kp_p[:, 1] < y2) &
                      (kp_p[:, 0] >= x1) & (kp_p[:, 0] < x2))
            p['scores'][in_occ] = 0.0
            p['visibility'][in_occ] = 0.0
            p['visibility_binary'][in_occ] = 0.0
            # Zero out heatmap in the occluded region
            # Heatmap may have different resolution than image
            hm_h, hm_w = p['heatmap'].shape[1], p['heatmap'].shape[2]
            hm_y1 = int(y1 * hm_h / h)
            hm_y2 = int(y2 * hm_h / h)
            hm_x1 = int(x1 * hm_w / w)
            hm_x2 = int(x2 * hm_w / w)
            p['heatmap'][:, hm_y1:hm_y2, hm_x1:hm_x2] = 0.0

        return img

    @staticmethod
    def _update_persons_for_erase(persons, erase_box, erased_channels=None):
        """Update person keypoint scores/visibility for erased regions."""
        if erase_box is None:
            return
        ex1, ey1, ex2, ey2 = erase_box
        for p in persons:
            kp = p['kp']
            in_box = ((kp[:, 0] >= ex1) & (kp[:, 0] < ex2) &
                      (kp[:, 1] >= ey1) & (kp[:, 1] < ey2))
            p['scores'][in_box] = 0.0
            p['visibility'][in_box] = 0.0
            p['visibility_binary'][in_box] = 0.0
        # For PGE, zero heatmap channels only within the erased spatial region
        if erased_channels is not None:
            for p in persons:
                for ch in erased_channels:
                    p['heatmap'][ch, ey1:ey2, ex1:ex2] = 0.0

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
    img_tuples, pids, camids, viewids, _, pose_dicts = zip(*batch)
    # img_tuples: list of tuples, each is (tensor,) or (full, roa, heavy)
    n_views = len(img_tuples[0])
    if n_views == 1:
        # Standard single-view: unwrap tuple
        imgs = torch.stack([t[0] for t in img_tuples], dim=0)
    else:
        # Parallel augmentation: stack each view separately
        # Return list of (B, C, H, W) tensors
        imgs = [torch.stack([t[v] for t in img_tuples], dim=0)
                for v in range(n_views)]
    return (imgs,
            torch.tensor(pids, dtype=torch.int64),
            torch.tensor(camids, dtype=torch.int64),
            torch.tensor(viewids, dtype=torch.int64),
            _collate_pose_dicts(pose_dicts))


def pose_val_collate_fn(batch):
    img_tuples, pids, camids, viewids, img_paths, pose_dicts = zip(*batch)
    # Val always single-view
    imgs = torch.stack([t[0] if isinstance(t, tuple) else t for t in img_tuples], dim=0)
    return (imgs,
            pids,
            camids,
            torch.tensor(camids, dtype=torch.int64),
            torch.tensor(viewids, dtype=torch.int64),
            img_paths,
            _collate_pose_dicts(pose_dicts))
