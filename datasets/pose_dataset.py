"""Pose-aware image dataset that loads per-person keypoints and heatmaps."""
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .bases import read_image


class PoseImageDataset(Dataset):
    """ImageDataset with multi-person pose data loading.

    Loads pre-extracted per-person keypoints, scores, bboxes, and heatmaps.
    Returns (img, pid, camid, trackid, img_path, pose_data_dict).

    pose_data_dict contains:
        - 'primary_keypoints': (17, 2) normalized coords of target person
        - 'primary_scores': (17,) keypoint confidence
        - 'primary_heatmap': (17, H, W) raw heatmap of target person
        - 'num_persons': int
        - 'all_keypoints': (max_persons, 17, 2) all persons' coords (padded)
        - 'all_scores': (max_persons, 17) all persons' scores (padded)
        - 'all_heatmaps': (max_persons, 17, H, W) all persons' heatmaps (padded)
        - 'all_bboxes': (max_persons, 4) all persons' bboxes (padded)
        - 'person_mask': (max_persons,) validity mask
    """

    MAX_PERSONS = 6  # max persons per image (pad/truncate to this)

    def __init__(self, dataset, transform=None, pose_data=None,
                 heatmap_dir=None, target_size=None):
        """
        Args:
            dataset: list of (img_path, pid, camid, trackid)
            transform: image transforms
            pose_data: dict mapping filename -> person list, OR
                       dict with 'filenames'/'keypoints'/'scores' (legacy NPZ format)
            heatmap_dir: directory or list of dirs containing .npy heatmap files
            target_size: (H, W) to resize heatmaps (e.g. (12, 4) for Swin stage3)
        """
        self.dataset = dataset
        self.transform = transform
        self.target_size = target_size
        self.hm_h = target_size[0] if target_size else 64
        self.hm_w = target_size[1] if target_size else 48

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
                # Legacy NPZ format: convert to new format
                for idx, fname in enumerate(pose_data['filenames']):
                    self.pose_lookup[fname] = [{
                        'keypoints': pose_data['keypoints'][idx],
                        'kp_scores': pose_data['scores'][idx],
                        'bbox': np.array([0, 0, 1, 1], dtype=np.float32),
                        'heatmap_file': fname.replace('.jpg', '_p0.npy'),
                    }]
            elif isinstance(pose_data, dict):
                # New PKL format: {filename: [person_list]}
                self.pose_lookup = pose_data
            elif isinstance(pose_data, list):
                # List of per-image dicts from PKL
                for entry in pose_data:
                    self.pose_lookup[entry['filename']] = entry['persons']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        fname = os.path.basename(img_path)
        persons = self.pose_lookup.get(fname, [])

        # Build primary person data
        if len(persons) > 0:
            p0 = persons[0]  # primary = largest bbox (sorted during extraction)
            primary_kp = torch.from_numpy(
                np.array(p0['keypoints'], dtype=np.float32)).float()
            primary_scores = torch.from_numpy(
                np.array(p0['kp_scores'], dtype=np.float32)).float()
            primary_hm = self._load_heatmap(p0.get('heatmap_file'))
        else:
            primary_kp = torch.zeros(17, 2)
            primary_scores = torch.zeros(17)
            primary_hm = torch.zeros(17, self.hm_h, self.hm_w)

        # Build all-persons data (padded to MAX_PERSONS)
        n_persons = min(len(persons), self.MAX_PERSONS)
        all_kp = torch.zeros(self.MAX_PERSONS, 17, 2)
        all_scores = torch.zeros(self.MAX_PERSONS, 17)
        all_hm = torch.zeros(self.MAX_PERSONS, 17, self.hm_h, self.hm_w)
        all_bboxes = torch.zeros(self.MAX_PERSONS, 4)
        person_mask = torch.zeros(self.MAX_PERSONS)

        for i in range(n_persons):
            p = persons[i]
            all_kp[i] = torch.from_numpy(
                np.array(p['keypoints'], dtype=np.float32))
            all_scores[i] = torch.from_numpy(
                np.array(p['kp_scores'], dtype=np.float32))
            all_hm[i] = self._load_heatmap(p.get('heatmap_file'))
            bbox = p.get('bbox', np.array([0, 0, 1, 1], dtype=np.float32))
            all_bboxes[i] = torch.from_numpy(
                np.array(bbox, dtype=np.float32))
            person_mask[i] = 1.0

        pose_dict = {
            'primary_keypoints': primary_kp,       # (17, 2)
            'primary_scores': primary_scores,       # (17,)
            'primary_heatmap': primary_hm,          # (17, H, W)
            'num_persons': n_persons,
            'all_keypoints': all_kp,                # (MAX, 17, 2)
            'all_scores': all_scores,               # (MAX, 17)
            'all_heatmaps': all_hm,                 # (MAX, 17, H, W)
            'all_bboxes': all_bboxes,               # (MAX, 4)
            'person_mask': person_mask,             # (MAX,)
        }

        return img, pid, camid, trackid, img_path, pose_dict

    def _load_heatmap(self, hm_filename):
        """Load raw heatmap .npy file and optionally resize."""
        if hm_filename is None:
            return torch.zeros(17, self.hm_h, self.hm_w)

        # Search through all heatmap directories
        for hdir in self.heatmap_dirs:
            npy_path = os.path.join(hdir, hm_filename)
            if os.path.exists(npy_path):
                heatmap = np.load(npy_path)  # (17, 64, 48)
                heatmap = torch.from_numpy(heatmap).float()

                # Resize if needed
                if self.target_size is not None:
                    heatmap = F.interpolate(
                        heatmap.unsqueeze(0),
                        size=self.target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                return heatmap

        return torch.zeros(17, self.hm_h, self.hm_w)


def load_pose_data(pose_dir, split):
    """Load pose data from PKL or fallback to NPZ.

    Args:
        pose_dir: directory containing pose data files
        split: 'train', 'gallery', or 'query'

    Returns:
        list of per-image dicts (from PKL), or
        legacy dict with 'filenames'/'keypoints'/'scores' (from NPZ)
    """
    # Try new PKL format first
    pkl_path = os.path.join(pose_dir, f'pose_data_{split}.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded pose data (PKL): {pkl_path} ({len(data)} images)")
        return data

    # Fall back to legacy NPZ
    npz_path = os.path.join(pose_dir, f'pose_data_{split}.npz')
    if not os.path.exists(npz_path):
        # Try old naming convention
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
