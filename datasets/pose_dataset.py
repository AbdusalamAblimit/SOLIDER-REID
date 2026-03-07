"""Pose-aware image dataset that loads keypoints alongside images."""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .bases import read_image


class PoseImageDataset(Dataset):
    """ImageDataset with pose keypoint loading.

    Loads pre-extracted keypoints and scores from NPZ files.
    Returns (img, pid, camid, trackid, img_path, keypoints, kp_scores).
    """

    def __init__(self, dataset, transform=None, pose_data=None):
        """
        Args:
            dataset: list of (img_path, pid, camid, trackid)
            transform: image transforms
            pose_data: dict with 'filenames', 'keypoints', 'scores' arrays
                       If None, returns zero keypoints.
        """
        self.dataset = dataset
        self.transform = transform

        # Build filename -> index lookup
        self.pose_lookup = {}
        if pose_data is not None:
            filenames = pose_data['filenames']
            self.pose_keypoints = pose_data['keypoints']  # (N, 17, 2)
            self.pose_scores = pose_data['scores']        # (N, 17)
            for idx, fname in enumerate(filenames):
                self.pose_lookup[fname] = idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        # Look up pose data by filename
        fname = os.path.basename(img_path)
        if fname in self.pose_lookup:
            pose_idx = self.pose_lookup[fname]
            keypoints = torch.from_numpy(
                self.pose_keypoints[pose_idx].copy()).float()
            kp_scores = torch.from_numpy(
                self.pose_scores[pose_idx].copy()).float()
        else:
            keypoints = torch.zeros(17, 2)
            kp_scores = torch.zeros(17)

        return img, pid, camid, trackid, img_path, keypoints, kp_scores


def load_pose_data(pose_dir, split):
    """Load pose data from NPZ file.

    Args:
        pose_dir: directory containing pose_rtmpose_*.npz
        split: 'train', 'gallery', or 'query'

    Returns:
        dict with 'filenames', 'keypoints', 'scores'
    """
    npz_path = os.path.join(pose_dir, f'pose_rtmpose_{split}.npz')
    if not os.path.exists(npz_path):
        print(f"Warning: Pose data not found at {npz_path}")
        return None
    data = np.load(npz_path)
    return {
        'filenames': data['filenames'],
        'keypoints': data['keypoints'],
        'scores': data['scores'],
    }
