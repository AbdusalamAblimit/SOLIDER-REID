"""Pose-aware image dataset that loads keypoints and heatmaps alongside images."""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .bases import read_image


class PoseImageDataset(Dataset):
    """ImageDataset with pose keypoint and heatmap loading.

    Loads pre-extracted keypoints/scores from NPZ and raw heatmaps from .npy files.
    Returns (img, pid, camid, trackid, img_path, keypoints, kp_scores, heatmap).
    """

    def __init__(self, dataset, transform=None, pose_data=None,
                 heatmap_dir=None, target_size=None):
        """
        Args:
            dataset: list of (img_path, pid, camid, trackid)
            transform: image transforms
            pose_data: dict with 'filenames', 'keypoints', 'scores' arrays
            heatmap_dir: directory or list of directories containing .npy heatmaps
            target_size: (H, W) to resize heatmaps to match model feature map
        """
        self.dataset = dataset
        self.transform = transform
        # Normalize to list for uniform handling (val set has query + gallery dirs)
        if heatmap_dir is None:
            self.heatmap_dirs = []
        elif isinstance(heatmap_dir, (list, tuple)):
            self.heatmap_dirs = list(heatmap_dir)
        else:
            self.heatmap_dirs = [heatmap_dir]
        self.target_size = target_size  # e.g. (12, 4) for Swin stage3

        # Build filename -> index lookup for coordinates
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

        fname = os.path.basename(img_path)

        # Load keypoint coordinates + scores from NPZ (backup / for Gaussian generation)
        if fname in self.pose_lookup:
            pose_idx = self.pose_lookup[fname]
            keypoints = torch.from_numpy(
                self.pose_keypoints[pose_idx].copy()).float()
            kp_scores = torch.from_numpy(
                self.pose_scores[pose_idx].copy()).float()
        else:
            keypoints = torch.zeros(17, 2)
            kp_scores = torch.zeros(17)

        # Load raw heatmap from .npy file
        heatmap = self._load_heatmap(fname)

        return img, pid, camid, trackid, img_path, keypoints, kp_scores, heatmap

    def _load_heatmap(self, fname):
        """Load raw heatmap .npy file and optionally resize."""
        npy_name = fname.replace('.jpg', '.npy')
        heatmap = None

        # Search through all heatmap directories
        for hdir in self.heatmap_dirs:
            npy_path = os.path.join(hdir, npy_name)
            if os.path.exists(npy_path):
                heatmap = np.load(npy_path)  # (17, 64, 48) float32
                heatmap = torch.from_numpy(heatmap).float()
                break

        if heatmap is None:
            if self.target_size is not None:
                return torch.zeros(17, self.target_size[0], self.target_size[1])
            return torch.zeros(17, 64, 48)

        # Resize if target_size specified
        if self.target_size is not None:
            # (17, H_in, W_in) -> (1, 17, H_out, W_out) -> (17, H_out, W_out)
            heatmap = F.interpolate(
                heatmap.unsqueeze(0),
                size=self.target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        return heatmap


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
