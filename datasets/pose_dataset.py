"""Training-only image dataset that pairs raw RGB with clean pose targets."""

import torch
from torch.utils.data import Dataset

from .bases import read_image


class PoseImageDataset(Dataset):
    def __init__(self, dataset, pose_store, transform, verify_image_sha=False):
        self.dataset = dataset
        self.pose_store = pose_store
        self.transform = transform
        self.verify_image_sha = verify_image_sha

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path, pid, camid, trackid = self.dataset[index]
        image = read_image(image_path)
        pose = self.pose_store.get(
            image_path, verify_image_sha=self.verify_image_sha
        )
        image, pose = self.transform(image, pose)
        return image, pid, camid, trackid, image_path, pose


def pose_train_collate_fn(batch):
    images, pids, camids, viewids, image_paths, poses = zip(*batch)
    pose_batch = {
        "relative_paths": tuple(item.relative_path for item in poses),
        "image_sha256": tuple(item.image_sha256 for item in poses),
        "keypoints": torch.stack([item.keypoints for item in poses], dim=0),
        "scores": torch.stack([item.scores for item in poses], dim=0),
        "valid": torch.stack([item.valid for item in poses], dim=0),
    }
    return (
        torch.stack(images, dim=0),
        torch.tensor(pids, dtype=torch.int64),
        torch.tensor(camids, dtype=torch.int64),
        torch.tensor(viewids, dtype=torch.int64),
        pose_batch,
    )
