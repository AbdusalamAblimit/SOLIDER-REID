"""Dataset with offline pose data and RE-aware visibility updates.

Loads pre-extracted keypoints and visibility from .npz files. During training,
tracks random erasing regions and updates keypoint visibility accordingly.
"""

import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from timm.data.random_erasing import RandomErasing

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    got_img = False
    if not osp.exists(img_path):
        raise IOError(f"{img_path} does not exist")
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            pass
    return img


class RandomErasingWithRegion(RandomErasing):
    """Wrapper around timm's RandomErasing that records the erased region.

    After calling, check self.last_erase_region for (top, left, h, w) or None.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_erase_region = None

    def __call__(self, input):
        self.last_erase_region = None
        if len(input.size()) == 3:
            self._erase_tracked(input, *input.size(), input.dtype)
        return input

    def _erase_tracked(self, img, chan, img_h, img_w, dtype):
        """Same as timm's _erase but records the erased region."""
        import random
        import math
        from timm.data.random_erasing import _get_pixels

        if random.random() > self.probability:
            return

        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel,
                        self.rand_color,
                        (chan, h, w),
                        dtype=dtype,
                        device=self.device,
                    )
                    self.last_erase_region = (top, left, h, w)
                    break


class PoseImageDataset(Dataset):
    """Image dataset with offline pose keypoints and RE-aware visibility.

    Args:
        dataset: list of (img_path, pid, camid, trackid) tuples
        transform: torchvision transforms (WITHOUT RandomErasing)
        pose_data: dict with 'filenames', 'keypoints', 'visibility' arrays
        re_prob: random erasing probability (0 = no RE)
        img_size: (H, W) of the resized image
    """

    def __init__(self, dataset, transform, pose_data, re_prob=0.5, img_size=(384, 128)):
        self.dataset = dataset
        self.transform = transform
        self.img_h, self.img_w = img_size

        # Build filename -> index mapping for pose data
        self.pose_kpts = pose_data['keypoints']   # [N, 17, 2]
        self.pose_vis = pose_data['visibility']    # [N, 17]
        filenames = pose_data['filenames']

        self.fname2idx = {}
        for i, fn in enumerate(filenames):
            fn_str = fn if isinstance(fn, str) else fn.decode('utf-8') if isinstance(fn, bytes) else str(fn)
            self.fname2idx[fn_str] = i

        # Random erasing with region tracking
        if re_prob > 0:
            self.random_erasing = RandomErasingWithRegion(
                probability=re_prob, mode='pixel', max_count=1, device='cpu'
            )
        else:
            self.random_erasing = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        # Apply transform (without RE)
        if self.transform is not None:
            img = self.transform(img)

        # Get offline pose data
        fname = osp.basename(img_path)
        pose_idx = self.fname2idx.get(fname, None)

        if pose_idx is not None:
            kpts = self.pose_kpts[pose_idx].copy()    # [17, 2] int16
            vis = self.pose_vis[pose_idx].copy()       # [17] float32
        else:
            # Fallback: no pose data available
            kpts = np.zeros((17, 2), dtype=np.int16)
            vis = np.zeros(17, dtype=np.float32)

        # Apply random erasing and update visibility
        if self.random_erasing is not None:
            img = self.random_erasing(img)

            if self.random_erasing.last_erase_region is not None:
                top, left, h, w = self.random_erasing.last_erase_region
                # Check which keypoints fall in the erased region
                # kpts are in image space (x=0..127, y=0..383 for 128x384 image)
                kp_x = kpts[:, 0]  # [17]
                kp_y = kpts[:, 1]  # [17]

                in_erase = (
                    (kp_x >= left) & (kp_x < left + w) &
                    (kp_y >= top) & (kp_y < top + h)
                )
                vis[in_erase] = 0.0

        # Convert to tensors
        kpts_tensor = torch.from_numpy(kpts).short()     # [17, 2]
        vis_tensor = torch.from_numpy(vis).float()        # [17]

        return img, pid, camid, trackid, img_path, kpts_tensor, vis_tensor


class PoseValDataset(Dataset):
    """Validation dataset with offline pose (no RE)."""

    def __init__(self, dataset, transform, pose_data):
        self.dataset = dataset
        self.transform = transform

        self.pose_kpts = pose_data['keypoints']
        self.pose_vis = pose_data['visibility']
        filenames = pose_data['filenames']

        self.fname2idx = {}
        for i, fn in enumerate(filenames):
            fn_str = fn if isinstance(fn, str) else fn.decode('utf-8') if isinstance(fn, bytes) else str(fn)
            self.fname2idx[fn_str] = i

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        fname = osp.basename(img_path)
        pose_idx = self.fname2idx.get(fname, None)

        if pose_idx is not None:
            kpts = self.pose_kpts[pose_idx]
            vis = self.pose_vis[pose_idx]
        else:
            kpts = np.zeros((17, 2), dtype=np.int16)
            vis = np.zeros(17, dtype=np.float32)

        kpts_tensor = torch.from_numpy(kpts.copy()).short()
        vis_tensor = torch.from_numpy(vis.copy()).float()

        return img, pid, camid, trackid, img_path, kpts_tensor, vis_tensor
