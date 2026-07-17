"""Official-SOLIDER-compatible RGB augmentation paired with COCO-17 pose."""

from dataclasses import dataclass

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from timm.data.random_erasing import RandomErasing

from .pose_targets import PoseTarget


COCO17_FLIP_INDEX = torch.tensor(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    dtype=torch.long,
)


@dataclass
class AugmentedPoseTarget:
    relative_path: str
    image_sha256: str
    image_size: tuple
    keypoints: torch.Tensor
    scores: torch.Tensor
    valid: torch.Tensor
    flipped: bool
    crop_offset: tuple


def _padding_left_top(padding):
    if isinstance(padding, int):
        return padding, padding
    padding = tuple(padding)
    if len(padding) == 1:
        return padding[0], padding[0]
    if len(padding) == 2:
        return padding[0], padding[1]
    if len(padding) == 4:
        return padding[0], padding[1]
    raise ValueError("Padding must have length 1, 2, or 4")


class PairedPoseTransform:
    """Apply one sampled geometry to RGB and pose, then erase RGB only."""

    def __init__(
        self,
        size_train,
        flip_probability,
        padding,
        pixel_mean,
        pixel_std,
        erasing_probability,
    ):
        self.size_train = tuple(size_train)
        self.flip_probability = float(flip_probability)
        self.erasing_probability = float(erasing_probability)
        if len(self.size_train) != 2 or min(self.size_train) <= 0:
            raise ValueError("size_train must contain two positive values")
        if not 0.0 <= self.flip_probability <= 1.0:
            raise ValueError("flip_probability must be in [0, 1]")
        if not 0.0 <= self.erasing_probability <= 1.0:
            raise ValueError("erasing_probability must be in [0, 1]")
        self.padding = padding
        _padding_left_top(padding)
        self.resize = T.Resize(
            self.size_train, interpolation=T.InterpolationMode.BICUBIC
        )
        self.pad = T.Pad(padding)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=pixel_mean, std=pixel_std)
        self.random_erasing = RandomErasing(
            probability=self.erasing_probability,
            mode="pixel",
            max_count=1,
            device="cpu",
        )

    def __call__(self, image, pose=None):
        keypoints = scores = valid = None
        if pose is not None:
            if not isinstance(pose, PoseTarget):
                raise TypeError("pose must be PoseTarget or None")
            if image.size != pose.image_size:
                raise RuntimeError("RGB and pose target size mismatch")
            keypoints = pose.keypoints.clone()
            scores = pose.scores.clone()
            valid = pose.valid.clone()

        original_width, original_height = image.size
        image = self.resize(image)
        resized_width, resized_height = image.size
        if keypoints is not None:
            keypoints[:, 0] *= resized_width / float(original_width)
            keypoints[:, 1] *= resized_height / float(original_height)

        flipped = bool(torch.rand(1) < self.flip_probability)
        if flipped:
            image = F.hflip(image)
            if keypoints is not None:
                keypoints[:, 0] = resized_width - 1 - keypoints[:, 0]
                keypoints = keypoints[COCO17_FLIP_INDEX]
                scores = scores[COCO17_FLIP_INDEX]
                valid = valid[COCO17_FLIP_INDEX]

        image = self.pad(image)
        pad_left, pad_top = _padding_left_top(self.padding)
        if keypoints is not None:
            keypoints[:, 0] += pad_left
            keypoints[:, 1] += pad_top

        crop_top, crop_left, crop_height, crop_width = T.RandomCrop.get_params(
            image, output_size=self.size_train
        )
        image = F.crop(image, crop_top, crop_left, crop_height, crop_width)
        if keypoints is not None:
            keypoints[:, 0] -= crop_left
            keypoints[:, 1] -= crop_top
            valid = (
                valid
                & (keypoints[:, 0] >= 0)
                & (keypoints[:, 0] <= crop_width - 1)
                & (keypoints[:, 1] >= 0)
                & (keypoints[:, 1] <= crop_height - 1)
            )

        image = self.to_tensor(image)
        image = self.normalize(image)
        image = self.random_erasing(image)

        if keypoints is None:
            return image, None
        return image, AugmentedPoseTarget(
            relative_path=pose.relative_path,
            image_sha256=pose.image_sha256,
            image_size=(crop_width, crop_height),
            keypoints=keypoints,
            scores=scores,
            valid=valid,
            flipped=flipped,
            crop_offset=(crop_left, crop_top),
        )
