"""Pose utility functions for heatmap processing.

Provides body part grouping and multi-person heatmap merging.
"""
import torch
import torch.nn.functional as F


# COCO 17-keypoint body part groups
PART_GROUPS = {
    'head': [0, 1, 2, 3, 4],           # nose, eyes, ears
    'upper_torso': [5, 6],              # shoulders
    'arms': [7, 8, 9, 10],             # elbows, wrists
    'lower_torso': [11, 12],            # hips
    'legs': [13, 14, 15, 16],          # knees, ankles
}
PART_NAMES = list(PART_GROUPS.keys())
NUM_PARTS = len(PART_NAMES)


def merge_person_heatmaps(heatmaps, person_mask):
    """Merge multi-person heatmaps into a single scene-level heatmap.

    Takes per-person heatmaps and combines them via element-wise max,
    producing one 17-channel heatmap that covers all detected persons.

    Args:
        heatmaps: (B, MAX_PERSONS, 17, H, W) per-person heatmaps
        person_mask: (B, MAX_PERSONS) binary mask for valid persons

    Returns:
        scene_heatmap: (B, 17, H, W) merged scene-level heatmap
    """
    # Mask out invalid persons before merging
    mask = person_mask[:, :, None, None, None]  # (B, MAX, 1, 1, 1)
    masked = heatmaps * mask
    # Max over persons dimension
    scene_heatmap = masked.max(dim=1)[0]  # (B, 17, H, W)
    return scene_heatmap


def heatmaps_to_parts(heatmaps_17, scores_17=None, threshold=0.0):
    """Group 17 keypoint heatmaps into 5 body part heatmaps.

    Args:
        heatmaps_17: (B, 17, H, W) keypoint-level heatmaps
        scores_17: (B, 17) optional keypoint scores; if provided,
                   parts with all scores below threshold are marked invalid
        threshold: minimum score for a keypoint to be considered valid

    Returns:
        part_heatmaps: (B, NUM_PARTS, H, W) body part heatmaps
        part_valid: (B, NUM_PARTS) binary validity mask
    """
    B = heatmaps_17.shape[0]
    device = heatmaps_17.device

    parts = []
    valid = []

    for name, indices in PART_GROUPS.items():
        # Max over keypoints in this body part
        group = heatmaps_17[:, indices]  # (B, len(indices), H, W)
        part_hm = group.max(dim=1)[0]  # (B, H, W)
        parts.append(part_hm)

        if scores_17 is not None:
            group_scores = scores_17[:, indices]  # (B, len(indices))
            v = (group_scores > threshold).any(dim=1).float()  # (B,)
        else:
            # Valid if heatmap has any non-zero response
            v = (part_hm.sum(dim=(1, 2)) > 0).float()  # (B,)
        valid.append(v)

    part_heatmaps = torch.stack(parts, dim=1)  # (B, NUM_PARTS, H, W)
    part_valid = torch.stack(valid, dim=1)  # (B, NUM_PARTS)
    return part_heatmaps, part_valid
