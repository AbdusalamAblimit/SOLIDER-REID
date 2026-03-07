"""Pose utility functions for generating heatmaps from keypoints."""
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


def generate_gaussian_heatmap(keypoints, scores, H, W, sigma=2.0, threshold=0.3):
    """Generate Gaussian heatmaps from keypoint coordinates.

    Args:
        keypoints: (B, 17, 2) normalized coordinates in [0, 1]
        scores: (B, 17) confidence scores
        H, W: spatial dimensions of output heatmap
        sigma: Gaussian std in pixel space
        threshold: minimum score to include a keypoint

    Returns:
        heatmaps: (B, 17, H, W) Gaussian heatmaps
    """
    B, K, _ = keypoints.shape
    device = keypoints.device

    # Create coordinate grids
    yy = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)

    # Convert normalized coords to pixel coords
    cx = keypoints[:, :, 0:1].unsqueeze(-1) * W  # (B, K, 1, 1)
    cy = keypoints[:, :, 1:2].unsqueeze(-1) * H  # (B, K, 1, 1)

    # Gaussian: exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
    heatmaps = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

    # Zero out low-confidence keypoints
    mask = (scores > threshold).float().unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
    heatmaps = heatmaps * mask

    return heatmaps


def generate_part_heatmaps(keypoints, scores, H, W, sigma=2.0, threshold=0.3):
    """Generate body part heatmaps by grouping keypoints.

    Args:
        keypoints: (B, 17, 2) normalized coordinates
        scores: (B, 17) confidence scores
        H, W: spatial dimensions
        sigma: Gaussian std
        threshold: minimum score threshold

    Returns:
        part_heatmaps: (B, NUM_PARTS, H, W) part attention maps
        part_valid: (B, NUM_PARTS) binary mask for valid parts
    """
    # Generate all 17 keypoint heatmaps
    all_heatmaps = generate_gaussian_heatmap(keypoints, scores, H, W, sigma, threshold)

    B = keypoints.shape[0]
    device = keypoints.device
    part_heatmaps = torch.zeros(B, NUM_PARTS, H, W, device=device)
    part_valid = torch.zeros(B, NUM_PARTS, device=device)

    for i, (name, indices) in enumerate(PART_GROUPS.items()):
        # Max over keypoints in this group
        group_heatmaps = all_heatmaps[:, indices]  # (B, len(indices), H, W)
        part_heatmaps[:, i] = group_heatmaps.max(dim=1)[0]  # (B, H, W)
        # Part is valid if any keypoint in the group has high score
        group_scores = scores[:, indices]  # (B, len(indices))
        part_valid[:, i] = (group_scores > threshold).any(dim=1).float()

    return part_heatmaps, part_valid
