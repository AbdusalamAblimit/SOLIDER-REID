"""
Pose-Guided Feature Inpainting (PGFI)

Recovers features in occluded regions using visible region features
and pose heatmap as condition. Applied after PSG+PAA on Stage 3 output.

Architecture:
  1. Generate visibility mask from pose heatmap
  2. Inpaint occluded regions: Conv(feat+heatmap) → inpainted features
  3. Merge: visible regions keep original, occluded regions use inpainted
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseFeatureInpainter(nn.Module):
    """Inpaints occluded features using pose-guided convolution.

    Args:
        feat_channels: Backbone feature dimension (768)
        pose_channels: Heatmap channels (17)
        hidden_dim: Inpainter hidden dimension
    """

    def __init__(self, feat_channels=768, pose_channels=17, hidden_dim=256):
        super().__init__()
        self.feat_channels = feat_channels

        # Inpainter: takes visible features + heatmap, produces inpainted features
        self.inpainter = nn.Sequential(
            nn.Conv2d(feat_channels + pose_channels, hidden_dim, kernel_size=3,
                      padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, feat_channels, kernel_size=3, padding=1, bias=True),
        )

        # Zero-init output for safe identity start
        nn.init.zeros_(self.inpainter[-1].weight)
        nn.init.zeros_(self.inpainter[-1].bias)

    def forward(self, feat_map, scene_heatmaps):
        """
        Args:
            feat_map: (B, C, fH, fW) backbone feature map
            scene_heatmaps: (B, 17, hH, hW) raw pose heatmaps

        Returns:
            feat_inpainted: (B, C, fH, fW) feature map with occluded regions inpainted
        """
        B, C, fH, fW = feat_map.shape

        # Resize heatmaps to feature map size
        if scene_heatmaps.shape[2:] != (fH, fW):
            hm = F.interpolate(scene_heatmaps, size=(fH, fW),
                               mode='bilinear', align_corners=False)
        else:
            hm = scene_heatmaps

        hm_sig = torch.sigmoid(hm)  # (B, 17, fH, fW)

        # Visibility mask: max across keypoints → body presence indicator
        # High where body visible, low where occluded/background
        vis_mask = hm_sig.max(dim=1, keepdim=True)[0]  # (B, 1, fH, fW)

        # Occluded mask: inverse of visibility
        occ_mask = 1.0 - vis_mask  # (B, 1, fH, fW)

        # Visible features (mask out occluded regions before inpainting)
        feat_visible = feat_map * vis_mask  # (B, C, fH, fW)

        # Inpainter input: visible features + heatmap condition
        inp_input = torch.cat([feat_visible, hm_sig], dim=1)  # (B, C+17, fH, fW)
        inpainted = self.inpainter(inp_input)  # (B, C, fH, fW)

        # Merge: keep original in visible regions, use inpainted in occluded
        feat_out = feat_map + occ_mask * inpainted

        return feat_out
