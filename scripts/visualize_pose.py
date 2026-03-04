"""Visualize ViTPose outputs on sample ReID images.

Checks:
1. Does the de-normalization pipeline correctly recover pixel values?
2. Does ViTPose produce meaningful heatmaps on 384x128 ReID images?
3. Are keypoints placed at correct body locations?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
from config import cfg

# COCO keypoint names
KP_NAMES = [
    'nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
    'L_shoulder', 'R_shoulder', 'L_hip', 'R_hip',
    'L_elbow', 'R_elbow', 'L_wrist', 'R_wrist',
    'L_knee', 'R_knee', 'L_ankle', 'R_ankle',
]

# COCO skeleton (pairs of keypoint indices)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # head
    (5, 6), (5, 7), (6, 8),                # upper body
    (7, 9), (8, 10), (9, 11), (10, 12),    # arms -> these are wrong for COCO
    (5, 11), (6, 12),                       # torso
    (11, 13), (12, 14), (13, 15), (14, 16) # legs
]

# Fix: correct COCO skeleton
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                  # shoulders
    (5, 7), (7, 9),                          # left arm
    (6, 8), (8, 10),                         # right arm
    (5, 11), (6, 12),                        # torso
    (11, 12),                                # hips
    (11, 13), (13, 15),                      # left leg
    (12, 14), (14, 16),                      # right leg
]


def load_config():
    """Load VPReID config."""
    cfg.merge_from_file('configs/occluded_duke/vpreid_tiny.yml')
    cfg.freeze()
    return cfg


def get_sample_images(cfg, n=6):
    """Get sample images from the dataset directory."""
    train_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'occluded_duke', 'bounding_box_train')
    img_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
    # Pick evenly spaced samples
    step = max(1, len(img_files) // n)
    selected = [img_files[i * step] for i in range(n)]
    return [os.path.join(train_dir, f) for f in selected]


def build_reid_transform(cfg):
    """Build the same transform as ReID dataloader (test mode, no augmentation)."""
    return T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])


def denorm_to_uint8(tensor, mean, std):
    """De-normalize a tensor image back to uint8 [H, W, 3]."""
    img = tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1) * 255
    img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return img


def visualize_heatmaps(img_rgb, heatmaps, visibility, kp_names, save_path):
    """Visualize all 17 keypoint heatmaps overlaid on the image."""
    K = heatmaps.shape[0]
    cols = 6
    rows = (K + 1 + cols - 1) // cols  # +1 for the original image

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 4 * rows))
    axes = axes.flatten()

    # Original image
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original', fontsize=9)
    axes[0].axis('off')

    H, W = img_rgb.shape[:2]

    for k in range(K):
        ax = axes[k + 1]
        hm = heatmaps[k].cpu().numpy()  # [h, w]
        # Resize heatmap to image size
        hm_resized = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
        ax.imshow(img_rgb)
        ax.imshow(hm_resized, cmap='jet', alpha=0.5, vmin=0, vmax=hm_resized.max() + 1e-8)
        vis_val = visibility[k].item() if visibility is not None else -1
        ax.set_title(f'{kp_names[k]}\nvis={vis_val:.2f}', fontsize=8)
        ax.axis('off')

    # Hide extra axes
    for i in range(K + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved heatmap grid: {save_path}')


def visualize_skeleton(img_rgb, heatmaps, visibility, save_path, vis_threshold=0.3):
    """Draw detected skeleton on the image."""
    H, W = img_rgb.shape[:2]
    K = heatmaps.shape[0]

    # Get keypoint coordinates from heatmap peaks
    kp_coords = []
    for k in range(K):
        hm = heatmaps[k].cpu().numpy()
        hm_resized = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
        y, x = np.unravel_index(hm_resized.argmax(), hm_resized.shape)
        kp_coords.append((x, y, hm_resized.max()))

    img_draw = img_rgb.copy()

    # Draw skeleton lines
    for (i, j) in SKELETON:
        if i >= K or j >= K:
            continue
        vi = visibility[i].item() if visibility is not None else 1.0
        vj = visibility[j].item() if visibility is not None else 1.0
        if vi > vis_threshold and vj > vis_threshold:
            x1, y1, _ = kp_coords[i]
            x2, y2, _ = kp_coords[j]
            cv2.line(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw keypoints
    for k in range(K):
        x, y, conf = kp_coords[k]
        vis_val = visibility[k].item() if visibility is not None else 1.0
        if vis_val > vis_threshold:
            color = (0, 255, 0)  # visible: green
        else:
            color = (255, 0, 0)  # occluded: red
        cv2.circle(img_draw, (x, y), 3, color, -1)
        cv2.putText(img_draw, f'{k}', (x + 4, y - 4),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Save
    img_bgr = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)
    print(f'  Saved skeleton: {save_path}')


def visualize_preprocessing(img_path, cfg, save_dir):
    """Show preprocessing pipeline step by step."""
    # Step 1: Original image
    pil_img = Image.open(img_path).convert('RGB')
    orig_np = np.array(pil_img)

    # Step 2: After ReID transform (what Swin sees)
    transform = build_reid_transform(cfg)
    reid_tensor = transform(pil_img)  # [3, H, W], normalized

    reid_np = denorm_to_uint8(reid_tensor, cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)

    # Step 3: What ViTPose receives (de-normed from ReID, then re-normed for ViTPose)
    # Replicate pose_predictor logic
    solider_mean = torch.tensor(cfg.INPUT.PIXEL_MEAN).view(3, 1, 1)
    solider_std = torch.tensor(cfg.INPUT.PIXEL_STD).view(3, 1, 1)
    vitpose_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    vitpose_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

    # De-norm from SOLIDER space to [0,1]
    img_01 = reid_tensor * solider_std + solider_mean
    # To [0,255]
    img_255 = img_01 * 255.0
    # Apply ViTPose norm
    img_vitpose = (img_255 - vitpose_mean) / vitpose_std

    # De-norm ViTPose back to [0,255] for visualization (verify round-trip)
    img_vitpose_denorm = (img_vitpose * vitpose_std + vitpose_mean).clamp(0, 255)
    vitpose_np = img_vitpose_denorm.permute(1, 2, 0).numpy().astype(np.uint8)

    # Step 4: Show comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))

    axes[0].imshow(orig_np)
    axes[0].set_title(f'Original\n{orig_np.shape[1]}x{orig_np.shape[0]}', fontsize=10)

    axes[1].imshow(reid_np)
    axes[1].set_title(f'ReID input\n{reid_np.shape[1]}x{reid_np.shape[0]}\nmean={cfg.INPUT.PIXEL_MEAN}', fontsize=9)

    axes[2].imshow(vitpose_np)
    axes[2].set_title(f'ViTPose input (denormed)\n{vitpose_np.shape[1]}x{vitpose_np.shape[0]}\nImageNet norm', fontsize=9)

    # Show pixel value distributions
    axes[3].hist(reid_tensor.flatten().numpy(), bins=50, alpha=0.5, label='ReID tensor', color='blue')
    axes[3].hist(img_vitpose.flatten().numpy(), bins=50, alpha=0.5, label='ViTPose tensor', color='red')
    axes[3].set_title('Tensor value distribution', fontsize=10)
    axes[3].legend(fontsize=8)
    axes[3].set_xlabel('value')

    for ax in axes[:3]:
        ax.axis('off')

    basename = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(save_dir, f'preprocess_{basename}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved preprocessing comparison: {save_path}')

    return reid_tensor, reid_np


def main():
    print("=" * 60)
    print("ViTPose Visualization on ReID Images")
    print("=" * 60)

    cfg = load_config()
    save_dir = 'experiments/pose_visualization'
    os.makedirs(save_dir, exist_ok=True)

    # Get sample images
    img_paths = get_sample_images(cfg, n=6)
    print(f"\nSelected {len(img_paths)} sample images")

    # Build ViTPose predictor
    print("\nLoading ViTPose predictor...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from model.backbones.pose_predictor import MMPoseTopDownPredictor
    predictor = MMPoseTopDownPredictor(
        cfg.MODEL.VPREID.POSE_CFG,
        cfg.MODEL.VPREID.POSE_CKPT,
        device
    )
    predictor.eval()
    print("ViTPose loaded.")

    # Config info
    print(f"\n--- Config ---")
    print(f"ReID input size: {cfg.INPUT.SIZE_TEST}")
    print(f"ReID PIXEL_MEAN: {cfg.INPUT.PIXEL_MEAN}")
    print(f"ReID PIXEL_STD:  {cfg.INPUT.PIXEL_STD}")
    print(f"ViTPose expected input: 256x192 (from codec)")
    print(f"ViTPose mean: [123.675, 116.28, 103.53]")
    print(f"ViTPose std:  [58.395, 57.12, 57.375]")
    print()

    transform = build_reid_transform(cfg)

    for i, img_path in enumerate(img_paths):
        print(f"\n[{i+1}/{len(img_paths)}] {os.path.basename(img_path)}")

        # Preprocessing visualization
        reid_tensor, reid_np = visualize_preprocessing(img_path, cfg, save_dir)

        # Run ViTPose
        batch = reid_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            heatmaps, visibility = predictor(batch)

        heatmaps = heatmaps[0]    # [17, h, w]
        visibility = visibility[0]  # [17]

        print(f'  Heatmap shape: {heatmaps.shape}')
        print(f'  Heatmap range: [{heatmaps.min():.4f}, {heatmaps.max():.4f}]')
        print(f'  Visibility: {[f"{v:.2f}" for v in visibility.tolist()]}')

        basename = os.path.splitext(os.path.basename(img_path))[0]

        # Heatmap grid
        visualize_heatmaps(
            reid_np, heatmaps, visibility, KP_NAMES,
            os.path.join(save_dir, f'heatmaps_{basename}.png')
        )

        # Skeleton overlay
        visualize_skeleton(
            reid_np, heatmaps, visibility,
            os.path.join(save_dir, f'skeleton_{basename}.png'),
            vis_threshold=0.3
        )

    # Summary figure: all skeletons side by side
    print(f"\nAll results saved to: {save_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
