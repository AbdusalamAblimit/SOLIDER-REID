"""End-to-end VPReID pipeline visualization test.

Runs the FULL VPReID model (Swin + ViTPose + PosePartHead) on real
dataloader samples, and saves intermediate results at every stage:

1. Dataloader output (what Swin receives)
2. ViTPose input (after denorm + resize + ImageNet norm)
3. ViTPose heatmaps (raw 64x48)
4. Visibility scores
5. PosePartHead: part attention masks on Swin feature map
6. Part visibility per sample
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

from config import cfg
from datasets import make_dataloader
from model import make_model

SAVE_DIR = 'experiments/vpreid_pipeline_test'
os.makedirs(SAVE_DIR, exist_ok=True)

KP_NAMES = [
    'nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
    'L_shoulder', 'R_shoulder', 'L_hip', 'R_hip',
    'L_elbow', 'R_elbow', 'L_wrist', 'R_wrist',
    'L_knee', 'R_knee', 'L_ankle', 'R_ankle',
]

PART_NAMES = ['Head', 'Torso', 'Arms', 'Thighs', 'Calves']

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def denorm_tensor(tensor, mean, std):
    """[C,H,W] normalized tensor -> [H,W,3] uint8 RGB."""
    img = tensor.clone().cpu().float()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1) * 255
    return img.permute(1, 2, 0).numpy().astype(np.uint8)


def draw_skeleton(img_rgb, heatmaps, visibility, vis_thr=0.3):
    """Draw skeleton on image from heatmap peaks."""
    H, W = img_rgb.shape[:2]
    K = heatmaps.shape[0]
    img = img_rgb.copy()

    kps = []
    for k in range(K):
        hm = heatmaps[k].cpu().numpy()
        hm_r = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
        y, x = np.unravel_index(hm_r.argmax(), hm_r.shape)
        kps.append((x, y))

    for (i, j) in SKELETON:
        if i >= K or j >= K:
            continue
        vi = visibility[i].item()
        vj = visibility[j].item()
        if vi > vis_thr and vj > vis_thr:
            cv2.line(img, kps[i], kps[j], (0, 255, 0), 2)

    for k in range(K):
        color = (0, 255, 0) if visibility[k].item() > vis_thr else (255, 0, 0)
        cv2.circle(img, kps[k], 3, color, -1)

    return img


def visualize_vitpose_internals(pose_predictor, images, sample_idx, save_prefix):
    """Hook into pose_predictor to capture and save intermediate tensors."""
    # Manually replicate the pipeline to capture intermediates
    solider_mean, solider_std = pose_predictor._get_solider_norm(images.device)

    # Stage 1: denorm to [0,255]
    img_01 = images * solider_std + solider_mean
    img_01 = img_01.clamp(0, 1)
    img_255 = img_01 * 255.0

    # Stage 2: resize to ViTPose size
    import torch.nn.functional as F
    img_resized = F.interpolate(
        img_255,
        size=(pose_predictor.pose_input_h, pose_predictor.pose_input_w),
        mode='bilinear', align_corners=False
    )

    # Stage 3: ImageNet norm
    pose_mean = pose_predictor.pose_mean.to(images.device)
    pose_std = pose_predictor.pose_std.to(images.device)
    img_vitpose = (img_resized - pose_mean) / pose_std

    # Stage 4: forward
    with torch.no_grad():
        if hasattr(pose_predictor.model, 'backbone') and hasattr(pose_predictor.model, 'head'):
            feat = pose_predictor.model.backbone(img_vitpose)
            out = pose_predictor.model.head(feat)
        else:
            out = pose_predictor.model(img_vitpose, mode='tensor')

    if isinstance(out, (list, tuple)):
        heatmap = out[0]
        visibility = out[1] if len(out) > 1 else None
    else:
        heatmap = out
        visibility = None

    if visibility is None:
        B, K, h, w = heatmap.shape
        visibility = heatmap.view(B, K, -1).amax(dim=-1)

    # --- Save visualizations for each sample ---
    for b in range(min(images.shape[0], 4)):
        prefix = f'{save_prefix}_s{sample_idx + b}'

        # 1. Original dataloader image (what Swin sees)
        swin_img = denorm_tensor(images[b], cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)

        # 2. Denormalized image [0,255]
        denorm_img = img_255[b].permute(1, 2, 0).cpu().clamp(0, 255).numpy().astype(np.uint8)

        # 3. Resized for ViTPose [0,255]
        resized_img = img_resized[b].permute(1, 2, 0).cpu().clamp(0, 255).numpy().astype(np.uint8)

        # 4. ViTPose normalized (denorm back for viz)
        vitpose_denorm = (img_vitpose[b] * pose_std[0] + pose_mean[0]).clamp(0, 255)
        vitpose_viz = vitpose_denorm.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # --- Figure 1: Pipeline stages ---
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        axes[0].imshow(swin_img)
        axes[0].set_title(f'1. Swin input\n{swin_img.shape[1]}x{swin_img.shape[0]}\nmean={cfg.INPUT.PIXEL_MEAN}')
        axes[1].imshow(denorm_img)
        axes[1].set_title(f'2. Denorm [0,255]\n{denorm_img.shape[1]}x{denorm_img.shape[0]}')
        axes[2].imshow(resized_img)
        axes[2].set_title(f'3. Resized for ViTPose\n{resized_img.shape[1]}x{resized_img.shape[0]}')
        axes[3].imshow(vitpose_viz)
        axes[3].set_title(f'4. ViTPose input\n(ImageNet norm, shown denormed)')
        for ax in axes:
            ax.axis('off')
        plt.suptitle(f'ViTPose Input Pipeline — Sample {sample_idx + b}', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f'{prefix}_1_pipeline.png'), dpi=120, bbox_inches='tight')
        plt.close()

        # --- Figure 2: Heatmaps ---
        hm_b = heatmap[b].cpu()  # [17, h, w]
        vis_b = visibility[b].cpu()  # [17]
        H_img, W_img = swin_img.shape[:2]

        fig, axes = plt.subplots(3, 6, figsize=(18, 10))
        axes_flat = axes.flatten()
        axes_flat[0].imshow(swin_img)
        axes_flat[0].set_title('Original', fontsize=9)
        axes_flat[0].axis('off')
        for k in range(17):
            ax = axes_flat[k + 1]
            hm_k = hm_b[k].numpy()
            hm_r = cv2.resize(hm_k, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
            ax.imshow(swin_img)
            ax.imshow(hm_r, cmap='jet', alpha=0.5, vmin=0, vmax=max(hm_r.max(), 0.01))
            ax.set_title(f'{KP_NAMES[k]}\nv={vis_b[k]:.2f}', fontsize=7)
            ax.axis('off')
        plt.suptitle(f'ViTPose Heatmaps (64x48) — Sample {sample_idx + b}', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f'{prefix}_2_heatmaps.png'), dpi=120, bbox_inches='tight')
        plt.close()

        # --- Figure 3: Skeleton ---
        skel_img = draw_skeleton(swin_img, hm_b, vis_b)
        cv2.imwrite(
            os.path.join(SAVE_DIR, f'{prefix}_3_skeleton.png'),
            cv2.cvtColor(skel_img, cv2.COLOR_RGB2BGR)
        )

    return heatmap, visibility


def visualize_part_head(feat_map, heatmaps, visibility, part_head, images, sample_idx, save_prefix):
    """Visualize PosePartHead intermediate: part masks and attention."""
    import torch.nn.functional as Fn

    B, C, H, W = feat_map.shape
    K = part_head.n_parts

    # Replicate PosePartHead logic to capture intermediates
    hm = Fn.interpolate(heatmaps.float(), (H, W), mode='bilinear', align_corners=False)
    vis = visibility.float().unsqueeze(-1).unsqueeze(-1)
    hm_masked = hm * vis

    part_masks = []
    part_vis_list = []
    for group in part_head.part_groups:
        pmask = hm_masked[:, group].max(dim=1)[0]
        part_masks.append(pmask)
        pvis = visibility[:, group].float().max(dim=1)[0]
        part_vis_list.append(pvis)

    part_masks_t = torch.stack(part_masks, dim=1)  # [B, K, H, W]
    part_vis_t = torch.stack(part_vis_list, dim=1)  # [B, K]

    # Softmax attention
    flat = part_masks_t.view(B, K, -1)
    flat_clamped = (flat / part_head.temp).clamp(-20.0, 20.0)
    attn = torch.softmax(flat_clamped.float(), dim=-1).view(B, K, H, W)

    # Visualize for each sample
    for b in range(min(B, 4)):
        prefix = f'{save_prefix}_s{sample_idx + b}'
        swin_img = denorm_tensor(images[b], cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        H_img, W_img = swin_img.shape[:2]

        # --- Figure 4: Part masks (raw) + attention (after softmax) ---
        fig, axes = plt.subplots(2, K + 1, figsize=(3 * (K + 1), 7))

        # Row 1: raw part masks
        axes[0, 0].imshow(swin_img)
        axes[0, 0].set_title('Original', fontsize=9)
        axes[0, 0].axis('off')
        for k in range(K):
            pm = part_masks_t[b, k].cpu().numpy()
            pm_r = cv2.resize(pm, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
            axes[0, k + 1].imshow(swin_img)
            axes[0, k + 1].imshow(pm_r, cmap='hot', alpha=0.6, vmin=0, vmax=max(pm_r.max(), 0.01))
            axes[0, k + 1].set_title(f'{PART_NAMES[k]}\nvis={part_vis_t[b, k]:.2f}', fontsize=8)
            axes[0, k + 1].axis('off')

        # Row 2: softmax attention
        axes[1, 0].imshow(swin_img)
        axes[1, 0].set_title('Original', fontsize=9)
        axes[1, 0].axis('off')
        for k in range(K):
            att = attn[b, k].cpu().numpy()
            att_r = cv2.resize(att, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
            axes[1, k + 1].imshow(swin_img)
            axes[1, k + 1].imshow(att_r, cmap='hot', alpha=0.6)
            axes[1, k + 1].set_title(f'Attn: {PART_NAMES[k]}', fontsize=8)
            axes[1, k + 1].axis('off')

        axes[0, 0].set_ylabel('Raw Part Mask', fontsize=10)
        axes[1, 0].set_ylabel('Softmax Attention', fontsize=10)
        plt.suptitle(f'PosePartHead — feat_map {H}x{W}, temp={part_head.temp} — Sample {sample_idx + b}', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f'{prefix}_4_part_attn.png'), dpi=120, bbox_inches='tight')
        plt.close()

        # --- Figure 5: Visibility bar chart ---
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        vis_vals = part_vis_t[b].cpu().numpy()
        colors = ['green' if v > 0.5 else 'red' for v in vis_vals]
        ax.bar(PART_NAMES, vis_vals, color=colors)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Visibility')
        ax.set_title(f'Part Visibility — Sample {sample_idx + b}')
        for i, v in enumerate(vis_vals):
            ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f'{prefix}_5_part_vis.png'), dpi=100, bbox_inches='tight')
        plt.close()


def main():
    print("=" * 60)
    print("VPReID End-to-End Pipeline Test")
    print("=" * 60)

    # Load config
    cfg.merge_from_file('configs/occluded_duke/vpreid_tiny.yml')
    cfg.merge_from_list([
        'DATASETS.ROOT_DIR', 'data',
        'MODEL.WITH_CP', 'True',
        'SOLVER.IMS_PER_BATCH', '16',
        'DATALOADER.NUM_WORKERS', '4',
        'TEST.IMS_PER_BATCH', '16',
    ])
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # Build dataloader
    print("\n[1/4] Building dataloader...")
    train_loader, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    print(f"  Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # Build model
    print("\n[2/4] Building VPReID model...")
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model = model.cuda().eval()

    # Verify it's VPReID
    assert hasattr(model, 'is_vpreid') and model.is_vpreid, "Model is not VPReID!"
    vpreid_backbone = model.base
    assert vpreid_backbone.pose_predictor is not None, "Pose predictor not loaded!"
    print(f"  VPReID backbone: {type(vpreid_backbone).__name__}")
    print(f"  Pose predictor: loaded ({sum(p.numel() for p in vpreid_backbone.pose_predictor.parameters())/1e6:.1f}M params)")
    print(f"  N_parts: {vpreid_backbone.n_parts}")

    # --- Test on TRAIN samples ---
    print("\n[3/4] Testing on train samples...")
    train_iter = iter(train_loader)
    img, vid, target_cam, target_view = next(train_iter)
    img = img.cuda()

    print(f"  Batch shape: {img.shape}")
    print(f"  Tensor range: [{img.min():.3f}, {img.max():.3f}]")

    # Full VPReID forward
    with torch.no_grad():
        outputs = vpreid_backbone(img)

    print(f"  global_feat: {outputs['global_feat'].shape}")
    print(f"  part_feats:  {outputs['part_feats'].shape}")
    print(f"  part_vis:    {outputs['part_vis'].shape}")
    print(f"  fg_feat:     {outputs['foreground_feat'].shape}")

    # Visualize ViTPose internals
    print("\n  Visualizing ViTPose pipeline...")
    heatmaps, visibility = visualize_vitpose_internals(
        vpreid_backbone.pose_predictor, img, 0, 'train'
    )
    print(f"  Heatmaps: {heatmaps.shape}, range [{heatmaps.min():.4f}, {heatmaps.max():.4f}]")
    print(f"  Visibility stats: min={visibility.min():.3f}, max={visibility.max():.3f}, mean={visibility.mean():.3f}")

    # Visualize PosePartHead
    print("\n  Visualizing PosePartHead...")
    _, outs = vpreid_backbone.base(img)
    feat_map = outs[-1]
    print(f"  Swin feat_map: {feat_map.shape}")
    visualize_part_head(feat_map, heatmaps, visibility, vpreid_backbone.part_head, img, 0, 'train')

    # --- Test on VAL samples ---
    print("\n[4/4] Testing on val samples...")
    val_iter = iter(val_loader)
    img_v, vid_v, camid_v, camids_v, target_view_v, _ = next(val_iter)
    img_v = img_v.cuda()

    print(f"  Batch shape: {img_v.shape}")

    with torch.no_grad():
        # Eval mode returns (feat_dict, None)
        feat_dict, _ = model(img_v, cam_label=camids_v.cuda(), view_label=target_view_v.cuda())

    if isinstance(feat_dict, dict):
        print(f"  Eval output keys: {list(feat_dict.keys())}")
        print(f"  global: {feat_dict['global'].shape}")
        print(f"  parts:  {feat_dict['parts'].shape}")
        print(f"  part_vis: {feat_dict['part_vis'].shape}")
    else:
        print(f"  Eval output: tensor {feat_dict.shape}")

    # Visualize val ViTPose
    heatmaps_v, visibility_v = visualize_vitpose_internals(
        vpreid_backbone.pose_predictor, img_v, 0, 'val'
    )

    _, outs_v = vpreid_backbone.base(img_v)
    feat_map_v = outs_v[-1]
    visualize_part_head(feat_map_v, heatmaps_v, visibility_v, vpreid_backbone.part_head, img_v, 0, 'val')

    print(f"\n{'=' * 60}")
    print(f"All visualizations saved to: {SAVE_DIR}/")
    print(f"{'=' * 60}")

    # Print summary of files
    files = sorted(os.listdir(SAVE_DIR))
    for f in files:
        sz = os.path.getsize(os.path.join(SAVE_DIR, f))
        print(f"  {f} ({sz // 1024}KB)")


if __name__ == '__main__':
    main()
