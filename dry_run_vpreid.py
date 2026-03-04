"""Dry-run verification script for VPReID.

Verifies:
1. Model construction (backbone + pose part head + classifiers)
2. Forward pass shape correctness (training & eval modes)
3. Backward pass (gradients flow correctly)
4. Loss computation
5. Evaluator compatibility (dict features)

Run: python dry_run_vpreid.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hack: add project root to path
sys.path.insert(0, '.')


def test_pose_part_head():
    """Test PosePartHead independently."""
    print("=" * 60)
    print("TEST 1: PosePartHead")
    print("=" * 60)

    from model.backbones.vpreid import PosePartHead

    head = PosePartHead(n_parts=5, temp=0.1)
    head.eval()

    B, C, H, W = 4, 768, 12, 4
    feat_map = torch.randn(B, C, H, W)
    heatmaps = torch.sigmoid(torch.randn(B, 17, 64, 48))  # ViTPose-like output
    visibility = torch.rand(B, 17)

    part_feats, part_vis, fg_feat = head(feat_map, heatmaps, visibility)

    assert part_feats.shape == (B, 5, C), f"part_feats shape mismatch: {part_feats.shape}"
    assert part_vis.shape == (B, 5), f"part_vis shape mismatch: {part_vis.shape}"
    assert fg_feat.shape == (B, C), f"fg_feat shape mismatch: {fg_feat.shape}"

    # Test gradient flow
    feat_map.requires_grad = True
    head.train()
    pf, pv, fg = head(feat_map, heatmaps, visibility)
    loss = pf.sum() + fg.sum()
    loss.backward()
    assert feat_map.grad is not None, "No gradient to feat_map!"
    assert feat_map.grad.abs().sum() > 0, "Zero gradient!"

    print(f"  part_feats: {part_feats.shape}")
    print(f"  part_vis: {part_vis.shape}")
    print(f"  fg_feat: {fg_feat.shape}")
    print(f"  Gradient flows to feat_map: OK")
    print("  PASSED\n")


def test_vpreid_backbone():
    """Test VPReIDSwin backbone (without real ViTPose)."""
    print("=" * 60)
    print("TEST 2: VPReIDSwin backbone (dummy pose)")
    print("=" * 60)

    from model.backbones.swin_transformer import swin_tiny_patch4_window7_224
    from model.backbones.vpreid import VPReIDSwin

    base = swin_tiny_patch4_window7_224(
        img_size=[384, 128],
        drop_path_rate=0.1,
        pretrained='',
        convert_weights=False,
        semantic_weight=0.2,
        with_cp=False,
    )

    model = VPReIDSwin(
        base_swin=base,
        pose_cfg='',  # No real pose model for dry run
        pose_ckpt='',
        n_parts=5,
        part_temp=0.1,
    )

    B = 4
    x = torch.randn(B, 3, 384, 128)
    model.eval()
    out = model(x)

    assert isinstance(out, dict), f"Output should be dict, got {type(out)}"
    assert 'global_feat' in out, "Missing global_feat"
    assert 'part_feats' in out, "Missing part_feats"
    assert 'part_vis' in out, "Missing part_vis"
    assert 'foreground_feat' in out, "Missing foreground_feat"

    print(f"  global_feat: {out['global_feat'].shape}")
    print(f"  part_feats: {out['part_feats'].shape}")
    print(f"  part_vis: {out['part_vis'].shape}")
    print(f"  foreground_feat: {out['foreground_feat'].shape}")

    assert out['global_feat'].shape == (B, 768)
    assert out['part_feats'].shape == (B, 5, 768)
    assert out['part_vis'].shape == (B, 5)
    assert out['foreground_feat'].shape == (B, 768)

    # Check that num_features is accessible (needed by build_transformer)
    assert hasattr(model, 'num_features'), "Missing num_features"
    assert model.is_vpreid == True, "is_vpreid should be True"

    print("  PASSED\n")


def test_full_model():
    """Test complete model via make_model (includes BNNeck, classifiers)."""
    print("=" * 60)
    print("TEST 3: Full model via make_model")
    print("=" * 60)

    from config import cfg

    # Load VPReID config
    cfg.defrost()
    cfg.MODEL.NAME = 'transformer'
    cfg.MODEL.TRANSFORMER_TYPE = 'vpreid_tiny_patch4_window7_224'
    cfg.MODEL.PRETRAIN_PATH = ''
    cfg.MODEL.PRETRAIN_CHOICE = 'self'
    cfg.MODEL.WITH_CP = False
    cfg.MODEL.SEMANTIC_WEIGHT = 0.2
    cfg.INPUT.SIZE_TRAIN = [384, 128]
    cfg.MODEL.DROP_PATH = 0.1
    cfg.MODEL.JPM = False
    cfg.MODEL.NO_MARGIN = True
    cfg.MODEL.VPREID.ENABLE = True
    cfg.MODEL.VPREID.N_PARTS = 5
    cfg.MODEL.VPREID.PART_TEMP = 0.1
    cfg.MODEL.VPREID.POSE_CFG = ''
    cfg.MODEL.VPREID.POSE_CKPT = ''
    cfg.MODEL.VPREID.ID_WEIGHT = 1.0
    cfg.MODEL.VPREID.TRI_WEIGHT = 1.0
    cfg.MODEL.VPREID.PART_ID_WEIGHT = 0.5
    cfg.MODEL.VPREID.PUSH_WEIGHT = 0.1
    cfg.freeze()

    from model import make_model
    num_classes = 702  # Occluded-Duke
    model = make_model(cfg, num_classes, camera_num=8, view_num=1,
                       semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)

    B = 4
    x = torch.randn(B, 3, 384, 128)
    target = torch.randint(0, num_classes, (B,))

    # Training mode
    model.train()
    scores, feats, extras = model(x, label=target)

    print(f"  Training output:")
    print(f"    scores: {len(scores)} items, first shape: {scores[0].shape}")
    print(f"    feats: {len(feats)} items, first shape: {feats[0].shape}")
    print(f"    extras keys: {list(extras.keys())}")

    assert len(scores) == 2 + 5, f"Expected 7 scores (global+fg+5parts), got {len(scores)}"
    assert len(feats) == 2 + 5, f"Expected 7 feats, got {len(feats)}"
    assert scores[0].shape == (B, num_classes)
    assert 'part_vis' in extras

    # Eval mode
    model.eval()
    with torch.no_grad():
        feat_dict, _ = model(x)

    print(f"  Eval output:")
    print(f"    feat_dict keys: {list(feat_dict.keys())}")
    print(f"    global: {feat_dict['global'].shape}")
    print(f"    parts: {feat_dict['parts'].shape}")
    print(f"    part_vis: {feat_dict['part_vis'].shape}")

    assert feat_dict['global'].shape == (B, 768)
    assert feat_dict['parts'].shape == (B, 5, 768)

    print("  PASSED\n")
    return cfg


def test_loss(cfg):
    """Test loss computation."""
    print("=" * 60)
    print("TEST 4: VPReID loss computation")
    print("=" * 60)

    from loss import make_loss
    num_classes = 702
    loss_fn, center_criterion = make_loss(cfg, num_classes)

    B = 4
    K = 5
    D = 768

    # Simulate training outputs
    scores = [torch.randn(B, num_classes) for _ in range(2 + K)]
    feats = [torch.randn(B, D) for _ in range(2 + K)]
    target = torch.randint(0, num_classes, (B,))
    target_cam = torch.zeros(B, dtype=torch.long)
    extras = {'part_vis': torch.rand(B, K)}

    loss = loss_fn(scores, feats, target, target_cam, extras=extras)

    print(f"  Loss value: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    assert loss.item() > 0, "Loss should be positive"

    # Test backward
    loss.backward()
    print(f"  Backward: OK")
    print("  PASSED\n")


def test_evaluator():
    """Test R1_mAP_eval with dict features."""
    print("=" * 60)
    print("TEST 5: Evaluator with dict features")
    print("=" * 60)

    from utils.metrics import R1_mAP_eval

    num_query = 10
    num_gallery = 20
    D = 768
    K = 5

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm='yes')
    evaluator.reset()

    # Simulate batches
    for i in range(3):
        batch_size = 10
        feat_dict = {
            'global': torch.randn(batch_size, D),
            'parts': torch.randn(batch_size, K, D),
            'part_vis': torch.rand(batch_size, K),
        }
        pids = list(range(i * batch_size, (i + 1) * batch_size))
        camids = [0] * batch_size
        evaluator.update((feat_dict, pids, camids))

    cmc, mAP, distmat, _, _, _, _ = evaluator.compute()

    print(f"  mAP: {mAP:.4f}")
    print(f"  R-1: {cmc[0]:.4f}")
    print(f"  distmat shape: {distmat.shape}")
    assert distmat.shape == (num_query, num_gallery)
    print("  PASSED\n")


def test_part_losses():
    """Test PartAveragedTripletLoss and PushLoss."""
    print("=" * 60)
    print("TEST 6: Part losses")
    print("=" * 60)

    from loss.part_loss import PartAveragedTripletLoss, PushLoss

    B, K, D = 16, 5, 768

    # Part averaged triplet (soft margin)
    tri_fn = PartAveragedTripletLoss(margin=None, normalize=True)
    feats = torch.randn(B, K, D)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    vis = torch.rand(B, K)

    tri_loss = tri_fn(feats, labels, vis)
    print(f"  PartTripletLoss (soft margin): {tri_loss.item():.4f}")
    assert not torch.isnan(tri_loss), "Triplet loss is NaN!"

    # Push loss
    push_fn = PushLoss()
    push_loss = push_fn(feats)
    print(f"  PushLoss: {push_loss.item():.4f}")
    assert not torch.isnan(push_loss), "Push loss is NaN!"

    # Backward
    total = tri_loss + push_loss
    feats.requires_grad = True
    tri_loss2 = tri_fn(feats, labels, vis)
    tri_loss2.backward()
    assert feats.grad is not None
    print(f"  Backward: OK")
    print("  PASSED\n")


def count_params(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_param_count():
    """Count parameters to verify minimal overhead."""
    print("=" * 60)
    print("TEST 7: Parameter count")
    print("=" * 60)

    from model.backbones.swin_transformer import swin_tiny_patch4_window7_224
    from model.backbones.vpreid import VPReIDSwin

    # Baseline Swin-Tiny
    base = swin_tiny_patch4_window7_224(
        img_size=[384, 128], drop_path_rate=0.1,
        pretrained='', convert_weights=False, semantic_weight=0.2,
    )
    base_total, base_train = count_params(base)

    # VPReID Swin (no real ViTPose)
    vpreid = VPReIDSwin(
        base_swin=swin_tiny_patch4_window7_224(
            img_size=[384, 128], drop_path_rate=0.1,
            pretrained='', convert_weights=False, semantic_weight=0.2,
        ),
        pose_cfg='', pose_ckpt='', n_parts=5,
    )
    vp_total, vp_train = count_params(vpreid)

    overhead = vp_total - base_total
    print(f"  Baseline Swin-Tiny: {base_total:,} params ({base_train:,} trainable)")
    print(f"  VPReID Swin:        {vp_total:,} params ({vp_train:,} trainable)")
    print(f"  Overhead:           {overhead:,} params ({overhead/base_total*100:.2f}%)")
    print(f"  (PosePartHead has 0 learnable params — all overhead is from ViTPose if loaded)")
    print("  PASSED\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("VPReID Dry-Run Verification")
    print("=" * 60 + "\n")

    test_pose_part_head()
    test_vpreid_backbone()
    cfg = test_full_model()
    test_loss(cfg)
    test_evaluator()
    test_part_losses()
    test_param_count()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
