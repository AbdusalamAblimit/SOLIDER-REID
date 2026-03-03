"""Dry-run script for PAMS: verifies shapes, forward/backward pass, and loss computation.

Usage:
    python dry_run_pams.py

No dataset or pretrained weights needed - uses random data.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Patch config before imports
from config import cfg

# Minimal config for PAMS dry run (no file needed)
cfg.defrost()
cfg.MODEL.NAME = 'transformer'
cfg.MODEL.TRANSFORMER_TYPE = 'pams_tiny_patch4_window7_224'
cfg.MODEL.PRETRAIN_CHOICE = 'imagenet'  # skip loading weights
cfg.MODEL.PRETRAIN_PATH = ''
cfg.MODEL.DEVICE_ID = '0'
cfg.MODEL.IF_LABELSMOOTH = 'off'
cfg.MODEL.METRIC_LOSS_TYPE = 'triplet'
cfg.MODEL.NO_MARGIN = True
cfg.MODEL.SEMANTIC_WEIGHT = 0.2
cfg.INPUT.SIZE_TRAIN = [384, 128]
cfg.DATALOADER.SAMPLER = 'softmax_triplet'
cfg.SOLVER.MARGIN = 0.3

# PAMS config
cfg.MODEL.PAMS.ENABLE = True
cfg.MODEL.PAMS.N_PARTS = 5
cfg.MODEL.PAMS.MSF_TARGET_HW = [24, 8]
cfg.MODEL.PAMS.MSF_OUT_DIM = 768
cfg.MODEL.PAMS.BPA_WEIGHT = 1.0
cfg.MODEL.PAMS.PUSH_WEIGHT = 0.1
cfg.MODEL.PAMS.ID_WEIGHT = 1.0
cfg.MODEL.PAMS.TRI_WEIGHT = 1.0
cfg.MODEL.PAMS.VIS_THRESHOLD = 0.5
cfg.MODEL.PAMS.LOSS_STRATEGY = 'pams'

# Pose disabled for dry run (no mmpose needed)
cfg.MODEL.POSE.ENABLE = False

cfg.freeze()

print("=" * 60)
print("PAMS Dry Run - Shape Verification")
print("=" * 60)

# ---- 1. Build model ----
from model import make_model
num_classes = 702  # Occ-Duke
model = make_model(cfg, num_class=num_classes, camera_num=8, view_num=1, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
model = model.cuda()

# Print model structure summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params / 1e6:.1f}M")
print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")

# Check PAMS detection
print(f"\nis_pams: {model.is_pams}")
assert model.is_pams, "PAMS not detected!"
print(f"in_planes (D): {model.in_planes}")
print(f"num_classes: {model.num_classes}")

# ---- 2. Training forward pass ----
print("\n" + "=" * 60)
print("Training Forward Pass")
print("=" * 60)

B = 16  # batch size
x = torch.randn(B, 3, 384, 128).cuda()
target = torch.randint(0, num_classes, (B,)).cuda()

model.train()
scores, feats, extras = model(x, label=target)

print(f"\nScores (list of {len(scores)} tensors):")
for i, s in enumerate(scores):
    print(f"  scores[{i}]: {s.shape}")

print(f"\nFeatures (list of {len(feats)} tensors):")
for i, f in enumerate(feats):
    print(f"  feats[{i}]: {f.shape}")

print(f"\nExtras keys: {list(extras.keys())}")
for k, v in extras.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")

# Verify expected shapes
assert len(scores) == 2, f"Expected 2 scores (global, fg), got {len(scores)}"
assert scores[0].shape == (B, num_classes), f"Global score shape mismatch: {scores[0].shape}"
assert scores[1].shape == (B, num_classes), f"FG score shape mismatch: {scores[1].shape}"
assert len(feats) == 7, f"Expected 7 feats (global, fg, 5 parts), got {len(feats)}"
D_global = model.pams_global_dim
D_part = model.pams_part_dim
assert feats[0].shape == (B, D_global), f"Global feat shape mismatch: {feats[0].shape}"
assert feats[1].shape == (B, D_part), f"FG feat shape mismatch: {feats[1].shape}"
for i in range(2, 7):
    assert feats[i].shape == (B, D_part), f"Part feat[{i}] shape mismatch: {feats[i].shape}"

print("\n[OK] Training forward shapes verified!")

# ---- 3. Loss computation ----
print("\n" + "=" * 60)
print("Loss Computation")
print("=" * 60)

from loss import make_loss
loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)

loss = loss_fn(scores, feats, target, torch.zeros(B).cuda().long(), extras=extras)
print(f"\nTotal loss: {loss.item():.4f}")
assert loss.isfinite(), "Loss is not finite!"
assert loss.requires_grad, "Loss doesn't have grad!"

# Backward pass
loss.backward()
print("[OK] Backward pass completed!")

# Check gradients exist
grad_count = sum(1 for p in model.parameters() if p.grad is not None)
print(f"Parameters with gradients: {grad_count}/{trainable_params}")

# ---- 4. Eval forward pass ----
print("\n" + "=" * 60)
print("Eval Forward Pass")
print("=" * 60)

model.eval()
with torch.no_grad():
    feat_dict, _ = model(x)

print(f"\nFeat dict keys: {list(feat_dict.keys())}")
for k, v in feat_dict.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")

assert 'global' in feat_dict, "Missing 'global' in eval output"
assert 'parts' in feat_dict, "Missing 'parts' in eval output"
assert 'part_vis' in feat_dict, "Missing 'part_vis' in eval output"
assert feat_dict['global'].shape == (B, D_global), f"Global eval shape mismatch: {feat_dict['global'].shape}"
assert feat_dict['parts'].shape == (B, 5, D_part), f"Parts eval shape mismatch: {feat_dict['parts'].shape}"
assert feat_dict['part_vis'].shape == (B, 5), f"Part vis shape mismatch: {feat_dict['part_vis'].shape}"

print("\n[OK] Eval forward shapes verified!")

# ---- 5. Evaluator test ----
print("\n" + "=" * 60)
print("Evaluator (Visibility-Aware Part Distance)")
print("=" * 60)

from utils.metrics import R1_mAP_eval

num_query = 4
evaluator = R1_mAP_eval(num_query, max_rank=10, feat_norm='yes')
evaluator.reset()

# PIDs must overlap between query and gallery, with different cam IDs
# Query: 4 items with pids 0,1,2,3, cam 0
# Gallery: 12 items with pids 0,1,2,3 repeated, cam 1
pids = [0, 1, 2, 3] + [i % 4 for i in range(12)]
camids = [0]*4 + [1]*12
evaluator.update((feat_dict, pids, camids))

results = evaluator.compute()
if isinstance(results, dict):
    for name, metrics in results.items():
        print(f"  [{name}] mAP: {metrics.mAP:.4f}, R1: {metrics.cmc[0]:.4f}")
else:
    print(f"  mAP: {results.mAP:.4f}, R1: {results.cmc[0]:.4f}")

print("\n[OK] Evaluator test passed!")

# ---- 6. Individual loss components ----
print("\n" + "=" * 60)
print("Individual Loss Components")
print("=" * 60)

from loss.part_loss import PartAveragedTripletLoss, PushLoss

model.train()
with torch.no_grad():
    scores2, feats2, extras2 = model(x, label=target)

part_feats_bn = torch.stack(feats2[2:], dim=1)  # [B, K, D]

# Part-averaged triplet
tri_fn = PartAveragedTripletLoss(margin=0.3)
tri_loss = tri_fn(part_feats_bn, target, extras2['part_vis'])
print(f"  Part-averaged triplet loss: {tri_loss.item():.4f}")

# Push loss
push_fn = PushLoss()
push_loss = push_fn(part_feats_bn)
print(f"  Push diversity loss: {push_loss.item():.4f}")

# BPA loss
if 'bpa_logits' in extras2:
    bpa_loss = F.cross_entropy(extras2['bpa_logits'], extras2['bpa_targets'])
    print(f"  BPA cross-entropy loss: {bpa_loss.item():.4f}")
else:
    print("  BPA loss: N/A (pose disabled in dry run)")

# ID loss
ce_loss = F.cross_entropy(scores2[0], target) + F.cross_entropy(scores2[1], target)
print(f"  ID loss (global + fg): {ce_loss.item():.4f}")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED!")
print("=" * 60)
