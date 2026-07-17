# Claude Broad Review — exp335 ViT-base + LGPA-D 复现

**审查范围**：design.md + scripts/exp335_train_vit_lgpa.py（主）+ config，交叉核对 clip_part_head.py / make_dataloader.py / pose_dataset.py / make_loss.py / make_optimizer.py / metrics.py / vit_pytorch.py。

**结论：NEEDS-FIX（2 High，均已修）→ 代码运行正确，无 Critical/runtime/NaN。**

## 已修 High
- **H1 损失权重不匹配原 LGPA-D**：原版把 `[global, pooled, p1..p5]` 7 元素 list 喂 make_loss list-branch → `0.5·global + 0.5·mean(pooled+5parts)` ID & **每部位 triplet**。我原来：global 全权重 1.0、pooled 单独全权重、parts 仅 CE 无 triplet、不平衡 → **可能复现不出 +4.4**。**修复**：构造 7 元素 list 调 loss_func 一次（走 list-branch，忠实原权重）+ assign。
- **H2 pose vs 非-pose dataloader 混淆 A/B**：off 臂走非-pose loader（timm RE max 0.333），on 臂走 pose loader（RE max 0.40）→ 增强不同、混淆。**修复**：两臂都 POSE_ENABLED=True 走 pose dataloader，只 gate LGPA head → 干净单变量 + alpha=0 sanity 才有效。

## 逐项 PASS（源码核对）
- token→spatial 布局正确：patch_embed (B,768,16,8) flatten y-major，`patches.transpose(1,2).reshape(B,768,16,8)` 重建 [y,x] 精确（fH=16=num_y, fW=8=num_x）。
- `feat_map.detach()`（LGPA-D，gated POSE_LGPA_DETACH=True）✓；CLIPPartHead 调用签名匹配 ✓。
- AMP（kl_div log-sum-exp + clamp + isfinite 守卫）安全 ✓；优化器覆盖 lgpa 子模块参数、CLIP text buffer 正确排除 ✓；scheduler/eval/R1_mAP_eval ✓。
- `_heatmaps` 处理 (B,6,17,H,W)→[:,0] 目标人 ✓（实测非零）；与原版 POSE_USE_TARGET_HEATMAP=True 喂 heatmaps[:,0] 一致 ✓。
- 融合 cls(before-BN)+alpha·pooled(各 L2 归一)concat，model.eval()，无泄漏 ✓。
- smoke：heatmaps 非零、assign≈0（pose-bias 已对齐注意力→KL 已满足，正常非 bug）、5 alpha eval 跑通。

## 结论
代码运行正确；2 个 High（损失忠实度 + 增强混淆）已修，确保复现可被诚实解读为"CLIP 是否带 +X"。审查通过（修复后），待 Codex 复审。
