# 实验 exp220: Gradient-Scaled Part Branch (GSPB) — Tiny

## 动机

### 根本问题
所有 per-keypoint training innovations 失败的根本原因是 `detach()`:
- detach=True: 梯度不到 backbone → 无效
- detach=False: Part 梯度全量到 backbone → 与 CE 冲突 → 灾难

### 解决方案: Gradient Scaling
不是 0 或 1 的选择，而是让 Part 分支的梯度以**缩放因子**传回 backbone。
- `scale=0.0` = 完全 detach (现状)
- `scale=1.0` = 完全 non-detach (exp215 灾难)
- `scale=0.01~0.1` = **微弱但持续的 Part→Backbone 梯度信号**

### 为什么这可能有效
- exp022 证明 detach 是必要的——Part CE/triplet 的全量梯度干扰 Global
- 但 0.01x 的梯度信号可能足够让 backbone "知道" Part 分支的需求
- 这相当于 "Part 分支作为辅助任务，以极低学习率影响共享 backbone"
- 类似于 multi-task learning 中的 task-specific learning rate

### 实现
```python
# 现状:
feat_map_detached = featmaps[-1].detach()

# GSPB:
scale = 0.05  # Part gradients scaled to 5% of original
feat_map_scaled = featmaps[-1].detach() + scale * (featmaps[-1] - featmaps[-1].detach())
# This is equivalent to: feat_map_scaled has gradient = scale * original_gradient
```
这个 trick 使用 `detach() + scale * (x - x.detach())`，在 forward 时值不变，
但在 backward 时梯度被 scale 缩放。

## 核心假设
0.05x 的 Part→Backbone 梯度让 backbone 缓慢学习 part-level discriminability，
不破坏 Global CE 收敛，并有机会超过 `exp191 = 63.2/75.4`，或至少在 `MaxSim` 上表现出更强的 per-keypoint signal。

## 技术方案
- `model/pose_backbone_model.py`: 替换 `detach()` 为 gradient scaling
- `config/defaults.py`: POSE_PART_GRAD_SCALE

## 对照组
- exp191 OA-SD (scale=0.0): 63.2/75.4
- exp215 BA-PKC (scale=1.0): 灾难 0.5%
- exp220 GSPB (scale=0.05): 目标 65%+
