# 实验 exp150: PCVT-Random（随机遮挡对照）

## 动机

`exp148 PCVT` 在 ep30 展示了 +2.4 mAP 的强正向信号。但我们必须回答一个关键机制问题：

**这个增益来自 pose-guided complementary masking（PCVT 核心创新），还是仅仅来自三视图多样化训练 + consistency loss（通用正则化效果）？**

如果 PCVT ≈ Random masking → pose-guided 部分不重要，创新 claim 不成立
如果 PCVT >> Random masking → pose-guided complementary structure 是关键，创新 claim 成立

这是论文消融实验的核心对照，无论 exp148 最终结果如何，这个实验都有价值。

## 核心假设

1. PCVT 的增益不仅来自"多视图 + consistency"这种通用正则化
2. Pose-guided body-part complementary 结构提供了比随机遮挡更有针对性的训练信号
3. 如果假设成立，exp150 应该显著弱于 exp148

## 技术方案

### 唯一改动：masking 策略从 pose-guided 变为 random

保持与 exp148 **完全相同**的：
- 架构：PSG + GCN + equal_concat
- 三视图前向：full + view_a + view_b
- consistency loss：`1 - cos(f_full, f_union)`，权重 0.25
- 训练参数：0.5x global loss, 120 epochs, seed 1234
- 评估方式：equal_concat

**唯一不同**：
- exp148：由 pose heatmap body-group response 做 greedy balanced partition → 互补 body-part masking
- exp150：将图像划分为 8×4 网格（32 块），随机均衡分配给 A/B → 随机空间块 masking

### 随机遮挡实现细节

1. 将图像（384×128）划分为 8×4 = 32 个 48×32 的空间块
2. 随机打乱 32 个块的索引
3. 前 16 个分给 A（view_a 中遮掉），后 16 个分给 B（view_b 中遮掉）
4. 这保证了：
   - 覆盖率 ≈ 50%/50%（与 PCVT 的 `pcvt_cov_a/b ≈ 0.50` 匹配）
   - 完全互补（A ∩ B = ∅，A ∪ B = 全图）
   - 空间连续性（块级而非像素级，更接近真实遮挡模式）

### config 实现

在 `defaults.py` 中新增：
- `MODEL.POSE_PCVT_RANDOM = False`

当 `POSE_PCVT_RANDOM = True` 时，`_make_pcvt_views` 跳过 body-part partition，改用随机块分配。

## 对照组

- 主对照：`exp148 PCVT`（pose-guided complementary masking）
- 间接对照：`exp030a-eq`（无 PCVT 基线）

## 预期结果

### 场景 1: PCVT >> Random（最佳情况，支撑论文 story）
- exp150 弱于 exp148 至少 1% mAP
- 说明 pose-guided complementary structure 是核心机制
- 论文可以写："pose-guided body-part complementary masking significantly outperforms random masking, confirming that the pose-structured view decomposition, not just multi-view regularization, drives the improvement"

### 场景 2: PCVT ≈ Random（中性，削弱 pose-guided claim）
- exp150 与 exp148 相差 <0.5% mAP
- 说明增益主要来自三视图 + consistency 正则化
- 需要重新定位 PCVT story：从 "pose-guided complementary" 改为 "multi-view consistency training"

### 场景 3: Random 也为正但弱于 PCVT（理想消融结果）
- exp150 > exp030a 但 < exp148
- 说明三视图 consistency 本身有价值，而 pose guidance 提供额外增益
- 论文可以做完整消融：baseline < random < pose-guided

## 关键日志

复用 PCVT 的所有日志键：
- `pcvt_cov_a/b/u`: 在随机模式下应分别 ≈ 0.5/0.5/1.0
- `pcvt_ovr`: 应 = 0.0（互补保证）
- `pcvt_fb`: 应 = 0.0（random 模式不需要 fallback）
- `pcvt_cos_fa/fb/fu/gap`: 核心机制指标，与 exp148 直接可比

## 风险

1. 若 ep30 已明显弱于 exp030a → 说明随机遮挡太强/太不相关，但这本身也是有信息的
2. 若与 exp148 几乎相同 → 需要想新的对照来区分
