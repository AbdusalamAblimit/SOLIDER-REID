# 实验 exp118: PAA+ROA+VCGA 组合

## 动机

exp117 表明 VCGA 在基础配置（exp030a）上完全中性。但 VCGA 的效果可能在更强的基线上更明显——当 PAA 已经改善了多人场景的 backbone 特征，VCGA 对 GCN 的 visibility-conditioned routing 可能更有信息量。

同时，这个实验也是为了建立 **PAA+ROA+VCGA 完整配置**的性能参考点。

## 核心假设

VCGA 在 PAA+ROA 基础上可能正交叠加，因为：
- PAA 改善 backbone 特征质量 → GCN 接收的 keypoint features 更好 → VCGA 的 visibility routing 更有效
- ROA 增加遮挡训练样本 → 更多低可见度 keypoint → VCGA 的差异化更明显

## 技术方案

在 exp085（PAA+ROA p=0.7）基础上添加 `POSE_VCGA: True`。

## 对照组

- 主对照: `exp085-eq`（PAA+ROA p=0.7）= **62.6% / 75.3%**

## 预期结果

- 若正交: +0.3~0.5% mAP
- 若中性: = exp085（更可能，鉴于 exp117 的中性结论）
- 若负面: 不太可能（VCGA 设计为退化安全）
