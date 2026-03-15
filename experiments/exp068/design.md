# 实验 exp068: Reliability-Routed PAA (RR-PAA)

## 动机
- exp066 PAA: +0.87%/+1.63%（加法 adapter 对所有 token 均匀加）
- 但遮挡区域的 token 比可见区域更需要补全
- **RR-PAA**: 只在低置信度区域加 adapter content，高置信度区域保留 PSG gate

## 创新点
- PSG 抑制背景（乘法 gate 降权非人体区域）
- RR-PAA 补全遮挡（加法 adapter 只在遮挡区域激活）
- 分工明确：PSG suppress + PAA complete → 对应用户建议的 Suppress-and-Complete 方向

## 技术方案
- 修改 PoseAdditiveAdapter.forward():
  - 计算 body_confidence = sigmoid(max(heatmap, dim=1))
  - occlusion_mask = (1 - body_confidence)  → 遮挡区域 ≈ 1，可见区域 ≈ 0
  - adapter_out = adapter_out * occlusion_mask → 只在遮挡区域加 content
- 零额外参数，只改 forward 路由逻辑

## 对照组
- exp066 PAA (uniform additive): 61.6%/74.2%
- exp030a baseline: 60.73%/72.57%
