# exp009: Multi-Stage PSG — 层次化姿态注入

## 设计动机
exp007 证明 PSG (Pose Spatial Gate) 在 Stage 3 注入有效 (+1.7% mAP)。但 Stage 3 只有 2 个 block，且空间分辨率仅 12×4 — pose heatmap 的空间信息在此分辨率下被大量压缩。

Stage 2 有 6 个 block 和 24×8 的空间分辨率（4× Stage 3），pose heatmap 可以提供更精确的空间引导。虽然 exp005 证明 Stage 2 特征不适合直接做 identity classification，但 PSG 不做分类 — 它只提供 spatial attention bias，这在语义不够深的层也可能有用。

## 假设
更早注入 pose spatial gate（Stage 2）可以让 backbone 在更高空间分辨率下学习 pose-aware 特征，这些改善会传播到后续层，最终产生比仅 Stage 3 注入更好的全局特征。

## 架构
- Stage 0-1: 无修改
- Stage 2: 6 个 SwinBlock 后各加 1 个 PSG (feat_channels=384, hidden_dim=64)
- Stage 3: 2 个 SwinBlock 后各加 1 个 PSG (feat_channels=768, hidden_dim=64)
- 总计 8 个 PSG 模块，258K extra params (vs exp007 的 102K)
- Test feature: global (768-dim)

## 配置
- Config: `configs/occluded_duke/pose_multi_psg.yml`
- Output: `./log/occluded_duke/exp009_multi_psg`
- `POSE_PSG_STAGES: [2, 3]`

## 对照
| 实验 | PSG 位置 | Extra params | mAP | R1 |
|------|----------|-------------|-----|-----|
| exp000 | 无 | 0 | 56.6% | 66.5% |
| exp007 | Stage 3 only | 102K | 58.3% | 67.9% |
| exp009 | Stage 2+3 | 258K | 58.3% | 67.2% |
