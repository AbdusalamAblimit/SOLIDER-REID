# 实验 exp146: PSG+GCN with 0.25x Global Loss Scale

## 动机
- 0.5x global loss 在 PSG 上稳定有效 (+1.37% mAP, p=0.006)
- exp007b 测试了 PSG-only 0.25x → 58.3% (= 1.0x，无效)
- 但 PSG+GCN 的 loss landscape 不同（list-loss path），0.25x 可能在 GCN 基础上有不同表现
- 零代码改动，仅 config 差异

## 对照组
- exp030a (PSG+GCN, 隐式 0.5x): 61.1% / 73.7% (equal_concat)

## 预期
- 如果 0.25x 有效：说明全局 loss 正则化有更优 sweet spot
- 如果中性：与 exp007b 结论一致
- 如果负面：说明 GCN 需要更多 global gradient 信号
