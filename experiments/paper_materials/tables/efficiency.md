# 效率分析

## 模型效率对比 (RTX 3090, 384x128 input)

| 方法 | Params | FLOPs | 推理速度 (ms/img) | mAP | R1 |
|------|--------|-------|-------------------|-----|-----|
| Baseline (Swin-Tiny) | 28.07M | 5.33G | 10.1 | 56.6 | 66.5 |
| **Ours (PCFC+GiLt)** | **30.78M (+9.6%)** | **5.33G (+~0%)** | **12.3 (+21.3%)** | **58.0** | **68.0** |

## PCFC 模块参数分解

| 组件 | 参数量 | 说明 |
|------|--------|------|
| Swin-Tiny backbone | 27.53M | 共享，不变 |
| Global classifier + BN | 0.54M | 共享，不变 |
| **Part classifiers (5x)** | **~2.70M** | **5个 part-specific 768->702 FC + BN** |
| PCFC alpha | 1 | 可学习标量，忽略不计 |
| **总 overhead** | **+2.70M (+9.6%)** | |

## 关键结论
- PCFC 不增加 backbone FLOPs（只在 GAP 层做 visibility weighting）
- 参数 overhead 来自 5 个 part classifier（每个 768x702 + BN）
- 推理时间增加 2.2ms 主要来自 part head 的 forward pass（5x BN + FC）
- **性价比**: +9.6% 参数换来 +1.4% mAP, +1.5% R1 的提升
- 注意：离线 pose 提取是一次性成本（ViTPose），不影响 ReID 推理效率
