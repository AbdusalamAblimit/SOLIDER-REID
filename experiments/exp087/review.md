# exp087 Momentum Memory 审查报告

## 审查范围
- `experiments/exp087/design.md`
- `configs/occluded_duke/pose_psg_gcn_paa_mm.yml`
- `model/modules/momentum_memory.py`
- `processor/processor.py`
- `log/occluded_duke/exp087_mm_local/train_log.txt`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | LOW | processor.py | `MomentumMemory` 是在 processor 里临时构建的，不随 checkpoint 保存；若中途 resume，memory bank 会重置 | 接受 |
| 2 | LOW | processor.py | 训练类别数通过 `train_loader.dataset.dataset` 这种内部结构推断，可用但写法偏脆弱 | 接受 |

## 审查通过项

- `POSE_MOMENTUM_MEMORY`、温度、权重、动量都已正确接线
- memory bank 是 buffer，不会误入优化器
- 每步先更新类原型再做 CE loss，整体逻辑与 proxy/memory 对比学习一致
- 该 loss 直接作用在 global 分支特征上，确实会影响主干训练
- 日志完整，120 epoch 结束，`mm` 分量被稳定记录

## 结论

✅ **通过**

`exp087` 的实现是成立的。它的主要问题是工程封装不够完美，而不是方法没接上。
