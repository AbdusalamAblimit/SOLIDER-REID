# exp231 Claude Review — BT-PKD Cosine Decay

## 审查范围

配置变更实验。仅新增 `MODEL.POSE_BT_PKD_DECAY_EPOCH 60`。

## 代码审查

新增代码在 `processor/processor.py` (6行):
```python
bt_pkd_decay_ep = int(getattr(cfg.MODEL, 'POSE_BT_PKD_DECAY_EPOCH', 0))
if bt_pkd_decay_ep > 0 and epoch > 0:
    import math
    if epoch >= bt_pkd_decay_ep:
        bt_pkd_weight = 0.0
    else:
        bt_pkd_weight *= 0.5 * (1 + math.cos(math.pi * epoch / bt_pkd_decay_ep))
```

- 默认 `POSE_BT_PKD_DECAY_EPOCH = 0` → 不影响已有实验
- `epoch >= decay_epoch` 时 weight=0 → BT-PKD 完全关闭
- cosine schedule 平滑衰减，无突变
- `math.cos` 在 AMP 下安全（标量运算）

## 单变量

vs exp229: 仅添加了 decay schedule。其余完全相同。

## 审查通过
