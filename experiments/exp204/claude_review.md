# exp204 Claude Broad Review: SupCon + PLBOA + ROA (dual occlusion augmentation)

## 审查范围
a. design.md -- 动机、假设合理性
b. 代码路径 -- PLBOA 与 ROA 的交互、执行顺序
c. 配置文件 -- 参数是否正确引用、默认值安全性
d. config/defaults.py -- POSE_ROA, POSE_ROA_PATH, POSE_ROA_PROB 检查
e. 与前序实验对照 -- 单变量隔离

## Critical: 实验无效 -- ROA 已经在 baseline 中生效

### 问题分析

`make_dataloader.py:80` 的 occluder 加载条件是:
```python
if getattr(cfg.MODEL, 'POSE_ROA', False) or getattr(cfg.MODEL, 'POSE_LOWER_BODY_OCC', False):
```

exp204 的 base config (`pose_psg_stdpr_pertoken_plboa_pape_ms_supcon.yml`) 已经设置 `POSE_LOWER_BODY_OCC: True`。因此 occluders **已经被加载**。

然后在 `pose_dataset.py:230` 的 standard 单视图路径中:
```python
if self.occluders and random.random() < self.roa_prob:
```

这里 `self.occluders` 非 None（因为已加载），`self.roa_prob` 默认 0.5（`defaults.py:108`，base config 没有覆盖）。所以 **exp176 (baseline) 已经在以 50% 概率执行标准 ROA**。

执行顺序:
1. PLBOA (line 174-179): p=0.7，paste VOC 物体遮挡下半身
2. 标准 ROA (line 230): p=0.5，paste VOC 物体到随机位置

两者独立概率，已在 baseline 中同时生效。

### 结论

设置 `MODEL.POSE_ROA True` 只影响 `make_dataloader.py:80` 的 OR 条件。由于 `POSE_LOWER_BODY_OCC=True` 已满足该条件，`POSE_ROA=True` 是空操作 (no-op)。设置 `POSE_ROA_PROB 0.5` 也是默认值，同样无效果。

**exp204 的训练行为与 exp176 完全相同。** 这不是一个有效的对照实验。

## 历史证据

- exp159 (PLBOA+ROA, 非 SupCon): 62.4% mAP，弱于 PLBOA-only 62.7%
- exp161e (STD-PR+PLBOA+ROA): 63.2%，ROA 不帮助 STD-PR (-0.2 vs 161b)
- 结论: ROA + PLBOA 在历史实验中已证明不正交，且组合效果略差

## 如果要真正测试"加 ROA"

要做的是 **反过来**: 以 exp176 为 baseline，设置 `POSE_ROA_PROB 0.0` 跑一个 **去掉 ROA** 的对照。因为 exp176 已经隐含了 ROA。

或者，修改 `pose_dataset.py` 在 PLBOA 场景下屏蔽标准 ROA 路径（加一个 `if cfg.MODEL.POSE_ROA:` 的显式检查），让两者真正独立控制。

## 配置安全性检查

- `POSE_ROA`, `POSE_ROA_PATH`, `POSE_ROA_PROB`: defaults.py 中均有定义，安全
- 远程 VOC 数据路径 `data/VOCdevkit/VOC2012`: 本地确认存在，远程需确认
- 远程 16GB 无 parallel_aug: 正确，1-view 不会额外增加显存

## 创新性质疑

此实验为纯配置组合，不涉及任何代码修改或新机制。即使假设 ROA 确实没在 baseline 中生效，这也只是组合两个已有增强方法，不满足创新门槛。

## 审查结论

**审查未通过 (Critical)**

| 级别 | 问题 | 状态 |
|------|------|------|
| Critical | ROA 已在 baseline 隐含生效，exp204 是 no-op | 需要修正 |
| High | 实验设计无创新价值（纯配置组合） | 需要重新设计 |
| Medium | ROA+PLBOA 在历史实验 (exp159) 中已证伪 | 建议跳过 |

### 建议行动

1. 修复 `pose_dataset.py` 中 ROA 的隐式触发 bug（当 POSE_ROA=False 时应跳过标准 ROA 路径，即使 occluders 被加载）
2. 放弃 exp204，因为 PLBOA+ROA 组合已在 exp159 中测试过且为负结果
3. 将 GPU 时间用于真正的创新实验

---

审查通过（条件性）: 仅在修复 ROA 隐式触发 bug 后，且确认实验目的不再是"加 ROA"而是其他有意义的对照时，方可继续。
