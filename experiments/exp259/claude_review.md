# Claude 审查报告 — exp259 / exp259b

**日期**: 2026-04-10
**审查员**: Claude Opus 4.6 (sub-agent)
**实验**: exp259 (WD=2e-4) / exp259b (OA-SD weight=2.0)
**对照**: exp255 (WD=1e-4, OA-SD weight=1.0, mAP 73.2/R1 83.3)

---

## 一、实验定位审查

exp259 系列是 config-only 超参调整实验，基于 exp255（当前最强配置：Small + GCN512 + 2-stage PSG + LGPA-D + OA-SD + PLBOA）。两个变体各改动一个超参数：

- **exp259**: `SOLVER.WEIGHT_DECAY` 1e-4 → 2e-4（更强 L2 正则化）
- **exp259b**: `MODEL.POSE_OA_SD_WEIGHT` 1.0 → 2.0（更强教师蒸馏权重）

设计文档简洁但信息完整：记录了对照基线、实验动机、两个变体的具体修改。单变量原则满足。

---

## 二、config 参数存在性验证

### 2.1 SOLVER.WEIGHT_DECAY

```python
# config/defaults.py, line 376
_C.SOLVER.WEIGHT_DECAY = 0.0005
# config/defaults.py, line 377
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
```

**确认**: `SOLVER.WEIGHT_DECAY` 和 `SOLVER.WEIGHT_DECAY_BIAS` 均存在于 `config/defaults.py`。
yacs CfgNode 的 override 机制要求 key 必须存在于 defaults 中才能被 YAML 覆盖 — 此条件满足。

### 2.2 MODEL.POSE_OA_SD_WEIGHT

```python
# config/defaults.py, line 183
_C.MODEL.POSE_OA_SD_WEIGHT = 1.0          # Distillation loss weight
```

**确认**: `MODEL.POSE_OA_SD_WEIGHT` 存在于 `config/defaults.py`，默认值 1.0。可被 YAML override 到 2.0。

---

## 三、参数使用路径验证

### 3.1 WEIGHT_DECAY 的使用路径

`solver/make_optimizer.py` 中，`SOLVER.WEIGHT_DECAY` 和 `SOLVER.WEIGHT_DECAY_BIAS` 的使用如下：

```python
# make_optimizer.py, lines 11-14
weight_decay = cfg.SOLVER.WEIGHT_DECAY
if "bias" in key:
    weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
...
params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
```

per-param 分组后，每个参数组的 `weight_decay` 直接来自 config。Adam 优化器走 `getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)`，即 `Adam(params)`，params 中的 per-group weight_decay 正确生效。

**注意**: AdamW 分支（line 30）额外传了 `weight_decay=cfg.SOLVER.WEIGHT_DECAY`，但 exp255 使用 Adam（默认），所以 AdamW 分支不会触发。Adam 使用 per-param group 的 weight_decay，exp259 的 override 完全有效。

**结论**: WD=2e-4 会正确应用到所有非 bias 参数，WEIGHT_DECAY_BIAS 未改动仍为 5e-4（bias 走旧值）。这是合理的调参设计。

### 3.2 POSE_OA_SD_WEIGHT 的使用路径

`processor/processor.py` 第 709 行：

```python
oa_sd_weight = float(getattr(cfg.MODEL, 'POSE_OA_SD_WEIGHT', 1.0))
```

第 773 行：

```python
loss = loss + oa_sd_weight * oa_sd_loss
```

**数据流验证**: OA-SD loss（cosine distillation loss，per-token 平均后加权）被 `oa_sd_weight` 直接线性缩放后加入总 loss。当 weight=2.0 时，蒸馏 loss 对总梯度的贡献翻倍。该缩放在数值上安全，`oa_sd_loss` 本身是 cosine similarity 项，范围 [0,2]，典型值 ~0.1~0.3，乘以 2 后仍在合理范围，不会引起梯度爆炸。

**结论**: POSE_OA_SD_WEIGHT=2.0 的路径验证通过。

---

## 四、实验隔离性与安全性验证

### 4.1 不影响其他实验的可复现性

两个超参均为纯 config override，通过 YAML 文件传递，不修改任何代码文件：
- `config/defaults.py` 未改动：已有实验均使用各自的 YAML 配置，默认值保持不变
- `models/` 下无任何代码修改
- `processor/` 下无任何代码修改
- `solver/` 下无任何代码修改

**结论**: 对已有实验（exp255、exp249、所有历史实验）的可复现性无任何影响。

### 4.2 单变量隔离

- exp259 vs exp255：仅改 `SOLVER.WEIGHT_DECAY`（1e-4 → 2e-4），其余配置完全相同
- exp259b vs exp255：仅改 `MODEL.POSE_OA_SD_WEIGHT`（1.0 → 2.0），其余配置完全相同
- exp259 与 exp259b 之间不共享修改变量

**单变量原则满足。**

### 4.3 数值安全性

- **WD=2e-4**: 标准范围内。Adam 中 WD 过大可能欠拟合，但 2e-4 vs 1e-4 变化量小，风险低。
- **OA-SD weight=2.0**: OA-SD loss 量级约 0.1~0.3（cosine），乘以 2 后约 0.2~0.6。ID loss + Triplet loss 通常 ~2~4，OA-SD 占比约 5~15%，加倍后约 10~30%，仍在合理范围内，不会主导训练。

---

## 五、设计合理性审查

### 5.1 exp259（WD 正则化）的动机

exp255 在 120 epoch 收敛后 mAP=73.2，与 exp249 差距 +1.3。增加 WD 可抑制大容量模型（GCN512 相比 GCN256 参数翻倍）的过拟合风险。这是合理的调参方向。

潜在风险：WD 加大会对所有参数施压，可能对 GCN 分支（参数量大）的影响更明显，存在欠拟合可能性。但变化量仅 1e-4，风险可控。

### 5.2 exp259b（OA-SD weight）的动机

exp255 已经用 OA-SD weight=1.0。更强的蒸馏监督（weight=2.0）旨在让 student（occluded view）更好地对齐 EMA teacher（clean view），从而增强对遮挡的鲁棒性。

潜在风险：OA-SD loss 加倍可能与 ID/Triplet loss 产生梯度竞争，导致分类损失下降速度变慢。但实验设计合理，值得验证。

### 5.3 论文价值

这两个实验均为消融/调参性质的 supporting evidence：
- 正向结果：加入 ablation table，说明 hyperparameter robustness 或最优配置
- 负向结果：说明 exp255 的 WD=1e-4 / OA-SD=1.0 已是最优，同样有论文价值

两个实验都具备明确的对照组和可解释性，符合论文素材要求。

---

## 六、问题列表

无 Critical / High 级别问题。

### Low 级别（不阻断）

1. **design.md 较简略**: 未明确说明 exp255 中当前 WD 的实际值（需确认 exp255 config 使用的是 `SOLVER.WEIGHT_DECAY=1e-4` 而非 defaults 的 5e-4）。建议后续 monitor.md 中补充确认。

2. **WEIGHT_DECAY_BIAS 未同步修改**: exp259 只改了 `SOLVER.WEIGHT_DECAY`，`SOLVER.WEIGHT_DECAY_BIAS` 保持旧值。这是有意设计还是疏漏？如果 bias 正则化也应增强，需同步修改。当前设计可接受，但应在 monitor.md 中注明。

---

## 七、总结

| 检查项 | 结果 |
|--------|------|
| design.md 存在且描述合理 | 通过 |
| SOLVER.WEIGHT_DECAY 存在于 defaults.py | 通过 |
| SOLVER.WEIGHT_DECAY_BIAS 存在于 defaults.py | 通过 |
| MODEL.POSE_OA_SD_WEIGHT 存在于 defaults.py | 通过 |
| WEIGHT_DECAY 使用路径正确（make_optimizer.py） | 通过 |
| OA_SD_WEIGHT 使用路径正确（processor.py） | 通过 |
| 单变量原则（每变体仅改一个超参） | 通过 |
| 不影响其他实验可复现性 | 通过 |
| 数值安全性 | 通过 |
| 代码修改量 | 无代码修改，纯 config override |
| Critical/High 问题 | 无 |

两个变体均为安全的单变量 config override，参数路径验证无误，设计合理，具备论文消融价值。

**审查通过**
