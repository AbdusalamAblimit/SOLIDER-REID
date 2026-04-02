# exp210 Small + GCN+PAA+CE+OA-SD + PKC (远程 1-view) 监控

配置: exp206 + Per-Keypoint Contrastive (PKC) loss
对照: exp206 (GCN+PAA+CE+OA-SD 1v Small): 70.5/82.3 (equal_concat), 72.1/82.9 (maxsim_hybrid)
**目标**: 73-74% mAP (with maxsim_hybrid)

**修正**: 本实验使用修复后的 OA-SD teacher (BN/Dropout/DropPath eval, train() mode forward)

## 检查点

### [18:26] 检查点 #1

**状态**: 正常
**进度**: Epoch 1/120

| 指标 | 当前值 |
|------|--------|
| oa_sd | 0.588 |
| pkc | 3.725 (初始 SupCon loss) |
| pkc_nk | 17.0 (所有 17 keypoints 参与) |
| id_global | 6.554 |

PKC loss 正常工作。17 个 keypoint 全部参与对比学习。
oa_sd=0.588 (修复后 teacher 更稳定)。
**决策**: 继续

### [18:30] 检查点 #2

**状态**: 正常
**进度**: Epoch 2/120

| 指标 | 当前值 |
|------|--------|
| oa_sd | 0.586 |
| pkc | 3.942 (从 3.725 略升——epoch 内波动正常) |
| pkc_nk | 17.0 |
| id_global | 6.554 |
| tri_global | 7.431 |

训练正常。pkc 和 oa_sd 都在合理范围。
**决策**: 继续

### [18:43] 检查点 #3 — teacher pose 修复后重启

发现 PLBOA 修改 persons 导致 teacher pose 被污染，修复后重启。
ep1 running. pkc=3.140 (下降中), oa_sd=0.462.
**决策**: 继续

### [18:49] 检查点 #4

ep3 done. pkc=3.892, oa_sd=0.243. ETA 5h58m.
oa_sd 从 0.462→0.243 快速下降——teacher 在追 student。
pkc 稳定在 ~3.9（初始阶段）。
**决策**: 继续

### [19:00] 检查点 #5

ep7. pkc=3.883 (几乎不变——keypoint contrastive 尚未真正发力), oa_sd=0.014.
id_global=6.536 (刚开始下降), Acc=0.014.
ep10 eval ~10min。
**决策**: 继续

### [19:06] 检查点 #6

ep9. id_global=6.513, Acc=0.015, pkc=3.873.
ep10 eval ~4min。
**决策**: 等 eval

### [19:11] 检查点 #7

ep10 done. id_global=6.469, Acc=0.040. pkc=3.871.
Eval 运行中。
**决策**: 等 eval

### [19:13] 检查点 #8 — ep10 ⚠️⚠️⚠️

**ep10: 3.6/5.3% — 灾难性失败！** (vs exp206 ep10: 47.9/60.3)

id_global=6.469 at ep10 几乎没有下降 (6.554→6.469 仅 -0.085)。
CE 完全没有收敛。Acc=0.040 at ep10 (exp206 ep10 约 0.09 at ep5)。

**可能原因**:
1. PKC 与 CE 在 GCN keypoint features 上梯度冲突
2. 修复后的 OA-SD teacher 本身有问题

**诊断**: 
已在远程启动 exp206r (相同配置但无 PKC) 作为对照。
如果 exp206r ep10 正常 (~47%) → PKC 是问题
如果 exp206r ep10 也低 → OA-SD teacher fix 有 bug

**exp210 已终止。**
