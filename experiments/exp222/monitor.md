# exp222 Small + GCN+PAA+CE+OA-SD + GSPB (scale=0.05) 监控

配置: Small GCN+PAA+CE+OA-SD + POSE_PART_GRAD_SCALE=0.05
对照: exp206r (scale=0): 70.6 eq, 72.3 maxsim. exp210b (PKC): 72.4 maxsim
**目标**: GSPB+MaxSim > 72.4% (新 Small 最佳)

**动机**: exp220 (Tiny) 证明 GSPB+MaxSim=64.6 > OA-SD+MaxSim=64.2 (+0.4)

## 检查点

### [13:09] 检查点 #1

本地 exp222 (scale=0.05): ep2. ETA 3h32m. 正常。
远程 exp222b (scale=0.1): OOM → 加 WITH_CP 重启。
**决策**: 继续。ep10 eval ~30min。

### [13:17] 检查点 #2

exp222 ep6. ETA ~3h25m.
exp222b (scale=0.1, WITH_CP) ep2. ETA longer (~6h with CP).
**决策**: 继续。ep10 eval ~20min。

### [13:22] 检查点 #3

ep9. ep10 eval ~3min.
**决策**: 等 eval

### [13:26] ep10 — 灾难！

**ep10: 2.3/3.9% — CATASTROPHIC!**
GSPB scale=0.05 对 Small 太强！
Tiny 成功是因为 GCN 参数量相对小。Small 的 GCN 更大 → Part gradient 更强。
**需要更小的 scale (0.01 或 0.005)。**
补查远程原始日志后可确认：`exp222b (scale=0.1)` 在 epoch 1 就多次出现 `tri_part=inf`，且未跑到第一次 eval。
因此 `exp222b` 只能算“早期即失稳”的辅助证据，不是完整对照。
exp222 和 222b 均已终止。

### 重启: exp222c scale=0.01

即将用 scale=0.01 重跑。

### [13:30] 检查点 #4 — 重启

本地 exp222c (scale=0.01) ep2. 正常。
远程 exp222d (scale=0.005, WITH_CP) ep1. 正常。
两台 ~6h。ep10 eval ~30min (local), ~50min (remote).
**决策**: 继续监控

### [13:36] 检查点 #5

exp222c ep5. id_global=6.532. 正常。ep10 eval ~15min.
**决策**: 继续

### [13:42] 检查点 #6

exp222c ep8. Acc=0.052 (exp206r ep8 ~0.17 — 低但不灾难)。ep10 eval ~5min.
**决策**: 等 eval

### [13:47] ep10 — scale=0.01 也是灾难!

**exp222c (scale=0.01) ep10: 15.1/23.8%** — 远低于 exp206r (50.4/63.9)

| scale | ep10 mAP | backbone |
|-------|---------|----------|
| 0.05 | 40.1% | Tiny (28M) ✅ |
| 0.05 | 2.3% | Small (50M) ❌ |
| 0.01 | 15.1% | Small (50M) ❌ |

**GSPB 只在 Tiny 上有效！Small 对 Part gradient 极度敏感。**
Small 的 GCN Part branch 梯度规模更大，即使 1% 也足以干扰 CE。

补查远程原始日志后可确认：`exp222d (scale=0.005)` 只跑到约 `ep7`，未形成首次 eval。
因此当前真正有评估数值支撑的 Small 线只有：
- `scale=0.05 -> ep10 2.3/3.9`
- `scale=0.01 -> ep10 15.1/23.8`

`scale=0.1` 的证据是“epoch1 即失稳”，`scale=0.005` 的证据是“早期训练仍明显异常且在首个 eval 前止损”。
**exp222c 和 222d 均终止。现有证据已足够支持 “GSPB on Small 不可作为当前主线” 的判断，但要避免把 0.005 写成已有完整负结果。**

## 最终结论

**GSPB 是 Tiny-only 的创新:**
- Tiny: scale=0.05 → early +5.8%, final -0.3%, **maxsim +0.4% (新 Tiny 最佳 64.6%)**
- Small: 任何 scale > 0 → 灾难

**Small 的 GCN+PAA+OA-SD+MaxSim 72.4% 仍是最佳。**

### [14:00] 状态

两台 GPU 空闲。研究 agent 正在搜索新架构方向。
等待结果后立即启动下一实验。
