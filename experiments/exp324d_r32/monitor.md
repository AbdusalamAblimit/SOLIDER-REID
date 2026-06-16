# exp324d_r32 监控日志 (DINOv2-base + LoRA rank32)

机器：hyy-5060ti-double GPU1 (RTX 5060Ti 16G)
脚本：`scripts/exp324d_lora.py --dino_model facebook/dinov2-base --lora_rank 32 --lora_alpha 32 --micro_bs 32 --epochs 30 --eval_period 5`
输出：`/hy-tmp/reid-clean/SOLIDER-REID/log/occluded_duke/exp324d_r32`
日志：`/tmp/exp324d_r32.log`

## 基线（对照）
- 冻结 DINOv2-base (exp324b)：part-MaxSim 重遮挡 mAP **8.65** / 全部 **14.61**（e20 触顶）
- Swin baseline：72.57
- base-rank16：lab-3090-d 在跑（隔离 LoRA rank 16 vs 32）
- 平行 exp324d_large (large rank16)：hyy GPU0

## Dry-run 验证 (08:00)
- hidden=768 正确；LoRA params=1,179,648（rank32，是 rank16 ~0.59M 的 2 倍），head=1.11M
- 3 步 loss 下降（tri 8.34→6.61），梯度到 LoRA 确认
- peak GPU=2.81G (micro_bs=32)，~2.6s/step（比 large 快 3x）
- ~244 batch/epoch × 2.6s ≈ 10.5min/epoch × 30 ≈ 5.5h

## 启动 (08:05)
- nohup 后台启动 GPU1
- 加载 query/gallery pool 缓存 OK，heavy-occ queries 989/2210
- 进入 epoch 1：GPU1 3299 MiB / 75% util，进程 ALIVE

## 训练进度

### Epoch[1] (630s, ~10.5min as预期)
`loss=10.07 id=5.47 tri=1.69 part=5.82 d_ap=23.15 d_an=23.80 acc=0.412 lr=1.0e-4`
- **acc 0.412**：ID 分类准确率 1 epoch 即 41%，loss 从 init ~18 降到 10.07，tri 8.34→1.69
- d_ap(23.15) < d_an(23.80)：positive 比 negative 近，triplet 在工作（margin 仍小）
- **信号积极**：LoRA 确实在让 backbone 自适应学判别性——冻结 baseline 做不到的事
- 等 e5 part-MaxSim eval（决定性 heavy/all mAP vs 冻结 8.65/14.61）

### Epoch[2-4] 训练端快速收敛
- e2: loss=4.31 acc=0.842 d_ap=34.0 d_an=41.0（margin 7）
- e3: loss=1.53 acc=0.953 d_ap=39.7 d_an=50.1（margin 10）
- e4: loss=0.76 acc=0.974 d_ap=40.1 d_an=51.5
- e5: loss=0.43 acc=0.985 d_ap=40.5 d_an=52.9

### ★ Epoch[5] EVAL — 决定性突破 ★
```
cos  ALL  : mAP=43.45 R1=55.57 R5=72.35 R10=77.65
part ALL  : mAP=44.54 R1=57.47 R5=73.76 R10=78.64
cos  HEAVY: mAP=34.52 R1=44.19 R5=62.08 R10=68.66
part HEAVY: mAP=36.72 R1=49.44 R5=65.93 R10=71.59
```

| 片 | LoRA r32 e5 | 冻结 baseline | 增益 |
|----|-------------|---------------|------|
| **part HEAVY mAP** | **36.72** (R1 49.44) | 8.65 | **+28.07 (4.2×)** |
| **part ALL mAP** | **44.54** (R1 57.47) | 14.61 | **+29.93 (3.0×)** |

- **核心问题回答：是。** LoRA 解冻把重遮挡 mAP 从 8.65 推到 36.72，进入 competitive 区间。
  FM-adaptation 确实让 DINO 特征判别化、突破冻结天花板——瓶颈是"冻结"不是"表征结构"。
- 仅 e5/30，仍在涨。等 e10/15/.../30 看上限。part 略优于 cos（pose-anchor 部位匹配仍有边际增益）。

### Epoch[10] EVAL — 继续上涨，未 plateau
```
cos  ALL  : mAP=46.15 R1=58.91   part ALL  : mAP=47.12 R1=59.64
cos  HEAVY: mAP=36.89 R1=47.32   part HEAVY: mAP=38.85 R1=50.05
```
- part HEAVY 36.72→**38.85** (+2.1)，part ALL 44.54→**47.12** (+2.6)，e5→e10 仍稳涨。
- e6-9 训练端饱和（acc 0.990→0.996，loss 0.29→0.11），但 retrieval 还在升 → 还没到上限，等 e15+。

### Epoch[15] EVAL — 明显 plateau
```
cos  ALL  : mAP=47.08 R1=59.86   part ALL  : mAP=47.71 R1=60.00
cos  HEAVY: mAP=38.03 R1=49.04   part HEAVY: mAP=39.44 R1=49.75
```
- part HEAVY 38.85→**39.44** (+0.6)，part ALL 47.12→**47.71** (+0.6)。e10→e15 增益骤降。

### 趋势汇总（part-MaxSim mAP）
| epoch | HEAVY | ALL | Δheavy/5ep |
|-------|-------|-----|-----------|
| frozen | 8.65 | 14.61 | — |
| e5 | 36.72 | 44.54 | — |
| e10 | 38.85 | 47.12 | +2.13 |
| e15 | 39.44 | 47.71 | +0.59 |
| e20 | 40.58 | 48.89 | +1.14 |

e20: `part ALL mAP=48.89 R1=61.18 / part HEAVY mAP=40.58 R1=52.17`（cos ALL 48.32 / cos HEAVY 39.25）

- **plateau 在 ~40 heavy / ~49 all**。LR cosine 衰减中(3e-5)、acc 0.999 训练饱和；e15→e20 +1.1 是低 LR 微调小涨，未脱离 plateau 带。
- **核心问题已 firm 回答**：LoRA 解冻彻底破冻结 8.65 天花板(~4.6×)，但**上限远低于 SOTA(60-72 all-query)**。
- 结论收敛新颖性 agent 判断：这是 **building block 不是 standalone method**。后续价值在机制重组(LoRA↔visibility)或问题 reframe(common-visible support/reliability)，不在继续刷这条曲线。
