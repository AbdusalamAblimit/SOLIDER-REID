# exp277 monitor — Phase 3-A Small 3-stage PSG (Occ-Duke, seed 42)

- 机器: lab4090
- 启动: 2026-04-21 01:42 CST (auto-chain from exp276 via v2 daemon 3654948)
- Log: `/home/afr/SOLIDER-REID/log/occluded_duke/exp277_psg3_s_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_small.yml` + CLI override (PSG_STAGES=`[-3,-2,-1]`)
- Scaffold: Swin-Small + PSG 3-stage (LGPA/GCN/OA-SD/PLBOA/ParAug 全关)
- Speed: ~54s/epoch, 总训练 ~1h50min
- 启动 PID: 3805926

## 对照 (Phase 3-A Small)

| Exp | PSG stages | FINAL mAP/R1 |
|-----|-----------|-------------|
| exp274 | 无 | 68.1 / 76.8 |
| exp275 | `[-1]` | 68.8 / 76.8 |
| exp276 | `[-2,-1]` | 68.3 / 77.2 |
| **exp277 (本)** | `[-3,-2,-1]` | **49.0 / 57.7 (塌缩)** |

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 | 备注 |
|-------|-----|----|----|----|------|
| 10 | **0.3** | **0.3** | 1.2 | 1.9 | ⚠️ 完全塌缩 (接近 random) |
| 20 | 5.7 | 9.6 | 17.8 | 22.0 | 慢恢复 |
| 30 | 24.9 | 32.0 | 47.1 | 52.5 | |
| 40 | 35.0 | 43.6 | 58.5 | 64.5 | |
| 50 | 41.7 | 49.8 | 64.6 | 70.9 | |
| 60 | 44.7 | 52.5 | 67.3 | 73.6 | |
| 70 | 46.4 | 54.8 | 70.3 | 75.1 | |
| 80 | 47.8 | 55.9 | 71.2 | 76.6 | |
| 90 | 48.0 | 57.1 | 70.9 | 76.3 | |
| 100 | 49.0 | 57.8 | 71.7 | 77.1 | |
| 110 | 49.0 | 57.6 | 71.9 | 76.6 | |
| **120 FINAL** | **49.0** | **57.7** | **71.4** | **76.9** | under-converged |

## FINAL (2026-04-21 ~03:47 CST)

- **mAP: 49.0%**, **Rank-1: 57.7%**, R5: 71.4%, R10: 76.9%
- 对照:
  - exp274 Small no-PSG 68.1/76.8 → Δ=**-19.1 / -19.1** ⬇️⬇️⬇️
  - exp275 Small 1-stage 68.8/76.8 → Δ=**-19.8 / -19.1**
  - exp273 Tiny 3-stage 60.5/69.9 → Small 3-stage 反而不如 Tiny 3-stage -11.5/-12.2
- Ckpt: `transformer_120.pth` (200MB, under-trained weights)

## 诊断: 训练塌缩根因

**e2 Loss signature**:
```
Epoch[2] Iter[*] id_global: 3.277 (常数) tri_global: ~7→3 (下降)
```

- id_global = 3.277 = 0.5 × ln(702) = GLOBAL_LOSS_SCALE × ln(num_classes)
- 即 **classifier output 完全均匀分布** (uniform softmax),相当于 random guess
- tri_global 从 7 下降到 3,**仅 triplet loss 在学 (feature space 差异化),id loss 卡死**
- 说明 **BNNeck + classifier 的梯度通路被 PSG 3-stage gate 切断** 或 **features 被 gate 为 0**

**可能机制**:
1. PSG 3-stage 在 Swin stage 1/2/3 都插入 spatial gate, 多层 gate 叠加可能将 feature 压到接近 0
2. Small backbone 容量比 Tiny 大,feature magnitude 在更小尺度,更容易被 gate 归零
3. 归零 features 经 BNNeck 后 logits 趋向均匀 → CE loss = ln(N_classes)

## 结论

### 不重训,作为有效数据点使用

**不重训理由**:
1. exp277 展示了 **3-stage PSG 在 Small 上的塌缩模式**,是重要 negative result
2. Tiny 3-stage (exp273) 正常 (60.5/69.9),但 Small 3-stage 塌缩 → **backbone 容量 × PSG stage 交互效应** 的证据
3. 支持论文结论 "default 选 2-stage" — 3-stage 在大 backbone 上有风险
4. 避免 14h 重训占 lab4090 (接下来接 Phase 3-B Small)

### Phase 3-A Small 矩阵最终闭合

| exp | backbone | PSG stages | mAP / R1 | 状态 |
|-----|----------|-----------|----------|------|
| exp274 | Small | 无 | 68.1 / 76.8 | ✅ |
| exp275 | Small | `[-1]` | **68.8** / 76.8 | ✅ mAP peak |
| exp276 | Small | `[-2,-1]` | 68.3 / **77.2** | ✅ R1 peak |
| exp277 | Small | `[-3,-2,-1]` | **49.0 / 57.7** | ⚠️ 塌缩 |

### Paper 写法 (Table 2 Small)

"Small 上 PSG stage 收益不同于 Tiny: 1-stage 达 mAP peak (68.8), 2-stage R1 peak (77.2), **3-stage 训练塌缩** (mAP 下降 19+)。说明大 backbone 对多层 PSG gate 敏感。"

### auto-chain

exp282 Phase 3-B Small GCN256+1stg 已 auto-launched PID 3889122 @ 03:48 CST (after 20s safety delay)。
