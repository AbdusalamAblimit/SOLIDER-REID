# 实验 exp294: Base Full-GCN (LGPA-only) + 2-stage PSG on Occ-Duke, seed 41

## 动机

Phase 3-C 消融了 Tiny/Small 上的 "Full - GCN" 对照 (exp286-289), 得出结论:
- Tiny: GCN 基本 0 贡献 (exp287 65.9/77.0 ≈ exp261 Full 65.9/77.4)
- Small: GCN 微贡献 (exp289 73.8/83.3 vs exp285b Full 74.0/84.1, Δ=-0.2/-0.8)

**缺口**: Base backbone 上 "Full - GCN" 对照从未做。用户指示在 lab4090 上跑 **Base + Full-GCN + seed 41** (目前最强 seed), 看能否达到 SOTA (exp263d Base Full s41 = 74.1/83.3 / MaxSim 75.2/84.8)。

**假设**: Base + LGPA + OA-SD + ParAug + PLBOA + 2-stage PSG (无 GCN) ≈ Base Full (带 GCN)。若成立, 证明 GCN 在所有 backbone 容量下都冗余, 简化模型。

## 核心假设

一句话: **Base 上 GCN 是冗余的, LGPA 已捕获 semantic part 结构。**

## 技术方案

### 修改文件
无新代码修改。仅 CLI override 关 GCN。

### 数据流 (无 GCN)
1. Swin-Base backbone + 2-stage PSG at `[-2,-1]`
2. LGPA 模块从 detached features 生成 semantic part features
3. OA-SD 蒸馏 student/teacher
4. PLBOA lower body 遮挡增强
5. ParAug 动态遮挡
6. **无 GCN branch** (POSE_SKELETON_GCN=False)
7. Eval: eq_concat (global + LGPA parts) + flip-test

### 关键超参数
- `SOLVER.SEED`: 41 (对齐 exp263d best seed)
- `MODEL.POSE_SKELETON_GCN`: **False** (关 GCN, 唯一变量)
- `MODEL.POSE_PSG_STAGES`: `[-2,-1]` (保持 2-stage)
- `MODEL.POSE_LGPA`: True
- `MODEL.POSE_OA_SD`: True
- `MODEL.POSE_PARALLEL_AUG`: True
- `MODEL.POSE_LOWER_BODY_OCC`: True
- `TEST.IMS_PER_BATCH`: 64 (防 Base eval OOM)

## CLI 配置

```bash
cd /home/afr/SOLIDER-REID
PYTHONUNBUFFERED=1 nohup /usr/local/anaconda3/envs/mmpose-abu/bin/python train.py \
  --config_file configs/occluded_duke/prcv_best_base.yml \
  SOLVER.SEED 41 \
  MODEL.POSE_SKELETON_GCN False \
  TEST.IMS_PER_BATCH 64 \
  OUTPUT_DIR /home/afr/SOLIDER-REID/log/occluded_duke/exp294_lgpaOnly_2stg_b_od_s41 \
  > /tmp/exp294.log 2>&1 &
```

## 预期结果

### 成功标准 (假设成立)
- FINAL mAP 73.8-74.2 / R1 82.5-83.5 (≈ exp263d 74.1/83.3)
- MaxSim 75.0-75.5 / 84.5-85.0 (≈ exp263d MaxSim 75.2/84.8)
- 证明 Base 上 GCN 也冗余, 论文可 claim "LGPA alone 足够, GCN 可去"

### 失败情况 (假设不成立)
- FINAL mAP 72.5-73.5 / R1 81-82 (显著低于 exp263d)
- 证明 Base + GCN 组合有增益 (与 Small 一致), GCN 不能去
- 论文 Base OD 主配置必须保留 GCN

## 对照组

**同 seed 41 Base OD 矩阵 (Full vs Full-GCN)**:

| Exp | seed | Full-GCN | mAP/R1 (eq+flip) | mAP/R1 (MaxSim+flip) |
|-----|------|----------|------------------|---------------------|
| exp263d | 41 | Full | 74.1/83.3 | 75.2/84.8 |
| **exp294 (本)** | **41** | **Full-GCN** | **pending** | **pending** |

**跨 backbone 横向比较 (Full-GCN)**:

| Backbone | 1-stg | 2-stg |
|----------|-------|-------|
| Tiny | exp286 66.0/76.6 | exp287 65.9/77.0 |
| Small | exp288 73.8/83.8 | exp289 73.8/83.3 |
| **Base (本 exp294)** | — | **pending** |

## 消融变量隔离性

**单变量**: POSE_SKELETON_GCN True → False
其他与 exp263d 完全相同: Base + 2-stage PSG + LGPA + OA-SD + ParAug + PLBOA + seed 41 + Occ-Duke

## 输出

- 机器: lab4090 (24GB 4090, mmpose-abu env)
- 预计 speed: 4.2 min/epoch (对齐 exp263b)
- 总训练时长: ~8h 30min
- ETA: tmr ~02:30 CST

## 论文价值

若 Full-GCN Base ≈ Full Base (SOTA), 补 ablation 主表:
- Claim: **GCN 在所有 backbone cap 下冗余** (Tiny/Small/Base 3 个 backbone 统一结论)
- 简化模型 → efficiency 章节的卖点
- 验证 "LGPA 已经隐式捕获 pose 结构, GCN branch 多余"

若显著下降, 作反例:
- Small/Base 分界点: Base 受益于 GCN (容量大, 能吸收额外信号)
- 论文叙事调整: "Small 简化模型, Base 保留 GCN"
