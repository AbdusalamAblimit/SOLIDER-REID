# Phase 1 主表实验设计（共 9 runs）

## 动机

PRCV 2026 需冲 SOTA 主表。单一训练 scaffold（`exp255` 最强配置）在 3 个 backbone × 3 个训练集上系统评测。

前序依据：
- exp255 (old protocol, 不含 default flip-test): Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA = 73.2/83.3
- exp260b (Base 同上): 73.9/83.2
- P0 改动后，smoke test 上 exp255 ckpt 带 flip = 74.1/83.6（+0.9）

## 核心假设

一句话：**Swin + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA** 在 3 个 backbone × 3 个训练集（OccDuke / OccPTrack / Market）上都能给出有竞争力的 SOTA 数字；默认测试协议含 flip-test。

## 技术方案

### 共享 scaffold（9 个实验完全一致）

```yaml
POSE_BACKBONE_PSG: True
POSE_PSG_STAGES: [-2, -1]        # 2-stage injection (after Stage 2 + Stage 3)
POSE_PFM_HIDDEN: 64

POSE_LGPA: True                  # CLIP-guided 5-part semantic branch
POSE_LGPA_CLIP_DIM: 512
POSE_LGPA_NUM_HEADS: 8
POSE_LGPA_ASSIGN_WEIGHT: 0.5
POSE_LGPA_DETACH: True           # LGPA-D: detached gradient

POSE_SKELETON_GCN: True          # structural pose branch
POSE_GCN_LAYERS: 2
POSE_GCN_HIDDEN: 512

POSE_OA_SD: True                 # Occlusion-Asymmetric Self-Distillation
POSE_OA_SD_WEIGHT: 1.0
POSE_OA_SD_EMA_DECAY: 0.999

POSE_TEST_FEAT: 'equal_concat'
GLOBAL_LOSS_SCALE: 0.5
```

训练超参：
- 120 epoch，SGD cosine，WARMUP_EPOCHS=20，BATCH_SIZE=64
- BASE_LR=8e-4（Small / Base / Tiny 都用，历史 exp260b 证实）
- CHECKPOINT_PERIOD=20（每 20 epoch 存一次）
- EVAL_PERIOD=10（含 flip-test）
- SEED=42

### 9 个实验差异（只变 backbone + dataset）

| Exp ID | Backbone | Dataset | Config | Machine | 备注 |
|--------|----------|---------|--------|---------|------|
| exp261_best_t_od_s42 | Swin-Tiny | OccDuke | `configs/occluded_duke/prcv_best_tiny.yml` | srvB | PLBOA ON |
| exp262_best_s_od_s42 | Swin-Small | OccDuke | `configs/occluded_duke/prcv_best_small.yml` | srvA | PLBOA ON |
| exp263_best_b_od_s42 | Swin-Base | OccDuke | `configs/occluded_duke/prcv_best_base.yml` | local | PLBOA ON |
| exp264_best_t_op_s42 | Swin-Tiny | OccPTrack | `configs/occluded_posetrack/prcv_best_tiny.yml` | srvC | PLBOA ON |
| exp265_best_s_op_s42 | Swin-Small | OccPTrack | `configs/occluded_posetrack/prcv_best_small.yml` | srvC→ | PLBOA ON |
| exp266_best_b_op_s42 | Swin-Base | OccPTrack | `configs/occluded_posetrack/prcv_best_base.yml` | local→ | PLBOA ON |
| exp267_best_t_m_s42 | Swin-Tiny | Market | `configs/market/prcv_best_tiny.yml` | srvB→ | PLBOA OFF |
| exp268_best_s_m_s42 | Swin-Small | Market | `configs/market/prcv_best_small.yml` | srvA→ | PLBOA OFF |
| exp269_best_b_m_s42 | Swin-Base | Market | `configs/market/prcv_best_base.yml` | local→ | PLBOA OFF |

数据流：训练 → 每 10 epoch eval（flip-test ON）→ 120 epoch 后 ckpt_120.pth → test.py 带 flip 出主结果行 → 再跑 `eval_fliptest_maxsim.py` 出 MaxSim 行。

## 预期结果

| Backbone | Occ-Duke | Occ-PTrack | Market | OR ← Market |
|----------|---------|-----------|--------|-------------|
| Tiny | ≥60/72 | — | ≥90/95 | ≥80/84 |
| Small | ≥74/83 | ≥65/72 | ≥94/97 | ≥86/88 |
| Base | ≥74/83 | ≥65/72 | ≥94/97 | ≥86/88 |

如果主表某格明显低于预期 → 检查数据加载 / flip 逻辑 / hyperparameter 不同点。

## 对照组

- 旧 exp255/exp260b（无默认 flip-test）作为历史对照
- 同 scaffold 的 Phase 3 消融（PSG stage / GCN / LGPA 变体）

## 单变量原则

Phase 1 的 9 个实验每两格之间**差至少两个变量**（backbone 或 dataset）。不是单变量消融，而是 SOTA 主表覆盖。单变量消融留 Phase 3。

## 风险

- Tiny + Occ-PTrack: Tiny 容量小，Occ-PTrack 规模大（>30K 训练图）可能欠拟合。若 < 60% mAP，改用 Small/Base。
- Market PLBOA 关闭：一致性需验证 PLBOA=False 在 Market 上不劣于 True。
- Base 14h × 3 runs = 42h，local 3090 慢但比 5060Ti 快 1.3-1.5x。
