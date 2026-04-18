# Phase 3 消融实验设计（共 20 runs，可选 4 runs）

> 待 Phase 1 Tiny/Small 6 run 全部完成后启动。每个 exp 创建独立目录并引用本文档作为共享设计。

## 动机

Phase 1 拿到主表后，主创新 `PSG` 和结构补充 `GCN` 的证据需要干净的消融来支撑论文 method + ablation 两节。旧 multi-stage PSG 结果里 stage / 容量 / scaffold 三个变量混杂，没法直接搬进论文。本 Phase 就是把这三个消融重跑干净。

## 核心问题（依次回答）

1. **PSG 本体**在纯 backbone setting 下是否稳定正增益？（Phase 3-A）
2. `2-stage PSG` 是不是**高容量 GCN 分支**的必要条件？（Phase 3-B）
3. `2-stage PSG` 的收益是偏 structural branch 还是 semantic branch 也吃？（Phase 3-C, optional）

## Phase 3-A — PSG stage 消融（8 runs, priority 1）

**scaffold（纯 PSG，全部无 LGPA / GCN / OA-SD / PLBOA）**:

```yaml
POSE_BACKBONE_PSG: <stage 数> # 由 PSG_STAGES 决定
POSE_PSG_STAGES: <变量>
POSE_LGPA: False
POSE_SKELETON_GCN: False
POSE_OA_SD: False
POSE_LOWER_BODY_OCC: False
POSE_PARALLEL_AUG: False
POSE_TEST_FEAT: 'global'       # 无 branch, 直接走 global
GLOBAL_LOSS_SCALE: 0.5          # 维持
```

### 矩阵

| Exp ID | PSG_STAGES | Backbone | 期望 mAP (OD) | 机器 |
|--------|-----------|----------|--------------|------|
| exp270_psg0_t_od_s42 | 不启用 | Tiny | ~56 (baseline) | srvB |
| exp271_psg1_t_od_s42 | `[-1]` | Tiny | ~58 (PSG stage 3) | srvB |
| exp272_psg2_t_od_s42 | `[-2,-1]` | Tiny | ~58-59 | srvB |
| exp273_psg3_t_od_s42 | `[-3,-2,-1]` | Tiny | 不确定，历史持平或弱 | srvB |
| exp274_psg0_s_od_s42 | 不启用 | Small | ~65 | srvA |
| exp275_psg1_s_od_s42 | `[-1]` | Small | ~67 | srvA |
| exp276_psg2_s_od_s42 | `[-2,-1]` | Small | ~67-68 | srvA |
| exp277_psg3_s_od_s42 | `[-3,-2,-1]` | Small | 不确定 | srvA |

若 `2-stage > 1-stage > no-PSG` 在 Tiny + Small 都稳定成立 → PSG 本体 + stage 数选择的消融扎实。  
若 `2-stage ≈ 1-stage`（Tiny 可能出现）→ 写作中明确这点，把 2-stage 的收益定位在"高容量结构分支场景"。

### 启动方式

基于 `prcv_best_tiny.yml / prcv_best_small.yml`，用 CLI override 关闭其他模块:

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_tiny.yml \
  MODEL.POSE_BACKBONE_PSG False \
  MODEL.POSE_LGPA False \
  MODEL.POSE_SKELETON_GCN False \
  MODEL.POSE_OA_SD False \
  MODEL.POSE_LOWER_BODY_OCC False \
  MODEL.POSE_PARALLEL_AUG False \
  MODEL.POSE_TEST_FEAT 'global' \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp270_psg0_t_od_s42
```

以此类推。PSG 打开时用 `MODEL.POSE_BACKBONE_PSG True MODEL.POSE_PSG_STAGES "[-1]"` 等。

### 时间预算

Tiny 单 run ~8h on 5060 Ti, Small ~14h。8 runs 两机并行:
- srvB: 4 × Tiny = 32h
- srvA: 4 × Small = 56h
- srvC 空闲 → 可拉去跑 Phase 3-B

## Phase 3-B — GCN capacity × PSG stage（8 runs, priority 2）

**scaffold（full, 仅改 GCN_HIDDEN + PSG_STAGES）**:

基于 `prcv_best_{t,s}.yml`，其余模块全开（LGPA-D + GCN + OA-SD + PLBOA）。

### 矩阵

| Exp ID | GCN_HIDDEN | PSG_STAGES | Backbone | 机器 |
|--------|-----------|------------|----------|------|
| exp278_gcn256_1stg_t_od_s42 | 256 | `[-1]` | Tiny | srvC |
| exp279_gcn256_2stg_t_od_s42 | 256 | `[-2,-1]` | Tiny | srvC |
| exp280_gcn512_1stg_t_od_s42 | 512 | `[-1]` | Tiny | srvC |
| **exp281_gcn512_2stg_t_od_s42** | 512 | `[-2,-1]` | Tiny | — | ≡ Phase 1 exp261，共享不重跑 |
| exp282_gcn256_1stg_s_od_s42 | 256 | `[-1]` | Small | srvC |
| exp283_gcn256_2stg_s_od_s42 | 256 | `[-2,-1]` | Small | srvC |
| exp284_gcn512_1stg_s_od_s42 | 512 | `[-1]` | Small | srvC |
| **exp285_gcn512_2stg_s_od_s42** | 512 | `[-2,-1]` | Small | — | ≡ Phase 1 exp262，共享不重跑 |

共享 2 个（exp281/285 = Phase 1），实际新跑 6 runs。

希望回答：`exp255 vs exp255b` 观察到的「高容量 GCN 下 2-stage PSG 更优」是否跨 backbone 稳定。  
若 Tiny/Small + GCN512 都呈现 `2-stage > 1-stage`，而 GCN256 上差距缩窄 → 明确写「2-stage PSG 是高容量结构分支的必要条件」。

### 启动方式

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_tiny.yml \
  MODEL.POSE_GCN_HIDDEN 256 \
  MODEL.POSE_PSG_STAGES "[-1]" \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp278_gcn256_1stg_t_od_s42
```

### 时间预算

srvC 单机 6 × 8-14h = 60h（Tiny 4 runs × 8h + Small 2 runs × 14h + overlap）。实际 6 runs on srvC 跑 ~45-50h（可与 A/B 的 Phase 3-A 同步开跑）。

## Phase 3-C — Semantic branch 依赖性消融（4 runs, optional）

基于 `prcv_best_{t,s}.yml`，只开 LGPA、关 GCN。

### 矩阵

| Exp ID | 语义分支 | 结构分支 | PSG_STAGES | Backbone | 状态 |
|--------|---------|---------|-----------|----------|------|
| exp286_lgpaOnly_1stg_t_od_s42 | LGPA-D | × | `[-1]` | Tiny | 可选 |
| exp287_lgpaOnly_2stg_t_od_s42 | LGPA-D | × | `[-2,-1]` | Tiny | 可选 |
| exp288_lgpaOnly_1stg_s_od_s42 | LGPA-D | × | `[-1]` | Small | 可选 |
| exp289_lgpaOnly_2stg_s_od_s42 | LGPA-D | × | `[-2,-1]` | Small | 可选 |

仅在 Phase 3-A/B 全部跑完后时间允许时再排。

### 启动方式

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_tiny.yml \
  MODEL.POSE_SKELETON_GCN False \
  MODEL.POSE_PSG_STAGES "[-1]" \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp286_lgpaOnly_1stg_t_od_s42
```

## 预期输出（供论文 Method/Ablation 引用）

1. **Table 2 (ablation)**: PSG stage 选择（Phase 3-A）
2. **Table 3 (ablation)**: GCN 容量 × PSG stage 交互（Phase 3-B）
3. **Table 4 (ablation, optional)**: Semantic branch 是否吃 2-stage 收益（Phase 3-C）

Table 2 和 Table 3 是论文硬素材，Table 4 有则加、无则在文字中简述。

## 与 Phase 1 的共享

- **exp281 = exp261**（Tiny Phase 1 scaffold），**exp285 = exp262**（Small Phase 1 scaffold）。Phase 3-B 的四个格子里这两个直接共享 Phase 1 ckpt，不重跑。
- Phase 3-A 的 exp270（no-PSG + 关一切）不是 `exp260b / exp262` 的 baseline，它是一个**纯 baseline**（SOLIDER + default data aug）。严格来说和 `4090-OD-PSG-small-lr8` 系列 baseline 一致。已有历史数据 `baseline ≈ 56/66 for Tiny, ~65/76 for Small`，但若时间允许可以重跑一次在 5060 Ti 上拿一致硬件基线。

## 风险与降级方案

- 若 Phase 3-A 就发现 `PSG 本体已经不稳`（多 seed 下 PSG != > no-PSG） → 论文主故事需重写，但这种可能性极低（`exp007` 3-seed 已强证明）
- 若 Phase 3-B 结果发现 `GCN512 + 2-stage` 并不比 `GCN512 + 1-stage` 强（即旧 exp255 vs exp255b 是方差） → 把 `2-stage` 在 method 部分位置从"scalable extension"降级为"default setting"，但仍保留
- 若时间不够跑完 8 runs 的 Phase 3-B → 至少必须跑完 `GCN512 1-stage vs 2-stage` 两组（Tiny + Small）= 4 runs，这是回答 `exp255 vs exp255b` 的最小闭环

## 审查要求

每个 Phase 3 exp 启动前必须有：
- `experiments/exp{NNN}/design.md`（引用本 phase3_design.md）
- `experiments/exp{NNN}/claude_review.md`（含"审查通过" + ≥30 行）
- `experiments/exp{NNN}/codex_review.md`（`/codex:review` 产出，verdict approve）

Phase 3-A/B/C 中若配置的 delta 只是 CLI override，review 可集中说明"与共享 scaffold 的唯一差异"，不重复 backbone / loss / optimizer 细节。
