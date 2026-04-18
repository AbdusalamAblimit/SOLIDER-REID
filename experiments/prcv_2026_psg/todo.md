# PRCV 2026 PSG 实验 TODO（主表优先版）

**Deadline**: 2026-04-30（今 4/17，12 天）
**策略**：主表 → 立刻开始写论文 → 消融/补充与写作并行

---

## 约束

- 所有旧实验结果**全部作废**（新 default eval 含 flip-test）
- 新默认测试协议 = `equal_concat + flip-test`
- MaxSim 只在训练结束后用 final checkpoint 跑一次（单独一行）
- 单 seed 先跑完矩阵；后期多 seed **必须在同机器上**（控硬件方差）
- log 和 experiments 目录都带 seed 后缀：`exp{NNN}_{desc}_s{seed}`
- 最强配置：Swin-{T/S/B} + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA

---

## 4 台机器

| 机器 | GPU | python | 磁盘 | 数据 |
|------|-----|--------|------|------|
| local | RTX 3090 24G | torch 1.13 conda | 982G free | 全 ✓ |
| srvA | 5060 Ti 16G | torch 2.9 system | /root 14G free, /hy-tmp 50G | OD ✓, 其他按需 OSS 拉 |
| srvB | 5060 Ti 16G | 同 | /root 27G, /hy-tmp 31G free | 全 ✓ |
| srvC | 5060 Ti 16G | 同 | 同 | 全 ✓ |

---

## Phase 0 — 代码改造（1-2h）

- [ ] **P0-1**：`config/defaults.py` 加 `_C.TEST.FLIP_TEST = True`
- [ ] **P0-2**：`processor/processor.py::do_inference` 加 flip-test 分支（L2-norm orig + flip 后取均值）
- [ ] **P0-3**：`processor/processor.py::do_train` 的 eval 路径同样加 flip-test（训练中间 eval 也含）
- [ ] **P0-4**：确认 MaxSim 只在 `scripts/eval_fliptest_maxsim.py`，train/test eval 不混入
- [ ] **P0-5**：Claude 广范围审查代码改动
- [ ] **P0-6**：`/codex:review` 审查，approve 后 commit
- [ ] **P0-7**：commit `p0_fliptest_default`，push

---

## Phase 1 — 主表 9 runs（冲 SOTA，1.5-2 天）

3 backbones × 3 train sets，最强配置：

| # | Exp ID | Train | Backbone | 机器 | 预计时长 | 状态 |
|---|--------|-------|----------|------|---------|------|
| 1 | exp261_best_t_od_s42 | Occ-Duke | Tiny | srvB | 8h | RUNNING e106/120 |
| 2 | exp262_best_s_od_s42 | Occ-Duke | Small | srvA | 14h | RUNNING e70/120 |
| 3 | exp263_best_b_od_s42 | Occ-Duke | Base | srvA/B/C 任一 | 17h @ with_cp | **DEFERRED**（Tiny/Small 完成后排入） |
| 4 | exp264_best_t_op_s42 | Occ-PTrack | Tiny | srvC | 8h | RUNNING e83/120 |
| 5 | exp265_best_s_op_s42 | Occ-PTrack | Small | srvC | 14h (接 #4) | QUEUED |
| 6 | exp266_best_b_op_s42 | Occ-PTrack | Base | srvA/B/C 任一 | 17h @ with_cp | **DEFERRED**（Tiny/Small 完成后排入） |
| 7 | exp267_best_t_m_s42 | Market | Tiny | srvB | 8h (接 #1) | QUEUED |
| 8 | exp268_best_s_m_s42 | Market | Small | srvA | 14h (接 #2) | QUEUED |
| 9 | exp269_best_b_m_s42 | Market | Base | srvA/B/C 任一 | 17h @ with_cp | **DEFERRED**（Tiny/Small 完成后排入） |

> **2026-04-19 更新**：本地 3090 挂了，Base 3 个 run 全部推迟（decision C）。但用户确认 `MODEL.WITH_CP: True`（已在 prcv_best_base.yml 打开）下 Base 显存只 6–8 GB，5060 Ti 16G 完全够。当前三台正在跑 Tiny/Small 6 run，完成后 Base 3 run 并排进 Phase 1。论文主表 Base 行短期仍可引 `exp260b = 73.9/83.2`（旧协议）。

**Wall time 估算**（不含 Base）：srvA 28h，srvB 16h，srvC 22h。最慢 srvA ~1.2 天。

**每 ckpt 测试**：
- `*_od_*` → test Occ-Duke + Occ-ReID cross
- `*_op_*` → test Occ-PTrack
- `*_m_*` → test Market + **Occ-ReID cross (主表 OR 那列用这个)**

**主表输出**（从 9 个 ckpt 的 log + MaxSim eval）：

| Backbone | Occ-Duke (mAP/R1) | Occ-PTrack (mAP/R1) | Market (mAP/R1) | Occ-ReID ← Market (mAP/R1) |
|----------|-------------------|---------------------|-----------------|-------|
| Tiny | exp261 | exp264 | exp267 | exp267 |
| Small | exp262 | exp265 | exp268 | exp268 |
| Base | exp263 | exp266 | exp269 | exp269 |

每格 2 行：`Ours (eq+flip) / Ours+MaxSim`。

### Phase 1 任务清单

- [ ] 写 P1 总 design.md（9 exp 共享）
- [ ] 9 个实验分别建目录 + claude review + codex review
- [ ] 启动 Phase 1 训练（按机器分配）
- [ ] Monitor 流事件跟所有 log
- [ ] 每训练完一个：monitor.md 追加 final + MaxSim eval 补一次
- [ ] 9 个都完 → 装 `main_results.md`

---

## Phase 2 — 论文主体写作（Phase 1 完成即启动，与 P3/P4 并行）

- [ ] 主结果表 `paper_materials/tables/main_results.md`
- [ ] Method section（PSG + GCN + LGPA-D + OA-SD + PLBOA + 测试协议）
- [ ] Abstract + Introduction
- [ ] Related work + KPR 边界
- [ ] 主图（method overview）

---

## Phase 3 — 消融（约 60h / 4 机 = 15h 有效，与写作并行）

只在 Tiny + Small 上（Base 不做消融），主 set = Occ-Duke，跨域 OR 用 OccDuke ckpt 直接 inference：

### A. PSG stage 消融（4 配置 × 2 backbone = 8 runs）

| # | Exp | PSG_STAGES | Backbone | Train |
|---|-----|-----------|----------|-------|
| - [ ] | exp270_psg0_t_od_s42 | 无 | Tiny | OD |
| - [ ] | exp271_psg1_t_od_s42 | `[-1]` | Tiny | OD |
| - [ ] | exp272_psg2_t_od_s42 | `[-2,-1]` | Tiny | OD |
| - [ ] | exp273_psg3_t_od_s42 | `[-3,-2,-1]` | Tiny | OD |
| - [ ] | exp274_psg0_s_od_s42 | 无 | Small | OD |
| - [ ] | exp275_psg1_s_od_s42 | `[-1]` | Small | OD |
| - [ ] | exp276_psg2_s_od_s42 | `[-2,-1]` | Small | OD |
| - [ ] | exp277_psg3_s_od_s42 | `[-3,-2,-1]` | Small | OD |

这组不含 LGPA/GCN/OA-SD/PLBOA（纯 PSG scaffold），回答 "PSG 本体是否稳定"。

### B. GCN capacity × PSG stage（4 配置 × 2 backbone = 8 runs）

| # | Exp | GCN_HIDDEN | PSG_STAGES | Backbone | Train |
|---|-----|-----------|------------|----------|-------|
| - [ ] | exp278_gcn256_1stg_t_od_s42 | 256 | `[-1]` | Tiny | OD |
| - [ ] | exp279_gcn256_2stg_t_od_s42 | 256 | `[-2,-1]` | Tiny | OD |
| - [ ] | exp280_gcn512_1stg_t_od_s42 | 512 | `[-1]` | Tiny | OD |
| - [ ] | exp281_gcn512_2stg_t_od_s42 | 512 | `[-2,-1]` | Tiny | OD |
| - [ ] | exp282_gcn256_1stg_s_od_s42 | 256 | `[-1]` | Small | OD |
| - [ ] | exp283_gcn256_2stg_s_od_s42 | 256 | `[-2,-1]` | Small | OD |
| - [ ] | exp284_gcn512_1stg_s_od_s42 | 512 | `[-1]` | Small | OD |
| - [ ] | exp285_gcn512_2stg_s_od_s42 | 512 | `[-2,-1]` | Small | OD |

full scaffold（LGPA-D + OA-SD + PLBOA + GCN + PSG）。

注：exp281 Tiny 和 exp285 Small 分别等同于主表 exp261 和 exp262，可共享。

### C. LGPA × stage（2 配置新 × 2 backbone = 4 runs，**可选**）

| # | Exp | 语义分支 | PSG_STAGES | Backbone |
|---|-----|---------|-----------|----------|
| - [ ] | exp286_lgpaOnly_1stg_t_od_s42 | LGPA only (no GCN) | `[-1]` | Tiny |
| - [ ] | exp287_lgpaOnly_2stg_t_od_s42 | LGPA only | `[-2,-1]` | Tiny |
| - [ ] | exp288_lgpaOnly_1stg_s_od_s42 | LGPA only | `[-1]` | Small |
| - [ ] | exp289_lgpaOnly_2stg_s_od_s42 | LGPA only | `[-2,-1]` | Small |

### Phase 3 任务

- [ ] Phase 3 每 run 的 design.md + review
- [ ] 20 次训练（8 A + 8 B + 4 C），按机器并行排期
- [ ] 消融表 `paper_materials/tables/ablation.md`
- [ ] 用 OccDuke ckpt 跑 Market + Occ-ReID cross eval（只 inference）→ 补进消融表

---

## Phase 4 — Multi-seed（关键配置 3-seed）

3 个最重要配置补 seed 1234 + 2024，**同机器锁硬件**：

| 配置 | 原机器 | 原 exp | multi-seed exp | 状态 |
|-----|--------|--------|---------------|------|
| - [ ] | srvA | exp262_best_s_od | exp{302}_best_s_od_s1234 / s2024 | 等 Phase 1+3 |
| - [ ] | srvB | exp261_best_t_od | exp{304}_best_t_od_s1234 / s2024 | 等 Phase 1+3 |
| - [ ] | srvC | — (原 Base 推迟) | 若时间允许改跑 Small 补一组 | DEFERRED |

Base multi-seed 因本地 3090 挂同步推迟（2026-04-19 决策 C）。4 runs × 12h = 48h / 2 机 = 24h；若补 srvC 一组 Small 则 6 runs × 12h / 3 机 = 24h。

### Phase 4 任务
- [ ] 6 个 multi-seed 目录 + design（短版 design）
- [ ] 训练
- [ ] 结果并入主表（加 ± std 行）

---

## Phase 5 — 收尾

- [ ] Abstract + Introduction + Figure 1
- [ ] t-SNE / attention / retrieval 可视化（若有时间）
- [ ] Codex 审查全文
- [ ] 投稿

---

## 时间预算（12 天）

| 阶段 | wall time | 累计 |
|------|----------|------|
| P0 代码改造 | 2-3h | 0.1d |
| P1 主表 9 runs | 42h (local 最慢路径) | 1.8d |
| **P2 开始写作** | 并行 | — |
| P3 消融 20 runs | 50h / 4 机 = 13h | 2.3d |
| P4 multi-seed 6 runs | 24h | 3.3d |
| 写作 | 8-10 天（与 P3/P4 并行） | 11-12d |

训练总 wall ~3.3 天。主表 1.8 天后即可开始写论文。12 天从容。

---

## 注意事项

- 每个 exp 启动前：**先写 design.md + claude_review.md + codex_review.md**（hook 强制）
- **Monitor 流事件监控**，不 sleep 轮询
- 训练结束即补 `monitor.md` 最终结果，再跑一次 MaxSim eval
- 数据来源永远 log 文件，不凭记忆
- 同配置不同 seed **必须同机器**
- `srvA` 按需用 OSS 拉 Market / OccReID / Ptrack（现在只有 OccDuke）
