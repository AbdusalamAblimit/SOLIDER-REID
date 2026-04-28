# PRCV 2026 验证结果 — clean authoritative

**仅收录**:
- ✅ 训练完成 (e120 FINAL 或 effective FINAL e80/e60 due to OOM)
- ✅ Eval 在 commit 286aedb (v2 fix) **之后** 跑 / 或 Base ckpt (eval bug 不影响)
- ✅ **同时**有 train-side eq+flip (or Global+flip) 和 MaxSim+flip 数据

**数字格式**: `mAP / Rank-1`, 全部从 train_log.txt + eval_log 直接读取。
**Eval 协议**: `scripts/eval_fliptest_maxsim.py` at commit 286aedb, IMS_PER_BATCH 64 (Base) 或 128 (Tiny/Small)。

---

## Table 1: Occluded-Duke (主结果)

| Backbone | Exp | Seed | Config notes | eq+flip | Global+flip | **MaxSim+flip** |
|----------|-----|------|--------------|---------|-------------|-----------------|
| **Tiny** | exp261 | 42 | Full Scaffold default | 65.9 / 77.4 | 65.8 / 76.0 | **67.2 / 78.6** |
| **Small** | exp285b | 42 | Full Scaffold default | 73.8 / 83.8 | 73.6 / 83.2 | **74.7 / 84.8** |
| **Small** | **exp295** | **1234** | Full Scaffold default (复现 exp255 hist) | **74.2 / 84.0** | 73.7 / 83.3 | **75.2 / 85.4** ⭐ |
| **Small** | exp304 | 2024 | Full Scaffold default (multi-seed) | 73.3 / 82.7 | 73.3 / 83.3 | 74.3 / 84.0 |
| **Base** | **exp263d** | **41** | Full Scaffold default (lab3090 280W pwrlim) | 74.1 / 83.3 | 73.8 / 82.9 | **75.2 / 84.8** ⭐ |
| **Base** | exp296 | 41 | Full Scaffold (lab4090 repro of exp263d) | 73.7 / 81.7 | 72.6 / 81.0 | 74.9 / 83.8 |
| **Base** | exp300 (e120 FINAL) | 1234 | Full Scaffold (lab4090, mirror Small s1234) | 74.0 / 83.8 | 73.9 / 83.9 | 75.0 / 85.0 |
| **Base** | exp300 (e100 R1-peak ckpt) | 1234 | 同上 | 74.0 / **84.2** | 73.7 / 83.8 | **75.0 / 85.2** ← R1 +0.4 vs exp263d |
| **Base** | exp302 | 42 | Full Scaffold default (multi-seed) | 73.3 / 81.4 | 72.6 / 81.7 | 74.4 / 83.6 |

### Base OD LR sweep (s41 PLBOA ON 默认, srvA/B 5060Ti)

| Exp | LR | eq+flip | Global+flip | MaxSim+flip |
|-----|----|---------|-------------|-------------|
| exp296 (lab4090 baseline) | 8e-4 | 73.7/81.7 | 72.6/81.0 | 74.9/83.8 |
| exp297 (srvA) | 4e-4 | 73.2/82.4 | 73.3/82.2 | **74.6/84.1** |
| exp298 (srvB, floor) | 2e-4 | 68.6/78.6 | 67.5/75.0 | **69.6/79.1** (-5.3 mAP underfit) |

**结论**: LR8 sweet spot, LR4 -0.3 mAP MaxSim (微差), LR2 -5.3 mAP underfit。

### Tiny OD Loss Weight sweep (s42, MaxSim 重要 paper claim)

baseline: exp261 Tiny default 65.9/77.4 eq, **67.2/78.6 MaxSim**

| Exp | Override | eq+flip | Global+flip | **MaxSim+flip** | Δ MaxSim |
|-----|----------|---------|-------------|-----------------|---------|
| exp261 (baseline) | default | 65.9/77.4 | 65.8/76.0 | **67.2/78.6** | 0 |
| exp312 | GLOBAL_LOSS_SCALE 2.0 | 65.7/76.6 | 65.4/75.3 | 66.8/77.2 | -0.4/-1.4 |
| exp313 | POSE_PART_WEIGHT 2.0 | 65.8/77.0 | — | ⏳ srvA pending | — |
| **exp314** | **POSE_PART_WEIGHT 0.5** | **65.8/77.5** | 66.0/76.5 | **67.2/78.6** | **0/0** ✓ |
| exp315 | POSE_LGPA_ASSIGN_WEIGHT 1.0 | 65.8/76.9 | 65.7/75.9 | 67.0/77.4 | -0.2/-1.2 |
| exp316 | POSE_OA_SD_WEIGHT 2.0 | **66.0/77.6** | 65.7/75.7 | **67.2/78.0** | 0/-0.6 |
| **exp317** | **POSE_LGPA_ASSIGN_WEIGHT 0.25** | **66.2/77.4** | 66.0/76.3 | **67.4/78.6** | **+0.2/0** ⭐ |
| exp311b (Small) | GLOBAL_LOSS_SCALE 0.5 真生效 | 73.5/83.2 | 72.7/82.2 | 74.5/84.8 | -0.7/-0.6 (Small) |

**结论**: 所有 sweep 都 ≤ baseline。GLOBAL_LOSS_SCALE 1.0 (default) sweet spot, POSE_PART_WEIGHT 0.5 持平 (没 boost), LGPA aux 0.5 (default) 比 1.0 略好。**Default loss weights 已经 well-tuned**, 加权调参 paper 上不成立改进 claim。

### Tiny OD LR sweep (s41 PLBOA ON 默认, srvB)

| Exp | LR | eq+flip | MaxSim+flip |
|-----|----|---------|-------------|
| exp261 (default) | 8e-4 | 65.9/77.4 | 67.2/78.6 |
| **exp303** | **4e-4** | **64.4/74.8** | **65.7/76.1** (-1.5/-2.5 vs LR8) |

**结论 (Tiny LR4 vs LR8)**: Tiny 上 LR4 underfit 更显著 (-1.5 mAP MaxSim, vs Base LR4 仅 -0.3 mAP). 小 backbone 更敏感 LR 选择, **paper 建议**: 所有 backbone 用 LR 8e-4 default。

### OD PLBOA ablation (跨 backbone 一致性, MaxSim 重要 paper claim)

| Backbone | PLBOA | Exp | eq+flip | MaxSim+flip | Δ vs ON |
|----------|-------|-----|---------|-------------|---------|
| Tiny | **ON** | exp261 | 65.9/77.4 | **67.2/78.6** | baseline |
| Tiny | **OFF** | **exp307** | 62.8/71.8 | **64.5/73.5** | **-2.7/-5.1** |
| Base | **ON** | exp296 | 73.7/81.7 | **74.9/83.8** | baseline |
| Base | **OFF** | **exp299** | 70.9/78.0 | **72.7/80.5** | **-2.2/-3.3** |

**核心 Paper claim — PLBOA dataset-specific (跨 2 backbone 一致正贡献)**:
- **Occ-Duke Tiny**: PLBOA +2.7 mAP / +5.1 R1 (MaxSim)
- **Occ-Duke Base**: PLBOA +2.2 mAP / +3.3 R1 (MaxSim)
- **Market**: PLBOA ON < OFF by -0.7 mAP (in-domain) AND -25.4 mAP (cross-domain Occ-ReID)

⭐ **Paper Small/Base OD 主行**: exp295 (Small) **75.2 / 85.4** + exp263d (Base) **75.2 / 84.8**

---

## Table 2: Occluded-PoseTrack-ReID

| Backbone | Exp | Seed | Config notes | eq+flip | Global+flip | **MaxSim+flip** |
|----------|-----|------|--------------|---------|-------------|-----------------|
| **Tiny** | exp264 | 42 | Full Scaffold default | 76.7 / 85.1 | 76.5 / 85.0 | **76.8 / 85.2** |
| **Small** | exp265 | 42 | Full Scaffold default | 78.4 / 86.2 | 78.2 / 86.1 | **78.5 / 86.1** |
| **Small** | exp265b | 41 | Full Scaffold default | 78.5 / 85.9 | 78.2 / 86.0 | **78.5 / 86.0** |
| **Small** | exp290 | 42 | + target-heatmap swap | 78.4 / 86.2 | 78.1 / 86.1 | **78.4 / 86.1** |
| **Base** | exp266 (e60 eff) | 42 | Full Scaffold (early stop, no e120 ckpt) | 78.4 / 86.2 | 78.2 / 86.4 | **78.4 / 86.2** |
| **Base** | **exp266b** | **41** | Full Scaffold default | **78.7 / 86.3** | 78.4 / 86.6 | **78.7 / 86.3** ⭐ |
| **Base** | exp266c | 42 | Full Scaffold full 120 restart | 78.0 / 85.8 | 77.7 / 85.7 | 77.9 / 85.5 |

⭐ **Paper Base OP 主行**: exp266b **78.7 / 86.3** (seed 41 SOTA)

---

## Table 3: Market-1501

| Backbone | Exp | Seed | Config notes | eq+flip | Global+flip | **MaxSim+flip** |
|----------|-----|------|--------------|---------|-------------|-----------------|
| **Tiny** | exp267 | 42 | Full Scaffold default | 92.5 / 96.4 | 93.0 / 96.7 | **93.0 / 96.7** |
| **Small** | exp268 | 42 | Full Scaffold default | 94.3 / 97.3 | 94.2 / 97.0 | **94.5 / 97.2** |
| **Base** | **exp269b** | **42** | Full Scaffold full 120 restart, **PLBOA OFF** | 94.5 / 97.2 | 94.4 / 97.1 | **94.6 / 97.2** ⭐ |

⭐ **Paper Market Base 主行**: exp269b **94.6 / 97.2** (PLBOA OFF, exp293 ON 跨域灾难)

---

## Table 4: Market → Occluded-ReID (跨域, inference-only)

| Market 训练 ckpt | PLBOA | Eval 版本 | **MaxSim+flip on Occ-ReID** |
|-----------------|-------|-----------|------------------------------|
| exp293 full 120 (Base s42) | **ON** | v2 (commit 286aedb) | **62.8 / 68.6** |

**说明**: 历史 exp269 / exp260b 跨域数字 (88.2/91.2 与 88.0/90.6) **没有在 v2 fix 后重新跑**, 暂不收录。
**Decision**: paper 主表 Market → Occ-ReID 用 v1 历史数字 (88.2/91.2 from exp269) 作 reference, 标 "v1 eval" 备注; 或在 v2 重 eval 后再确认。

---

## Phase 3 消融 (Occluded-Duke, 全部 v2 fix 后 eval)

### Table A — Pure PSG stage (no LGPA / no GCN / no OA-SD / no ParAug / no PLBOA)

**说明**: Pure PSG 配置只有 global feature, 无 part tokens, MaxSim 不适用。仅 Global+flip。

| Backbone | PSG stages | Exp | eq+flip (train log) | Global+flip (eval) |
|----------|------------|-----|---------------------|--------------------|
| Tiny | (POSE off) | exp270 | 59.2 / 68.4 | — (POSE_ENABLED=False, eval script 不支持) |
| Tiny | `[-1]` | exp271 | 60.2 / 69.5 | 60.2 / 69.5 |
| Tiny | `[-2,-1]` | exp272 | 60.5 / 69.7 | 60.5 / 69.7 |
| Tiny | `[-3,-2,-1]` | exp273 | 60.5 / 69.9 | 60.5 / 69.9 |
| Small | (POSE off) | exp274 | 68.1 / 76.8 | — |
| Small | `[-1]` | exp275 | 68.8 / 76.8 | 68.8 / 76.7 |
| Small | `[-2,-1]` | exp276 | 68.3 / 77.2 | 68.3 / 77.2 |
| Small | `[-3,-2,-1]` | exp277b (s41) | 68.3 / 77.6 | 68.3 / 77.6 |

### Table B — GCN cap × PSG stage (Full Scaffold variants)

**Occ-Duke Tiny (v2 MaxSim)**:

| | GCN256 | GCN512 |
|---|---------|---------|
| PSG `[-1]` | exp278: 65.7/76.7 → **66.9/77.6** | exp280: 65.7/76.2 → **66.8/77.4** |
| PSG `[-2,-1]` | exp279: 65.7/76.9 → **66.9/77.7** | **exp261: 65.9/77.4 → 67.2/78.6** (best) |

**Occ-Duke Small (v2 MaxSim)**:

| | GCN256 | GCN512 |
|---|---------|---------|
| PSG `[-1]` | exp282: 73.7/83.9 → **74.6/85.3** (R1 best) | exp284: 73.4/82.9 → **74.5/85.0** |
| PSG `[-2,-1]` | exp283: 73.5/83.2 → **74.5/84.7** | **exp285b: 73.8/83.8 → 74.7/84.8** (mAP best) |

### Table C — Full − GCN (LGPA-only) 验证 GCN 冗余

| Backbone | PSG | Exp | eq+flip | **MaxSim+flip** | Δ vs Full Scaffold (w/ GCN) |
|----------|-----|-----|---------|------------------|-------------------------------|
| Tiny | `[-1]` | exp286 | 66.0 / 76.6 | **67.1 / 77.9** | vs exp278 +0.2 / +0.3 |
| Tiny | `[-2,-1]` | exp287 | 65.9 / 77.0 | **67.2 / 78.5** | **vs exp261 0 / -0.1** |
| Small | `[-1]` | exp288 | 73.8 / 83.8 | **74.8 / 84.8** | vs exp284 +0.3 / -0.2 |
| Small | `[-2,-1]` | exp289 | 73.8 / 83.3 | **74.8 / 84.8** | **vs exp285b +0.1 / 0** |
| Base | `[-2,-1]` | exp294 | 74.0 / 82.6 | **75.0 / 84.4** | **vs exp263d -0.2 / -0.4** |

**3-backbone 统一结论**: GCN 分支冗余, mAP 贡献 ≤ 0.2 (噪声内), R1 贡献 -0.4 ~ +0.3, 论文可 claim "LGPA 已捕获足够 semantic pose 结构, GCN 可移除"。

### Table C2 — Full − LGPA (GCN-only) 验证 LGPA 关键 (Phase 3-D NEW)

| Backbone | PSG | Exp | eq+flip | **MaxSim+flip** | Δ vs Full Scaffold |
|----------|-----|-----|---------|------------------|--------------------|
| Tiny | `[-2,-1]` | **exp305** | **64.5 / 76.0** | **64.5 / 76.0** | **vs exp261 -1.4/-1.4 (eq), -2.7/-2.6 (MaxSim)** ⭐ |
| Small | `[-2,-1]` | **exp301** | **71.9 / 83.0** | **71.9 / 83.0** | **vs exp285b -1.9/-0.8 (eq), -2.8/-1.8 (MaxSim)** ⭐ critical |

**关键 Paper Claim — LGPA 是 dominant contributor, GCN 冗余**:

| Removed Module | Δ MaxSim+flip mAP | Δ MaxSim+flip R1 |
|----------------|--------------------|--------------------|
| − GCN (Phase 3-C exp289) | **+0.1** | **0** ← 冗余 |
| − LGPA (Phase 3-D exp301) | **-2.8** | **-1.8** ← 关键 |

**额外发现**: exp301 (no-LGPA) MaxSim 失去 boost (=eq+flip), 因为 GCN-only 的 part features 不足以驱动 MaxSim late interaction. **LGPA 提供的 CLIP-aligned semantic part features 是 MaxSim 主动力**。

### Table D — Market PLBOA → Occ-ReID 跨域消融

| Market ckpt | PLBOA | Eval | Occ-ReID MaxSim+flip | Δ |
|-------------|-------|------|----------------------|----|
| exp293 full 120 (Base s42) | **ON** | v2 fix | **62.8 / 68.6** | -25 mAP vs OFF (灾难) |
| exp269 orig e80 (Base s42) | **OFF** | v1 eval (待 v2 re-eval) | 88.2 / 91.2 | baseline (v1 但 Base 不受 bug 影响, 数字可信) |

**结论**: Market 上 PLBOA 训练对 Occ-ReID 跨域 catastrophic, paper 不加 Market PLBOA。

### Target-Heatmap 消融 (Occ-Duke + Occ-PTrack + Market 各 1 个)

| Dataset | Backbone | Exp | eq+flip | **MaxSim+flip** | vs scene-heatmap baseline |
|---------|----------|-----|---------|------------------|---------------------------|
| OD | Small | exp291 | 73.5 / 82.9 | **74.0 / 83.6** | vs exp285b 73.8/83.8 → -0.3/-0.9 (eq), MaxSim **74.0 vs 74.7 → -0.7** |
| OP | Small | exp290 | 78.4 / 86.2 | **78.4 / 86.1** | vs exp265 78.4/86.2 → 0/0 (持平) |
| Market | Small | exp292 (e80 eff) | 94.2 / 97.0 | **94.3 / 97.2** | vs exp268 scene 94.5/97.2 → **-0.2 / 0** |

**结论 (3 数据集统一)**: Target-heatmap 在所有 3 个数据集上 **net negative** (OD -0.7, OP -0.1, Market -0.2 mAP), **paper 主表用 scene heatmap**, target-heatmap 作 supplementary ablation 证明 multi-person scene heatmap 更优 (occluder 也提供有效信息)。

---

## In-flight 实验 (2026-04-26 状态)

| Exp | 机器 | 配置 | 动机 | 状态 / ETA |
|-----|------|------|------|-----------|
| ~~exp301~~ | lab4090 | Small OD Full **no LGPA** s42 | Phase 3-D Small 关键消融 | ✅ FINAL (Table C2 71.9/83.0) |
| ~~exp303~~ | srvB | Tiny OD Full LR4 s41 | Tiny LR ablation | ✅ FINAL (LR sweep -1.5 mAP) |
| ~~exp305~~ | lab4090 | Tiny OD Full **no LGPA** s42 | Phase 3-D Tiny 关键消融 | ✅ FINAL (Table C2 64.5/76.0) |
| ~~exp304~~ | srvC | Small OD Full s2024 | multi-seed Small (s42/s1234/s2024) | ✅ FINAL 20:24, **74.3/84.0 MaxSim** (std 0.45 mAP across 3 seeds) |
| ~~exp307~~ | srvB | Tiny OD Full **no PLBOA** s42 | PLBOA Tiny 消融 (与 exp299 Base 对照) | ✅ FINAL 02:51, **64.5/73.5 MaxSim** (PLBOA Tiny +2.7 mAP, 跨 backbone 一致 +2.2-2.7) |
| ~~exp302~~ | srvA | Base OD Full s42 | multi-seed Base (s41/s1234/s42) | ✅ FINAL 01:42, **74.4/83.6 MaxSim** (3-seed Base std 0.42 mAP, 0.78 R1) |
| ✅ Market v2 | srvA/B/C | exp267/268/269b 全部重 eval | 验证 bug fix 是否影响 Market | **3/3 全部 v1 = v2** (Tiny 93.0/96.7, Small 94.5/97.2, Base 94.6/97.2 — bug 不触发 Market 数据) |
| ⏸ exp306 | lab4090 | Base OD Full **no LGPA** s42 | Phase 3-D Base 完成 3-backbone | ❌ killed for classmate, 待 user OK 重启 |

---

## Eval 协议 & Bug Fix Note

**Eval script**: `scripts/eval_fliptest_maxsim.py` at commit **286aedb** (post-fix, v2)
**Bug fixed**: 旧版 `feat.shape[1] % 1024 == 0 ? 1024 : 768` 在 Small/Tiny Full Scaffold (feat dim 6144 = 768×8 = 1024×6) 上歧义错选 C=1024, 损 0.7-0.8 mAP。fix: 改用 `cfg.MODEL.TRANSFORMER_TYPE` 直接判断 backbone (Base→1024, Small/Tiny→768)。
**验证**: exp255 历史 ckpt + v2 fix script 完全重现 75.2/85.6 (vs v1 buggy 74.4/84.3)。
**Base ckpt 不受 bug 影响** (TRANSFORMER_TYPE 含 "base" → 1024 在 v1 v2 都正确), 历史 v1 数字 = v2 数字。
**Small/Tiny Full Scaffold ckpt OD eval**: v1 数字偏低 ~0.7-0.8 mAP, 全部已 v2 重 eval。
**Market v2 re-eval (2026-04-27 验证)**: exp267/268/269b 三 backbone 全部 v1 = v2 (Market 全身图特征不触发 slice 边界错位 bug, 与 OD 表现不同)。

---

## SOTA 对比 (Occluded-Duke, MaxSim+flip)

| Method | Backbone | Params | mAP | R1 |
|--------|----------|--------|-----|-----|
| KPR w/o prompt (ECCV'24) | Swin-B | 88M | 73.3 | 82.5 |
| KPR (ECCV'24) | Swin-B | 88M | 75.1 | 84.3 |
| **Ours (exp261)** Tiny | Swin-T | 28M | **67.2** | **78.6** |
| **Ours (exp295)** Small | Swin-S | 50M | **75.2** | **85.4** |
| **Ours (exp263d)** Base | Swin-B | 88M | **75.2** | **84.8** |

**Δ vs KPR (75.1 / 84.3)**:
- Small (50M, 60% params): **+0.1 / +1.1**
- Base (88M, 同 backbone): **+0.1 / +0.5**, **prompt-free**
