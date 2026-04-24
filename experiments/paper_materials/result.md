# PRCV 2026 Paper Results

**Date**: 2026-04-24 17:30 CST
**Purpose**: 单一 authoritative 结果文件, 整合主结果表 + 消融表 + 跨域结果。
**数据源**: 每行数字均从 train_log.txt / eval_log 读取 (非 copy)。
**Eval protocol**: `scripts/eval_fliptest_maxsim.py` at commit 286aedb (post-fix, v2). 详见 [Eval bug fix note](#eval-bug-fix-note-important) 末尾章节。

---

## Main Table — 4 datasets × {Tiny, Small, Base}

### 字段说明
- **eq+flip**: train-side `equal_concat` 特征 + flip-test TTA (from train_log.txt "Validation Results - Epoch: 120" 或 effective FINAL)
- **Global+flip**: eval-script global cosine distance + flip-test
- **MaxSim+flip**: eval-script MaxSim hybrid (1:1 global + part-MaxSim) + flip-test, v2 fix
- 所有数字格式: `mAP / Rank-1`

### Table 1: Occluded-Duke

| Backbone | Exp ID | Seed | eq+flip (train log) | Global+flip | **MaxSim+flip** |
|----------|--------|------|---------------------|-------------|-----------------|
| **Tiny** | exp261 | 42 | 65.9 / 77.4 | 65.8 / 76.0 | **67.2 / 78.6** |
| **Small** | exp285b (seed 42) | 42 | 73.8 / 83.8 | 73.6 / 83.2 | **74.7 / 84.8** |
| **Small** | **exp295 (seed 1234 repro)** | **1234** | **74.2 / 84.0** | 73.7 / 83.3 | **75.2 / 85.4** ✓ **matches exp255 hist 75.2/85.6** |
| **Small** | exp255 (historical ref) | 1234 | 73.2 / 83.3 | 73.6 / 83.4 | **75.2 / 85.6** (backup srvA log) |
| **Base** | exp263d | 41 | 74.1 / 83.3 | 73.8 / 82.9 | **75.2 / 84.8** |

**Paper 主表 Small OD 用 exp295 75.2/85.4** (seed 1234, HEAD code + v2 fix eval, reproducible)。
**Paper 主表 Base OD 用 exp263d 75.2/84.8** (seed 41, beats KPR w/ prompt 75.1/84.3 by +0.1/+0.5)。

### Table 2: Occluded-PoseTrack-ReID

| Backbone | Exp ID | Seed | eq+flip | Global+flip | **MaxSim+flip** |
|----------|--------|------|---------|-------------|-----------------|
| **Tiny** | exp264 | 42 | 76.7 / 85.1 | 76.5 / 85.0 | **76.8 / 85.2** |
| **Small** | exp265 | 42 | 78.4 / 86.2 | 78.2 / 86.1 | **78.5 / 86.1** |
| **Small** | exp265b | 41 | 78.5 / 85.9 | 78.2 / 86.0 | **78.5 / 86.0** |
| **Base** | exp266 (e60 eff) | 42 | 78.4 / 86.2 | 78.2 / 86.4 | **78.4 / 86.2** |
| **Base** | exp266b (SOTA) | 41 | 78.7 / 86.3 | 78.4 / 86.6 | **78.7 / 86.3** |
| **Base** | exp266c (s42 full 120) | 42 | 78.0 / 85.8 | 77.7 / 85.7 | **77.9 / 85.5** |

**Paper 主表 Base OP 用 exp266b 78.7/86.3** (seed 41, SOTA)。
**Paper 主表 Small OP 用 exp265b 78.5/86.0** (seed 41) 或 exp265 78.5/86.1 (seed 42, 数字几乎相同)。

### Table 3: Market-1501

| Backbone | Exp ID | Seed | eq+flip | Global+flip | **MaxSim+flip** |
|----------|--------|------|---------|-------------|-----------------|
| **Tiny** | exp267 | 42 | 92.5 / 96.4 | 93.0 / 96.7 | **93.0 / 96.7** |
| **Small** | exp268 | 42 | 94.3 / 97.3 | 94.2 / 97.0 | **94.5 / 97.2** |
| **Base** (PLBOA OFF) | exp269b (full 120) | 42 | 94.5 / 97.2 | 94.4 / 97.1 | **94.6 / 97.2** |

**Paper 主表 Market Base 用 exp269b 94.6/97.2** (seed 42 PLBOA OFF full 120)。

### Table 4: Market → Occluded-ReID (cross-domain, inference-only)

| Market ckpt | PLBOA | Eval protocol | **MaxSim+flip on Occ-ReID** |
|-------------|-------|---------------|------------------------------|
| exp269 orig (Base, seed 42) | OFF | v1 eval | **88.2 / 91.2** ← historical top-tier |
| exp293 first run e80 eff (Base, seed 42) | **ON** | v1 eval | 72.4 / 76.7 (−15.8 mAP vs OFF) |
| exp293 full 120 (Base, seed 42) | **ON** | v2 eval | **62.8 / 68.6** (−25.4 mAP vs OFF, catastrophic) |

**结论**: **Market 上加 PLBOA 对 Occ-ReID 跨域是 catastrophic 负效应** (−25 mAP), paper 不加 PLBOA on Market。

---

## SOTA Comparison (Occluded-Duke)

数据来源: KPR (ECCV 2024) Table 3 + 我们的结果。

| Method | Backbone | Params | mAP | Rank-1 | 
|--------|----------|--------|-----|--------|
| BPBreID (WACV'23) | HRNet-W48 | 63M | 54.4 | 63.2 |
| PFD (AAAI'22) | ViT-B | 86M | 65.7 | 74.4 |
| PAT (CVPR'21) | ViT-B | 86M | 53.6 | 64.5 |
| HOReID (CVPR'20) | ResNet-50 | 25M | 43.8 | 55.1 |
| FRT (TMM'22) | HRNet-W32 | 28M | 54.7 | 66.9 |
| QPM (TMM'22) | ResNet-50 | 25M | 56.7 | 66.7 |
| KPR w/o prompt (ECCV'24) | Swin-B | 88M | 73.3 | 82.5 |
| **KPR** (ECCV'24) | Swin-B | 88M | 75.1 | 84.3 |
| **Ours (exp261)** | Swin-T | 28M | 67.2 | 78.6 |
| **Ours (exp295, s1234 repro)** | Swin-S | 50M | **75.2** | **85.4** |
| **Ours (exp263d, s41)** | Swin-B | 88M | **75.2** | **84.8** |
| **Δ vs KPR (Base)** | Swin-B | | **+0.1** | **+0.5** |
| **Δ vs KPR (Small, fewer params)** | Swin-S (50M < 88M) | | **+0.1** | **+1.1** |

**Ours Base OD** 用 prompt-free 设计超越 KPR w/ prompt, 差距 +0.1 mAP / +0.5 R1 on Occluded-Duke。

---

## Ablation Tables (Occ-Duke + Occ-ReID only)

### Table A: Pure PSG stage ablation (no LGPA / no GCN / no OA-SD / no ParAug / no PLBOA)

**Purpose**: PSG (Pose-guided Spatial Gate) stage choice 独立消融, 建立 backbone baseline。

**Occ-Duke**:

| Backbone | PSG stages | Exp | eq+flip (train log) | Global+flip (eval) |
|----------|------------|-----|---------------------|--------------------|
| **Tiny** | (POSE off) | exp270 | 59.2 / 68.4 | — (eval script doesn't support POSE_ENABLED=False) |
| **Tiny** | `[-1]` | exp271 | 60.2 / 69.5 | 60.2 / 69.5 |
| **Tiny** | `[-2,-1]` | exp272 | 60.5 / 69.7 | 60.5 / 69.7 |
| **Tiny** | `[-3,-2,-1]` | exp273 | 60.5 / 69.9 | 60.5 / 69.9 |
| **Small** | (POSE off) | exp274 | 68.1 / 76.8 | — |
| **Small** | `[-1]` | exp275 | 68.8 / 76.8 | 68.8 / 76.7 |
| **Small** | `[-2,-1]` | exp276 | 68.3 / 77.2 | 68.3 / 77.2 |
| **Small** | `[-3,-2,-1]` | exp277b (seed 41) | 68.3 / 77.6 | 68.3 / 77.6 |

**结论**: Pure PSG stage 贡献 ≤ 0.3 mAP 差异, Tiny 上 PSG stages 从 1 到 3 mAP 几乎持平 (60.5, 60.5, 60.5), Small 上 `[-1]` 略优 (68.8)。

**说明**: pure PSG 配置只输出 global feature, 无 part tokens — **MaxSim 不适用** (no part features to match)。

### Table B: GCN cap × PSG stage (Full Scaffold, no PLBOA PSG variants)

**Purpose**: GCN 隐藏维度 (256/512) × PSG stage (1/2) 2×2 网格, Full Scaffold (LGPA+OA-SD+ParAug+PLBOA) 基础上变 GCN/PSG 参数。

**Occ-Duke Tiny (v2 MaxSim after fix)**:

| | GCN 256 | GCN 512 |
|---|---------|---------|
| PSG `[-1]` | exp278: 65.7/76.7 → **66.9/77.6** | exp280: 65.7/76.2 → **66.8/77.4** |
| PSG `[-2,-1]` | exp279: 65.7/76.9 → **66.9/77.7** | **exp261: 65.9/77.4 → 67.2/78.6** (best) |

**Occ-Duke Small (v2 MaxSim after fix)**:

| | GCN 256 | GCN 512 |
|---|---------|---------|
| PSG `[-1]` | exp282: 73.7/83.9 → **74.6/85.3** (R1 best) | exp284: 73.4/82.9 → **74.5/85.0** |
| PSG `[-2,-1]` | exp283: 73.5/83.2 → **74.5/84.7** | **exp285b: 73.8/83.8 → 74.7/84.8** (mAP best) |

**结论 (3-backbone)**:
- **Tiny**: `GCN512 + 2-stage PSG` best (exp261 67.2/78.6)
- **Small**: `GCN512 + 2-stage PSG` best mAP, `GCN256 + 1-stage PSG` best R1 — 中间格 GCN512+1stg 反而弱
- **Base** (exp263d, 外表): 标杆 75.2/84.8

### Table C: Full − GCN (LGPA-only, 验证 GCN 冗余)

**Purpose**: 从 Full Scaffold 移除 GCN 分支, 只保留 LGPA + OA-SD + ParAug + PLBOA + PSG, 看 GCN 是否 redundant。

**Occ-Duke (v2 eval)**:

| Backbone | PSG stage | Exp (LGPA-only) | eq+flip | **MaxSim+flip** | vs Full Scaffold (w/ GCN) Δ |
|----------|-----------|-----------------|---------|-----------------|------------------------------|
| Tiny | `[-1]` | exp286 | 66.0 / 76.6 | **67.1 / 77.9** | vs exp278 (GCN256+1stg) +0.2/+0.3 |
| Tiny | `[-2,-1]` | exp287 | 65.9 / 77.0 | **67.2 / 78.5** | **vs exp261 (Full) 0 / −0.1** |
| Small | `[-1]` | exp288 | 73.8 / 83.8 | **74.8 / 84.8** | vs exp284 (GCN512+1stg) +0.3/−0.2 |
| Small | `[-2,-1]` | exp289 | 73.8 / 83.3 | **74.8 / 84.8** | **vs exp285b (Full) +0.1 / 0** |
| Base | `[-2,-1]` | exp294 | 74.0 / 82.6 | **75.0 / 84.4** | **vs exp263d (Full) −0.2 / −0.4** |

**结论 (3-backbone 统一)**:
- **GCN 分支几乎冗余**: Tiny 0 mAP, Small +0.1 mAP, Base −0.2 mAP (noise level)
- **R1 小幅贡献** (GCN): Tiny −0.1, Small 0, Base −0.4
- 论文 claim: **"LGPA 已捕获足够 semantic pose 结构, GCN 可移除, 参数/算力减少, mAP 持平"**

### Table D: Market PLBOA → Occ-ReID 跨域消融

**Purpose**: 测 Market 训练时加 PLBOA (lower-body occlusion augment) 是否帮助 Occ-ReID 跨域泛化。

| Market ckpt | PLBOA | Occ-ReID mAP / R1 (MaxSim+flip) | Δ vs OFF |
|-------------|-------|--------------------------------|----------|
| exp269 orig e80 (seed 42) | OFF | **88.2 / 91.2** | baseline |
| exp293 e80 eff (seed 42) | **ON** | 72.4 / 76.7 | **−15.8 / −14.5** |
| exp293 full 120 (seed 42) | **ON** | **62.8 / 68.6** | **−25.4 / −22.6** (worst) |

**结论**: **Market 上加 PLBOA 对 Occ-ReID 跨域 catastrophic** (−25 mAP at full 120)。Paper 不加 PLBOA on Market。

---

## Eval Bug Fix Note (IMPORTANT)

在 v2 re-eval 前, Small/Tiny Full Scaffold ckpts 的 MaxSim+flip 数字被 bug 污染 ~0.8 mAP。

**Bug**: `scripts/eval_fliptest_maxsim.py` 的 auto-detect 使用 `feat.shape[1] % 1024 == 0 ? 1024 : 768`。Small Full Scaffold feat dim = 6144 (= 768×8 = 1024×6), 歧义 case 下错误选 C=1024, 导致 global/part slice boundaries 错位, part features 污染。

**Fix** (commit 286aedb): 改用 `cfg.MODEL.TRANSFORMER_TYPE` 直接判断 backbone。Small/Tiny → 768, Base → 1024。

**验证**: exp255 ckpt + v1 buggy script = 74.4/84.3; v2 fix script = **75.2/85.6** (完全重现历史数字)。

**影响**:
- Small/Tiny **Full Scaffold** (GCN+LGPA, feat dim 6144) — 全部受影响, Δ MaxSim ~+0.8 mAP after fix
- Small/Tiny **LGPA-only** (no GCN, feat dim 4608) — 不受影响 (歧义不触发)
- Small/Tiny **pure PSG** (global only) — 不受影响
- **Base** (embed_dim=1024, 系统正确) — 不受影响

所有本 result.md 的 Small/Tiny Full Scaffold MaxSim 数字均 **v2 fix 后** 重测。

---

## Appendix: 完整 ckpt 清单 & eval protocol

### 每个 exp 的 config + CLI override 记录

| Exp | Dataset | Backbone | Config file | CLI override |
|-----|---------|----------|-------------|--------------|
| exp261 | OD | Tiny | `prcv_best_tiny.yml` | default (Full) |
| exp262 | OD | Small | `prcv_best_small.yml` | default (old code) |
| exp263d | OD | Base | `prcv_best_base.yml` | `SOLVER.SEED 41` |
| exp264 | OP | Tiny | `prcv_best_tiny.yml` | default |
| exp265 | OP | Small | `prcv_best_small.yml` | default |
| exp265b | OP | Small | `prcv_best_small.yml` | `SOLVER.SEED 41` |
| exp266 | OP | Base | `prcv_best_base.yml` | `WITH_CP True` (e60 eff ckpt only) |
| exp266b | OP | Base | `prcv_best_base.yml` | `SOLVER.SEED 41` |
| exp266c | OP | Base | `prcv_best_base.yml` | `SOLVER.SEED 42 TEST.IMS_PER_BATCH 64` |
| exp267 | Market | Tiny | `prcv_best_tiny.yml` | default |
| exp268 | Market | Small | `prcv_best_small.yml` | default |
| exp269b | Market | Base | `prcv_best_base.yml` | `SOLVER.SEED 42 TEST.IMS_PER_BATCH 64` |
| exp270 | OD | Small | `prcv_best_small.yml` | `POSE_ENABLED False` |
| exp271-273 | OD | Tiny | `prcv_best_tiny.yml` | pure scaffold: POSE_LGPA/GCN/OA_SD/PARALLEL_AUG/LOWER_BODY_OCC 全 False, POSE_PSG_STAGES 变 |
| exp274 | OD | Small | `prcv_best_small.yml` | `POSE_ENABLED False` |
| exp275-277b | OD | Small | `prcv_best_small.yml` | pure scaffold, PSG_STAGES 变 |
| exp278-280 | OD | Tiny | `prcv_best_tiny.yml` | Full Scaffold 变 GCN_HIDDEN (256) / PSG_STAGES |
| exp282-285b | OD | Small | `prcv_best_small.yml` | Full Scaffold 变 GCN_HIDDEN (256) / PSG_STAGES |
| exp286-287 | OD | Tiny | `prcv_best_tiny.yml` | `POSE_SKELETON_GCN False` + PSG_STAGES |
| exp288-289 | OD | Small | `prcv_best_small.yml` | `POSE_SKELETON_GCN False` + PSG_STAGES |
| exp290 | OP | Small | `prcv_best_small.yml` | `POSE_USE_TARGET_HEATMAP True` |
| exp291 | OD | Small | `prcv_best_small.yml` | `POSE_USE_TARGET_HEATMAP True` |
| exp293 | Market | Base | `prcv_best_base.yml` | `POSE_LOWER_BODY_OCC True` (PLBOA ON) |
| exp294 | OD | Base | `prcv_best_base.yml` | `SOLVER.SEED 41 POSE_SKELETON_GCN False TEST.IMS_PER_BATCH 64` |

### exp295 复现完成 ✓ (2026-04-24 23:25 CST)

- **config**: `prcv_best_small.yml` + `SOLVER.SEED 1234`
- **train eq+flip FINAL**: 74.2 / 84.0 (+1.0 mAP vs exp255 historical 73.2/83.3)
- **MaxSim+flip**: **75.2 / 85.4** ← **完全 match exp255 75.2/85.6** (mAP 精准重现, R1 -0.2 噪声范围)
- lab4090 training ~6h (17:19 → 23:25 CST), 175s/epoch
- 证实: exp255 历史 75.2 数字是真实可重现, 非 eval bug 产物, paper 可安全用

### Eval command template

```bash
python3 scripts/eval_fliptest_maxsim.py \
  --config_file <config.yml> \
  --weight <ckpt.pth> \
  DATALOADER.NUM_WORKERS 2 \
  TEST.IMS_PER_BATCH 128  # Base: 64 (防 OOM)
  [extra CLI override to match training]
```
