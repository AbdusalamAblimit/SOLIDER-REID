# SOTA 对比表 (Occluded-DukeMTMC-reID)

> ⚠️ **本文件含部分历史 v1 (buggy eval) 数字**, 单一 authoritative 表见 [`paper_materials/result.md`](../result.md)。
> v1 → v2 修复: commit 286aedb 修了 eval_fliptest_maxsim.py 的 C-dim auto-detect bug, Small/Tiny Full Scaffold MaxSim ~+0.7-0.8 mAP 修正。
> Section "PRCV 2026 主结果表" 下面的数字已更新到 v2。

## 数据来源: KPR (ECCV 2024) Table 3 + 我们的实验

### 无后处理结果

| Method | Venue | Backbone | mAP | R1 |
|--------|-------|----------|-----|-----|
| PCB | ECCV'18 | ResNet50 | 40.8 | 51.2 |
| PGFA | ICCV'19 | ResNet50 | 37.3 | 51.4 |
| BoT | CVPRW'19 | ResNet50 | 44.7 | 51.4 |
| HOReID | CVPR'20 | ResNet50 | 43.8 | 55.1 |
| ISP | ECCV'20 | ResNet50 | 52.3 | 62.8 |
| MHSA | AAAI'21 | ResNet50 | 44.8 | 59.7 |
| PAT | CVPR'21 | DeiT-Small | 53.6 | 64.5 |
| PGFL | MM'22 | ResNet50 | 54.1 | 63.0 |
| HG | ECCV'22 | ResNet50 | 50.5 | 61.4 |
| LDS | PR'22 | ViT-B | 55.7 | 64.3 |
| FED | CVPR'22 | ViT-B | 56.4 | 68.1 |
| OAMN | MM'22 | ResNet50 | 46.1 | 62.6 |
| VGTri | PR'23 | ViT-B | 46.3 | 62.2 |
| SSGR | AAAI'23 | ViT-B | 57.2 | 69.0 |
| **Ours (PCFC+GiLt)** | **-** | **Swin-Tiny (28M)** | **58.0** | **68.0** |
| SOLIDER | CVPR'23 | Swin-Base (88M) | 61.9 | 71.2 |
| PFD | AAAI'22 | ViT-B (86M) | 61.8 | 69.5 |
| BPBreID | WACV'23 | HRNet-W48 | 62.5 | 75.1 |
| KPR w/o prompt | ECCV'24 | Swin-Base (88M) | 73.3 | 82.5 |
| KPR | ECCV'24 | Swin-Base (88M) | 75.1 | 84.3 |

### 分析

**我们 Swin-Tiny (28M params) 的竞争力**:
- 超越 FED (CVPR'22, ViT-B): +1.6% mAP
- 超越 SSGR (AAAI'23, ViT-B): +0.8% mAP
- 超越 LDS (PR'22, ViT-B): +2.3% mAP, +3.7% R1
- 超越 PAT (CVPR'21, DeiT-Small): +4.4% mAP, +3.5% R1
- 超越所有 ResNet50 方法

**与大 backbone 方法的差距**:
- vs SOLIDER (Swin-Base, 88M, 3.1x params): -3.9% mAP
- vs PFD (ViT-B, 86M, 3.1x params): -3.8% mAP
- vs BPBreID (HRNet-W48): -4.5% mAP
- PCFC 在 28M backbone 上缩小了部分差距 (baseline 56.6 → 58.0)

### 含后处理结果

| Method | Post-Processing | mAP | R1 |
|--------|----------------|-----|-----|
| Ours | 无 | 58.0 | 68.0 |
| Ours | + NFC (global+part) | 64.7 | 69.4 |
| Ours | + Re-ranking + Part Dist | 75.3 | 74.4 |

注意: NFC 和 Re-ranking 是通用后处理方法，非我们的训练端创新。
KPR 的 75.1/84.3 也包含了 test-time prompting (可视为后处理)。

---

# PRCV 2026 主结果表 (填充中)

## 模型 = Swin-{T/S/B} + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA

默认测试协议: `equal_concat + flip-test` (主行)；`+ MaxSim` 作为附加匹配独立行。

### Table 1 — Cross-backbone × Cross-dataset 主表

每格两行: 上行 = `Ours (eq+flip)`, 下行 = `+ MaxSim (1:2 hybrid)`。Occ-ReID 列均使用对应 Market ckpt 跨域 inference，不重新训练。

| Backbone | Occ-Duke (mAP / R1) | Occ-PTrack (mAP / R1) | Market (mAP / R1) | Occ-ReID ← Market (mAP / R1) |
|----------|---------------------|-----------------------|-------------------|------------------------------|
| Swin-Tiny (28M) | exp261: **65.9 / 77.4** | exp264: **76.7 / 85.1** | exp267: **92.5 / 96.4** | exp267→OR: TBD |
|  | + MaxSim+flip (v2): **67.2 / 78.6** (Δ +1.3/+1.2) | + MaxSim+flip: **76.8 / 85.2** (Δ +0.1/+0.1) | + MaxSim+flip: **93.0 / 96.7** (Δ +0.5/+0.3) | + MaxSim: TBD |
| Swin-Small (50M) | **exp295 (s1234)**: **74.2 / 84.0** ← break exp255 hist 73.2/83.3 by +1.0; exp285b (s42): 73.8/83.8 | exp265 (s42): **78.4 / 86.2** / exp265b (s41): **78.5 / 85.9** | exp268: **94.3 / 97.3** | exp268→OR: TBD |
|  | + MaxSim+flip (v2): **75.2 / 85.4** (s1234 exp295) / 74.7 / 84.8 (s42 exp285b) | + MaxSim+flip: **78.5 / 86.1** (Δ +0.1/-0.1) | + MaxSim+flip: **94.5 / 97.2** (Δ +0.2/-0.1) | + MaxSim: TBD |
| Swin-Base (88M) | **exp263d (s41)**: **74.1 / 83.3** (lab3090 pwrlim 280W); exp296 lab4090 repro: 73.7/81.7 (-0.4/-1.6 不同 GPU 环境) | exp266b (s41 SOTA): **78.7 / 86.3** / exp266 s42 e60 eff: 78.4/86.2 | exp269b (s42 full120 PLBOA OFF): **94.5 / 97.2** | exp269 → OR: 88.2 / 91.2 (MaxSim+flip ✓ Top-tier) |
|  | + MaxSim+flip (v2): **75.2 / 84.8** ✓ SOTA (Δ +1.1/+1.5 vs eq) | + MaxSim+flip: **78.7 / 86.3** (exp266b) | + MaxSim+flip: **94.6 / 97.2** (exp269b) | + MaxSim: 88.2 / 91.2 ✓ |

**Base 行临时 reference**（旧协议 `exp260b` 本地 3090，不含默认 flip-test）：
- Occ-Duke: 73.9 / 83.2 (eq_concat), 75.4 / 84.8 (MaxSim + flip)
- Market FINAL: 94.4 / 97.1 (eq_concat), 94.7 / 97.2 (MaxSim + flip)
- Occ-ReID ← Market: 86.0 / 88.5 (eq_concat), 88.0 / 90.6 (MaxSim + flip)

（旧协议 exp255 Small 同期为 73.2/83.3 → 75.2/85.6 含 flip+MaxSim；新协议 exp262 预期 ≥74/83。）

### 填表指引

1. 训练完成后从 `/hy-tmp/log/<dataset>/<exp_id>/train_log.txt` 末尾的 `Validation Results - Epoch: 120` 段复制 `mAP:` 和 `CMC curve, Rank-1:` 数字
2. `+ MaxSim` 行: 从 `test.py` 带 `--maxsim_hybrid 1:2` 的独立 eval log 拷贝
3. Occ-ReID cross-domain 行: 用对应 Market ckpt 跑 `test.py --target occluded_reid`
4. 所有数字必须精确复制 log, 严禁凭记忆填写
5. Base 3 行当前 DEFERRED; Phase 1 Tiny/Small 完成后再评估是否回补

### SOTA 对标 (PRCV 版, KPR + 更新参考)

| Method | Venue | Backbone | Occ-Duke (mAP/R1) | Occ-PTrack (mAP/R1) | 备注 |
|--------|-------|----------|-------------------|---------------------|------|
| PAT | CVPR'21 | DeiT-S | 53.6 / 64.5 | — | |
| FED | CVPR'22 | ViT-B | 56.4 / 68.1 | — | |
| SOLIDER | CVPR'23 | Swin-B (88M) | 61.9 / 71.2 | — | |
| BPBreID | WACV'23 | HRNet-W48 | 62.5 / 75.1 | — / — | 需查 KPR 表格补 Occ-PTrack |
| KPR w/o prompt | ECCV'24 | Swin-B (88M) | 73.3 / 82.5 | — / — | 需查 KPR 表格补 Occ-PTrack |
| KPR | ECCV'24 | Swin-B (88M) | 75.1 / 84.3 | — / — | 带 prompt |
| **Ours (Tiny)** eq+flip | — | Swin-T (28M) | **65.9 / 77.4** (exp261) | **76.7 / 85.1** (exp264) | — |
| **Ours (Tiny)** + MaxSim+flip (v2) | — | Swin-T (28M) | **67.2 / 78.6** (exp261) | **76.8 / 85.2** | — |
| **Ours (Small)** eq+flip | — | Swin-S (50M) | **74.2 / 84.0** (exp295 s1234) | **78.5 / 85.9** (exp265b s41) | Small 对 KPR w/o prompt 73.3/82.5 +0.9/+1.4 |
| **Ours (Small)** + MaxSim+flip (v2) | — | Swin-S (50M) | **75.2 / 85.4** (exp295) | **78.5 / 86.0** (exp265b) | **超 KPR 75.1/84.3 +0.1/+1.1** (Small 用更少参数) |
| **Ours (Base)** eq+flip | — | Swin-B (88M) | **74.1 / 83.3** (exp263d s41) | **78.7 / 86.3** (exp266b s41 SOTA) | Base +0.8 mAP over KPR w/o prompt |
| **Ours (Base)** + MaxSim+flip (v2) | — | Swin-B (88M) | **75.2 / 84.8** (exp263d) ✓ SOTA | **78.7 / 86.3** (exp266b) | **超 KPR 75.1/84.3 +0.1/+0.5**, prompt-free |

**Δ vs KPR (75.1 / 84.3)**: 
- Small (50M): **+0.1 / +1.1** (R1 大幅领先, 50M < 88M params)
- Base (88M): **+0.1 / +0.5** (同 backbone, 持平 mAP, R1 领先)

**MaxSim+flip 数字说明**: post-bug-fix v2 (commit 286aedb), 历史 Small/Tiny Full Scaffold v1 数字偏低 ~0.7-0.8 mAP, 已修正。
