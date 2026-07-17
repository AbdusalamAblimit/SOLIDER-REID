# exp360 Intruder — monitor

## 阶段 0 地基机制验证（frozen probe，2026-06-26）

**脚本**: `experiments/cargo_cvpb/cvpb_intruder_probe.py`（market noLMloss frozen baseline + 合成 target+donor，occ_frac 0.45 下半身遮挡）
**log**: 4090 `/tmp/exp360_intruder_full.log`（50 donor×20 + 700 query vs full gallery）

| 判据 | 结果 | 通过 |
|---|---|---|
| H1 donor-ID 泄漏可测 | probe acc 73% vs chance 2% = **36.5x** | ✅ PASS（泄漏巨大确凿）|
| person >> rand control | leak person 0.15 >> rand −0.01 | ✅ PASS |
| **H2 leak ↔ AP drop（控 #false）** | raw spearman **+0.120** → **partial\|#false = −0.028（≈0）** | ❌ **FAIL** |

AP drop 大（clean 0.835 → mix 0.409，−0.43，遮挡确实大幅损害检索）。

### 诚实判读
- donor 泄漏**存在且巨大**（H1 36.5x，能从被遮挡 target 的 embedding 73% 认出遮挡者），但**泄漏量不独立于 #false-in-topk 预测检索损害**（H2 控 #false 后 ≈0）。
- 即：leak 和检索损害都是遮挡的"症状"，但 leak 不是损害的**独立原因**（损害主由 top-k 混入错 ID 驱动）。
- **memory 铁律再次发挥**（#false 控制把 +0.120 打回 −0.028），避免 over-claim "泄漏导致损害"。Hubness/evidence/d17 同款。
- 对 Intruder 的实质打击：核心假设"压 donor 泄漏 → 救检索"因果地基不稳（= codex 头号风险=退化 target ambiguity 墙）。

### 不收敛停（deep work 模式）
- H2 是 frozen per-query 相关（哪个 query 损害大），FAIL ≠ "训练压 donor-ID 无效"（相关 ≠ 干预效果，H3 训练才是终判）。
- codex 评估（`codex_h2fail_decision.md`）：H2 FAIL 不数学杀死(但杀强叙事)，建议 Stage0.5 frozen donor-null projection 因果测试（GRL 上界代理）再裁决。

## 阶段 0.5 donor-null projection 因果测试（2026-06-26，codex 7/10）

frozen rank-r 抠掉 donor 判别方向，看 mix AP 是否回升（= GRL 软压的上界代理）：

| r | donor acc | mixAP Δ | cleanAP Δ | #false |
|---|---|---|---|---|
| 5 | 0.72→0.575 | +0.018 | −0.002 | 5.89→5.71 |
| 10 | →0.500 | +0.025 | −0.005 | →5.63 |
| 20 | →0.480 | +0.028 | −0.008 | →5.57 |
| 40 | →0.480 | **+0.031** | −0.009 | →5.58 |

**裁决：Intruder DEAD（因果证伪）**：
- donor 信息**高度分布式**：r=40 抠 40 维，donor acc 只 0.72→0.48（r=20→40 饱和，抠不干净）。
- 即使抠 40 维，mixAP 只回 **+0.031**（mix 0.409→0.440 vs clean 0.835，gap 0.4 几乎没动），clean 被伤 −0.009，#false 几乎不降。
- **坐实 codex 头号风险 + memory target ambiguity 墙**：donor-ID 可读(H1 36x)但压它不救排序。
- **codex Stage0.5 完美奏效**：frozen 因果测试(零训练)裁决 Intruder，省多日 GRL 训练 + 审查。上界 projection 救 0.031，GRL 软压更不行。

**→ Intruder DEAD（有据：H2 #false 控制 + Stage0.5 r-sweep 双重因果证伪）。按 codex 裁决转 B PSC-JEPA（6.5，从 SOLIDER continued-pretrain + pose-defined pseudo-support-bank latent JEPA，真正换量级）。**
