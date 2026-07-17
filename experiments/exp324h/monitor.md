# exp324h monitor — adapted-DINO (LoRA) vs SOTA Swin ORACLE

## 设置
- 机器：lab-3090-d，system python3，eval-only，无训练，无 commit。
- DINO retriever：exp324d base-r16 LoRA **e10** checkpoint（lora_10/ + head_10.pth）。
- Swin：exp255 MaxSim distmat（experiments/exp324f/swin_distmat.npz，mAP 75.16 / R1 85.57）。
- heavy 子集：query pose vis≤8（与 exp324g 同源，989 queries）。
- 对照 baseline = **exp324g 冻结 DINO**：oracle 上界 +0.12，P_dino_only 0.20%（2/989），
  Jaccard 0.062，DINO-only heavy mAP 8.65，Swin heavy mAP 72.57。

## 启动验证（smoke）
- head: 702 classes, embed 512 ✓
- LoRA: 589,824 params 加载，sum|lora_B|=1469.6（>0 证明训练好的 adapter 真加载，非 random init）✓
- Claude broad review: VERDICT PASS（M1 已修：WARN 只对 lora_ key 报）✓

## 运行
- PID 313112，与 exp324d(309591) 并行共享 3090（~20G free，no_grad eval 无 OOM 风险）。
- 阶段：load Swin npz → load head+LoRA → prepare_split(cache hit) → encode query(2210)+gallery(17661) → part-MaxSim distmat → align → oracle → fusion sweep。

### [完成 2026-06-16] ORACLE 结果（989 heavy queries，topk=10）

adapted-DINO part-MaxSim ALONE: **all mAP=44.67 R1=57.01**（= exp324d e10，验证 LoRA 加载+encode+distmat 全链路正确）。
align OK（filename join，pid 校验，camid offset=1）。

| 指标 | 冻结 DINO (exp324g) | adapted-DINO (exp324h) | 变化 |
|------|--------------------|------------------------|------|
| DINO-only heavy mAP | 8.65 | **36.78** | ×4.3 更强 |
| top-10 Jaccard | 0.062 | **0.253** | ×4 更重叠（**更不正交**） |
| P_dino_only | 0.20% (2/989) | **0.71% (7/989)** | ×3.5，但绝对仍极小 |
| Swin heavy mAP | 72.57 | 72.57 (同) | — |
| oracle 上界 heavy mAP | 72.69 | **73.16** | — |
| **oracle gain** | **+0.12** | **+0.59** | 比冻结好，但 **< +1 clean-go 门槛** |

### fusion sweep（z-score / min-max，w∈{0.05..0.5}）—— 测 "beat 75"

| arm | ALL mAP | vs Swin75.16 | HVY mAP | vs Swin72.57 |
|-----|---------|--------------|---------|--------------|
| Swin alone | 75.16 | — | 72.57 | — |
| **best ALL = minmax w0.2** | **75.53** | **+0.37** | 72.79 | +0.22 |
| best HVY = zscore w0.15 | 75.50 | +0.35 | **72.83** | **+0.26** |
| w≥0.4（任一 norm） | 75.0~74.1 | 转负 | 71~72 | 转负 |

re-rank：**主动跳过**（repo `re_ranking(only_local)` 需完整 (Q+G)² distmat 的 q-q/g-g block，
仅 dump 了 q-g，不可重建，不伪造）。fusion sweep 即可行的 beat-75 测试。

### 判定：AMBIGUOUS（脚本字面）→ 实质 STOP-LOSS，确认 analysis 结论

- adapted-DINO 确比冻结明显更强（heavy 8.65→36.78），也确实多救回几个 Swin 漏检（P_only 0.20%→0.71%，7 个）。
- 但 **oracle 上界仅 +0.59 mAP**（远低于 +1 clean-go 门槛），**绝对天花板 73.16 仍低于 Swin 自身 all-query 75**。
  perfect fuser 都推不过 +1，实际 distmat 融合更只 +0.37（ALL）/ +0.26（HVY）。
- **关键反直觉发现**：adaptation 让 DINO 变强的**同时也让它和 Swin 更一致**（Jaccard 0.062→0.253，×4）。
  "变判别" = 学到与 Swin 相似的判别方向 → **互补性反而随判别性上升而下降**。
  adapted-DINO 救回的多是 Swin 也接近能救的样本，不是 Swin 系统性盲区。
- **+0.37 ALL 是 test-time distmat 融合的微小后处理收益**（把一个 44-mAP 弱检索器 z/minmax 归一后掺 20%），
  按项目铁律属 NFC/RR 同级 test-time trick，**不算训练端方法贡献**，且远不构成"beat 75 = 真方法"。

**结论**：exp324g（冻结）→ exp324h（LoRA-adapted）两端夹逼，确认 "DINO completes Swin" 家族
（含已判别化的 adapted-DINO）对 75-mAP SOTA Swin **无足够独立信息**。融合/re-rank 不能 beat 75 = 真方法。**止损**。
诚实价值：这给 overnight FM 分析补上最后一块 —— "变判别 ≠ 变互补，反而更冗余"，是干净的负向方法结论。

artifacts：`experiments/exp324h/lora_dino_distmat.npz`（对齐后 adapted-DINO distmat）、`oracle_summary.json`。
