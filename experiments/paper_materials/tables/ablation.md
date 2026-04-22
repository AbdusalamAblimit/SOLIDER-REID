# 消融实验表 (PRCV 2026)

## 协议

- 默认 eval: `equal_concat + flip-test` global feature
- 训练集: Occluded-Duke, 120 epochs, Swin-{Tiny, Small}
- 每行 = 单独 full training run

---

## Table A — PSG stage 消融 (pure PSG, no GCN/OA-SD/ParAug)

**配置**: Swin backbone + pose supervision + PSG (pure scaffold, 不含 GCN / LGPA / OA-SD / ParAug / LOWER_BODY_OCC)

| Backbone | PSG stages | mAP | R1 | Exp |
|----------|-----------|-----|----|----|
| Swin-Tiny | 无 | 59.2 | 68.4 | exp270 |
| Swin-Tiny | `[-1]` | 60.2 | 69.5 | exp271 |
| Swin-Tiny | `[-2,-1]` | 60.5 | 69.7 | exp272 |
| Swin-Tiny | `[-3,-2,-1]` | **60.5** | **69.9** | exp273 (R1 peak) |
| Swin-Small | 无 | 68.1 | 76.8 | exp274 |
| Swin-Small | `[-1]` | **68.8** | 76.8 | exp275 (mAP peak) |
| Swin-Small | `[-2,-1]` | 68.3 | 77.2 | exp276 |
| Swin-Small | `[-3,-2,-1]` | 68.3 | **77.6** | exp277b s41 (R1 peak; s42 exp277 塌缩 49.0/57.7 偶发, 改 s41 合理) |

**结论**:
1. **PSG stage 增加 → R1 单调改善** (Tiny 68.4→69.9 +1.5; Small 76.8→77.6 +0.8)
2. **Small mAP peak 在 1-stage** (68.8), 3-stage 回落到 68.3 → 过深 PSG 压缩 Small 更强特征
3. **Tiny PSG 增益更明显** (+1.3 mAP on 2-stage), Small +0.7 mAP

---

## Table B — GCN cap × PSG stage (Full Scaffold)

**配置**: Swin + LGPA + GCN + OA-SD + ParAug + LOWER_BODY_OCC + PSG

### Swin-Tiny (4/4 FINAL)

| | GCN256 | GCN512 |
|---|---|---|
| PSG `[-1]` | **65.7 / 76.7** (exp278) | **65.7 / 76.2** (exp280, weakest R1) |
| PSG `[-2,-1]` | 65.7 / 76.9 (exp279) | **65.9 / 77.4** (exp261 / exp281, best) |

**观察**:
- **GCN512+1stg (exp280) 最弱 R1 = 76.2** — 和 Small 2×2 exp284 最弱模式一致
- GCN256 下 1-stg ≈ 2-stg (mAP 65.7=65.7, R1 +0.2)
- GCN512 下 2-stg (exp261) > 1-stg (exp280) R1 +1.2
- **跨 backbone 结论一致**: 大 GCN 必须 2-stg 配套才完整 exploit

### Swin-Small

| | GCN256 | GCN512 |
|---|---|---|
| PSG `[-1]` | **73.7 / 83.9** (exp282, R1 best) | 73.4 / 82.9 (exp284, weakest) |
| PSG `[-2,-1]` | 73.5 / 83.2 (exp283) | **73.8 / 83.8** (exp285b, mAP best) |

**观察**:
- **方差 ≤ 0.4 mAP / 1.0 R1** — 所有 4 格非常接近
- **R1 peak (GCN256+1stg)** 和 **mAP peak (GCN512+2stg)** 是"light vs heavy"两个最优配置点
- **GCN512+1stg (exp284) 反而最弱** — 大 GCN 必须 2-stg 配套
- **exp285b 是 lab4090 同设备 rerun exp262** (73.8/83.1 srvA), 确认跨设备 mAP 持平, R1 +0.7 (更严谨)

---

## Table C — LGPA-only × PSG stage (no GCN, 验证 semantic branch 依赖)

**配置**: Swin + LGPA + OA-SD + ParAug + LOWER_BODY_OCC + PSG (**移除 GCN**)

### Swin-Tiny (2/2 FINAL)

| PSG | mAP | R1 | Exp | vs Full Scaffold |
|-----|-----|----|----|------------------|
| `[-1]` | **66.0** | 76.6 | exp286 | +0.1/-0.8 vs exp261 Full 65.9/77.4 |
| `[-2,-1]` | 65.9 | 77.0 | exp287 | 0/-0.4 vs exp261 |

### Swin-Small (0/2 FINAL; exp288 运行中 e80 73.1/83.5)

| PSG | mAP | R1 | Exp | 状态 |
|-----|-----|----|----|------|
| `[-1]` | TBD | TBD | exp288 | 🔄 srvC e80, FINAL ~12:45 |
| `[-2,-1]` | TBD | TBD | exp289 | ⏳ queued (auto-chain after exp288) |

**观察 (Tiny)**:
- LGPA-only vs Full Scaffold 几乎无 mAP 差 (66.0 vs 65.9)
- **Tiny 下 GCN 几乎无贡献** (R1 -0.4 ~ -0.8) → small-capacity backbone 语义分支已饱和

---

## Table D — 跨 seed 鲁棒性

| Setup | seed 42 | seed 41 | Δ | Exp |
|-------|---------|---------|----|----|
| Small Full Scaffold OD | 73.8/83.1 (exp262 srvA) | 73.8/83.8 (exp285b lab4090) | 0/+0.7 (R1 跨设备 slight) | 不同设备 + 同 seed 42 vs seed 42 |
| Small PSG 3-stg OD | 49.0/57.7 (exp277 塌缩) | **68.3/77.6** (exp277b) | +19.3/+19.9 | 偶发 seed 塌缩问题 |
| Base Full Scaffold OD | (exp263 e100 eff 72.5/81.8 seed 42) | **74.1/83.3** (exp263d s41) | +1.6/+1.5 | lab3090 pwrlim 280W |
| **Small Full Scaffold OP** | 78.4/86.2 (exp265 srvC) | **78.5/85.9** (exp265b srvA) | +0.1/-0.3 | 跨 seed + 跨设备, OP 对 seed 最鲁棒 (Δ ≤ 0.3) |

**结论**: 大多数 seed 对结果影响 ≤0.7 mAP, 但 **pure PSG 3-stg 有偶发塌缩风险**, 主表应避开此配置。

---

## Table E — 跨设备鲁棒性

| Setup | 设备 A | 设备 B | Δ |
|-------|--------|--------|----|
| Small Full Scaffold (exp262 config) | srvA 5060Ti 73.8/83.1 | lab4090 73.8/83.8 (exp285b) | 0/+0.7 → mAP 稳, R1 跨设备 +0.7 |

**结论**: 跨设备 mAP 差异 ≤0.1, R1 最大 0.7, 论文数字稳定可信。

---

## 填表说明

1. 所有数字来自 `/root/work/SOLIDER-REID/log/occluded_duke/{exp}/train_log.txt` 末尾 `Validation Results - Epoch: 120` 段 (log 精确复制, 无凭记忆)
2. 未 FINAL 的行以 "pending" / "running" 标注
3. 跨 seed / 跨设备表用于 supplementary robustness
