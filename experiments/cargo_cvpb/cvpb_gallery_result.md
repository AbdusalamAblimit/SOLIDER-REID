# Gallery-Composition Kill-Switch — 结果 (cvpb_gallery_killswitch.py)

零训练 frozen 特征 + numpy。Market exp260b(全量 mAP 94.61) / Occluded-Duke exp255(全量 mAP 73.05)。
缓存特征 `/tmp/hub_market_feats.npz`(Nq=3368 Ng=15913 dim=1024) / `/tmp/hub_oduke_feats.npz`(Nq=2210 Ng=17661 dim=768)。
log: `/tmp/cvpb_gallery_market.log`(57s) / `/tmp/cvpb_gallery_oduke.log`(42s)。
★每个 per-query 相关都控了 trivial 代理(#false-in-topk / max-of-N / per-image)——吸取 HUBNESS §7.6 教训。

---

## 测试 A — Gallery-Growth Tax 【LIVE：两个 trivial 对照都活下来】

固定 CORE 任务(core_frac=0.2, core gallery cap 8/ID), 逐步注入 held-out 同域 ID(+gallery-only ID)当纯 distractor。
**frozen 模型(表示完全没变), 旧 query mAP 随 gallery 膨胀显著掉:**

| mult | Market dmAP | Market dR1 | OD dmAP | OD dR1 |
|------|------------|-----------|---------|--------|
| 1.5x | −0.55 | −0.27 | −2.60 | −2.56 |
| 2x   | −1.11 | −0.65 | −4.28 | −3.61 |
| 3x   | −1.76 | −0.83 | −6.74 | −5.71 |
| 5x   | −2.92 | −1.56 | −9.13 | −7.10 |
| 10x  | **−4.43** | −2.57 | **−12.86** | −9.16 |

→ 量级接近 LReID 报告的 "forgetting"(OD 10x 掉 12.9 mAP), 但**模型没变**——是 gallery-growth tax 的误归因。

### CONTROL1 (#false-in-topk, HUBNESS 致命代理) — tie-aware Spearman(对拍 scipy)
| | Market | OD |
|---|---|---|
| Spearman(−dAP, d#false) | +0.7154 | +0.8170 |
| partial(−dAP, base#false \| d#false) | +0.0922 | **+0.3587** |
| "TOP-10 #false 不变"子集 mean dAP (n) | **−1.23** (578) | **−2.56** (306) |

→ 大部分掉点确实是 trivial 计数(ρ+0.72/+0.82), **但**控住 #false-in-topk 后仍有成分存活:
partial OD +0.36; "top-10 #false 没变"的 query 仍掉 1.2(Market)/2.6(OD) mAP(=top-10 窗外的 rank reorder)。
**注**: 这是 PARTIAL 结构证据(AP 可经 top-10 之外重排下降); 决定性结构证据是 CONTROL2。

### CONTROL2 (结构 vs 纯 count) ★决定性
| at 10x | Market | OD |
|---|---|---|
| real held-out distractor dmAP | **−4.45** | **−13.16** |
| 列洗牌(毁方向, 同 count) dmAP | −0.002 | −0.006 |

→ **碾压式结论**: 同样数量的 distractor, 真实 held-out 人体几何 −13 mAP, 随机方向向量 ≈0。
**tax 是结构性的**(真邻居咬人), 不是机械 count。distractor 的*身份几何*才是膨胀税的来源。

---

## 测试 B — Gallery-Size Rejection 【DEAD：trivial EVT max-of-N】

watchlist size {10,50,100,250}, impostor 源 = gallery-only + 非 enrolled shared ID。
GLOBAL vs SIZE-CONDITIONED 阈值用 CAL/EVAL 折(偶/奇 trial)消除 in-sample 循环。

RANDOM-null 现为**列洗牌**(同 count、真实 norm、无 genuine 泄漏; codex#3 修复后)。

| | Market | OD |
|---|---|---|
| impostor max-cosine 随 size: REAL ρ | +1.0 | +1.0 |
| 同上 RANDOM-null ρ | +1.0 | +1.0 |
| size-cond 净增益 drift-red (REAL−RANDOM) | **−0.293** | **−0.303** |
| 净 dDIR@FPIR5% (REAL−RANDOM) | −0.016 | −0.025 |

→ impostor max 确实随 size 升, **但 max-of-N null 升得一样甚至更猛**(纯极值效应)。
强 backbone 上 genuine~0.97 / impostor~0.5 近乎完全可分, 拒识接近饱和; size-conditioning 表观增益
**全在 max-of-N trivial floor 里**, 净增益为负。**诚实判死**: 被 max-of-N 极值统计吃掉。
(阈值全用 CAL/EVAL 折 out-of-sample, 无 in-sample 循环。)

---

## 测试 C — Singleton Merge 【DEAD：trivial count "更多彩票"】

Zipf gallery(head 多图 head support 2-12, tail singleton), tail query 移除自身 singleton→真 unknown。

per-head-ID Spearman 用 tie-aware(对拍 scipy); 阈值用 CAL(偶 seed)/EVAL(奇 seed)折 + 整体 tail-probe 分母。

| | Market | OD |
|---|---|---|
| NN-is-head 比例 | 0.674 | 0.720 |
| per-head-ID Spearman(support, attraction-count) | +0.043 | +0.059 |
| per-head-ID Spearman(support, attraction-PER-IMAGE) | **−0.013** | **−0.009** |
| support-cal vs global OVERALL false-merge d (recall0.90) | +0.000 | −0.001 |
| 同上 recall0.95 | −0.001 | −0.014 |
| fallback-to-global 比例 | 0.00 | 0.14 |

→ NN-is-head=0.72 只反映 head 占了 72% gallery 图像质量(机械)。控住 count 后(per-image)
heads **不超额吸附**(ρ≈0 甚至略负)。CAL/EVAL 折后 support-calibrated 阈值**零增益**(d≈0)。
**诚实判死**: 被 "head 图多→NN 彩票多" 的 trivial count 吃掉。

---

## VERDICT

- **唯一 LIVE = 测试 A (Gallery-Growth Tax)**: 干净、非 trivial-proxy(过了 #false-in-topk + 列洗牌双控)。
  headline = "frozen 强 ReID 的旧 query mAP 随同域 gallery 膨胀结构性下降(OD 10x −12.9 mAP), 且该下降
  在控住 #false-in-topk 与 count(列洗牌)后仍存活——LReID 把这部分误记为 catastrophic forgetting"。
- **测试 B / C 诚实判死**: B 被 max-of-N 极值吃掉(净增益负, 强特征拒识饱和); C 被 count 吃掉(per-image 吸附≈0)。
- vs HUBNESS: 上一个诊断被 #false-in-topk 杀; **本次 A 的 CONTROL1 子集 + CONTROL2 列洗牌正是为这条教训设计, A 活下来**。

### 注意事项(写作前需诚实标注)
1. 测试 A 的 CONTROL2 是最强证据(real −13 vs shuffle −0)。CONTROL1 的"结构残差"较弱(Market partial 仅 +0.05, OD +0.28),
   主结构性证据应靠 CONTROL2, 而非 CONTROL1 的 partial。
2. A 是"诊断/问题重定义", remedy(distractor-aware training)未验证——需独立实验, 且要警惕撞 backward-compatible LReID。
3. 与 k-reciprocal 互补性、跨 backbone 普适性未测(留后续)。
