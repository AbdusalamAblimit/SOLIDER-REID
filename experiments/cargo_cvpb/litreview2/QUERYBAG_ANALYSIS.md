# Ambiguous Query-Bag ReID — kill-switch 结果 + VERDICT

> 脚本 `experiments/cargo_cvpb/cvpb_querybag_killswitch.py`（frozen exp255/exp260b 特征 + numpy, 零训练）。
> log: lab-3090 `/tmp/cvpb_querybag_od.log`、`/tmp/cvpb_querybag_market.log`。
> 设计见 `QUERYBAG_KILLSWITCH_DESIGN.md`。bag 真目标数 m=4, 污染率 = c/(m+c)。
> **consensus 列是 oracle-tuned 上界**（每 cell 在 mode×thr×topL=2×5×3=30 配置里取 mAP 最优）——给提案方法最强一枪。

## VERDICT: **NO-GO**（两数据集一致, 干净判死）

核心一句: **k-reciprocal（现成 test-time 后处理）在每个污染率下都是最好的去污染手段; oracle-tuned Target-Consensus 即使被最优调参, 也从未比 k-reciprocal 高 ≥2 mAP（OD@50% 甚至只打平: 85.18 vs 85.40）。** 这正是 HUBNESS §7.6 的教训复现: trivial baseline 已经解决, 提案机制无增量价值。污染确实造成大 protocol gap（GO 第一条满足）, 但"修复"被 k-reciprocal 占。

## Occluded-Duke (exp255, 519 query-id, 17661 gallery)

### mAP
| 污染率 | avg | single-best | median | trimmed | **consensus★** | k-recip | camera |
|---|---|---|---|---|---|---|---|
| 0%  | 96.67 | 96.91 | 96.41 | 95.35 | 95.60 | **97.47** | 97.12 |
| 25% | 94.14 | 69.43 | 94.62 | 94.68 | 94.76 | **96.62** | 94.90 |
| 50% | 75.77 | 42.65 | 77.29 | 75.16 | 85.18 | **85.40** | 78.92 |
| 75% | 42.14 | 24.89 | 40.85 | 33.95 | 37.96 | **52.34** | 46.93 |

### R1
| 污染率 | avg | single-best | median | trimmed | consensus★ | k-recip | camera |
|---|---|---|---|---|---|---|---|
| 0%  | 99.61 | 99.81 | 99.61 | 99.04 | 99.61 | 99.61 | 99.61 |
| 25% | 98.27 | 71.48 | 98.07 | 97.69 | 97.88 | 98.46 | 98.84 |
| 50% | 80.54 | 41.23 | 81.12 | 77.46 | 86.90 | 85.55 | 83.24 |
| 75% | 37.19 | 19.08 | 34.10 | 24.08 | 33.33 | 43.74 | 40.66 |

### 逐 cell 判定
- **25%**: avg 仅掉 2.53（<10, 无 protocol gap）; k-recip 96.62 ≥ avg → trivial-solved。NO-GO。
- **50%**: avg 掉 20.90（gap 够大 ✓）; consensus 85.18 追回 +9.41（半跌幅 bar 10.45, 差一点 ✗）; **consensus − k-recip = −0.22**（打平, 远不到 +2）。NO-GO。
- **75%**: consensus 37.96 < avg 42.14（负追回）; k-recip 52.34 碾压全部。NO-GO。
- best-consensus config 逐 cell 漂移（component thr0.1 / component thr0.05 / component thr0.3 / medoid thr0.5）→ 连 oracle 也要换配置, 无单一稳健设定。

## Market-1501 (exp260b, 750 query-id, 15913 gallery)

### mAP
| 污染率 | avg | single-best | median | trimmed | **consensus★** | k-recip | camera |
|---|---|---|---|---|---|---|---|
| 0%  | 98.19 | 98.52 | 97.99 | 97.49 | 97.46 | 98.16 | **98.32** |
| 25% | 97.36 | 71.46 | 97.48 | 97.50 | 97.55 | 97.66 | **97.71** |
| 50% | 83.05 | 43.54 | 85.61 | 83.80 | 87.54 | **89.42** | 85.04 |
| 75% | 53.22 | 25.97 | 51.16 | 41.97 | 41.96 | **57.87** | 56.70 |

### R1
| 污染率 | avg | single-best | median | trimmed | consensus★ | k-recip | camera |
|---|---|---|---|---|---|---|---|
| 0%  | 99.73 | 100.00 | 99.73 | 100.00 | 100.00 | 99.73 | 99.73 |
| 25% | 98.93 | 68.00 | 99.07 | 99.20 | 99.20 | 99.20 | 99.33 |
| 50% | 86.93 | 34.40 | 86.93 | 84.53 | 86.93 | 87.87 | 88.53 |
| 75% | 47.20 | 15.20 | 43.60 | 30.13 | 31.87 | 48.40 | 48.53 |

### 逐 cell 判定
- **25%**: avg 仅掉 0.83（无 gap）; consensus 97.55 与 k-recip 97.66/camera 97.71 打平偏低。NO-GO。
- **50%**: avg 掉 15.14（gap 够 ✓）; consensus 87.54 追回 +4.50（半 bar 7.57 ✗）; **consensus − k-recip = −1.87**（输 k-recip）。NO-GO。
- **75%**: consensus 41.96 < avg 53.22（负追回）; k-recip 57.87 领先。NO-GO。

## 为什么死（机制层面, 不是没调参）
1. **hard contaminant 与真目标在 top-L 空间太像**（污染图本就采自 anchor 的 top-20, 是模型已经混淆的 hard neg）→ 它们能通过 consensus 一致性测试: 即使 oracle-tuned, consensus purity 50% 顶到 0.80-0.85, 75% 直接坍到 0.34-0.36。一致性图切不开"看起来很像目标"的污染。
2. **k-reciprocal 用的就是同一个 neighborhood-consistency 信号, 但在 gallery 层面做得更彻底**——它在每个污染率都是最佳去污染手段（OD 97.5/96.6/85.4/52.3, Market 98.2/97.7/89.4/57.9）, oracle-tuned consensus 从未超过它 ≥2。这与 HUBNESS 诊断同构: neighborhood/topology 类想法被 k-reciprocal re-ranking 完全吸收。
3. **single-best 是被钓走最惨的**（OD 42.7 / Market 43.5 @50%）——hard contaminant 与某些异身份 gallery 强匹配, max-over-bag 直接锁定错身份; 这验证了"多给错图比单图坏"的部分前提, 但 average 靠稀释存活, 不需要 consensus。
4. **camera baseline 平庸**（gamma 总选 0.5, 增益 OD +3 / Market +2 @50%）, 远不及 k-recip。

## 结论
- **协议前提部分成立**: 50% hard 污染下 average 确实大跌（OD −21 / Market −15 mAP）, "多给错图更坏"在 single-best/trimmed 上尤甚——协议本身有 protocol gap。
- **机制证伪**: Target-Consensus 即使 oracle-tuned（30 配置取最优、逐 cell 换配置）也 (a) 追不回一半跌幅, (b) 从未比现成 k-reciprocal 高 ≥2 mAP（多数 cell 还输 1-2）。**trivial test-time 后处理（k-reciprocal）已解决污染, 提案机制无增量价值。** 注: consensus 是 oracle 上界, 单一固定配置只会更差, 故 NO-GO 稳健, 无需补单配置实验。
- 与 HUBNESS §7.6 同一教训: 不让 trivial baseline 赢了还包装成功。**判死。**
