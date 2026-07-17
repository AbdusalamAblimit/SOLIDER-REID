# OVLI 方法稿 — 诚实可投性评估(2026-06-23,用户追问"10 codex 也都这么认为吗")

## 10 codex 独立评估结果
**10/10 全判「悬」(borderline)**。2 个「不能投」(c2/c9), 1 个「能投」带一堆条件(c5), 7 个纯「悬」。平均 ~4-5/10。**没有一个认为现状能稳发 B 类。** 与最初 5 codex(~3.5-4)一致。

## 共识(核心病,不许再自我安慰)
- **强实验现象 + 弱方法 novelty**。
- **方法稿太薄**: 原 novel headline = token-set late-interaction(MaxSim/ColBERT 式), 但被自己消融降级(avg-pool 52.37 >> MaxSim 45.19); 两个涨点机制(OVC-SetVLAD 角度2 / ACVP 角度6)都**实验证伪** → 核心 OVLI 外**没有第二贡献**。
- **撞车面大**: opposite-view 对比 + 多 token 邻近 AlignedReID/ColBERT/一堆 cross-view contrastive ReID。

## 现实天花板
- workshop / PRCV / 中文核心: **有戏**。
- CCF-B 主会(ICME 等): **偏险, 多半弱拒**。
- 补齐 Swin SOTA + AG-ReID.v2 + 严格对照 + 重新包装后, codex 共识也只「**从 4 拉到 6.5**」, 中下 B / 中文核心稳一点, 顶 B 仍悬。

## 路(从「悬」到「可投」)
1. **★ Swin SOTA(决定性 go/no-go)**: 必须证明增益来自 OVLI 不是 Swin 本身(**baseline-Swin vs OVLI-Swin 大增益**)。在跑(fix256 lab-3090 256×128 / fix384 lab-4090 384×128)。eval bug 已修 = resnet50 的 LR 3.5e-4 太猛把 Swin transformer 在 ep8 训崩成常数输出 → Swin backbone 单独 0.1× LR(3.5e-5)。
2. AG-ReID.v2 第二 benchmark 跨数据集。
3. **重新包装(最便宜的 novelty 杠杆)**: 证明 OVLI 专门解决 aerial-ground view gap(不是普通 cross-camera contrastive), 按 view gap / altitude / protocol 分桶分析(CARGO 有 aerial-aerial / aerial-ground / ground-ground 协议)。贡献定位主打"opposite-view 身份证据", 别主打 MaxSim。
4. 严格同设置公平对照(baseline 同 backbone 同 recipe)。

## ★★ Swin gap 实测(go/no-go,2026-06-23)——坏消息那边
| | ep10 | ep20 | gap |
|---|---|---|---|
| OVLI-Swin | 45.38 | 48.35 | |
| baseline-Swin(无OVLI)| **43.79** | 待 | **ep10 +1.59** |

**resnet50 gap +19.9 → Swin gap ~+1.6(大幅缩水)。** baseline-Swin 43.79 已强, OVLI 只加 +1.59 = **弱机制被强预训练 backbone 吃掉**(codex 警告的场景)。待 final(ep60)确认, 但趋势明确: OVLI 在强 backbone 上 headroom 很小。

## 结论(钉死,不反复 + Swin gap 后更新)
OVLI = **数字扎实 + 创新平庸 + 强 backbone 上增益缩水** 的稿子。
- resnet50 上 +19.9 好看, 但那是 backbone 太弱给的虚高 headroom。
- 强 Swin 上只 +1.6 → **机制本身价值有限**, 审稿人一跑 baseline-Swin 对照就看穿。
- **现实目标降级: 中文核心**(PRCV/中文期刊), 不是中下 B。顶 B / 中下 B 主会基本无望(gap 太小, 创新太薄)。
- 诚实讲: 这条线作为"方法稿"接近天花板。要么接受中文核心, 要么这套实证当**经验研究/技术报告**, 主力创新另起炉灶。

## ★★★ ep20 gap 翻负 — 决定性死刑(2026-06-23)
| | ep10 | ep20 | ep30 |
|---|---|---|---|
| OVLI-Swin | 45.38 | 48.35 | 待 |
| baseline-Swin(无OVLI)| 43.79 | **48.98** | **51.30** |
| gap | +1.59 | **−0.63** | |

**baseline-Swin(无 OVLI)ep20 反超 OVLI-Swin, ep30 51.30 继续拉开。** → **OVLI 在强 backbone 上无增益甚至微负**。resnet50 的 +19.9 = 纯弱 backbone headroom artifact。**这比"中文核心"还糟: 机制被证无内在价值, 任何审稿人跑一次 baseline-Swin 对照就废。**

## 最终结论(2026-06-23,不再反复)
**OVLI 作为方法稿(连中文核心)都危险。** resnet50 上的好看数字是 backbone 太弱给的虚高, 一上强 backbone 机制原形毕露。
- **不再在 OVLI/cross-view contrastive 上投入。** 这个角度在 CARGO 上彻底耗尽。
- 这套实证最多当**"弱 backbone 上跨视角对齐现象"的技术报告/负结果记录**。
- **B 类必须全新角度**(5 codex 探索中: 极端尺度 gap 显式建模 / altitude-pose-conditioned / 几何相机先验 / 生成补视角 / 新问题定义)。航拍-地面子领域仍是金矿, 但入口不能是对比学习。
