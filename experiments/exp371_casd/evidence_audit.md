# exp371 CASD：LGPA 证据审计

## 结论先行

LGPA 的稳定正信号是真实的，但当前只能严谨表述为：

> 在 Swin-Tiny 上，detached 的结构化局部描述子与 global 做标准拼接后，相对同一 checkpoint 的 global 描述子稳定提高约 `+0.9 mAP`；同权重推理干预进一步表明，精确的当前图姿态只解释其中很小一部分。

它尚不能被表述为 CLIP 文本语义增益、正确逐图姿态的全部因果增益、标准 768-D global 增益、GCN 增益或 matching 增益。

## 已成立的证据

| 问题 | 证据 | 当前可下结论 |
|---|---|---|
| LGPA 是否有稳定增益 | `exp336` 三 seed 的 equal-concat/global 差均为 `+0.9 mAP` | structured local descriptor 融合稳定有效 |
| 无 pose 时是否仍涨 | `exp337` 三 seed 为 `-0.1/-0.1/-0.3 mAP` | 纯 CLIP-text query 在无空间先验时无增益 |
| 固定空间先验是否够用 | `exp340` canonical：global `58.8`，part-only `59.4`，equal-concat `59.5` | 通用人体布局已取得大部分局部增益 |
| 正确逐图 pose 是否贡献全部增益 | `exp357` correct `60.5`、cross-image pose `59.8`；差约 `0.7` | 正确对应只提供较小增量，不是全部来源 |
| 解剖通道身份是否关键 | `exp358` channel-shuffle `60.2`，相对 correct 仅 `-0.3` | 固定的“头/躯干/腿”语义身份几乎不是核心 |
| detach 是否重要 | 强系统 `exp320` un-detach 约 `-6.4 mAP`；固定 canonical un-detach 也严重下降 | 不允许让普通 part loss 直接塑形 backbone |
| GCN 是否必要 | Tiny/Small/Base 的 LGPA-only 与 LGPA+GCN 基本持平 | GCN 可留代码，不进入论文主方法 |
| matching 是否是根增益 | equal-concat 先有正增益，MaxSim 再追加约 `0.7～1.7 mAP` | MaxSim 是评测资产，不是方法创新 |
| standard global 是否被改善 | `exp370` 同机 e60：PBSR `54.4/63.7`，B0 `54.5/63.8` | 结构路由写回 global 已正式 NO-GO |

精确来源：

- `experiments/decisions.md` 中 `exp336/337` 三 seed 记录；
- `experiments/exp336_swin_lgpa_nopsg/monitor.md`；
- `experiments/exp337_swin_lgpa_nopose/design.md`；
- `experiments/exp340_swin_lgpa_fixedbands/design.md`；
- `experiments/exp357_pose_shuffle_ks/design.md`；
- `experiments/exp358_pose_channel_shuffle/design.md`；
- `experiments/exp370_pbsr/monitor.md`；
- `experiments/results.md` 中 LGPA-off、GCN、MaxSim 与 un-detach 对照。

## 必须纠正的旧表述

### 1. “增益来自 CLIP 文本”没有证据

`clip_part_head.py` 中冻结文本向量后仍有 trainable `text_proj/q_proj/k_proj/v_proj/out_proj`。因此六个文本向量可能只充当可区分的 query ID。

仓库已有 `exp340c` 的 fixed random query 设计，但没有训练结果。`exp353` 说明 random query 在另一套 un-detach 弱隔离设置中可以工作，却不能替代 detached、同 pose 条件下的单变量控制。

所以当前只能说：

- 无 pose 时 CLIP-text-alone 无效；
- 不能说 CLIP 在 pose 条件下已经被严格证明为零贡献。

2026-07-13 在 4090 找回了 `exp340c` 原始 checkpoint、train log 和三种 test log，补上了此前缺失的 canonical-pose 单变量证据：

| Query 来源 | global | part-only | equal-concat | 相对 global |
|---|---:|---:|---:|---:|
| frozen CLIP | 58.8/67.8 | 59.4/68.1 | 59.5/68.1 | +0.7/+0.3 |
| seed-42 fixed random | 58.8/67.8 | 59.8/68.7 | 59.9/68.7 | +1.1/+0.9 |

两臂均为 seed 1234、fixed canonical pose、detach、同协议 e120；两个 global 完全一致。原始 SHA：

- CLIP checkpoint：`5acb031981a0cccd0fcfad38fe161ee593589ebe004f93f99f128a37fee97b7f`
- random checkpoint：`885bc90e28c49b9660a2c509990cd6cef48c0fd7c028808f7a4d59638852af62`
- CLIP 三份 test log：`70aea744... / 9617c972... / 4248203...`
- random 三份 test log：`8ced06b9... / a60555cd... / d91d357d...`

这足以把“CLIP 词义是 LGPA 涨点来源”判负；random 并未损失增益，反而高 `0.4/0.6`。learned query 仍需与同一 random 初始化做单变量对照，但它只回答可学习性，不再影响去 CLIP 化裁决。

### 2. “正确逐图 pose 是增益核心”表述过强

`exp340/357/358` 联合说明：

- canonical 先验已取得约 `+0.7 mAP`；
- correct 相对 cross-image pose 只多约 `+0.7 mAP`；
- 解剖通道身份打乱只掉约 `0.3 mAP`。

更可信的解释是：

> LGPA 的大头来自稳定的局部结构读取与局部监督；正确逐图 pose 只提供较小的实例级路由校正。

### 3. `exp357/358` 还不是完整 inference intervention

这两项在训练时扰动 pose，但测试仍使用正确 pose。它们能回答“训练期 pose 对应是否重要”，不能完全替代在同一 checkpoint 上将测试 pose 改成 correct/canonical/shuffled/uniform/no-pose 的干预矩阵。

2026-07-13 已在同一 exp336 s0 checkpoint 上完成完整同权重干预：

| arm | mAP/R1 | 相对 global mAP | 相对 correct mAP |
|---|---:|---:|---:|
| global | 58.9908/67.3756 | — | -0.8449 |
| correct | 59.8357/67.6018 | +0.8449 | — |
| target-only | 59.8121/67.5113 | +0.8213 | -0.0236 |
| canonical | 59.7374/67.6471 | +0.7465 | -0.0984 |
| shuffled | 59.8037/67.7376 | +0.8129 | -0.0320 |
| uniform | 59.3689/66.8326 | +0.3781 | -0.4668 |
| no-pose | 59.4014/66.6063 | +0.4106 | -0.4344 |

五臂的 global descriptor SHA 完全一致；shuffled 在 query/gallery 内分别满足异 PID、无碰撞、严格双射。结果支持：

1. 局部融合增益真实存在；
2. part-specific structured spatial support 有贡献，因为 canonical/shuffled 明显高于 uniform/no-pose；
3. 当前图的精确 pose 对应几乎不是主来源，因为 shuffled/canonical 与 correct 的差只有 `0.03/0.10 mAP`；
4. scene-merged 与 target-only 只差 `0.024 mAP`，说明旁人 heatmap 不是该结果的主要混淆，但也没有证明目标人物精确 pose 具有独立优势；
5. 因此后续不能把 `anatomical support` 当作默认成立的解释，必须由 support-routing 对照另行证明。

## 当前描述子的真实成本

当前 `equal_concat` 由以下七个 768-D 块组成：

1. global；
2. pooled part；
3. 五个 individual parts。

因此总维度为 `7 × 768 = 5376-D`，且原 eval 路径读取 heatmap。单 seed train-only packing oracle 已证明：

- full 5376-D：`59.8357/67.6018`；
- fixed JL-768：`58.8011/67.5566`，paired-gain retention=`-0.2245`；
- train-only PCA-768：`59.9336/67.8733`，paired-gain retention=`1.1158`；
- train/eval path overlap=`0`，PCA 不读取 query/gallery 做 fit。

所以“LGPA 增益必然来自 5376-D 扩维”已被单 seed 证据否定；但该结论仍需三 seed paired 复核。现有证据仍没有证明：

- 测试完全无 pose 时能保留这 `+0.9 mAP`；
- 普通 ViT 上能够成立。

`exp335` 在修复 heatmap bug 后仍显示普通 ViT 的 equal-concat 在 e40 比 global 低约 `4.0 mAP`，所以跨骨干不能先写成既成事实。

## exp371 的证据任务

正式训练前必须先补三组廉价门禁：

1. **query 归因**：canonical 条件的 CLIP/fixed-random 已闭合；correct-pose learned-query 只剩低优先级优化归因，不再影响去 CLIP 化结论；
2. **inference intervention**：s0 已完成，三 seed 只在进入最终机制报告时补齐；
3. **同维 oracle**：s0 PCA-768 provisional GO，最终需三 seed paired 验证；
4. **support 归因**：必须 target-only、strict-path LOO、class-free、shared-mask、loss-matched，并直接对照 identity-only、slot permutation 与 exp123-style relational teacher。

若 target-only/part routing 不能比 controls 组织出更完整、更可靠的跨图 support，CASD 必须去掉 pose-specific/anatomical claim；若 CASD 又不能超过 exp123-style 强对照，则主方法判负。PCA-768 已允许同维首验，但不能把单 seed oracle 冒充最终三 seed 成本结论。
