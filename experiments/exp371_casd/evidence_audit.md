# exp371 CASD：LGPA 证据审计

## 结论先行

LGPA 的稳定正信号是真实的，但当前只能严谨表述为：

> 在 Swin-Tiny 上，detached 的姿态定位局部描述子与 global 做标准拼接后，相对同一 checkpoint 的 global 描述子稳定提高约 `+0.9 mAP`。

它尚不能被表述为 CLIP 文本语义增益、正确逐图姿态的全部因果增益、标准 768-D global 增益、GCN 增益或 matching 增益。

## 已成立的证据

| 问题 | 证据 | 当前可下结论 |
|---|---|---|
| LGPA 是否有稳定增益 | `exp336` 三 seed 的 equal-concat/global 差均为 `+0.9 mAP` | pose-aware local descriptor 融合稳定有效 |
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

## 当前描述子的真实成本

当前 `equal_concat` 由以下七个 768-D 块组成：

1. global；
2. pooled part；
3. 五个 individual parts。

因此总维度为 `7 × 768 = 5376-D`，且 eval 路径读取 heatmap。现有证据还没有证明：

- 768-D 固定维度下能保留这 `+0.9 mAP`；
- 测试完全无 pose 时能保留这 `+0.9 mAP`；
- 普通 ViT 上能够成立。

`exp335` 在修复 heatmap bug 后仍显示普通 ViT 的 equal-concat 在 e40 比 global 低约 `4.0 mAP`，所以跨骨干不能先写成既成事实。

## exp371 的证据任务

正式训练前必须先补三组廉价门禁：

1. **query 归因**：canonical 条件的 CLIP/fixed-random 已由找回的原始日志闭合；再在 exp336 correct-pose 协议比较 frozen CLIP、fixed random、learned query ID，确认结论不依赖 canonical；
2. **inference intervention**：同一 `exp336` checkpoint 比较 correct/canonical/shuffled/uniform/no-pose；
3. **同维 oracle**：将已验证 5376-D teacher descriptor 压到 768-D，检查能否保留至少 80% 的相对增益。

若 correct pose 不能比 controls 组织出更完整、更可靠的跨图 support，CASD 的 pose-privileged 故事立即失去地基；若 768-D oracle 也失败，则同维化只能降为长期问题，不能与首个机制实验捆绑。
