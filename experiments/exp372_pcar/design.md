# 实验 exp372：PCAR（Pose-Conditioned Attention Residual）新颖性门禁

## 动机

历史 LGPA 在 Swin 上有稳定约 `+0.9 mAP` 的局部结构收益，但 exp371 Gate B/C 同时表明：correct pose 相对 shuffled/canonical 只高 `+0.0320/+0.0984 mAP`，逐图精确姿态不是主要燃料。用户提出一个与旧 LGPA head 正交的新候选：不再做后置 part assembly，而是在官方 CLIP-ReID 的真实 CLIP ViT 视觉塔内部，用姿态残差修改少量 self-attention heads/layers，同时保留标准 global descriptor。

## 核心假设

候选公式为：

\[
L' = L + \gamma\,[B(P_{instance})-B(P_{canonical})],
\]

其中 `L` 是原始 CLIP attention logits，`B` 将姿态映射为 token-pair bias，`gamma` 零初始化；未选中的 heads/layers 保持原始 CLIP attention，作为语义锚。

原假设是：相对 canonical layout 的实例残差可能比普通 pose mask 更能隔离“逐图姿态变化”，并避免把固定人体布局重复包装成创新。

## 技术方案（仅在新颖性门禁通过后）

1. 以官方 `Syliz517/CLIP-ReID` 的 ViT-B/16 Occluded-Duke checkpoint 为 matched baseline；
2. 在 `ResidualAttentionBlock.attention()` 处显式传入每批次 `[B×heads,129,129]` additive mask；
3. 只修改预注册的少量 heads/layers，未选 head 的 mask 恒为零；
4. `gamma=0` 时 descriptor 与官方模型严格一致；
5. 测试输出仍为官方 `768-D CLS + 512-D projected CLS = 1280-D` global descriptor，不增加 part token、part branch 或特殊 matching；
6. 六臂为 `untouched/correct/canonical/shuffled/uniform/no-pose`，且 correct 必须相对 untouched 至少 `+0.5 mAP`、相对最强 pose control 至少 `+0.3 mAP`，才允许完整训练。

## 对照组

- 官方 untouched CLIP-ReID；
- `gamma=0` parity；
- fixed canonical；
- path-hash 固定、按 visible-kp-count/body-scale/pose difficulty 分箱的 cross-image derangement；
- uniform foreground；
- no-pose；
- 若进入最小适配，还应有 correct-train 与 shuffled-train 的 matched 2×2 train/eval 对照，避免只做 OOD inference control。

## 预期结果

在性能门禁之前先过新颖性门禁：相对 PAFormer、KPR、ProFD、PeVL、PAAB、MUVA，以及仓库 exp012/052/143/354，PCAR 必须是不可归约的机制差异。若最终只是普通 additive pose bias、pose prompt 或 attention mask 的中心化参数化，则按预注册规则直接 NO-GO，不启动训练。

## 风险与失败解释

1. `B(Pinstance)-B(Pcanonical)` 可能只是一种中心化参数化，不扩大函数族；
2. 若 canonical bias 对 softmax 某一行是常数，它甚至会被 softmax 完全消去；
3. PeVL/PAAB/MUVA 可能已覆盖“pose/part mask 修改 CLIP/ViT attention”的核心操作；
4. exp371 correct≈shuffled/canonical 说明实例姿态残差缺乏内部燃料证据；
5. frozen intervention 的性能失败不能否定可训练 adapter，因此 frozen screen 只能做 parity、attention 变化、尺度和安全审计，不能用手工 alpha 的 mAP 直接杀方法。

## 最终状态

**新颖性 Gate NO-GO。** 详见 `literature_novelty.md` 与 `codex_review.md`。未改代码、未下载官方 checkpoint、未占用 3090/4090、未进入六臂性能 screen。
