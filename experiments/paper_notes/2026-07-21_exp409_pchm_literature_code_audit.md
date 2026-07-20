# exp409 PCHM 文献与代码审计（2026-07-21）

## exp408 后的对象约束

exp408 已实证 pose-indexed CLIP relation 可以被 Stage-2 学到，却没有提高最终 mAP。因此下一候选必须直接改变
identity descriptor 的训练几何；继续做 part KD、attention、router、prompt、temperature 或局部关系 loss 只会
重复拥挤对象。

## 近期近邻

1. CLIP-ReID及后续 VLM-ReID：主要把 image/text prompt 或 semantic alignment 写入 descriptor，并不使用
   pose×CLIP 联合规则离散替换标准 ReID triplet 的实际正负 index。
2. ProFD、PAFormer、MUVA 及 pose/part ReID：主要做 part decomposition、alignment、visibility、fusion 或
   auxiliary supervision；这些对象与 exp408 已验证失败的局部 binding 更接近。
3. Pose2ID：核心是跨姿态生成/融合和 identity centralization，不是 official PK batch 中的 train-only pair mining。
4. Pose-guided Enriched Feature Learning（CVPR 2026）：包含 pose-related decomposition 与跨 ID pose swap hard
   positive，封住“pose-guided hard positive”宽泛 claim，但没有发现其使用 frozen CLIP appearance consistency 与
   pose coverage rank 共同选择 standard ReID final-metric 正负 pair。
5. Composite-Attribute Person Re-Identification via Pose-Guided Disentanglement（CVPR 2026）：使用属性/身份
   组合与 batch triplet，近邻在 disentanglement，不同于 PCHM 的外生联合离散 miner。
6. 普通 batch-hard、CLIP hard-negative mining、pose-aware sampling分别是已知原子；PCHM不能声称这些原子新颖。

## 可以争取的窄差分

PCHM 的主张只能是：在标准遮挡 ReID 中，以增强后 pose coverage 与 frozen region-isolated CLIP visual
appearance 的无权 ordinal
联合排序，直接替换 final descriptor soft-margin triplet 的正负 index；CLIP不作为teacher target，pose/CLIP不进
测试路径，也不改变margin或loss weight。

按创新门判断：

- 问题门：PASS。明确针对跨遮挡同 ID 支持与同姿态同外观异 ID 混淆未被 batch-hard显式区分的问题；
- 机制门：CONDITIONAL PASS。必须确实替换 pair index；若变成 loss weighting/top-k调参则FAIL；
- 证据门：PASS。D0、pose-shuffle、CLIP-only、wrong-RGB/generic/zero和selected-pair统计可形成明确反事实。

因此只按 C 类会议候选推进，不做宽 novelty 声称。自然 e120 性能不过双门则立即封板。
