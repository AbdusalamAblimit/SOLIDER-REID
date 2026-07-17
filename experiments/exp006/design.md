# exp006: L2-Normalized Part-Global Feature Fusion

## 动机

exp001 的关键发现：
- part_only mAP 57.5% (+0.9%) — 最好
- concat_scaled mAP 57.1% (+0.5%) — 1/N scaling 稀释 part 信号
- equal_concat mAP 57.0% (+0.4%) — 维度不匹配(768 vs 5×768)导致 global 被淹没
- global_only mAP 57.1% (+0.5%) — global 本身也提升了

**问题**: Global feature (768-dim) 和 Part features (5×768-dim) 的 L2 norm 不匹配。在余弦距离度量下，维度更多的一方会主导相似度计算。

## 方案

**Feature-level L2 normalization before concatenation**:
1. 对 global_feat 做 L2 normalize → unit sphere
2. 对每个 part_feat 做 L2 normalize → unit sphere
3. 拼接: [global_norm, part1_norm, ..., part5_norm]
4. 再做整体 L2 normalize（如果 FEAT_NORM=yes）

这样每个 sub-feature 对最终距离的贡献相等，不受维度差异影响。

## 训练改动

**无需改动训练代码**。只改 test-time 的特征组合方式。在 `test_feature_modes.py` 中加一个 `normalized_concat` 模式。也可以改 `pose_model.py` 的 inference path。

## 配置

新增 POSE_TEST_FEAT 选项: 'norm_concat'

## 与 baseline 对比

这个实验不需要重新训练。直接用 exp001 的已训练模型，改变 test-time 特征组合方式即可。
