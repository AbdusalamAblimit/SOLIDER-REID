Good. Here is the complete review:

---

# Claude 审查结论

## 结论
- **不允许启动**

## Blocking

**B1: 测试时 cvk_residual 路径维度不匹配 — 必崩（Critical）**

模型初始化（`pose_backbone_model.py:511-516`）在 `context_mode='query_ctx'` 时设 `input_dim=11`，`PairResidualScorer` 的第一层 Linear 为 `Linear(11, 32)`。

但测试时评估路径（`utils/metrics.py:290-304`）只调用了 `build_pair_descriptors()` 构建 6 维描述子，然后直接喂进 `head(desc.view(-1, desc.shape[-1]))`。`desc.shape[-1] = 6`，而 head 期望 11 维输入。

这会在第一个 eval period（epoch 10）触发 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (N x 6) and (11 x 32)`，实验必崩。

**B2: 测试时无法计算 query context — 设计缺陷（Critical）**

训练时 `_compute_lpcs_loss`（`processor.py:402-420`）计算 query context 依赖于 ground-truth labels（`row_pos_mean`、`row_neg_mean`、`row_margin` 均由 label equality 决定）。测试时没有 labels，无法计算这 5 维 context。即使用 zeros 补齐维度避免崩溃，模型训练时学习的是 11 维联合空间，测试时 5 维恒为 0 会导致：
- 训练-测试分布严重不对齐
- 训练时 scorer 可能过度依赖 context 维度，测试时全部失效
- 实验指标不可信——无法判断是方法无效还是训练-测试不对称导致的退化

**必须在启动前解决这个基本设计问题。** 可选方案：
1. 测试时用无标签代理（如用距离排名估算 pseudo-pos/neg mean），但这本身也是一个新的设计变量
2. 在 desc 中补零并接受不对称性，但需在 design.md 中明确说明这是实验约束并调整预期
3. 在训练时也用无标签代理统计（如 row-wise mean、row-wise std），这样训练测试一致

## Major

- 无

## Medium

**M1: `same_label` 冗余计算**

`processor.py:404` 和 `processor.py:424` 都执行 `same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))`。第一次是在 `query_ctx` 分支内部，第二次是在外部。由于 Python 无块级作用域，第二次赋值覆盖第一次的值（虽然结果相同）。不是 bug，但容易造成混淆。建议将 line 404 改为直接使用局部变量名以避免误读。

**M2: line 404 多重赋值风格混乱**

`pos_mask_float = same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))` — 这行同时给 `pos_mask_float` 和 `same_label` 赋了同一个 bool tensor，紧接着 line 405 又重新赋值 `pos_mask_float`。虽然功能正确，但可读性差，容易被未来维护者误读。

## Low

**L1: design.md 未提及测试时行为**

design.md 详细描述了训练时的 11 维 descriptor 构成和日志指标，但完全没有提及测试时 `cvk_residual` 路径如何处理 context 维度。实验设计文档应当覆盖"训练时做什么 + 测试时怎么评"的完整数据流。

**L2: 单变量原则验证**

对比 exp135 config（`pose_psg_gcn_lpcs_fix.yml`）和 exp139 config（`pose_psg_gcn_lpcs_query_ctx.yml`），唯一差异是 `POSE_LPCS_CONTEXT_MODE: 'query_ctx'`（exp135 缺省为 `'none'`）和 `OUTPUT_DIR`。单变量原则满足。

## 建议

**不可启动。** 必须先解决 B1 和 B2：

1. **最优方案**：重新设计 context 特征使其不依赖 labels。例如用 row-wise distance statistics（row mean、row std、row min、row max、row support mean）替代 pos/neg 统计。这样训练和测试一致，不需要 labels。
2. **最低限度修复**：在 `utils/metrics.py` 的 `cvk_residual` 路径中，检查 head 的 `input_dim`，若为 11 则补 5 维 zeros。同时在 design.md 中明确说明：这是一个"训练时有 oracle context、测试时无 context"的对照实验，测试指标应被理解为下界。
3. 修复后需重新提交审查。
