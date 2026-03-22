# exp141 Claude Review: Competition-Context LPCS

## 结论：不允许启动

---

## Blocking 问题

### [Critical] exp141 config 使用了完全不同的模板，至少有 8 个变量与 exp135 不同

exp141 config (`pose_psg_gcn_lpcs_comp_ctx.yml`) 与 exp135 config (`pose_psg_gcn_lpcs_fix.yml`) 使用了不同的 config 模板。exp135 显式设置了大量关键字段，exp141 缺失了这些字段，导致它们退回到 `defaults.py` 中的默认值，与 exp135 的设置不同。

逐项差异如下：

| 字段 | exp135 显式设置 | exp141 | defaults.py 默认值 | 差异影响 |
|------|----------------|--------|-------------------|---------|
| `MODEL.NAME` | `'transformer'` | **缺失** | `'resnet50'` | **致命：会构建错误的模型架构** |
| `MODEL.IF_LABELSMOOTH` | `'off'` | **缺失** | `'on'` | 训练端 loss 不同 |
| `MODEL.NO_MARGIN` | `True` | **缺失** | `False` | Triplet loss margin 不同 |
| `MODEL.PRETRAIN_HW_RATIO` | `2` | **缺失** | `1` | 预训练权重加载方式不同 |
| `INPUT.PIXEL_MEAN` | `[0.5, 0.5, 0.5]` | **缺失** | `[0.485, 0.456, 0.406]` | 输入预处理不同 |
| `INPUT.PIXEL_STD` | `[0.5, 0.5, 0.5]` | **缺失** | `[0.229, 0.224, 0.225]` | 输入预处理不同 |
| `SOLVER.WEIGHT_DECAY` | `1e-4` | **缺失** | `5e-4` | 正则化强度不同 |
| `SOLVER.WEIGHT_DECAY_BIAS` | `1e-4` | **缺失** | `5e-4` | 正则化强度不同 |
| `SOLVER.BIAS_LR_FACTOR` | `2` | **缺失** | `1` | 偏置学习率不同 |
| `TEST.NECK_FEAT` | `'before'` | **缺失** | `'after'` | 评估用的特征层不同 |

**其中 `MODEL.NAME='resnet50'`（默认值）几乎必定导致启动时构建 ResNet50 而非 Swin-Tiny，可能直接 crash 或生成完全无意义的结果。**

**修复方案**：将 exp141 config 改为以 exp135 config 为基础，仅改动 `POSE_LPCS_CONTEXT_MODE: 'comp_ctx'` 和 `OUTPUT_DIR`。最安全的做法是直接复制 `pose_psg_gcn_lpcs_fix.yml`，只添加一行 `POSE_LPCS_CONTEXT_MODE: 'comp_ctx'` 并修改 `OUTPUT_DIR`。

---

## 非阻塞提醒

### [Low] 训练/测试 competition context 粒度差异

训练中 `build_query_competition_descriptors` 在 batch 内（约 64 候选）计算 rank；测试中在全部 gallery（约 17000 候选）计算 rank。由于 rank 已经归一化到 [0, 1]，语义上是一致的（"当前 candidate 在本 query 候选集中的相对位置"），但分布特征可能有差异。这是设计层面的 trade-off，不是 bug。

### [Low] `_normalized_rank` 的 invalid 位置处理

- 训练中 `valid_mask=~eye` 排除对角线，invalid 位置用 inf/-inf 填充后参与排序，最终被 0.5 中性值替换
- 测试中 `valid_mask=None`（全部有效），不存在 invalid 位置
- 逻辑正确，无问题

### [Low] exp141 config 中多余的显式设置

exp141 config 显式设置了一些与 defaults.py 相同的值（如 `JPM: False`, `RE_ARRANGE: True`, `POSE_LPCS_PAIR_MODE: 'all'` 等）。这些不会造成 bug，但在修复 config 时应一并清理，保持与 exp135 一致的风格。

---

## 审查判断

### 1. 是否是相对 exp135 的单变量实验？

**否。** 当前 config 存在至少 10 个变量差异（见上方 Blocking 表格），远非单变量。修复 config 后（以 exp135 为基础只改 `CONTEXT_MODE`）才能成为单变量实验。

### 2. 是否真的不同于 exp139 query_ctx？

**是的，机制上确实不同。**

- exp139 (`query_ctx`)：追加 5 维 **query 级常量**（`row_mean, row_std, row_min, row_support_mean, row_gap_mean`），同一 query 内所有 pair 共享同一组 context 值
- exp141 (`comp_ctx`)：追加 5 维 **pair-specific 相对竞争位置**（`base_rank, kp_rank, support_rank, gain_rank, gain_zscore`），同一 query 内每个 pair 的 context 值不同

两者虽然最终都是 11 维 descriptor -> `PairResidualScorer`，但 context 的语义完全不同：
- `query_ctx` 回答"这个 query 整体有多难"
- `comp_ctx` 回答"这个 candidate 在本 query 中排第几"

这是有效的对比轴。

### 3. 训练/测试路径是否对称？

**对称。** 训练和测试使用同一个 `build_query_competition_descriptors` 函数，区别仅在于：
- 训练：`valid_mask=~eye`（排除自身对角线）
- 测试：`valid_mask=None`（全部 gallery 有效）

这是合理的，因为训练中 batch 内包含自己对自己的距离（=0），需要排除；测试中 query 和 gallery 不重叠，无需排除。

### 4. 是否无标签、无 oracle/label 泄漏？

**是的，无泄漏。** `build_query_competition_descriptors` 的所有输入（`base_dist, kp_dist, support_ratio`）均来自特征距离计算，不涉及任何标签。排名和 z-score 也都是纯统计量。

### 5. 默认行为是否被破坏？

**`POSE_LPCS_CONTEXT_MODE` 的默认值为 `'none'`**，在 `defaults.py` L257 正确设置。当不设置 `comp_ctx` 时，代码走 `else` 分支（L518-519），`lpcs_input_dim = 6`，不会追加任何 context。已有实验不受影响。

### 6. `build_query_competition_descriptors` 实现审查

- `_normalized_rank` 正确处理了 ascending/descending、invalid 填充、rank 归一化
- `gain_z`（z-score）正确排除 invalid 位置，invalid 位置被置为 0
- 返回的 5 维 tensor shape 正确：`(Q, G, 5)`
- rank 归一化使用 `(valid_count - 1).clamp(min=1.0)` 避免除零
- 无广播错误

---

## 修复建议

**唯一需要的修复**：用 exp135 config 作为基础，只改两处：

```yaml
# 在 exp135 config 基础上添加：
  POSE_LPCS_CONTEXT_MODE: 'comp_ctx'

# 修改输出目录：
OUTPUT_DIR: './log/occluded_duke/exp141_lpcs_comp_ctx'
```

修复后再次送审即可。
