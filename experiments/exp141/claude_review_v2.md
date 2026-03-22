# exp141 Claude Review V2: Competition-Context LPCS（修复后二次审查）

## 结论：允许启动

---

## Blocking 问题

无。

首轮审查发现的唯一 blocking 问题（config 模板错误导致 ≥10 个变量差异）已完全修复。当前 config 已通过逐行 diff 确认。

---

## 非阻塞提醒

### [Low] 训练/测试 competition context 候选集规模差异

训练中 `build_query_competition_descriptors` 在 batch 内（约 64 候选，排除对角线后 63）计算 rank；测试中在全部 gallery（约 17000 候选）计算 rank。rank 已归一化到 [0, 1]，语义一致（"当前 candidate 在本 query 候选集中的相对位置"），但分布的分辨率差异很大（训练中只有 ~63 个离散位置 vs 测试中 ~17000 个离散位置）。这是设计层面的 trade-off，design.md 已将其列为风险项。不构成 blocking。

### [Low] `gain_zscore` 的 batch-vs-gallery 差异比 rank 更大

`gain_z`（z-score）的 mean 和 std 是在候选集上计算的。训练中 batch 内 64 样本的 gain 分布 vs 测试中 17000 样本的 gain 分布可能差异更大。但这同样是设计上预期内的 trade-off。

---

## 逐项审查

### 1. Config 是否相对 exp135 真正单变量？

**是。** `diff` 输出仅有两处差异：

```
28a29
>   POSE_LPCS_CONTEXT_MODE: 'comp_ctx'
76c77
< OUTPUT_DIR: './log/occluded_duke/exp135_lpcs_fix'
---
> OUTPUT_DIR: './log/occluded_duke/exp141_lpcs_comp_ctx'
```

所有其他字段（MODEL、INPUT、DATASETS、DATALOADER、SOLVER、TEST）完全一致。单变量原则严格满足。

### 2. `build_query_competition_descriptors()` 训练/测试对称性

**对称。** 训练和测试使用同一函数，语义差异合理：

| 维度 | 训练侧 (processor.py:419-427) | 测试侧 (metrics.py:312-318) |
|------|------|------|
| 函数 | `build_query_competition_descriptors` | 同一函数 |
| `base_dist` | `base_dist.detach()` (batch x batch) | `base_dist[start:end]` (chunk_q x gallery) |
| `kp_dist` | `kp_dist.detach()` (batch x batch) | `kp_dist[start:end]` (chunk_q x gallery) |
| `support_ratio` | `support_ratio.detach()` (batch x batch) | `support_ratio[start:end]` (chunk_q x gallery) |
| `valid_mask` | `~eye` (排除自身) | `None` -> 全 True (q/g 不重叠，无自身) |
| `base_dist` 公式 | `(gw*global + kw*kp) / (gw+kw)` | 同一公式，同一 gw/kw 来源 |

训练排除对角线（因为 batch 内 q=g 包含自身），测试无需排除（query 与 gallery 不重叠）。语义对称，无遗漏。

关键确认：测试侧的 chunking（每次 256 queries）不影响 rank 计算，因为 `_normalized_rank` 沿 dim=1（gallery 维度）排序，每个 query 始终看到完整的 gallery。

### 3. comp_ctx 与 query_ctx (exp139) 的机制差异

**确实不同。**

| | exp139 `query_ctx` | exp141 `comp_ctx` |
|---|---|---|
| 追加 5 维内容 | `row_mean, row_std, row_min, row_support_mean, row_gap_mean` | `base_rank, kp_rank, support_rank, gain_rank, gain_zscore` |
| 语义 | 同一 query 内所有 pair 共享同一组值（query 级常量广播） | 同一 query 内每个 pair 有不同的值（pair 级相对位置） |
| 回答的问题 | "这个 query 整体有多难" | "这个 candidate 在本 query 中排第几" |
| 信息粒度 | 粗（per-query） | 细（per-pair） |

这是有效的对比轴，两者的 input_dim 都是 11 但承载的信息完全不同。

### 4. 无标签/无 oracle 泄漏

**确认无泄漏。**

`build_query_competition_descriptors` 的所有输入：
- `base_dist`: 来自特征距离计算（detached）
- `kp_dist`: 来自 common-support 距离计算（detached）
- `support_ratio`: 来自 keypoint weight 计算（detached）
- `valid_mask`: 纯布尔掩码（排除自身或全 True）

排名和 z-score 都是纯统计量，不依赖任何标签信息。训练中的 `labels` 仅在 `_compute_lpcs_loss` 内部用于构建 `pos_mask`/`neg_mask`（triplet loss 的标准用法），不参与 context 构建。

### 5. 默认行为是否被破坏

**未被破坏。**

- `defaults.py` L257: `_C.MODEL.POSE_LPCS_CONTEXT_MODE = 'none'`
- `pose_backbone_model.py` L515-519: 当 `context_mode` 不是 `query_ctx`/`comp_ctx` 时，`lpcs_input_dim = 6`，不追加 context
- `processor.py` L194, L212-213: 验证 context_mode 合法性，默认 'none' 跳过 context 追加
- `metrics.py` L304: `getattr(self.cfg.MODEL, 'POSE_LPCS_CONTEXT_MODE', 'none')` 默认 'none' 跳过

所有已有实验（exp135 等不设此字段的 config）不受影响。

### 6. `_normalized_rank` 实现审查

- `dim=1` 排序：正确，沿 gallery/candidate 维度排序
- ascending（距离越小排名越前）用 `inf` 填充 invalid；descending（support_ratio 越大排名越前）用 `-inf` 填充 invalid：正确
- `rank / (valid_count - 1).clamp(min=1.0)` 归一化到 [0, 1]：正确处理了只有 1 个 valid 元素的退化情况
- invalid 位置返回 0.5 中性值：正确
- 无数值不稳定风险

### 7. 模型维度一致性

- `pose_backbone_model.py` L516-517: `comp_ctx` -> `lpcs_input_dim = 11`
- `PairResidualScorer.__init__`: `nn.Linear(input_dim=11, 32)` -> 第一层接受 11 维
- 训练侧: `desc` = 6 (base) + 5 (comp_ctx) = 11 维
- 测试侧: `desc` = 6 (base) + 5 (comp_ctx) = 11 维
- 自检中已验证 `concat = [4, 4, 11]`

### 8. 梯度流

- `base_dist.detach()`, `kp_dist.detach()`, `support_ratio.detach()` -> 所有 descriptor 输入无梯度
- `comp_ctx` 从 detached 输入构建 -> 无梯度
- 梯度仅通过 `lpcs_head` 的 MLP 权重流动，与 exp135/exp139 行为一致

---

## 最终判断

当前 exp141 **已经是相对 exp135 的单变量 clean run**。唯一差异为 `POSE_LPCS_CONTEXT_MODE: 'comp_ctx'`。

- 首轮 blocking（config 模板错误）：已修复
- 训练/测试对称性：确认
- 与 exp139 的机制区分：确认
- 无标签泄漏：确认
- 默认行为安全：确认
- 数值/维度正确性：确认

**审查通过，可以启动训练。**
