I have all the information needed. Here is the review:

---

# exp137 Hard-Rank LPCS 启动前审查

## 1. 审查结论：允许启动

无 Critical 或 High 级问题。代码变更干净，单变量隔离清晰，默认行为安全。

## 2. Critical

无。

## 3. High

无。

## 4. Medium

无。

## 5. Low

**L1. `_select_top` 对空张量的理论边界**
- `values.numel() == 0` 时走 `<= 1` 分支，`torch.ones_like` 返回空 bool tensor → 后续 `pos[mask]` 仍为空 → 被外层 `continue` 兜住。不会 crash，但在逻辑上这个分支的"防护意图"并不是为空张量设计的。
- **影响**：None。外层已有 `pos.numel() == 0 or neg.numel() == 0` 三重 guard（pair routing 前一次、后一次、rank selection 后一次），空张量不可能真正进入 `_select_top`。
- **处置**：接受，无需修改。

**L2. config 中显式写出 `POSE_LPCS_PAIR_MODE: 'all'` 和 `POSE_LPCS_PAIR_TOP_RATIO: 1.0`**
- 这两个值等于 defaults.py 中的默认值，exp135 未显式写。
- **影响**：None。纯文档性质，让 config 更自解释。实际运行行为完全等价。
- **处置**：接受，是好实践。

## 6. 逐项检查表

### 6.1 单变量原则（vs exp135）

| 参数 | exp135 (lpcs_fix.yml) | exp137 (lpcs_hard_rank.yml) | 差异 |
|------|-----------------------|-----------------------------|------|
| POSE_LPCS_PAIR_MODE | default='all' | explicit 'all' | 无实质差异 |
| POSE_LPCS_PAIR_TOP_RATIO | default=1.0 | explicit 1.0 | 无实质差异 |
| **POSE_LPCS_RANK_MODE** | default='all' | **'hard_top'** | **唯一变量** |
| **POSE_LPCS_RANK_TOP_RATIO** | default=1.0 | **0.25** | **唯一变量（配套参数）** |
| OUTPUT_DIR | exp135_lpcs_fix | exp137_lpcs_hard_rank | 隔离输出 |
| 其他所有参数 | 同 | 同 | ✅ |

结论：满足单变量原则。rank_mode + rank_top_ratio 构成一个逻辑变量。

### 6.2 hard_top 排序聚合实现正确性

代码路径 (`processor.py:424-446`):

```
1. pair routing (pair_mode='all') → 保留所有 pos/neg
2. routed_pos_w / routed_neg_w 保存 routing 后计数
3. hard_top → pos: _select_top(pos, 0.25, largest=True) → 保留距离最大的 25% positive（最难正样本）
4. hard_top → neg: _select_top(neg, 0.25, largest=False) → 保留距离最小的 25% negative（最难负样本）
5. 只对这些 hardest pairs 计算 softplus ranking loss
6. loss 归一化：除以 rank_weight.sum()，量级稳定
```

- 正样本取最远（hardest positive）✅
- 负样本取最近（hardest negative）✅
- `_select_top` 的 `largest` 参数添加正确，默认值 `True` 不影响已有 pair routing 调用（line 408-409 未传 `largest`，沿用默认值，与旧行为一致）✅
- `max(1, ceil(N * 0.25))` 保证至少保留 1 个样本 ✅
- 梯度流：`pos`/`neg` 来自 `final_dist = base_dist + delta`，`base_dist` 全程 `.detach()`，梯度仅通过 `delta`（lpcs_head 输出）。Boolean indexing 对选中元素可导。✅

### 6.3 默认行为安全性

- `defaults.py` 新增：`RANK_MODE='all'`, `RANK_TOP_RATIO=1.0`
- 当 `rank_mode='all'`：代码走 `else` 分支创建 all-True masks → 等价于不做 rank selection
- 当 `rank_mode='all'` 且 `ratio=1.0`：`_select_top` 在 `ratio >= 1.0` 处直接返回 all-True
- 所有已有实验（不设置这两个参数）行为完全不变 ✅
- 启动校验 (line 198-201)：非法 rank_mode 或 hard_top 下 ratio 非法会 raise ValueError ✅

### 6.4 lpcs_rsr 诊断指标

```python
rank_selected_ratio = rank_selected_pair_count / max(selected_pair_count, 1e-6)
```

- `selected_pair_count` = pair routing 后、rank selection 前的样本数
- `rank_selected_pair_count` = rank selection 后的样本数
- 预期：exp137 中 `lpcs_rsr ≈ 0.25`；exp135 中 `lpcs_rsr = 1.0`
- 已添加到 log 输出 (`details['lpcs_rsr']`, line 735) ✅
- **如果 lpcs_rsr ≈ 1.0 则说明 hard_top 退化，需排查** — 设计文档已预判此风险 ✅

### 6.5 pair_selected_ratio / pair_focus 指标回归

- `selected_pair_count` 现在用 `routed_pos_w.numel()` 计数（rank selection 前），与旧代码中 `pos_w.numel()` 在 pair_mode='all' 时完全等价 ✅
- `selected_pair_weight_sum` 同理用 `routed_pos_w.sum()` ✅
- exp137 中 pair_mode='all'，所以 `lpcs_psr=1.0, lpcs_pf=1.0` 不变 ✅

### 6.6 启动日志确认项

Logger (line 246-253) 会输出:
```
[LPCS] enabled: ... pair_mode=all, top_ratio=1.0, rank_mode=hard_top, rank_top_ratio=0.25, ...
```
启动后应确认此行出现。

### 6.7 优化器/参数注册

无新增可训练参数。rank selection 是纯计算逻辑，不涉及新 `nn.Module`。lpcs_head 结构不变。✅

## 7. 最终结论

**允许启动。**

变更范围小且干净：defaults.py 加 2 行默认值，processor.py 在 `_compute_lpcs_loss` 内部加 ~25 行 rank selection 逻辑 + 诊断指标。单变量隔离清晰，默认行为无回归，诊断指标 `lpcs_rsr` 足以判断 hard ranking 是否真正激活。
