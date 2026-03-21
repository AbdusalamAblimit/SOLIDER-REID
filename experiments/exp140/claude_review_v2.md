# exp140 Clean Rerun 审查报告 v2

## 结论：允许启动

---

## 审查背景

首轮 `exp140` 在 `epoch 21+` 崩溃，根因是：

- `PairResidualConfidenceScorer` 原始实现输出 `sigmoid(conf_head)` 再传入 `F.binary_cross_entropy()`
- 在 AMP/autocast 下，`binary_cross_entropy` 不接受 float16 输入，运行时报错

本次审查目标：确认修复后的 logits 形式是否正确，以及是否仍然保持单变量、train/test 对称。

---

## 审查维度

### 1. AMP 修复是否正确

**结论：✅ 已正确修复**

修复前（崩溃版本）：
```python
conf = sigmoid(conf_head(feat))  # float16 under autocast
F.binary_cross_entropy(conf, target)  # AMP-unsafe → crash
```

修复后（当前代码）：

| 文件 | 行号 | 代码 | AMP 安全性 |
|------|------|------|-----------|
| `pair_adaptive_fusion.py` | 164 | `conf_logits = self.conf_head(feat)` — 输出原始 logits | ✅ safe |
| `processor.py` | 422 | `conf = torch.sigmoid(conf_logits)` — 仅用于计算 `delta` | ✅ safe（sigmoid 是 AMP-safe） |
| `processor.py` | 511 | `F.binary_cross_entropy_with_logits(conf_logits, conf_target)` | ✅ safe（`_with_logits` 变体在 AMP 下安全） |

**关键**：`binary_cross_entropy_with_logits` 在内部先做 log-sigmoid 操作，全程数值稳定，不需要外部预先 sigmoid。这是 PyTorch 官方推荐的 AMP-safe 写法。

### 2. Train/Test 对称性

**结论：✅ 完全对称**

| 路径 | 文件:行号 | 逻辑 |
|------|----------|------|
| 训练 | `processor.py:418-423` | `raw_delta, conf_logits = lpcs_head(desc)` → `conf = sigmoid(conf_logits)` → `delta = conf * raw_delta` |
| 测试 | `metrics.py:310-315` | `raw_delta, conf_logits = head(desc)` → `conf = sigmoid(conf_logits)` → `delta = conf * raw_delta` |

两个路径的语义完全一致：

1. 都从 `PairResidualConfidenceScorer.forward()` 获取 `(raw_delta, conf_logits)`
2. 都用 `sigmoid(conf_logits)` 得到 `conf`
3. 都用 `delta = conf * raw_delta` 计算最终修正
4. 都用 `final_dist = base_dist + delta` 得到最终距离

### 3. 单变量隔离（exp140 vs exp135）

**结论：✅ 真正单变量**

逐项比对 `pose_psg_gcn_lpcs_conf.yml`（exp140）与 `pose_psg_gcn_lpcs_fix.yml`（exp135）：

| 参数 | exp135 | exp140 | 差异 |
|------|--------|--------|------|
| `POSE_LPCS_HEAD_MODE` | 未设置 → 默认 `'residual'` | `'residual_conf'` | **唯一差异** |
| `POSE_LPCS_CONF_WEIGHT` | 未设置 → 默认 `0.25` | `0.25` | 等价（显式写出默认值） |
| 所有其他 LPCS 参数 | 相同 | 相同 | — |
| 所有其他 MODEL/SOLVER/INPUT 参数 | 相同 | 相同 | — |
| OUTPUT_DIR | `exp135_lpcs_fix` | `exp140_lpcs_conf` | 仅输出路径 |

实际改变的只有 `POSE_LPCS_HEAD_MODE`：
- `'residual'`：使用 `PairResidualScorer`（单输出 delta，无 conf）
- `'residual_conf'`：使用 `PairResidualConfidenceScorer`（双输出 delta + conf_logits，加 conf calibration loss）

### 4. 默认行为保护

**结论：✅ 不受影响**

当 `POSE_LPCS_HEAD_MODE` 为默认值 `'residual'` 时：

- `processor.py:424-428`：走 `else` 分支，`conf_logits = None, conf = None, delta = raw_delta`
- `processor.py:508-509`：`if conf is not None` → 不进入，`conf_loss` 永远不被计算
- `metrics.py:310`：`if lpcs_head_mode == 'residual_conf'` → 不进入，走原来的 `else` 分支
- `defaults.py:250`：默认值为 `'residual'`，已有实验不受影响

### 5. 其他正确性检查

| 检查项 | 结论 |
|--------|------|
| `conf_target` 无梯度泄漏 | ✅ `pair_change` 由 `.detach()` 特征计算 |
| `conf_target` 范围 [0, 1] | ✅ `1 - exp(-x/mean)` 当 `x≥0` 时输出在 [0, 1] |
| `desc` 无 backbone 梯度 | ✅ 所有输入特征均 `.detach()`，梯度只流经 `lpcs_head` 自身参数 |
| 零初始化保持恒等启动 | ✅ `delta_head` 和 `conf_head` 均零初始化 → 初始 `delta=0, conf=0.5` → `delta=0.5*0=0` |
| 模型 save/load | ✅ `lpcs_head` 是 `nn.Module` 子模块，自动进入 `state_dict` |
| 优化器覆盖 | ✅ `lpcs_head` 参数通过 `model.parameters()` 自动被优化器收集 |
| 评估时 head 绑定 | ✅ `evaluator.pair_residual_head = _eval_model.lpcs_head`（3 处一致） |

---

## Blocking 问题

**无**

---

## 非阻塞提醒

1. **`pair_weight` 用于两处**：同一个 `pair_weight`（基于 `pair_change` 归一化）同时用于排名损失的 pair 加权和 `conf_loss` 的 pair 加权。这是合理的设计选择（高 change 的 pair 更重要），但如果后续 `conf_loss` 表现异常，可以考虑给 `conf_loss` 使用均匀权重作为消融。

2. **`conf_target` 的 scale**：当 `pair_change = mean` 时，`conf_target ≈ 0.632`；当 `pair_change = 2*mean` 时，`conf_target ≈ 0.865`。这意味着大多数 pair 的 conf target 在 0.3-0.7 区间，不太会出现极端 0/1 的 target。这是 soft calibration 的预期行为。

3. **`lpcs_conf_weight = 0.25`**：这是 conf_loss 相对于主排名损失的权重。如果 conf_loss 太强导致 head 过早收缩到低 conf（保守策略），可以适当降低。如果 conf 学不到有意义信号（长期 ~0.5），可以适当提高。但这是后续调参范畴，不影响启动。

---

## 最终判断

修复后的 `residual_conf` 实现：

1. ✅ AMP 安全问题已正确解决（`binary_cross_entropy` → `binary_cross_entropy_with_logits`）
2. ✅ Train/test 路径完全对称
3. ✅ 相对 exp135 是真正单变量
4. ✅ 不破坏任何已有实验的默认行为
5. ✅ 无 blocking 问题

**允许作为 exp140 clean rerun 放行。**
