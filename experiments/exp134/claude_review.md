Now I have all the information needed for a complete review. Here is the report:

---

# exp134 代码审查报告：Changed-Pair Sparse LPCS

## 1. 审查结论：❌ 不允许启动

存在一个 **Critical** 级别 bug，会在 epoch 21（LPCS warmup 结束后第一次计算 loss 时）立即导致 `NameError` 崩溃。必须修复后重新审查。

---

## 2. Critical

### C1: `import math` 缺失 — 训练必崩

- **位置**: `processor/processor.py:339`
- **现象**: `_select_top()` 函数调用了 `math.ceil(values.numel() * ratio)`，但文件顶部（第 1-19 行）没有 `import math`。
- **影响**: exp134 配置 `PAIR_MODE='delta_top'` + `TOP_RATIO=0.25`，会在 warmup 结束后（epoch 21）第一次进入 `_compute_lpcs_loss` → `_select_top(ratio=0.25)` → 触发 `NameError: name 'math' is not defined`，训练直接崩溃。
- **为什么 exp133 不受影响**: exp133 的 `PAIR_MODE` 默认为 `'all'`，走 `else` 分支，`_select_top` 从未被调用。
- **修复**: 在 `processor/processor.py` 顶部添加 `import math`。

---

## 3. High

无。

---

## 4. Medium

### M1: 训练/测试 base_dist 权重硬编码耦合（沿袭自 exp133，非新引入）

- **训练**: `base_dist = 0.5 * (global_dist + kp_dist)` — 硬编码 1:1
- **测试**: `base_dist = (gw * global_dist + kw * kp_dist) / (gw + kw)` — 由 `CVK_GLOBAL_WEIGHT` / `CVK_KP_WEIGHT` 控制
- **当前状态**: 两个 config 中 `CVK_GLOBAL_WEIGHT = CVK_KP_WEIGHT = 1.0`（默认值），所以 train-test 一致
- **风险**: 如果未来有人只改 test 端权重而不改 train 端，会导致 train-test 不一致
- **本次判定**: 与 exp133 完全一致，exp133 审查已接受此边界，不阻塞 exp134

---

## 5. Low

### L1: `_select_top` 对 ties 不确定性

- `torch.topk` 在存在并列值时的选择是 nondeterministic 的
- 对于 float 型 `pair_change` 值，完全并列概率极低，实际不构成问题

### L2: `defaults.py` 中 `PAIR_TOP_RATIO` 默认值 1.0 的含义可能令人困惑

- 当 `PAIR_MODE='delta_top'` 且 `TOP_RATIO=1.0` 时，`_select_top` 会因 `ratio >= 1.0` 提前返回全 True，功能等价于 `'all'`
- 这本身是安全的（failsafe），但语义上可能让人误以为 `delta_top` 在生效
- 建议：在 logging 中加一行 warning，当 `delta_top` + `ratio >= 1.0` 时提示

---

## 6. 逐项检查表

| 检查维度 | 结论 | 说明 |
|---------|------|------|
| **单变量性** | ✅ 通过 | exp134 与 exp133 yml 的唯一差异是 `PAIR_MODE='delta_top'` + `TOP_RATIO=0.25` + `OUTPUT_DIR`。其余所有超参完全一致 |
| **默认行为是否被破坏** | ✅ 通过 | `defaults.py` 新增的 `PAIR_MODE='all'` + `TOP_RATIO=1.0` 使得所有未设置这两项的已有实验行为不变（走 `else` 全选分支） |
| **config 接线** | ✅ 通过 | `processor.py:184-185` 正确读取 `PAIR_MODE` 和 `TOP_RATIO`；`do_train` 启动时 logging（L237）正确打印两个新参数 |
| **train loss 接线** | ❌ **Critical 阻塞** | `_select_top` 内 `math.ceil` 会 crash（C1） |
| **test 路径** | ✅ 通过 | `cvk_residual` 测试路径（metrics.py:277-307）与 `PAIR_MODE` 无关，仅使用 `pair_residual_head` 做推理，exp133/134 完全一致 |
| **PairResidualScorer** | ✅ 通过 | `pair_adaptive_fusion.py` 未被修改，模型结构与 exp133 一致 |
| **teacher bank** | ✅ 通过 | `SupportCompleteBank` 配置一致，更新/replace 逻辑未改动 |
| **统计量是否足以验证机制** | ✅ 通过 | `lpcs_psr`（保留比例）和 `lpcs_pf`（保留 pair 的平均 focus 强度）两个统计量可清晰验证：(1) 稀疏路由是否生效（psr < 1.0）；(2) 选中的 pair 是否确实是 teacher-change 更大的（pf > 1.0） |
| **远程启动风险** | ⚠️ 需注意 | 修复 C1 后需 `git push` + 远程 `git pull` 同步代码；远程 Python 环境应与本地一致 |
| **梯度流** | ✅ 通过 | `base_dist` 全 detach，仅 `delta`（由 `lpcs_head` 产出）携带梯度。`pos_sel`/`neg_sel` mask 由 detach 的 `pair_weight` 派生，不影响梯度链 |
| **pair descriptor** | ✅ 通过 | `build_pair_descriptors` 未改动，6 维 descriptor 与 exp133 一致 |
| **design.md 与实现一致性** | ✅ 通过 | design.md 描述的 5 项改动全部在代码中对应：pair_mode config、top_ratio config、per-anchor top-k selection、psr/pf logging |

---

## 7. 最终结论

**不允许启动。** 原因：

1. **Critical C1** 会导致 exp134 在 epoch 21+ 必然崩溃（`NameError: name 'math' is not defined`）
2. 修复方式明确且单一：在 `processor/processor.py` 顶部添加 `import math`
3. 修复后，该改动不影响 exp133 或任何其他实验（`math.ceil` 仅在 `delta_top` 路径被调用）

**修复后需要重新审查确认修复正确性，然后方可启动。**
