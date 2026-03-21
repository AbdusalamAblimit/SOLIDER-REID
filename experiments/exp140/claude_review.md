# exp140 代码审查报告

## 结论：允许启动

---

## 一、单变量判断

**判断：是真正的单变量实验。**

逐行对比 exp135 config (`pose_psg_gcn_lpcs_fix.yml`) 与 exp140 config (`pose_psg_gcn_lpcs_conf.yml`)，差异仅有：

| 配置项 | exp135 (默认值) | exp140 |
|--------|----------------|--------|
| `POSE_LPCS_HEAD_MODE` | `'residual'` (default) | `'residual_conf'` |
| `POSE_LPCS_CONF_WEIGHT` | `0.25` (default) | `0.25` (显式写出) |
| `OUTPUT_DIR` | `exp135_lpcs_fix` | `exp140_lpcs_conf` |

`POSE_LPCS_CONF_WEIGHT` 在 `head_mode='residual'` 时完全不生效（processor.py L507 的 `if conf is not None:` 分支不会进入），所以 exp140 显式写 0.25 与不写等价。唯一有效变量是 `HEAD_MODE: residual → residual_conf`。

---

## 二、Train / Test 路径对称性

**判断：对称，无遗漏。**

| 步骤 | 训练路径 (processor.py L418-422) | 测试路径 (metrics.py L310-314) |
|------|----------------------------------|-------------------------------|
| 读取 mode | `lpcs_head_mode` 在函数外层捕获 (L186) | `getattr(self.cfg.MODEL, 'POSE_LPCS_HEAD_MODE', 'residual')` (L285) |
| 前向 | `raw_delta, conf = lpcs_head(desc)` | `raw_delta, conf = head(desc)` |
| 合并 | `delta = conf * raw_delta` | `delta = conf * raw_delta` |
| 最终距离 | `final_dist = base_dist + delta` | `corrected = base_dist[start:end] + delta` |

测试路径同样按 chunk (256) 处理，在 `cvk_residual` 分支内完整处理了 `residual_conf` case。模型对象的传递链：`evaluator.pair_residual_head = getattr(_eval_model, 'lpcs_head', None)` 在三处（L1263, L1291, L1337）都有设置。

---

## 三、`residual_conf` 是否在测试时真正生效

**判断：是的，完整生效。**

追踪链：
1. `pose_backbone_model.py` L520-525：当 `head_mode='residual_conf'` 时，模型创建 `PairResidualConfidenceScorer` 实例作为 `self.lpcs_head`
2. `processor.py` L1263/1291/1337：`evaluator.pair_residual_head = getattr(_eval_model, 'lpcs_head', None)` 把训练好的 head 传给 evaluator
3. `metrics.py` L285：读取 `lpcs_head_mode` 从 config
4. `metrics.py` L310-314：当 mode 为 `residual_conf` 时，调用 head 获得 `(raw_delta, conf)` 并做 `delta = conf * raw_delta`

没有路径断裂。

---

## 四、`conf_target` 是否引入 label/oracle 泄漏

**判断：无泄漏。**

`conf_target` 的计算（processor.py L433）：
```python
conf_target = 1.0 - torch.exp(-pair_change / pair_change[~eye].mean().clamp(min=1e-6))
```

其中 `pair_change = |teacher_dist - base_dist|`（L399），来源于：
- `base_dist`：当前 student 的 global+kp 加权距离（label-free）
- `teacher_dist`：support-complete bank 替换后的 global+kp 加权距离（label-free）

两者都是在 batch 内 features 上计算的欧式距离，不使用 labels。

Labels 仅在 `pos_mask / neg_mask`（L429-431）中使用，用于 LPCS 排序损失的正负对挖掘——这与 exp135 完全一致，不是新引入的。

测试时，`conf` 完全由 learned head 的 forward 产出，不依赖任何 oracle。

---

## 五、默认行为是否被破坏

**判断：未破坏。**

- `defaults.py` L250：`POSE_LPCS_HEAD_MODE = 'residual'`，非 exp140 实验使用默认值，走原有 `PairResidualScorer` 路径
- `defaults.py` L251：`POSE_LPCS_CONF_WEIGHT = 0.25`，在 `head_mode='residual'` 时完全不使用（L507 的 `if conf is not None:` 不会进入）
- 不使用 LPCS 的实验（`POSE_LPCS=False`）完全不受影响

---

## 六、实现问题检查

### Blocking 问题

**无。**

### 非阻塞提醒

1. **`pair_weight` 对 `conf_loss` 的双重加权**（Low）
   - L509-510：`conf_loss` 被 `pair_weight` 加权
   - `pair_weight = pair_change / mean(pair_change)`
   - `conf_target` 本身也由 `pair_change` 导出
   - 效果：高 `pair_change` 的 pairs 既有高 `conf_target`，又有高 loss 权重。双重强调可能导致 conf head 过度偏向高改动 pairs
   - 不会导致错误，但如果 `lpcs_cf` 长期接近 0.5-0.6 不动，这可能是原因之一

2. **共享 backbone 的两个 head 可能互相竞争**（Low）
   - `PairResidualConfidenceScorer` 中 `backbone`（两层 MLP）被 `delta_head` 和 `conf_head` 共享
   - 排序损失（通过 delta）和校准损失（通过 conf）的梯度都流回同一个 backbone
   - 如果两个目标冲突，backbone 的优化方向可能不稳定
   - 这是设计选择，不是 bug

3. **参数量差异极小**（Info）
   - `PairResidualScorer`：6→32→32→1 = 1313 params
   - `PairResidualConfidenceScorer`：shared 6→32→32 backbone + delta Linear(32,1) + conf Linear(32,1) = 1346 params
   - 多了 33 个参数（一个 Linear(32,1)），可忽略

4. **初始化行为正确**（Info）
   - `delta_head` 和 `conf_head` 都 zero-init（L156-159）
   - 初始 conf ≈ sigmoid(0) = 0.5，raw_delta ≈ tanh(0) × 0.5 = 0
   - 初始 delta = 0.5 × 0 = 0，与 exp135 的初始 delta = 0 一致
   - 不会引入初始化偏差

5. **梯度流正确**（Info）
   - L385-388：backbone features 全部 `.detach()`，LPCS loss 梯度只流经 `lpcs_head`
   - 与 exp135 行为一致

6. **优化器自动包含新参数**（Info）
   - `make_optimizer.py` 遍历 `model.named_parameters()`
   - `PairResidualConfidenceScorer` 作为 `self.lpcs_head` 属于 model，其参数自动被遍历和加入优化器

---

## 总结

| 审查维度 | 结论 |
|----------|------|
| 单变量 | ✅ 唯一有效变量：`HEAD_MODE: residual → residual_conf` |
| Train/test 对称 | ✅ 两端同样处理 `(raw_delta, conf)` → `conf * raw_delta` |
| 测试生效 | ✅ head 类型、config 读取、forward 路径完整 |
| 无 label/oracle 泄漏 | ✅ `conf_target` 仅依赖 teacher-student 距离差 |
| 默认行为保护 | ✅ 不影响任何已有实验 |
| 实现正确性 | ✅ 无 blocking issue，4 个 low/info 级提醒 |
