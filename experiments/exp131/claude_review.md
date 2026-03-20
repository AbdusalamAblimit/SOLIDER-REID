Confirmed. Here is the full review:

---

# exp131 Claude 审查

## 结论

**禁止启动。** 存在一个阻塞性 bug，会导致 queue 永远为空，实验退化为 exp125 的精确复制。

## 关键发现

### 1. 单变量隔离：通过

Config diff 干净。`exp131` 相对 `exp125` 只增加了 `POSE_CSRD_QUEUE_SIZE: 256`，其余参数（teacher、routing、target_mode、alpha、top_ratio）完全一致。`OUTPUT_DIR` 也正确区分。defaults.py 的默认值 `POSE_CSRD_QUEUE_SIZE = 0` 不影响已有实验。

### 2. 阻塞性 bug：queue 永远不会被填充

**位置**: `processor/processor.py:768`

```python
queue_teacher_feats = kp_data.get('csrd_teacher_feats')  # ← BUG
```

**原因**: `csrd_teacher_feats` 在 line 386 被写入的是 `kp_aux_data`（`dict(kp_data)` 的浅拷贝），而不是 `kp_data`（模型原始输出）。`kp_data` 永远不含 `csrd_teacher_feats` 这个 key，所以 `.get()` 永远返回 `None`。

**后果链**:
- `queue_teacher_feats` 始终为 `None`
- `if queue_teacher_feats is not None:` 永远不满足
- `_enqueue_csrd_queue()` 永远不被调用
- queue 始终空 → `_get_csrd_queue_payload()` 始终返回 `None`
- `_compute_csrd_loss` 的 `queue_data` 始终为 `None`
- 日志中 `csrd_qn` / `csrd_qr` 永远不会出现

**修复**: 将 line 768 改为 `kp_aux_data.get('csrd_teacher_feats')` 即可。注意 `kp_aux_data` 在同一函数作用域内，Python 不会因为 `with amp.autocast` 块结束而失效。

### 3. loss 侧代码逻辑：无 bug

`_compute_csrd_loss` 中新增的 queue 逻辑本身是正确的：
- cross-distance 计算 `_aggregate_teacher_cross` 的 shape 正确：`(B,1,K) × (1,Q,K) → (B,Q,K)`
- queue 特征全部 `.detach()`，无梯度泄漏
- `dist_s_q = euclidean_dist(feat_s, feat_q)` 只对 `feat_s`（当前 batch）有梯度，正确
- pos/neg mask 对 queue labels 的广播方向正确
- `pair_delta_q` 的计算与 batch 内 `pair_delta` 保持一致
- stats 中新增 `queue_size` 和 `queue_ratio` 指标，有助于监控

### 4. processor 侧 queue 管理逻辑：无 bug（除上述 key 问题外）

- enqueue 在 `scaler.update()` 之后（梯度已完成），时序正确
- FIFO 截断 `[-csrd_queue_size:]` 正确
- warmup 保护 `epoch > CSRD_WARMUP` 与 loss 端一致
- 首 batch queue 为空时 `_get_csrd_queue_payload` 返回 `None`，正确 fallback 为纯 batch-only

## 必须先修的问题

| # | 严重度 | 文件 | 行 | 问题 | 修复 |
|---|--------|------|----|------|------|
| 1 | **Critical** | `processor/processor.py` | 768 | `kp_data.get('csrd_teacher_feats')` 应为 `kp_aux_data.get('csrd_teacher_feats')` | 替换变量名 |

修复后需要重新审查确认。

## 主要实验风险

即使 bug 修复后，仍需注意：

1. **Stale features**: queue 中的 student/teacher features 来自之前的 batch（最多 4 batch 前）。随着训练推进，模型参数变化导致 queue 中的 embedding 与当前 batch 的 embedding 不在同一"语义空间"——student_feat 和 kp_feats 的语义在几个 step 前就过时了。queue_size=256 ≈ 4 batch，staleness 中等。

2. **Softmax 稀释**: `_distill_subset` 中 KL 的 softmax 在 `dim=0` 上对 batch+queue relations 一起归一化。当 queue 填满后，一个 anchor 的 neg 候选从 ~48 个暴增到 ~300 个，softmax distribution 变得更平坦，每个 pair 的信号被稀释。这可能抵消 coverage 增益。

3. **显存额外开销可控**: 256 样本 × (768 + 17×768×2 + 17 + 1) ≈ 27MB，可忽略。但 `_compute_csrd_loss` 中 per-anchor loop 内做 `torch.cat` 的次数翻倍，可能略微拖慢训练速度。
