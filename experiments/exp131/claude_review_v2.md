I have all the information needed. Here is my review:

---

# exp131 Claude 审查（二次）

## 结论

**允许启动。** 第一次审查指出的 Critical bug（queue 永远为空）已被正确修复。无新的阻塞性问题。相对 exp125 仍为单变量实验（唯一新增：`POSE_CSRD_QUEUE_SIZE: 256`）。

## 关键发现

### 1. Critical Bug 修复验证：通过

**原始 bug**：`processor.py` 原 line 768 使用 `kp_data.get('csrd_teacher_feats')`，而 `csrd_teacher_feats` 写入的是 `kp_aux_data`（line 386），`kp_data` 是模型原始输出，永远不含此 key → queue 永远为空。

**修复后代码**（line 767-768）：
```python
if kp_aux_data is not None:
    queue_teacher_feats = kp_aux_data.get('csrd_teacher_feats')
```

**验证**：`kp_aux_data = dict(kp_data)` 在 line 376 创建，`csrd_teacher_feats` 在 line 386 写入 `kp_aux_data`。Line 768 现在从同一个 dict 读取。修复正确。

### 2. 完整数据流追踪：通过

逐步追踪 queue 的写入→读出→使用全链路：

1. **Enqueue**（processor line 765-773）：`epoch > warmup` 时，从 `kp_aux_data` 取 teacher feats，连同 student feat/kp_feats/kp_weights/labels 一起入队。所有 `.detach()` 正确，无梯度泄漏。
2. **Dequeue**（processor line 398-401）：下一个 batch 的 forward pass 前，`_get_csrd_queue_payload()` 返回 detached 拷贝，写入 `kp_aux_data['csrd_queue']`。
3. **Loss 使用**（make_loss line 109-139, 226-293）：queue 数据参与 cross-distance 计算、pos/neg mask、delta_top focusing，与 batch 内 pairs 拼接后一起 distill。
4. **Stats 报告**（make_loss line 318-319, 526-528）：`csrd_qn`（queue size）和 `csrd_qr`（queue ratio）正确输出到日志。

### 3. Shape 兼容性：通过

- `pair_w_q = sqrt(w[B,K].unsqueeze(1) * w_q[Q,K].unsqueeze(0))` → `(B,Q,K)` ✓
- `euclidean_dist(feat_s[B,D], feat_q[Q,D])` → `(B,Q)` ✓
- `pos_mask_q = labels[B,1].eq(queue_labels[1,Q])` → `(B,Q)` ✓
- Per-anchor loop 中 `torch.cat(pos_parts_s)` 正确拼接 batch+queue pairs ✓

### 4. 梯度流：通过

- Queue 内所有 tensor 均 `.detach()`，无反向传播到旧 batch ✓
- `dist_s_q = euclidean_dist(feat_s, feat_q)` 中 `feat_q` 来自 detached queue，梯度仅流过 `feat_s` ✓
- Teacher 距离在 `_distill_subset` 中 `.detach()` ✓

### 5. 单变量隔离：通过

Config diff 干净。exp131 相对 exp125 唯一差异：
- `POSE_CSRD_QUEUE_SIZE: 256`（exp125 为 0 / 缺省默认 0）

其余参数完全一致：teacher、routing、target_mode、alpha、top_ratio、warmup。`defaults.py` 中 `POSE_CSRD_QUEUE_SIZE = 0` 不影响已有实验。

### 6. 边界情况处理：通过

- 首 batch after warmup：queue 为空 → `_get_csrd_queue_payload()` 返回 None → 纯 batch-only fallback ✓
- Queue 填满后 FIFO 截断：`[-csrd_queue_size:]` 正确保留最新 256 条 ✓
- Warmup 期间 enqueue 被 `epoch > CSRD_WARMUP` 门控，queue 保持空 ✓

## 必须先修的问题

无。

## 主要实验风险

1. **Softmax 稀释**（中风险）：queue 填满后，每个 anchor 的 neg 候选从 ~48 暴增到 ~304。`_distill_subset` 中 `F.log_softmax(s_logits, dim=0)` 在全量 batch+queue 上归一化，分布更平坦，每个 pair 的 KL 信号被稀释。这可能抵消 coverage 增益。**监控指标**：观察 `csrd_qr`（queue ratio）和 CSRD loss 值变化。

2. **Stale features**（低-中风险）：queue 中 student/teacher features 来自最多 4 个 batch 前（256/64）。随模型参数更新，这些 embedding 与当前 batch 不在完全相同的语义空间。staleness 程度中等，类似 MoCo 机制。

3. **训练速度下降**（低风险）：per-anchor loop 中 `torch.cat` 次数和 `euclidean_dist` 计算量约翻 5-6 倍（64→64+256），可能拖慢训练。显存增量 ~27MB 可忽略。
