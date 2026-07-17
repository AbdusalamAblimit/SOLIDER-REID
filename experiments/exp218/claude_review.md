# exp218: PACI (Pose-Anchored Compositional Identity) — Claude Review

## 审查范围

a. `design.md` — 合理性、单变量原则、假设清晰度
b. `model/modules/part_prototype_bank.py` — 逐行审查
c. `processor/processor.py` — PACI 相关代码（初始化、consistency loss、bank update）
d. `config/defaults.py` — PACI 默认值
e. 梯度流分析 — detached kp_feats 是否导致 no-op
f. OA-SD 交互 — 是否冲突

---

## 发现

### Critical

#### C1: `num_classes` 未定义 — 运行时 NameError

**文件**: `processor/processor.py`, line 429

```python
paci_bank = PartPrototypeBank(
    num_classes=num_classes, num_parts=17, feat_dim=feat_dim,
    ...
```

变量 `num_classes` 在 `do_train` 函数作用域中从未定义。其他使用类似 bank 的代码（LTCS line 83, LPCS line 111）使用 `num_train_classes`，该变量仅在各自的 `if` 块内定义。

**当 PACI 启用时，将立即在训练开始前抛出 `NameError: name 'num_classes' is not defined`。**

**修复**: 在 PACI 初始化块中计算 `num_train_classes`:
```python
if paci_enabled:
    from model.modules.part_prototype_bank import PartPrototypeBank
    base_m = model.module if hasattr(model, 'module') else model
    feat_dim = base_m.in_planes
    num_train_classes = len(set([d[1] for d in train_loader.dataset.dataset]))
    paci_bank = PartPrototypeBank(
        num_classes=num_train_classes, ...
```

---

### High

（无）

---

### Medium

#### M1: 缺少 bank 统计日志

`PartPrototypeBank` 有 `stats()` 方法但 processor 从未调用。无法在日志中看到 bank coverage、平均更新次数等关键信息。建议在每个 epoch 结束时或至少每 `LOG_PERIOD` 次迭代时输出 bank stats，例如：

```python
if n_iter % log_period == 0 and paci_bank is not None:
    s = paci_bank.stats()
    logger.info(f"[PACI] coverage={s['coverage']:.2%} avg_count={s['avg_count']:.1f}")
```

这对监控 bank 是否正常工作至关重要。

#### M2: `update()` 中双层 Python 循环 + `.item()` 的性能

`update()` 内部的 `for i in range(B): for k in range(17):` 循环 + `count = self.update_count[label, k].item()` 每次调用产生 B*17 次 GPU->CPU 同步。对于 batch_size=64，这是 1088 次 `.item()` 调用。

这不会导致错误，但可能增加 ~10-20ms/step 的开销。可以改为向量化实现：
```python
for i in range(B):
    label = labels[i].item()  # 1 sync per sample
    vis_mask = visible[i]     # (17,) bool
    if not vis_mask.any():
        continue
    is_first = (self.update_count[label] == 0) & vis_mask
    is_update = (self.update_count[label] > 0) & vis_mask
    self.bank[label][is_first] = kp_feats[i][is_first]
    self.bank[label][is_update] = (self.momentum * self.bank[label][is_update] +
                                    (1 - self.momentum) * kp_feats[i][is_update])
    self.update_count[label][vis_mask] += 1
```
将 `.item()` 调用从 B*17 降至 B。不阻塞但值得优化。

#### M3: consistency loss 中 `paci_loss.item() > 0` 的 GPU 同步

Line 1089 在热路径中调用 `.item()` 进行条件判断，触发 GPU 同步。可以改为无条件加到 loss 上（当 loss 为 0 时对梯度无影响）：
```python
loss = loss + paci_weight * paci_loss
details['paci'] = paci_loss.item()  # 仅用于日志
```

---

### Low

#### L1: Warmup 条件是 `epoch > paci_warmup` 而非 `epoch >= paci_warmup`

当 `PACI_WARMUP=5` 时，consistency loss 从 epoch 6 开始。这是可以接受的行为，但如果意图是 "5 个 epoch 的 warmup，第 6 个开始"，应该更明确地文档化。（当前实现是正确的。）

#### L2: `get_negative_prototypes` 中的 per-sample 循环

Line 117-124 的 per-sample 循环可以向量化，但 B=64 的循环开销很小，不影响训练速度。

---

## 梯度流分析

**关键问题**: kp_feats 来自 `feat_map_detached = featmaps[-1].detach()`，经过 GCN 层处理后成为 `kp_feats_enhanced`。

- `kp_feats_enhanced` **有 grad_fn**（GCN 层的 Linear/Conv 参数在计算图中）
- PACI consistency loss 对 `kp_feats_enhanced` 求梯度 → **梯度流向 GCN 参数**（但不到 backbone）
- 这与 PKC 的"no-op"情况**不同**：PKC 的问题是 SupCon 在同 batch 内自引用；PACI 使用跨 batch 积累的 momentum prototypes 作为 target，提供了 GCN 之前未见的监督信号
- Bank prototypes 正确使用 `.detach()` 查询（line 1074），不引入循环梯度

**结论**: PACI consistency loss 是有效的——它为 GCN 提供额外的基于记忆的监督信号。

---

## OA-SD 交互分析

- OA-SD 模式下，`kp_data` 来自 student（occluded image）的前向传播
- PACI bank update 使用 student features → bank 中会积累 occluded view 的特征
- 这是**可以接受的**：visibility filter (`kp_w > 0.3`) 会过滤掉 occluded keypoints
- 不与 OA-SD 的 EMA teacher 冲突（PACI bank 是独立的记忆机制）

---

## 设计合理性

1. **不是小调参**: 引入了新的模块和新的 loss，有明确的创新点（per-ID per-part memory）
2. **单变量**: 在 OA-SD baseline 上加 PACI，变量隔离
3. **假设清晰**: "per-identity per-part prototypes 能让 GCN 学习 identity-specific part appearance"
4. **消融设计**: 有清晰的 Phase 1/2/3 分解

---

## 修复后状态

Critical C1 必须修复后才能训练。M1 (bank stats 日志) 强烈建议修复以便监控。M2/M3 建议修复但不阻塞。

---

**修复 C1 后：审查通过**
