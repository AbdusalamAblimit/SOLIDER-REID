Now I have a complete picture. Let me compile the full review.

---

# exp142 SKC 全面审查报告

## 一、研究设计合理性

### 1.1 与 exp109 oracle 发现的关系

**结论：SKC 确实在直接回答 support incomplete 问题，方向正确。**

exp109 证明了 oracle 上界为 +8.5% mAP（feature+weight replacement），说明 headroom 巨大。SKC 试图通过训练一个 self-contained completion module 来捕获这个 headroom，而不是在 test-time 直接抄 bank。这是一个合理的技术路径。

### 1.2 与 SCRC / SCKD / LPCS 的关系

**结论：SKC 是同一家族（feature-level completion with bank guidance）的显著升级，不是伪创新，但也不是范式转换。**

具体区别：

| 维度 | SCKD | SCFR | SCRC | **SKC** |
|------|------|------|------|---------|
| 特征修改方式 | 无（仅梯度） | 硬替换 | 标量门控 × (proto−feat) | self-attn + skeleton graph → delta + gate |
| test-time bank 依赖 | 否 | 否 | **是**（直接用 proto） | **否** |
| 跨关键点推理 | 无 | 无 | 无 | **有**（attention + graph） |
| 新增参数量 | 0 | 0 | ~200K | **~1.1M** |
| 典型退化模式 | 梯度太弱无效果 | 分布不匹配 | gate→0.999 塌缩 | 待观察 |

SKC 相对前者的两个真正新点：
1. **Train/test 完全分离**：bank 只是训练 supervisor，test 时 module 纯靠自图证据
2. **跨关键点结构推理**：而非逐点独立操作

### 1.3 能否成为论文主方法候选

**条件性回答：可以，但需要 global 模式也涨。**

如果只有 `equal_concat` 涨而 `global` 不涨，那说明补全改善了 fusion 但没有改善单图表征，论文 story 会弱。exp109 的 oracle 是作用在单图表征上的，SKC 的论文价值取决于能否在 global 上也展现正收益。

### 1.4 失败时的诊断价值

**清晰。** 设计文档已经列出了 5 种失败模式及其解释，每种都有对应的日志指标。负结果不会被浪费。

---

## 二、代码正确性

### 2.1 Completion 路径隔离 — PASS

`self.skc` 默认为 False，只在 config 开启时生效。已通过 `if self.skc:` 保护所有 SKC 相关路径（init L327、forward L648、aux_data L772）。默认 baseline 完全不受影响。

### 2.2 Low/high confidence mask — PASS (with 1 observation)

- SSKC module: `low_mask = kp_scores <= self.low_thr` (L196)，`gate = sigmoid(...) * low_mask.float()` (L222)
- Bank update: `vis_mask = kp_weights[b_idx] >= self.update_thr` (L77)
- Low threshold: 0.3, update threshold: 0.7, **无重叠区间 [0.3, 0.7]**

这个设计非常干净：高置信关键点 (≥0.7) 更新 bank，低置信关键点 (≤0.3) 被补全，中间带 (0.3, 0.7) 两者都不碰。**无信息泄漏。**

关键发现：bank 更新使用 `skc_completed_feats`（L1247），但由于 gate = 0 对所有 score > 0.3 的关键点，**高置信关键点的 completed 特征 === raw 特征**。所以 bank 实际上用的是未修改的特征来更新，不存在循环污染。

### 2.3 Completion 写回对 downstream 的影响 — PASS

追踪数据流：
```
kp_feats (pre-SSKC)
    → SSKC → kp_feats (completed)
        → GCN → kp_feats_enhanced
            → weighted pooling → skeleton_feat
                → BN → classifier → ID loss / triplet loss
```

Completed features 确实进入了 downstream 的 pooling、fusion 和 loss。不是只改了日志。

### 2.4 AMP 安全 — PASS

- SSKC module 内部全是 `nn.Linear`、`nn.LayerNorm`（AMP 自动保持 fp32）、`nn.MultiheadAttention`、`nn.Sigmoid` — 均是 AMP 安全操作
- `adj_norm` buffer (float32) 在 `torch.matmul` 中会被 autocast 到 float16，这是正常行为
- `vis_scale.clamp(min=1e-6)` — float16 最小表示约 6e-8，1e-6 安全
- gate × delta 和 kp_feats + gate × delta 均为标准 elementwise 操作，float16 安全

### 2.5 `skc_raw_feats` 引用安全 — PASS

L649: `skc_raw_feats = kp_feats`（引用，非 clone）。L651: `kp_feats, skc_stats = self.skc_block(kp_feats, kp_scores)` 返回新 tensor。此后 `skc_raw_feats` 仍指向原始 pre-completion tensor。后续代码（SCRC/GCN）不会 in-place 修改原始 tensor。安全。

### 2.6 梯度流 — PASS (with 1 minor observation)

两条梯度路径到达 SSKC module：
1. **Main loss** → classifier → BN → pooling → GCN → completed kp_feats → SSKC
2. **Completion loss** → comp_norm → skc_completed_feats → SSKC

两条路径正确叠加。Proto 经过 `.detach()`，不会反向传播到 bank。

**Minor observation**: L1088 `raw_norm = F.normalize(skc_raw, dim=2)` 建立了不必要的计算图。`raw_norm` 只用于 `pre_dist` 的 logging（`.item()` 路径），不参与 loss 计算。虽然 backward 不会实际流梯度到这里（因为 loss 不依赖 pre_dist），但计算图仍被构建，浪费少量内存。建议包裹 `with torch.no_grad()`。

### 2.7 日志统计与真实计算一致性 — **有一个问题**

**`applied_ratio` 定义问题（Medium）**：

```python
# skeleton_gcn.py L228
applied_mask = (gate > 0.05) & low_mask
stats = {
    'applied_ratio': float(applied_mask.float().mean().item()),  # ← 除以 B*17
}
```

分母是 `B × 17`（所有关键点），不是 `low_count`。如果一个 batch 中只有 30% 的关键点是低置信的，且全部被有效补全，`applied_ratio` 也只会显示 ~0.3，看起来像"大部分跳过了"。

建议同时记录 `applied_in_low = applied_count / max(low_count, 1)` 以避免日志误判。

---

## 三、训练/测试一致性

### 3.1 Train/test 使用同一套 completion 逻辑 — PASS

SSKC module 的 `forward()` 没有 `self.training` 分支。Train 和 test 执行完全相同的计算：`kp_feats → token_proj → attention → skeleton → FFN → delta + gate → completed`。`LayerNorm` 在 train/eval 模式行为相同。`MultiheadAttention` 未传 dropout 参数（默认 0.0），train/eval 相同。

### 3.2 `_skc_active` 在 test 时的行为 — PASS

- `__init__` 设 `self._skc_active = True`
- Processor 每个 epoch 开始时设 `_skc_active = (epoch > skc_warmup)`
- Eval（训练中途）继承当前 epoch 的值
- `test.py` 加载 checkpoint 时，`__init__` 重新设为 True（`_skc_active` 不是 buffer/param，不在 checkpoint 里）

**结果**：warmup 期间 eval 不运行 SSKC（harmless identity）；warmup 后 eval 运行 SSKC（正确）；standalone test.py 总是运行 SSKC（正确）。

### 3.3 Support bank 泄漏检查 — PASS

Bank 在 forward pass 中**完全不参与**。Bank 只出现在 processor.py 的：
1. Loss 计算（L1081: `skc_bank.get_support()` → 用于算 loss）
2. Bank 更新（L1251: `skc_bank.update()`）

两者都在 forward pass **之后**执行。SSKC module 本身不接收任何 bank 输入。**无隐蔽泄漏。**

---

## 四、止损能力

### 4.1 当前日志设计评估

| 问题 | 对应指标 | 能否回答？ |
|------|---------|-----------|
| 模块是否真正作用？ | `skc_arr`, `skc_dn` | **部分** — `skc_arr` 分母是全部关键点，可能误导 |
| 是否整体跳过？ | `skc_gm → 0`, `skc_arr → 0` | **是** |
| 是否整体强覆盖？ | `skc_gm → 1.0`, `skc_gs → 0` | **是** |
| low-confidence tokens 是否被拉近 support target？ | `skc_pre > skc_post` | **是** — 这是最关键的指标 |
| 补全退化成常数？ | `skc_gs → 0`, `skc_dn` 低方差 | **需要 delta 方差**，当前只有均值范数 |

### 4.2 建议补充的日志

1. **`skc_applied_in_low`** = `applied_count / max(low_count, 1)` — 在低置信关键点中的有效补全率，避免分母稀释
2. **`skc_delta_std`** = delta 范数的 std — 检测 delta 是否塌缩成常数方向
3. 可选：warmup 期间也记录 `skc_spr`、`skc_pc`、`skc_pcnt` — 验证 bank 建设进度

### 4.3 止损规则评估 — 合理

设计文档中的 4 条止损规则（L190-197）均有对应的日志指标支撑，可以执行。

---

## 五、关键风险评估

### 风险 1: 结构补全信号不足（最核心风险）

SKC 的核心假设是"自图高置信关节点 + skeleton topology 能提供足够的补全信号"。但在重遮挡图中：

- 可能只有 3-5 个高置信关键点（头、肩）
- 这些关键点在 skeleton 图上与缺失的下半身关键点距离 2-3 跳
- Visibility-weighted 传播 `A_norm @ (tokens * vis_scale) / (A_norm @ vis_scale)` 会在多跳后极度衰减

**预测**：SKC 可能只对轻度遮挡有效（少量关键点缺失、邻居可见），对重度遮挡基本退化成 identity。而 exp109 的 oracle 收益恰恰集中在重度遮挡上。这是一个根本性的张力。

**观测方法**：分可见关键点数量分组看 `skc_pre - skc_post` 的改善。如果只有 vis≥12 的样本有改善而 vis≤6 的没有，则验证此风险。

### 风险 2: 退化为"花哨的 SCRC"

SCRC 的核心机制是 `gate * (proto - feat)`。SKC 的核心机制是 `gate * delta`，其中 delta 来自 self-attention + skeleton graph + FFN。如果 attention 和 graph 没有学到有意义的跨关键点交互，SKC 本质上就是一个更大的 MLP 在做 `feat → delta`，与 SCRC 的差异只在参数量。

**观测方法**：如果 SKC 最终 global 涨幅与 SCRC 的 -0.6% 相近（即 ±0.5%），则说明结构推理没有额外贡献。

### 风险 3: Completion loss 与 main loss 竞争

Main loss（ID + triplet）希望 kp_feats 为 identity discrimination 服务；completion loss 希望 kp_feats 接近 bank prototype。当两者方向一致时互利，但如果 bank prototype 质量不高或方向不一致，会产生梯度冲突。

**缓解**：weight 0.5 + 20 epoch warmup 是合理的保护。

---

## 六、结论

### 这条线是否比 LPCS 家族更大？

**是，毫无疑问。** LPCS 是 test-time pair correction，SKC 是 training-time representation improvement。两者不在同一个层面上。SKC 直接回应 exp109 的核心发现，LPCS 只是绕路。

### 这条线是否与旧方法本质重复？

**不完全重复，但有进化关系。** SKC 是 SCRC 的显著升级（train/test 分离 + 结构推理），不是换名字重做。但它仍属于同一个方法家族（feature-level bank-supervised completion）。如果这个家族整体被证伪，SKC 的结果也不例外。

### 代码是否容易退化成旧机制？

**有风险但可监控。** zero-init + negative-bias gate 保证了保守启动。completion loss 直接激励非零 delta。SCRC 的 gate→1.0 塌缩问题在 SKC 中不太可能重现，因为 SKC 的 gate 被 `low_mask.float()` 硬限制（高置信关键点 gate 恒为 0），没有全局塌缩的路径。但 delta 可能塌缩为常数方向——当前日志中缺少 delta std 来检测这一点。

### 需要在训练前修复的问题

1. **[Medium]** `applied_ratio` 分母应改用 `low_count`，或补充 `applied_in_low` 指标
2. **[Low]** processor L1088 `raw_norm` 计算应包裹 `torch.no_grad()`
3. **[Low]** 补充 `skc_delta_std` 日志防止 delta 方向塌缩

### 最终判断

**代码实现正确，设计合理，可以启动训练。** 核心风险不在代码层面，而在方法层面（结构补全信号是否足够强）。这只能通过实验验证。
