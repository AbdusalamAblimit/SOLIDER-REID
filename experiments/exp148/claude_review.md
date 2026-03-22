# exp148 PCVT 广范围审查报告

**审查日期**: 2026-03-22
**审查范围**: 想法合理性 + 单变量隔离 + 代码正确性 + train/test 对称性 + 默认行为 + 行为日志 + AMP/shape/collate + 创新风险

---

## 阻塞问题

**无阻塞问题。** 代码实现与设计一致，无 runtime crash 级 bug。

---

## 高风险问题

### H1. 主损失被 1/3 稀释 — 可能是最大单一风险

**位置**: `processor/processor.py:918-927`

```python
for vi in range(1, len(all_scores)):
    v_loss = loss_fn(all_scores[vi], all_feats[vi], target, ...)
    loss = loss + v_loss
loss = loss / len(all_scores)  # ← 三视图平均
loss = loss + pcvt_weight * pcvt_loss  # ← 再加 PCVT consistency
```

效果：full view 的 ID+Triplet 损失从权重 1.0 降到 **1/3**。partial views 各自分到 1/3。
- baseline：`L = L_main`
- PCVT：`L = (L_full + L_a + L_b) / 3 + 0.25 * L_pcvt`

这意味着 backbone 在主任务上的有效学习率被缩小为 1/3。如果最终结果不好，**无法区分是"PCVT 思路无效"还是"主损失被稀释太多导致的"**。

**建议**：这不是 bug，是设计选择。但必须在 monitor.md 中记录这个已知混淆因子，并在实验结束后如果结果为负，优先考虑"dilution 才是原因"的解释。可选的修复：把 full view 权重调高（如 full:a:b = 2:1:1 再除以 4），但这需要另一个实验。

**严重程度**: High — 不阻塞启动，但是最大的解释性风险。

### H2. 3x 前向传播 — OOM 风险

**位置**: `processor/processor.py:741-757`

三视图在同一个 `amp.autocast` block 内顺序执行，所有三组激活的计算图都需要保留到 backward。

- 标准训练：1x forward，约 15-18GB
- PCVT：3x forward 的梯度图同时驻留，预估 **20-22GB**

RTX 3090 有 24GB，应该刚好够。但如果 batch 中有大 heatmap 或 GCN 分支较重，可能 OOM。

**建议**：第一个 epoch 必须密切监控 GPU 显存。如果 OOM，考虑在 partial view forward 时 `with torch.no_grad()` + 只从 `pcvt_loss` 回传，但这会牺牲 partial view 的 ID/Triplet 训练信号。

**严重程度**: High — 不阻塞，但首 epoch 如果 OOM 需要立即止损。

### H3. PSG 在 masked views 上仍使用完整热图 — 设计层面的不确定性

**位置**: `processor/processor.py:744-746`

```python
for v_img in img_views:
    m_out = model(v_img, ..., pose_dict=pose_dict)  # ← pose_dict 是完整热图
```

三个视图共享相同的 `pose_dict`（包含完整的 scene heatmap）。PSG gate 会告诉 backbone"这里有 torso"，但 view_a 的该区域像素已经被 fill=0.0 覆盖。

两种可能：
- **有益**：backbone 学会了"被告知这里有人但看不到像素 → 这是遮挡 → 调整表征"
- **有害**：PSG 给了矛盾信号 → backbone 困惑

这不是 bug，但它意味着 PCVT 的实际训练动力学和 design.md 描述的"互补 partial support"有微妙差异。

**建议**：在 monitor.md 中标记这个设计选择。如果 `pcvt_cos_fa/fb` 比预期低得多，这可能是原因。

**严重程度**: High — 可能是决定实验成败的隐藏因子。

---

## 中低风险问题

### M1. POSE_TEST_FEAT 从 `concat_scaled` 改成 `equal_concat` — 不影响训练，但需注意报告一致性

**位置**: `configs/occluded_duke/pose_psg_gcn_pcvt.yml:23` vs `pose_psg_gcn.yml:25`

这只影响评估时的特征拼接方式。设计文档明确说对照的是 `exp030a-eq`，所以是正确的。但在 results.md 中必须明确标注对照组是 `exp030a equal_concat`，不是 `exp030a concat_scaled`。

**严重程度**: Low — 只是报告层面。

### M2. Fallback 路径的 PCVT loss 仍然计算 — 信号噪声

**位置**: `datasets/pose_dataset.py:518-541` + `processor/processor.py:928-933`

当 `visible_parts < pcvt_min_parts` 时，两张视图退化为独立随机 erase。但 `pcvt_cov_a/b/u` 被设为 0.0，`_compute_pcvt_loss` 仍然计算 cosine consistency。

在这种情况下，PCVT loss 变成"两个随机 erase 视图的平均应接近 full view" — 这是普通 dual-view consistency，不是互补 support 训练。

probe 显示 `pcvt_fb = 0.000`，说明当前 fallback 率为 0%。所以实际上不影响训练。但如果数据分布变化导致 fallback 率上升，日志不会区分"有效互补训练"和"退化为随机 consistency"。

**建议**：在 fallback 时可以考虑跳过 PCVT loss（直接 `pcvt_lc = 0`），或至少用 `pcvt_fb` 比率来调节 `pcvt_lc` 的有效权重。当前 fallback=0% 时不需要改动。

**严重程度**: Low（当前 fallback=0%）。

### M3. fill_value=0.0 在归一化空间对应 mid-gray — 不是"黑色遮挡"

`pixel_mean = [0.5, 0.5, 0.5]`, `pixel_std = [0.5, 0.5, 0.5]`
→ `tensor = (pixel - 0.5) / 0.5`
→ fill=0.0 对应 pixel=0.5 (mid-gray)

这是合理的中性 fill。但需要意识到：
- 如果 Swin 对全黑/全灰区域有不同的内部处理模式，fill=0.0 可能不是最优选择
- Random Erasing 默认也是 pixel-mode erase（不是常数 fill），所以 PCVT 的 fill 策略和 RE 不完全一致

**严重程度**: Low — 合理的默认值。

### M4. PCVT_BODY_PARTS 中关键点 11/12 同时出现在 torso 和 legs

```python
'torso': [5, 6, 11, 12],
'left_leg': [11, 13, 15],
'right_leg': [12, 14, 16],
```

keypoint 11 (left hip) 同时属于 torso 和 left_leg。这导致 hip 区域的热图响应同时出现在两个 body group 中。`argmax` 分配解决了像素级排他性，但 body-group 的语义定义是模糊的。

**严重程度**: Low — `argmax` 正确处理了排他性。

### M5. 训练速度降低约 3x

每个 iteration 需要 3 次完整前向 + 1 次 PCVT loss。120 epoch 的实际训练时间从约 4-5 小时变为 12-15 小时（粗估）。

**建议**：monitor.md 中记录实际每 epoch 耗时，确认 3090 能在合理时间内完成。

**严重程度**: Medium — 不影响正确性，但影响迭代效率。

---

## 想法合理性评估

### A1. PCVT 是否真的在回答 "single-image support incomplete"？

**部分是。** exp109 证明的是：如果有同 ID 跨图完整 support，matching 大幅提升。PCVT 的 claim 是：通过在单图内构造"伪多 support"训练，让 backbone 对 partial support 更鲁棒。

但"让 backbone 更鲁棒"和"真正补全 support"是两个不同的目标。PCVT 走的是前者（robustness training），不是后者（feature completion）。design.md 对此是诚实的。

**结论**：PCVT 回答的更准确地说是"能否用训练范式让编码器对 partial support 稳定"，而不是"能否补全 support"。这个问题的答案本身有价值，无论正负。

### A2. PCVT 和 PAMC (exp050) 的根本区别

| 维度 | PAMC | PCVT |
|------|------|------|
| 遮挡视图数 | 1 (masked) + 1 (full) | 2 (complementary A/B) + 1 (full) |
| 遮挡方式 | 随机 body group masking | 互补 body group 划分 |
| Consistency 目标 | projector(full) ≈ projector(masked)（BYOL-style） | `avg(f_a, f_b)` ≈ `f_full`（union-consistency） |
| 关键约束 | 单个 masked view 需匹配 full | 两个 partial views 的 union 需匹配 full |
| 梯度流 | 双向 + stop-grad target | full.detach()（只训练 partial views） |

**区别是真实的**，不只是换名。PCVT 的"互补性"和"union = full"约束是 PAMC 没有的。PAMC 是 self-supervised 预训练风格（BYOL），PCVT 是 multi-view supervised training + union consistency。

但差异的**实质大小**取决于实验结果。如果 PCVT 也是 neutral，那这个差异在实践中不重要。

### A3. PCVT 和 PADE / FCFormer / SSSC 的区别

- **vs PADE**：PADE 用 pose 调节增强强度（哪里擦多、哪里擦少），但仍然是单视图。PCVT 是多视图 + 互补约束。机制不同。
- **vs FCFormer**：FCFormer 是双流架构 + feature-level decoder completion。PCVT 是图像级 masking + representation-level union consistency。层次不同。
- **vs SSSC**：SSSC 是 self-supervised contrastive + severity-aware。PCVT 是 pose-defined complementary + supervised + union loss。策略不同。

**结论**：PCVT 与这些工作在机制上确实不同。但"不同"不等于"足够新颖"。作为论文的唯一主贡献可能偏薄；作为论文主方法的一个组件或 supporting technique 更合理。

### A4. 成为论文主贡献的潜力

**单独作为主贡献：偏弱。** "Pose-defined complementary masking for robust training" 可以写成一个 section，但要支撑一篇 B 类论文的主叙事，仅此一个组件不太够。

**作为主方法的 training recipe 组件：合理。** 如果结果为正，它可以是 "Method Section 3.3: Complementary View Training" + 消融证明其必要性。

**如果结果为 neutral/负：也有价值。** 排除了"单图伪多 support 训练"这条线，为论文的 motivation section 提供反向证据。

---

## 行为日志评估

### 当前日志是否足够支撑及时止损？

**是的，当前日志设计很好。** 具体来说：

| 日志字段 | 用途 | 止损判据 |
|----------|------|----------|
| `pcvt_fb` | fallback 比例 | >0.5 说明数据不适合互补划分 |
| `pcvt_cov_u` | union coverage | 应持续≈1.0；如果下降说明 partition 退化 |
| `pcvt_ovr` | overlap | 应持续≈0.0；>0 说明互补性被破坏 |
| `pcvt_gap` | union优于单视图的 gap | 核心指标；若持续≤0 说明 union 无增益 |
| `pcvt_cos_fu` | union-full cosine | 应随训练上升 |
| `pcvt_cos_fa/fb` | partial-full cosine | 应低于 cos_fu |
| `pcvt_lc` | PCVT loss 值 | 应下降 |
| `pcvt_mga/mgb` | mask 面积比 | 应接近 0.5 |

### 缺少的日志

1. **每 epoch 实际时间**：3x forward 应导致每 epoch 变慢，需要记录 wall-clock 来确认不会让 120 epoch 超过可接受范围。
2. **partial views 的 ID accuracy**：当前只记录 view 0 的 acc（`all_scores[0]`），没有 view_a/view_b 的独立 acc。如果 partial views 的 acc 极低，说明 masked 太重。

**建议**：在 monitor.md 的前几次检查中手动关注这两点。如果需要正式追踪，可以在 processor 中添加 `pcvt_acc_a/b` 日志，但不阻塞启动。

---

## AMP / Shape / Collate / Parallel-View 路径检查

### AMP
- 三次前向在 `amp.autocast(enabled=True)` 内：**OK**
- `_compute_pcvt_loss` 使用 `F.cosine_similarity` + 标量运算：**AMP safe**
- `full.detach()` 防止混精度梯度问题：**OK**

### Shape
- `base_tensor`: (3, 384, 128) → `_make_pcvt_views` 的 mask 在 (384, 128) 上操作 → `view_a[:, drop_a] = fill` → shape 一致
- `hm = persons[0]['heatmap']`: (17, 384, 128) at this point (pre-downsample) → matches image resolution
- PCVT meta scalars: 0-dim tensors → `_collate_pose_dicts` 用 `torch.stack` → (B,) → `pose_dict['pcvt_cov_a'].float().mean()` → scalar：**OK**

### Collate
- `pose_train_collate_fn` 检测 `n_views > 1` → 返回 `list of (B,C,H,W)`：**OK**
- `pose_val_collate_fn` 始终单视图（val dataset 没有 pcvt=True）：**OK**
- PCVT 的 pcvt_meta keys（pcvt_cov_a 等）出现在所有训练样本中（self.pcvt 是 dataset 级别 flag）：**OK**
- Val 样本不含 pcvt_meta keys → val collate 不会崩：**OK**

### Parallel-View 路径
- PISD 已检查 `not parallel_aug`（`processor.py:1196`）：无冲突
- PAMC 未在 PCVT config 中启用：无冲突
- PACD 不与 parallel_aug 冲突（它用 `feat_maps` from last view）：**OK**
- `kp_data` 取自 `all_kpdata[0]`（第一个视图）：**OK**，因为 GCN 对三个视图都用相同 pose_dict

---

## "实验看起来大，其实只是旧 recipe" 风险评估

### 风险：PCVT ≈ "更精细的 Random Erasing + dual-view consistency" 吗？

**部分是。** 如果剥离 PCVT 的所有叙事，它的核心操作是：
1. 用 pose heatmap 决定 erase 位置（vs Random Erasing 随机位置）
2. 创建两个互补 erase 版本（vs 单个 erase）
3. 加一个 union-consistency loss（vs 无 consistency）

其中 (1) 已被 PADE 做过（pose-aware erase），(2) 是 dual-view 标准操作（BYOL/MoCo 变体），(3) 是 consistency regularization 标准操作。

**但 PCVT 的组合方式（互补 body-group 划分 + union ≈ full 约束）确实是新的。** 没有现有工作把这三步按这种方式组合在一起。

### 结论

PCVT 不是"旧 recipe 换名"，但也不是全新范式。它是对现有 masking-consistency 范式的一个 **有针对性的改进**（从随机遮挡到互补遮挡，从单视图 consistency 到 union consistency）。

作为实验来说值得做，因为：
- 如果有正信号：至少可以作为训练 recipe 组件
- 如果为 neutral/负：排除了"单图伪多 support"整条线

---

## 总体结论

### 可启动，但有以下注意事项：

1. **必须在 monitor.md 记录的风险**：
   - H1: 主损失 1/3 稀释是已知混淆因子。如果最终结果为负，不能直接归因于"PCVT 思路无效"，必须先考虑 dilution 的影响。
   - H2: 首个 epoch 密切监控 GPU 显存，如果 OOM 立即止损。
   - H3: PSG 在 masked views 上使用完整热图是设计选择，需要监控 `pcvt_cos_fa/fb` 是否异常低。

2. **止损标准**：
   - `pcvt_gap` 在 epoch 10 后持续 ≤ 0 → PCVT 的 union 假设不成立
   - `pcvt_lc` 在 epoch 20 后不下降 → consistency 学不动
   - OOM → 立即终止

3. **如果结果为 neutral**：
   - 先做消融：去掉 PCVT loss 但保留 3-view 训练 → 看是否是 augmentation 本身的贡献
   - 先做消融：调整 view 权重比（如 full:a:b = 2:1:1） → 看是否是 dilution 的问题
   - 如果消融仍然 neutral → 排除"单图伪多 support"整条线

4. **创新评价**：
   - 作为论文 **唯一主贡献**：偏弱
   - 作为论文 **方法组件之一 + 消融证据**：合理
   - 作为 **方向排除实验**：有价值（无论结果正负）

### 审查结论：**可启动。**
