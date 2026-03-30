# Claude Review: exp199 — Occlusion-Asymmetric Relational Distillation (OA-RD)

**审查日期**: 2026-03-30
**审查范围**: design.md, config/defaults.py, processor/processor.py, datasets/make_dataloader.py

---

## a. Design Innovation 审查

### OA-RD 是否真正避免了与 SupCon 的梯度冲突？

**结论：理论上是的，但需要仔细分析。**

- SupCon 作用于 per-token features (feat[1:])，推拉 token 在 embedding space 中的绝对位置
- OA-SD (feature distillation) 作用于 individual features (global 或 token)，约束它们向 teacher 的方向收敛 — 这与 SupCon 在同一空间上施加相反方向的力
- OA-RD 作用于 batch-level pairwise similarity matrix of GLOBAL features (feat[0]) — 它只关心样本间的"相对距离排序"，不约束任何单个 feature 的绝对方向

**关键区别**：OA-SD 的梯度形如 ∂L/∂f_i ∝ (f_i - f_teacher_i)，直接推动 f_i 向特定方向移动。OA-RD 的梯度形如 ∂L/∂f_i ∝ Σ_j w_j * (f_j - <f_i,f_j>*f_i)，它通过 pairwise cosine 传播，更关心保持相对结构。而且 OA-RD 仅在 global feature 上操作，SupCon 在 token features 上操作，特征空间完全不重叠。

**结论：梯度冲突风险极低。设计合理。**

### 创新性评估

- Relational Knowledge Distillation (RKD) 是成熟方法，但在 occluded ReID 的 EMA self-distillation 框架下应用无先例
- 与 OA-SD 失败 (exp188/196) 形成对照 → 如果 OA-RD 成功，证据链完整
- 满足创新门槛 2/3：问题层面重定义 (feature→relation)、证据层面可消融

**通过。**

---

## b. Code Correctness 审查

### b.1 Pairwise Similarity 计算 (L761-766)

```python
s_norm = F.normalize(s_global, p=2, dim=1)  # (B, D)
t_norm = F.normalize(t_global.detach(), p=2, dim=1)  # (B, D)
sim_s = s_norm @ s_norm.t() / oa_rd_temp  # (B, B)
sim_t = t_norm @ t_norm.t() / oa_rd_temp  # (B, B)
```

- `F.normalize` + 矩阵乘法 → cosine similarity matrix。**正确。**
- 维度：(B, D) @ (D, B) → (B, B)。**正确。**
- Teacher detach：`t_global.detach()`。**正确，梯度不回流到 teacher。**
- Temperature 除法应用于 student 和 teacher 两侧。**正确（对称温度）。**

### b.2 KL Divergence (L768-771)

```python
log_p_s = F.log_softmax(sim_s, dim=1)
p_t = F.softmax(sim_t, dim=1)
oa_rd_loss = F.kl_div(log_p_s, p_t, reduction='batchmean')
```

- `F.kl_div` 签名：`F.kl_div(input=log_prob, target=prob)`。student 是 log_softmax，teacher 是 softmax。**正确。**
- `reduction='batchmean'`：PyTorch 文档指出此模式除以 batch size，是 KL divergence 的数学正确归约。**正确。**
- Row-normalized softmax (dim=1)：每行独立归一化为概率分布。**正确 — 每个样本独立地 match 其与所有其他样本的关系模式。**

### b.3 Teacher Forward — OA-RD only (L740-755)

```python
if not oa_sd_enabled:
    # Need to run teacher forward
    with torch.no_grad():
        ema_teacher.train()
        teacher_out = ema_teacher(img_teacher, ...)
        ema_teacher.eval()
        ...
```

- 当 OA-SD 禁用但 OA-RD 启用时，需要单独运行 teacher forward。**逻辑正确。**
- `img_teacher` 在此场景下是否已定义？检查：
  - `oa_sd_mode` (2-view): L453-454 定义 `img_teacher`
  - `parallel_oa_sd` (4-view): L448 定义 `img_teacher`
  - OA-RD 代码在 L736 要求 `oa_sd_mode or parallel_oa_sd`。**因此 `img_teacher` 一定已定义。正确。**
- `teacher_feat` 变量：如果 OA-SD 已经运行 (L684-701)，`teacher_feat` 已经赋值。如果 OA-SD 未运行，OA-RD 自己赋值 (L750-755)。**正确，无 NameError 风险。**

### b.4 双模式兼容：OA-SD + OA-RD 同时启用

- OA-SD (L684) 先运行 teacher forward → 设置 `teacher_feat`
- OA-RD (L741) 检查 `if not oa_sd_enabled` → 跳过重复 forward
- OA-RD 直接使用已有的 `teacher_feat`。**正确，teacher forward 只运行一次。**

### b.5 EMA Update (L798-803)

```python
if ema_teacher is not None:
    ...
```

- 条件检查 `ema_teacher is not None`，对 OA-SD 和 OA-RD 共用。**正确。**
- EMA decay 使用 `POSE_OA_SD_EMA_DECAY`（即使只启用 OA-RD）。这是一个设计选择而非 bug — OA-RD 没有自己的 decay 参数，复用 OA-SD 的。**可接受。** 如果未来需要区分，可以添加 `POSE_OA_RD_EMA_DECAY`，但目前不影响正确性。

### b.6 AMP 安全性

- cosine similarity 和 softmax 在 float16 下可能有精度问题，但 PyTorch `autocast` 会自动将 softmax 提升到 float32。`F.kl_div` 也是安全的。**无风险。**

---

## c. Interactions 审查

### c.1 与已有 OA-SD 实验的兼容性

- `POSE_OA_RD` 默认 `False`。**不破坏已有实验。**
- OA-SD 代码路径 (L684-732) 完全独立于 OA-RD (L734-776)，通过 `if oa_sd_enabled` 和 `if oa_rd_enabled` 分开。**正确。**

### c.2 OA-RD + SupCon + 3-view (4-view pipeline)

- `make_dataloader.py` L123: `parallel_aug=True`（3-view）
- `make_dataloader.py` L126: `_oa_sd_mode=True`（OA-RD 触发 → teacher 作为第 4 view）
- `processor.py` L444: `parallel_oa_sd` 正确检测 4-view 模式
- Student 使用 view 0-2，teacher 使用 view 3。**正确。**

### c.3 `oa_sd_mode` / `parallel_oa_sd` 命名

- 变量名用 `oa_sd_mode` 但实际也为 OA-RD 服务。这是命名问题，不是 bug。代码逻辑正确。**低优先级，可后续重命名。**

### c.4 Loss 加权顺序

- 3-view 平均 (L681) → OA-SD loss (L730) → OA-RD loss (L774)
- OA-RD loss 在 3-view 平均之后添加，与主 loss 的量级独立。**正确。**

---

## d. Config Safety 审查

- `POSE_OA_RD = False`：默认禁用。**安全。**
- `POSE_OA_RD_TEMP = 0.1`：合理范围（RKD 论文用类似值）。
- `POSE_OA_RD_WEIGHT = 1.0`：初始权重 1.0，作为探索起点合理。
- 新增 3 行 config，不影响任何已有默认值。**安全。**

---

## e. 日志充分性

- OA-RD loss 通过 `details['oa_rd']` 记录在 detail_meters 中。**可以观察到。**
- 如果同时启用 OA-SD，会分别记录 `oa_sd` 和 `oa_rd`。**可区分。**

---

## f. 潜在问题

### f.1 [Low] Diagonal Self-Similarity

- Pairwise similarity matrix 的对角线 sim[i,i] = 1.0/temp = 10.0 (temp=0.1)。这意味着 softmax 后对角线占主导。
- Teacher 和 student 都有相同的对角线特征，这可能导致 KL loss 主要被对角线支配，off-diagonal 信号被稀释。
- **评估**：实际上对角线在 teacher 和 student 侧都同样为 1.0/temp，所以 KL 中对角线贡献趋近 0（相同的 softmax 输出）。真正的学习信号来自 off-diagonal。这是正确的行为。**无需修改。**

### f.2 [Low] Temperature 选择

- temp=0.1 使得 similarity 范围 [-10, 10]，softmax 分布比较尖锐。如果需要更平滑的 relational matching，可以考虑更高的温度（如 0.5）。但 0.1 作为初始值是合理的探索起点。

### f.3 [Low] 变量命名

- `oa_sd_mode`, `parallel_oa_sd`, `_oa_sd_mode` 这些变量名已经同时服务于 OA-SD 和 OA-RD。建议后续重命名为更通用的名称（如 `teacher_mode`, `parallel_teacher`），但不影响正确性。

---

## 审查结论

| 级别 | 发现 | 状态 |
|------|------|------|
| Low | 对角线 self-similarity 可能稀释信号 | 分析后无问题 |
| Low | temperature=0.1 较尖锐 | 合理探索起点 |
| Low | 变量命名 `oa_sd_*` 服务两种模式 | 不影响正确性 |

所有代码路径正确。Teacher forward 在 OA-RD-only 和 OA-SD+OA-RD 两种模式下都正确处理。KL divergence 计算正确。Config 默认值安全。日志可观察。梯度流分析表明与 SupCon 不冲突。

**审查通过**
