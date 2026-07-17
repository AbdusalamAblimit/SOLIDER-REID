# exp197 Structural Token Mixup (STM) — Claude Review v2

## 审查范围（与 v1 完全相同的全范围审查）
- design.md — 动机、创新门槛、技术方案
- config/defaults.py — 新配置项（行 182-186）
- processor/processor.py — STM 实现（行 535-597）
- loss/make_loss.py — loss_fn 对 STM 输入的处理（行 121-248）
- loss/triplet_loss.py — hard_example_mining 等正样本数假设（行 64-84）
- loss/supcon_loss.py — SupCon 对 STM batch 的兼容性

---

## a. 设计评审

创新门槛、单变量原则、非小调参：与 v1 结论一致，均通过。不再重复。

---

## b. v1 Critical #1 修复验证：等正样本数

**状态: 已修复。**

v1 问题：`stm_prob=0.5` 对每个样本独立做概率选择，导致各 ID 产生的混合样本数量不等，triplet loss 的 `dist_mat[is_pos].view(N, -1)` 会因 reshape 维度不一致而崩溃。

v2 修复方案：改为 batch 级别概率判定 + 固定数量生成。

1. 行 547：`random.random() < stm_prob` 是 batch 级别的单次掷骰，决定整个 batch 是否做 STM。通过。
2. 行 554-581：双层循环 `for id_idx in range(num_ids)` + `for i in range(num_instance)`，每个 ID 恰好产生 `num_instance` 个混合样本。
3. 最终 `stm_target` 长度 = `num_ids * num_instance = B`（与原 batch 完全相同大小）。
4. 标签排列：外层按 ID 遍历，内层按 instance 遍历，所以 `stm_target` 的排列是 `[id0, id0, id0, id0, id1, id1, id1, id1, ...]`，与原始 sampler 的排列一致。triplet loss 的 `is_pos` 掩码每行恰好有 `num_instance - 1` 个 True（自己除外），`view(N, -1)` 中 `-1` = `num_instance - 1`，与原 batch 完全一致。

**结论：Critical #1 修复正确，triplet loss 不会崩溃。**

---

## c. v1 Medium #2 修复验证：num_instance 硬编码

**状态: 已修复。**

行 544：`num_instance = cfg.DATALOADER.NUM_INSTANCE`。从配置读取，不再硬编码。通过。

---

## d. v1 Medium #3 修复验证：kp_data 不传的注释

**状态: 已修复。**

行 589：`# Compute loss on mixed batch (no kp_data — mixed tokens lack kp correspondence)`。注释已添加，清楚说明不传 kp_data 的设计意图。通过。

---

## e. 代码正确性（全范围重新审查）

### e1. Gradient flow through slicing

行 573-574：`score[k][idx_j:idx_j+1]` 和 `feat[k][idx_j:idx_j+1]` 都是 PyTorch slice 操作，保留计算图。行 584-585 的 `torch.cat` 同样保留梯度。`loss_fn` 返回的 `stm_loss` 通过 `loss = loss + stm_weight * stm_loss`（行 592）加入总 loss。反向传播会经由 `stm_loss` → `loss_fn` 内的 CE/triplet → `stm_score`/`stm_feat` → 原始 `score`/`feat` → 模型参数。梯度流正确。

### e2. stm_cam 语义

行 587：`stm_cam = target_cam[:len(stm_target)]`。由于混合 batch 大小 = B（与原 batch 相同），`stm_cam` 就是完整的 `target_cam`。`target_cam` 在 `loss_func` 中不被使用（只在签名中出现），不影响计算。Low，可接受。

### e3. _loss_details 保存

行 591-595：`getattr(loss, '_loss_details', {})` 从已有的 loss 上获取 details dict，添加 `stm` 和 `stm_n`，再赋回给新的 `loss`（`loss + stm_weight * stm_loss`）。注意：`loss = loss + stm_weight * stm_loss` 创建了一个新的 tensor，旧 tensor 上的 `_loss_details` 不会自动继承，所以行 595 `loss._loss_details = details` 必须执行。逻辑正确。

### e4. loss_fn 内对 STM batch 的行为

当 `kp_data=None` 传入 `loss_func` 时：
- Evidential: `evid_enabled and kp_data is not None` → False，跳过。回退到 SupCon 或普通 CE。
- SupCon: `POSE_STR_SUPCON` 从 cfg 读取，不依赖 kp_data。如果 SupCon 启用，STM batch 也会计算 SupCon loss。SupCon 不要求等正样本数（用 `num_pos.clamp(min=1)` 保护），安全。
- MaxSim triplet: `maxsim_tri_enabled and kp_data is not None` → False，回退到普通 per-token triplet。安全。
- Visibility-weighted SupCon: `vis_weighted and kp_data is not None` → False，回退到均等权重。安全。

### e5. 边界情况

- `stm_num_swap > num_parts`: 行 566 `min(stm_num_swap, num_parts)` 保护。正确。
- `num_ids = 0`（空 batch 或 B < num_instance）：行 547 `num_ids > 0` 保护。正确。
- 最后一个 batch 不足 B：`num_ids = B // num_instance` 截断尾部，不会越界。正确。
- `num_instance = 1`：partners 列表为空，`random.choice([])` 会抛 IndexError。但 `NUM_INSTANCE=1` 本身不可能用于 triplet 训练（原始 loss 已经会崩溃），所以这不是 STM 引入的新问题。可接受。

### e6. 与 parallel_aug (3-view) 的交互

STM 在行 535 执行，仅操作 view 0 的 `score`/`feat`。parallel_aug 在后续行添加 view 1/2 的 loss。STM 不影响 view 1/2。这意味着 STM 只增强 view 0 的训练信号，不增强 augmented views。这是合理的设计——augmented views 已经是多样化的输入，再做 token mixup 可能过度正则化。

### e7. 与 OA-SD 的交互

OA-SD 在 parallel_aug 之后，操作 teacher forward 的输出，与 STM 完全独立。无冲突。

---

## f. Config 安全

- `POSE_STM = False` 默认关闭。已有实验不受影响。
- `POSE_STM_NUM_SWAP = 2`、`POSE_STM_PROB = 0.5`、`POSE_STM_WEIGHT = 0.5` 默认值合理。
- 没有 yml 文件设置这些参数（通过命令行 args 传入），不影响其他配置文件。

通过。

---

## g. AMP 安全

STM 执行在 `with amp.autocast(enabled=True):` 块内。`torch.cat`、slice、`loss_fn` 调用都在 autocast 范围内。无手动类型转换，无精度风险。通过。

---

## h. 内存影响

v2 固定数量生成使得混合 batch 大小恒定为 B（与原 batch 相同），不再是 v1 的 ~B/2（期望值）。额外峰值内存 = 原 batch feature tensor 的 100%（而非 50%）。feature tensor 尺寸为 7 * B * D（~7 * 64 * 768 = 344K floats = ~1.3MB），加上 score tensor 7 * B * num_classes（~7 * 64 * 702 = 315K floats = ~1.2MB）。加上 loss_fn 内部的 dist_mat (B*B) 和中间计算，估计额外 ~50-100MB。在 3090 24GB 上安全。建议首次训练时在 monitor.md 记录 GPU 显存。

---

## 问题汇总

| # | 严重程度 | 问题 | 状态 |
|---|---------|------|------|
| 1 | ~~Critical~~ | triplet loss 等正样本数 | **已修复** — 固定每 ID 产生 num_instance 个混合样本 |
| 2 | ~~Medium~~ | num_instance 硬编码 | **已修复** — 从 cfg.DATALOADER.NUM_INSTANCE 读取 |
| 3 | ~~Medium~~ | kp_data 不传未注释 | **已修复** — 注释已添加 |
| 4 | Low | stm_cam 语义不正确 | 不影响计算，可接受 |

无新问题发现。

---

## 结论

**审查通过。** v1 中发现的全部 3 个需修复问题（1 Critical + 2 Medium）均已正确修复。固定数量生成策略保证了 triplet loss 的等正样本数假设，梯度流正确，与 SupCon/OA-SD/parallel_aug 等现有功能无冲突。可以启动训练。
