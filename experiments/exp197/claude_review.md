# exp197 Structural Token Mixup (STM) — Claude Review

## 审查范围
- design.md — 动机、创新门槛、技术方案
- config/defaults.py — 新配置项（行 182-186）
- processor/processor.py — STM 实现（行 535-597）
- loss/make_loss.py — loss_fn 对 STM 输入的处理
- loss/triplet_loss.py — 硬样本挖掘假设

---

## a. 设计评审

**创新门槛**: 满足。Token-level cross-instance mixup 在 ReID 中确实没有先例。CutMix/Mixup 在 pixel/feature map 级别操作，而 STM 在 semantic structural token 级别操作——这些 token 已经经过 STD-PR 的 part routing，每个 token 对应一个语义身体部位。在 token 级别交换的合理性在于：同一个人的不同照片的 head token 应该可互换（同 ID），但像素不能直接互换（不同视角/光照/遮挡）。

**单变量原则**: 满足。在 exp187 (3v+SupCon) 或 exp193 (3v+OA-SD+CE) 基础上只加 STM。

**非小调参**: 满足。虽然代码量不算大（~60 行），但机制本身是新的——不是调超参或改权重，而是引入一种新的数据增强范式。

---

## b. 代码正确性

### Critical: Triplet Loss 将在 STM 混合 batch 上崩溃

**严重程度: Critical**

`hard_example_mining()` (triplet_loss.py:79) 执行:
```python
dist_mat[is_pos].contiguous().view(N, -1)
```

这要求**每个 label 拥有完全相同数量的正样本**。但 STM 的混合 batch 违反了这个假设：

- `stm_prob=0.5` 意味着每个 ID 的 4 张图各有 50% 概率被选中
- 一个 ID 可能贡献 1 个混合样本，另一个 ID 可能贡献 3 个
- 当 triplet loss 尝试将 `is_pos` 的布尔掩码结果 reshape 为 `(N, -1)` 时，由于正样本数量不均，**会抛出 RuntimeError**

**修复方案**: 必须保证每个 ID 在混合 batch 中有相同数量的样本。两个选项：
1. 改为"每个 ID 固定生成 K 个混合样本"（不用概率采样），或
2. 后处理：丢弃贡献不足的 ID，或补齐到相同数量

### Medium: `num_instance` 硬编码为 4

**严重程度: Medium**

行 546: `num_instance = 4  # from dataloader config`

默认 `_C.DATALOADER.NUM_INSTANCE = 16`，虽然所有 occluded_duke 配置都将其覆盖为 4，但硬编码仍然脆弱。应该从 `cfg.DATALOADER.NUM_INSTANCE` 读取。

如果 B 不能被 `num_instance` 整除（例如最后一个 batch），`num_ids = B // num_instance` 会丢掉尾部样本，不会崩溃但会静默忽略部分样本。这是可接受的。

### Medium: STM loss_fn 未传 kp_data

**严重程度: Medium**

行 592: `stm_loss = loss_fn(stm_score, stm_feat, stm_target, stm_cam)`

未传 `kp_data` 参数。影响分析：
- **Evidential**: `evid_enabled and kp_data is not None` → 不会触发，回退到普通 CE。可接受。
- **SupCon**: `POSE_STR_SUPCON` 从 cfg 读取，不依赖 kp_data。如果 SupCon 启用，**STM batch 也会计算 SupCon loss**。这在语义上是合理的（混合 token 仍属于同一 ID），但需要注意 SupCon 在小 batch 上的行为。
- **MaxSim triplet**: `maxsim_tri_enabled and kp_data is not None` → 不会触发，回退到普通 triplet。可接受。
- **Visibility-weighted SupCon**: `vis_weighted and kp_data is not None` → 不会触发，回退到均等权重。可接受。

不传 kp_data 的设计决策是合理的——混合 token 没有对应的 kp_feats/kp_weights。但应在代码注释中说明这个选择。

### Low: dummy cam labels

**严重程度: Low**

行 590: `stm_cam = target_cam[:len(stm_target)]  # dummy cam labels`

`target_cam` 在 loss_fn 中完全未使用（签名接收但不引用），所以这不会导致错误。但语义上不正确——混合样本 i 不一定来自 `target_cam[i]` 的 camera。由于不影响计算，标记为 Low。

---

## c. 边界情况

- **`stm_num_swap > num_parts`**: 行 568 用 `min(stm_num_swap, num_parts)` 处理。正确。
- **所有样本被跳过**: 行 585 `if len(mixed_labels) > 0` 保护。正确。
- **`num_ids * num_instance > B`**: 不会发生，因为 `num_ids = B // num_instance` 是整除。

---

## d. Config 安全

- `POSE_STM = False` 默认关闭。不影响已有实验。通过。
- 其他默认值（NUM_SWAP=2, PROB=0.5, WEIGHT=0.5）合理。

---

## e. 与其他功能的交互

- **OA-SD**: STM 在 main loss 计算后、OA-SD 在 parallel_aug 处理后。两者独立。但 STM 对 `score` 和 `feat` 操作，OA-SD 对 teacher forward 输出操作。无冲突。
- **parallel_aug (3-view)**: STM 只操作 `score` 和 `feat`（view 0 的输出）。parallel_aug 在行 672 加 view 1/2 的 loss。**但 STM 在行 535 执行，parallel_aug 在行 672 执行，所以 STM 对 view 0 做 mixup，然后 parallel_aug 又加了 view 1/2 的 loss**。这意味着 STM 只增强 view 0，view 1/2 不受 STM 影响。这是否是有意设计？如果是，应在注释中说明。
- **`_loss_details` 保存**: 行 593-597 正确获取、更新、重新赋值。与后续 STD-PR stats (行 599) 和 LTCS (行 616) 的 details 更新模式一致。

---

## f. 内存影响

STM 创建的额外张量：
- `mixed_scores_all` / `mixed_feats_all`: 最多 `B` 个 list（每个含 7 个 `(1, D)` slice）
- `stm_score` / `stm_feat`: 7 个 `(~B/2, D)` 张量（期望约 50% 被选中）
- 峰值额外内存 ~= 主 batch 的 50%（因为 stm_prob=0.5）

在 3090 24GB 上，当前训练约用 16-18GB，额外 50% 的 feature tensor 估计增加 ~0.5-1GB（feature 远小于 activation maps）。应该安全，但建议在 monitor.md 第一次检查时记录 GPU 显存。

---

## g. AMP 安全

STM 整体在 `with amp.autocast(enabled=True):` 内执行。`loss_fn` 内的计算自动受 AMP 管理。`torch.cat` 和 slice 操作不涉及数值精度问题。安全。

---

## 问题汇总

| # | 严重程度 | 问题 | 修复建议 |
|---|---------|------|---------|
| 1 | **Critical** | Triplet loss 要求每个 ID 正样本数相同，STM 的概率采样违反此假设 | 改为每 ID 固定 K 个混合样本，或在后处理中保证均匀 |
| 2 | Medium | `num_instance` 硬编码为 4 | 从 `cfg.DATALOADER.NUM_INSTANCE` 读取 |
| 3 | Medium | STM loss_fn 不传 kp_data 的设计选择未注释 | 添加注释说明 |
| 4 | Low | dummy cam labels 语义不正确（但不影响计算） | 可忽略，或添加注释 |

---

## 结论

**审查未通过。** Critical #1 必须修复后才能训练。Triplet loss 的等正样本数假设将导致运行时崩溃。修复后需要二次全范围审查。
