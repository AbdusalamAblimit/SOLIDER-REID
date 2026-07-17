# exp198 STM + OA-SD + CE — Claude Review

## 审查范围

本实验为纯配置组合实验：将 exp197 的 STM 功能叠加到 exp191 的 OA-SD + CE 配置上。
无新增/修改代码。审查重点是两个已审查通过的功能（STM、OA-SD）在组合使用时的交互安全性。

审查依据：
- exp197/claude_review_v2.md — STM 完整审查（已通过）
- processor/processor.py — 训练循环中 STM 和 OA-SD 的执行流

---

## a. 执行流分析

在 OA-SD 模式下（`len(img)==2`），训练循环的执行顺序如下：

1. **数据拆分**（行 448-452）：`img[0]` 为 student（post-PLBOA），`img[1]` 为 teacher（clean）
2. **Student forward**（行 482-483）：单视图前向，产出 `score`, `feat`, `kp_data`
3. **主 loss 计算**（行 533）：`loss = loss_fn(score, feat, target, target_cam, kp_data=kp_aux_data)`
4. **STM**（行 535-595）：操作 `score`/`feat`，生成混合 batch，计算额外 `stm_loss`，加入 `loss`
5. **Parallel aug**（行 670-678）：`parallel_aug=False`，跳过
6. **OA-SD**（行 680-728）：EMA teacher forward `img_teacher`，计算蒸馏 loss，加入 `loss`
7. **Backward**（行 745）：`scaler.scale(loss).backward()`
8. **EMA update**（行 750-755）：更新 teacher 参数

---

## b. STM 与 OA-SD 交互安全性

### b1. 数据独立性

- STM 操作的是 student forward 产出的 `score`/`feat`（行 538），通过 slice + cat 创建混合 batch
- OA-SD 操作的也是同一份 `feat`（行 704），但只读取不修改（`F.normalize` 创建新 tensor）
- Teacher forward（行 687）使用 `img_teacher`（clean 图像），完全独立于 student
- 两者均不修改 `score`/`feat` 的原始 tensor，只读取并计算各自的 loss

**结论：数据流无冲突。**

### b2. Loss 累加顺序

- 主 loss（行 533）→ STM loss 累加（行 592）→ OA-SD loss 累加（行 726）
- 每次累加都通过 `loss = loss + weight * component_loss`，创建新 tensor
- `_loss_details` dict 通过 `getattr(loss, '_loss_details', {})` 获取后追加 key，再赋回新 tensor
- STM 写入 keys: `stm`, `stm_n`
- OA-SD 写入 keys: `oa_sd`
- 无 key 冲突

**结论：Loss 累加安全，日志记录无冲突。**

### b3. 梯度流

- STM loss 的梯度：`stm_loss` → `loss_fn` → `stm_score`/`stm_feat`（由原始 `score`/`feat` slice 而来）→ 模型参数
- OA-SD loss 的梯度：`oa_sd_loss` → `F.normalize(feat[i])` → `feat[i]` → 模型参数
- Teacher 侧 `tf.detach()` 阻断梯度（行 707/714/723），只有 student 侧有梯度
- 两条梯度路径最终都汇聚到模型参数，通过 autograd 正常累加

**结论：梯度流正确，无冲突。**

### b4. EMA Teacher 不受 STM 影响

- EMA update（行 750-755）在 `optimizer.step()` 之后执行
- 更新基于 student 模型参数（`base_model.parameters()`），不基于 loss 的具体组成
- STM 通过影响 loss → 影响梯度 → 影响 optimizer step → 间接影响 EMA update
- 这是预期行为：STM 改善 student 特征 → EMA teacher 也逐步改善

**结论：EMA 更新逻辑不受 STM 直接干扰。**

---

## c. 内存评估

- OA-SD 模式下：student forward + teacher forward（no_grad）
- STM 额外内存：混合 batch 的 score/feat tensor（~2.5MB）+ loss_fn 计算（~50-100MB）
- 远程 16GB GPU：exp191 (OA-SD + CE) 已验证可运行，STM 额外开销 < 150MB
- 安全裕量充足

---

## d. 配置安全

- 默认值 `POSE_STM=False`，不影响其他实验
- exp198 通过命令行启用 `POSE_STM True`，与 OA-SD 配置项（`POSE_OA_SD True`）无冲突

---

## 问题汇总

无新问题。STM 和 OA-SD 是完全独立的 loss 组件，操作不同阶段，无数据/梯度/内存冲突。

---

## 结论

**审查通过。** STM（exp197 已审查通过）和 OA-SD（多次审查通过）在 processor 中的执行路径完全独立：STM 在主 loss 后立即执行（行 535-595），OA-SD 在 parallel_aug 之后执行（行 680-728），两者均只读取原始 `score`/`feat`，各自计算独立的 loss 分量并累加。无数据修改冲突、无 key 冲突、无梯度干扰。可以启动训练。
