# exp196 审查报告: 3-view + SupCon + OA-SD Global-Only

## 审查范围

- design.md 合理性与假设清晰度
- config/defaults.py 所有配置项存在性
- processor/processor.py 4-view 模式正确性、loss 计算顺序、梯度分离
- datasets/pose_dataset.py 4-view tuple 构建
- datasets/make_dataloader.py 标志位设置
- loss/make_loss.py SupCon + parallel_aug 交互
- 显存估计

## 审查结论

**审查通过** -- 无 Critical/High/Medium/Low 问题。所有代码路径已在前序实验中验证，本实验仅为配置组合。

## 逐项验证

### 1. 配置项存在性 (config/defaults.py)

全部 4 个关键配置项已注册:
- `POSE_PARALLEL_AUG` (L176): 默认 False
- `POSE_STR_SUPCON` (L161): 默认 False
- `POSE_OA_SD` (L177): 默认 False
- `POSE_OA_SD_GLOBAL_ONLY` (L180): 默认 False

辅助配置 (`POSE_OA_SD_WEIGHT`, `POSE_OA_SD_EMA_DECAY`, `POSE_STR_SUPCON_TEMP`) 均存在且默认值合理。无新配置需要添加。

### 2. 4-view tuple 构建 (datasets/pose_dataset.py)

- L167-172: `_oa_sd_mode=True` 时，PLBOA 之前保存 `img_clean_for_oa_sd`
- L194-226: `parallel_aug=True` 时构建 3 view (full/ROA/heavy)
- L221-224: 当 `img_clean_for_oa_sd is not None` 且 `parallel_aug=True`，第 4 个 view 作为 teacher 追加
- 最终 img_tensor = (full, roa, heavy, clean) 4-tuple

make_dataloader.py L123-127: `parallel_aug=True` 和 `_oa_sd_mode=True` 独立设置，无冲突。

collate_fn (L1054-1070): `n_views=4` 时正确进入 else 分支，返回 `[B_v1, B_v2, B_v3, B_teacher]` list。

### 3. 4-view 模式检测 (processor.py L433-454)

- L436: `parallel_aug = isinstance(img, list) and len(img) >= 3` -- True (len=4)
- L437: `oa_sd_mode = isinstance(img, list) and len(img) == 2` -- False (len=4)
- L439: `parallel_oa_sd = parallel_aug and oa_sd_enabled and len(img) == 4` -- True
- L441-443: 正确分离 3 student views 和 1 teacher view
- `img_teacher` 在此赋值，后续 OA-SD 块 (L624) 可访问

### 4. Forward pass (processor.py L460-476)

3 student views 顺序 forward，各产出 (score, feat, feat_maps, recon_loss, kp_data)。
- `score, feat` 设为 view 1 的输出 (L477)
- `kp_data` 设为 view 1 的输出 (L479)
- 所有 views 的 score/feat 存入 all_scores/all_feats

### 5. Loss 计算顺序 -- 核心验证

**Step 1**: View 1 loss (L532): `loss_fn(score, feat, target, target_cam, kp_data=kp_aux_data)`
- score 为 list (global + part logits) -> 进入 L128 分支
- feat 为 list (global + tokens) -> 进入 L160 SupCon 分支
- SupCon 对 `feat[1:]` (per-token features) 计算对比损失
- Global CE 对 `score[0]` 计算
- Triplet 对 `feat[0]` (global) + `feat[1:]` (tokens) 分别计算

**Step 2**: Views 2&3 loss (L606-615):
- `loss_fn(all_scores[vi], all_feats[vi], target, target_cam)` -- 注意 **无 kp_data 参数**
- SupCon 由 config flag `POSE_STR_SUPCON` 控制 (L160)，不依赖 kp_data
- 因此 views 2&3 也正确计算 SupCon
- visibility-weighted SupCon 路径 (L172) 因 `kp_data is None` 不触发，退回 uniform averaging (L180)
- 这是正确行为: view 1 可用 visibility 加权，views 2&3 用均匀加权

**Step 3**: 三 view loss 平均 (L614): `loss = loss / 3`

**Step 4**: OA-SD distillation (L617-665):
- 条件 `oa_sd_enabled and parallel_oa_sd` 满足
- Teacher forward (L622-628): no_grad，使用 `img_teacher`
- Global-only 模式 (L638-645): `feat[0]` 是 view 1 的 global feature
- OA-SD loss 加在 averaged loss 之后 (L663): `loss = loss + oa_sd_weight * oa_sd_loss`

### 6. 梯度分离验证

- **SupCon**: 操作 `feat[1:]` (per-token features)，梯度流向 structural token decomposition 模块
- **OA-SD global-only**: 操作 `feat[0]` (global pooled feature)，梯度流向 backbone + global pooling
- **CE global**: 操作 `score[0]` (global logits)，梯度流向 backbone + classifier
- **Triplet global**: 操作 `feat[0]`，梯度流向 backbone + global pooling

SupCon 和 OA-SD 在 feature 层面完全分离: SupCon on tokens, OA-SD on global。二者梯度不冲突。
Global CE 和 OA-SD 都作用于 global feature，但这是 by design -- CE 提供判别力，OA-SD 提供遮挡不变性。

### 7. EMA Teacher 更新 (processor.py L688-692)

在 optimizer.step() 之后更新。`ema_teacher` 在 L397-401 初始化。
EMA 更新代码与 exp193/exp195 相同，无变化。

### 8. 显存估计

- 3 student forward (with grad): 各约 5-6 GB activation memory (AMP)
- 1 teacher forward (no_grad): ~1 GB (no activation storage)
- SupCon loss: 计算 per-token cosine similarity matrix，~O(B^2 * K * D)，negligible
- 估计峰值: ~21-22 GB (exp193 = 20.9 GB + SupCon overhead)
- 在 24 GB 3090 内可行

### 9. Design.md 评估

- 动机清晰: 基于 exp187/exp193/exp195 三条已验证路线的合并
- 核心假设明确: "职责分离" 使三者增益 additive
- 对照组完整: exp187 (SupCon only)、exp193 (OA-SD+CE)、exp195 (SupCon+OA-SD)
- 预期结果合理: 上限 65.0-65.5%，有明确失败判定标准

**关于"小调参"质疑**: 本实验是纯配置组合，无新代码。但它组合的三个组件各自已单独验证有效，且 exp188 证明了 naive 全 token OA-SD + SupCon 失败，exp195 证明了 global-only 版本成功。因此本实验是对"职责分离"假说在完整 3-view 环境下的最终验证，具有论文消融表价值。可以接受。

### 10. 风险评估

唯一风险: 3-view 下 OA-SD global-only 的增益可能被 3-view 本身的 regularization 效果吸收（3-view 已提供多样遮挡模式）。这属于科学问题而非代码问题。

## 最终结论

**审查通过。** 所有代码路径在 exp187/exp190/exp193/exp195 中已验证。4-view 数据流、loss 计算顺序、梯度分离、EMA 更新均正确。显存在安全范围内。可以启动训练。
