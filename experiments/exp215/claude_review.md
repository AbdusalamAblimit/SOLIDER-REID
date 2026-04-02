# exp215 BA-PKC Review (Claude Broad Review v1)

## 审查范围

a. design.md 合理性
b. 新增/修改代码逐行审查 (pose_backbone_model.py, processor.py, defaults.py)
c. config 默认值安全性
d. 数据流/梯度流验证
e. 与前序实验对照 (exp210b PKC on detached, exp206r baseline)

---

## a. design.md 审查

**合理性**: 高。design.md 清晰阐述了核心问题：exp210b/exp211 中 PKC/MST 的梯度被 `featmaps[-1].detach()` 阻断，导致 per-keypoint loss 只更新 GCN 的 ~200K 参数，无法影响 backbone 50M 参数。BA-PKC 通过从 non-detached feature map 采样 keypoint features 来解决这个问题。

**单变量原则**: 满足。唯一变化是新增 BA-PKC loss 通道，GCN 仍使用 detached features，Global CE/Triplet 不受影响。

**假设清晰度**: 明确。"SupCon 梯度方向与 Global CE 一致"的论证合理——两者都推动同 ID 特征聚拢、异 ID 特征分离。不同于 Part CE（为每个 keypoint 学习分类边界，梯度方向复杂），SupCon 直接优化度量空间。

**创新性评估**: 这不是小调参。虽然代码量不大，但它解决了一个此前实验确认的结构性问题（梯度被阻断）。理论动机清晰，是 exp210b 失败的直接响应。通过。

---

## b. 代码逐行审查

### pose_backbone_model.py

**初始化 (line 115-118)**:
```python
self.ba_pkc = getattr(cfg.MODEL, 'POSE_BA_PKC', False)
if self.ba_pkc:
    print('[BA-PKC] Backbone-aware per-keypoint contrastive enabled')
```
- 正确。无需额外参数/层，BA-PKC 只是从现有 feature map 采样。

**Forward — BA-PKC 采样 (lines 521-535)**:
```python
if getattr(self, 'ba_pkc', False):
    raw_fm = featmaps[-1]  # (B, C, fH, fW) — NOT detached!
    kp_coords = pose_dict['keypoints'][:, 0, :, :]  # (B, 17, 2)
    input_h, input_w = x.shape[2], x.shape[3]
    grid_x = (kp_coords[:, :, 0] / input_w * 2 - 1).clamp(-1, 1)
    grid_y = (kp_coords[:, :, 1] / input_h * 2 - 1).clamp(-1, 1)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # (B, 17, 1, 2)
    sampled = F.grid_sample(raw_fm, grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
    ba_kp_feats = sampled.squeeze(-1).permute(0, 2, 1)  # (B, 17, C)
```

逐点检查:

1. **`featmaps[-1]` is not detached**: 正确。这是核心设计意图。梯度从 SupCon loss → ba_kp_feats → grid_sample → featmaps[-1] → backbone。

2. **`pose_dict['keypoints'][:, 0, :, :]`**: 取 person 0 的 17 个关键点。Shape (B, 17, 2)。正确。

3. **坐标映射**: `x.shape` 是原始输入 (B, 3, 384, 128)。keypoints 在原图像素空间。
   - **Low: 轻微精度偏差**: `align_corners=True` 时，理论上完美映射是 `grid = 2*p/(W-1) - 1`，代码用的是 `grid = 2*p/W - 1`。对于 W=128，差异约 0.8%。不会导致功能错误，对训练影响可忽略。已经被 `clamp(-1,1)` 保护。

4. **grid_sample 参数**:
   - `mode='bilinear'`: 正确，支持 backward。
   - `padding_mode='border'`: 安全，OOB 坐标用边界值。
   - `align_corners=True`: 标准用法。

5. **Shape 推导**:
   - `grid`: (B, 17, 1, 2) — grid_sample 期望 (N, H_out, W_out, 2)，这里 H_out=17, W_out=1。
   - `sampled`: (B, C, 17, 1)
   - `.squeeze(-1)`: (B, C, 17)
   - `.permute(0, 2, 1)`: (B, 17, C)
   - 正确。

6. **kp_data 安全**: `if kp_data is None: kp_data = {}` — 此分支是 `use_skeleton_gcn` 内部，kp_data 由 skeleton_head 返回（line 470），不会是 None。安全但防御性代码无害。

7. **梯度图**: `ba_kp_feats` 保持在计算图中。`kp_data` 是一个普通 dict，不会断开梯度。正确。

### processor.py (lines 870-903)

```python
ba_pkc_enabled = getattr(cfg.MODEL, 'POSE_BA_PKC', False)
if ba_pkc_enabled and kp_data is not None and 'ba_kp_feats' in kp_data:
    ba_pkc_weight = float(getattr(cfg.MODEL, 'POSE_BA_PKC_WEIGHT', 0.1))
    ba_vis_thr = float(getattr(cfg.MODEL, 'POSE_PKC_VIS_THR', 0.3))
    ba_kp_f = kp_data['ba_kp_feats']  # (B, 17, C) — NOT detached!
    ba_kp_w = kp_data['kp_weights']    # (B, 17) — visibility weights
```

逐点检查:

1. **`kp_data['kp_weights']` 存在性**: skeleton_head 的 aux_data 始终包含 `kp_weights` (skeleton_gcn.py line 865)。BA-PKC 只在 `use_skeleton_gcn` 分支内启用，所以 `kp_weights` 一定存在。正确。

2. **visibility threshold**: 使用 `POSE_PKC_VIS_THR` (默认 0.3)。BA-PKC 没有自己独立的阈值参数，复用 PKC 的。可接受，但如果同时启用 PKC 和 BA-PKC 且需要不同阈值则受限。当前场景无影响。

3. **SupCon 惰性初始化**:
   ```python
   if not hasattr(do_train, '_ba_pkc_supcon'):
       from loss.supcon_loss import SupConLoss
       do_train._ba_pkc_supcon = SupConLoss(temperature=0.07)
   ```
   - **Low: temperature 硬编码**: 0.07 是标准值，但未从 config 读取。BA-PKC 没有 `POSE_BA_PKC_TEMP` 配置项。如果需要调温度需改代码。当前可接受。
   - SupConLoss 无可学习参数，设备无关。正确。

4. **Per-keypoint 循环 (lines 886-895)**:
   - 每个 keypoint 独立计算 SupCon loss
   - `vis_mask.sum() < 4`: 最少 4 个可见样本。合理。
   - `label_k.unique().shape[0] < 2`: 至少 2 个不同 ID。合理。
   - `feat_k = ba_kp_f[vis_mask, k_idx, :]`: boolean indexing 保持梯度。正确。

5. **Loss 累加**:
   ```python
   ba_pkc_loss = sum(ba_losses) / len(ba_losses)
   loss = loss + ba_pkc_weight * ba_pkc_loss
   ```
   - 对所有可计算 keypoint 的 loss 取平均，再乘权重 0.1 加到总 loss。
   - `loss` 本身包含 Global CE + Global Triplet + GCN losses，BA-PKC 贡献 ~0.1 * SupCon_value。
   - 权重 0.1 较保守，合适作为初始实验。

6. **日志记录**: `details['ba_pkc']` 和 `details['ba_nk']` 正确记录。可在训练日志中观察到 BA-PKC loss 值和贡献 keypoint 数。

### defaults.py (lines 188-190)

```python
_C.MODEL.POSE_BA_PKC = False              # Enable BA-PKC
_C.MODEL.POSE_BA_PKC_WEIGHT = 0.1         # BA-PKC loss weight
```
- 默认关闭。安全，不影响已有实验。
- 权重 0.1 合理（保守起步）。

---

## c. 配置文件检查

尚未创建 exp215 专用 config。需要基于 `pose_pds_sg_gcn.yml`（或类似 skeleton GCN config）创建，添加:
```yaml
MODEL:
  POSE_BA_PKC: True
  POSE_BA_PKC_WEIGHT: 0.1
```

**注意**: BA-PKC 要求 `POSE_SKELETON_GCN: True`，因为它运行在 `use_skeleton_gcn` 分支内。

---

## d. 梯度流验证

**正向数据流**:
1. 输入 x → backbone with PSG → featmaps[-1] (B, 768, 12, 4)
2. BA-PKC: featmaps[-1] (NOT detached) → grid_sample → ba_kp_feats (B, 17, 768)
3. GCN: featmaps[-1].detach() → skeleton_head → gcn_cls_scores, gcn_feats, kp_data
4. Global: featmaps[-1] → GAP → BNNeck → cls_score, global_feat

**反向梯度流**:
- Global CE/Triplet → 通过 global_feat 回到 backbone (正常)
- GCN CE/Triplet → detach 阻断，不影响 backbone (正常)
- **BA-PKC SupCon → 通过 ba_kp_feats → grid_sample → featmaps[-1] → backbone** (新增)

**梯度冲突分析**:
- Global CE 推动: 相同 ID 的 global feature 在分类边界同侧
- BA-PKC SupCon 推动: 相同 keypoint 位置的 same-ID features 在超球面上靠近
- 两个方向兼容: 都要求 same-ID features 相似。SupCon 是更直接的度量学习信号。
- 风险: 如果 BA-PKC 梯度过强，可能干扰 Global CE 的收敛。权重 0.1 提供了缓冲。

**内存影响**:
- grid_sample 的 backward 需要保存 featmaps[-1] (B, 768, 12, 4) 和 grid (B, 17, 1, 2)。
- featmaps[-1] 已经因为 Global CE 需要保留在计算图中，所以 BA-PKC 不额外保存 feature map。
- 额外开销仅为 grid tensor 和 backward kernel 的中间结果。估算 < 50MB。可忽略。

---

## e. 与前序实验对照

| 实验 | Per-kp features 来源 | 梯度到 backbone? | 结果 |
|------|---------------------|-------------------|------|
| exp206r | GCN (detached) | No | 72.3 maxsim (baseline) |
| exp210b | GCN (detached) + PKC | No (SupCon only更新 GCN) | 72.4 maxsim (无效) |
| exp215 | Backbone (non-detached) + BA-PKC | **Yes** | 目标 73-74% |

差异明确: exp215 唯一变化是 SupCon 梯度流入 backbone。单变量隔离良好。

---

## f. 交互/边界检查

1. **dual_branch 互斥**: BA-PKC 代码在 `dual_branch_active` return 之后 (line 517-519)。如果同时启用 dual_branch + BA-PKC，BA-PKC 被跳过。不会崩溃但会静默无效。当前 exp215 不使用 dual_branch，无影响。**Medium: 建议添加警告 log 或 assert**，但不阻挡本次实验。

2. **`pose_dict is None` 保护**: BA-PKC 代码在 `self.use_skeleton_gcn and pose_dict is not None` 条件内 (line 438)。如果无 pose 数据，整个分支不执行。安全。

3. **AMP 安全**: grid_sample、F.normalize、matmul、exp、log 全部支持 float16。无问题。

4. **已有实验可复现性**: BA-PKC 默认 False，不影响任何已有配置。安全。

---

## 问题汇总

| 级别 | 问题 | 位置 | 说明 |
|------|------|------|------|
| Low | grid 坐标映射轻微偏差 | model L527-528 | `align_corners=True` 理论映射应为 `2*p/(W-1)-1`，实际用 `2*p/W-1`。差异 <1%，不影响训练。 |
| Low | SupCon temperature 硬编码 | processor L883 | 固定 0.07，无 config 可调。当前可接受。 |
| Medium | dual_branch + BA-PKC 互斥无警告 | model L500-535 | dual_branch return 在 BA-PKC 之前，BA-PKC 被静默跳过。当前实验不受影响，但建议添加日志。 |

**没有 Critical 或 High 级别问题。**

---

## 结论

代码实现正确，梯度流设计合理，内存开销可控，不破坏已有实验。Low/Medium 问题不影响 exp215 的正确执行和结果可靠性。

**审查通过**
