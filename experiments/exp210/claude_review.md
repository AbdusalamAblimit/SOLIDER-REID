# exp210 Code Review: Per-Keypoint Contrastive (PKC) Loss

## 审查范围

1. `experiments/exp210/design.md` — 设计合理性
2. `config/defaults.py` — 新默认值安全性
3. `processor/processor.py` — PKC loss 实现（lines 793-830）及 empty_cache 改动
4. `loss/supcon_loss.py` — SupCon loss 正确性
5. `model/modules/skeleton_gcn.py` — kp_data 生成（lines 650-906）
6. `model/pose_backbone_model.py` — kp_data 传递（lines 433-517）
7. 与已有 loss (CE, triplet, OA-SD) 的交互

---

## 1. Design Review

### 假设合理性: OK
- 动机清晰：MaxSim test-time 证明 per-keypoint features 有强 matching 信号，但训练时只受间接监督（pooled CE + triplet）。PKC 直接对每个 keypoint feature 做 SupCon，让每个 keypoint 独立学 discriminative representation。
- 这是对 GCN 头的直接强化，逻辑自洽。

### 创新性评估: Medium
- PKC 本身是将已有 SupCon 应用于 per-keypoint features 的简单扩展。作为独立创新论文价值有限，但作为 GCN+MaxSim pipeline 的重要组件，有消融实验价值。
- 不属于 CLAUDE.md 禁止的"小调参"类别 — 它确实给 GCN keypoint features 增加了全新的训练信号。

### 单变量原则: OK
- 在 exp206 (GCN+PAA+CE+OA-SD) 基础上只增加 PKC loss，单一变量。

---

## 2. Config Changes (`config/defaults.py`): PASS

```python
_C.MODEL.POSE_PKC = False                 # Default off — safe
_C.MODEL.POSE_PKC_WEIGHT = 0.5
_C.MODEL.POSE_PKC_TEMP = 0.07
_C.MODEL.POSE_PKC_VIS_THR = 0.3
```

- 所有新 config 默认为 False/合理值
- 不影响任何已有实验（POSE_PKC=False 时完全无操作）
- 位置合理（在 OA-RD 后，STM 前）

---

## 3. Processor Changes (`processor/processor.py`): PASS (with notes)

### 3a. PKC Loss 实现 (lines 793-830)

**数据可用性**: OK
- `kp_data['kp_feats']` 来自 `skeleton_gcn.py` line 864: `aux_data['kp_feats'] = kp_feats_enhanced`
- `kp_data['kp_weights']` 来自 line 865: `aux_data['kp_weights'] = kp_weights`
- GCN path 始终返回 5-tuple (line 517)，kp_data 始终非 None

**Feature 选择**: OK
- 使用 post-GCN `kp_feats_enhanced` 而非 pre-GCN `kp_feats`
- 这是正确选择：PKC 训练 GCN 增强后的特征，与 test-time MaxSim 使用的特征一致

**Visibility 阈值**: OK
- `kp_w` 是 `kp_weights`，默认 mode='score' 时等于 keypoint confidence scores (0-1)
- 注意：如果启用了 `kp_learnable_attn` 或 `kp_uncertainty`，`kp_weights` 会被乘以 attention/reliability
  - 这意味着 threshold 的语义会改变（不再是纯 confidence score）
  - 但当前 exp210 未启用这两个模块，所以不是问题

**最少样本检查**: OK
- `n_vis < 4` 跳过 — 合理，SupCon 需要足够的 positive/negative pairs
- `label_k.unique().shape[0] < 2` 跳过 — 正确，单一 ID 无法做 contrastive

**梯度流**: OK
- `kp_feats_enhanced` 有梯度（GCN forward 是正常计算图一部分）
- backbone features 已被 detach（line 434: `feat_map_detached = featmaps[-1].detach()`）
- 因此 PKC 梯度只流向 GCN 参数，不影响 backbone — 这是正确的设计

**Lazy-init SupCon**: OK
- `do_train._pkc_supcon = SupConLoss(temperature=pkc_temp)` — SupConLoss 无可学习参数
- 注意：SupConLoss 不需要加入 optimizer，因为它只有 `self.temperature` 常量

**Loss 求和**: OK
- `sum(pkc_losses) / len(pkc_losses)` 对 tensor list 正确
- 保持梯度图

**`_loss_details` 传递**: OK
- `details = getattr(loss, '_loss_details', {})` 获取旧值
- `loss = loss + pkc_weight * pkc_loss` 创建新 tensor
- `loss._loss_details = details` 挂到新 tensor — 与整个 codebase 一致

### 3b. `torch.cuda.empty_cache()` (line 926): OK
- 在 eval 前释放训练显存，合理优化
- 不影响正确性

---

## 4. SupCon Loss (`loss/supcon_loss.py`): PASS

- 标准 Supervised Contrastive Loss 实现
- `F.normalize` 确保输入在 unit hypersphere 上
- Log-sum-exp stability: `sim = sim - sim_max.detach()` — 正确
- 空 positive pair 处理: `has_pos.any()` check + 零梯度返回 — 正确
- Denominator 使用所有 non-self samples (positives + negatives) — 符合 Khosla et al.

**AMP 安全性**: OK
- Temperature=0.07 时 `sim` 范围 [-14.3, 14.3]
- Max subtraction 后 `exp` 值范围 [0, 1] — float16 安全
- `log(sum + 1e-8)` 提供数值下限 — 安全

---

## 5. Model Forward Path: PASS

### skeleton_gcn.py (lines 650-906)
- `kp_feats_enhanced` 是 GCN 输出 (line 812): `kp_feats_enhanced = self.gcn(kp_feats, kp_weights=gcn_kp_w)`
- Shape: (B, 17, C) where C=768 — 正确
- `kp_weights` shape: (B, 17) — 正确
- 返回 `aux_data` dict 包含两者 (lines 863-866) — 正确

### pose_backbone_model.py (lines 433-517)
- GCN 路径返回 5-tuple: `[cls_score] + gcn_cls_scores, [global_feat] + gcn_feats, featmaps, None, kp_data` (line 517)
- `kp_data` 直接从 `skeleton_head` 传出，包含 `kp_feats` 和 `kp_weights`
- Processor 在 line 490 正确接收: `score, feat, feat_maps, recon_loss, kp_data = model_out`

---

## 6. Runtime Error Checks

### Shape 兼容性: OK
- `kp_f.shape = (B, 17, C)` → `kp_f[vis_mask, k_idx, :].shape = (n_vis, C)` — 正确
- `target[vis_mask].shape = (n_vis,)` — 正确
- SupConLoss 接受 `(B, D)` features 和 `(B,)` labels — 匹配

### Device 兼容性: OK
- `kp_f` 和 `target` 都在 GPU 上（model forward 和 target.to(device) 保证）
- `vis_mask` 从 `kp_w` 衍生，同 device
- SupConLoss 内部创建的 tensors (eye, etc.) 使用 `device=features.device`

### Dtype 兼容性: OK
- 在 AMP autocast 下，`kp_f` 可能是 float16
- SupConLoss 的 `F.normalize`, `matmul`, `exp`, `log` 在 autocast 下安全
- `label_k` 是 long tensor — `unique()` 和 `==` 操作正确

---

## 7. Loss 交互分析

### PKC + CE/Triplet: OK
- CE/triplet 作用于 pooled `skeleton_feat`（从 kp_feats_enhanced 加权池化得到）
- PKC 作用于 per-keypoint `kp_feats_enhanced`
- 两者共享 GCN 参数但目标互补：CE/triplet 优化 pooled identity, PKC 优化 per-keypoint identity
- 不会冲突

### PKC + OA-SD: OK
- OA-SD 作用于 `feat` list (global + GCN pooled)，不涉及 per-keypoint features
- Teacher 的 kp_data 被丢弃（line 697）
- PKC 不影响 teacher，teacher 不影响 PKC

### PKC + OA-RD: OK
- OA-RD 作用于 global feature 的 pairwise similarity matrix
- 完全独立于 per-keypoint features

### 梯度累积: 注意但非问题
- GCN 参数同时接收 CE + triplet (via pooled feat) + PKC (per-keypoint) + OA-SD (via pooled feat) 的梯度
- 总梯度量较大但由 weight 控制（PKC_WEIGHT=0.5 合理）
- 如果训练不稳定可考虑降低 PKC_WEIGHT

---

## 8. 其他发现

### 缺少 Config 文件
- 未找到 exp210 的 YAML config 文件
- 需要创建 (e.g., `configs/occluded_duke/pose_psg_gcn_paa_pkc.yml`)
- 应基于 exp206 config 加上 `POSE_PKC: True`
- 还需要 `POSE_OA_SD: True` 及相关参数（exp206 base）

### 日志充分性: OK
- `pkc` (loss 值) 和 `pkc_nk` (参与计算的 keypoint 数量) 会被自动记录
- `pkc_nk` 可以帮助监控 visibility masking 是否过于激进

### `pkc_nk` 日志格式: Low
- `pkc_nk` 是整数 (0-17) 但被 `:.3f` 格式化输出为 "17.000"
- 功能无影响，仅显示略丑

---

## 审查结论

代码实现正确，无 bug。所有新增代码遵循 codebase 现有模式，defaults 安全（不破坏已有实验），loss 交互无冲突，AMP 安全，shape/device/dtype 一致。

审查通过
