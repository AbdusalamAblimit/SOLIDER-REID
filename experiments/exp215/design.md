# 实验 exp215: Backbone-Aware Per-Keypoint Contrastive (BA-PKC)

## 动机

### 关键发现 (exp210b/exp211)
- PKC/MST 的梯度被 `featmaps[-1].detach()` (line 434) 阻断
- 所有 per-keypoint loss 的梯度只更新 GCN 的 ~200K 参数，无法影响 backbone 的 50M 参数
- 这就是为什么 PKC/MST 完全不改变 equal_concat 性能

### 解决方案: Backbone-Aware PKC (BA-PKC)
- 从 **non-detached** backbone feature map 直接采样 keypoint features
- 这些 raw keypoint features 用于 PKC SupCon loss
- 梯度直接回到 backbone → backbone 为 per-keypoint discriminability 学习
- GCN 继续用 detached feature map → Part CE/triplet 不干扰 Global

### 为什么不直接去掉 detach?
- exp022 证明了去掉 StopGrad (detach) 会让 Part 梯度干扰 Global，导致 -1.4% mAP
- BA-PKC 只让 **contrastive** 梯度回传，不让 **classification** 梯度回传
- SupCon 的梯度方向是 "推同 ID 近，异 ID 远"，这与 Global CE 方向一致，不会冲突

## 核心假设
Backbone 直出的 keypoint features + SupCon loss 可以让 backbone 学习 keypoint-level discriminability。
与 MaxSim test-time matching 结合后，预期 +1-3% mAP 超过基线。

## 技术方案

### 实现
在 `pose_backbone_model.py` 的 forward (training) 中：
```python
# 现有: GCN 用 detached feature map
feat_map_detached = featmaps[-1].detach()
gcn_output = skeleton_head(feat_map_detached, ...)

# 新增: BA-PKC 用 non-detached feature map
if self.ba_pkc:
    raw_kp_feats = bilinear_sample(featmaps[-1], keypoints)  # (B, 17, C) — gradients flow to backbone!
    kp_data['ba_kp_feats'] = raw_kp_feats
```

在 `processor.py` 中：
```python
if ba_pkc_enabled and 'ba_kp_feats' in kp_data:
    # SupCon on backbone-direct keypoint features
    for k in range(17):
        vis_mask = kp_w[:, k] > 0.3
        if vis_mask.sum() >= 4:
            loss_k = supcon(ba_kp_feats[vis_mask, k], target[vis_mask])
            ba_pkc_losses.append(loss_k)
```

### 修改文件
- `config/defaults.py`: POSE_BA_PKC, POSE_BA_PKC_WEIGHT
- `model/pose_backbone_model.py`: 新增 non-detached keypoint sampling
- `processor/processor.py`: BA-PKC loss 计算

## 对照组
- exp206r (no per-kp loss): 72.3 maxsim
- exp210b (PKC on detached GCN feats): 72.4 maxsim
- exp215 (BA-PKC on backbone feats): 目标 73-74% maxsim
