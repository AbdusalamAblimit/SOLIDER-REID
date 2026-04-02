# 实验 exp210: GCN+PAA+CE+OA-SD + Per-Keypoint Contrastive (PKC)

## 动机
- MaxSim hybrid 在 exp206 checkpoint 上无需重训给 +1.8% mAP (70.3→72.1)
- 这证明 GCN-enhanced per-keypoint features 有强大的 matching 信号
- 但这些 keypoint features 在训练时只受到 pooled-CE + triplet 的间接监督
- **如果直接用 per-keypoint contrastive loss 训练，MaxSim test-time gain 会更大**

## 核心假设
Per-keypoint SupCon loss 让每个 keypoint feature 独立地学习 discriminative representation。
结合 MaxSim hybrid test-time matching，预期比 exp206 + MaxSim (72.1) 更强。

## 技术方案

### PKC (Per-Keypoint Contrastive) Loss
对 17 个 keypoint features 分别做 SupCon:
```
For each keypoint k in [0, 17):
    feat_k = kp_feats[:, k, :]  # (B, C)
    w_k = kp_weights[:, k]      # (B,) visibility
    # Only include samples where this keypoint is visible (w_k > threshold)
    valid = w_k > 0.3
    if valid.sum() < 4: continue
    loss_k = SupCon(feat_k[valid], target[valid])
total_pkc_loss = mean(loss_k for all valid k)
```

### 关键设计
1. **Visibility masking**: 只对可见 keypoint 计算 contrastive loss
2. **与 CE/triplet 共存**: PKC 作为额外 loss，不替换已有 CE/triplet
3. **与 OA-SD 共存**: PKC 在 student forward 的 kp_feats 上计算，不涉及 teacher

### 修改文件
- `config/defaults.py`: 添加 `POSE_PKC`, `POSE_PKC_WEIGHT`, `POSE_PKC_TEMP`
- `processor/processor.py`: 在 training loop 添加 PKC loss 计算
- `loss/make_loss.py`: 或复用已有 SupCon loss

### 超参数
- PKC weight: 0.5 (不要太大，避免抢 CE 梯度)
- PKC temp: 0.07 (标准 SupCon temperature)
- Visibility threshold: 0.3 (匹配 MaxSim test-time threshold)

## 预期结果
- 假设成立: mAP 73-74% (with maxsim_hybrid), R1 83-85%
- 如果 PKC 与 OA-SD 冲突: 降 PKC weight 或只对高 visibility keypoint 计算

## 对照组
- exp206 (GCN+PAA+CE+OA-SD, equal_concat): 70.3/81.8
- exp206 (GCN+PAA+CE+OA-SD, maxsim_hybrid): 72.1/82.9
- exp210 (+ PKC, maxsim_hybrid): 目标 73-74%
