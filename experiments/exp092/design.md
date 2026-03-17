# 实验 exp092: Learned Skeleton Recovery Module (LSRM)

## 动机
- SGCFR (exp090) test-time +2.6% mAP — 证明 cross-image kp recovery 有效
- 但 SGCFR 用 naive weighted average 做 recovery — 不够智能
- TTSFR (exp091) batch-level recovery 中性 — batch 太小 (4 imgs/ID)
- **改进**: 训练一个 learned recovery module，让 recovery 更精准

## 创新点
**从 naive average 到 learned cross-attention recovery**

训练 SRM (Skeleton Recovery Module):
- 输入: query 的遮挡 kp feat + candidate 的可见 kp feat + visibility
- 输出: recovered kp feat (比 naive average 更精准)
- 训练: 用同 ID pair 做监督——用 A 的 kp 恢复 B 的遮挡部分，target 是 B 的真实 kp

这让 SGCFR 从 "test-time trick" 升级为 "learned recovery module"。

## 技术方案
```python
class SkeletonRecoveryModule(nn.Module):
    # Cross-attention: query's occluded kp attend to candidate's visible kps
    # Returns: recovered kp features
    
    def forward(self, query_kp, cand_kp, query_vis, cand_vis):
        # 1. 标记遮挡和可见
        occ_mask = query_vis < threshold
        vis_mask = cand_vis >= threshold
        
        # 2. Cross-attention: occluded queries attend to visible keys
        Q = proj_q(query_kp[occ_mask])   # occluded keypoints as queries
        K = proj_k(cand_kp[vis_mask])    # visible keypoints as keys
        V = proj_v(cand_kp[vis_mask])    # visible keypoints as values
        recovered = softmax(QK^T/sqrt(d)) @ V
        
        # 3. Replace occluded kp with recovered
        output = query_kp.clone()
        output[occ_mask] = recovered
        return output
```

## 训练方式
- 在 GCN forward 中，对同 ID pair 做 recovery + 重建 loss
- Loss: MSE(recovered_B_from_A, original_B) for occluded keypoints of B
- 同时用恢复后的特征参与主 ID+Triplet loss

## 测试方式
- 和 SGCFR 结合: 先找 top-K candidates，用 SRM 做 learned recovery
- 代替 naive weighted average

## 对照
- exp090 SGCFR (naive average): 64.2%/75.7%
- exp066 PAA (no recovery): 61.6%/74.2%
