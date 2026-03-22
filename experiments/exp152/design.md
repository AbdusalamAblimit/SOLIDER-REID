# 实验 exp152: Set-to-Set Metric Learning via Soft-MaxSim Triplet

## 动机

MaxSim test-time 在所有 checkpoint 上稳定 +1.0~1.5% mAP。这说明 set-to-set matching 优于 vector matching。但当前存在 **train-test metric mismatch**：

- 训练：keypoint features → pooled vector → euclidean triplet
- 测试：keypoint features → MaxSim (set matching) → distance

如果训练也用 MaxSim 作为距离函数，实现 train-test metric symmetry，应该进一步释放 set-level 特征的判别力。

## 文献定位

- **MoS (AAAI 2021)**: Jaccard set matching in ReID — 不同的 set 距离，不是 MaxSim
- **BPBreID/KPR (WACV23/ECCV24)**: per-part average distance — 不是 max-based alignment
- **ColPali (2024)**: MaxSim training for document retrieval — 证明 MaxSim training 可行，但在 vision-to-text 而非 ReID
- **Video-ColBERT (CVPR 2025)**: MaxSim training for video — 进一步验证

**结论：MaxSim training 在 person ReID 中从未被做过。这是一个真正的创新空白。**

## 核心假设

1. 用 Soft-MaxSim 替换 GCN branch 的 pooled triplet，让训练直接优化 set-level matching
2. 训练时每个 query keypoint 通过 temperature-softmax 对齐到最佳 gallery keypoint
3. 这应该让每个 keypoint feature 更具独立判别力
4. 最终表现为：MaxSim test 进一步提升，且 equal_concat test 也改善

## 技术方案

### Soft-MaxSim 距离

```python
# 对每个 query keypoint k，softmax 对齐到 gallery keypoints
attn_kj = softmax(cos(q_k, g_j) / tau)  # j = 1..17
sim_k = sum_j attn_kj * cos(q_k, g_j)

# 加权求和
SoftMaxSim(Q, G) = sum_k [w_k * sim_k] / sum_k w_k
distance = 1 - SoftMaxSim
```

tau=0.05: 低温使 softmax 尖锐（接近 hard max），但保留非零梯度到所有 gallery keypoints。

### 在 GCN branch 中的集成

| 组件 | 改动 |
|------|------|
| Global ID + Triplet | 不变 |
| GCN ID | 不变（仍用 pooled feature 做分类） |
| **GCN Triplet** | **替换为 Soft-MaxSim Triplet** |

### Batch Hard Mining with MaxSim

1. 计算 batch 内 (B, B) MaxSim 距离矩阵
2. 对每个 anchor，找同 ID 最远的 positive（hardest positive）
3. 找不同 ID 最近的 negative（hardest negative）
4. loss = max(0, d_ap - d_an + margin)

### 计算效率

- (B=64, K=17, D=768): einsum `ikh,jlh->ijkl` → (64,64,17,17)
- 4.7 MB tensor，~950M FLOPs ≪ backbone 4.5G FLOPs
- GPU 额外显存 < 50 MB

## 关键日志

- `tri_maxsim`: MaxSim triplet loss 值
- `maxsim_d_ap`: 均值 MaxSim 距离到 hardest positive
- `maxsim_d_an`: 均值 MaxSim 距离到 hardest negative
- `maxsim_margin`: d_an - d_ap（应该为正且增长）

## 预期结果

- MaxSim test: +0.3~0.8% mAP 超过 test-only MaxSim (62.2% → 62.5~63.0%)
- equal_concat test: +0.0~0.5% mAP 超过 exp030a (60.73% → 61.0~61.2%)
- 如果训练确实改善了 keypoint 判别力，per-keypoint retrieval mAP 应显著提升

## 对照组

- A: exp030a + equal_concat (60.73% mAP, 3-seed mean)
- B: exp030a + maxsim_hybrid test-only (62.2% mAP)
- C: **exp152 + maxsim_hybrid** (核心实验)
- D: exp152 + equal_concat (消融：训练是否也改善了标准 metric)

## 风险

1. **训练集 95.8% 可见 → MaxSim 退化为 1-to-1 fixed correspondence**: 但这等价于 per-keypoint independent triplet，仍然比 pooled triplet 更精细
2. **Soft-MaxSim 的 tau 敏感性**: 先用 0.05，必要时扫 0.02/0.1
3. **梯度冲突**: pooled ID loss 和 MaxSim triplet 方向一致（都要求同 ID 相似），不冲突

## 止损条件

- ep60 equal_concat mAP 低于 exp030a 同期 1.5% 以上 → 终止
- MaxSim triplet loss 出现 NaN → 终止
- maxsim_margin 长期 ≤ 0 → 机制失效，终止
