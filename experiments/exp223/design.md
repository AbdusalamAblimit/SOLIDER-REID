# 实验 exp223: PADPQ — Pose-Anchored Deformable Part Queries (Tiny)

## 动机

### 当前 GCN 的局限
GCN 的 keypoint feature sampling 使用 `grid_sample` 在精确的 keypoint (x,y) 坐标处采样。
问题:
1. Keypoint 坐标不精确（pose estimator 的误差）
2. 遮挡时 keypoint 位置的 feature 可能是 occluder 的，不是 person 的
3. 判别性信息可能不在精确坐标处，而在附近区域

### PADPQ 的核心思想
不在固定的 keypoint 坐标采样，而是学习 **每个 keypoint 周围的 K 个偏移采样点**。
模型可以自适应地将采样点移到更有判别力的区域。

### 与已有方法的区别
| 方法 | 采样方式 | 问题 |
|------|---------|------|
| GCN (当前) | 固定 keypoint grid_sample | 依赖 pose 精度 |
| STD-PR (exp175等) | Cross-attention over all tokens | 太 diffuse, 不收敛 |
| XCAD (exp053) | Cross-attention decoder | 同上 |
| **PADPQ** | **Deformable: keypoint + learned offsets** | **局部、高效、可学习** |

### 创新门槛检查
1. ✅ **问题层面**: 重新定义 "part feature extraction" 为 deformable sampling (vs fixed sampling)
2. ✅ **机制层面**: Pose-anchored deformable attention 在 ReID 中是新的
3. ✅ **证据层面**: 直接 ablation: fixed sampling (GCN) vs deformable sampling (PADPQ)

## 核心假设
Keypoint 附近的 learned receptive field 比精确坐标采样更好，
特别在 keypoint 不精确或被遮挡时。

## 技术方案

### PoseAnchoredDeformableHead
替换 SkeletonGCNHead 中的 `_sample_keypoint_features`:

```python
# 当前: 精确采样
kp_feats = grid_sample(feat_map, keypoints)  # (B, 17, C)

# PADPQ: deformable 采样
# 1. 在 keypoint 处初始采样
initial_feats = grid_sample(feat_map, keypoints)  # (B, 17, C)
# 2. 预测 K 个偏移
offsets = offset_head(cat(initial_feats, keypoints))  # (B, 17, K*2)
# 3. 在偏移位置采样
sample_points = keypoints + offsets  # (B, 17, K, 2)
sampled = grid_sample(feat_map, sample_points)  # (B, 17, K, C)
# 4. Attention 加权聚合
attn = attn_head(initial_feats).softmax(-1)  # (B, 17, K)
kp_feats = (sampled * attn).sum(2)  # (B, 17, C)
```

### 修改文件
- `model/modules/skeleton_gcn.py`: 在 SkeletonGCNHead 中添加 deformable sampling
- `config/defaults.py`: POSE_DEFORMABLE_SAMPLE, POSE_DEFORMABLE_K

### 关键设计
1. **Zero-init offsets** — 初始行为 = 原始 GCN（identity start）
2. **K=4 采样点** — 每个 keypoint 周围 4 个偏移（上下左右）
3. **在 detached feature map 上操作** — 与现有 GCN 兼容，不引入梯度冲突
4. **与 GCN 兼容** — 采样后仍经过 skeleton GCN propagation

## 预期结果
- PADPQ 在遮挡图像上应该优于 fixed sampling
- 预计 mAP +0.5-1% on Tiny (63.2 → 63.7-64.2)
- MaxSim matching 应该进一步受益（更好的 per-keypoint features）

## 对照组
- exp191 OA-SD (fixed sampling): 63.2/75.4
- exp220 GSPB (fixed sampling + gradient scale): 62.9/74.3 (maxsim 64.6)
