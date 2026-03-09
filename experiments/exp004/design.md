# exp004: Pose Feature Modulation (PFM)

## 核心想法

**不只用 pose 做空间 pooling（part 提取），还用 pose 做特征调制（feature modulation）。**

之前的 exp001-003 证明了：
1. Pose-guided part pooling 有效（+0.9% mAP with part-only）
2. 但 id_part 收敛极慢（最终 ~2.0 vs id_global ~0.2）
3. 降低 global weight 反而伤害 backbone（exp003 负面结果）

**根本问题**：backbone 产出的特征图不是 pose-aware 的。Part pooling 只是在"不知道人体结构"的特征图上做空间选择。

**PFM 方案**：在 backbone 特征图和 GAP/part pooling 之间插入一个轻量调制模块：
```
pose heatmaps (17, H, W) → pose_encoder → modulation weights (C, H, W)
feat_map (C, H, W) × (1 + modulation) → modulated_feat_map
```

这样：
- Global feature 从 pose-conditioned 特征图 GAP 得到（更好）
- Part features 从 pose-conditioned 特征图 pooling 得到（更好）
- 两者都受益于 pose 调制

## 技术细节

### PFM 模块

```python
class PoseFeatureModulation(nn.Module):
    def __init__(self, pose_channels=17, feat_channels=768, hidden=64):
        self.encoder = nn.Sequential(
            nn.Conv2d(pose_channels, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, feat_channels, 1),
        )

    def forward(self, feat_map, pose_heatmaps):
        # Resize heatmaps to feature map size
        hm = F.interpolate(pose_heatmaps, size=feat_map.shape[2:])
        # Generate modulation weights
        mod = self.encoder(hm)  # (B, C, fH, fW)
        # Residual modulation
        return feat_map * (1 + mod)
```

参数量：17×64 + 64×768 ≈ 50K（极轻量）

### 架构变化
- 在 PoseReIDModel 中，backbone 输出后、GAP/part pooling 前，插入 PFM
- Global feature = GAP(PFM(feat_map))
- Part features = PosePartPooling(PFM(feat_map), heatmaps)

### 关键区别 vs 现有方法
- **现有 Part Pooling**: pose → 空间选择权重（where to look）
- **PFM**: pose → 特征调制权重（what to enhance/suppress）
- PFM 改变特征内容，Part Pooling 改变特征选择位置
- 两者正交，可以叠加

## Config

- `POSE_PFM_ENABLED: True`
- `POSE_PFM_HIDDEN: 64`
- 其他参数与 exp001 相同（sigmoid, 50/50 weight, part-only test）

## 预期

- Global feature 应该更好（因为 GAP 在 pose-modulated 特征上进行）
- Part features 也应该更好（因为 pooling 的底层特征更 discriminative）
- id_part 可能收敛更快（更好的底层特征 → 更易区分的 part features）

## 论文价值

这是潜在的核心创新之一。消融实验可以展示：
1. PFM alone（无 part pooling）vs baseline
2. Part pooling alone vs baseline
3. PFM + Part pooling vs baseline
4. 证明 PFM 和 part pooling 是互补的
