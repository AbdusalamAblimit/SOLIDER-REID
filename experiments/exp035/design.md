# 实验 exp035: Visibility 最小闭环消融

## 动机
- 当前 GCN/KPP 的 confidence-weighted pooling 使用 ViTPose `scores`（检测置信度）
- `scores` 在遮挡关节上仍然偏高（0.6-0.8），因为 ViTPose 对位置估计有信心
- `visibility` 更有区分力：可见关节 0.9-1.0，遮挡关节 0.0-0.1
- 假设：用 visibility 替代/补充 scores 做加权池化，可以更好地降权遮挡关节

## 核心假设
- Visibility 能提供比 score 更准确的遮挡信号
- 在 keypoint pooling 中，遮挡关节应该被更强力地降权
- Visibility 在 score 的基础上有独立价值

## 技术方案

### 新增配置
- `MODEL.POSE_KP_WEIGHT_MODE`: 选择权重模式
  - `'score'`: 使用 scores（当前行为，baseline）
  - `'visibility'`: 使用 visibility
  - `'score_visibility'`: 使用 scores * visibility
  - `'binary_visibility'`: 使用 visibility_binary

### 代码修改
1. `config/defaults.py`: 添加 `POSE_KP_WEIGHT_MODE = 'score'`（默认值 = 当前行为）
2. `model/modules/skeleton_gcn.py`:
   - `_sample_keypoint_features()`: 额外返回 visibility 数据
   - `forward()`: 根据 weight_mode 选择不同的权重向量
3. `model/pose_backbone_model.py`: 传递 weight_mode 到 GCN head

### 四种消融模式

| 模式 | 权重公式 | 预期行为 |
|------|---------|---------|
| score | w = scores | 当前行为，大多数关节权重 0.6-0.95 |
| visibility | w = visibility | 遮挡关节 ~0，可见关节 ~1 |
| score_visibility | w = scores * visibility | 双重降权遮挡关节 |
| binary_visibility | w = vis_binary | 0/1 硬掩码，完全丢弃遮挡关节 |

## 预期结果
- 如果 visibility 有独立价值：visibility / score_visibility 模式应优于 score
- 如果无独立价值：所有模式性能接近，说明 score 已足够
- binary_visibility 可能因信息丢失而略差（过于激进）

## 对照组
- Baseline: exp030a (PSG + GCN, score weighting, equal_concat mAP 60.73% 3-seed mean)
- 消融变量：仅 keypoint pooling 的权重模式

## 优先执行顺序
1. score（确认 baseline 可复现）
2. score_visibility（预期最强）
3. visibility_only（如果时间允许）
4. binary_visibility（如果时间允许）
