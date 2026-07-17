# 实验 exp170: Pose-Guided Multi-Part Occlusion Augmentation (PGMPOA)

## 动机
- PLBOA 是目前最强的单项改进（+2.8 mAP），证明数据增强是最有力的改进来源
- 但 PLBOA 只遮挡**下半身**（hip 以下），而 Occluded-Duke 测试集有**各种类型的遮挡**
- exp167/168/169 全部中性，证明模型侧小修改已到天花板
- **核心假设：更多样化的遮挡训练 → 更鲁棒的模型**

## 核心假设
在 PLBOA 基础上，额外随机遮挡一个上半身部位（头/左臂/右臂），让模型训练时接触到全身各部位遮挡场景。

## 技术方案

### 增强管道
```
原始图像
  ↓
PLBOA (p=0.7): 遮挡下半身
  ↓
PGMPOA (p=0.3): 额外遮挡 {头 / 左臂 / 右臂} 中随机一个
  ↓
标准增强 (flip, crop, random erasing)
```

### 上半身部位定义
| 部位 | Keypoints | Padding |
|------|-----------|---------|
| Head | nose(0), L-eye(1), R-eye(2), L-ear(3), R-ear(4) | 30% |
| Left Arm | L-shoulder(5), L-elbow(7), L-wrist(9) | 20% |
| Right Arm | R-shoulder(6), R-elbow(8), R-wrist(10) | 20% |

### 遮挡概率分析
- P(下半身遮挡) = 0.7
- P(上半身部位遮挡) = 0.3
- P(两者同时) = 0.7 × 0.3 = 0.21
- P(只有下半身) = 0.7 × 0.7 = 0.49
- P(只有上半身) = 0.3 × 0.3 = 0.09
- P(无遮挡) = 0.3 × 0.7 = 0.21

### 修改文件
1. `datasets/pose_dataset.py`: 新增 `_apply_upper_body_part_occlusion` 方法 (~80 行)
2. `datasets/make_dataloader.py`: 接线 `upper_body_occ` 配置
3. `config/defaults.py`: 新增 `POSE_UPPER_BODY_OCC` 和 `POSE_UPPER_BODY_OCC_PROB`

### 关键实现细节
- 用 keypoints 坐标计算身体部位 bbox（带 padding 扩展）
- 至少 2 个可见 keypoints 才执行遮挡（避免在不可见部位做无效操作）
- 遮挡后更新 keypoint scores/visibility/heatmap（与 PLBOA 行为一致）
- 使用与 PLBOA 相同的 VOC occluder patches（alpha blending）

## 预期结果
- 如果假设成立：mAP/R1 > exp166 (63.1/73.9)，因为模型见过更多遮挡模式
- 上半身遮挡可能伤害 R1（头和躯干是最判别的区域），但 mAP 应该提升
- 如果失败：p=0.3 上半身遮挡太激进，破坏了太多判别信息

## 对照组
- exp166 (per-token + PLBOA only): 63.1/73.9
- 消融变量：仅增加 POSE_UPPER_BODY_OCC: True, POSE_UPPER_BODY_OCC_PROB: 0.3
