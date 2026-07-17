# 实验 exp016: PSG + Pose-Guided Erasing (PGE)

## 动机
- 15 个实验证明 PSG 58.3% 是 backbone injection 的上限，所有组合/扩展都失败
- 需要从完全正交的角度提升：**训练数据增强**
- 当前使用 Random Erasing (RE)，但 RE 不模拟真实遮挡——真实遮挡是整个身体部件被遮挡（如下半身被挡），而不是随机矩形
- 假设：结构化的 pose-guided 擦除比随机擦除能更好地训练模型应对遮挡

## 创新点 / 核心想法
- **核心假设**：用 pose 关键点引导的身体部件级擦除（Pose-Guided Erasing, PGE），比 Random Erasing 更能模拟真实遮挡场景，从而提升 occluded ReID 性能
- **与 baseline/exp007 的区别**：仅改变数据增强策略（RE → PGE），模型架构不变（仍用 PSG）

## 技术方案

### 修改文件
1. `datasets/pose_dataset.py` — 新增 `_pose_guided_erase` 方法，5 个 body part group 定义
2. `config/defaults.py` — 新增 `POSE_GUIDED_ERASING` 配置开关
3. `configs/occluded_duke/pose_psg_pge.yml` — 实验 config
4. `datasets/make_dataloader.py` — 传递 `pose_guided_erasing` 参数

### PGE 实现细节
1. **身体部件分组** (5 组，基于 COCO 17 关键点):
   - head: [0,1,2,3,4] (nose, eyes, ears)
   - left_arm: [5,7,9] (left shoulder, elbow, wrist)
   - right_arm: [6,8,10] (right shoulder, elbow, wrist)
   - torso: [5,6,11,12] (shoulders + hips，界定躯干区域)
   - legs: [13,14,15,16] (knees, ankles)

2. **PGE 流程** (基于关键点 bounding box 方法):
   - 以概率 0.5 (与 RE 相同) 触发
   - 随机选择 1 个身体部件组
   - 获取 person 0 该部件组的关键点坐标（score > 0.3 的有效关键点）
   - 如果有效关键点 < 2 个，fallback 到 Random Erasing
   - 计算关键点 bounding box + margin (宽度 15%, 高度 8%)
   - 在 bounding box 区域填充随机噪声
   - 将 pose 热图中对应通道的**擦除空间区域**清零（不是整个通道）
   - 将擦除区域内的关键点 score 清零
   - 注意：PGE **替代** Random Erasing，不是叠加

3. **关键超参数**:
   - p_pge = 0.5 (与 RE 相同的触发概率，共用 RE_PROB)
   - 每次擦除 1 个部件组
   - 关键点有效性阈值 > 0.3
   - margin_x = 15% 宽度 (~19px)，margin_y = 8% 高度 (~31px)

### 数据流
```
原始图像 + pose 热图
  → JointResize → JointFlip → JointPadCrop
  → ToTensor + Normalize
  → PoseGuidedErasing (替代 RandomErasing)
    → 随机选 1 个部件组
    → 从 person 0 关键点计算 bounding box
    → 填充随机噪声 + 清零对应热图空间区域 + 清零关键点 score
    → (如果关键点不足 fallback 到 Random Erasing)
  → 输出 (image, pose_dict)
```

## 预期结果
- **如果假设成立**: mAP 在 PSG 58.3% 基础上再提 0.5-1.5%，因为模型学到了更好的遮挡鲁棒性
- **如果失败**: (1) 擦除区域不够精确（bounding box 而非精确轮廓） (2) 擦除太多信息导致训练不充分 (3) Random Erasing 已经足够模拟遮挡

## 对照组
- Baseline 对照: exp007 PSG-only (mAP 58.3%, R1 67.9%)
- 消融变量: 数据增强策略（RE → PGE），其他完全不变
