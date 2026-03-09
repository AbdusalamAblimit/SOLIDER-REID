# 实验 exp001: Pose Part Pooling (Soft Attention)

## 动机
- Phase 1 的 33 个实验证明 ViTPose visibility 向量方向已穷尽，最佳结果仅 +1.4% mAP
- 本轮实验转向使用 ViTPose-Huge 提取的**原始姿态热图**（17通道，模型 head.final_layer 输出）
- 首个实验验证最基本的姿态热图利用方式：基于热图的软注意力 part 池化
- 参考论文：BPBreID (WACV 2023) 的 part-based pooling 思路，但使用真实模型热图代替 human parsing

## 创新点 / 核心想法
- **核心假设**：使用真实 ViTPose 热图作为空间注意力权重，对 Swin 最后一层特征做 per-part 软池化，可以提取更具判别力的局部特征
- 与 baseline 相比：新增 5 个 body part 分支（head, upper_torso, arms, lower_torso, legs），每个使用热图加权的空间池化

## 技术方案
- **修改文件**: model/pose_model.py, model/modules/pose_part_pooling.py, model/modules/pose_utils.py
- **数据流**:
  1. 加载多人 NPZ 文件 → max 合并为场景级热图 (B, 17, H, W)
  2. Swin backbone → 最后一层特征 (B, 768, 12, 4)
  3. 热图 clamp(min=0) → 双线性插值到 (12, 4) → 17 通道分组为 5 body parts
  4. 每个 part：热图归一化 → 加权池化 → BN → 分类器
  5. 训练：0.5 * global_loss + 0.5 * mean(part_losses)
  6. 测试：concat [global | part0/5 | ... | part4/5]
- **关键超参数**: POSE_THRESHOLD=0.3, POSE_HEATMAP_SIZE=[96,32], 5 parts

## 预期结果
- 如果成立：mAP +1~2%（57.5~58.5%），与 Phase 1 的 GiLt+PCFC 相当或更好
- 如果失败：可能原因包括 (1) 热图分辨率在 12x4 特征图上信息不足 (2) 5 部位分组不够精细 (3) 多人合并热图引入噪声

## 对照组
- Baseline 对照：exp000 (mAP 56.6%, R1 66.5%)
- 消融变量：仅新增 PosePartPooling 模块，其他配置完全不变

## 论文定位
- 主实验表的 ablation 基线（"+ Part Pooling"行）
- 如果有效，将作为更复杂方法的基础组件
