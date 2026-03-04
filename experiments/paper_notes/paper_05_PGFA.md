# Paper 5: PGFA - Pose-Guided Feature Alignment
**来源**: ICCV 2019
**仓库**: https://github.com/lightas/ICCV19_Pose_Guided_Occluded_Person_ReID
**核心**: 经典姿态热图掩码加权 + 共同可见区域匹配

## 代码架构概览
- 模型: `model.py` L58-82 (PCB 基线)
- 特征提取: `test.py` L106-123 (pg_global_feature)
- 匹配: `shared_region_evaluate.py`

## 可拆解模块清单

### M1: 姿态热图掩码加权特征聚合
- 文件: `test.py` L106-123
- 功能: 18个关键点热图分别与空间特征逐元素相乘 → 各自池化 → max融合
- **移植可行性**: 高 | **显存**: <0.1G

### M2: 共同可见区域匹配
- 文件: `shared_region_evaluate.py`
- 功能: query/gallery 只比较双方都可见的关键点
- **移植可行性**: 高 | **显存**: 0

## 关键洞察
1. 设计最简单但有效: 热图×特征→池化，无可学习参数
2. 我们的 VPReID PosePartHead 本质上就是 PGFA 的升级版(分组+visibility+temperature)
3. 共同可见区域匹配的思路已在我们的 fused 评估中部分实现
4. **局限**: 18个关键点 max 池化为1个特征，信息损失大
