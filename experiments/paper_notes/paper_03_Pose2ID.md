# Paper 3: Pose2ID - From Poses to Identity
**来源**: CVPR 2025
**仓库**: https://github.com/yuanc3/Pose2ID
**核心**: 训练无关的特征中心化(NFC) + 身份引导的行人生成(IPG)

## 代码架构概览
- NFC: `NFC.py` (37行，核心模块)
- IPG: `IPG/inference.py` + `IPG/src/models/pose_guider.py`
- ID2指标: `ID2.py`

## 可拆解模块清单

### M1: NFC - 邻域特征中心化 (最核心)
- 文件: `NFC.py` L16-35
- 算法: 计算欧氏距离 → 找 k1 近邻 → 过滤为相互邻域(k2) → 特征聚合 → L2归一化
- 参数: k1=2, k2=2
- **移植可行性**: 极高(测试后处理，不改网络) | **显存**: 0 | **预期收益**: mAP +2-5%

### M2: IPG - 身份引导行人生成
- 基于 Stable Diffusion v1.5
- 需 IFR(Identity Feature Reformer): ReID特征→20个token序列
- **移植可行性**: 低(需SD权重4GB+，推理慢) | 暂不考虑

### M3: ID2 指标
- 功能: 量化特征紧凑度(每个样本到其身份中心的距离)
- **用途**: 可视化分析，论文图表素材

## 关键洞察
1. NFC 是"零训练成本"的通用提升方法，对任何 ReID 模型有效
2. 核心发现: 特征空间存在"隐藏正样本"(相互邻域)
3. 可探索**姿态感知的 NFC**: 结合姿态相似度找邻域，而非纯特征相似度
4. **Part-level NFC**: 对可见部分激进聚合，遮挡部分保守处理
