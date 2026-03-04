# Paper 2: PFD - Pose-guided Feature Disentangling
**来源**: AAAI 2022
**仓库**: https://github.com/WangTaoAs/PFD_Net
**核心**: 姿态引导的特征解耦 + Transformer Decoder

## 代码架构概览
- 核心模型: `model/make_pfd.py` (PFA + PVM + Decoder)
- 姿态模型: `model/pose_net.py` (HRNet-W48 在线推理)
- Push Loss: `loss/pose_push_loss.py`

## 可拆解模块清单

### M1: PFA (Pose-guided Feature Alignment)
- 文件: `model/make_pfd.py` L538-574
- 功能: 热图线性映射→与编码器部件特征逐元素相乘→余弦相似度匹配→特征重排
- 数据流: HRNet热图[B,17,2048] → FC→[B,17,768] → 与部件特征相乘 → 对齐
- **移植可行性**: 高 | **显存**: <0.1G

### M2: PVM (Part-View Matching)
- 文件: `model/make_pfd.py` L503-536
- 功能: Encoder 部件特征 ↔ Decoder 查询特征的余弦相似度匹配
- **移植可行性**: 高 | **显存**: <0.05G

### M3: Transformer Decoder (2层)
- 功能: 零初始化查询 + 位置编码 → 交叉注意力融合热图信息
- **移植可行性**: 中 | **显存**: ~0.5G

### M4: SKT 遮挡检测 (阈值=0.3)
- 根据热图最大激活值判断关键点是否被遮挡
- 高/低置信度部件分别池化，Push Loss 拉开二者距离

## 损失函数
- 0.5*loss_encoder + 0.5*loss_decoder + push_loss
- Push Loss: 同一样本的高置信度 vs 低置信度特征的余弦距离

## 关键洞察
1. PFA 的热图权重映射简单有效，显存开销小
2. 两路特征分支(可见/遮挡)可处理遮挡，与 SOLIDER 语义解耦互补
3. **在线推理 HRNet 爆显存** → 必须离线提取
4. 推理时特征拼接到 27K+ 维，效率低 → 我们应避免此问题
