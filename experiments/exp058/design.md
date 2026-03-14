# 实验 exp058: Realistic Occlusion Augmentation (ROA)

## 动机
- 当前训练使用 Random Erasing (RE) 做遮挡增强——随机矩形、均匀噪声填充
- DPEFormer (KBS 2025) 的 ROA 报告 +3.2% mAP vs RE; FCFormer 的 OIA 报告 +4.1% R1
- 真实遮挡不是矩形+噪声，而是有形状、纹理的物体（车、人、自行车等）
- ROA 从 Pascal VOC 2012 提取分割物体，以 alpha blending 粘贴到训练图像上
- **纯数据增强**——不修改模型架构或 loss

## 创新点 / 核心想法
- **核心假设**: 用真实物体遮挡替代随机矩形遮挡，能更好地模拟 Occluded-Duke 的真实遮挡模式
- **与 PGE (exp016) 的区别**: PGE 是 pose-guided body part erasing（擦除身体部位）。ROA 是物体粘贴（增加遮挡物）。方向相反
- **与 PAMC (exp050) 的区别**: PAMC 是双前向+一致性 loss。ROA 是纯数据增强，零额外计算
- **实现**: 从 ProFD 的 random_occlusion.py 适配，VOC 2012 ~5000 个 RGBA 遮挡物 patch

## 技术方案
- `datasets/occlusion_augmentation.py`: 独立模块，load_occluders() + occlude_with_objects()
- `datasets/pose_dataset.py`: 在 pad+crop 之后、tensor 转换之前应用 ROA
- `datasets/make_dataloader.py`: 启动时加载 occluders（一次性，~10s）
- 配置: POSE_ROA=True, POSE_ROA_PROB=0.5, POSE_ROA_PATH='data/VOCdevkit/VOC2012'
- ROA 不影响 heatmap/keypoints（遮挡物贴在图像上，但 pose 数据不变）
- 注意: ROA 与 RE 同时存在——ROA 贴物体后，RE 可能再叠加矩形擦除

## 预期结果
- 如果成功: 更真实的遮挡训练 → mAP/R1 提升 0.5-2%
- 如果失败: VOC 物体与 ReID 场景分布不匹配，或遮挡过强导致训练退化
- 如果中性: 当前 RE 已足够，更真实的遮挡物不提供额外信息

## 对照组
- Baseline: exp030a (PSG+GCN, 标准 RE)
- 消融变量: 增加 ROA 数据增强
