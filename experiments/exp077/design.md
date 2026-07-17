# 实验 exp077: Scene+Target Concat PAA (ST-PAA)

## 动机
- exp066 PAA 使用 scene heatmap (17ch) 作为 adapter 输入 → 无法区分 target 和 distractor
- exp070 S&C 把 PAA 从 scene 切到 target-only → 负面 (-0.2%/-0.8%)，说明 scene context 重要
- exp076 TDPC 手工计算 H_diff = H_target - H_distractor 并用独立 adapter → 初期中性
- **新想法**: 不手工计算差异，而是把 [H_scene, H_target] concat 为 34 通道直接喂给 PAA
- 让模型自己学习 scene 和 target 之间的关系

## 创新点 / 核心想法
- **核心假设**: 把 scene 和 target 热图同时提供给单个 adapter，比单独给 scene 或手工计算差异更好
- **与 exp066 的区别**: PAA 输入从 17ch → 34ch ([scene, target])
- **与 exp076 TDPC 的区别**: TDPC 需要 2 个模块 (PAA + TDDA)，ST-PAA 只需 1 个更宽的 PAA
- **与 exp070 S&C 的区别**: S&C 用 target 替换 scene；ST-PAA 保留 scene 并额外加 target

## 技术方案
- 仅修改 PAA 模块的输入通道数: Conv2d(17→32→768) → Conv2d(34→32→768)
- 输入: `torch.cat([scene_heatmaps, target_heatmaps], dim=1)` (34 channels)
- 其他全部不变: PSG 仍用 scene，GCN 不变，loss 不变

### 修改文件清单
1. `model/pose_backbone_model.py`: 在 PAA 调用处 cat scene + target
2. `config/defaults.py`: 新增 `POSE_PAA_SCENE_TARGET` 开关
3. `configs/occluded_duke/pose_psg_gcn_paa_st.yml`: 新配置

### 参数变化
- PAA 第一层 Conv2d: 17×32 = 544 params → 34×32 = 1088 params (+544)
- 总增加 ~1K params (忽略不计)

## 数据统计
- 训练集 26.4% 多人图: scene ≠ target，adapter 能学到差异
- 训练集 73.6% 单人图: scene ≈ target，34ch 中前后两半几乎相同 → adapter 学到退化模式

## 预期结果
- 如果成功: mAP 相对 exp066 提升 0.3-1.0%
- 如果失败: 34ch 中信息冗余太高（单人图 73.6% 时 scene≈target），adapter 无法有效利用

## 对照组
- **Baseline 对照**: exp066 PAA seed1234 = 61.6%/74.2%
- **消融变量**: 仅改变 PAA 输入通道 (17→34)
