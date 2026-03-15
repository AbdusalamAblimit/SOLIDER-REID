# 实验 exp070: Suppress-and-Complete (S&C)

## 动机
- exp066 PAA (additive adapter) 是突破性结果: +0.87%/+1.63% over 3-seed baseline
- 当前 PSG 和 PAA 都使用相同的 scene-level 合并热图（所有人 max-merge）
- 这存在语义混淆：PSG 本应"抑制背景"，PAA 本应"补全目标"，但两者用的是同一个信号源
- 用户建议的核心创新方向：**明确将 PSG 和 PAA 分工为两条 target-aware 路径**

## 创新点 / 核心想法
- **Suppress-and-Complete 双路径分工**：
  - **PSG (Suppress)**: 使用 scene-level 热图 → 抑制背景和无关区域 → `x = x * (1 + gate(scene))`
  - **PAA (Complete)**: 使用 target-person (person-0) 热图 → 添加目标人特有的身体结构信息 → `x = x + adapter(target)`
- 这是一个更原则性的设计：场景级信号用于空间选择（哪里有人），个体级信号用于内容补全（目标人的姿态结构）
- 解决了 CLAUDE.md 规则 8b 指出的多人图问题——PAA 现在只使用目标人数据

## 技术方案
- **修改文件**:
  1. `config/defaults.py`: 新增 `POSE_PAA_TARGET_ONLY` (bool, default False)
  2. `model/pose_backbone_model.py`:
     - `_prepare_pose()`: 新增返回 `target_heatmaps`（person-0 的原始热图）
     - `forward()`: 传递 `target_heatmaps` 到 backbone
     - `_run_backbone_with_psg()`: 接收并传递 `target_heatmaps`
     - `_run_stage_with_psg()`: PAA 使用 `target_heatmaps` 替代 `scene_heatmaps`

- **数据流**:
  - 输入: `pose_dict['heatmaps']` shape (B, max_persons, 17, 64, 48)
  - scene_heatmaps = max-merge 所有人 → (B, 17, 64, 48) → PSG
  - target_heatmaps = heatmaps[:, 0] → (B, 17, 64, 48) → PAA
  - 注意: person_mask[:, 0]=0 时，target_heatmaps 全零（安全，adapter 输出为零）

- **关键超参数**:
  - `POSE_PAA_TARGET_ONLY: True` (控制 PAA 使用 target 而非 scene 热图)
  - 其他参数沿用 exp066 (PAA bottleneck=32)

## 预期结果
- 如果假设成立（target-specific completion 优于 scene-level completion）：mAP +0.3~1.0% over exp066
- 核心价值不仅是数字提升，而是 PSG/PAA 分工的论文叙事更清晰
- 如果失败：说明 scene-level 多人信息对 PAA 也有价值（PAA 需要场景上下文）

## 对照组
- exp066 PAA b32 (scene for both PSG and PAA): 61.6%/74.2%
- 本实验相对于 exp066 只改了一个变量: PAA 的热图输入源从 scene → target
