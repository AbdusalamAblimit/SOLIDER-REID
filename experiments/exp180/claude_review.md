# Claude Broad Review: exp180 SupCon + PLBOA gradient (Opus 4.6)

## 审查通过

### 1. 设计合理性
单变量 vs exp176: 仅 POSE_LOWER_BODY_OCC_MODE 'lower' → 'gradient'。

### 2. gradient mode 实现验证
- `_apply_lower_body_occlusion` gradient 分支 (line 734-751) 已完整实现
- u^2 采样实现 bottom-heavy 分布
- occ_start clamp 到 [0, h-5]
- 与 lower mode 共享 occluder paste 和 keypoint 更新逻辑
- occ_x 在 paste 分支中正确赋值（occluders 和 fallback 两路都有）

### 3. CLI Override
- defaults.py: POSE_LOWER_BODY_OCC_MODE = 'lower'
- make_dataloader.py: 读取 cfg.MODEL.POSE_LOWER_BODY_OCC_MODE
- CLI override 'gradient' 正确覆盖

### 4. 无代码变更
纯配置实验。

### 5. 数值安全
gradient mode 的 `u^2` 采样: u ~ Uniform(0,1), occ_start = head_y + body_h * u^2.
当 u=0: occ_start = head_y (从头顶开始遮挡 = 最极端)
当 u=1: occ_start = head_y + body_h = foot_y (从脚开始 = 不遮挡)
P(occ_start > midpoint) = 1 - sqrt(0.5) ≈ 0.29, 即 71% 的遮挡从下半身开始。合理。

### 6. 与 SupCon 交互
gradient mode 改变输入图像（遮挡区域不同），但 SupCon 在特征空间操作。
两者独立：PLBOA 改变数据分布，SupCon 改变损失函数。无直接交互风险。

零 issue。
