# exp194 审查报告

## 审查范围
超参数实验: OA-SD weight 1.0 -> 2.0, 无代码修改

## 检查项

### 1. design.md
- 动机清晰: 验证 OA-SD distillation loss 权重敏感性
- 单变量: 仅改 POSE_OA_SD_WEIGHT (1.0 -> 2.0)
- 对照组明确: exp191 (weight=1.0) = 63.2/75.4

### 2. POSE_OA_SD_WEIGHT 定义 (config/defaults.py:178)
- `_C.MODEL.POSE_OA_SD_WEIGHT = 1.0` -- 默认值正确, 不影响其他实验

### 3. 权重使用 (processor/processor.py:619, 654)
- 行 619: `oa_sd_weight = float(getattr(cfg.MODEL, 'POSE_OA_SD_WEIGHT', 1.0))` -- 正确读取
- 行 654: `loss = loss + oa_sd_weight * oa_sd_loss` -- 正确应用
- 命令行 override `MODEL.POSE_OA_SD_WEIGHT 2.0` 会被 yacs 正确解析为 float

### 4. 数值安全性
- oa_sd_loss 是 cosine distance: `(1 - cos_sim).mean()`, 范围 [0, 2]
- weight=2.0 时最大贡献 = 4.0, 与 CE + triplet 量级兼容, 无溢出风险

### 5. 显存 (5060 Ti 16GB)
- 无额外模块/参数, 与 exp191 完全相同, 显存安全

## 结论

纯超参数实验, 代码路径与 exp191 完全一致, 仅 weight 从 1.0 变为 2.0. 无风险.

**审查通过**
