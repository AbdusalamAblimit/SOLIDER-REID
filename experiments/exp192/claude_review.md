# Claude Review: exp192 OA-SD EMA decay=0.99 (Opus 4.6 审查)

## 审查通过

### 代码改动
1. processor.py line 396: `ema_decay = 0.999` → `ema_decay = float(getattr(cfg.MODEL, 'POSE_OA_SD_EMA_DECAY', 0.999))`
   - 从 config 读取 EMA decay 而非硬编码
   - 默认值 0.999 不变 → exp191 可复现
   - getattr + float() 安全

2. config/defaults.py: 新增 `POSE_OA_SD_EMA_DECAY = 0.999`
   - 安全默认
   - 不影响任何已有实验

### 实验设计
单变量 vs exp191: 仅 EMA decay 0.999 → 0.99
base 架构 + CE + PLBOA + OA-SD

### EMA decay=0.99 的含义
- 0.999: teacher 更新很慢（需要 ~1000 步才能 50% 更新）
- 0.99: teacher 更新更快（~100 步 50% 更新）
- 更快更新 = teacher 更紧跟 student → distillation target 更 fresh
- 风险：太快可能导致 teacher 和 student 太相似，distillation 信号太弱

### 后向兼容
POSE_OA_SD_EMA_DECAY=0.999 默认不变。

### 无其他改动
纯 config 消融。零代码逻辑变更。

### 显存
与 exp191 完全相同（EMA teacher + 2x forward）。远程 16GB 够。
