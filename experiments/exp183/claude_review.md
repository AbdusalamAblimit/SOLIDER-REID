# Claude Broad Review: exp183 SupCon on base without PLBOA (Opus 4.6)

## 审查通过

### 消融矩阵完整性
| | CE | SupCon T=0.05 |
|---|---|---|
| **No PLBOA** | exp166r (60.3/72.8) | **exp183 (?)** |
| **PLBOA** | exp166 (63.1/73.9) | exp179 (64.2/74.9) |
exp183 填充最后一个空格。

### 配置验证
1. No PAPE: POSE_PATCH_EMBED 默认 False，base config 未设置 ✓
2. No multi-stage PSG: POSE_PSG_STAGES 默认 [-1] (Stage 3 only) ✓
3. PLBOA disabled: CLI override POSE_LOWER_BODY_OCC False ✓
4. SupCon enabled: CLI override POSE_STR_SUPCON True, TEMP 0.05 ✓
5. ADDITIVE defaults False: replace mode (CE replaced by SupCon) ✓

### 单变量隔离
- vs exp166r: 仅增加 SupCon
- vs exp179: 仅移除 PLBOA
- vs exp181: 仅移除 PAPE + multi-stage PSG

### 无代码变更
纯 CLI 覆盖实验。

零 issue。
