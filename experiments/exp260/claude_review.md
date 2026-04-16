# Claude Review — exp260 Base GCN512+2stage

**审查通过** — 本实验是 exp255 (Small) 的 backbone 升级，无代码修改。

## 审查范围

### a. design.md
- 动机合理：Small 已达 75.2/85.6 (MaxSim+flip)，Base 预计 +2-3%
- 非小调参：backbone 升级是论文标准消融（Tiny→Small→Base）
- 假设清晰：更大 backbone → 更强特征 → 所有模块增益保持或放大

### b. 代码修改
- **无新增/修改代码**。仅 config 文件更改：
  - `TRANSFORMER_TYPE`: swin_small → swin_base
  - `PRETRAIN_PATH`: swin_small.pth → swin_base.pth
  - `GCN_HIDDEN`: 256 → 512（与 exp255 对齐）
  - `PSG_STAGES`: [-1] → [-2,-1]（与 exp255 对齐）
  - `TEST.IMS_PER_BATCH`: 256 → 128（Base 模型更大）

### c. 配置文件
- `configs/occluded_duke/pose_psg_lgpa_gcn_base.yml`: WITH_CP=True ✓, LR=4e-4 ✓
- `configs/market/pose_psg_lgpa_gcn_base.yml`: PLBOA=False ✓（Market 非遮挡数据集）

### d. defaults.py
- 无修改

### e. processor
- 无修改

### f. 对照
- 对照 exp255 (Small): 73.2/83.3
- 单变量: backbone 尺度 (Small→Base)

## 结论
审查通过。纯 backbone 升级，无代码风险。
