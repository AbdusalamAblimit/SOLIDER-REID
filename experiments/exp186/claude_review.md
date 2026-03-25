# Claude Broad Review: exp186 SupCon without PSG (Opus 4.6)

## 审查通过（第二轮，修复 Critical bug 后）

### 第一轮发现的 Critical Bug
POSE_BACKBONE_PSG=False 会导致 make_model.py 构建 PoseReIDModel（旧模型）而非 PoseBackboneModel，
丢失 STD-PR, PAPE, SupCon, PLBOA 所有组件。

### 修复方案
使用 POSE_PSG_STAGES '[]'（空列表）代替 POSE_BACKBONE_PSG=False。
PoseBackboneModel 仍被构建（保留 STD-PR, PAPE, SupCon, PLBOA），
但 psg_stage_indices=set() 导致所有 stage 走 normal path（无 PSG gating）。

### 验证
- psg_stages=[] → psg_stage_indices=set() → 无 PSG modules 创建
- _run_backbone_with_psg: stages not in psg_stage_indices → stage(x, hw_shape) normal path
- psg_modules_dict 为空 → 无额外参数
- PAPE 仍生效（独立于 PSG，在 PatchEmbed 后注入）
- STD-PR 仍生效（在 backbone 输出后做 cross-attention）
- SupCon 仍生效（loss function 独立于架构）
- PLBOA 仍生效（dataset augmentation 独立于模型）

### 单变量
vs exp176: 仅 PSG gating 被禁用（POSE_PSG_STAGES [-1] → []）。

### backward compat 兼容

PSG_STAGES='[]' 是 CLI 覆盖。默认 [-1] 不变。不影响其他实验。

### 梯度流
无 PSG 时 backbone 是标准 Swin forward。STD-PR 从 detached feature map 提取 tokens。
Global CE + SupCon + triplet 梯度仅通过 backbone standard path。正确。

### 无其他 issue
