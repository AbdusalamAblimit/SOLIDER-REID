# Claude Broad Review: exp190 3-view + CE (Opus 4.6)

## 审查通过

### 实验设计
消融：3-view parallel aug + CE (无 SupCon)。单变量 vs exp187 (3-view + SupCon)。
目的：isolate parallel aug 的独立贡献。

### 代码
无代码修改。纯 CLI 覆盖：
- MODEL.POSE_STR_SUPCON False（关闭 SupCon → 走 CE 路径）
- MODEL.POSE_PARALLEL_AUG True（3-view 增强）
- OUTPUT_DIR 修改

### Loss 路径验证
SUPCON=False → make_loss.py line 192: `part_ids = [ce_fn(s, target) for s in score[1:]]`
CE 路径正确激活。每个 view 各自走完整 CE+triplet loss → 平均。

### 单变量隔离
vs exp187: 仅 POSE_STR_SUPCON True→False。其余完全相同。
vs exp166: 有多个变量不同（PAPE, multi-stage, parallel_aug），但非主要消融目标。

### defaults.py
无新默认值。SUPCON=False 和 PARALLEL_AUG=False 都是已有默认。

### 无代码层面风险
纯 config ablation。

### 实验价值
如果 3-view+CE > 1-view+CE → parallel aug 本身有效
如果 3-view+CE ≈ 1-view+CE → parallel aug 只在 SupCon 下有效（synergy）

零 issue。
