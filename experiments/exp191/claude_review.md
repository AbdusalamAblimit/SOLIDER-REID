# Claude Broad Review: exp191 OA-SD + CE (Opus 4.6)

## 审查通过

### 实验设计
OA-SD with CE (no SupCon)。消融 OA-SD 的独立有效性。
单变量 vs exp166: 仅增加 POSE_OA_SD=True。

### 代码路径验证
- Dataset: _oa_sd_mode → 保存 clean image → 2-view output (student, teacher)
- Collate: n_views=2 → list of 2 tensors
- Processor: oa_sd_mode=True → EMA teacher creation + distillation loss
- Loss: CE path (SUPCON=False default)。OA-SD distillation 独立于 CE/SupCon。
- EMA: deepcopy → eval → no_grad。Forward 时 train() → eval()。

### 显存
Student forward ~10GB + teacher no_grad ~3GB + EMA params ~112MB = ~13GB。5060 Ti 16GB 够。

### 已知 LOW issues (from exp188 审查)
1. BN running stats 在 teacher train() forward 时被污染 — 但 distillation 用 pre-BN features
2. EMA 只更新 parameters 不更新 buffers — 标准 trade-off

### PLBOA warning
Base config 有 PLBOA=True。warning 不会触发。Teacher/student asymmetry 正确建立。

### 单变量
vs exp166: 仅 OA_SD=True。其余 (PSG@S3, STD-PR, per-token CE, PLBOA) 不变。

### 实验价值
如果 OA-SD+CE > CE alone → OA-SD 独立于 SupCon 有效
如果 OA-SD+CE ≈ CE alone → OA-SD 需要 SupCon 才有效

零 blocking issue。
