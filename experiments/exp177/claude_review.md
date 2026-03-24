# Claude Broad Review: exp177 SupCon + PAPE (Opus 4.6)

## 审查通过

### a. design.md
单变量 vs exp171: 仅增加 POSE_STR_SUPCON=True。消融 SupCon 在简单架构上的效果。

### b. 代码审查
- SupCon loss 实现正确（supcon_loss.py: L2 normalize, max-subtraction, masking）
- make_loss.py: 条件链 evid→supcon→CE 正确匹配。SupCon 用 feat[1:]（features），不是 score[1:]（logits）
- per-token scores 仍被计算但在 SupCon 分支中未使用（harmless）
- Triplet 路径不受影响
- Global CE 保留

### c. 配置
- Base: pose_psg_stdpr_pertoken_plboa_pape.yml (PAPE + PSG@Stage3)
- No POSE_PSG_STAGES override → default [-1] → Stage 3 only
- POSE_STR_SUPCON=True 通过 CLI 添加
- POSE_STR_SUPCON_TEMP=0.07 使用默认值

### d. 单变量隔离
vs exp171: 仅 SupCon
vs exp174: SupCon 相同，但无 multi-stage PSG

### e. Eval 路径
Test features 不变（equal_concat: global + confidence-weighted pooled）

零 issue。
