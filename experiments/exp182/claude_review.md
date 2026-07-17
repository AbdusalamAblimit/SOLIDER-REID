# Claude Broad Review: exp182 SupCon+CE Joint (Opus 4.6)

## 审查通过

### 1. design.md
单变量 vs exp176: 仅增加 POSE_STR_SUPCON_ADDITIVE=True。CE + 0.5*SupCon 叠加训练。

### 2. Additive mode 实现 (make_loss.py:160-184)
- ADDITIVE=True: part_id_avg = part_ce_avg + 0.5 * supcon_avg — 正确
- ADDITIVE=False (default): part_id_avg = supcon_avg — 与原行为一致
- supcon_avg 作为中间变量，重构干净不改变结果

### 3. defaults.py
- POSE_STR_SUPCON_ADDITIVE = False: 安全默认
- POSE_STR_SUPCON_WEIGHT = 0.5: 仅在 ADDITIVE=True 时使用

### 4. 梯度流
- CE: score[1:] → classifier → BN → raw_tokens — 正确
- SupCon: feat[1:] → L2_norm → contrastive — 正确
- 两条路径独立但共享底层权重

### 5. No interference
score (logits) 和 feat (features) 是独立张量，来自模型不同计算路径。

### 6. 日志
ADDITIVE 时记录 supcon, id_part_ce, id_part (combined)。充分。

### 7. Loss weighting
最终: ID_W * (0.5*global_ce + 0.5*(part_ce + 0.5*supcon)) + TRI_W * triplet
SupCon effective weight = 0.25 * global CE。合理，不会导致不稳定。

### 8. Triplet unchanged
Per-token triplet 不受影响。

零 issue。
