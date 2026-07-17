# Claude Broad Review: exp184 SupCon on GCN (Opus 4.6)

## 审查通过

### 代码变更 (make_loss.py:160)
`len(feat) > 3` → `len(feat) > 1`: 让 GCN (feat len=2) 也能走 SupCon 路径。
- GCN: feat[1:] = [gcn_pooled] → SupConLoss on (B, 768)
- 单视图 SupCon 在 batch 内做 same-ID/different-ID contrastive，有效且有意义
- NUM_INSTANCE=4 → 3 positives per anchor，足够

### 后向兼容
- STD-PR per-token: len=7 > 1 (仍 True，无变化)
- STD-PR pooled: len=2 > 1 (现在也可以走 SupCon，但 SUPCON 默认 False)
- Triplet use_norm: len>3 条件不变
- 非 list feat 路径不受影响

### Loss 流
global CE + SupCon(gcn_pooled) + global tri + part tri
= 0.5 * CE + 0.5 * SupCon + triplet

### Config
Base: pose_psg_gcn_plboa.yml (GCN + PLBOA)
Add: POSE_STR_SUPCON True, TEMP 0.05
GCN and STD-PR mutually exclusive (不冲突)

### SupConLoss 安全性
sim_max subtraction, log epsilon, degenerate case handling — 已在 exp174-179 验证。

### 数值安全
T=0.05: max sim/T=20, fp32 safe。sim_max subtraction 保证 max exponent=0。

零 issue。
