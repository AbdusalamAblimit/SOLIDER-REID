# Claude Broad Review: exp185 SupCon on STD-PR pooled (Opus 4.6)

## 审查通过

### 设计
单变量: SupCon 替代 CE on pooled STD-PR (no per-token)
测试: SupCon 是否需要 per-token diversity structure

### Activation path
len(feat)=2 > 1 → SupCon triggered。feat[1:]=[str_feat] (B,768 pooled)。
SupConLoss 接受 (B,D) input。正确。

### Score path
score=[cls_score, str_cls]。SupCon 替代 str_cls CE。global CE 保留。

### Triplet path
len(feat)=2, use_norm=False (pooled 768-d 不需要 L2 norm)。不受影响。

### Config
Base: pose_psg_stdpr_plboa.yml (STD-PR + PLBOA, no PER_TOKEN)
CLI: SUPCON True, TEMP 0.05, OUTPUT_DIR exp185
默认 ADDITIVE=False → replace mode

### 无代码变更
len>1 fix 已在 exp184 应用。

### 数值安全
T=0.05 已在 exp176/178/179 验证。

零 issue。
