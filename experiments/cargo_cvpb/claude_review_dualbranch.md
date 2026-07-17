# Claude Broad Review — AIRL 单模型双分支(--airl_dualbranch,kill-switch #4)

**审查对象**: afd_model.py(2nd BNNeck L407-414 + forward L467-516)+ afd_train.py(f_rec CE+consistency 训练块 + airl_dualbranch_eval L1028-1099 + argparse L1474 + 验证 L1553);smoke_airl_dualbranch.py
**日期**: 2026-06-23
**结论**: 审查通过(无 Critical/High/Medium)

## 审查范围(全范围)
a. 设计合理性(双分支内化 #3 软融合 +1.46 进 1 forward)
b. forward 双 head 数据流(共享 global_feat,1 forward 出双特征)
c. airl_dualbranch_eval 融合公式(=#3 GATE-5)
d. off 字节级复现
e. 两 head 真分化 + 都进 optimizer
f. AMP fp32 + train/test 对称(w 固定)

## 逐项
### 1. 设计合理性
#3 oracle 验证: 软融合 AIRL+baseline 距离 → mean +1.46(合法 w=0.25)。双分支 = 1 backbone + 2 BNNeck head 内化这个融合: **f_full**(原 head,ID-CE+shared triplet,无 consistency,保 G→A)+ **f_rec**(新 BNNeck,自己 ID-CE + AIRL consistency,服务低清 A→G)。eval 软融合两 head cosine。framing 钉死"observation-limited evidence ceiling + 按 query 像素预算路由证据空间"(避 resolution-adaptive dual-branch 撞车)。

### 2. forward 双 head(afd_model.py L467-516)
- eval: `return_dual=False`(默认)→ 单 f_full(legacy 不变);`return_dual+airl_dualbranch` → (f_full, f_rec) 从**同一 pooled global_feat**(L490-491)= 1 forward 出双特征。✓
- train: dict 加 `bn_feat_rec`/`logits_rec`(L507-509),2nd BNNeck + classifier on shared global_feat。triplet 在 shared global_feat **不重复**(L505-506)。✓

### 3. airl_dualbranch_eval 融合(afd_train.py L1028-1099)
- 1 forward 提双特征(L1058)。✓
- 软融合 `dm_fuse = 2−2·(w·s_rec+(1−w)·s_full)`(L1093)= **#3 GATE-5 公式精确一致**,w=args.airl_fuse_w 固定。✓
- 报 full/rec/**fuse** 三档 × A→G/G→A,model-selection 用 fuse mean。✓
- 镜像 run_cross_view_eval → f_full 数字 bit-for-bit 复现 baseline。✓

### 4. off 字节级复现
`--airl_dualbranch` off → 不建 2nd head(L409 if),dict 无 `*_rec`,eval 单特征,loss 不触碰。smoke D1/D1b `max|df|=0`。✓

### 5. 两 head 真分化 + optimizer
- consistency 只读 f_rec → **f_full 零 consistency 梯度**(smoke D4),f_rec 拿 consistency+CE 梯度(D3/D8)。✓
- 两 head 在 model.parameters() 自动进 optimizer(Swin 上 full-LR 组,非 backbone-scaled),显式 assert + smoke D5/D10。✓

### 6. AMP/NaN + train/test 对称
consistency 在 autocast(enabled=False) fp32,clean 侧 detach。w 固定非 test 调(train/test 对称)。smoke D6/D7。✓

## Findings
- **Critical/High/Medium: 无。Low: 无实质问题。**

## 结论
审查通过。双分支 = 2 BNNeck head(f_full 保 G→A / f_rec consistency 服务低清 A→G,共享 1 forward)+ 融合 eval(=#3 GATE-5)+ off 字节级 + 两 head 真分化 + 都进 optimizer + NaN-safe + train/test 对称。smoke 11/11 + 回归 21/21。codex 审 + GPU 即训 kill-switch #4:**dualbranch fuse mean ≥ baseline 60.84 +1.0(=61.84)→ 机制成立(B 类候选);否则杀。**
