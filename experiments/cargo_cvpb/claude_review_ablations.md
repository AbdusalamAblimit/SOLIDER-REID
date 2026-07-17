# Claude Broad Review — OVLI 消融(--ovli_match / --ovli_align)

**审查对象**: experiments/cargo_cvpb/afd_train.py 的 OVLIHead 消融开关(subagent 实现)+ smoke_ovli_ablations.py
**日期**: 2026-06-22
**结论**: 审查通过(无 Critical/High/Medium)

## 审查范围(全范围)
a. 消融设计合理性 / 单变量隔离
b. 新增代码逐行(_reduce_other / _row_mask4 / argparse / 构造 / eval)
c. 默认值字节级复现
d. train/test 对称
e. AMP 安全 / NaN-safe
f. 旧 ckpt 兼容

## 逐项

### 1. 设计合理性(非小调参)
- `--ovli_match {maxsim,avg}`: 隔离"late-interaction 的 MAX 选择 vs soft average"——审稿人必问"你的晚交互是不是等价全局平均?"。avg 退化成近全局软匹配(smoke T4: sym_maxsim ≈ <mean_q,mean_g>)。
- `--ovli_align {free,ordered}`: 隔离"自由 partial set matching vs AlignedReID 式有序行对齐"——审稿人必问"和 AlignedReID 有序对齐比有何优势?"(codex novelty 列的撞车点)。
- 两者都是 novelty-defense 必做对照, 非逃避创新的小调参。✓

### 2. 代码逐行
- `_reduce_other`(L399-434): align='free'+match='maxsim' → `sim.max(dim).values`(原行为逐字)。avg → mean。ordered → masked(maxsim 用 floor -1e4 / avg 用 clamp≥1)。✓
- `_row_mask4`(L312-317): `arange(K)//gw` 行索引 → K×K 行等价 → (1,K,1,K)。`register_buffer(persistent=False)`(跟 .to(device)、不进 state_dict、非 parameter)。✓
- match/align assert 校验(L292/305), 非法值直接报错。✓
- argparse(L722/733): 默认 maxsim/free。✓
- sym_maxsim_matrix(L451/456)两方向都走 _reduce_other → 对称保持。✓

### 3. 默认字节级复现
- maxsim+free 路径 == `sim.max(dim).values`。smoke T1: sym_maxsim `torch.equal=True` / `max|diff=0`, loss `torch.equal=True`(1.803385)。✓
- 旧 smoke_ovli_allview 回归数值不变(loss=1.429576)。✓

### 4. train/test 对称
- eval maxsim_block(L592/595)用同一 `ovli._reduce_other(sim, other_dim=3/1)` + 同 pool/topk/thresh/tau → rerank 与训练 loss 同 match/align。L590-591 注释明确。smoke T5 矩形 eval-shape(Nq≠Ng)归约形状正确 == 行受限 ref。✓

### 5. AMP / NaN-safe
- 没动 OVLI loss 的 `autocast(enabled=False)` fp32 位置; 新代码全在已有 fp32 token 路径。✓
- ordered masked-max 用 finite floor(-1e4); cosine 相似度 ∈[-1,1] 永远 > -1e4, floor 不会被误选。masked-mean `clamp(min=1)`。固定 gh×gw 网格每行恒有 gw≥1 token → floor 永不触发(只有退化 gw=0 才可能, 不存在)。NaN-safe。✓

### 6. 旧 ckpt 兼容 / optimizer 自检
- buffer non-persistent → 不进 state_dict → 旧 OVLI ckpt `load_state_dict(strict=True)` 不缺 key。✓
- buffer 非 parameter → optimizer 自检 `proj.parameters() 在 param_groups` 与参数计数不变(smoke T2: params=['proj.weight','proj.bias'] #=2, buffer in_state_dict=False)。✓

## Findings
- **Critical: 无。High: 无。Medium: 无。**
- Low: 无实质问题(floor -1e4 远小于 cosine 下界 -1, 安全; ordered 在固定网格下每行非空)。

## 结论
审查通过。实现单变量、插件式、默认字节级复现、train/test 对称、AMP/NaN 安全、旧 ckpt 兼容。smoke 5 组(T1 复现/T2 隔离/T3 ordered/T4 avg/T5 对称+NaN)全过 + allview 回归不变。codex 审过 + GPU 空出后即可训练。
