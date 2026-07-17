# Claude Broad Review — OVLI SetPool(--ovli_setpool)

**审查对象**: afd_train.py 的 OVLISetPool 类(netvlad/attn/gated/secondorder)+ aggregate_tokens + sym_maxsim_matrix/eval 分支 + argparse + 优化器自检;smoke_ovli_setpool.py
**日期**: 2026-06-23
**结论**: 审查通过(无 Critical/High/Medium)

## 审查范围(全范围)
a. 设计合理性(DCVP / OVC-SetVLAD 机制 = codex 8 舰队角度2 推荐)
b. OVLISetPool 4 模式逐行(置换不变 + NaN-safe)
c. 默认 setpool=mean 字节级复现
d. train/test 对称
e. 新参数进 optimizer
f. AMP fp32 安全

## 逐项
### 1. 设计合理性(非小调参)
`--ovli_setpool` 把固定 mean-pool(=52.37 最强)换成可学习置换不变集合池化。netvlad = codex 角度2 推荐的 OVC-SetVLAD(跨视角无对应 token 集合的残差分布建模),是 DCVP headline 的核心机制验证。

### 2. 代码逐行(L232-326)
- **netvlad**(L309-316): softmax-assign(over C)+ residual(tok-centers)+ sum(over K)+ 簇内 norm + 全局 L2 + linear。标准 NetVLAD,sum over K = 置换不变。✓
- **attn**(L317-323): H 头 learned query, softmax(over K)+ sum(over K)+ linear。置换不变。✓
- **gated**(L324+): sigmoid 门 + gated sum /(Σg).clamp(eps)。置换不变 + NaN-safe。✓
- **secondorder**: low-rank reduce + 协方差(z^Tz/K)+ signed-sqrt + linear。协方差阶不变。✓
- 所有模式只对 K 轴 sum/softmax-sum/mean → 置换不变(smoke T2 打乱 token diff<1e-5)。

### 3. 默认字节级复现
setpool=mean → setpool_mod=None → 所有 `!= 'mean'` 分支为假 → 逐字落回原 MaxSim 代码。smoke T1a-d: maxsim+avg 两路径 torch.equal/1e-6, 不加参数。旧 2 个 smoke(ablations/allview)回归不变。✓

### 4. train/test 对称
train sym_maxsim_matrix(L596)与 eval maxsim_block(L735)共用同一 `aggregate_tokens`。✓

### 5. 新参数进 optimizer
setpool_mod 是 ovli 注册子模块 → list(ovli.parameters()) 递归收进。L1031 assert 自检 setpool params 全在 opt_ids。smoke T3 四模式 setpool=True。✓

### 6. AMP / NaN
aggregate 在 autocast(enabled=False) fp32 路径。gated clamp(min=eps), secondorder sqrt(|·|+eps), 最终 F.normalize 带 eps。smoke T6 全零/全等/矩形(Nq≠Ng)全有限。✓

## Findings
- **Critical/High/Medium: 无。Low: 无实质问题。**

## 结论
审查通过。OVLISetPool 4 模式置换不变 + NaN-safe + 默认字节级复现 + train/test 对称 + 参数进 optimizer。smoke 7 组全过 + 2 旧 smoke 回归。codex 审过 + GPU 空即跑 OVC-SetVLAD(netvlad)kill-switch: feature-only >52.37 则 DCVP 故事成立。

## 追加审查 — residual fix(2026-06-23,mean + zero-init residual)
**背景**: standalone 池化实验证伪(netvlad ep20 仅 14.66 < 纯 global 45.14, 随机初始化输出拖垮跨视角 cosine)。修成 **mean + 零初始化残差**(codex 角度2 的正确设计, subagent 一审漏了)。
**结论**: 审查通过(增量改动, smoke 字节级证明无损起步)
- **改动**: `pooled = mean_k(tok) + gate_res·residual(tok)`, `gate_res=nn.Parameter(zeros(1))`。初始 gate_res==0 → 输出字节级 == mean_k(tok)(=52.37 路径)。`--ovli_setpool_residual` default 1, =0 退回 standalone 对照。
- **逐行验证**: L289 self.residual; L338 gate_res=zeros(1); L375-381 forward(not residual→standalone; else m + gate_res·residual)。py_compile OK。
- **无损起步**: smoke R1 |forward-mean|=0.00e+00, |aggregate-F.normalize(mean)|=0.00e+00(标量零门, 非 out 层零, 与各分支内部 init 无关)。R3 full loss at init==mean 0.00e+00。
- **gate_res 进 optimizer**: OVLISetPool 子模块参数 → list(ovli.parameters()) 收进, L1031 assert 覆盖。
- **置换不变/NaN-safe/standalone fallback**: smoke R5/R7/R8 全过。
- 已双审 setpool 上的小增量(零初始化标量门), 风险低。codex 审 residual 进行中(pid 13397)。
- ★ residual netvlad 已在 lab-3090 跑(/tmp/cvpb_setvlad_residual.log, 从 52.37 无损起步): 残差能否 >52.37 = OVC-SetVLAD 成立判据。
