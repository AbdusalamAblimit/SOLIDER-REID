# Claude Broad Review — ACVP(--acvp,歧义负样本软化)

**审查对象**: afd_train.py 的 acvp_neg_bias(L732)+ loss(L827 off/L871 acvp 分支)+ acvp_mem(L1288)+ 优化器自检(L1349)+ argparse(L1170)+ warmup(L1377)+ loss 调用(L1431);smoke_acvp.py
**日期**: 2026-06-23
**结论**: 审查通过(无 Critical/High/Medium)

## 审查范围(全范围)
a. 设计合理性(ACVP = codex 角度6, detached 歧义传感器软化负样本, 非 prototype 对齐)
b. acvp_neg_bias 逐行(detached + 数值安全)
c. --acvp off 字节级复现
d. detach 无梯度泄漏
e. acvp_mem 不进 optimizer
f. AMP fp32 + train/test 对称

## 逐项
### 1. 设计合理性(非小调参)
ACVP 把 OVP 原型从"对齐目标"改成 **detached 歧义传感器**, 只软化 OVLI 对比的不可靠负样本: `δ_ij=cos(z_i,P[y_j,vj])-cos(z_i,P[y_i,vj])`, `w_ij=clamp(1-γ·sigmoid((δ-margin)/η),wmin,1)`, `neg_logit += log(w_ij)`。**无 prototype-positive InfoNCE → 避开 OVP/CMPC/PDPA**。codex 角度6 设计, 复活"prototype 信息有用"(OVP+MaxSim 52.76)但换用法。

### 2. acvp_neg_bias 逐行(L732-799)
- 全程 detached(proto 传入 `.detach()`, 整函数 no-grad 概念 → bias 是常量, 不回传梯度到原型/feature)。✓
- δ=cos_neg-cos_self(L779): 负样本 j 在对面视角离 i 多近=歧义度。✓
- w=clamp(1-γ·sigmoid((δ-margin)/η),wmin,1)(L781): 高歧义→低 w→软化。✓
- validity(L785-789): 只软化双原型都初始化的负样本对; 未初始化 w=1(不软化, 冷启动安全)。✓
- bias=log(w), w≥wmin>0 → finite 无 -inf/NaN(L790)。✓
- 只加 cand_logits(负样本分母), 正样本/分子不动 → **只软化负样本**。✓
- kill-switch stats frac(w<0.95 占比)/mean_w(L792-798)。✓

### 3. --acvp off 字节级复现
acvp_proto is None → loss 走原路径(L827)→ 不构造 bank, 不加 bias。smoke A1 torch.equal(loss/pos/neg)。旧 smoke(allview/residual)回归不变。✓

### 4. detach 无梯度泄漏
proto=`acvp_mem.bank.detach()`(L1431); acvp_neg_bias 用 detached proto。smoke A6: prototype.grad is None, proj.weight/layer4 map 仍有梯度。✓

### 5. acvp_mem 不进 optimizer
acvp_mem=OVPMemory(EMA buffer), read-only for ACVP, L1349 自检 acvp_buf_in_opt(确认 bank 不在 opt_ids)。无新可学习参数。✓

### 6. AMP/NaN + train/test 对称
acvp_neg_bias 在 autocast(enabled=False) fp32; cos/sigmoid/clamp/log fp32; w≥wmin>0。smoke A5 全零原型 gamma=1 → w∈[0.5,1] finite。ACVP 纯训练期 loss calibration, eval 默认不变(global + --ovli_rerank 不受影响)。✓

## Findings
- **Critical/High/Medium: 无。Low: 无实质问题。**

## 结论
审查通过。ACVP = detached 歧义传感器软化负样本, off 字节级复现 + detach 无泄漏 + 不进 optimizer + NaN-safe + train/test 对称 + kill-switch 日志全。smoke A1-A10 全过 + 2 旧 smoke 回归。codex 审 + GPU 空即跑 kill-switch(ep20/30 轨迹: mean mAP 不低于 OVLI 0.3 / G→A +0.5 / frac<30% / mean_w>0.75)。
