# Codex Review — ACVP(--acvp,歧义负样本软化)

**Verdict**: approve(第2轮修正后)
**Date**: 2026-06-23
**Review round**: 2

## 第1轮 findings(已全修 + smoke 验证)
- **Critical**: --acvp 第一个 batch UnboundLocalError(`bs` 用在 `bs=imgs.size(0)` 赋值前)→ 修: bs 移到 ACVP stats 块前。smoke A11(真跑一个 ACVP-on mock step)验证不崩。
- **Medium**: kill-switch stats 按 batch size 加权(文档说 #softenable-neg)+ 冷启动 (0,1) 被计入掩盖 → 修: `acvp_neg_bias` 返回 `n_soft=ok.sum()`, 主循环按 n_soft 加权 + 跳过 n_soft==0。
- **Medium**: 缺超参校验 → 修: parse 后 `assert wmin∈(0,1]/eta>0/gamma>=0`(--acvp 时)。
- **Low**: --acvp 配 --ovli_allview 语义冲突(allview 含同视角负样本)→ 修: 禁止(ap.error)。

## Checked(approve, 全过)
- --acvp **off 字节级复现 OVLI**(torch.equal A1/A2)。
- proto **detach 无梯度泄漏**(@torch.no_grad + .detach, A6)。
- acvp_mem buffer-only **不进 optimizer**(startup assert)。
- bias 只加 cand_logits(负样本分母), **正样本不动**。
- 未初始化原型 **w=1**(冷启动安全)。
- AMP fp32 + train/test 对称(ACVP 纯训练期)。

## ★ Novelty 提醒(codex 第1轮, 写 paper 必遵)
ACVP 邻近 false-negative debiasing(DCL/FNC)+ CMPC(跨模态原型抗假负/偏移正样本)。**不要 claim 宽泛的"prototype-based negative recalibration"为新, 只 claim 窄的"detached opposite-view ambiguity sensor inside OVLI contrastive"**。

## 结论
codex 第2轮 verdict=**approve**。Critical + 2 Medium + Low 全修, smoke A1-A11 全过。ACVP 双审完成(claude PASS + codex approve)。clean netvlad 完即 daemon 自动跑(kill-switch: ep20/30 mean mAP 不低于 OVLI 0.3 / G→A +0.5 / relaxed_neg_frac<30% / mean_w>0.75)。
