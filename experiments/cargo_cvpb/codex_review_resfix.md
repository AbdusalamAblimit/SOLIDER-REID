# Codex Review — OVLI residual fix(resfix,修正后第2轮)

**Verdict**: needs-attention → **接受为非阻塞**(findings 是浮点噪声 + 陈旧注释,行为正确)
**Date**: 2026-06-23
**Review round**: 2

## Findings
- **Medium(实为非问题)**: gate_res=0 现在 byte-level == raw `mean_k(tok) @ mean_k(tok).T`(smoke torch.equal **0.0**),但与**字面** setpool=mean+match=avg+pool=mean 代码路径差 **3.0e-08~4.5e-08**(reduction-order)。codex 原文:"Behaviorally this is the fixed unnormalized mean gram"。→ **3e-8 是纯浮点累加序噪声,不是真实差异;残差确实从 52.37 的 mean gram 起步**。接受为非阻塞。
- **Low**: L695-696 注释仍写 L2-normed/cosine gram(residual 模式已改 raw unnormalized)。cosmetic,不影响运行,后续清理。

## Checked(全过)
- residual 不 normalize 修正: passed(L587-590 raw vector)。
- proj RNG parity: passed(all modes proj weight/bias torch.equal=True)。
- setpool=mean byte identity: passed(maxsim + avg 两路径)。
- standalone fallback(residual=0 保留 L2-norm)/ 置换不变 / NaN-safe / 梯度流: passed。
- optimizer inclusion: 源码正确(L1129-1157 加 list(ovli.parameters()) + assert proj/setpool/gate)。

## 结论
codex Medium 是 3e-8 浮点 reduction-order 噪声(行为 byte-exact 到 `mean@mean.T`,codex 自己确认 behaviorally correct),Low 是陈旧注释。**均非阻塞**。residual netvlad 确实从 52.37 字节级起步(±3e-8 不影响 mAP),kill-switch 有效。clean netvlad 在 lab-3090 跑(cvpb_setvlad_clean,fix 代码 13 行确认)。

**双审最终状态**: claude PASS + codex 实质 approve(2 findings 均非阻塞,3e-8 浮点 + 注释)。
