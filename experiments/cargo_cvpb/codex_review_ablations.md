# Codex Review — OVLI 消融(--ovli_match / --ovli_align)

**Verdict**: approve
**Date**: 2026-06-22
**Review round**: 1
**Tokens**: 189,246(--search exec, xhigh）

## Findings
- **Low**: `experiments/cargo_cvpb/` 未 git-track → `git diff` 返回空, codex 改读实际文件内容审。建议 stage/track 后再依赖常规 review 工具。
- **Low**: `--ovli_align ordered` 实现为 same-row 限制是准确的, 但叫 "AlignedReID-style" 要保持限定——AlignedReID 是 shortest-path 局部对齐 + test 只用 global, 非此 row-equality mask。代码注释已写 "simplified monotonic/diagonal cut", 是对的 caveat。

## Checked items(全过)
- **Runtime/shape**: `_row_mask4` (1,K,1,K) 对 train (B,K,B,K) + rerank (Nq,K,Ng,K) 在 other_dim=3/1 双向广播正确。
- **Train/test 对称**: rerank 用同 `ovli._reduce_other()` + 同 pool/topk/thresh/tau → match/align 训练与 eval 对称(afd_train.py:590)。
- **AMP fp32**: OVLI path 在 autocast 外; cached map + BN feature 投影前 cast fp32, 再 cosine/MaxSim/logsumexp。
- **默认字节级复现**: 默认 maxsim+free; `_reduce_other()` 返回 `sim.max(dim).values`; buffer 不耗 RNG, non-persistent。
- **消融隔离**: match 只改内层 token 归约; align 只改候选 token mask; 外层 pool/双向/α/τ/loss 不变。
- **ckpt 兼容**: `_row_mask4` persistent=False → 不进 state_dict → 旧 strict OVLI-head load 不缺 key。
- **NaN-safe**: ordered max finite floor -1e4; ordered avg clamp count; loss masking finite floor before logsumexp。

## Prior-art(确认 ablation 有先例, framing=inspired-by 非 novel 机制)
- AlignedReID(arXiv 1711.08184): global+local, shortest-path 局部对齐, test 只 global。
- ColBERT(2004.12832): MaxSim late-interaction。
- top-k/softmax pooling: late-interaction retrieval 已显式研究。
- HPM(1804.05275): ReID avg/max pooling 做 partial/local evidence。

## 结论
codex 审查通过(verdict=approve)。无 Critical/High/Medium。2 个 Low 均不阻塞训练(git-track 待补; AlignedReID-style 措辞注释已限定)。两个消融是 OVLI 设计选择的**对照**(非 novelty 来源), prior-art 支持论文 inspired-by/切开写法。GPU 空出即可训练。
