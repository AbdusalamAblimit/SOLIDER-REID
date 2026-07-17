# Codex Review — OVLI

**Verdict**: approve
**Date**: 2026-06-22
**Review round**: 1(OVLI 首审）

## Findings
- **Critical**: none。**High**: none。
- **Medium**: RandomIdentitySampler 不强制 opposite-view composition(cargo_dataset.py:226), OVLI loss 正确跳过无 opp-view 正/负候选的 anchor → 建议加 `valid_anchor_frac/pos_per_valid/neg_per_valid` 日志, 防低 OVLI loss 被误读为"无有效 anchor"。注: cross-view 诊断已知 ~88% anchor 有 opp-view 正样本 → OVLI 非退化, 此为监控增强非 bug。
- **Low**: CLI 超参未校验(--ovli_tau 0 / 负 / alpha∉[0,1] 会崩或语义无效, 建议加 assert); pos/neg diagnostics 算全 pair 非仅 valid anchor(无害于 loss)。

## Confirmed Checks
- ★proj 新参数进 optimizer: `list(model.parameters())+list(ovli.parameters())` + assert 自检 `ovli.proj.parameters()` 在 param_groups(afd_train.py:559)✓
- AMP/数值: cached fmap 可能 fp16, 但 OVLI token proj/loss 在 `autocast(enabled=False)` + 显式 `.float()` 真 fp32(afd_train.py:619)✓
- MaxSim/logsumexp NaN-safe: 双向 mean-max 对称, 无效 logits 用有限 `-1e4` floor, valid 行 logsumexp 后选(afd_train.py:270/322)✓
- eval: 默认 `run_cross_view_eval` global-only, `--ovli_rerank` opt-in 额外报 global+MaxSim(afd_train.py:687)✓
- `--ovli` off 不构造 head/hook, optimizer 只 model params; `--ovp/--ovli` 互斥(afd_train.py:503)✓
- token path shape/device safe(layer4 2048ch → adaptive-pool grid → Conv2d 2048→dim → (B,K,D) L2-norm)✓

## Novelty
**未找到 training-time opposite-view supervised-contrastive ColBERT-style symmetric MaxSim token-set late-interaction 在 aerial-ground 或 person ReID 的 exact prior。** Frame narrowly: 主张是 "training-time opposite-view partial token-set late interaction", 不是 "MaxSim"(ColBERT 通用)或 "prototype alignment"(CMPC/OVP)。切开: ColBERT(IR late-interaction)/CVFT(geo-localization OT)/DTST(AGPReID token selection 非 retrieval loss)/CM-EMD(VI-ReID OT)/CMPC(prototype contrast=OVP 撞的, OVLI 无 memory/prototype/EMA)。

## 结论
codex 审查通过(verdict=approve)。engineering 无 Critical/High, proj 进 optimizer 关键点过, AMP/数值/train-test 对称/baseline 复现全 confirmed。novelty defensible(无 exact prior, narrow framing)。可启动 kill-switch #2′ 训练。
