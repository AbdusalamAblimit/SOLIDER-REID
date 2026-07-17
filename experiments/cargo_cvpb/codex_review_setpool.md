# Codex Review — OVLI SetPool(--ovli_setpool)

**Verdict**: approve
**Date**: 2026-06-23
**Review round**: 1

## Findings
- **Low**: `--ovli_vlad_clusters`/`--ovli_attn_heads`/`--ovli_so_rank` 缺正数校验(默认值安全, headline netvlad 默认 OK; 传 0 可能 cryptic error。建议加 `assert >=1`)。
- **Low**: `aggregate_tokens()` fp32 依赖调用点而非函数内自保证(当前训练在 `autocast(enabled=False)`、eval tokens 也 fp32, 主路径 OK; 未来复用建议函数内 `tok.float()`)。

## Checked(全过)
- **setpool=mean**: setpool_mod=None, 不构造新模块, 不耗 RNG, sym_maxsim_matrix 走原 MaxSim 分支 → 结构支持字节级复现。
- **置换不变**: netvlad K-residual-sum / attn K-softmax-sum / gated K-gated-clamped-sum / secondorder z^Tz/K, 都不依赖 token 顺序。
- **train/test 对称**: train sym_maxsim_matrix() 与 eval maxsim_block() 在 setpool!=mean 时共用 aggregate_tokens()。
- **optimizer**: setpool_mod 是 OVLIHead 子模块, list(ovli.parameters()) 递归收进, L1031 assert 覆盖 setpool 参数。
- **AMP/NaN**: OVLI fp32; gated denominator clamp; secondorder signed-sqrt; netvlad softmax+normalize 无明显 NaN 源。
- **OVC-SetVLAD**: 标准 learnable assignment + center residual + K-sum + cluster/global norm + linear, headline 机制主路径可跑。

## 结论
codex 审查通过(verdict=approve)。无 Critical/High/Medium。2 个 Low 不阻塞训练(默认值安全, netvlad headline 默认 clusters=8 OK)。setpool 双审完成(claude PASS + codex approve)。OVC-SetVLAD netvlad 已在 lab-3090 跑 kill-switch(>52.37?)。
