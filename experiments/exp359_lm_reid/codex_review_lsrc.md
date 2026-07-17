# Codex Review — exp359-LSRC

**Verdict**: needs-attention（修 3 项后放行对称 go/no-go；asym 主实验跑前重审）
**Date**: 2026-06-26 07:50
**Review round**: 1
**审查对象**: `cvpb_lm_reid_train.py` LSRC loss（loss_mode=='lsrc'）+ args

## Findings（codex --search exec, xhigh, 156k tokens）

- **Critical**：代码默认 `lam_marg=1.0 / lam_cons=0.2`，`loss_mode==lsrc` 只追加 LSRC，不显式传 `--lam_marg 0 --lam_cons 0` 就混入已证伪的 L_marg/L_cons，污染 LSRC 结论。
- **High**：train/test marginalization 不对齐。训练 `sim4` 是双边 M×M LR-lattice bag-to-bag；test 是 query-LR-K变体 × gallery-HR-单图。对称训练给正样本 gallery-side oracle、给负样本 test 不存在的 gallery-side worst-pair 约束。建议非对称 `LR-query set → gallery single`，或至少加该对照。
- **Medium-1**：self-pair/对角 mask 主体正确（pos.fill_diagonal_0 + logits 对角 -1e9 + neg 排同 id）。但若 PKSampler 某 id 图不足重复采同图，非对角"同图副本"会被当正样本（低概率，非严格自配对排除）。
- **Medium-2**：LSRC 机制上确与 consistency（非拉均值）/ L_marg（非分类头 posterior）有本质区别，但默认混旧项导致无法隔离证明区别。
- **Low-1**：coverage 项只最大化正样本 top-M cos，没让 top-M"赢过负样本"，更像相似度奖励非完整 ranking coverage；且 `cov_m=0` 时 `topk(...,0)` NaN，需校验。
- **Low-2**：logsumexp 没减 log(M*M)，严格说非 logmeanexp，但 batch 内 M 固定、进 softmax 常数抵消，非 bug。

## Novelty
没见"LR-ReID 把 sampling lattice 明确当 hidden variable 做 lattice decision marginalization"的直接先例。set contrastive/logsumexp bag score 本身不新（SupCon/MIL-NCE/set-ReID），novelty 只能落在 **sampling-lattice hidden variable + retrieval decision marginalization** 问题定义。不能声称发现 aliasing 本身（anti-aliasing 强先例）。

## 处理（主 agent）

1. **Critical** — 已核：正在跑的对称双跑**本就传了** `--lam_marg 0 --lam_cons 0`，log `L=id`（marg/cons 只诊断打印不进总 loss）证实**当前实验干净，不 kill**。另**已修代码默认值坑**：`loss_mode==lsrc` 自动 force lam_marg/cons=0（防以后忘传）。
2. **High** — 已加 `--lsrc_asym`（query M 变体 set × gallery slot-0 单图，对齐 test）。**策略**：对称版作便宜 go/no-go（acc 0.999 健康，~50min 跑完 eval）；对称 eval 涨 → **重跑 codex 审 asym diff → approve → 跑 asym 作主结果**；对称 eval 平 → 训练端这条基本凉，转 BLC / codex 找更多。
3. **Low** — 已加 `assert cov_m>=1`；logsumexp 常数无害（确认）。
4. **Medium** — mask 正确（确认）；隔离问题随 Critical force 解决。

## 结论
codex needs-attention 已逐条处理；对称 go/no-go 放行（实验本就隔离干净），asym 主实验跑前重审拿 approve。**这次审查暴露了代码默认值隐患 + 机制对齐问题——即便当前实验侥幸干净，价值在防未来踩坑。**
