# exp367 Single-Support CVaR Episodic Loss（训练侧创新，2026-06-28）

## 动机

用户 goal：找训练侧创新发 CCF-B，不收手，不轻易说穷尽，审查调研交 codex，严谨，文档记好。codex 训练侧深度调研 #1（最务实）：训练时每 ID 只用单图 support 定义身份，对 worst-support 选择做 CVaR 优化。回应 exp109 根问题（single-image support incomplete）。**纯训练侧**（episodic loss，输出常规 descriptor），严格非 test-time/检索侧/范式重定义。

codex 已避所有死区（visibility/masked/CLIP-align/synthetic/topology/DG-foundation/noisy-label/long-tail），2024-26 novelty 空白：few-shot/DG 有先例，但**标准 Market/MSMT/Occluded 监督训练里"单图 support 是否足够定义身份"做成主训练目标，2024-2026 没看到直接占位**。

## 核心假设

ReID 训练用 multi-shot gallery（每 ID 多图），但模型学到的身份边界可能依赖"见过该 ID 多个 view"。部署常 single-shot（单图 support 定义新身份）。训练时**强制单图 support + CVaR worst-support 优化**，逼模型学"从任意单图恢复完整身份边界"的鲁棒表征，而非依赖 multi-view 平均。

## cheap kill-switch（零训练，cvpb_single_support_probe.py）

复用 Market 特征 cache（frozen SOLIDER exp260b）。每 gallery ID 只留 1 图：
- full-gallery：上界
- best-support：每 ID 选最好单图（同 ID query 平均 sim 最高，oracle 上界）
- random-support：每 ID 随机 1 图
- worst-support：每 ID 选最差单图（CVaR worst-case 目标针对的）

**GO**（support 选择是真训练瓶颈）：worst 比 full 掉 > 3 mAP 且 **best−worst gap > 3 mAP**（哪张 support 图很重要 = support 选择 matters，值得 CVaR 优化）。
**DEAD**：best≈worst（哪张 support 都一样，没 support 选择价值）或 single≈full（单图够）。

★诚实设计要点：单图 vs 多图必掉 mAP（少正样本）是 trivial，所以**关键判据是 best−worst gap**（同样单图，选择重不重要），不是 single<full。codex 审 probe 验这个设计是否真有意义（用户要审查交 codex）。

## 审查（codex，用户要求）

codex 审 probe（codex_review.md）：kill-switch 设计是否有意义、best/worst per-ID 选择逻辑、#false-in-topk 控制。

## 预期

- GO → 设计 Single-Support CVaR episodic loss 训练（每 ID 单图 support + worst-case 风险优化），训练侧第一 contribution，full fine-tune 前 codex 三审 diff。
- DEAD → support 选择无训练价值，转 Equivariant Routing（codex 训练侧 #2，routing 等变非 embedding 一致）。

## 训练设计（codex 调研 63517，probe GO 后）

★**novelty 真空白（codex 确认）**：2024-26 标准监督 person ReID 没有"episodic single-support training + CVaR worst-support tail optimization"直接先例（检索 single-support/worst-support/CVaR-ReID/support-selection 都没命中）。邻近但不同：CFReID(continual few-shot)/DG-episodic(domain-invariant)/ProtoNet(novel-class 优化 prototype 平均非 tail)/batch-hard(hard pair mining 非 support tail)。**claim 写窄**：不发明 episodic/CVaR，是"标准 ReID 优化单图 support 定义身份的 tail risk"。

★**训练设计（two-level CVaR，加项不替换）**：
- episode N ID × K 图，每 ID 1 support + K-1 query。
- `risk(y,s)` = 该 support 对同 ID 多 query 的 CE 失败。
- `L_cvar_y = CVaR_α(support risks)`，`L_ss_cvar = mean_y`。α=0.7/0.8。
- `L = L_id_ce + L_triplet + λ·L_ss_cvar`（λ 0.1→0.3 warmup）。
- support/query 都梯度 + 保 CE+Triplet 防 collapse。两级聚合(support risk→CVaR)非 CVaR over all pairs(避 batch-hard)。

★**避坑（避六点定律）**：不写 support completion/feature alignment/prototype compression/query-dependent selector。训练用 label 算 worst support 可以，测试不选 best/不用 query label，输出常规 descriptor。

★**cheap 验证路径**：① frozen head smoke(10-20ep, worst/random +0.8~1.0, 失败不判死) ② last-stage(20-40ep, worst+2/random+1/gap 缩≥2/full 不降>0.5, 对照普通 CE+Triplet continued FT) ③ full FT。

★**风险+对照（防退化 hard-mining）**：报 batch-hard/pair-CVaR/random episodic CE 三对照，只 support-level CVaR 独立赢才站得住。报 missing-positive/cross-cam 覆盖/false10 random std。

★CCF-B 6.5/10（last-stage/full FT 抬 worst/random + 赢 hard-mining → 7.5；只改 diagnostic 但 full-gallery 不涨 → 4 附录）。

## 状态

probe v2 GO（best-worst 12.27 不被 #false 解释，codex 两轮审）。codex 训练设计 GO（novelty 空白 6.5/10）。下一步：写 frozen head smoke 训练（cheap 第一步，复用 Market cache + projection head + episodic CVaR loss）+ codex 三审 diff（full FT 铁律 + 审查交 codex）。
