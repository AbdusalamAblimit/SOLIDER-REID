# Claude Broad Review — cvpb_gallery_killswitch.py

**审查对象**: `experiments/cargo_cvpb/cvpb_gallery_killswitch.py`（零训练 gallery-组成 kill-switch, 三测试 A/B/C）
**日期**: 2026-06-25
**审查轮次**: v1 (Opus 子代理全范围逐行) → 主 agent 修复 → 复核

## 第一轮发现（5 个 blocking + should-fix）

| # | Sev | 位置 | 问题 | 状态 |
|---|-----|------|------|------|
| B1 | Critical | test_B 阈值 | size-conditioned DIR@FPIR/FPIR@TPIR 在同一样本上 calibrate+eval → 循环, 保证"赢" | ✅ 已修: CAL/EVAL 折(偶/奇 trial), 全程 out-of-sample |
| A1 | High | test_A CONTROL2 | shuffled-distractor null 从全 gallery 取(注入 core-身份向量)且 count 不匹配 → 偏向"structural" | ✅ 已修: 同 add_idx 特征 + per-row 列洗牌(毁方向保范数) + 重归一 + assert count 相等 |
| B2 | High | test_B | EVT max-of-N null 算了但没接进阈值比较 → "赢过 random"承诺未测 | ✅ 已修: threshold_eval 对 real 和 rnd 都跑, 报 NET = real − rnd |
| C2 | High | test_C 阈值 | support-calibrated false-merge 单 Zipf draw + sparse level 静默 fallback → 单点高方差 | ✅ 已修: 跨 n_zipf_seeds 平均, 报 fallback-to-global 比例 |
| E1 | High | test_C headline | 头条 Spearman 在 n=4 bins(exact-p 下限~0.08) | ✅ 已修: 头条改 per-head-ID(数百点)的 per-image Spearman; 分箱降级为 descriptive 并标注 n=4 |

## should-fix（已处理）
- B5/D: impostor 池改用 gallery-only IDs + 非 enrolled shared IDs; watchlist 上限 = len(shared)//2 防 impostor 池饥饿。
- C3: tail probe 时移除 tail 自己的 singleton → 真"unknown"(最近邻必跨身份)。
- E2: test_C 分箱 numerator/denominator 同 seed 累积(原 denominator 用单 draw)。
- E3: test_A CONTROL1 加 partial_spearman(结构 beyond count 增量); no-new-false 子集用严格 |d#false|<1e-6。
- A 池太小到不了 10x: core_frac=0.2 + core_cap=8/ID(相机感知保留跨相机正样本) → Market 可达 12.5x。
- CONTROL2 死代码 shuf_feat 行删除。

## 复核（v1 修复后 smoke 验证, market）
- 三测试端到端无 runtime error。
- Test A: 1x→10x mAP −4.5(frozen); CONTROL1 Spearman(-dAP,d#false)=+0.73 但"#false 完全不变"的 579 query 仍 −1.35 mAP(结构成分存活); CONTROL2 real −4.45 vs 列洗牌 −0.00 → 结构性吸附实锤(非纯 count)。
- Test B: CAL/EVAL 折跑通, Market 上 genuine~0.96/impostor~0.5 近饱和(强 backbone 上拒识接近完美), size-cond NET 增益 ≈0。
- Test C: per-head-ID per-image Spearman ≈0(Market 头不超额吸附); fallback 比例报出。

## 确认未动的正确部分
- per_query_ap_cmc 的 AP/CMC/junk 逻辑与 utils/metrics.py 一致; argsort 距离方向正确; #false-in-topk 在 junk 移除后计算(对上 §7.6 定义)。
- 零训练保证成立(仅 model.eval()+no_grad 提特征, 其余全 numpy)。
- L2/junk 处理、Test C per-IMAGE 归一(唯一真正击败"更多彩票"陷阱的对照)。

## 结论
全部 5 个 blocking 已修, should-fix 已处理, smoke 复核三测试无错且对照逻辑正确触发。代码满足零训练 + 每个 per-query 相关都控了 trivial 代理(#false-in-topk / max-of-N / per-image)。**审查通过**, 可进行全量双数据集运行。
