# Rank-Regret (Rank-Instability) Efficiency Kill-Switch — Design

> 零训练 kill-switch。**效率 / Pareto 轴的 re-frame**（不是 accuracy 隐藏变量——那 5 个 accuracy 方向已被 k-reciprocal 碾死）。novelty 已查 7/10 存活，但需证关键区分（test B 决定撞不撞 CFPER）。

## Re-frame
ReID 默认所有 query 跑 full 网络（uniform compute）。重定义：用 **cheap（早层 Swin stage GAP）vs full（最终 BNNeck embedding）的检索排名不一致度（Rank-Regret / Rank-Instability, RI）** 路由算力——RI 低（cheap 排名 ≈ full）早退在 cheap，RI 高走 full。

**关键（决定生死，区别于 CFPER 式难度路由）**：RI 是**检索结果级 / 关系级**变量（cheap 表示是否改变 gallery 排序），不是图像内部难度分。

## Cheap / Full（怎么 hook）
- 模型 `PoseBackboneModel._run_backbone_with_psg` 单次 forward 返回 `(test_feat, featmaps)`，`featmaps[0..3]` = 4 个 Swin stage 输出（已 norm）。
- 设 `POSE_TEST_FEAT='global'` → eval 分支里所有 gcn/part 计算被 `!= 'global'` 关掉，返回干净单向量。`NECK_FEAT='after'` → full = BNNeck embedding。
- **cheap = GAP(featmaps[cheap_stage])**，默认 **stage1（0-idx=1，cum 0.167，真省 83% 算力）**；也报 stage0/2/3。Swin depths(2,2,18,2) 把 ~75% FLOPs 压在 stage-2，只有 stage0/1 出口真省算力，stage2/3 省≈0。**full = test_feat（BNNeck）**。
- 单 forward 同时拿全部 stage GAP + BNNeck，零额外开销，frozen + no_grad。

## RI@K（≥2 种，按 spec）
每 query，cheap top-k vs full top-k 的排名距离：
1. **top-k overlap 不一致** `1 - |A∩B|/k`（集合级）
2. **RBO**（rank-biased overlap, p=0.9, top-weighted）—— 主 RI
3. **Kendall-tau**（union 上，缺失项给深 rank）

## 测试
- **A**：RI ~ (AP_full − AP_cheap) per query。spearman + perm-p。期望正（高 RI → full 帮助大）。
- **★B（生死, CFPER 区分）**：RI vs 静态难度代理（cheap top1-margin / entropy / feature-norm / top1-top2 gap / 邻域密度）谁更预测 AP_gap？**RI 必须 partial 控住全部静态代理后仍显著**（≥0.05 且 ≥0.4×marginal），否则 = CFPER 撞车死。加反向控制（静态代理 partial on RI，应缩小）。
- **C（可行性）**：cheap-only 量（margin/entropy/density，不看 full）能否预测 RI？单变量 spearman + rank 多元回归 R²。若估不出 → 推理时无法路由 → 无效率收益。
- **D（Pareto cascade）**：低 RI（oracle）走 cheap、高 RI 走 full，扫阈值画 compute-fraction vs mAP。**对照**：random routing / 静态 difficulty routing（最强静态代理 + cheap-margin）/ **cheap-only-estimated RI**（test C 的可部署 router）。
  - compute fraction：Swin stage 解析 FLOPs（depth × tokens × dim²，tokens 每次降采样 /4，dim ×2），cum_frac[s] = 跑完 stage s 的算力占比。routed-to-full 付 full（含 cheap stem，full 子集 cheap），cheap-exit 付 cheap_compute。avg = frac_full×1 + (1−frac_full)×cheap_compute。
  - **通过 = ~50-60% compute 拿 ≥99% full mAP 且明显 beat random + 静态 difficulty routing**。

## 生死判据
- **B**（撞不撞 CFPER）：RI partial-on-all-static 显著 → 活；≈静态 → 死（撞 CFPER）。
- **C**（有没有效率收益）：cheap-only 估得出 RI → 活；估不出 → 死（无 inference router）。
- 两数据集（Market exp260b / Occluded-Duke exp255），各一份。

## 坑
- 早层 hook：复用 `featmaps` 不用注册 hook（模型已返回）。
- cheap 维度：stage0=96/128, stage1=192/256, stage2=384/512（small/base）。早 stage 维度低、mAP 低 → headroom 大。
- FLOPs 估算：解析近似（线性投影主导），非 fvcore 实测——足够定 Pareto 相对位置；论文若要精确用 fvcore profile 单图。
- stage 选择：太早（stage0/1）cheap 太弱、RI 饱和（全高）失去区分；stage3 太强、headroom 小。默认 stage3（次强 cheap）平衡，附 stage2 对照。
- 诚实：若 RI ≈ 静态难度（B 死）或 cheap-only 估不出 RI（C 死），判死，别粉饰。
