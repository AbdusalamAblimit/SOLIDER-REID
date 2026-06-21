# LGPA "语义无关"洞察 → 能否做 B 类论文:8 路 codex 讨论综合 (2026-06-21)

## 总账
8/8 codex **没有一个**认为"random≈CLIP / 语义无用"本身能撑 B 类。一致: 它是**很强的因果消融 / negative finding, 不是方法**。要成论文, 必须在其上长出一个真新机制并实证打赢。

## 1. 机制共识
LGPA 增益与语义无关, 本质三件事: (a) pose_bias 在 softmax 前给强空间先验(锚到人体、抹背景/遮挡/邻人); (b) per-part ID/triplet 监督; (c) un-detach 梯度回流重塑 backbone。query 退化成纯 slot/seed(= Set Transformer PMA seed, arxiv 1810.00825)。**等价于 soft pose-weighted GAP / masked average pooling。**
★核心张力(file01): **因果消融做得越干净, "新方法"越危险**——QK=0 仍近最优 ⇒ 就是 soft pose pooling。

## 2. 新颖性裁决: 不是真新, 概念空间饱和
最危险近邻: **PAFormer(2408.05918, 最接近——pose 监督 cross-attn)**、PFD(2112.02466)、PVPM(2004.00230)、BPBreID(2211.03679)、KPR(2407.18112)、ProFD(2409.20081)。
"random≈CLIP 因果对照"是**新的**(无 ReID 论文做过此反事实), 但是 negative finding 非方法。

## 3. 与作者 PSG 区分
可立(PSG=backbone 门控; 新=post-backbone query-invariant part 分解), 但**不能再 claim LGPA**(已在 PRCV 系统公开)。致命前提: 新机制必须**独立**贡献增益, 不能寄生 PSG+GCN+OA-SD+PLBOA 系统。

## 4. ★ 三个真新机制提案(排名)
**🥇 #1 Query-Invariance 训练目标**: 两组无语义 query bank → 显式 loss 强制同 pose slot 在不同 query 下输出一致(KL on attn + cos on output), pose-shuffle 反向 kill-switch。把"query 语义是 nuisance"从观察升级成训练目标(无先例)。✅ 不踩禁区。风险: 必须涨, 持平=仍是消融。
**🥈 #2 SFPER freed-capacity appearance probes**: pose 只管 support, 释放的 query 容量 → M 个共享 appearance probe 在 pose support 内学局部证据。probe 部分新+安全; ⚠️"reliability 路由"≈visibility(死方向)必须砍。
**🥉 #3 Gradient-valve causal slot**: 固定 random slot + KL 对齐 + 梯度阀门 γ 扫描。最稳但最像消融。
**建议: 主推 #1 + #2 的 probes 增强, 彻底剥掉 reliability/visibility。**

## 5. B 类裁决: CONDITIONAL-GO, 高风险
单靠洞察=NO(8/8)。GO 需: 升级成"诊断原则+新训练机制"(#1); 去 CLIP/language 命名; 硬证据(多seed+跨数据集+pose/query干预+遮挡分组); 切开 8 个近邻。
最强路径: **rethinking/analysis-driven 论文**, 标题如 **"Pose, Not Prompt: Rethinking Semantic Part Queries for Occluded ReID"**。

## 6. Red-team 三大被拒风险
1. "就是 pose-pooling+part loss 换随机 seed"(概念饱和)→ 必须加真机制(#1)+逐项差异表。
2. 自我抄袭 vs PSG → PSG 只作 baseline, 绝不复述主贡献。
3. negative finding 太局部 + kill-switch 风险: **若 pose-shuffle 不掉点, 整个"pose causal"故事塌**。→ pose_bias λ 扫 + pose 跨部位/跨图 shuffle 双 kill-switch + 多seed跨数据集。

## 7. 最小实验集
因果矩阵: Query{CLIP/random/learnable/zero} × pose_bias λ{0,.25,.5,1,2} × supervision{on/off} × D1(无QK纯pose-pool) × D2(shuffled pose-map) × gradient-valve γ。
3-seed × 4 headline。量化可视化(KL(A_clip‖A_random)/PoseMass/PartPurity)。3层 SOTA。跨数据集(OccDuke+OccReID/Partial+Market sanity)。
**硬门槛: 最终方法 ≥ un-detach LGPA-no-CLIP +0.8~1.0 mAP(同 backbone/recipe)才闭环。**

## 一句话
**A 级的机制拆解证据, B-减的方法新意。** 唯一翻盘=#1 query-invariance + #2 probes(剥 visibility), 过 +0.8~1.0 硬门槛。打不过就当 PRCV 里强分析 section, 别单投。

## ★ kill-switch 地基结果 (2026-06-22): 正面方法死, 诊断材料活
exp353 60.5(真)/ exp357 59.8(乱图-pose -0.7)/ exp358 60.2(乱部位身份 -0.3)。
**pose 正确性+0.7, 部位身份~0 → LGPA 价值=部位池化结构+监督, 非pose具体内容。**
- 正面"Pose, Not Prompt"方法论文: **死**(#1 query-invariance 地基不稳, pose 价值太弱)。
- 诊断研究材料: **强**(干净拆解"pose-guided part ReID 凭什么涨")→ 并入 B 类诊断论文(《What Helps Occluded ReID? A Controlled Study》)的"pose 引导"章节。
