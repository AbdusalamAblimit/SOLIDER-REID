# 20-Codex 调研: 用 pose 改进 CLIP — 汇总 (2026-06-20)

## 元发现(重要)
- **6/20 撞同一个"正交残差相减/L_orth"死idea**(01/07/09/13/19 + 半个00)→ 不是独立背书, 是 mode-collapse。我们 exp333(β-feature null)+ de-occluded purify 已证负 → 排除整族。
- 可见性 channel-mask(08/11)= PSG 族(禁忌#1, 已 plateau)→ 排除。
- 测试时部位距离 additive(03/12)= 禁止的 retrieval-side scorer → 排除。

## ★ 突破方向: pose+CLIP文本决定 token "归属", 在 CLIP ViT 内部
**为什么这族避开了今晚所有坑:**
- **不冗余**: pose 回答"哪个 patch 是**目标人** vs 别人/遮挡物"——CLIP 全局 CLS 在多人遮挡下做不到(把遮挡物/第二人吸进去), LGPA 也做不到("像腿"但不知"是别人的腿")。**目标归属与 ID 判别正交。**
- **不被吸收**: router 由 pose 监督(非 ID)+ stop-grad 在 ID 路径 → ID 梯度改不动它。
- **真"深度进 CLIP 语义"**: 在 CLIP 自己的 ViT 视觉塔里, pose 决定哪些 token 影响 CLS。

## Top 6 排序
1. **PC-SOR (file15)**: pose+CLIP文本给每patch分配归属(目标部位/别人/遮挡物/bg), 当attention bias注入CLIP ViT最后2层, CLS只聚合目标token。router非ID监督+stop-grad。先例: SAGA-ReID(CLIP CLS是遮挡瓶颈,文本锚点修)、KPR(关键点消歧多人)、DROP(ReID与定位解耦)。
2. **PSR-CLIP (file02)**: 同族, token路由做遮挡抑制。与#1合并成一个实验(共享attention-bias hook)。
3. **PC-MSC (file16)**: pose mask 可见部位→小decoder从可见证据重建被mask部位的CLIP语义token(对齐frozen CLIP clean teacher)。**被删的token吸不走**(从输入删除)。先例: PersonMAE/PersonViT, MVP/RILS(CLIP-token MIM)。有kill-switch(random-mask vs pose-mask)。
4. **KCD-CLIP (file17)**: pose生成动态卷积核调制CLIP token + role头(目标/遮挡/别人/bg) + **pose-shuffle负样本**(打乱关键点role必须崩→强制真用pose)。先例: CondConv/DynamicConv。便宜kill-switch。
5. **PGPD (file14)**: 冻结CLIP-ReID prompt bank, pose选更完整的同ID teacher, 蒸馏ID-prompt硬负的暗分布到遮挡student + GRL pose对抗去姿态。**纯训练端,测试无pose,无架构改**。先例: PCL-CLIP, PGFL-KD, PromptSRC。
6. **PCSRA (file06)**: pose body-slot cross-attend CLIP patch, 跨同ID视图补全缺失slot, 残差用delta-over-baseline遮挡loss(遮挡特征必须比baseline更接近clean teacher)。先例: TokenFusion, PFD。风险: 仍加残差到global。

## 推荐
**首推 PC-SOR/PSR-CLIP(#1/#2合一)**: 同时满足 用户要的"深度进CLIP语义" + 结构避吸收(非ID router) + 避冗余(目标归属正交ID) + 有先例。
备选独立家族: PC-MSC(完成式, 不同机制)、PGPD(纯训练端最省)。
