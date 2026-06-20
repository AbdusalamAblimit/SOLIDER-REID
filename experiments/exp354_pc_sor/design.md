# 实验 exp354: PC-SOR (Pose-Conditioned Semantic Ownership Routing)

## 动机(20-codex 调研结论)
今晚所有失败的根子: 我一直做 **pose-as-ID**(pose 帮学 ID), 而 CLIP 也学 ID → 冗余/竞争。
20-codex 挖出的真正正交角度 = **pose-as-归属**: pose+CLIP文本 判断每个 patch "是谁的"——**目标人 / 另一个人 / 遮挡物 / 背景**。这是 CLIP 全局 CLS 在多人遮挡下做不到的(它把遮挡物/第二人吸进特征), LGPA 也做不到("像腿"≠"是别人的腿")。**目标归属 ⊥ ID 判别 → 结构上不冗余。**

## 核心机制
1. **用 CLIP 自己的 ViT 图像编码器**(clip_model.visual, 现被丢弃 → 保留)。
2. **归属路由**: pose 热图(目标人区域)+ CLIP 文本锚点 {head/torso/arm/leg/shoes, "another person", "occluding object/bag", "background"} → 每个 patch token 的归属分布(softmax: 目标部位/别人/遮挡/bg)。
3. **注入 ViT 最后 2 层的 attention bias**: `bias(CLS→token_i) = η·log(sg(A_target_i)) − η·log(sg(A_other_i + A_occ_i))` → CLS 只聚合目标 token, 推开别人/遮挡。
4. **router 由 pose 监督(非 ID)+ stop-grad 在 ID 路径** → ID 梯度改不动它(避开 A/C 吸收)。
5. 描述子 = 路由后的 CLS;ID 对齐用 CLIP-ReID prompt(纯 ID)。

## 为什么避开今晚所有坑
- 不被吸收: router 非 ID 监督 + stop-grad(A/C 被吸收正因新通路在 ID 梯度路径上)。
- 不冗余: 目标归属(哪个 patch 是目标)⊥ ID(CLIP/LGPA 都做不到多人消歧)。
- 非禁忌 PSG/visibility: 是**steer ViT 内在的 CLS 聚合**(ViT 本来就 CLS-aggregate), 不是外挂一个 gate 在描述子上。
- 真"深度进 CLIP 语义": 在 CLIP 视觉塔内部, pose 决定哪些 token 影响 CLIP 语义。

## ★ 廉价 kill-switch(全建之前必做)
**先冻结验证: pose+CLIP文本 的归属图 到底能不能分出 目标/遮挡/别人?**
- 取若干 Occluded-Duke 图, 冻结 CLIP ViT, 算每 patch 对文本锚点的相似度 + pose 先验 → 归属图。
- 可视化 + 量化: 遮挡区/背景的"目标"概率是否显著低于人体区? 多人图里第二人是否被标"another person"?
- 若归属图是垃圾(分不出)→ 直接 kill, 不浪费全建。若有信号 → 全建训练。

## 损失
`L = L_clipreid(纯ID prompt) + L_ownership(pose派生归属标签, 非ID) + L_occ_margin(遮挡token推离) + L_cf(反事实遮挡一致)`

## 对照/消融
- baseline: CLIP ViT + CLIP-ReID prompt(无归属路由)
- 关键消融: 去 stop-grad(确认会被吸收回 baseline)、router-as-pooling(复现失败的 A/D)、text-anchor-only(无 pose)、pose-only(无 text)
- 先例: SAGA-ReID(CLIP CLS 是遮挡瓶颈, 文本锚点修)、KPR(关键点消歧多人)、DROP(ReID 与定位解耦)

## 架构影响(需用户确认)
引入 CLIP ViT-L 当图像分支(~300M 冻结 + 路由), 是比"在 SOLIDER 上加模块"更大的改动。但这正是"pose 深度进 CLIP 语义"必须的——在 CLIP 自己的视觉塔里。

## 状态
设计完成, 等用户点头 → 先做 kill-switch 验证 → 有信号则全建+双审查+训练。
