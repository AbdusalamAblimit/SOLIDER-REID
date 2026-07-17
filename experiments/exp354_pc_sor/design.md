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

## ★ Kill-switch 结果 (2026-06-20): FAILED (简单形式)
冻结 CLIP ViT, 6 张 occluded_duke query 图, raw patch token 对文本锚点的归属:
- 单人裁剪里"OTHER-PERSON"锚点却抢 90-159 patch(占大头)→ CLIP 分不清 目标人 vs 泛指人。
- "OCCLUDER"锚点抓 ~0 → 遮挡物定位失败。"torso"=0。
**结论: raw CLIP patch-文本对齐太噪(CLIP 只为全局CLS-文本训, 非patch级), 归属路由核心假设在简单形式下失败。**
**救法**: MaskCLIP/CLIP-Surgery 式 dense 特征(去最后attn, 用value embedding) + pose 先验定位目标。但 occluder-via-text 仍存疑。
**判断**: kill-switch 省了多小时白建。PC-SOR 需 MaskCLIP 改造才可能, 较重且不确定。备选 PC-MSC(#3, 不依赖patch-文本归属, 靠CLIP语义token重建)、PGPD(#5, 纯训练端prompt蒸馏)更干净。

## ★ MaskCLIP 重试 (2026-06-20): 也 FAILED → PC-SOR 死
MaskCLIP value-embedding dense 特征下: 部位略好(torso 0→4-12), 但两致命问题在:
- "OTHER-PERSON"依然霸占 107-178 patch(单人裁剪!)→ CLIP 文本分不清"目标人"vs"任意人"("another person"匹配目标自己)。
- "OCCLUDER"依然 ~0-8 → 遮挡物定位失败。
**死因(深层): "目标人"是 pose 定义的非 CLIP 文本能定义(文本里"另一个人"匹配任何人); CLIP 文本定位不了多样遮挡物。所以归属里 CLIP 文本部分对 目标/遮挡 消歧毫无贡献——pose 自己做目标定位=就是 LGPA/PSG。**
**合力结论: CLIP=全局语义非空间; pose=空间先验。全局层面融=冗余(今晚证); 空间层面融=CLIP非空间(归属失败,刚证)。没有层面能让 pose+CLIP 真互补涨点。**
