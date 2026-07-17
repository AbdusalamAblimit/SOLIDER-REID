# Codex 新颖性审查 — AIRL(2026-06-24,联网 gpt-5.5 xhigh, 132k tokens)

## 四点评分

| 点 | 新颖性 | 最接近先例 | 区分 |
|---|---:|---|---|
| ① observation-limited identity recoverability | **4/5** | AG-ReID.v2 提 viewpoints/poses/resolutions;VDT/SeCap/GSAlign/ViSA 走 view discrepancy/decoupling/alignment | AIRL 不写"对齐视角",而是"航拍像素预算下哪些身份线索可恢复"——无同名/同机制先例 |
| ② 非对称降质一致性 | **3/5** | RAIN(HR→LR resolution-invariant)、DI-REID(self-degraded degradation-invariant)、MRJL(HR/LR 双表征) | 区别:只降 ground 不修 aerial + clean 约束 degraded。**先例多,别单独包装成"全新 DI-ReID"** |
| ③ 梯度隔离单模型双 head | **4/5** | DI-REID(degradation-invariant/sensitive 双特征)、MRJL(双分支)、VDT(identity/view token 解耦) | 关键不是双分支,是 clean trunk 与 recover 末段**梯度隔离**(robust 目标不污染 clean 证据)。绑定 ④ 讲 |
| ④ **检索方向特化双 head** | **★5/5** | query-adaptive late fusion(按 query 估特征有效性);AG-ReID.v2/VDT 分 A→G/G→A 报告但同 embedding/同融合 | **"clean head 强 A→G、robust head 强 G→A + 方向感知融合"无明确先例——最值得当主贡献** |

## B 类 headline(codex 建议)
- 主:**Observation-Limited Identity Recoverability for Aerial-Ground Person Re-ID**
- 副:**Directional Evidence Specialization via Asymmetric Degradation Learning**
- 主张排序:① view alignment → aerial pixel-budget recoverability;② 非对称降质 + 梯度隔离双 head 生成两类证据;③ **核心发现:两 head 非简单互补,而是按检索方向特化,direction-aware fusion +2.4~3.8**。
- ⚠️ codex 警告:别把 ② 单独包装成新 DI-ReID(先例太多)。真正能打的是 **directional evidence specialization**。

## 我的解读(诚实)
- 之前对用户说"无惊艳单点"——codex 联网查完 ④ 是 5/5 无先例,**比我估的好**。
- 但 ④ 是**经验发现**(direction-specialization)非深机制,direction-aware fusion 本身简单。撑 B 类靠"新发现 + 框架 + 两数据集 + 干净消融"。
- **★关键验证:AG-ReID.v2 必须复现方向特化**(CARGO 134 query 噪声大,2356/1811 query 的 AG-ReID.v2 能证明 clean→A→G、robust→G→A 不是偶然)。这是 point 4 成立与否的命门,现在正在跑。
- 撞车清单(写 related-work 切开):AG-ReID.v2 / VDT / ViSA / RAIN / DI-REID / MRJL / query-adaptive late fusion。
