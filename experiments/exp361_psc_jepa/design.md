# exp361 — PSC-JEPA：Pose-defined Support-Complete JEPA pretraining（范式转向核心 build）

## 选定经过（2026-06-26）

范式转向（用户授权换量级）→ 3 路 gap → Intruder(C-#2) 选中 → **Intruder DEAD**（exp360 阶段0.5 因果证伪：donor-ID 可读但压它不救排序，H2 #false 控制 + rank-r projection 双证）→ codex 裁决转 **B PSC-JEPA(6.5)**。这是真正的 continued-pretrain 换量级，不是 frozen probe。

## 核心 idea

给一张**不完整人体图**（pose-defined 部分遮挡），让 backbone 在 **latent body-part token 空间预测"完整身份 support"**——target 来自 EMA full-view teacher + 高置信 pseudo same-ID support bank。**不补像素、不补语义比例，而是补"身份证据"**。JEPA 路线：预测 embedding target 而非像素，避开 ReID 中重建背景/遮挡物的污染。

贴项目 exp109 oracle 证据：support-complete teacher 有真实 headroom（oracle 61.88/73.26 → 70.40/81.36），PSC-JEPA 把这个 headroom 尝试**蒸进预训练**（不像 exp109/FGEU 是测试端不可得的 oracle）。

## Novelty 边界（codex 标，诚实）

**已被占**：SOLIDER(human SSL+语义可控) / PersonMAE(occlusion masking + pixel+semantic 重建, 报 Occ-Duke) / HAP(part-guided MIM) / PersonViT 2024(MIM+contrastive) / SAIP 2025(cross-scale)。
**没吃掉的窄缝**：① 补身份 support 非像素 ② support bank/pseudo cross-view teacher 非 single-image MAE ③ pretraining 学"缺部位时如何形成可检索身份证据" ④ JEPA latent prediction。
→ **B 类空间只在"latent support completion 讲清 + 赢过 plain continued-pretrain/random-mask/PersonMAE-lite"才成立**。

## Pipeline（codex 设计）

1. **init**：SOLIDER/Swin-Tiny continued-pretrain（保持 fine-tune 兼容）；DINOv2 可选 frozen dense teacher（稳的 latent target，不全量 fine-tune）。
2. **data**：**train split only**（防 query/gallery 泄漏）Market+MSMT17+Occluded-Duke+Occluded-ReID+Occluded-PoseTrack；预缓存 pose/keypoint visibility/body-group mask + pose-defined complementary masks。
3. **pretext**：输入 partial view（保留一部分 body support）；teacher target = `T_full`(EMA full-view body-part latent) + `T_bank`(pseudo same-ID/NN support bank 的 body-part prototype)；student 输出 visible/missing/union tokens。
4. **loss**：`L_part_jepa`(missing token 预测 teacher/support, cos/L2) + `L_union`(union token ≈ full-view identity) + `L_gram`(part-token 关系矩阵对齐) + `L_visible_anchor`(可见 part 不被改坏) + `L_solider/dino_anchor`(可见区蒸原 backbone, 防遗忘) + 可选 `L_cluster_contrast`(高置信 pseudo cluster)。
5. **fine-tune**：continued-pretrain 完 → 标准 ReID fine-tune → 评估。

## 4-slot 排布（codex；对照是 novelty 生命线）

| slot | 任务 | 作用 |
|---|---|---|
| 4090 | **PSC-JEPA 主跑**（SOLIDER/Swin-Tiny multi-dataset 50-100ep）| 主结果 |
| 3090 | 去 support bank，只 same-image full teacher | 对照"是否只是 OA-SD/PCVT 换名" |
| 5060Ti-1 | random mask / PersonMAE-lite latent baseline | 对照"是否只是普通 MIM" |
| 5060Ti-2 | support bank 质量诊断 / DINOv2 frozen teacher variant / Occ-Duke smoke | 诊断 |

## Build 阶段（deep work，v0 5-8 天）

- **阶段 A 骨架**：continued-pretrain loop（SOLIDER init + EMA teacher + partial-view pretext + `L_part_jepa`+`L_union`+`L_visible_anchor`，先不加 support bank）→ smoke 跑通 + 不崩。
- **阶段 B support bank**：加 pseudo same-ID support bank（`T_bank`）+ 质量诊断（pseudo same-ID top-k 精度）+ `L_solider_anchor` 防遗忘。
- **阶段 C 对照 + 全量**：4-slot 并行主跑 + 3 对照 → fine-tune Occ-Duke/Market/MSMT → 判 novelty（主跑赢过对照 + Occ-Duke ≥+0.7）。

## Kill-switch（codex）

- continued-pretrain 后 Occluded-Duke fine-tune **≥+0.7 mAP** 或 heavy-occ 子集明显正向；
- **plain continued-pretrain / random mask / same-image teacher 不能同涨**（否则 novelty 塌，只是"更好的 MIM/continued-pretrain"）；
- pseudo support bank top-k 精度过线（否则噪声主导）。

## 风险

- pseudo support bank 噪声（第一风险）；novelty 拥挤（PersonMAE/HAP/PersonViT，靠对照证差异）；continued-pretrain 不涨（45-55% 成功率, codex）。

## 训练前纪律

continued-pretrain = backbone 训练 → **启动前 codex 审 diff**（[[pre-experiment-review-discipline]]，用户分工：审查交 codex）。

关联：`experiments/paradigm_shift/`、`experiments/exp360_intruder/`（DEAD 因果记录）、memory [[exp109-headroom-is-a-wall]]。
