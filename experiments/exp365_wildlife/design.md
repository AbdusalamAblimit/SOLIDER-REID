# exp365 Wildlife Species-Conditioned ReID（换领域换量级，2026-06-27）

## 动机

LM-ReID 训练端真 measure 六点穷尽（塑造/对齐/分离 K 变体全死，见 design_lcrs/lrfd + memory [[lm-reid-trainside-shaping-kills-marg]]）。现有标准行人 ReID 数据 + 4 单卡训练端机制窄。**探透一层 = 自主转下一层**：codex 换量级调研第二名 = Wildlife/Animal ReID（真训练端换领域，避全死区）。

codex 深化核查（codex_wildlife_check.md）：**不是伪命题**，真价值不在"animal 数据集"，而在 **多物种少样本场景下，训练端如何把 shared animal ReID prior 分解成 species-conditioned identity evidence**。主线风险中等。

## 核心假设

动物 ReID 的难点是 **same-species hard negatives**（同物种个体极像）+ **rare/unseen species 少样本**。统一 backbone 给的是 shared-animal embedding；若能 **species-conditioned** 地分解出"该物种内部真正区分个体的证据"（per-species metric/centering），应能在 rare/unseen species 上超过 species-agnostic 的 global fine-tune。

## 候选机制（过 kill-switch 后才写）

SCREA: Species-Conditioned Residual Evidence Adapter —— shared identity core + species-conditioned residual（per-species centering/whitening 的可学版），adapter/LoRA 不碰大规模 full fine-tune。

## cheap kill-switch（第一步零训练 frozen probe，codex 建议）

1. 下 WildlifeReID-10k（214k 图 / 10k 个体；wildlife-datasets 库统一 dataframe + similarity-aware split）。
2. 抽特征：**MegaDescriptor-L**（动物 ReID SOTA）/ **MiewID-msv2** / DINOv2 / SOLIDER。
3. similarity-aware split + 构造 gallery：rare species / unseen species / 5-shot / same-species-only / all-species。
4. 指标（控 trivial 代理）：mAP/R1 + **false_in_top5/top10** + wrong-species-in-topk + same-species hard-neg error + per-species mAP 方差。

**Go 线**：best frozen 在 rare/unseen 5-shot 仍明显低，错误集中 same-species hard negatives；per-species centering/whitening **oracle >= +3 mAP**；all-species gallery 明显比 same-species-only 差（species 干扰真实存在）；MegaDescriptor/MiewID 没饱和。

**Kill 线**：MegaDescriptor/MiewID frozen 已很强，per-species normalization <+1；错误主要来自标注噪声/裁剪/单图不可辨（非 species-conditioned）；ordinary ArcFace/LoRA 已吃掉 rare/unseen 增益；species conditioning 只在 seen/common 涨、不迁移 unseen。

## 单卡可行性

可行。先缓存 embedding，零训练 probe 一张 3090/4090 足够。过线后训练只做 adapter/LoRA。下载走本地代理 → rsync 3090。

## 状态

待：① 探查 WildlifeReID-10k 下载（wildlife-datasets 库）② 装库 + 抽 MegaDescriptor/DINO/SOLIDER 特征 ③ frozen probe（per-species oracle）。过线 → 写 SCREA design；不过线 → 直接杀，不补 adapter rank/head/loss 小变体（codex 明确）。
