# Lattice-Marginalized ReID — 零训练 kill-switch 设计

> 脚本: `cvpb_lattice_killswitch.py`  机会来源: `litreview2/explore20/clean/d_8.txt` 机会1（信心 7/10）
> 机器: lab-3090-d（docker, root, py3.8.20 conda solider-reid）。frozen ckpt: market `log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth`（HR sanity mAP 94.43，与 exp260b ref 94.4 一致）。

## 重定义（被测假设）
低分辨率 ReID 失败**不是**"模糊缺细节"，而是**采样格点不确定性**：一个人在 h=16-32px 时 = 一族 alias/crop-lattice 观测（亚像素采样相位 / ±1 LR-pixel bbox 量化 / antialias kernel / 轻微 detector crop error）。模型只见过其中一个。身份匹配应对这族观测**边缘化**，而不是把某个确定性 LR 图当真值。

## 实现要点（零训练，frozen + numpy/PIL，无 backward）
- **特征**：`POSE_TEST_FEAT='global'` + `pose_dict=None` → PSG/LGPA/GCN 全部因 `scene_heatmaps is None` 跳过，得到纯 SOLIDER/Swin backbone 全局向量（BN-neck after, L2-norm）。这样 LR/phase 变换不需要对齐 heatmap。HR sanity 94.43 确认提取忠实。
- **CR-ReID 设定**：HR gallery（一次提取）+ LR query。标准"远处/小目标 query, 入库 HR"场景。
- **LR 生成**：原图 → 384×128 HR 画布（BICUBIC）→ 下采样到 (h, round(h/3)) → 上采回 384×128（degrade-then-restore-size）。
- **K=9 lattice 变体**（每 LR query，从同一 HR 图）：variant0 = 确定性 bicubic LR（= 单图 baseline）；其余轮转覆盖三轴：亚像素相位平移（±0.5 LR 像素的 HR 仿射）、±1 LR-pixel bbox crop shift、bbox expand/contract 1 LR 像素 + kernel 切换（bicubic/bilinear/lanczos/box/hamming）。
- **测量**：(A) same-image phase 特征方差（K 个变体两两 1-cos 均值）；(B) rank volatility（top1 一致率 / top10 Jaccard / 跨相位 rank-1 身份翻转数）；(C) phase-var 是否解释 LR false match；(D) K-phase mean-feat / MaxSim ensemble vs 单一 bicubic LR 的 mAP。

## ★生死对照（区分 trivial）
1. **vs 普通 TTA**：相同 K、相同融合（mean / MaxSim），但 K 个 view 是普通 test-time aug（pad+RandomCrop + hflip）of 单一 bicubic LR，**无 lattice 语义**。phase-lattice 必须明显超过普通 TTA（否则只是 TTA 换名）。MaxSim 对照尤其关键：MaxSim 取 K 中最优可能仅靠"多枪"涨点，TTA-MaxSim 同样多枪 → 差值才是 lattice 特有增益。
2. **vs #false-in-topk**（Hubness §7.6 教训）：phase-var 解释失败必须在 partial out 这个 trivial 计数后仍 >0。另加 partial out LR severity（single-LR→HR drift）。Hubness 的 M(q) 正是没控这个代理被判死。

## GO / DEAD 判据
- **GO**：h≤32 时 rank volatility 明显（idFlip>1 / top1stab<1）**且** phase-ensemble ≥+2 mAP **且** LAT-TTA 明显>0 **且** partial rho(AP-err, phase-var | #false) 明显>0。
- **DEAD**：phase-var ≈ TTA-var / ensemble ≈ 单图 / ensemble ≈ 普通 TTA / phase-var 被 #false 吃掉。

## 坑（已踩/已规避）
- LR 喂法：用 384×128 HR 画布统一 degrade，gallery 也走同一画布，避免 aspect 混杂。
- pose 必须关（pose_dict=None），否则 LR 图与缓存 heatmap 错位。
- K 选 9：覆盖三轴 + variant0 = 单图 baseline（lattice 与 TTA 共享 variant0，保证 baseline 一致）。
- 变体生成是 CPU 单线程瓶颈（每 height 3368×9×2 ≈ 60k 张 PIL 变换），单 height ~8min，4 height 全跑 ~30min。GPU forward 不是瓶颈。
