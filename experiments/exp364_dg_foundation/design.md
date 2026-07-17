# exp364 — DG/Lifelong Foundation-Preserving Adaptation（跳出盒子第二个 build）

## 由来（2026-06-27）
- AG（exp363）杀（视频证据积累路死：frozen DINOv2-reg 全 8 protocol mean-single < +5；cheap kill-switch 一天验死）。
- codex 全 ReID gap analysis 排序 #2：DG/Lifelong foundation-preserving adaptation，信心 **6.5**（不是 7.5，窄缝窄，Not All Starting Points 抬高 direct-FT baseline）。

## 问题重定义（novelty 窄缝，codex）
DG-ReID 失败不只是 dataset-level domain shift，而是 **directed camera-pair / source-pair matching risk**——fine-tune 为源域 ID 判别，覆盖 foundation 保留的跨相机/跨数据集局部邻域几何，尤其伤高风险 camera-pair。

## 方法：Camera-Pair Foundation-Preserved Residual DG
- frozen F0（跨域 prior，不更新）+ 低秩 residual Rθ
- `z = norm(P(F0(x)) + β(pair-risk, x)·Rθ(x))`
- residual 只补源域判别，不重写高风险 pair 的 foundation 邻域拓扑
- preservation target = **cross-camera local rank topology**（保 F0 跨相机邻居 top-k soft dist，非单图 L2）
- β 由 source camera-pair risk 控（高风险小 residual）
- 避：normalization/IBN/MoE/meta/CLIP-prompt/CILP-FGDI

## ★PSC-JEPA 同质性 + U-shaped 判死（关键，避免 no-op 重演）
有同质风险（都"训练破坏 foundation prior + anchor 越强越 no-op"）。区别：PSC-JEPA continued-pretrain 改 backbone 保单域判别 vs DG frozen+residual 保跨域邻域结构。
**判死标准 = U-shaped sweet spot**：λ（preservation 权重）中间有峰值 = 真有救；单调 frozen↔FT 无峰值 = no-op 死（和 PSC-JEPA 一样）。

## cheap kill-switch（先验证前提，非写方法）
### 第一步：frozen cross-domain probe（零训练，本轮）
DINOv2-reg frozen 提特征，Market/MSMT/Duke in-domain mAP baseline。验证 frozen foundation 的 ReID 基线（fine-tune 要超它才有意义，破坏它则 fine-tune-harm）。

### 第二步：4 卡矩阵（frozen / head-only 短训 / direct-FT 30ep）
多源训 → held-out（Market+Duke→MSMT 等）。记：source gain / held-out gap vs frozen / neighbor overlap@20 topology drift / drift-failure Spearman。
- **Go**：direct-FT source 大涨但 held-out 比 frozen/head-only 低 ≥3 且 ≥2 协议 / head-only 优 full-FT held-out ≥2
- **Kill**：direct-FT 所有 held-out 赢 frozen/head-only ≥2（foundation-preserving 没燃料）→ 转 open-set/gallery-growth/distractor-aware lifelong

## 数据/基建
Market/MSMT17/occluded_duke 现成（3090 /root/work/SOLIDER-REID/data/）。DINOv2-reg timm `vit_base_patch14_reg4_dinov2.lvd142m`（transformers python3.8 不兼容用 timm）。foundation fine-tune 前 codex 三审 diff。

关联：`paradigm_shift/codex_dg_deepen.md`，memory [[paradigm-shift-occluded-reid-wall]]，exp363（AG negative control）。
