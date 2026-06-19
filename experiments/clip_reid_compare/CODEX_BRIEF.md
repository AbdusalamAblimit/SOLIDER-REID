# CODEX 调查 brief — CLIP-ReID 对比 + 我们 pose×CLIP 的创新性验证

## 我们的方法（LGPA = Language-Grounded Part Assignment）
- 在 SOLIDER/Swin ReID backbone 上加一个 CLIP 部位分支:
  - 6 个 CLIP 文本部位原型("head and face of a person"/"torso"/"arms"/"upper legs"/"lower legs"/"background",ViT-B-32 文本编码,**冻结**)当 query。
  - cross-attend backbone 空间 token(key/value),**pose 热图当 additive attention bias**(引导每个部位文本 attend 到对应身体区)。
  - KL 分配监督损失(pose-GT)+ 每部位 BN/分类器 + visibility 加权池化。
  - **"-D" = 喂进去的 feat_map 被 detach**(梯度不回 backbone)。
- 测试描述子 = equal_concat `[global_norm ‖ pooled ‖ p1..p5_norm]`。
- 代码:`model/modules/clip_part_head.py`(CLIPPartHead 模块)+ `model/pose_backbone_model.py`(集成,~line 600/800 调用)。

## ⭐ 我们的关键实验发现（3-seed 确认,非噪声）
| | pose-CLIP(有 pose) | no-pose(LGPA 收 heatmaps=None,纯 CLIP-text 部位) |
|---|---|---|
| equalcat vs global 增益 | **+0.9 ± 0.0**(s0/1/2 全 +0.9) | **−0.17 ± 0.1**(≈0) |
- **结论:增益来自 pose 注入,不是 CLIP 文本语义本身。** 去掉 pose,CLIP 文本部位原型对 global 零贡献(冗余)。pose-bias 引导注意力到对的身体区才是增益来源。CLIP 文本只是 query 壳。

## CLIP-ReID（对比对象,已下载本地）
- 代码:`experiments/clip_reid_compare/CLIP-ReID/`(官方 Syliz517/CLIP-ReID,AAAI 2023)
- 论文:`experiments/clip_reid_compare/clip_reid_paper.pdf`(arxiv 2211.13977)
- 核心:**可学习 ID-level 文本 prompt**(CoOp 式),2-stage 训练(stage1 学 prompt,stage2 微调 image encoder)。无 concrete text label。

## 调查目标（你被分配其中一个角度）
1. **彻底搞懂 CLIP-ReID** 的机制/训练/结果。
2. **对比 ours(LGPA: 部位级文本 + pose + detach)vs CLIP-ReID(ID 级 prompt,无 pose)** 的结构差异。
3. **诚实评估 pose×CLIP 的创新性**——尤其考虑我们的发现(pose 驱动、CLIP 文本冗余):这还算不算"CLIP 创新"?够不够 B 类?
4. **文献查新**(用 --search):pose+VLM、部位级 CLIP、遮挡+CLIP、pose-ReID、detach 辅助分支、text-query cross-attn 等。

## 你的输出
针对你的角度:具体发现(引代码行/论文段/web 来源)+ 对"pose×CLIP 创新性"的判断 + 信心分。**诚实**:如果发现 CLIP 部分冗余/不新颖,直说。
