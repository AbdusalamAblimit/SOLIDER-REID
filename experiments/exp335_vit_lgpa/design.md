# 实验 exp335: ViT-base + LGPA-D 复现（CLIP 创新方向的地基）

## 动机
- 用户方向：找一个**和别人都不一样的 CLIP-as-module 接法**，带来涨点 → B 类。LGPA-D 是起点/证据（证明 CLIP 确实带新信息源）。
- **step 1（本实验）= 在干净 ViT baseline 上复现 LGPA-D，确认 CLIP +X**；step 2 再设计新 CLIP 接法。
- SMPL 两用法（exp333 β 特征 / exp334 几何先验）均证负 → 转 CLIP（已验证有效的外部信息源）。

## LGPA-D 机制（复用 `model/modules/clip_part_head.py`）
6 个 CLIP 部位文本原型（head/torso/arms/upper-legs/lower-legs/bg，ViT-B-32 文本编码，冻结缓存）当 query，cross-attend ViT patch token（key/value），**pose 热图当 additive attention bias**（KL 分配监督）。**"-D" = 喂进 CLIPPartHead 的 feat_map 被 detach**（梯度不回 backbone，这是让它 work 的关键 trick，exp243 vs exp244 +4.4）。

## 核心假设
ViT baseline（53.09）+ LGPA-D（detached CLIP 部位分支）应 +X（CLIP 语义部位证据是 backbone 没有的外部信息）。

## 技术方案
- 自包含 trainer `scripts/exp335_train_vit_lgpa.py`（现有 LGPA 管线与 Swin+PSG 耦合，故自建以便后续改 CLIP 接法）。
- 数据复用 `make_dataloader` 的 pose 路径（`PoseImageDataset` 做图+热图**联合对齐增强**，用 npz 的 crop_bounds 正确对齐——自己写轻量 loader 会对齐错）。target 热图 = `pose_dict['heatmaps'][:,0]`（17,H,W）。
- 模型 = ViT backbone（_tokens 取 cls + patches→feat_map (B,768,16,8)）+ CLIPPartHead；LGPA-D 时 `feat_map.detach()`。
- 损失：cls ID+triplet（外观）+ pooled 部位 ID+triplet + 各部位 CE + 0.5·assign(KL)。
- 测试：`[cls || pooled-part]` L2 归一 concat，alpha 扫。
- **单变量**：`--use_lgpa` off = 纯 ViT 外观（POSE_ENABLED 也关）；on = +LGPA-D。两臂同 seed。

## 预期 / 判据
- headline = LGPA-D best-alpha mAP vs 自训 off 对照（≈53.09）。预期 +2~5（参考 Swin 上 exp244 +4.4）。
- 若复现到 +X → CLIP 信息源成立，进 step 2（新接法）。若复现不出 → 查对齐/损失，再定。

## 对照组（Claude review 修正 H1/H2）
- **H2 修复**：off 臂也走 **pose dataloader**（POSE_ENABLED 两臂都 True，只 gate LGPA head）→ 增强完全一致、干净单变量。否则 off(timm RE max0.333) vs on(pose RE max0.40) 增强混淆。
- **H1 修复**：损失改用 **make_loss list-branch**（传 7 元素 `[global, pooled, p1..p5]`，得 0.5·global + 0.5·mean(pooled+5parts) + 每部位 triplet），忠实复现原 LGPA-D 权重(否则可能复现不出 +4.4)。
- LGPA 分支 detached → 不回传 backbone，故 on 臂 alpha=0 应 ≈ off 对照（sanity，类似 exp333）；H2 修复后此 sanity 才有效。

## 机器
lab-3090-d（ControlMaster；CLIP 文本特征缓存 pretrained/clip_part_text_features.pt；open_clip 已装）。
