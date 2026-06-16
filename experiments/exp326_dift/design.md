# 实验 exp326: DIFT（Stable-Diffusion emergent correspondence）pose-anchored part-MaxSim 训练-free 首验

> **来源**：post-PRCV「搬范式」路线，exp324（frozen DINOv2 emergent correspondence）的姐妹探针。
> **性质**：training-free 廉价首验，零训练。**机器**：hyy GPU0（5060 Ti 16G）。
> **决定性问题**：DIFT 训练-free 重遮挡 pose-part mAP 能否超 exp324 frozen DINOv2-base 的 **1.86**？

## 动机

- exp324 证明 frozen DINOv2-base + 姿态锚定 5-part + mutually-visible MaxSim 在重遮挡子集有干净相对信号（pose 锚定 ×3.4 vs 整图），但绝对天花板低（重遮挡 1.86 mAP，训头后 e20 即到顶 14）。
- 对应特征综述（DIFT, Tang NeurIPS'23 / SD-DINO 等）报告：**SD UNet 中间特征在遮挡 / 姿态对应基准上比 DINO 高 14-19 PCK**。若 SD 特征在「跨图 part 对应」上确实更强，训练-free 阶段就应当看到更高的重遮挡 mAP。
- 绿地 niche 与 exp324 一致：没人用 SD 的 emergent correspondence 接 occluded part 匹配。这里先用**最便宜的训练-free eval** 判断 SD 特征值不值得后续上头。

## 核心假设

frozen SD-v1.5 UNet，对图像 VAE 编码 → 小 timestep t 加噪 → 单步 UNet forward → 取 up_block 中间激活作 dense 特征，按 pose 锚定 5-part + mutually-visible part-MaxSim，在重遮挡子集 mAP **超过 exp324 DINOv2-base 的 1.86**。超了 → SD 特征值得上轻量头；不超 → SD 训练-free 不优于 DINO，路线降级。

## 技术方案（training-free）

1. **DIFT 特征**：`StableDiffusionPipeline`（SD-v1.5，fp16，safety_checker=None）。VAE encode → latent（用 latent_dist.mean，确定性）→ `scheduler.add_noise(lat, noise, t)` → `unet(noisy, t, empty_text_emb)` 单步 forward → forward hook 抓 `unet.up_blocks[up_block]` 输出 dense feature map (B,C,h,w)。空文本（无条件）。
2. **ensemble**：对 N 个随机噪声样本平均特征（DIFT 论文用 8，这里默认 4 控成本）。
3. **几何**：输入 256W×512H，VAE ×8 下采样 → latent 32×64，up_block[1] 输出分辨率由 probe 一张图动态读取（不硬编码），keypoints 缩放到该 grid。
4. **pose 锚定 5-part + MaxSim**：与 exp324/327 **完全相同**的 build_part_pose / part_maxsim_distmat（只比 mutually-visible part，per-part L2-norm 后 cosine 均值）。
5. **对照**：(a) holistic mean-pool cosine；(b) pose part-MaxSim；(c) grid part-MaxSim（均匀 5 横带，隔离 pose 锚定贡献）。
6. **eval**：mAP/R1/R5/R10，ALL query + 重遮挡子集（query visibility_binary.sum()<=8），与 exp324 同口径。

## 关键超参及依据

- `t=100`（of 1000）：小 timestep → 少加噪 → 保留细空间结构（DIFT 语义对应用 t≈261，细对应用更小，先取 100，必要时 smoke 扫 t∈{50,100,200}）。
- `up_block=1`：DIFT-sd 默认中高分辨率层（语义 + 空间折中）。
- `ensemble=4`：成本/稳定折中。
- 输入 256×512：保 1:2 行人比，VAE 友好（÷8 整除）。

## 预期结果

- 成立：DIFT pose-part 重遮挡 mAP > 1.86，且 pose>grid（锚定有效）→ 写赢家 → 上轻量 part 头。
- 失败最可能原因：(1) SD 特征在 256×512 低清脏 crop 上对应漂移，pose 降噪不足；(2) DIFT 优势在 PCK（关键点对应）不直接转 ReID 判别性；(3) ensemble 太小噪声大。任一即降级。

## 对照组

- baseline = exp324 frozen DINOv2-base 重遮挡 pose-part 1.86（同 pose data、同 5-part、同 MaxSim、同重遮挡口径）。唯一变量 = **特征源**（DINOv2 → SD-DIFT）。
- 内部消融：pose vs grid（隔离锚定）；t / up_block（smoke 扫，决定细 vs 语义层）。

## Kill-switch / 下一步

- 重遮挡 > 1.86 且 pose>grid → exp326b：轻量 part-projection 头（复用 exp324b 思路）训 SD 特征，天花板 check vs DINO 的 14。
- 否则 → 降级，SD 特征训练-free 不优于 DINO，记录止损。

## 备注（数据来源一致性）

- pose_data 为 **slim npz**（keypoints(17,2)+visibility_binary(17,)），由 lab-3090-d 原 exp324 npz 剥离 heatmap 生成（仅为加速跨机传输，keypoints/visibility_binary 与 exp324 逐字节一致）。
- find_pose 取 p0=首个排序检测，与 exp324 完全相同 → 与 1.86 baseline apples-to-apples。
