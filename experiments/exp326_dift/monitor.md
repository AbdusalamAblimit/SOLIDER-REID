# exp326 监控记录 — DIFT（Stable-Diffusion 特征对应）pose-part-MaxSim

脚本：`scripts/exp326_dift.py`（training-free，frozen SD-v1.5 UNet 中间特征，纯推理）
机器：hyy-5060ti-double GPU0（5060 Ti 16G），py3.11 + torch 2.9.1+cu128 + diffusers 0.38.0 + transformers 5.12.0
特征源：`stable-diffusion-v1-5/stable-diffusion-v1-5` VAE encode → t=100 加噪 → 单步 UNet forward → hook `up_blocks[1]` 输出 → ensemble=4 平均

## 关键实现参数
- 输入 256W×512H（保 1:2 行人比，VAE ÷8 整除）。up_block[1] 输出 feature map **C=1280, grid 32(行)×16(列)**（与 exp324 DINO grid 同形状，空间分辨率相当）。
- pose 锚定 5-part + mutually-visible MaxSim + 重遮挡口径（vis.sum()<=8）与 exp324 **逐字节一致**（同 PART_GROUPS / POOL_RADIUS=1 / find_pose p0 / eval_func）。
- pose_data 为 slim npz（keypoints+visibility_binary），由 lab-3090-d 原 exp324 npz 剥 heatmap 生成（数值一致），已 rsync 到 hyy。
- 特征不写盘（--cache 默认关），全留 503G 主存（float16，gallery ~0.7GB/500img 量级，full ~26GB RAM，host 480G free 充足）。

## 双审查
- Claude Broad Review：通过（C1 pose 缺失静默 0.00 已加 assert+guard；H1 disk 溢出改 in-RAM；M3 hook stale 加 clear+assert）。`claude_review.md`
- Codex --search exec：approve（exp327 assert 收紧、dinov3 默认改 ungated；DIFT premise 联网核验 SPair +19 PCK 但 SD/DINO 互补）。`codex_review.md`

## [smoke] 30 query × 500 gallery（ensemble=2，小 gallery 绝对值偏高，仅看趋势）
| 方法 | ALL mAP/R1/R5/R10 | HEAVY mAP/R1/R5/R10 |
|------|-------------------|---------------------|
| (a) holistic mean-pool | 5.34/13.33/16.67/23.33 | 5.06/9.09/9.09/18.18 |
| **(b) pose part-MaxSim** | **14.68/26.67/46.67/76.67** | **9.92/9.09/18.18/63.64** |
| (c) grid part-MaxSim | 7.27/20.00/26.67/43.33 | 6.54/18.18/18.18/27.27 |

趋势：pose-part(9.92) > grid(6.54) > holistic(5.06)，pose vs grid heavy **+3.38 mAP** → 姿态锚定有效。流程跑通（hook 正常 fire，无报错），DIFT loaded 102s（模型下载 + 加载），feature ~0.06s/img(e2)。**绝对值因 gallery=500 偏高，需 full gallery 才能与 1.86 apples-to-apples**。

## [FULL] 2210 query × 17661 gallery（ensemble=4，2026-06-16，hyy GPU0）

heavy-occ 989/2210（与 exp324 一致）。no-pose 0。feature map C=1280, grid 32×16。耗时 2065s（feature 1650s = ensemble4×4 UNet forward 慢 + rep building 405s + distmat 0.9s）。gallery 特征 23.1GB in-RAM。

| 方法 | ALL mAP/R1/R5/R10 | HEAVY mAP/R1/R5/R10 |
|------|-------------------|---------------------|
| (a) holistic mean-pool | 0.21/0.14/0.77/1.31 | 0.22/0.20/0.71/1.21 |
| **(b) pose part-MaxSim** | **0.92/2.58/6.29/9.28** | **0.73/1.42/4.45/7.79** |
| (c) grid part-MaxSim | 0.39/1.09/2.35/2.99 | 0.35/0.81/1.92/2.12 |

**>>> vs exp324 DINOv2-base heavy pose-part 1.86：DIFT heavy 0.73（−1.13 mAP）—— DIFT 全量明显劣于 DINOv2-base。**

机制方向仍在（pose 0.73 > grid 0.35 > holistic 0.22，pose vs grid +0.38），但**绝对判别性远低于 DINO**。

## 结论（exp326）：DIFT/SD 特征不值得做，决定性负结果

- **决定性问题答案 = 否**：DIFT 训练-free 重遮挡 **0.73 << DINOv2-base 1.86**（−1.13），更不及 dinov2-registers 2.15。SD 特征**不值得上头**。
- **smoke 误导剖析**：smoke（500 gallery）DIFT pose-part heavy 9.92，但 full（17661 gallery）塌到 0.73。原因——500 张 gallery distractor 极少，所有方法虚高；**DINO 从 smoke 2.55 → full 1.86 仅小降，DIFT 从 9.92 → 0.73 灾难性塌**。说明 **SD/DIFT 特征是 category-level 语义对应强（PCK 高），但 instance-level 身份判别弱**（与 SD-DINO/Tale-of-Two-Features 文献一致：SD 与 DINO 互补，SD 不主导 instance retrieval）。
- **教训**：训练-free probe 必须用**全量 gallery** 判定，小 gallery smoke 只看流程不看绝对值——DIFT 是这条铁律的活教材（smoke 排第一、full 垫底）。
- **不上 exp326b 头**：SD 特征 instance 判别性弱，训头起点（0.73）远低于 DINO（1.86→14）。SD 线止损。
- DIFT 其他配置（t∈{50,200}、up_block∈{0,2}、ensemble=8）理论上可能更好，但 instance-discrimination 是 SD 特征的**结构性短板**（非超参问题），不值得继续扫。
