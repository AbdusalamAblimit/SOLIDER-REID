# 实验 exp382：Video TAPF 查新与数据门禁

## 动机

exp378/380/381 已把完整 `anchor+PSG` 原子方法闭合为训练期使用姿态监督、测试期 RGB-only 的方法对象。三个骨干上 D0 相对各自 B0 的 mAP 增益分别为 Swin `+1.1`、ResNet-50 `+3.1`、ViT-B `+2.0`。本轮只判断该对象能否自然扩展为一个有独立方法新意的视频 ReID 贡献，不预设一定训练。

## 核心假设

候选假设原为：逐帧原子 TAPF 之外，跨帧姿态可靠性、运动连续性和遮挡恢复可以形成一个训练期 pose-privileged、测试期 RGB-only 的内部时序状态；该状态应提供普通 temporal pooling、帧质量加权或参数量增加无法解释的增益。

## 查新门禁

正式设计和下载数据前，必须同时满足：

1. 没有先例同时覆盖“视频结构特权教师、时序知识迁移、RGB-only 视频学生”；
2. 相对 KPRTrack、PSTA、STMN、TF-CLIP、PAFormer 的差分不是换名后的部位聚合、质量加权或普通蒸馏；
3. 可以定义一个可直接观测、可被 correct/shuffle/constant/temporal-order 干预裁决的中间变量；
4. 有可靠、许可清楚且规模可控的视频数据来源。

## 查新证据

### 直接碰撞

- [GAE-Net](https://doi.org/10.1016/j.neunet.2025.107946) 已把视频 ReID 重定义为训练期 RGB+gait 多模态教师、再通过 local perceptual complementary distillation 蒸馏到 RGB-only 视频学生。其公开正文审计记录显示：RGB ReID 分支 `84.3 mAP`，多模态 DTA-Net `85.8`，RGB-only 学生 GAE-Net `87.7`；部署参数由 `164.1M` 降为 `24.8M`。这直接覆盖“时序人体结构作为特权信息、测试期只保留 RGB 视频分支”的大叙事。
- [PAFormer](https://arxiv.org/abs/2408.05918) 已覆盖 pose heatmap 监督 pose-token attention、visibility teacher forcing 与测试期不使用 pose heatmap。
- [KPR/KPRTrack](https://arxiv.org/abs/2407.18112) 已覆盖 keypoint-prompted parts、共同可见部位比较以及 tracklet 内同部位 moving average。

### 视频时序空间已经拥挤

- PSTA、CTL、GRL、STMN、TF-CLIP 等已覆盖时空关系、局部线索、遮挡/干扰记忆、sequence memory 与 temporal memory diffusion。
- AG-VPReID（CVPR 2025）已用 Temporal-Spatial、Normalized Appearance、Multi-Scale Attention 三流处理 temporal discontinuity、视角/分辨率和尺度；AG-VPReID.VIR 又覆盖跨平台、跨视角 memory 与 intermediary-guided temporal learning。
- 因此，把跨帧 pose state 写成 temporal evidence routing，仍很容易被归类为已有的结构特权蒸馏、同部位 tracklet 聚合或 occlusion-aware temporal memory。

本轮证据来源包括：

- `experiments/cargo_cvpb/litreview2/pivot/clean/video_feasibility.txt`
- `experiments/cargo_cvpb/litreview2/reviews/deep_13.md`
- `experiments/exp371_casd/critical_prior_audit_2026.md`
- `experiments/exp371_casd/frozen_support_oracle_v2_design.md`

## 数据门禁

2026-07-17 远端只读审计结果：

- `/home/afr/datasets/AG-VPReID.VIR` 仅 `4.0K`，为空目录；
- 未发现 MARS、DukeMTMC-VideoReID、iLIDS-VID 或 PRID2011 数据目录；
- 4090 为 `2 MiB / 0%`，没有训练进程。

目录存在不等于数据可用。本轮禁止下载 AG-VPReID 9.6M frames，也不以下载 MARS 代替新颖性门禁。

## 若未来以应用扩展重新开启

只有论文主方法已经闭合、视频数据有可靠来源，并且目标明确降级为“跨任务外部验证”时，才允许做同一 video backbone、同 sampler、同帧数、同 seed 的三臂：

1. `B0`：RGB temporal baseline；
2. `D0`：逐帧原子 `anchor+PSG` 后做完全相同的 temporal aggregation；
3. `T0`：在 D0 上增加真正跨帧 pose-state，直接报告 `T0-D0`。

其中 `D0-B0` 只能回答原子方法能否迁移到视频；`T0-D0` 才回答跨帧机制。普通 temporal pooling、更多帧、额外参数或测试期 pose 输入都不能算 TAPF 贡献。

## 风险与失败解释

- 若只做 D0，结果可作为 backbone/task transfer，不构成新视频方法。
- 若 T0 上涨但不能领先 temporal attention、top-k frame quality、KPRTrack-style part average 与 GAE-Net-style privileged distillation，对新颖性仍是失败。
- 当前新颖性门禁和数据门禁均未通过，因此不进入实现、预检、下载或训练。

## 当前决定

**NO-GO（作为独立视频方法主线，信心 9/10）。**

保留 Video TAPF 作为未来论文的应用扩展候选，但不把“训练期姿态、推理期 RGB-only”在视频场景重新包装成新贡献。论文中心继续是已经跨 Swin/ResNet/ViT 得到正向证据的完整单图 `anchor+PSG` 原子方法；Hierarchical 仍只作 backbone-conditional 扩展。
