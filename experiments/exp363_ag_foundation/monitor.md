# exp363 AG Foundation Adaptation — monitor

## 数据/foundation 打通（2026-06-27，3090）
AG-VPReID.VIR（本地代理 gdown 下载 → rsync 3090，all/ 54739 图像 547M）+ DINOv2-reg-B（timm `vit_base_patch14_reg4_dinov2.lvd142m`；transformers 4.46/python3.8 不支持 dinov2_with_registers → timm 退路）+ 官方 dataset_reader.py（AG_ReID_IR_Enhanced，8 protocol exp5-12）。工程坎：4090 连不上 Google→本地代理 / gdown --remaining-ok 错参数 / transformers→timm。

## frozen DINOv2-reg baseline — cheap kill-switch 第一步（无训练）

### exp7 (cross_platform aerial_ground, visible_infrared) nframes=8
| pooling | mAP | R1 |
|---|---|---|
| single | 6.87 | 3.08 |
| mean | 8.62 | 4.62 |
| max | 8.43 | 5.00 |
| topk | 7.90 | 2.69 |
| oracle | 15.11 | 8.46 |

**codex 硬判定（复杂，一过一不过）**：
- mean − single = **+1.76 < +5**（不过，视频证据积累弱）
- oracle − mean = **+6.48 > +3**（过，选择空间大）

**诚实判读（不 cherry-pick）**：严格"任一不过即杀"→ mean-single 不过 = 简单 temporal mean 没多少视频增益。但 oracle-mean +6.48 大 = 帧质量差异大、选好帧空间存在（anchored-LoRA 机会），简单 mean 没利用。整体 mAP 极低（8.62）= frozen DINOv2 在 AG cross A-G IR 域 gap 大（aerial 视角 + IR 模态 vs DINOv2 RGB 自然图预训练）。exp7 = 最难 protocol（双域 gap）。

### 全 protocol（exp5-12，nframes=8）
| protocol | mean−single | oracle−mean | mAP(mean) |
|---|---|---|---|
| exp5 cross G-A V→IR | **−1.82** | +10.07 | 7.15 |
| exp7 cross A-G V→IR | +1.76 | +6.48 | 8.62 |
| exp8 cross A-G IR→V | +0.32 | +3.36 | 1.52 |
| exp9 same G-G V→IR | +1.60 | +10.16 | 9.13 |
| exp10 same G-G IR→V | +1.82 | +6.21 | 4.89 |
| exp11 same A-A V→IR | +0.28 | +5.28 | 6.04 |
| exp12 same A-A IR→V | +0.09 | +0.63 | 0.59 |

## ★结论：视频证据积累（temporal mean）路死（2026-06-27）
- **mean−single 全 8 protocol < +5**（最高 +1.82，exp5 还 −1.82）→ codex 硬判定全不过，简单 temporal mean 无视频增益。
- oracle−mean 大部分 +3~+10（选好帧空间大），但 oracle 用真 label（upper bound），anchored-LoRA 要无 test-label 学 frame quality（可能另一个坑：quality estimation 没监督）。
- 整体 mAP 极低（IR→V 0.59-1.52，V→IR 6-9）= frozen DINOv2-reg 对 AG aerial+IR 域太弱。
- **cheap kill-switch 价值**：几小时 frozen baseline 验死"视频证据积累"核心假设，没闷头训 anchored-LoRA（省多日）。
- codex 判方向生死（codex_frozen_verdict.md）见下。

## ★★codex 判：杀 AG 主线，保留资产，切 DG/Lifelong（2026-06-27）
- (a) temporal mean 路**确认死**（7 protocol mean-single 平均 +0.58 << +5，简单视频证据积累不成立）。
- (b) frame-quality selection **是坑**（oracle=retrieval-label upper bound，无监督 quality 学成清晰度/模态/中心度非 identity utility；generic selector 不新）。除非零训练 probe（label-free top-k 稳定 +2）否则别上。
- (c) 换 foundation **不救方向**（CLIP/EVA 仍 RGB prior 不解 IR；抬绝对值 ≠ 方法创新）。最多 CLIP-L frozen sanity 对照。
- **执行**：停 AG 主线（不补 attention/view-gate/LoRA-rank 小变体=移动 kill-switch）；不先做 anchored-LoRA（oracle-chasing）；**主资源切 #2 DG/Lifelong foundation-preserving adaptation**（问题=fine-tune 时保住跨域泛化 prior，数据 Market/MSMT/Duke 现成，kill-switch 清楚）。
- 沉没成本：半天不亏（数据+DINOv2+dataloader 链路 + 干净负结论），诚实止损。
- **AG 资产保留**（数据+DINOv2+dataloader+frozen baseline 脚本 on 3090），可当 negative control / 未来对照。
- codex 深化 DG/Lifelong 中（paradigm_shift/codex_dg_deepen.md）：novelty 窄缝 / **PSC-JEPA 同质性核查（DG 会不会同样 fine-tune-harm 死）** / cheap kill-switch / 信心。
