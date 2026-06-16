# VC-Norm 前提探针（无训练，lab-3090-d）

## 一句话结论

**VERDICT = 有燃料 / PROCEED。** 遮挡确实在 per-part-token 的归一化统计上造成
**巨大且可分离**的分布 shift（per-keypoint 高可见 vs 低可见 token：median KL≈288，
LDA AUC≈0.97），且经过 3 个对照证明**不是采样伪影**。VC-Norm
（把遮挡当 domain factor 做 visibility-conditioned normalization）的前提成立，
下一步可上 1-2d dual-forward Market 30ep（目标 Occ-ReID mAP > 88.0 baseline）。

> 注意这是 NECESSARY 前提（有可对齐的 domain 轴），不是 SUFFICIENT 证明
> （对齐这条轴是否真涨 mAP 只能训练验证）。本探针只回答"有没有燃料"。

## 设定

- ckpt：`log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth`
  （Market-trained Swin-Base + PSG + LGPA + GCN512，主线 exp260b，Occ-ReID baseline 88.0 mAP MaxSim+flip）
- 数据：Occluded-ReID（query 1000 + gallery 1000，200 ids）+ pose_data
- env：`/root/miniconda3/envs/solider-reid`（torch1.13+mmcv，与训练一致）
- per-part token = SkeletonGCNHead 在 PSG-modulated Stage-3 特征图上按 17 个
  COCO 关键点 bilinear 采样得到的 token（dim=1024）。pre_gcn=采样原始 token，
  post_gcn=GCN 增强后 token（即 `aux_data['kp_feats']`）。
- 可见性 = pose 置信度 `scores`（ViTPose conf）。high-vis: score≥0.7；low-vis（遮挡）: score≤0.2。
- 距离：逐通道对角高斯的对称 KL（=均值/方差归一化统计 shift）、2-Wasserstein、
  均值 L2、Fisher-LDA 方向上 held-out ROC AUC（可分性）。

脚本：`scripts/vcnorm_probe.py`（主探针）+ `scripts/vcnorm_probe_control.py`（对照）。
原始结果：`vcnorm_probe_full.json` / `vcnorm_control_result.json`。

## 主探针：per-keypoint 高可见 vs 低可见 token 统计（KL 表）

只列 low-vis 计数足够（≥50）的关键点。上半身（肩/肘/髋）几乎总是高置信，low-vis
样本不足被跳过——这本身说明 Occ-ReID 的遮挡集中在头部边缘 + 下肢。

| keypoint | n_hi | n_lo | KL_sym(pre) | KL_sym(post) | LDA_AUC(pre) | LDA_AUC(post) |
|----------|------|------|-------------|--------------|--------------|---------------|
| nose     | 1700 | 64   | 289.5 | 167.4 | 0.972 | 0.982 |
| l_eye    | 1676 | 69   | 288.1 | 174.9 | 0.962 | 0.982 |
| r_eye    | 1652 | 67   | 270.6 | 169.9 | 0.979 | 0.993 |
| l_ear    | 1789 | 56   | 257.9 | 164.7 | 0.948 | 0.963 |
| r_ear    | 1766 | 57   | 297.7 | 171.2 | 0.970 | 0.988 |
| l_knee   | 1653 | 204  | 122.0 | 93.5  | 0.953 | 0.976 |
| r_knee   | 1652 | 208  | 126.7 | 94.4  | 0.965 | 0.984 |
| l_ankle  | 1571 | 318  | 300.9 | 194.7 | 0.986 | 0.988 |
| r_ankle  | 1552 | 324  | 293.5 | 193.1 | 0.984 | 0.987 |
| **median** | | | **288.1** | **169.9** | **0.970** | **0.984** |

读法：KL 量级在 **94–300**（远非 ≈0），AUC **0.95–0.99**（接近完美可分）。
遮挡 token 与可见 token 在归一化统计上几乎线性可分 → 有强 domain 轴可对齐。
post-GCN KL 略降（GCN 沿骨架传播部分修复了被遮挡 token），但 AUC 反而更高，
说明 GCN 没有抹掉这条 domain 轴。

## 对照：是真 domain factor 还是关键点采样伪影？

担心：遮挡关键点 score 低 + 坐标常被 ViTPose 钉到边界/幻觉位置，bilinear
（border padding）采到 off-body 特征——若如此，shift 只是"on-body vs off-body
采样"，不是 VC-Norm 能用的遮挡 domain factor。

| keypoint | n_lo | %border | KL(hi,lo) | KL(hi,**rand**) | KL(hi,**lo_onbody**) | n_onbody |
|----------|------|---------|-----------|-----------------|----------------------|----------|
| nose   | 64  | 10.9% | 289.5 | 121.5 | 294.5 | 57  |
| l_eye  | 69  |  8.7% | 288.1 | 133.6 | 293.8 | 63  |
| r_eye  | 67  | 17.9% | 270.6 | 132.9 | 291.1 | 55  |
| l_ear  | 56  |  3.6% | 257.9 | 124.6 | 277.6 | 54  |
| r_ear  | 57  |  8.8% | 297.7 | 123.0 | 345.0 | 52  |
| l_knee | 204 |  5.4% | 122.0 |  90.9 | 126.6 | 193 |
| r_knee | 208 |  7.2% | 126.7 |  90.2 | 126.0 | 193 |
| l_ankle| 318 |  7.2% | 300.9 | 188.0 | 315.9 | 295 |
| r_ankle| 324 |  6.5% | 293.5 | 181.2 | 314.6 | 303 |
| **median** | | **7.2%** | **288.1** | **124.6** | **293.8** | |

三个对照全部通过：
1. **%border 仅 7%**：遮挡关键点坐标绝大多数仍在体内，不是被钉到边界的退化坐标。
2. **KL(hi,lo)=288 ≫ KL(hi,rand)=125**（~2.3×）：遮挡 token 的 shift 显著大于
   "在体内随机位置采样"的 shift → 这条 shift 是**遮挡特有**的，不是泛泛的 off-kp 采样噪声。
3. **KL(hi,lo_onbody)=294 ≈ KL(hi,lo)=288**：剔掉那 7% 边界坐标后 shift 几乎不变 →
   **不是边界采样伪影**，是表示层真实的遮挡 domain shift。

## 机制解读

- exp260b 是 Market（几乎全可见）训练的，从未见过重遮挡；拿到 Occ-ReID 上，
  被遮挡部位的 token 落到了一个与可见 token 明显不同的统计区域（mean/var 都漂）。
  这正是"遮挡 = 未对齐的 domain factor"的直接证据。
- 这条轴用对角高斯 KL 就能量到、用一个线性方向（LDA）就能近完美分开 →
  说明它**结构简单、可被一个归一化/对齐模块吸收**，符合 VC-Norm 的设计假设
  （per-part-token visibility-conditioned normalization）。
- GCN 已经沿骨架做了部分修复（post KL↓），但远没有抹平这条轴 → 还有 VC-Norm
  可吃的 headroom，且 VC-Norm 与 GCN 不重复（一个对齐归一化统计，一个传播结构）。

## 重要 caveat（诚实）

- **这是必要条件，不是充分条件。** 有可对齐 domain 轴 ≠ 对齐后一定涨 mAP。
  风险：沿这条轴归一化可能连带抹掉一部分身份判别信号（occluded 与 visible 的
  统计差异里，可能既有"遮挡噪声"也有"真身份差异"）。这只能靠下一步训练验证。
- 上半身（肩/肘/髋）low-vis 样本太少（Occ-ReID 遮挡分布所致），KL 表只能覆盖
  头部 + 膝/踝；但这些已足够给出明确的 PROCEED 信号。
- 跨域口径：本探针在 **test 端 Occ-ReID**（遮挡多）上做，不受"95.8% 训练全可见墙"
  直接限制——那堵墙限制的是训练端能学到的遮挡推理，这里量的是测试端的分布失配。

## 下一步（若推进）

1-2d dual-forward Market 30ep：训练时构造 occluded/clean 两路 forward，在
per-part-token 上对齐归一化统计（visibility-conditioned），目标
**Occ-ReID mAP > 88.0**（exp260b MaxSim+flip baseline）。先用 30ep 短训当 kill-switch，
不涨就止损。
