# exp324 监控记录

脚本：`scripts/exp324_dino.py`（training-free，frozen DINOv2-base，纯推理）
机器：lab-3090-d（RTX 3090 24G），系统 python3（torch 2.7.1+cu118, transformers 5.2.0）

## 关键实现参数
- 输入分辨率 224W × 448H（保持行人竖长比，patch=14 → grid 32 行 × 16 列 = 512 patch token + 1 CLS）
- 坐标缩放：keypoints 已在原图像素空间 → `col = x/W*16`, `row = y/H*32`
- 部位池化：每个可见 keypoint 映射到 grid cell，取 Chebyshev 半径 1（3×3 窗）内 patch token 均值池化；部位向量 = 该部位所有可见 keypoint 窗口的 patch 并集均值
- COCO-17→5 部位：head[0-4] / torso[5,6,11,12] / arms[7-10] / legs[13,14] / feet[15,16]
- 部位可见 = 该部位任一 keypoint 可见且非 (0,0) sentinel
- part-MaxSim：只在双方都可见的部位算 per-part cosine，对 common 部位求均值；无 common → 距离 2.0
- 多人选人：每图取 p0（primary 检测，最高置信），与 exp323 gallery 选人逻辑一致
- 重遮挡子集：query `visibility_binary.sum() <= 8`
- 注意：`utils.metrics.eval_func` 顶层 import model 包需要 mmcv/mmengine（系统 python 无），故把 eval_func 逐行 inline 复制进脚本（纯 numpy，无依赖）

## [smoke] 50 query × 2000 gallery（小 gallery 不代表全量）
| 方法 | ALL mAP/R1 | HEAVY mAP/R1 |
|------|-----------|-------------|
| (a) holistic CLS | 1.05/0.00 | 0.72/0.00 |
| (a) holistic mean-pool | 1.75/0.00 | 0.97/0.00 |
| (b) pose part-MaxSim | 8.19/12.00 | 2.55/0.00 |
| (c) grid part-MaxSim | 2.20/0.00 | 1.30/0.00 |

流程跑通；趋势符合预期 (b)>(c)>(a)，pose 锚定有正增益。绝对分低符合 DINO 零样本预期。
feature 抽取 ~80 img/s（gallery 2000 张 26s）。

## [FULL] 2210 query × 17661 gallery（lab-3090-d，2026-06-16）

重遮挡子集 989/2210（query visibility_binary.sum() <= 8）。no-pose 图 0（全部有 pose）。

| 方法 | ALL mAP/R1/R5/R10 | HEAVY mAP/R1/R5/R10 |
|------|-------------------|---------------------|
| (a) holistic CLS | 0.64/0.90/3.12/4.62 | 0.55/0.81/2.43/3.24 |
| (a) holistic mean-pool | 0.70/1.27/3.94/5.38 | 0.57/0.71/3.34/4.15 |
| **(b) pose part-MaxSim** | **3.21/7.87/15.29/19.32** | **1.86/3.54/8.59/11.63** |
| (c) grid part-MaxSim | 0.89/1.63/4.34/5.84 | 0.67/1.21/2.93/3.84 |

**相对（重遮挡子集）**：
- pose-part vs holistic CLS：**+1.31 mAP / +2.73 R1**（mAP ×3.4，R1 ×4.4）
- grid-part vs holistic CLS：+0.12 mAP / +0.40 R1（几乎无增益）
- **pose-part vs grid-part：+1.19 mAP / +2.33 R1** → pose 锚定贡献占绝对主导（grid 那点增益只是 part 分解本身的微弱效果）

ALL 子集同向更明显：pose-part 3.21 mAP / 7.87 R1，holistic 仅 0.64/0.90，grid 仅 0.89/1.63。

**耗时**：feature 抽取 293s（query 30s + gallery 251s，~75 img/s）；rep building 327s（含每图重开 PIL 读尺寸缩放 kp，2 万次 PIL open，是主要瓶颈）；distmat 0.8s；总 629s（~10.5 min）。

## 结论

- **机制有明确相对信号**：frozen DINOv2 dense token + pose 锚定 5-part + mutually-visible part-MaxSim 在重遮挡子集**显著超过整图基准**（mAP/R1 提升 3-4 倍），且 **pose 锚定 >> 均匀网格**——证明涨点来自"姿态把 token 约束到身体部位语义"，不是部位分解本身的 trivial 效果。
- **绝对分仍低**（pose-part heavy 1.86 mAP），符合 DINO 零样本 ReID 文献区间（0.3-4.7 mAP）。training-free 不足以做可用 ReID，但**信号方向正确**。
- **值得下一步**：上轻量 part-projection 头 / LoRA 把 DINO 特征投到 ReID-judiciable 空间，再全量对比 KPR（design.md kill-switch 命中"重遮挡组超 holistic 且 pose 锚定有效" → exp324b）。
