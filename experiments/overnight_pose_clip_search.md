# 通宵 pose+CLIP 深度融合涨点搜索 (2026-06-20 夜)

## 目标
找一个 pose+CLIP 融合机制, test.py >59.8 且清噪声(>~+0.5)。基线: exp341 CLIP=59.8, baseline=57.6, exp342外挂=60.0(+0.2 marginal)。

## 机制洞察(指导所有设计, 来自白天 A/B/C 全负)
exp341 +2.2 来自 **global 对齐纯 ID 文本原型**(把 global 塑成纯 ID-判别)。三个坑:
- A(pose-bias 池化, learnable query/k_proj)→ 通路吸收 i2t/t2i 梯度 → 57.6
- B(pose 调制 prompt)→ 原型 pose-aware → global 编码姿态非纯ID → 57.6
- C(K 部位对齐 ID 原型, learnable query)→ 部分吸收 → 58.0
**绕坑原则**: 姿态只帮"对齐信号干净", 不碰对齐目标(纯ID), 用无参数池化(不吸收)。

## 通宵 bet 矩阵(全双审查通过)
| | 机制 | 假设 | 状态 |
|---|---|---|---|
| b exp342b | un-detach LGPA(姿态塑造backbone,部位独立ID监督) | 塑造≠竞争对齐 → 破冗余 | 4090跑 |
| c exp342c | clean global 1.0x(GLOBAL_LOSS_SCALE 2.0 修M1) | 干净global → external从+0.2到+0.4 | 3090跑 |
| exp347 | param-free de-occluded 对齐(无参数池化) | 梯度直进backbone(修A吸收)+对齐去遮挡global到纯ID | 排4090 |
| exp348 | exp347 + occluder repulsion | 显式分离 可见=ID/遮挡=非ID → GAP更干净 | 排3090 |
| exp349 | Swin-Small全pose(73.2)+CLIP | 容量大+互补 → 组合>73.2 | 排4090 |

## 备用角度(若上述全 marginal)
- pose-reliability-weighted 对比: 高可见样本在 i2t/t2i 权重大 → 原型更干净(不碰feature,不编码姿态)
- CLIP-supervised pose parts 当独立描述子(per-part 学习原型, 不进 global 对齐 → 不吸收)
- pose-guided 遮挡增强 + CLIP 一致性(已部分在 exp349 的 PLBOA+CLIP)

## 结果(待填)

## ★★★ 突破 (2026-06-20 凌晨): exp342b un-detach LGPA + CLIP = +0.9!
| | 描述子 | mAP | R1 | vs exp341(59.8/68.4) |
|---|---|---|---|---|
| exp342 (detached LGPA) | equal_concat | 60.0 | 68.9 | +0.2/+0.5 (marginal) |
| **exp342b (UN-detach LGPA)** | equal_concat | **60.7** | **69.3** | **+0.9/+0.9 清噪声!** |
exp342b global=58.8 (un-detach 姿态塑造与纯ID竞争↓), 但部位塑造后判别↑ → equal_concat 净 +0.9。
**机制**: un-detach 让 LGPA 部位的梯度流进 backbone, 部位有独立 id_part/tri_part 监督(不进CLIP对齐故不吸收CLIP), 姿态真塑造 backbone → 破"只当外挂"的冗余。**这是 pose 深度参与 + CLIP 的真涨。**
**深挖方向**: (1) un-detach + 保护global(scale 2.0); (2) scale-up 到 Swin-Small 全系统(exp349, 若其LGPA本就un-detach); (3) un-detach + de-occluded对齐叠加。

## 深挖结果
- **exp350 (un-detach + clean global 2.0x) = 59.7 equal_concat** (global 57.9) — **比 exp342b 60.7 差!** 2.0x 过度加权 global 抢了部位注意力(部位carry +0.9)→ 掉。**原 1.0x(exp342b)最优, 保护global想法错。** NO-GO。
- exp351 (un-detach + de-occluded) 跑中
- exp349b (Swin-Small scale-up) 跑中

- **exp351 (un-detach + de-occluded) = 60.3** — 比 exp342b 60.7 略低。un-detach 已塑造backbone, de-occluded净化对齐多余。NO-GO(不加成)。
- **深挖小结**: exp342b(un-detach 纯净, 1.0x scale)= 60.7/69.3 = +0.9 仍是最优。变体(clean global 2.0x=59.7, de-occluded叠加=60.3)均未超。un-detach 本身是关键, 加料不如纯净。

## exp349b (Swin-Small 全系统 + un-detach + CLIP) = 65.7/64.7 — 大跌 -7.5 vs exp255 73.2!
un-detach 破坏了全系统(强系统为 detached LGPA 调好, un-detach 让 LGPA 塑造 backbone 干扰 PSG/GCN/OA-SD 平衡)。
**关键: un-detach 突破(exp342b +0.9)是 Swin-Tiny 纯 LGPA+CLIP 特有, 不泛化到全系统。** NO-GO。
→ 正确 scale-up = detached + CLIP(exp349, 测 CLIP 加到正确强系统)。

- **exp348 (de-occluded + occluder repulsion 独立) = 57.6 = baseline, -2.2 vs exp341** — de-occluded 对齐把 global 拉回 baseline。整条 de-occlusion(exp347/348)NEGATIVE 死。
- **完整结论: un-detach(exp342b 60.7/69.3 +0.9)是唯一赢家。所有 7 变体(clean global/de-occluded×2/scale-up/occluder)均未超。纯净 un-detach 最优。**

## ★★★ 重要纠正 (exp353 隔离, 用户问题戳穿): +0.9 大部分是 pose 不是 CLIP
exp353 (un-detach LGPA **无CLIP**) = 60.5/68.4 equal_concat (global 57.7≈baseline)。
| | mAP | over baseline 57.6 |
|---|---|---|
| 只 CLIP (exp341) | 59.8 | +2.2 |
| 只 un-detach LGPA (exp353) | 60.5 | **+2.9 (已>CLIP!)** |
| un-detach LGPA + CLIP (exp342b) | 60.7 | +3.1 |
**CLIP 加到 un-detach LGPA 上只 +0.2(60.5→60.7)。** CLIP(+2.2)与 un-detach LGPA(+2.9)**冗余**(合+3.1 << 和5.1, 都塑造backbone学ID)。equal_concat 被部位主导, CLIP 改善的global被稀释。
**纠正结论: exp342b 的 +0.9 大部分是 pose(un-detach LGPA, 用户自己的机制), CLIP 只 +0.2 边际/冗余。无真正 CLIP+pose 协同。CLIP 给强pose系统加东西加不动(冗余), 与白天"强backbone上互补信号被吃掉"一致。**

## ★ exp349 (强系统 scale-up) 最后确认: CLIP 有害 -1.8
exp349 = exp255 全 pose 系统(Swin-Small + 2-stage PSG + LGPA + GCN512 + OA-SD + PLBOA)+ CLIP prompt, e120 训练eval mAP = **71.4%** vs exp255 **73.2%** → **CLIP 拉低强系统 -1.8**(test.py poll 待精确分解)。
**完整画面: CLIP 在弱裸baseline +2.2; 弱系统+pose 冗余(+0.2); 强pose系统 -1.8 有害。** CLIP 只在裸弱baseline有用, 有pose结构后从冗余变累赘(纯ID对齐与全系统多loss/部位结构冲突)。
**pose+CLIP 终极结论: 全局层冗余/有害, 空间层CLIP非空间(PC-SOR死)。无任何productive fusion。今晚交付=完整诊断(8实验+20codex+2 kill-switch+强系统确认)。**
