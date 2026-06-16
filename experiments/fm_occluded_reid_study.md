# 基础模型用于 Occluded Person ReID — 一夜诊断研究

> 夜间自主探索成果（2026-06-15 夜）。用户目标：复刻 CLIP→CLIP-ReID / 扩散→Pose2ID 的"搬范式"打法，找一个有用的重量级创新。
> **诚实结论先行**：没有挖到 beat-SOTA 的方法（通用 FM 即便 adaptation 也远不及 purpose-built 的 SOLIDER/Swin），**但得到一个完整、证据扎实、对领域有用的诊断结论 + 一个干净的 ×4 adaptation 发现**。负面/诊断结果本身有用——它量化并劝退了"把通用基础模型搬进 occluded ReID 就能赢"这个诱人但错误的直觉。

## 核心问题
能不能把当前 AI/CV 热点基础模型（MLLM / DINOv2 / SD扩散特征）搬进 occluded person ReID，做出超过现有 SOTA（Swin/SOLIDER 全栈 Occ-Duke 75.2 mAP）的重量级成果？

## 五个发现（每个都有干净实验证据）

**1. MLLM 推理：冻结大模型不会做被遮挡行人匹配，姿态提示也救不了。**
288 重遮挡 pair 同人判定：GPT-5.5(codex) 裸 55.9% / 文字提示 55.6%(无效)；Qwen2.5-VL-3B 裸 54.2% / 文字 49.3%(有害) / 视觉裁剪 35.8%(严重有害)。连最强模型也~56%(近瞎猜)。呼应 CAPTURe(ICCV'25)/O-Bench 对 MLLM amodal 推理缺口的实锤。→ MLLM-as-reasoner 路死。

**2. 冻结 FM 特征对 SOTA 没有独立信息（"假正交"）。**
DINOv2 pose-anchored part-MaxSim vs Swin MaxSim，重遮挡 oracle 检查：top-10 Jaccard 0.062，**P_dino_only 0.20%**(989 query 里只 2 个 DINO 命中而 Swin 漏)，**oracle 上界仅 +0.12 mAP**。低 Jaccard 是"假正交"——DINO 不是补充，是全局太弱(8.65 vs 72.57)。Swin 错的地方 DINO 也错。→ DINO⊕Swin 融合/re-rank 家族全死（两 agent 独立确认）。

**3. 瓶颈是"冻结"本身，不是模型新旧或对应质量。**
冻结部位匹配重遮挡 mAP：DINOv2-base 1.86 · DINOv2-registers 2.15(+0.29) · **DIFT(SD-v1.5) 0.73(更差)**。SD 特征强于 category 对应、弱于 instance ID 判别(SD-DINO 文献一致)。换更强/更新的冻结源都不破天花板。

**4. 极小参数 adaptation 解锁冻结特征 ×4（干净的正面发现）。**
LoRA 解冻 DINOv2-base(~1M 可训：0.6M LoRA + 0.4M 头，DINO 主体冻结) + 可微 pose-part-MaxSim：重遮挡 mAP **8.65 → 36.78**(e10，×4.2)，全部 14.61 → 44.67。rank32 36.72、large 待定。**证明瓶颈是 adaptation 不是特征**——这是 oracle"冻结无独立信息"的对照解释。

**5. 但通用 FM 即便 adaptation，单分支仍远低于 purpose-built SOTA。**
LoRA-DINO 单 pose-part 分支 **plateau ~37 heavy**(e5 34.92→e10 36.78，train acc 0.997 饱和)，对比 Swin **全栈** 72.57。即便 branch-to-branch 估计(Swin 裸 backbone+part ~65-70)，DINO-LoRA 仍明显低。原因：SOLIDER 是 human-specific 自监督预训练 + 全量微调；通用 DINO + 受限 LoRA 补不上这个差。

**6. ⭐ 判别性-互补性张力（headline 洞察）：adaptation 让 FM 既有用又"趋同 SOTA"，无法兼得。**
测"adapted-DINO 是否互补 Swin"（exp324h，adapted vs frozen 对比）：
| 指标(重遮挡) | frozen DINO | **adapted DINO** |
|---|---|---|
| DINO-only mAP | 8.65 | 36.78 (×4.3) |
| top-10 Jaccard(vs Swin) | 0.062 | **0.253**(×4，更重叠) |
| P_dino_only | 0.20% | 0.71% |
| oracle 上界 gain | +0.12 | **+0.59**(仍<+1) |
| 融合 best | — | ALL 75.53(+0.37)/HEAVY 72.83(+0.26)，w≥0.4 转负 |

**关键**：让 DINO 判别化(8.65→37)的同时，它和 Swin **更一致了**(Jaccard 0.062→0.253)。adaptation 把 DINO 推向 Swin-like 判别方向 → **判别性升、互补性降，无法兼得**。adapted-DINO 救的是 Swin 也快对的 case，不是 Swin 的系统盲点。融合 +0.37 是 NFC/RR 级 test-time 后处理(项目规则不算训练端贡献)，远非"beat 75"。
→ **冻结 FM 互补但无用(无信息)；adapted FM 有用但趋同 SOTA(失互补)。通用 FM 无论冻结/adaptation 都无法 beat 或 boost purpose-built SOTA。这是 fundamental 的判别性-互补性张力。**

**6b. ⭐⭐ 张力对显式干预鲁棒（exp324i decorr 对照，把张力从"观察"做成"打不破"）。**
直接攻击张力：DINO-LoRA 训练加跨网络跨协方差解相关损失，逼 DINO-global 与 frozen-Swin-global **线性无关**（Barlow-Twins 跨网络版，Codex 查无直接先例）。λ=0(无decorr) vs λ=1(decorr) e10 matched oracle：
| 指标(heavy) | λ=0 | λ=1 decorr | Δ |
|---|---|---|---|
| DINO-only mAP | 36.78 | 36.49 | -0.29 |
| top-10 Jaccard vs Swin | 0.253 | 0.2513 | **≈0** |
| P_dino_only | 0.71% | 0.81% | +0.10 |
| oracle 上界 | +0.59 | +0.58 | ≈0 |
| fusion best ALL/HEAVY | 75.53/72.83(+0.37) | 75.52/72.84(+0.37) | **≈0** |

**decorr loss 全程活跃(稳 0.041)却完全没移动 Jaccard/oracle/fusion。** 机制：强迫 global 线性解相关对"排哪些 gallery"(part-MaxSim 排序)是**正交**的——检索由 part-MaxSim over 相同可见身体部位证据决定，两模型受**同一份可见证据**约束犯**同样的错**(Swin-only-r1-hit 370/989=37%，DINO 补 8=0.81%)。global 线性相关只是排序"装饰"。→ **显式施压也打不破张力 = 张力鲁棒、fundamental**。这是张力洞察最强的对照证据(诊断论文核心实验)。

**收敛点(e30 matched)双确认**：λ=0 vs λ=1 同 rank16/seed/script 跑到 e30 oracle 仍**完全一致**——Jaccard 0.2646 vs 0.2627、oracle +0.85 vs +0.80、fusion best ALL 75.74 vs 75.73（λ=1 甚至略低）。**早期(e10)+收敛(e30) 双证据：解相关在任何训练阶段对互补性零效果。** 加上 decorr-floor 证据（λ=2 双倍权重只把 0.041 降 ~1% → ~0.041 是 ID-constrained floor，共享判别方向是 ID load-bearing 的、删不掉）。→ **张力对显式干预(e10/e30、λ∈{0,1,2}、λ=10 进行中)全程鲁棒，是 fundamental 的，不是可调超参。**

## 机制副发现（有用）
- **姿态锚定是关键**：冻结时均匀网格 part-MaxSim 几乎不涨(0.67 vs 整图 0.55)，只有 pose 锚定涨(1.86)——涨点来自"用姿态把 dense token 约束到身体部位语义"，不是 trivial 分部位。单变量隔离干净。
- **part-MaxSim > 整图 cosine** 在冻结和 adaptation 后都成立 → pose-part 机制不冗余。

## 总判断 / 有用贡献
**"搬通用 FM 进 occluded ReID 赢 SOTA"这条路，本夜证据判为不通**（generic FM ≠ purpose-built human FM）。但产出三个对领域有用的东西：
1. **诊断/负面结论**（劝退 + 量化）：MLLM 不会遮挡推理、冻结 FM 无独立信息、换源无用——省别人踩坑。
2. **×4 adaptation 发现**："冻结基础模型对 occluded ReID 无用，但 ~1M 参数 pose-anchored adaptation 解锁 ×4"——bottleneck-is-adaptation 的干净论点（Codex 称 LoRA-DINO+可微 pose-part-MaxSim 组合无直接先例）。
3. **可复用诊断工具**：rank-disagreement oracle（0-GPU 判两表征是否互补 + motivation 图）、按可见度子集拆 mAP。

## 诚实局限 / 未尽
- DINO-LoRA 是**单 pose-part 分支**，没上 Swin 主线的 PSG/GCN/LGPA 全栈（跨环境无单一 env 同时有 mmengine+transformers，没硬上）。"DINO-LoRA+全栈能否逼近 75"未测（大工程，留 future work）。
- 没测 full-finetune（vs LoRA）、没测 human-pretrained FM（那才可能匹敌 SOLIDER）。
- **large(DINOv2-large+LoRA) e5/e30 待定**——若 large 意外冲高(>50)，结论上调；若也 plateau ~40，确认通用-FM-adaptation 路封顶。

## 待补（实验跑完）
- [x] base-r16（killed e20=38.69 heavy）/ rank32（**e20=40.58 heavy / 48.89 all**，e25/e30 收尾中）→ **plateau ~40 heavy / ~49 all 确认**（远低于 SOTA 60-72）
- [ ] large e5/e30（capacity 对照，~30min/ep，hyy GPU0）：large 也 plateau ~40 则 capacity 非瓶颈，坐实"机制/问题"才是瓶颈
- [ ] **exp324i（解相关感知 DINO-LoRA）跑完** → 若 λ=1 fusion 真超 Swin = 张力被打破=真 method；若不超 = 张力升级为"显式施压也打不破"的强结论（本研究关键对照）
- [ ] 跑完后并入 results.md / decisions.md

## exp324i 进行中（张力的对照实验）
直攻"判别性-互补性张力"：给 DINO-LoRA 加跨网络跨协方差解相关损失（逼 DINO-global 与 frozen-Swin-global 线性无关），λ=0 vs λ=1 单变量。判 fusion(decorr-DINO⊕Swin) 能否超 Swin。这是把张力从"观察"做成"显式干预也无法打破"（或被打破=真 method）的决定性实验。详见 `exp324i/design.md`、`overnight_innovation_log.md`。
