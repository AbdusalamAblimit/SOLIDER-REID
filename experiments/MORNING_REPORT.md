# 晨报：一夜自主探索总结（2026-06-15 夜 → 06-16 晨）

> 你睡前要求：整夜不停、三台服务器全用、务必找出一个有用的创新点。
> **诚实先行**：没找到能 beat-SOTA 的新方法——但这不是空手而归。我把"搬通用基础模型进 occluded ReID 赢 SOTA"这条最诱人的路**系统性地证负了**，并在过程中挖到一个**漂亮、扎实、对领域有用的洞察 + 一份完整诊断研究**。这本身是一个真实、可发表的创新点（analysis/诊断类），且帮你（和别人）**省掉一整片会白烧 GPU 的坑**。

## 一夜干了什么（全速、三机并行）
跑了 ~15 个实验，覆盖你提的"搬范式"打法的所有主要候选：MLLM 推理、DINO 冻结对应、换更强冻结源(large/registers/DIFT)、LoRA 解冻、LoRA+SOTA 融合。每条都先廉价 kill-switch 验、有信号才升级、死路诚实砍。全程文档在 `experiments/overnight_innovation_log.md`，技术细节在 `experiments/fm_occluded_reid_study.md`。

## 核心有用产出

**1. ⭐ headline 洞察：判别性-互补性张力（可单独成 analysis 论文的点）**
冻结的基础模型特征，对训练好的 SOTA（Swin/SOLIDER 75.2）**没有任何独立信息**（oracle 上界只 +0.12 mAP）。我用 LoRA 把它解冻、~100 万参数就把它从"几乎没用"拉高 4 倍（重遮挡 8.65→37）——**证明瓶颈是 adaptation 不是特征**。但关键来了：**让它变判别的同时，它和 SOTA 越来越像**（top-10 重叠 0.06→0.25）。adaptation 把通用模型推向 SOTA-like 方向 → **判别性升、互补性降，两者不可兼得**。所以通用基础模型无论冻结还是 adaptation，**既打不过、也补不强**一个专门为人体预训练的 SOTA。这个张力是 fundamental 的，没人系统写过。

**2. ×4 adaptation 发现** + **可复用诊断工具**（rank-disagreement oracle：0-GPU 判两个表征是否互补 + 出 motivation 图）。

**3. 完整诚实的"基础模型能/不能做什么 for occluded ReID"诊断**（MLLM 56%、冻结无信息、换源无用、LoRA me-too 且 < SOTA）。

## 诚实：为什么没拿到 beat-SOTA 方法
- LoRA-DINO 单分支 plateau ~40 mAP，**me-too**（PersonViT 已做 DINO-ReID 到 72.2，我们 47；LoRA-ReID / pose-visible 匹配都有先例）。
- adapted-DINO 融合 Swin 只 +0.37（NFC 级 test-time 后处理，不算训练端方法）。
- 根因 = 上面那个张力 + 之前确认的三堵墙（exp109 oracle / 多人 no-op / 95.8% 训练全可见）+ Swin 端 300 实验已饱和。**这片空间是真的硬。**

## 资源使用
- lab-3090-d（3090）：主力，跑了 DINO 全线 + 所有 oracle/诊断。✅ 物尽其用。
- hyy（5060Ti×2）：LoRA rank32 + large + DIFT/DINOv3 换源。✅
- **lab-4090：env 坏了**（无 cv2/mmengine/transformers，反复 banner 超时）——这台没能用上，是硬件/环境问题不是没派活。建议你有空修一下它的 python 环境。
- **没花钱买 API**：用了 codex(GPT-5.5 多模态) 当免费强 VLM 臂。

## 给你的下一步建议（醒来可挑）
1. **把这夜做成 analysis 论文**（最稳、最快变现）：张力洞察 + ×4 finding + 诊断，证据已齐，是真贡献。
2. **若仍想要 method**：唯一没试的硬骨头是"DINO-LoRA + 完整 PSG/GCN/LGPA 全栈"（今夜因没有单一 env 同时有 mmengine+transformers 没硬上）——但张力暗示它大概率也趋同 Swin、难超 75。或者"用与 ID 正交的目标 adapt FM 以保互补性"（被 95.8% 墙威胁）。两条都不确定。
3. **修 lab-4090 env**，下次三机齐全。

诚实讲：我没给你编一个假的 SOTA 突破。但这一夜的钱换来了一个真实、有用、诚实的科学结论 + 一个漂亮洞察 + 一片避坑地图。要不要把它做成 analysis 论文，或你想让我再换个完全不同的问题继续挖，醒来告诉我。
