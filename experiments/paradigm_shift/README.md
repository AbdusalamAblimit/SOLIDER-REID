# 范式转向（Paradigm Shift）——2026-06-26

## 触发

用户授权重大转向（2026-06-26 深夜）：
> "全面转得换量级（自己训一个新预训练范式、或上新的大规模监督信号）。不要 cheap 实验，可以不 cheap，可以花时间！可以做任何范式级别的创新。如果有必要改实验规则 claude.md 和 rules 那就改！"

## 背景：为什么转向

LM-ReID(exp359) 6.5 是最强 B 类候选，但训练端探索（守 SOLIDER/Swin 强 backbone 加小模块 + 历史模块）已 **100% 实测穷尽**。诚实结论：**不是"ReID 没训练端创新"，而是"在不换 backbone、不换量级、cheap kill-switch 的约束下，小修小补探到底了"**。真正的训练端创新是"范式级"的（换预训练/监督/数据来源）——CLIP-ReID(language)/SOLIDER(SSL pretraining)/Pose2ID(generation) 都是。用户授权松开所有约束去够这个量级。

## 算力现实（诚实边界）

4 个单卡 slot（RTX 4090 24G / 3090 24G / 5060Ti 16G ×2）。**无 foundation-model-from-scratch 算力**（训不了 LUPerson 4M images + 几十卡）。够得着的范式级动作：
1. **生成式数据引擎**：diffusion(SD/ControlNet) + pose/SMPL 控制造大规模 ID-consistent 训练数据
2. **新自监督预训练 pretext**：从 SOLIDER/DINOv2 权重 continued-pretraining（省算力，非 from-scratch）
3. **新监督信号 / 跨界范式 import**

## 规则变更（本转向生效）

- ❌ 不再要求 cheap kill-switch 优先（那是穷尽阶段的纪律）
- ❌ 不再"连续负结果→止损→收敛"（范式级创新允许长周期、允许失败重来）
- ✅ 允许多日/多周训练、允许 from-checkpoint continued-pretraining、允许造数据
- ✅ 仍守：动手前查 novelty（避先例）、full fine-tune/backbone 训练前 codex 审 diff、文档先行、诚实报告
- ✅ 死区仍避：SMPL 几何无独特 ReID 信号(多次证负)/FM-import MLLM-DINO-SD(判别性-互补性张力证负)/test-time trick 当主创新

## 进度

- [✅] 范式 gap analysis：3 路 codex（A 生成 / B 预训练 / C 自由）。三路 + 项目 exp109 oracle 收敛到根问题=**single-image support incomplete**。
- [✅] 终局对比（codex `decision_tscd_vs_intruder.md`）：T-SCD **5.0 否**（撞项目自己 `fgeu_realizability_result.md` tracklet 只恢复 oracle 16.3% + MVI²P/UMTS/VKD 先例）；**Intruder Identity Suppression 7.0 选定**。
- [✅] design `experiments/exp360_intruder/design.md`；CLAUDE.md 已改（范式转向 + codex 分工铁律落最高优先级段）。
- [进行中] **build exp360 Intruder**：阶段 0 地基机制验证（H1 donor-ID 泄漏可测 + H2 泄漏 vs AP drop 正相关，frozen probe）→ 阶段 1 小训验 H3 → 阶段 2 全量+多数据集+消融（deep work）。

## codex 分工（用户 2026-06-26 明确指令）

调研（novelty/gap/文献）+ 审查（code review/diff）→ **codex**；build（写码/训练/造数据/debug/迭代/决策/文档）→ **Claude**。本转向的 3 路 gap + 终局对比 + fgeu 翻档全是 codex，省了我两三周撞墙（T-SCD）。

关联 memory: [[autonomous-exploration-mandate]] [[fm-import-occluded-reid-closed]] [[historical-module-total-account-trap]]
