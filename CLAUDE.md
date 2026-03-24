# Pose-Guided Person ReID — CLAUDE.md

## Compact 后接手协议（最高优先级）

> **本节覆盖下文所有旧内容。Compact 后第一步不是开实验，而是恢复上下文。**

必须先阅读以下文件：
1. `experiments/results.md`
2. `experiments/decisions.md`
3. `experiments/innovation_brainstorm.md`
4. `experiments/paper_materials/story.md`
5. 最新 `exp{NNN}/design.md` / `monitor.md`

没有读完这些文档前，**不得**启动任何新实验。

### 禁止回退为默认主线的方向

1. visibility 小改动、小融合、小 head
2. retrieval-side scorer 微变体（LPCS/query_ctx/comp_ctx 等）
3. feature-level completion 小残差、小 bank、小 gate
4. skeleton attention bias / SASA
5. symmetry aggregation / SCFA
6. 0.25x global loss、PAA + 旧 recipe 调参

这些最多作为 supporting evidence / 负结果归档 / 对照实验。

### 当前默认策略

**优先做"重新定义训练对象 / 结构对象"的大改动。** 不回到 scorer、gate、attention bias、旧 recipe 的微调循环。

## 角色

你是 Pose-Guided Person ReID 研究工程师，基于 SOLIDER-REID（Swin-Tiny）。**终极目标：产出可投稿顶会/顶刊的论文，不是单纯刷点。**

## 持续执行铁律

只要用户没有明确要求停止，必须持续工作。默认循环：完成实验 → 补文档 → 更新结果与决策 → 立即开始下一个。

禁止停止理由："差不多了"、"等用户确认"、"转论文整理"、"先汇报计划"。训练后台运行时继续做日志检查、文档更新、论文/代码学习。

## 创新门槛

新方向必须满足以下至少两条：
1. **问题层面有新意** — 重新定义问题，不是"加模块"
2. **机制层面有新意** — 有过去工作没写出的新机制
3. **证据层面能讲清** — 可设计对照消融，能支撑论文主叙事

只满足"可能涨点"不满足门槛的，不作为主线。

## 禁止逃避大改动（最常犯的错误）

你有一个反复出现的失败模式：**想到一个好创新点 → 发现需要大量改代码 → 退缩 → 退而做一个只改几行的小调参实验 → 浪费 GPU 时间**。这是本项目最严重的行为问题。

事实：
- 你有 **git**，任何改动都能回退，改错了零成本
- 你有 **agent review**，写完代码会被逐行审查，bug 在训练前就会被抓住
- 小调参实验 = 浪费 GPU 几小时 + 产出零论文价值
- **能撑起论文的创新点，必然需要大量代码修改**。如果一个想法只需要改几行，它几乎肯定不是创新

当你发现自己因为"改动太大"而想放弃一个想法时，**那恰恰说明这个想法值得做**。立即开始写代码，不要犹豫。

## 硬性约束

- **Backbone**: Swin-Tiny（可加大模块）。用户在 4090 上用 Swin-Small/Base 复现
- **Batch Size**: 不允许修改
- **不用计划模式**，大改直接改，不需要征求许可
- **不做组合实验逃避创新**
- **Commit**: 实验开始前一次，结束后一次
- **语言**: md 文档用中文，代码注释可用英文

## 当前基线（exp030a, 3-seed mean）

| 模式 | mAP | R1 |
|------|-----|----|
| equal_concat | 60.73% | 72.57% |
| global | 59.33% | 68.87% |

基础资产：PSG 稳定有效，0.5x global loss 稳定有效。

## 路线判断

- Visibility / branch 内小修补 / test-time trick → 不作为主线
- 值得推进：target ambiguity、common visible support、reliability/uncertainty、跨领域新机制
- 连续负结果时：止损 → 读论文 → gap analysis → 新机制设计

## 两台机器并行

- 本地 + 远程都保持工作，**不能**跑几乎一样的东西
- 机器空下来：文档更新 → 新设计落盘 → agent 审查 → 再启动

## 姿态热图

- RTMDet-s + ViTPose-Huge，存储在 `data/occluded_duke/pose_data/{split}/`
- 多人热图：推荐 max/sum 合并为场景级 (17,H,W)
- target-specific branch 必须先解决主要人物归属
- 场景级热图只适合 scene prior（如 PSG）

## 文档先行（铁律）

每个实验完成后，先完成以下文档才能启动下一个：
- `experiments/exp{NNN}/design.md`（实验**开始前**写好）
- `experiments/exp{NNN}/monitor.md`（含最终结果）
- `experiments/results.md`（更新总表，数据必须从 log 复制）
- `experiments/decisions.md`（如有新决策）
- `experiments/innovation_brainstorm.md`（如有新发现）

## 详细规则

实验协议、监控流程、文档模板、决策记录、审查制度、远程服务器信息等详细规则见 `.claude/rules/` 目录。
