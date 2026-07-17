# exp360 — Intruder Identity Suppression（范式转向第一个 build）

## 选定经过（2026-06-26 深夜，用户授权范式转向后）

3 路 codex gap analysis → 2 候选终局对比（codex `decision_tscd_vs_intruder.md`）：

| 候选 | codex 综合 | 否/选理由 |
|---|---|---|
| T-SCD（tracklet support 蒸馏） | 5.0/10 ❌ | 撞项目自己的 `fgeu_realizability_result.md`（posetrack tracklet 每条≤2帧、同机位冗余，只恢复 oracle 16.3% < 40% 门槛）+ MVI²P/UMTS/VKD 先例 |
| **Intruder Identity Suppression** | **7.0/10 ✅** | 避开 completion/visibility/occluder-gate 死区；novelty 窄缝（KPR/QPM/DPEFormer/OGFR 相邻但无"显式 donor-ID 泄漏可测+训练对抗+测试单图"直接同构） |

## 动机 / 问题重定义（核心）

遮挡 ReID 的根症结，**不是**"target 信息缺失要补全"（completion / support-complete 这条线已反复证负：exp109 墙、fgeu 16.3%、各种 feature completion 小残差），**而是**：

> **遮挡物（尤其另一个行人 donor）把 donor 的身份信息泄漏进了 target 的 embedding，污染了检索。要做的是 source separation——把 donor-ID 从 target embedding 里分离/压制掉。**

这是"换问题定义"的范式动作，不是在强 backbone 上加模块。

## 核心假设（可证伪）

H1: 行人遮挡的 target crop，其 embedding 里**可测地**含 donor-ID 信息（donor-ID probe 显著 > 随机）。
H2: donor-ID 泄漏量与 target 检索错误**正相关**（cos(f_mix,f_donor) − cos(f_clean,f_donor) 越大，AP drop 越大）。
H3: 训练时显式压制 donor-ID（对抗）能降低泄漏，**且**降低泄漏带来真实人遮挡 split 的 ReID 提升（不是只压表征不涨点）。

## Pipeline（codex 终选）

1. 合成样本：target 图 + donor 行人 crop/mask（donor 有已知 PID），遮挡比例分档。
2. 三路 forward：clean target `x_t` / intruded `x_t+d` / donor alone `x_d`。
3. 主 ReID loss：`x_t`、`x_t+d` 都用 **target** PID 做 CE/triplet。
4. clean-occluded consistency：`f(x_t+d)` → `stopgrad(f(x_t))`，保护 target identity。
5. donor suppression：在 `f(x_t+d)`（或 residual `f(x_t+d)−f(x_t)`）接 donor-ID classifier + **GRL**；加 margin/contrastive 让 mixed feature 不靠近 donor feature。
6. 测试：纯单图，去掉 donor head，无外部信息。

## Build 阶段（deep work，非 cheap-收敛）

- **阶段 0（地基机制验证，先做）**：frozen strong baseline 上验证 H1+H2——合成 target+donor，测 donor-ID probe acc vs 随机、测泄漏量 vs AP drop 相关性。**这是 build 地基不是 cheap 逃避**：H1/H2 不成立则对抗压制无的放矢，需调整 donor 合成或换机制变体。
- **阶段 1**：小规模训练验证 H3（donor suppression head + GRL + consistency），20-40 epoch 看 donor probe 是否降 + 真实人遮挡 split 是否涨。
- **阶段 2（成立则）**：全量训练 + 多数据集（Occluded-Duke/Occluded-ReID/Market/MSMT）+ 消融（GRL on/off、consistency on/off、donor margin）+ 迭代。deep work，允许多日训练。

## Kill-switch / 风险（codex）

- **头号风险**：donor-ID probe 证明泄漏存在，但压掉后 ReID 不涨 = 退化成 **target ambiguity 墙**（项目 memory `post-prcv-directions-exhausted` 记该方向红蓝队判死过；本方向靠"训练端 source separation + 可测泄漏"区别于 KPR test-time prompt，窄缝）。
- **硬阈值**：donor 泄漏下降 ≥30% 但真实人遮挡 mAP 没 ≥+0.5 → kill（说明压泄漏不是有效 ReID 机制）。
- **对照**：matched synthetic-only baseline（同样合成数据但不压 donor）、Market/物体遮挡 split（不该涨，涨了说明是通用增强非 source-separation）。

## 与死区的关系（诚实标注）

- target ambiguity（memory 判死）相邻 → 靠"训练端可测 donor 泄漏 + 对抗压制"区别，阶段 0 的 H1/H2 就是验证这个区别真实。
- 不碰：SMPL 几何 / FM-import / test-time prompt(KPR) / completion 残差。

关联：`experiments/paradigm_shift/`（README + 3 路 gap + 决策）、memory [[post-prcv-directions-exhausted]] [[exp109-headroom-is-a-wall]] [[fm-import-occluded-reid-closed]]。
