# TAPF 论文主张—证据—缺口矩阵

> 口径冻结于 2026-07-17。所有性能数字均为 e120/final、seed 1234；只比较同一骨干内部的 matched
> B0/D0。跨骨干绝对值不可横比，单 seed 不得写成显著性结论。

## 一、可写主张

| 编号 | 候选主张 | 现有直接证据 | 当前口径 |
|---|---|---|---|
| C1 | 完整 `anchor+PSG` 在训练期使用姿态监督、推理期只读 RGB | Swin/ResNet/ViT 的 D0−B0 mAP 分别为 `+1.1/+3.1/+2.0` | 可写“跨三骨干描述性正证据”，不可写统计稳定或普适提升 |
| C2 | 该方法消除部署时外部姿态模型依赖 | 三骨干 final 的 external correct/shuffle/None/exploding descriptor exact parity 全部通过 | 可写严格 RGB-only inference；训练期仍需要离线姿态 teacher |
| C3 | D0 能在不读测试姿态的条件下保留原始 PSG 的实用收益 | Swin D0=`56.2/67.6/79.8/83.4`，external-pose R0=`56.1/67.4/79.5/83.7` | 可写“匹配原始 external-pose PSG 的描述性性能”，不可写等价定理 |
| C4 | 原子方法可迁移到不同归纳偏置的 backbone | Swin-T、ResNet-50、ViT-B 三个独立 matched B0/D0 | 可写 backbone transfer；必须同时展示各自 baseline，不能横比绝对值 |

## 二、必须收紧或禁止的主张

| 编号 | 过强主张 | 反证/边界 | 论文处理 |
|---|---|---|---|
| B1 | Hierarchical refinement 跨架构稳定优于单层 | HT0−D0 mAP：Swin/ResNet/ViT=`-0.1/+0.8/-0.3` | 只作 backbone-conditional 扩展或消融，不进 headline |
| B2 | 精确关节名称、confidence 或 geometry residual 各自带来检索收益 | permutation、teacher-off、residual-OFF 与冻结语义审计均不支持独立因果贡献 | 方法按完整 `anchor+PSG` 原子对象讨论，不拆子部件刷贡献 |
| B3 | 所有检索指标在所有骨干上都提高 | Swin D0−B0 的 R10 为 `-0.4`，其余骨干四项为正 | 主表完整报告 mAP/R1/R5/R10，不选择性隐去负项 |
| B4 | Video TAPF 是新的独立贡献 | GAE-Net、PAFormer、KPRTrack 与成熟 temporal memory/aggregation 已占直接近邻；无可用视频数据 | exp382 已 NO-GO，不下载、不训练、不写 headline |
| B5 | ViT 配置中的三个 G3 consumer 都有效 | post-block11 位于最后 CLS–patch 交互之后，final projection 全轨迹 `0/2 changed` | 只声称 post-block9/10 是有效 G3 consumer |

## 三、强先例差分

| 先例 | 已覆盖内容 | TAPF 只能争取的差分 | 风险 |
|---|---|---|---|
| PAFormer | pose heatmap 监督 pose-token attention、visibility teacher forcing、测试期 pose-free | 在 backbone 内学习 anchor/state，并用 PSG 直接调制后继视觉特征 | pose-free、pose token、visibility 都不能单独列为首创 |
| PGFL-KD / TSD | pose/parsing teacher 向普通 student 蒸馏，测试丢弃结构分支 | 结构监督不是只做 logit/feature KD，而是形成可审计的内部 spatial support→PSG 路径 | “privileged pose teacher”大叙事已不是空白 |
| KPR / KPRTrack | keypoint-prompted parts、共同可见部位比较、tracklet 同部位聚合 | TAPF 单图部署不运行外部 keypoint prompt；主对象不是 test-time part matching | 不能把部位对应或 pose-guided ReID 本身写成创新 |

## 四、证据缺口与优先级

| 优先级 | 缺口 | 当前状态 | 最小必要动作 |
|---:|---|---|---|
| P0 | 数据集/域覆盖 | 只有 Occluded-Duke 训练与测试 | exp383：fresh Market B0/D0，报告 Market 域内和 Occluded-ReID 跨域 |
| P0 | TAPF 专属参数、FLOPs、训练开销与推理开销 | 旧 `efficiency.md` 是其他历史方法，不能挪用 | 在 exp383 预检中对 B0/D0 同输入静态计数和 batch64 CUDA timing；单列离线 pose teacher 成本 |
| P1 | 随机种子稳定性 | 三骨干都只有 seed 1234 | exp383 后根据跨域结果决定是否只对论文主骨干补 2 个 seed；不为 HT0 补 seed |
| P1 | 方法与 PAFormer/PGFL-KD/TSD 的实现级差分 | 已有文献审计，但尚无 paper-ready related-work 对照表 | 基于正文/代码路径写固定差分，不以标题或名字判断 |
| P2 | 第二个原生遮挡训练集 | 远端没有 P-Duke 等可训练 split；Occluded-ReID 是 test-only | 不伪造“第二训练集”；若未来补数据，需单独许可与协议审计 |

## 五、下一算力决策

优先执行 exp383 的 Market B0/D0 两臂，而不是立即在 Occluded-Duke 重复多 seed。理由是同等两次训练
能同时回答“第二训练域能否复现”和“独立遮挡目标能否跨域受益”，信息量高于只重复现有域。若
exp383 的跨域与域内方向均不支持 D0，再补 Occluded-Duke seed 不能修复数据集泛化缺口；若方向
支持，再只对最终论文主骨干补必要 seed。
