# LGPA 自有化 Goal 逐项闭合审计

日期：2026-07-14

## 审计结论

该 Goal **不能标记为完成**：目标结果“把 LGPA 改造成可投稿的自有方法”没有实现。与此同时，Goal 自身预注册的 kill-switch 已被正式数据触发，因此继续 IPER、student 或相邻小变体会违反原目标，而不是推进原目标。

## 要求与证据

| Goal 要求 | 当前证据 | 状态 |
|---|---|---|
| 保留 detached pose-localized local descriptors 与 `global+parts` 性能资产 | exp336 三 seed约 `+0.9 mAP`；Gate B s0 target-only 相对 global `+0.8213 mAP` | 已证明 |
| CLIP 文本、GCN、matching、普通 pose KD、pose token、slot/write-back 不作为创新 | `design.md`、`innovation_brainstorm.md`、`story.md` 均已明确排除；PBSR 已 NO-GO | 已完成 |
| correct/canonical/shuffled/uniform/no-pose 归因 | 同 checkpoint Gate B 全部完成，correct 只比 shuffled/canonical 高 `0.0320/0.0984 mAP` | 已完成，且反驳实例 pose 主效应 |
| frozen-random/learned query 归因 | canonical CLIP `59.5/68.1`、fixed-random `59.9/68.7` 已足以移除 CLIP 语义；correct-pose random-frozen/random-learned 未执行 | 未完整执行；主 kill-switch 失败后不再消耗资源 |
| `5376D→768D` 可行性 oracle | fixed JL retention=`-0.2245`；train-only PCA retention=`1.1158`，train/eval path overlap=`0` | 单 seed可行性已完成 |
| 近邻查新 | PAFormer、UMTS、MVI²P、MVCD、MHSF、PGFL-KD、TSD、NNCL 等两轮审计已落盘 | 已完成；外部新颖性仍受限 |
| IPER correct-effect 领先最强 control至少 `0.5 mAP` | Gate B correct−shuffled=`+0.0320 mAP`，correct−canonical=`+0.0984 mAP` | **失败，触发停止** |
| pose-free student 恢复旧增益至少 80% | 只有上游 correct-effect 过门才允许执行；上游已失败 | 按预注册规则禁止执行 |
| frozen support 的 pose-specific routing 成立 | Gate C POSE-RESP−PART-EQUAL=`-0.0766 pp`，五折全负，bootstrap CI跨零；scene五折全负 | **失败，CASD NO-GO** |
| 完整单 seed、e60、三 seed、ResNet、ViT、多数据集 | 只有前置门禁通过才允许 | 前置失败，禁止执行 |
| 最终形成可投稿自有方法 | IPER、PBSR、CASD 三条正交机制均未通过预注册门禁 | **未实现** |

## 阻塞性质

这不是代码、算力或外部资源阻塞，而是目标内部的科学停止条件已经成立：

1. 实例级正确 pose 没有相对强 controls 达到 `+0.5 mAP`；
2. pose-response 跨实例 routing 也没有独立价值；
3. Goal 明确禁止失败后转 OT/MoE/slot/温度/超参小变体；
4. 普通 same-ID multi-view support 又不能绕过 UMTS/MVI²P 的外部邻近性。

因此当前没有既遵守 Goal 又能继续逼近其成功终态的动作。把主线切到 PSG-only 论文、LM-ReID 或新的问题定义，都属于**新目标**，不能用来把本 Goal 伪装成完成。

## 状态纪律

- 不标记 `complete`；
- 不恢复 IPER/PBSR/CASD 训练；
- 不把 LGPA 的性能增益改写成自有创新；
- 只有用户明确建立新的论文/PSG目标后，才按新范围继续。
