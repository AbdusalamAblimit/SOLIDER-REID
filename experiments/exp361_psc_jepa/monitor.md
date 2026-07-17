# exp361 PSC-JEPA — monitor

## Stage-A continued-pretrain（2026-06-26，4090，50 ep）
- 健康收敛：L 3.0→0.13，**tok_std 0.49→0.998（C1 防坍缩成功，全程不坍缩）**，cos_drop 0.95，var hinge 满足。
- ckpt pscjepa_10/20/30/40/50.pth（205 keys，backbone. prefix，H1 修复生效）。
- smoke 抓到 SOLIDER swin `.train()/.eval()` 不返回 self 的运行时 bug（审查覆盖不到），已修。

## Stage-A fine-tune kill-switch（2026-06-26，4090 PSC-JEPA vs 3090 plain，Occ-Duke 120ep，同 config 只差 PRETRAIN_PATH）

**★负结果（同 epoch 对比）**：
| epoch | PSC-JEPA mAP | plain mAP | 差 |
|---|---|---|---|
| 10 | 15.9% | 33.1% | **−17.2** |
| 20 | 27.8% | 42.8% | **−15.0** |
| 70 | 39.9%（平台）| — | — |

PSC-JEPA epoch 70 才 39.9%（平台），plain epoch 20 就 42.8%（趋势更高）。**continued-pretrain 让 fine-tune 更差，不是更好。**

**诊断：Stage-A 裸 continued-pretrain 破坏 SOLIDER backbone 判别性（catastrophic forgetting）** —— Stage-A **无 L_solider_anchor 防遗忘**（design 里 Stage-B 才加）。partial-view JEPA 把 backbone 从 ReID 判别表征拉偏，fine-tune 50+ ep 拉不回。

**★final 确认（2026-06-26）**：PSC-JEPA **120ep final mAP = 41.0%**（已 ENDED，平台）；plain @ epoch 60 已 **52.9%**（还在涨，120ep ~55%）。差 **−12 且会更大**。

**结论**：kill-switch Stage-A **FAIL（严重）**（PSC-JEPA 41.0 << plain ~55，差 ~−14，远不是 ≥+0.7）。**catastrophic forgetting 坐实** = 裸 continued-pretrain（无防遗忘）严重破坏 SOLIDER 判别性，fine-tune 拉不回。但**诊断清楚 = forgetting**，design 预期内（Stage-A = 骨架/防坍缩验证，不主张 novelty；Stage-B 才防遗忘 + support bank）。不是死路，是诊断明确的迭代。

## Stage-B 修复方向（防遗忘 + 真 novelty）
1. **L_solider_anchor（防遗忘，关键）**：frozen SOLIDER backbone（swin_tiny.pth 不更新）= anchor teacher；student 可见区 part token 蒸 frozen SOLIDER token（cos，gvis 掩码 visible）。锚住可见区判别性不遗忘，JEPA 只在 dropped 区学 completion。
2. **pseudo same-ID support bank（B 类 novelty）**：T_bank 同 ID NN 的 body-part prototype，dropped 区预测 support。
3. 重训 continued-pretrain（3 backbone：student + EMA teacher + frozen SOLIDER anchor）→ fine-tune 再验 kill-switch（≥+0.7 vs plain）。

## Stage-B 重训 + fine-tune（防遗忘 v2，2026-06-27）

- **continued-pretrain 50ep 健康**：防遗忘 sol_p 0.6→0.11 / sol_g 0.05→0.03 活跃，tok_std 不坍缩，L 收敛。codex 三审（R1 抓"只锚 5 part token 覆盖窄"→补 global GAP distillation→R2/R3 approve）。
- **★fine-tune early signal（epoch 10）**：Stage-B **23.0%** vs Stage-A 15.9%（防遗忘 **+7.1**，机制部分生效）vs plain 33.1%（仍 **−10.1**，没完全修）。
- **诚实判读**：防遗忘 anchor（part + global GAP）**减轻 forgetting 但不充分**——continued-pretrain 仍损害判别性。完全符合 codex 守的诚实"修复尝试成立非 forgetting 已解决"。
- 可能原因：anchor 权重不够 / part+global GAP 还不够（codex 提 dense/stage-wise distill）/ partial-view JEPA 与 ReID 判别性本质张力。
- 待：epoch 50 趋势（追平 plain or 平台卡 < plain）→ final 判 kill-switch。趋势平行 plain（差固定 −10）=防遗忘不够；收敛追平=够。

## ★PSC-JEPA continued-pretrain 主范式判死（2026-06-27，codex 诊断 8/10）

**kill-switch FAIL**：Stage-B 防遗忘 fine-tune 趋势 23→36.7→44→46.2→46.7（epoch 10-50 平台）<< plain 58.5（−11.5 平台差）。防遗忘缩 early gap（−10→−2）但**没改平台**。

**codex 诊断（非代码问题，范式本质冲突）**：
1. partial-view JEPA 补"**不可观测身份细节**"（被遮 part 真实 identity 单图不存在）→ 学生学上下文均值/人体先验 → 特征推向"可预测平滑不变"，而 ReID 要"细粒度可区分"（纹理/颜色/局部差异）。**目标方向冲突**。
2. continued-pretrain 覆盖 SOLIDER 已调好的 appearance/semantic 平衡，anchor 拉不回全部 dense feature/attention/层间几何/margin。
3. 防遗忘越强 JEPA 越 no-op（终点≈原 SOLIDER 不超 plain）。
- 外部先例支持：TransReID-SSL(Catastrophic Forgetting Score) / PersonMAE/HAP(需完整预训练体系) / continued-pretrain 普遍报 forgetting。

**结论：不再 pretrain backbone**。尸检 4-run mini grid（救回 2/10，跳过）。
**Pivot（codex）**：① support bank 改 fine-tune 侧 **detached auxiliary**（5.5，plain SOLIDER fine-tune + support 只监督轻量 completion head/part branch/pair scorer，global backbone hard guard ≥plain−0.3，避 pretrain harm 保 exp109 信号）② **生成数据引擎**（6.5，扩训练分布让监督 fine-tune 自学遮挡鲁棒，不蒸不可观测 support，更长期范式空间）。
