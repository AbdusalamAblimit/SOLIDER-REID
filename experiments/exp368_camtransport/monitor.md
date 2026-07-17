# exp368 Camera-Pose Transport — monitor

## cheap kill-switch（frozen SOLIDER Market exp260b, 零训练, 2026-06-28）—— DEAD

| | mAP | R1 | Δ vs baseline |
|---|---|---|---|
| baseline cosine | 94.43 | 97.15 | — |
| camera-centering | 94.41 | 97.15 | -0.03 |
| transport | 93.86 | 96.76 | **-0.58** |

按 baseline #false@10 分桶 ΔAP(transport-baseline):
| bucket | n | baseAP | ΔAP |
|---|---|---|---|
| [0,0.5) 易 | 2929 | 97.11 | -0.47 |
| [0.5,0.9) 中 | 383 | 82.12 | -1.28 |
| [0.9,1.0) 难 | 56 | 38.62 | -1.26 |

★**DEAD**: transport Δ=-0.58（掉，成功线 +0.5 反向），所有桶都掉（难桶 -1.26 更多，非 trivial）。camera-centering Δ=-0.03（几乎不变）。

★Why: **SOLIDER 特征已 camera-invariant**（camera-centering Δ-0.03 = camera bias 已极小，去 cam mean 无用）→ codex 假设"camera 间系统性 shift, transport 有用"前提错。transport（cam pair ID-mean ridge map）反而引入噪声/过拟合，破坏判别特征（难桶掉最多）。

## 决定

Camera-Pose Transport DEAD（camera invariance 已够，transport 有害）。cheap kill-switch 干净快杀（frozen 零训练，省 Top3 训练版大投入）。转 codex Top2 Counterfactual Part-Contradiction（part swap 造 hard neg, cheap 验 feature swap 测 donor/target false）。记 memory（camera-aware transport 死: 强 backbone 已 camera-invariant, camera-conditional metric 无 headroom）。
