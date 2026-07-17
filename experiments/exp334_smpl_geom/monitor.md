# exp334 监控

## 配置
- 机器：lab-3090-d（ControlMaster 持久连接，conda env solider-reid）
- 两臂顺序：`exp334_geom`（--use_geom on）→ `exp334_control`（off）。同 config(exp333_vit_base_smpl.yml)/seed=1234，120 epoch，EVAL_PERIOD 10，TEST.IMS_PER_BATCH 64。
- log `/tmp/exp334_train.log`。

## 审查
- Claude broad review PASS（代码正确，仅报告口径修正）；Codex v1 needs-attention（**抓到翻转不同步 High bug** + bn_body 污染 + 口径）→ 全修 → Codex v2 approve。smoke 端到端跑通。

## 判据
- **headline = geom-on best-alpha mAP vs control(geom-off) mAP**（control ≈ exp333_baseline 53.09）。
- alpha=0 是诊断量（body loss 对 backbone 的正则），非 baseline。
- **重遮挡子集单列**（location≠visibility 风险在此最显形）。
- 诚实预期：location≠visibility（遮挡 patch 被 body-pool 进遮挡物 token）可能令其中性。

## 参考：exp333_baseline 曲线（geom-on best-alpha 对标这条看趋势）
| ep | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mAP | 41.07 | 45.52 | 46.81 | 49.95 | 50.64 | 50.75 | 51.54 | 52.48 | 52.83 | 52.88 | 53.02 | 53.09 |

## 进度记录
### [07:34] 启动 geom-on 臂

### geom-on eval 曲线（mAP）
| epoch | a=0(cls) | a=1.0 | a=2.0 | baseline参考 | a0−baseline |
|---|---|---|---|---|---|
| 10 | 43.17 | 38.07 | 34.06 | 41.07 | **+2.10** |
| 20 | 38.40 | 34.14 | 30.50 | 45.52 | **−7.12** |
| 30 | 41.07 | 37.00 | 33.06 | 46.81 | **−5.74** |
| 40 | 46.48 | 42.12 | 37.26 | 49.95 | **−3.47** |
| 50 | 47.94 | 43.70 | 38.50 | 50.64 | **−2.70** |
| 60 | 48.39 | 43.94 | 38.64 | 50.75 | **−2.36** |

### [08:17] e60 后 kill — 判决 = NEGATIVE
- gap 序列：+2.1(e10,噪声) → −7.1(e20,峰值LR伤) → −5.7 → −3.5 → −2.7 → **−2.36(e60)**。恢复**急剧减速**（deltas +2.3,+0.8,+0.34）→ **plateau ~−2.3，不会追平 baseline**。
- **best 永远是 alpha=0**（test 加 body 特征始终更差）；alpha=0（被 body-loss 正则的 backbone）收敛在 baseline 之下 ~2。
- **结论**：SMPL 几何 body-pool **两头都伤**——test 特征伤（location≠visibility，池进遮挡物 token）+ 训练正则也伤（噪声梯度损 backbone）。**location≠visibility 经验证实**。
- kill geom-on（plateau 负，e120 不会翻）；control 不需要（geom-on << 53 参考）。lingering D-state worker 无害，GPU 0% 空闲。

## exp334 最终判决
**NEGATIVE**。SMPL 完整-身体 2D 关节当空间先验：best(alpha=0) ≈ baseline−2，不涨反伤。继 exp333(β 特征=随机) 后**第二个 SMPL 经验失败**。
（gap +2.1→−7.1→−5.7→**−3.5**：body-loss 在 LR 峰值伤 backbone，过峰后**稳定缩小**恢复。e10 +2.1 可能不是纯噪声=正则信号被峰值不稳淹没。**不 kill**（避免过早结论）；body 特征(alpha>0)仍伤，但 alpha=0（被正则 backbone）在追平。e120 converged 结果待定：中性 or 微正。盯 e60/80/120。）

- **关键模式**：alpha>0（test 融合 body 特征）变差（location≠visibility，body-pool 进遮挡物 token，同 β 失败）；**但 alpha=0（cls-only）比 baseline +2.1** → body-loss 对 backbone 的**训练端正则**有益。
- 若持续 → SMPL 几何当**训练端辅助**（LGPA-D 式）有效，测试丢掉 body 分支、只用被正则的 cls。**全 SMPL 线首个正信号**。
- ⚠️ e10 噪声大，+2.1 可能是方差。真判据 = geom-on alpha=0 是否**全程压住 baseline 曲线**。盯 e20/30/60/120。
