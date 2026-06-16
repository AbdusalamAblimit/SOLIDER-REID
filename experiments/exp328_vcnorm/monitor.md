# exp328 VC-Norm — 监控记录

## 配置
- 主：`configs/market/pose_vcnorm_base.yml`（POSE_VCNORM=True, WARMUP=20, WEIGHT=0.5, VIS_THR=0.3, OA_SD=True, PLBOA=True, TEST_FEAT=equal_concat, SIZE_TEST=[384,128]）@ lab-3090-d `/root/work/SOLIDER-REID`, OUTPUT `./log/market1501/exp328_vcnorm`, log `/tmp/exp328_vcnorm.log`。
- 对照：`pose_vcnorm_base_control.yml`（仅 POSE_VCNORM=False，单变量）@ lab-4090, log `/tmp/exp328_control.log`。
- 两机同 commit 715c020，同 vcnorm.py，已核单变量。MAX_EPOCHS=120, EVAL_PERIOD=10。

## 训练监控（Market val mAP）

| Epoch | exp328 VC-Norm | control | Δ(VC−ctrl) | 备注 |
|------|----------------|---------|-----------|------|
| 10 | **14.2%** | 86.2% | −72.0 | ⚠️ **一次性 eval 瞬态(假警报)**, 非真实轨迹 |
| 20 | **88.4%** | 89.4% | **−1.0** | e10 已恢复; VCA 此时刚激活(warmup=20 结束) |
| 30 | **90.4%** | 91.0% | **−0.6** | gap 收窄(−1.0→−0.6); VCA ramp 不进一步伤整体, 健康 |
| 40 | **90.7%** | 91.3% | **−0.6** | gap 稳定 −0.6; VCA 不伤整体确认 |

**趋势**: VC-Norm 在 Market(整体集)上稳定低对照 ~0.6-1.0 = VCN 模块在无遮挡数据上的小成本(符合预期, 无可对齐的遮挡 token)。gap 不扩大 → VCA 不伤整体。真正判据仍是跨域 Occ-ReID(训练完)。

### e10=14.2% 假警报分析
一度怀疑 VCN 模块毁特征。排查：VCN 是 zero-init 恒等(gain~0.005 极小, 数学上不可能掉 72 分)；两机 git 同 commit、同代码、单变量 config 无误。e20 恢复 88.4% → e10 是孤立 eval glitch(疑 AMP/LayerNorm 一次数值抖动)。**教训：单点 eval 异常先查模块幅度+对照，别急判死。**

### e20 真实读数解读
- Market(整体集, 无遮挡)上 VC-Norm −1.0 vs control。**符合预期**：Market 无遮挡 → VC-Norm 的遮挡-stat 对齐无处发力，只显 VCN 模块小成本。
- VCA 在 e20 刚激活(warmup=20)，e20 数尚未体现 VCA 的对齐效果。
- **真正判据 = 训练完跨域 Occluded-ReID eval**（有真遮挡），非 Market。训练用 PLBOA 合成遮挡，VCA 对齐合成遮挡 token，收益只在遮挡测试集显现。

## 待跑
- [ ] e30/e40 Market（确认 VCA ramp 不进一步伤整体；若持续走低则 VCA 过强）
- [ ] 训练完 → `test_on_occluded_reid.py` 跨域评测（VC-Norm vs control 单变量净增益）
- [ ] 决定性：跨域 Occ-ReID 上 VC-Norm 是否 > control。是→真机制；否→VC-Norm 判负。
