# exp129 监控

## 实验信息
- 方法: Residual-Correction SCRD
- 类型: 训练端单变量改进
- 运行位置: 本地 3090
- 主配置: `configs/occluded_duke/pose_psg_gcn_pair_top_resid_scrd.yml`
- 核心变量: 相对 `exp125` 只新增 `POSE_CSRD_TARGET_MODE = 'residual'`
- 直接对照: `exp125`
- 次对照: `exp123`, `exp120`

## 启动记录

### [2026-03-20 18:35] 启动前记录
- 启动原因:
  1. 用户已明确否定继续试 `freeze`；`exp128` 已手动停止
  2. `exp127` 已说明 feature-level direct completion 不是当前主突破口
  3. `exp120/123/125` 的共同缺口是：support-complete teacher 的新增 correction 很可能被完整 teacher target 稀释
- 当前判断: 待启动
- 原因:
  - `exp129` 是相对 `exp125` 的下一跳：保留当前最强的在线 relational 主线，只改 distillation target，从 full teacher 改成 residual correction

### [2026-03-20 18:22] 启动确认（本地 3090）
- 运行位置: 本地 3090
- 输出目录: `log/occluded_duke/exp129_pair_top_resid_scrd`
- 配置: `configs/occluded_duke/pose_psg_gcn_pair_top_resid_scrd.yml`
- 关键确认:
  1. 主训练进程已成功启动：
     - `python -u train.py --config_file configs/occluded_duke/pose_psg_gcn_pair_top_resid_scrd.yml`
  2. `support-complete teacher` 仍为在线版本：
     - `[CSRD-ST] enabled: ... stop_epoch=-1`
  3. residual target 已正确接线：
     - `[CSRD-TARGET] mode=residual`
  4. `delta_top` pair routing 保持不变：
     - `[CSRD-PW] mode=delta_top, alpha=1.0, top_ratio=0.25`
- 当前判断: 继续
- 原因:
  - 当前已经通过“配置是否按设计生效”的第一关；接下来只需盯 early warmup 稳定性与 `epoch 21+` 的新统计

### [2026-03-20 18:24] 检查点 #1 — Epoch 1-2 warmup 前段
- 当前局部训练状态:
  - `Epoch 1 done`: `59.8s/epoch`
  - `Epoch[1] Iter[200/227] Loss: 15.486, Acc: 0.001`
  - `Epoch[2] Iter[160/227] Loss: 11.227, Acc: 0.004`
- 当前观察:
  1. warmup 前段曲线与 `exp123/125` 同量级，没有新增不稳定
  2. residual target 没有污染默认 early training，因为 `epoch<=20` 时 `CSRD` 尚未激活
  3. 当前阶段最重要的确认不是指标，而是：
     - target 改动已接上
     - 早期收敛没有被破坏
- 当前判断: 继续
- 原因:
  - 真正有信息量的节点仍是 `ep10 / ep20`，以及 `epoch 21+` 后 `csrd / csrd_tr / csrd_gr / csrd_psr`
