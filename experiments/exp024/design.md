# 实验 exp024: PDS+StopGrad without PSG (消融实验)

## 动机
- exp023 (PDS+StopGrad) 取得了全实验最佳结果 (mAP 59.5%, +2.9% vs baseline)
- 需要验证 PSG 在 PDS+StopGrad 架构中是否仍然必要
- 论文消融表需要分离 PSG 的独立贡献
- 当前消融链缺少 "PDS+StopGrad 但无 PSG" 这一数据点

## 创新点 / 核心假设
- **核心假设**: PSG 是 PDS+StopGrad 架构中不可或缺的组件。没有 PSG，即使有 PDS+StopGrad 的梯度隔离，Global 分支也无法获得 pose 空间先验，性能会大幅下降
- **预期**: Global-only mAP ≈ 56.6% (baseline 水平)，因为 StopGrad 阻断了 Part 梯度，而 Global 分支没有 PSG，相当于标准 Swin-Tiny

## 技术方案
- 修改 `model/pose_dual_stream_model.py`: 添加 `POSE_GLOBAL_PSG` 配置项
  - 当 `POSE_GLOBAL_PSG=False` 时，`_run_global_branch` 跳过 PSG 门控
  - PSG 模块仍会被创建（不影响模型结构），但不参与计算
- 新增 `config/defaults.py`: `POSE_GLOBAL_PSG = True` (默认启用，向后兼容)
- 配置: `configs/occluded_duke/pose_pds_stopgrad_nopsg.yml`
  - 与 exp023 完全相同，唯一区别: `POSE_GLOBAL_PSG: False`

## 预期结果
- **global-only mAP**: ~56.6% (baseline 水平)
  - 理由: 无 PSG + StopGrad 阻断 Part 梯度 → Global 分支等同于标准 Swin-Tiny
- **part-only mAP**: 可能低于 exp023 part-only (56.7%)
  - 理由: 没有 PSG 增强的共享特征质量更差 → Part 分支获得的输入质量下降
- **如果 global-only 接近 baseline (56.6%)**: 证明 PSG 贡献 ~2.9% (59.5% - 56.6%)
- **如果 global-only 明显高于 baseline**: 说明 PDS 架构本身（即使无 PSG）有正面效果

## 对照组
- **直接对照**: exp023 (PDS+StopGrad+PSG, mAP 59.5%) — 唯一区别是有无 PSG
- **间接对照**: exp000 baseline (56.6%) — 如果 exp024 ≈ baseline，证明 PSG 是全部提升来源
- **间接对照**: exp007 PSG-only (58.3%) — 对比 PSG 在单分支 vs 双分支架构中的效果

## 消融变量
- 仅改变一个变量: `POSE_GLOBAL_PSG: True → False`
- 其他所有配置 (PDS, StopGrad, 训练超参数) 与 exp023 完全一致

## 论文价值
完成 PSG 贡献的消融证据:
| 配置 | PSG | PDS | StopGrad | 预期 mAP |
|------|-----|-----|----------|---------|
| Baseline (exp000) | ❌ | ❌ | ❌ | 56.6% |
| PSG-only (exp007) | ✅ | ❌ | ❌ | 58.3% |
| PDS+StopGrad 无 PSG (**exp024**) | ❌ | ✅ | ✅ | ~56.6% |
| PDS+StopGrad+PSG (exp023) | ✅ | ✅ | ✅ | 59.5% |
