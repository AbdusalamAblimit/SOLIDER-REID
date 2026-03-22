# exp148 PCVT 监控

## 实验信息
- 方法: PCVT（Pose-Complementary View Training）
- 类型: 训练范式大改动
- 主基线: `exp030a-eq`
- 当前状态: 仅完成设计，尚未实现/审查/启动

## 启动前检查清单
- [ ] 数据管线支持生成 `full / view_a / view_b`
- [ ] complement partition 只依赖当前图 pose
- [ ] 三视图主损失与 `exp030a` 对齐
- [ ] `pcvt_*` 行为日志接好
- [ ] Claude 广范围审查通过

## 当前判断
- 这是对 `support incomplete` 的训练范式级重写，不是小调参
- 若实现失败，也应能明确排除“单图伪多 support”这条主线

## 启动记录

### [2026-03-22 09:58] 代码接入完成，开始工程自检
- 已修改:
  - `config/defaults.py`
  - `datasets/make_dataloader.py`
  - `datasets/pose_dataset.py`
  - `processor/processor.py`
  - `configs/occluded_duke/pose_psg_gcn_pcvt.yml`
- 关键实现点:
  1. 数据侧新增 `PCVT` 三视图：`full / view_a / view_b`
  2. `view_a/view_b` 不再是随机增强，而是由 person-0 pose heatmap 的 body-group 响应做互补划分
  3. 训练侧新增 `pcvt_lc`，约束 `0.5*(f_a+f_b)` 逼近 `f_full`
  4. 行为日志已接入 `pcvt_*`

### [2026-03-22 10:08] 数据级 probe 通过，并修正了“伪互补”问题
- 初版 probe 发现:
  - `pcvt_cov_u ≈ 0.74`
  - `pcvt_ovr ≈ 0.26`
- 原因:
  - part mask 直接取各自 active 区域，空间上会重叠，导致两张视图并不是真正互补
- 已修正:
  - 改为像素级独占分配：每个可见像素只归属响应最大的 body group
- 修正后新 probe:
  - `pcvt_cov_a ≈ 0.499`
  - `pcvt_cov_b ≈ 0.501`
  - `pcvt_cov_u = 1.000`
  - `pcvt_ovr = 0.000`
  - `pcvt_fb = 0.000`
- 当前判断: 现在才算真正实现了“互补 support”

### [2026-03-22 10:18] 模型级 probe 通过，训练路径闭合
- 使用 `pose_psg_gcn_pcvt.yml` 构造:
  - dataloader
  - `PSG+GCN` 模型
  - `loss_fn`
- 取一个 `bs=8` 的真实 batch 做三视图前向
- 结果:
  - `main_loss = 12.296`
  - `pcvt_lc = 0.317`
  - `pcvt_gap = 0.047`
  - `pcvt_cov_u = 1.000`
- 解释:
  1. 主损失与新增 `pcvt_lc` 可以同时正常计算
  2. `pcvt_gap > 0`，说明 union 表示初始就略优于单个 partial view
  3. 当前已具备送 Claude 做广范围审查的工程完整性

### [2026-03-22 10:20] 当前判断
- 继续
- 原因:
  1. 这不是空设计，数据/模型/日志三条链已打通
  2. 下一步不是直接开跑，而是按规则先做广范围 Claude 审查
