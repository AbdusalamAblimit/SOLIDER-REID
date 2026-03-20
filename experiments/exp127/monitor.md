# exp127 监控

## 实验信息
- 方法: SCRC（Support-Conditioned Residual Completion）
- 类型: 训练端单变量改进
- 运行位置: 本地 3090
- 主配置: `configs/occluded_duke/pose_psg_gcn_scrc.yml`
- 核心变量: `POSE_SCRC = True`
- 直接对照: `exp116 SCFR`
- 主基线: `exp030a-eq seed1234`

## 启动记录

### [2026-03-20 16:14] 启动前记录
- 目标:
  1. 检验“support-complete prior 作为可学习残差”是否优于 `SCFR` 的硬替换
  2. 不再停留在 `loss / routing` 微调，而直接测试 feature-level support completion 的下一阶段机制
- 当前判断: 准备启动

### [2026-03-20 16:15] 启动失败（解释器错误）
- 现象:
  1. 第一次直接用系统 `python` 启动，后台进程秒退
  2. 前台复现报错：`ModuleNotFoundError: No module named 'torch'`
- 原因:
  1. 默认解释器是 `/root/miniconda3/bin/python`
  2. 实际训练环境应使用 `/root/miniconda3/envs/solider-reid/bin/python`
- 当前判断: 已修正后重启

### [2026-03-20 16:16] 启动确认（修正后）
- 运行位置: 本地 3090
- 输出目录: `log/occluded_duke/exp127_scrc`
- 启动方式:
  - `/root/miniconda3/envs/solider-reid/bin/python -u train.py --config_file configs/occluded_duke/pose_psg_gcn_scrc.yml`
- 关键确认:
  1. 配置已正确加载：`POSE_SCRC=True`, `POSE_SCFR=False`, `POSE_SCKD=True`
  2. 控制台打印：`[SCRC] Support-Conditioned Residual Completion enabled: hidden=128`
  3. 控制台打印：`[SCKD] enabled: weight=0.5, warmup=20, low_thr=0.3, update_thr=0.5, mom=0.9, stop_epoch=-1`
  4. 数据集与模型均已成功构建，进入 `start training`
  5. 本地主进程 PID `3914446`，GPU 占用约 `8032MiB`
- 当前判断: 继续
- 原因:
  1. `SCRC` 路径已真正接入训练
  2. 默认行为未被破坏，启动阶段未见新报错
  3. 下一关键点是 `Epoch 1-5` 的 loss/速度，以及 `epoch 21+` 后 `scrc_*` 统计

### [2026-03-20 16:18] 检查点 #1 — Epoch 1-2
- 结果:
  - `ep1` 末尾:
    - `Loss = 16.124`
    - `Acc = 0.002`
    - `Time per epoch = 59.7s`
  - `ep2` 末尾:
    - `Loss = 11.212`
    - `Acc = 0.005`
    - `Time per epoch = 58.1s`
- 当前观察:
  1. warmup 形状健康，`loss` 快速下降，没有 NaN / Inf
  2. 训练速度与 `exp116/119/120` 同量级，没有明显额外开销
  3. `epoch<=20` 时尚未激活 `SCRC`，因此当前阶段主要是在验证“默认训练没被新模块拖坏”
- 当前判断: 继续
- 原因:
  1. 启动稳定，GPU 占用与 DataLoader worker 均正常
  2. 下一次有信息量的点是 `ep10` warmup 验证

### [2026-03-20 16:20] Claude 代码审查结论
- 审查文件: `experiments/exp127/claude_review.md`
- 审查结论:
  1. **无阻塞性问题**
  2. 唯一中等优先级问题是：
     - `SCFR replace_ratio` 的语义在重构后从“guard 前”变成了“guard 后”，会影响与历史日志的直读对比
  3. 其余仅为低优先级问题：
     - 一个多余的 `detach`
     - `SCRC` warmup 阶段与 `SCFR` 一样只更新 bank、不引入额外 support 信号
     - gate 对全 `(B,17)` 位置做前向，有少量冗余计算
- 已处理:
  1. 已将 `SCFR replace_ratio` 改回历史语义
  2. 额外保留 `applied_ratio` 以记录真正执行替换后的比例
  3. 去掉了多余的 `detach`
- 当前判断: 继续，不重启
- 原因:
  1. 审查未发现会导致训练崩溃或逻辑失真的问题
  2. 已修正的问题仅影响未来代码口径，不值得为此打断正在正常运行的 `exp127`
