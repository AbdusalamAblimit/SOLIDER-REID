# exp370 PBSR 监控

## 状态

- 当前阶段：查新、设计、隔离实现与本地机制审计完成；等待远程 CUDA smoke
- 训练状态：未启动
- 输出目录：尚未创建
- GPU：未占用

## 预训练门禁

- [x] 核对 PAFormer 与 LGPA 的直接重合
- [x] 核对 ProFD reverse cross-attention 的真实数据流
- [x] 核对 KPR、BPBreID、PFD、PAT、TSD、PGDS/PGFL-KD
- [x] 写明可主张与不可主张的新颖性边界
- [x] 纳入 exp161、exp320 等历史负证据
- [x] 完成代码实现前独立 design 审查
- [x] 完成默认关闭与零初始化逐元素退化测试
- [x] 完成 pose-loss/backbone 梯度防火墙测试
- [x] 完成 eval 无姿态依赖测试
- [x] 完成 CPU bfloat16 autocast smoke test
- [ ] 完成远程真实 dataloader + CUDA AMP smoke test
- [ ] 冻结 kill-switch manifest 后方可启动训练

## 事件记录

### [2026-07-13] ProFD 风险裁决

- 官方实现的 reverse cross-attention 会更新 visual token 副本，但该副本只作为 part decoder 内部 key/value。
- 最终 global descriptor 仍取原始 CLIP CLS；更新后的 visual tokens 不返回主干。
- 结论：不能声称“首次双向 attention”，但“共享路由重组实际 global 主表征”仍保有差异空间。

### [2026-07-13] 机制收紧

- 删除 CLIP text prototype 依赖。
- pose 从前向 bias 改成纯训练监督 target。
- read/write 强制共享 routing matrix。
- pose assignment loss 使用 detached backbone 输入。
- 最终只返回标准 global descriptor，不使用 part concat/MaxSim。
- 当前判断：允许进入无训练实现与审计，不允许直接开正式训练。

### [2026-07-13] 本地实现审计 PASS

- 新增独立模块 `model/modules/pose_bidirectional_router.py`，所有 config 默认关闭。
- read/write 共享 routing；independent-write 对照不增加参数。
- route loss 用 `feat_map.detach()` 重新计算路由，router 有梯度、input/backbone 无梯度。
- `write_scale=0` 时输出与输入 bitwise 相同；首步 `write_scale.grad` 非零。
- 门打开后 identity probe 可到达 key/query/out projection。
- eval 对 None/correct/random heatmap bitwise 不变。
- coupled 与 independent 参数量相同。
- CPU bfloat16 autocast forward/backward finite。
- YACS 成功读取冻结 `configs/occluded_duke/exp370_pbsr.yml`，batch size 保持 64。
- 本地测试输出：`PBSR mechanism checks: PASS`。
- 当前判断：允许远程单批次 CUDA smoke，仍不允许直接启动 120 epoch 正式训练。
