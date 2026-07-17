# exp378 显式 SGD relaxation 实现审查

## 审查范围

- `relaxation_design.md`；
- TAPF transition config、module、processor optimizer boundary；
- MR-F0/MR-P0 paired configs；
- CPU单元、full-model invariant与CUDA preflight；
- PyTorch 1.13 legacy zero-gradient parity与真实 GradScaler overflow语义。

本审查由两个独立 Codex 子审查完成，未调用 Claude。审查只读进行；所有修复由主执行者落盘后
重新送审。

## 首轮结论与阻塞项

核心数学路径正确：e11后 anchor `requires_grad=False`、objective `.grad=None`，只在真实
`GradScaler.step(optimizer)` 边界显式写入零梯度，因此标准 SGD 参数组继续执行 momentum、
weight decay与当前 LR日程；`finally`清理 synthetic grad，overflow时与主 optimizer同步跳步。

首轮指出两个 High 证据缺口：

1. 原 overflow单测只是省略 `optimizer.step()` 的 CPU模拟，不足以证明真实 `found_inf`；
2. 原 parity只在本地新版 PyTorch运行，未在生产 PyTorch 1.13 CUDA路径逐步比较参数与
   momentum buffer。

另指出每 iteration同步计算anchor/momentum norm存在不必要开销。

## 修复与复审

已完成：

1. CUDA preflight真实向 adapter gradient注入 `Inf`，要求 GradScaler scale下降，并逐位证明
   全部模块参数、anchor momentum buffer均不变，synthetic anchor grad被清空；
2. CUDA preflight在当前运行时构造 legacy/explicit两条独立 SGD轨迹，运行两轮各5步：
   - production组：weight/bias weight decay均为 `1e-4`；
   - stress组：bias LR保持 `2×`，bias weight decay改为 `2e-4`，验证逐参数组语义；
   每步要求anchor参数与momentum buffer逐位相同；
3. 复审发现 optimizer `load_state_dict` 的同device tensor可能共享 storage。已改为
   `copy.deepcopy(state_dict)`，并在步进前逐对要求 momentum值相同但 `data_ptr()`不同；
4. relaxation norm只在 production `LOG_PERIOD`采集，常规 iteration不触发额外norm归约/sync；
5. CPU单元扩展为12项，覆盖 hard-freeze、legacy parity、MR-F0/MR-P0 matched anchor、
   adapter隔离、模拟skip清理、非法mode/optimizer拒绝；当前 `12/12 PASS`；
6. 两份MR config去除 mode、transition与 output后与 hard对照完全一致，batch64、seed1234、
   `RE_PROB=0`不变；默认 transition仍为 `hard`。

## 最终裁决

`PASS_FOR_4090_EXACT_RUNTIME_PREFLIGHT`。

未见 Critical/High 实现阻塞。exact execution候选 `8af76a1` 已在空闲3090完成只诊断 CUDA
preflight；但3090实际 runtime 为 PyTorch `2.4.1+cu121`，不是4090生产使用的
`1.13.1+cu117`，因此只能作为跨运行时兼容证据，不能替代最终放行。

3090两臂均为 `TAPF_CUDA_PREFLIGHT_PASS`：

- MR-F0：`runtime_parity_steps=10`、overflow `128→64`、anchor delta
  `9.9231e-4`、adapter delta `0`；
- MR-P0：`runtime_parity_steps=10`、overflow `128→64`、anchor delta
  `8.9309e-4`、adapter delta `3.7493e-6`；
- 两臂均为 batch64，anchor objective gradient为零、pose loss为 `None`、外部 pose不被读取；
- full-model paired init/strict reload与12项CPU单元均通过；
- 原始日志SHA256：MR-F0
  `1db3f48a9a7e7290eeab86ce6754d6e744ab49a3cc5bfddfcb00c1969d74221a`，MR-P0
  `76cfa4108198bbab4fdd9a1a5fbc5dcd9714732756d8b519be32f37f537d068b`。

在以下生产证据落盘前仍不得启动120 epoch MR训练：

- 当前4090 hard F0自然完成并释放GPU；
- 同一 candidate在4090 PyTorch `1.13.1+cu117`输出 `runtime_parity_steps=10`及真实
  overflow scale下降；
- 4090 MR-F0/MR-P0 e11 batch64 AMP重新确认上述梯度、delta、pose-input parity与full-model
  invariants；
- 生成新 exact execution commit/bundle，确保production代码与已审查candidate逐文件一致。

正式运行后必须记录每个 checkpoint 的anchor漂移、momentum norm、AMP scale/skip事件；MR-F0与
MR-P0若出现不同overflow序列，不能假设其anchor轨迹天然matched，必须以checkpoint逐位比较为准。
