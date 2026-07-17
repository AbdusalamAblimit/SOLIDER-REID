# exp377 最终实现与执行审查

## 结论

**GO。** 当前实现可以进入预注册的首轮训练：4090 运行 P0，3090 运行 D0。跨机结果只作趋势判断；正式结论仍须在 4090 同一解释器、同一 execution commit 下补齐 B0/D0/M0。

本结论仅覆盖实现正确性、数值稳定性、配置隔离和执行安全性，不提前宣称性能收益或论文创新成立。

## 代码与机制核对

- selective SSM 是真实的连续到离散 recurrence：`A=-exp(A_log)`、`dA=exp(delta*A)`，状态更新与输出在 FP32 中显式执行。
- RGB token 产生基础 `delta/B/C`；姿态分支只以无偏置 MLP 产生有界 residual，并分别调制 `delta/B/C`。
- D0 的全零姿态严格产生零 pose residual；SSM/RGB 路径仍正常前向、反向和更新。
- 模块只在最终 `12x4` feature map 后、GAP 前插入一次；双向 serpentine scan 的正反方向及空间逆变换已覆盖测试。
- P0 使用 person-0 target heatmap；production forward 中 target-only swap 发生在 selective SSM 前。input-pose 缺失时已增加 fail-fast，不允许静默退化为 D0。
- 新参数会被通用 `model.named_parameters()` 优化器路径纳入；默认关闭时不会注册模块。
- B0 与 exp375 legacy B0 的 state dict、descriptor 和最终 feature map 在相同随机种子下逐元素完全一致。
- 冻结 evaluator 已覆盖 8 个预注册 arm；support/composition 在最终 `12x4` 网格构造并由模块
  自身 `_local_pose` 做误差硬门禁，避免原分辨率交换经二次缩放后失真。

## 配置隔离

- P0/D0/M0 只在 `POSE_SELECTIVE_SSM_POSE_SOURCE` 和独立 `OUTPUT_DIR` 上不同。
- B0 只关闭 `POSE_SELECTIVE_SSM` 并使用独立 `OUTPUT_DIR`。
- 四臂均为 seed 1234、batch 64、120 epochs、Swin-Tiny、global descriptor；PSG/PAA/PRSM/PBSR/GCN/LGPA 等其他机制全部关闭。
- P0/D0/M0 在相同随机种子下共享模型 state dict 逐键完全一致，避免初始化混杂。

## 双机最终 preflight

最终本地实现已重新同步到两台训练机；核心模型、模块、处理器、测试和四份配置哈希均与本地一致。同步后重新执行了 standalone、production integration 和 batch-64 CUDA AMP/GradScaler smoke。

### 4090 P0

- standalone：`6 tests passed`；production integration：`PASS`；batch-64 AMP：`PASS`。
- loss `0.598913`；峰值显存 `5.651 GiB / 23.643 GiB`。
- `delta`、`A/dA`、state、输出、全部关键梯度和 optimizer update 均 finite。
- pose `delta/B/C` residual RMS 分别为 `0.001591/0.001643/0.001668`，确认实例姿态路径实际激活。

### 3090 D0

- standalone：`6 tests passed`；production integration：`PASS`；batch-64 AMP：`PASS`。
- loss `0.596020`；峰值显存 `5.672 GiB / 23.690 GiB`。
- RGB-selective core、SSM、输出投影和 residual scale 均有有限非零梯度并实际更新。
- pose `delta/B/C` residual RMS 精确为 `0/0/0`，pose 分支数据梯度精确为零。

## 启动要求

1. 先只显式暂存 exp377 目标文件并创建干净 execution commit；禁止 `git add -A`，不得带入当前工作树中的无关用户改动。
2. 从该 exact commit 分别部署到两机；4090 使用 `/root/solider-venv/bin/python`，3090 使用 `/root/miniconda3/envs/solider-reid/bin/python`。
3. 启动前确认四个 `OUTPUT_DIR` 不存在；启动后确认每机只有一个 main 和 8 个 DataLoader workers。
4. 每次完整 eval 记录 mAP/R1/R5/R10；小于 epoch 60 的差值不触发负裁决。
5. e60 checkpoint 形成后先生成 exp377-specific frozen donor/preflight，再运行正式反事实；
   旧 exp375 donor 只能作为代码通路测试，不能继承其 nuisance PASS。

在满足上述执行纪律后，无剩余代码级阻塞项。
