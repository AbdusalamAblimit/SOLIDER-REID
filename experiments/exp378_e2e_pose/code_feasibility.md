# exp378 TAPF 代码可行性审查

## 审查范围与裁决

本审查只评估 `design.md` 中已经收敛的 TAPF 实现：从 Swin-Tiny 的 stage index 2
特征预测冻结姿态锚点，再用一个低带宽几何控制器预测每关节
`(dx, dy, dlogsigma_x, dlogsigma_y)`，经可靠性约束和 Gaussian 重渲染后，只控制
stage index 3 的两个 PSG block。没有修改代码、config 或 manifest，也没有启动训练。

**裁决：代码上可实现，且不需要第二遍 backbone 或在线 ViTPose；有条件 GO。** 当前代码已经有
合适的 stage 级插入点、target-person 离线 teacher、PSG 以及可复用的训练返回接口。正式实现前
必须先解决两个阻塞项：

1. 明确定义 Gaussian 场进入现有 `PoseSpatialGate` 前的 raw 数值域，确保 handoff 前后只经过
   一次 sigmoid，不能把空间 softmax 概率直接送入现有 PSG；
2. 修正或规避普通 Random Erasing 只清零 score/visibility、却不一定清除对应 heatmap 响应的
   RGB/teacher 不一致。

当前 `design.md` 与 `design_review.md` 已统一为 epoch 1–5 teacher、6–10 线性交接、11 起
predicted-only；实现不得自行改回 10 epoch 后硬切换，也不得使用 `FREEZE_BACKBONE_EPOCHS`。

## 一、现有 backbone 的真实 shape 与因果边界

输入为 `384×128` 时，Swin-Tiny 的张量形状是：

| 位置 | token/map 形状 |
|---|---|
| PatchEmbed | `B×(96×32)×96` |
| stage index 0 输出 | `B×96×96×32` |
| stage index 1 输出 | `B×192×48×16` |
| stage index 2 输出 | `B×384×24×8` |
| stage index 3 输出 | `B×768×12×4` |

`model/pose_backbone_model.py::_run_backbone_with_psg()` 手工遍历 `self.base.stages`。每个 stage
返回的 `out` 是该 stage 下采样前的 token，`x` 是下采样后供下一 stage 使用的 token。对
`out` 应用 `norm2` 并 reshape 后，得到 TAPF 所需的 `F2: B×384×24×8`。完整 forward 结束后，
`outs[-2]` 是该 map，`outs[-1]` 才是最终 `B×768×12×4`。

因此单次前向的唯一合法顺序是：

```text
stage 0 -> stage 1 -> stage 2 -> F2/TAPF -> stage 3 PSG -> descriptor
```

由 `F2` 生成的场只能影响 stage index 3。它不能回灌 stage 0–2；为了多层注入再跑一遍
backbone 会改变计算量、drop-path 与优化路径，不属于 exp378。

## 二、推荐模块边界

建议新增单一模块文件：

```text
model/modules/task_adaptive_pose_field.py
```

模块内部包含三个清晰隔离的部分。

### 2.1 LiteHR-style anchor

首版建议宽度 64，最多 96，总参数强制小于 `0.5M`：

```text
F2.detach(): B×384×24×8
 -> 1×1 Conv 384→64 + GroupNorm + SiLU
 -> 高分辨率分支：bilinear ×2 + DWConv 3×3 + PWConv 1×1
 -> 再一次 bilinear ×2 + DWConv 3×3 + PWConv 1×1
 -> 低分辨率 context refinement 后上采样融合
 -> 1×1 Conv 64→17
 -> 17 个 96×32 空间分布

GAP(F2.detach()) -> Linear 384→17 -> 逐关节 confidence
```

不使用完整 HRNet，不接 stage-1 lateral，不加 deformable conv、GCN、PAA、Mamba 或额外
identity head。归一化使用 GroupNorm，不引入依赖 batch 统计的 BatchNorm。ReID 路径只能读取
最终 17 个关节分布、17 个置信度以及由它们得到的几何量，不能读取 decoder hidden feature。

### 2.2 低带宽几何 controller

对 anchor 的每个关节空间分布提取 `mu_x/mu_y/sigma_x/sigma_y`。用该分布在
`F2.detach()` 上做逐关节加权池化，得到 `B×17×384`，再由所有关节共享的轻量 MLP 输出
`B×17×4`：

```text
delta_j = (dx_j, dy_j, dlogsigma_x_j, dlogsigma_y_j)
release_j = (1 - stopgrad(c_j)) ** gamma
mu'_j = mu_j + release_j * max_shift * tanh(delta_mu_j)
logsigma'_j = logsigma_j
              + release_j * max_log_scale * tanh(delta_logsigma_j)
H_adapt_j = c_j * Gaussian(mu'_j, sigma'_j)
```

`max_shift`、`max_log_scale`、`gamma` 必须由 config 预注册。坐标建议在 `[0,1]` 归一化域计算，
方差使用带上下限的 log-space 参数化，Gaussian 归一化和矩计算强制 FP32。controller 只能输出
`17×4` 几何量，不得退回任意 `17×96×32` residual map；Gaussian 重渲染必须是正式 forward
的一部分，而不是训练后的可视化。

### 2.3 PSG 输入数值域

现有 `model/modules/pose_spatial_gate.py::PoseSpatialGate.forward()` 会先 resize，再执行
`torch.sigmoid(raw_heatmap)`。离线 ViTPose NPZ 保存的是 MSE heatmap head 的 raw activation，
可有小负数，并非概率图。

因此实现必须遵守以下约束：

1. teacher、anchor 与 adapted field 都按“PSG sigmoid 前”的 raw 场交接；
2. 空间 softmax 只用于提取分布、矩与 bootstrap shape loss，不能直接作为 PSG 输入；
3. `H_adapt=c×Gaussian` 的幅值定义要与 teacher raw peak 对齐，之后仍只由现有 PSG sigmoid
   一次；不得先 sigmoid 成概率后又被 PSG sigmoid；
4. 每次 preflight 记录 teacher/anchor/adapted 的 raw `min/max/mean/peak` 以及 PSG sigmoid 后
   的对应统计。

如果 raw Gaussian 无法复现 teacher 消费域，允许为 exp378 给 PSG 增加显式
`input_is_probability` 分支，但默认值必须为 `False`，且 teacher 先转换为现有 PSG 实际消费的
`sigmoid(H_teacher)`，保证交接两端只做一次变换。该选择必须在编码前固定，不能训练后再切换。

## 三、最小代码接入点

### 3.1 `model/pose_backbone_model.py`

在 `PoseBackboneModel.__init__()` 中：

- 只在 exp378 开关为真时构造 `TaskAdaptivePoseField`；
- 保存并恢复模块构造前后的 RNG state，避免可选模块初始化改变 B0 的 backbone、classifier 或
  训练随机序列；
- 校验 `POSE_PSG_STAGES` 精确等于 `[3]`，并拒绝 PAA、GCN、LGPA、PRSM、Pose Hyper-LoRA
  等混合开关；
- 注册当前 arm、bootstrap epoch、handoff epoch、可靠性和几何边界参数。

在 `_run_backbone_with_psg()` 中：

- stage index 2 正常完成并得到 norm 后的 `F2: B×384×24×8`；
- 立即用 `F2.detach()` 计算 anchor 和 geometry controller；
- 根据 epoch/arm 选择 external teacher、冻结 anchor 或 adapted field，更新**仅供 index 3**
  使用的局部 heatmap 变量；
- stage index 3 的两个现有 PSG block读取该变量；
- 不修改 stage 0–2 的输入，不启动第二次 backbone；
- 将 bootstrap loss 与审计统计显式返回给 `forward()`。

不能等完整 backbone 完成后再从 `outs[-2]` 计算 pose，因为那时 stage 3 已经执行完毕。计算点
必须位于 stage index 2 的 `outs.append(stage2_map)` 之后、进入下一轮 stage index 3 之前。

在 `forward()` 中：

- bootstrap teacher 只能取
  `pose_dict['heatmaps'][:, 0] * pose_dict['person_mask'][:, 0]`；
- confidence target 取 `pose_dict['scores'][:, 0]`，同样乘 person-0 mask；
- train epoch 11 起不再读取 teacher tensor，不只是把 pose loss 权重设为 0；
- eval 从第一次开始始终 predicted-only，对传入的真实、shuffle 或 `None pose_dict` 完全忽略；
- 训练返回可复用现有第 4 项 `recon_loss` 携带**已经乘好权重**的 FP32 bootstrap loss，第 5 项
  `kp_data` 携带 TAPF 统计。

现有 `_prepare_pose()` 已提供 target-person heatmap，但没有返回 person-0 scores。exp378 可在
`forward()` 中直接提取 target scores，避免改变这个被大量旧实验复用的四返回值接口。

### 3.2 `processor/processor.py`

现有 processor 已兼容模型的 3/4/5 项训练返回值，并会把非空 `recon_loss` 加到总 loss，因此
不需要另建训练器。需要的最小改动是：

- 每个 epoch 开始时，对 `model.module` 或 `model` 调用固定的 `set_tapf_epoch(epoch)`；
- 从 `kp_data['tapf_stats']` 记录 source、blend、pose loss、raw/post-sigmoid 范围、anchor 与
  controller 梯度、置信度分组 residual RMS；
- bootstrap loss在 autocast 外或显式 `.float()` 后计算；
- epoch 11 后断言 P0 的 `recon_loss is None`，而不是一个数值为零但仍连接 teacher 的 tensor；
- 记录 `GradScaler` scale、NaN/Inf 和 optimizer step 后参数 delta。

不要复用 `SOLVER.FREEZE_BACKBONE_EPOCHS`。历史 freeze/unfreeze 会改变 PSG 主干训练，而且
动态切换 `requires_grad` 还可能让参数未被 optimizer 正确管理。exp378 的梯度语义只通过
`.detach()`、是否构建 loss、source blend 和 arm 开关表达。

### 3.3 `solver/make_optimizer.py`

模块只要在 optimizer 构造前注册到 model，当前 `named_parameters()` 遍历会自动纳入 anchor 和
controller。当前 optimizer 只对 classifier 与旧 part branch 做 LR 特判；若 TAPF 需要独立
LR factor，必须按明确参数前缀加入一次，并在日志输出每组参数数、LR 和 weight decay。

必须测试：

- 每个可训练参数只出现一个 param group；
- anchor/controller 参数没有遗漏；
- epoch 11 后冻结 anchor 的 `grad is None`，从而 AdamW 不对其执行 weight decay；
- controller 在 P0 有有限非零 delta，在 F0/D0 为零。

### 3.4 `config/defaults.py`、独立 YAML 与 `make_model.py`

新增配置必须默认关闭，至少覆盖：enable、arm、stage、head width、bootstrap/handoff epoch、
confidence threshold、`gamma`、`max_shift`、`max_log_scale`、bootstrap loss 权重与 TAPF LR factor。
exp378 使用独立 OUTPUT_DIR。

`model/make_model.py` 不需要改动：`MODEL.POSE_ENABLED=True` 且
`MODEL.POSE_BACKBONE_PSG=True` 已经选择 `PoseBackboneModel`。

## 四、课程与梯度链

### 4.1 固定课程

| epoch | Stage-3 PSG source | anchor pose loss | ReID 到 anchor | ReID 到 controller |
|---|---|---:|---:|---:|
| 1–5 | person-0 external teacher | ON | OFF | OFF |
| 6–10 | teacher→internal anchor 线性 handoff | ON | OFF | OFF |
| 11–120，F0 | frozen anchor | OFF | OFF | OFF |
| 11–120，D0 | distilled anchor | ON | OFF | OFF |
| 11–120，P0 | adapted Gaussian field | OFF | OFF | ON |
| 11–120，J0 | adapted Gaussian field | ON | OFF | ON |

epoch 6–10 同一份 anchor 输出要分两条使用：bootstrap loss读取非 detach 输出，ReID/PSG
交接读取 `anchor.detach()`。这样 pose loss只更新 anchor，ReID 不会提前进入 anchor。epoch 11
以后 P0/J0 的 controller 同时读取 `F2.detach()` 与 `anchor.detach()`；anchor 不接收 ReID
梯度。Swin 主干仍通过正常 descriptor→stage3→stage2 主路径接受 ReID 梯度，但不存在
controller→F2 的第二条梯度。

F0/D0/P0/J0 在 epoch 1–10 必须具有逐键一致的模型初值、相同 source、blend、sampler 和
bootstrap 更新。epoch 11 才按两个因子分叉。P0 后期的唯一损失是标准 ID+triplet；不能残留
flip、pose、part、matching 或 confidence regularizer。

### 4.2 bootstrap loss

teacher 不是 GT。建议在 FP32 中把 raw heatmap 的正响应归一化为空间 shape，只对
`teacher_score >= threshold` 的关节做 KL/heatmap shape 监督，再用 soft-target BCE 或回归
监督独立 confidence。低置信关节不应被强迫复制一个可能错误的峰。

所有精度措辞必须是 `teacher agreement`、`pseudo-PCK`、normalized coordinate error 或
heatmap cosine，不能写成真实 pose accuracy。

## 五、数据链风险

`datasets/pose_dataset.py` 已提供：

```text
heatmaps:             B×max_persons×17×96×32
scores:               B×max_persons×17
visibility:           B×max_persons×17
visibility_binary:    B×max_persons×17
person_mask:          B×max_persons
```

原始 NPZ heatmap 为 `17×64×48 float16`，keypoints/scores 为 float32；dataloader 会投影到
整图并同步 resize、水平翻转、pad/crop。`_load_persons()` 会把 index 中的 target person 重排到
person 0，水平翻转也已交换 COCO 左右关节通道。

仍有以下必须处理的风险：

1. **普通 Random Erasing 不完全同步。** `_update_persons_for_erase()` 会把擦除框内的
   scores/visibility 清零，但只有 `erased_channels is not None` 的 pose-guided erasing 才清对应
   heatmap；普通 RE 可能保留原空间响应。首轮应禁用普通 RE，或实现对所有擦除框同步清 heatmap，
   至少 bootstrap loss必须按更新后的 score mask排除不一致关节。
2. **target-person 歧义。** teacher 是 person 0，但内部 head只看整张 crop，图中仍可能有
   distractor。必须按 target-person 数量/遮挡分层记录 agreement；不能退回 scene max-merge。
3. **visibility 不是可靠 GT。** 历史相关性较弱，confidence target优先使用 ViTPose `scores`，
   visibility只作辅助审计。
4. **缓存泄漏。** Gate 0 必须确认 target assignment 不读取 pid/camid/文件名身份 token，且
   train/query/gallery cache 无跨 split prototype 或检索标签信息。

## 六、AMP、显存与两机环境

### 6.1 AMP 与 batch 64

LiteHR conv/upsample 可在 autocast 中运行；空间 softmax、矩、log-variance、Gaussian 重渲染和
bootstrap loss必须用 FP32。高分辨率 `B×64×96×32` 激活是新增显存主项。batch 固定 64，OOM
时按顺序处理：

1. head width 96 降到 64；
2. 所有 3×3 refinement 改为 depthwise separable；
3. 开启 Swin `WITH_CP=True`；
4. 不改 batch size，不换完整 HRNet，不接 live teacher。

首轮 AMP init scale 可沿用近期实验的 `1024`，但必须由 B2 和 B64 实测决定，不能仅凭配置
推断安全。

### 6.2 已核对的两机资产

4090 已有可用离线 teacher 生成环境与缓存：

- ViTPose-Huge checkpoint SHA256：
  `e32adcd41ab0b0ef0b5bf3d167ddae7cdbd45fcf45e7f6a834815ef04d641f2b`，
  2,548,954,167 bytes；
- ViTPose config SHA256：
  `c4fee8723dc3ec74d9d57e75d9b22138480fe556c1f5278f319e9ae5b65b6e16`；
- visibility checkpoint SHA256：
  `f6ebe8240672d1ddd7003ca709ac9ed51deca089676077d54ba713af6b0f5d0b`，
  452,638,455 bytes；
- mmpose 环境：`/usr/local/anaconda3/envs/mmpose-abu/bin/python`，torch 1.13.1、
  mmpose 1.3.2、mmcv 2.1.0、mmdet 3.2.0；
- Occluded-Duke pose cache：`/mnt1/afrdata/Occluded_Duke/pose_data`，约 4.8 GB。

4090 常用训练 venv `/root/solider-venv` 没有 mmpose/mmdet。3090 虽有同 SHA checkpoint，
但 ASFlubi 环境缺 mmpose/mmcv/mmdet，旧 mmpose 0.24.0 与 mmcv 1.7.2 当前 import 不兼容。

结论：两机正式训练都只能读取离线 NPZ teacher，不能在训练图中实例化 ViTPose-Huge。在线
teacher 不仅造成两机依赖差异，在 24 GB GPU、batch 64 下也几乎必然 OOM。

## 七、与 exp020 的代码/科学区别

`exp020` 已做过 final-stage pose reconstruction：从 `B×768×12×4` 最终特征重建 pose，并在
全程保留辅助 MSE，结果为 57.8 mAP，较 PSG 低 0.5，存在后期辅助梯度干扰。

exp378 不能复用该失败叙事包装。它的可区分实现对象是：

- 从更高分辨率 stage-2 `B×384×24×8` 生成 anchor；
- predicted field 在同一次 forward 中真实进入 stage-3 PSG；
- epoch 11 后 P0 pose loss精确为 0；
- anchor冻结，ReID只更新受置信度约束的 `17×4` 几何 residual；
- controller 输入对 F2 detach，Gaussian 是正式信息瓶颈；
- eval 完全 pose-free。

任何实现若变回“final feature + 持续 pose reconstruction”，都应直接判为偏离设计。

## 八、Gate 0 完整测试清单

以下全部 PASS 后才能启动正式训练。

### 8.1 默认行为与前向拓扑

- [ ] exp378 config关闭时，与 exact B0 的 state dict keys、descriptor、final featmap逐元素一致；
- [ ] 输入 `384×128` 时，head 输入精确为 `B×384×24×8`，anchor/adapted 输出精确为
  `B×17×96×32`，confidence 为 `B×17`，delta 为 `B×17×4`；
- [ ] TAPF 总参数 `<0.5M`，逐子模块记录参数量；
- [ ] 每个样本只有一次 PatchEmbed 和一次四-stage forward；
- [ ] predicted field只影响 stage index 3，stage 0–2 完全不读取姿态；
- [ ] PSG 只接收最终 17 maps/17 confidence派生场，不读取 decoder hidden feature；
- [ ] P0/D0/F0/J0 初始 state dict逐键一致，构造可选模块不改变 B0 RNG。

### 8.2 梯度与 optimizer

- [ ] bootstrap：`L_boot→anchor` 有限非零，`L_boot→F2/backbone` 精确为零；
- [ ] epoch 6–10：ReID→anchor/controller 精确为零，anchor仍由 bootstrap 更新；
- [ ] F0 e11：anchor/controller 两类梯度均为 `None`，参数 delta精确为零；
- [ ] D0 e11：pose loss→anchor 有限非零，ReID→anchor/controller 精确为零；
- [ ] P0 e11：pose loss未构建，ReID→controller 有限非零，ReID→anchor 精确为零；
- [ ] J0 e11：pose loss→anchor、ReID→controller均有限非零，交叉梯度仍受 detach 隔离；
- [ ] P0 的 stage-2 主路径有正常有限非零 ReID 梯度，但 controller→F2 梯度精确为零；
- [ ] optimizer 参数无重复/遗漏，LR/weight decay 与预注册一致；
- [ ] 在同一 batch 完成 backward、optimizer step 和参数 delta审计，而非只打印 loss 权重。

### 8.3 teacher-off 与 eval pose-free

- [ ] P0 epoch 11 后任意改写 teacher heatmap、scores、person mask，不改变 descriptor 和总 loss；
- [ ] P0 epoch 11 后移除 `pose_dict` 也能完成训练前向；
- [ ] eval 下 correct、matched-shuffle、全零和 `None pose_dict` 的 descriptor/featmap逐元素一致；
- [ ] 第一次 eval 即 predicted-only，不随 epoch 偷用 teacher；
- [ ] correct-start/end 复现，排除审计脚本本身的随机性。

### 8.4 数据与几何语义

- [ ] person-0 teacher、scores 与 mask逐 batch对应，禁止 scene max-merge；
- [ ] resize、flip、pad/crop 后 RGB/keypoint/heatmap 坐标一致；
- [ ] horizontal flip 同时空间翻转并交换 COCO 左右通道，逆变换误差在预注册容差内；
- [ ] ordinary RE 与 teacher 语义一致，或 Gate A 明确禁用 RE；
- [ ] target assignment、cache、split 无 pid/camid/prototype 泄漏；
- [ ] Gaussian 的 `mu/sigma/confidence/delta/release` 全 finite，方差在上下限内；
- [ ] anchor/adapted map非全零、全一、单点或全均匀 collapse；
- [ ] 低置信组 residual RMS 高于高置信组，高置信关节位移受预注册上限约束；
- [ ] joint-channel permutation、release-uniform/permuted 和 residual-off 干预真正改变 controller输入；
- [ ] Gaussian 语义重渲染后的 descriptor可复现正式 forward，排除 raw-map side channel；
- [ ] confidence correct/shuffle 与 spatial shape correct/shuffle 可独立审计，排除 17 个标量的身份编码。

### 8.5 数值、checkpoint 与设备

- [ ] teacher/anchor/adapted raw 域以及 PSG sigmoid 后的 min/max/peak均符合预注册范围；
- [ ] CPU FP32 batch 2 forward/backward/step PASS；
- [ ] 4090 CUDA AMP batch 2 forward/backward/step PASS；
- [ ] 4090 CUDA AMP batch 64 forward/backward/step PASS，显存峰值有记录；
- [ ] 3090 相同 archive/config 的 CUDA AMP batch 64 preflight PASS；
- [ ] GradScaler scale有限，无 NaN/Inf/overflow 静默跳步；
- [ ] strict checkpoint save/load roundtrip 后 descriptor、anchor、adapted field逐元素一致；
- [ ] 两机使用同一代码 archive、离线 NPZ schema 与 checkpoint/config SHA，不依赖在线 mmpose。

### 8.6 bootstrap 质量仪表

- [ ] 每次 eval列出 mAP/R1/R5/R10；
- [ ] 记录高置信关节 pseudo-PCK、normalized coordinate error、heatmap cosine；
- [ ] 记录 confidence calibration/AUROC，但不称真实 pose accuracy；
- [ ] 记录 e10 teacher-mode 与 100% student-mode差值；
- [ ] 记录高/低置信 residual RMS、坐标漂移、尺度变化和 flip equivariance；
- [ ] 对 raw 17-map、坐标/协方差/置信度、Gaussian 重渲染分别做同容量 ID probe。

## 九、启动前最终结论

exp378 不需要等待新 pose 数据，也不需要把 ViTPose 或完整 HRNet塞进训练。最小实现可直接建立在
`PoseBackboneModel` 的单次 stage loop、离线 person-0 NPZ 和现有 processor 返回协议上。

但正式编码必须锁死三条边界：

1. `F2` 对 anchor/controller 全程 detach，只有标准主路径继续训练 Swin；
2. ReID 只能更新 `17×4` 低带宽几何 residual，anchor在 bootstrap 后冻结，禁止任意 map residual；
3. epoch 11 后 P0 对 external teacher具有严格的计算图独立性，eval从一开始 predicted-only。

在 PSG 数值域和 Random Erasing 一致性两项被写成可执行测试并通过前，不应启动正式训练。
