# 实验 exp377：Pose-Conditioned Selective SSM

## 动机

exp375 的 PRSM 用姿态把 RGB token 写入六个手工身体槽，但冻结反事实显示实例姿态没有改变
检索排序；exp376 又说明，在 stage 2/3 每个 block 后生成低秩动态算子，仍稳定低于 clean
baseline。两次失败共同提示：不能继续只改变姿态门控或局部残差的函数复杂度。

本实验改动真正的状态空间动力学。在 Swin 最终 `12×4` token 与 GAP 之间加入一个标准的
输入选择性对角 SSM：稳定连续状态矩阵 `A` 经过输入相关步长 `Δ` 离散化，RGB 内容生成基础
`Δ/B/C`，实例姿态只作为外生观测修正这三个 selective 参数。第一版固定扫描顺序，不同时
引入解剖扫描、身体槽、graph、额外 loss 或多 stage 注入。

## 核心假设

遮挡会使不同身体位置的视觉 token 需要不同的记忆时间尺度、写入方向和读取方向。RGB 可以
决定“当前看到了什么”，但只靠 RGB 难以稳定判断当前 token 在人体结构中的位置。若局部关节
组成与置信度能够修正真实 selective SSM 的 `Δ/B/C`，正确 target-person pose 应同时：

1. 优于无 SSM 的 clean Swin；
2. 优于参数与初始化完全相同的 RGB-only selective SSM；
3. 优于固定 canonical pose；
4. 在冻结 checkpoint 上对匹配姿态、关节通道置换和 pose-off 产生可测排序退化。

否则，任何涨点只能归因于普通 Mamba/SSM 容量，而不能归因于实例姿态。

## 技术方案

只对最终 featmap 使用长度 `L=48` 的固定 serpentine raster 序列；共享参数分别正向与反向
扫描并平均。对第 `t` 个 token：

```text
z_t = LN(x_t)
u_t = SiLU(W_u z_t)

delta_t = softplus(delta_bias + W_delta z_t
                   + visibility_t * g_delta(q_t))
B_t     = W_B z_t + visibility_t * g_B(q_t)
C_t     = W_C z_t + visibility_t * g_C(q_t)
A       = -exp(A_log)

Abar_t = exp(delta_t * A)
h_t    = Abar_t * h_(t-1) + delta_t * B_t * u_t
y_t    = sum(C_t * h_t, state) + D * u_t
out_t  = x_t + alpha * W_out((y_t_forward + y_t_reverse) / 2)
```

- `d_inner=128`，`d_state=16`；
- `q_t` 是局部 17 关节热图按通道和归一化后的关节组成；
- `visibility_t` 取局部最大关节响应，只缩放 pose residual，不直接门控输出；
- pose MLP 无 bias，输出经 `tanh` 限幅；zero pose 的 `Δ/B/C` 必须逐 forward 精确等于
  RGB-only arm，但 SSM 本身仍工作；
- `A` 始终为负，递推强制使用 FP32，再转换回输入 dtype；
- `alpha` 采用近 identity 初始化，但不得与 pose gain 同时为零；训练前必须证明 SSM 与
  pose `Δ/B/C` 各分支均有有限非零梯度和参数更新；
- 使用纯 PyTorch reference recurrence，避免两机不同 CUDA/PyTorch 下引入 fused kernel
  混淆。首轮性能成立后再考虑可选加速后端。

这不是 exp375 PRSM 换名：PRSM 没有连续 `A`、`exp(ΔA)` 离散化或 RGB-selective `B/C`，
且状态是六个手工解剖槽；exp377 使用单个对角连续状态、完整 48-token scan，姿态直接修正
输入选择性动力学。但两者都涉及状态保留/更新，论文中必须诚实承认邻近关系。

## 对照组

### Gate A：训练对照

1. `B0`：exact-commit clean Swin-Tiny global-only，无 selective SSM；
2. `D0`：与 P0 完全相同的 selective SSM 和 state dict，但 pose source 为 zero，得到
   RGB-only `Δ/B/C`；
3. `M0`：与 P0 参数和初始化相同，使用固定 canonical pose；
4. `P0`：使用当前图像 target-person pose 同时修正 `Δ/B/C`。

首轮由 4090 跑 P0、3090 跑 D0，仅作快速趋势；跨机差值不作正式裁决。e60 clean 参考预注册为
exp375 已完成的同一 4090 clean B0 曲线（e60 `55.2/65.0/77.6/83.1`，e120
`58.4/67.1/81.2/85.6`）。exp377 已在生产模型上证明其 B0 与该 legacy B0 的 state dict、
descriptor 和最终 featmap 逐元素相同；因此它只用于 e60 早停，不充当最终论文 control。
P0 有燃料后，补同一 4090/解释器/exact commit 的 B0、D0、M0；未补齐前不得作正式 GO。
所有 arm 使用 seed=1234、batch=64、120 epochs、
相同 pretrained checkpoint、优化器与数据顺序，测试只用 global descriptor。

### Gate B：冻结 checkpoint 反事实

至少评测：

- correct-start / correct-end；
- target-matched pose；
- correct visibility + matched joint composition；
- matched visibility + correct joint composition；
- joint-channel permutation（保持逐像素总量与 support）；
- canonical pose；
- pose-off。

pose-off 必须与同 checkpoint D0 mode 的 descriptor 和指标精确一致；它不等于 clean B0。
matched donor 必须按本模块实际看到的 final-grid visibility、joint composition 与
`Δ/B/C` residual 做专用 preflight，不能沿用 exp375/376 donor 结论。

若全量 pose shuffle 退化、但 correct-visibility + matched-composition 不退化，收益只能解释为
foreground/support 对齐；若 `Δ/B/C` 联合版本成功，随后必须追加 `Δ-only`、`B/C-only` 消融，
不能直接把联合收益归给某一个 selective 参数。

## 预注册判断门槛

- `< epoch 60` 不作负裁决，完整记录每次 eval 的 mAP/R1/R5/R10；
- e60 前必须证明 `A/Δ/B/C/state/output` 全部 finite，正反 scan 均执行，pose residual 非零，
  P0 与 D0 的核心 SSM 参数均实际更新；
- e60 若 P0 相对上述预注册同机 clean B0 低 `>=0.5 mAP`，**或**正确 pose 相对
  matched 与 support-preserved composition/pose-off 均无 `>=0.1 mAP` 的冻结燃料，判
  NO-GO 并停止；3090 D0 只作趋势，不能触发正式差值裁决；
- 若 P0 接近或超过 controls 且反事实有燃料，跑满 e120 并补同机 B0/D0/M0；
- 正式性能门槛：`P0-B0 >= +0.8 mAP`、`P0-D0 >= +0.5 mAP`、
  `P0-M0 >= +0.4 mAP`；
- 正式因果门槛：`correct-matched >= +0.3 mAP`、
  `correct-(correct visibility + matched composition) >= +0.3 mAP`、
  `correct-pose-off >= +0.3 mAP`；
- 首 seed 通过后至少补同机多 seed；单 seed 只作为探索证据。

## 预期结果

成功时，exp377 证明的不是“把 Mamba 放进 ReID 就涨点”，而是实例姿态能作为外生结构观测，
校准 RGB-selective 状态动力学并对排序产生因果贡献。失败时，封板当前固定 serpentine、最终层、
联合 `Δ/B/C` 实现；后续解剖扫描顺序或 pose-controlled selective forgetting 必须另立实验，
不能作为 exp377 的临场救场变体。

## 风险与失败解释

1. `exp(ΔA)` 的长乘在 AMP 下可能下溢：递推强制 FP32，并记录 Δ 分位数、A/dA 范围与
   state max/RMS；
2. 近 identity 初始化可能让模块未学动：batch64 CUDA preflight 必须证明 correct 与 pose-off
   descriptor 不同，`A/Δ/B/C/pose MLP/alpha` 均有梯度和 optimizer update；
3. pose residual 可能只编码 foreground：必须执行 support/composition split；
4. 联合修正 `Δ/B/C` 的解释较宽：只有整体机制先过性能/因果 Gate 后，才投入分支消融；
5. 两机运行时不同：跨机仅筛查，正式结论全部要求同机 exact control。
