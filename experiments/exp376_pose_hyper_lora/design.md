# 实验 exp376：逐层 Pose Hyper-LoRA

## 动机

exp071 已验证过固定低秩投影与热图特征逐元素相乘：
`W_up(W_down(x) * f(P))`。严格地说，它的有效矩阵已经是随 token 姿态变化的
`W_up·diag(f(P))·W_down`，不能把 exp376 描述为“第一次动态算子”。exp071 的限制是 A/B
两侧固定，姿态只能在预定义的 rank coordinate 上做 diagonal modulation；而且其历史实验处于
PSG+GCN+equal-concat、scene pose、Stage3 scaffold，不能充当当前 clean-global 配方的直接对照。

exp375 PRSM 进一步说明，仅以姿态分配 recurrent memory 写入位置不能产生可测的身份排序贡献。
下一步不再修改 memory，而是直接检验一个更贴近用户设想的对象：让每个关节热图向量生成
Swin 中间层每个 token 的低秩变换参数。

## 核心假设

遮挡行人图像中，不同空间 token 需要的特征更新方向并不相同。局部 17 关节响应若能动态生成
低秩变换的 A/B 两侧 basis mixture，而非只提供 diagonal rank weight，应当使同一视觉特征在
不同身体位置接受不同的更新。若收益确实来自 factor-wise 动态参数，则它应优于同配方、
projection 参数量精确匹配的 exp071-style diagonal control；若收益确实来自实例姿态，则真实姿态还应同时
优于 image-only 和固定 canonical pose，且
冻结 checkpoint 后打乱姿态—图像对应关系应造成可测退化。

## 技术方案

对 stage 2、3 的每个 Swin block，在原 block 输出后加入一个独立 Pose Hyper-LoRA，共 8 层：

```text
A_i(P_n) = Σ_m a_im(P_n) A_im
B_i(P_n) = Σ_m b_im(P_n) B_im
Δx_in = B_i(P_n) A_i(P_n) LN(x_in)
y_in  = x_in + α · visibility(P_n) · Δx_in
```

- `i` 为 block，`n` 为空间 token；
- 17 通道热图双线性对齐到当前 token 网格；
- 两个独立系数组分别混合 A/B basis，因此有效矩阵随图像和 token 改变；
- rank `r=4`，basis 数 `M=4`，pose MLP hidden dim `32`；
- `α` 可学习，初值 `1e-3`；
- `visibility=max_k P_k` 只控制残差强度；zero pose 必须 exact identity；
- 各 block 不共享参数，以允许逐层学习不同的姿态作用方式；
- 不启用 PSG、PAA、PRSM、LGPA、GCN 或额外 loss，测试只用 global descriptor。

这与 exp071 的可证伪差别是 factorization，而不是“静态/动态”：exp376 的 A、B 两侧均由
局部姿态独立混合；exp071/D0 保持 A、B 固定，只在二者之间使用姿态生成的 diagonal。本文只将
该模块称为“逐 token、逐 block 的 factor-wise pose-hypernetwork low-rank residual”，不声称
直接改写 Swin 原生 Q/K/V/FFN 权重。

## 对照组

### Gate A：训练对照

1. `B0`：exp375 同机 clean image-only，58.4 mAP / 67.1 R1，仅作首轮筛查参考；任何正式
   GO 都必须补 exp376 exact commit、同 4090/解释器/配置的 B0。
2. `D0`：与 P0 相同的 8 blocks、visibility、训练配方，A/B projection 参数量精确匹配，改为固定 A/B 的
   `B·diag(f(P))·A`，直接复刻 exp071-style 核心 factorization。
3. `M0`：参数、初始化、训练配方与 P0 相同，所有样本使用固定 canonical pose。
4. `P0`：使用当前图像 target-person 的实例姿态和 factor-wise A/B basis mixture。

首轮使用相同 seed=1234、batch=64、120 epochs，4090 跑 P0、3090 跑 D0。跨机器只作快速
趋势；若 P0 不能优于历史 B0/D0，停止本机制。若 P0 有希望，再补 4090 exact B0 与 M0，
任何正式 GO 均以同运行时控制为准。

### Gate B：同 checkpoint 反事实

对 P0 的冻结 checkpoint 评测：

- correct pose；
- target-matched shuffle pose；
- correct visibility + matched coefficients；
- matched visibility + correct coefficients；
- canonical pose；
- zero pose。

correct-start/end 必须精确复现；各 intervention 必须确认 descriptor 已改变；zero 路径逐 forward
验证 exact identity。exp375 donor map 只作为候选，正式使用前必须针对 exp376 实际看到的
stage2/3 多层 per-joint、visibility 与 coefficient 统计重新做 nuisance preflight；不能直接沿用
PRSM 的 12×4 write-profile PASS 结论。

上述 visibility-split 是硬控制：`correct-matched` 的差值只有在“correct visibility + matched
coefficients”也退化时，才能归因给 factor-wise 动态矩阵；否则只能归因给显式 foreground gate。
canonical 同时改变 support、幅度和 coefficients，只作诊断，不设硬门槛。

## 预注册判断门槛

- `< epoch 60` 不作负裁决，只记录每次 eval；
- 性能门槛：`P0-B0 >= +0.8 mAP`；
- factorization 门槛：`P0-D0 >= +0.4 mAP`；
- 实例姿态门槛：`P0-M0 >= +0.4 mAP`；
- 因果门槛：`correct-matched >= +0.3 mAP`，理想值为 `>= +0.5`；
- 若 P0 有增益但不优于 M0，结论只能是动态低秩容量/固定解剖先验有效，不能声称实例姿态有效；
- 若 P0 有增益但不优于 D0，结论只能是 pose-conditioned low-rank adapter 有效，不能声称
  factor-wise A/B mixture 有额外价值；
- 首个 seed 通过全部门槛后，至少补同机多 seed；单 seed 只作为探索结果；
- 若训练差值和反事实差值均不达门槛，停止本机制，不补 rank/basis/层数小变体，进入真实 Mamba
  selective `Δ/B/C` 的下一个单变量实验。

## 预期结果

成功时，exp376 可将 exp071 的 diagonal rank modulation 升级为 factor-wise A/B mixture，并由
D0、M0、visibility-split matched shuffle 分别排除旧 factorization、普通容量/固定姿态先验和
前景门控解释。失败时，则说明在当前 Swin-ReID 配方中，
更强的局部动态参数化仍未把姿态信息转化为身份排序贡献。

## 风险与失败解释

1. `α=1e-3` 可能使模块早期作用较弱：A/B 必须逐个二维 basis 正确初始化；训练前用真实
   GPU AMP + GradScaler 验证 8 层 applied residual 不被 FP16 全量舍入、关键参数梯度非零、
   optimizer step 后参数确实改变。训练日志每 50 iter 记录 α、系数幅值、visibility 与 delta RMS，
   用以区分“机制无效”和“模块未学动”。
2. 热图在深层网格上可能过平滑：本实验不临时改插值、rank 或 stage，以保持单变量和可解释性。
3. 3090/4090 运行时不同：跨机只作并行筛查，接近门槛必须补同机控制。
4. 动态矩阵可能增加显存：保持 batch=64；若 OOM，只优化等价 einsum 实现，不改实验定义。
