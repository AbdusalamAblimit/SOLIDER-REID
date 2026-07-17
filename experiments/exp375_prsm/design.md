# 实验 exp375：PRSM 姿态路由选择性记忆

## 动机

exp374 已把两个事实拆开：PSG 相对 bypass 稳定提高 `+3.8577 mAP`，但正确姿态相对
matched 错姿态只有 `+0.0012 mAP`。因此继续改变逐位置 `gamma(H)*X+beta(H)` 没有
实例姿态因果燃料。新方向不再调制特征幅值，而把遮挡问题重新定义为：普通视觉扫描中，
后出现的遮挡物或背景 token 会覆盖先前形成的身份状态。

## 核心假设

当前图像的实例姿态可以为每个 RGB token 提供部位归属与可写可信度。若姿态直接控制
recurrent memory 的 write/retain，而读取端只由 RGB 内容决定，就能把可靠身份线索写入
稳定的部位状态，同时让低可信遮挡 token 只读取、不污染状态。若该机制真正使用实例姿态，
同一模型的 correct pose 必须明确优于 matched-shuffle 与 canonical/foreground controls。

## 技术方案

在 Swin-Tiny 最终 `12×4` 特征图与 GAP 之间插入
Pose-Routed Selective Memory（PRSM）。每列构成长度 12 的纵向序列，并用共享参数做
头到脚、脚到头双向扫描。维护 6 个状态槽：head、torso、left/right arm、left/right leg。

对 token `t`、部位 `k`：

```text
w_tk = visibility_t * soft_part_tk
carried_tk = retain_k * H_(t-1)k
H_tk = carried_tk + w_tk * (candidate_tk - carried_tk)
```

其中 pose 只生成 `w_tk`。每个 token 先读取 pre-write carried state，再执行自己的写入，
因此当前 token 不能通过自身 pose gate 形成无记忆的局部捷径。状态读取权重由 RGB query
与 learnable part keys 计算：

```text
read_t = sum_k softmax(q_rgb_t · key_k) * carried_tk
Y_t = X_t + alpha * W_out(read_t)
```

`alpha` 以 `1e-3` 初始化。pose 不进入 RGB candidate、read query、part keys 或 output
projection；背景/低可信 token 默认不写，但仍可从已保护的状态读取。

第一版只做固定纵向双向扫描和部位状态写入，不加入 skeleton graph transfer、动态扫描、
额外 loss、GCN、LGPA、PAA 或 PSG。

## 对照组

第一轮训练只做三个核心臂，均为 Swin-Tiny、Occluded-Duke、seed 1234、batch 64、标准
768-d global descriptor：

1. `B0`：无状态模块的干净 Swin；
2. `M0`：参数完全相同的 PRSM，但所有图像使用固定 canonical pose；
3. `P0`：PRSM 使用当前目标实例 pose。

P0 同一 checkpoint 额外做 correct-start、matched-shuffle、canonical、
foreground-uniform、zero、correct-end 六个无需重训的评测。matched-shuffle 使用冻结的、
query/gallery 分离、无 fixed point、异 PID、与 batch/worker/order 无关的双射 donor map；
donor 只替换 target-person pose，并审计 PRSM 实际看到的 write mass、support、纵向中心和
跨度、横向中心/跨度、active columns、12 行/4 列 write-mass profile、六部位质量、
visibility 幅度分位数及 zero-write 状态。正式 map 必须逐 index 绑定其生成时的
query/gallery path、PID、camera 顺序；target-only nuisance 采用 median/MAD robust z，固定
20 个异 PID、无 fixed point 的 constrained-random 双射作为对照。**在读取任何 exp375
反事实指标前**冻结验收线：实际 mean cost / random median `<=0.75`、任一维 median
absolute-z `<=0.65`、zero-write concordance `=1.0`；不通过则该 arm 只能称普通 shuffle，
不得进入 `correct−matched-shuffle` 硬门禁。foreground-uniform 保留 correct 的逐像素
visibility 和总写入量，只删除部位槽归属。
zero 必须产生 PRSM 输入特征的 exact identity，用于区分推理时 memory 贡献与另训 B0。
full canonical 同时改变 route、support 和 write mass，只作诊断；只有 canonical route 与
correct visibility 组合的 mass-matched canonical 才能进入硬门禁。correct-start/end 必须
精确复现，排除跨 arm 状态漂移。exp374 的 PSG/SFT correct/bypass/shuffle 作为外部强机制
对照，不重跑大矩阵。

## GO / NO-GO

Gate A 的 GO 条件：

- `P0(correct)−B0 >= +0.8 mAP`；
- `P0(correct)−M0 >= +0.4 mAP`；
- 同权重 `correct−matched-shuffle >= +0.5 mAP`；
- 同权重 `correct−foreground-uniform >= +0.3 mAP`；
- 同权重 `correct−zero >= +0.5 mAP`；
- 若使用 mass-matched canonical，同权重 `correct−canonical >= +0.3 mAP`；
- 无 NaN/Inf，clean query 不出现明显崩塌。

若 `correct−matched-shuffle < +0.2 mAP`，或 P0 不优于 M0，则正式 NO-GO，不用 graph、
scan order、更多状态槽或额外 loss 救场。介于 `0.2～0.5` 时只允许完成当前 full epoch，
不新开变体矩阵。**任何 `< MAX_EPOCHS/2`（本实验即 epoch 60 之前）的结果都只作轨迹、
评测管线 smoke 或异常诊断，不得据此判负、终止训练或触发上述 NO-GO。**

## 风险与失败解释

- 若 P0≈M0：收益来自通用 state capacity 或 canonical anatomy，不是实例姿态；
- 若 correct≈shuffle：再次重复 exp374 的通用人体先验现象；
- 若 correct≈foreground-uniform：机制只是 pose foreground mask，不能声称部位状态路由；
- 若 correct≈zero：训练正则或 backbone 适配可能有效，但推理时 selective memory 未贡献；
- 若 P0 低于 B0：state adapter 或 GAP 前写回损伤预训练表征；
- 若 alpha 不增长或 state/update 梯度为零：属于实现/优化失败，不作科学结论；
- 若跨机器差异接近门槛：补同机 control，不用跨机差值下最终结论。
