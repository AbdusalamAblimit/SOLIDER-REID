# 实验 exp389：官方干净代码上的双层 TAPF HT0

## 动机

exp387 已在官方最后代码、fresh ViTPose-H train-only target 与严格 RGB-only eval 下完成最小单层 TAPF D0，final=`57.6/67.7/80.8/84.6`；其 matched B0 为 `57.4/67.4/80.6/85.2`。exp388 又在 Market 上得到 D0=`92.0/96.5/98.8/99.3`，相对 matched B0 为 `+0.4/+0.2/+0.1/+0.1`。两组 clean 证据支持完整 `anchor+PSG` 原子方法，但效应较小。

旧实验中的 hierarchical 运行时代码、旧 pose cache 与路径映射均禁止复用。用户要求在 Market D0 封板后，基于当前官方干净实现重新验证“多阶段那个”。本实验只从零加入一个早期层级，直接比较 exp387 D0；它是 backbone-conditional 扩展的干净确认，不预注册为论文 headline，也不能覆盖既有 Swin/ResNet/ViT 上 HT0−D0 不稳定的事实。

## 核心假设

Stage-1 的较高分辨率特征可能更适合学习遮挡下的粗姿态场；若这个内生场在 Stage-2 的六个 block 后分别调制特征，其影响会经过剩余 Stage-2、下采样、Stage-3 两个 block 与最终全局池化，因而每个 consumer 都有到最终 descriptor 的真实下游路径。Stage-2 的既有 anchor 与 Stage-3 两个 PSG 完整保留，用于检验新增早期层级是否在同一干净 recipe 下带来 HT0−D0 增量。

## 单变量对照

| 项目 | exp387 D0 | exp389 HT0 |
|---|---|---|
| 官方 backbone / teacher | Swin-Tiny / SOLIDER teacher | 相同 |
| 数据 / pose artifact | Occluded-Duke / exp386 fresh train-only ViTPose-H | 相同 |
| batch / seed / epoch | 64 / 1234 / 120 | 相同 |
| sampler / RGB 增强 | identity sampler / 384×128 / flip-pad-crop-RE | 相同 |
| optimizer / LR /主 ReID loss | SGD / 0.0008 / ID+triplet | 相同 |
| 晚期层级 | Stage-2 anchor → Stage-3 两个 PSG | 逐参数与路由完整保留 |
| 唯一新增结构 | 无 | Stage-1 anchor → Stage-2 六个独立 PSG |
| 测试输入 | RGB-only | RGB-only |

正式 output 预注册为 `log/occluded_duke/exp389_clean_swin_tiny_ht0_s1234`。不得从 exp387 checkpoint 续训；两臂比较依赖相同 seed 的 fresh 初始化与同一 teacher，而不是 warm start。

## 技术方案

### 1. 晚期 D0 路径保持不变

保留 exp387 的全部晚期实现：Stage-2 pre-downsample `384×24×8` feature 经既有 `norm2` 后输入 `384→128` anchor；两个独立 Stage-3 PSG 分别位于两个 block 之后。其参数名、初始化顺序、handoff、pose loss、consumer detach 与 eval 行为都必须与 exp387 D0 exact。

### 2. 新增早期 anchor

在 Stage-1 pre-downsample 输出上使用既有 `norm1`，得到 `192×48×16` source。新增 anchor 复用同一种最小结构，但不共享参数：

1. `1×1 Conv 192→128`；
2. `3×3 depthwise Conv`；
3. GroupNorm + GELU；
4. `1×1 Conv 128→34`，输出 17 个 heatmap 与 17 个 confidence。

source 继续 detach，pose objective 不回流 Swin。paired keypoint 由同一个 FP32 Gaussian renderer 现场映射到早期 grid；reliability、空 valid、越界 mask 与 score clamp 规则均不改变。

### 3. Stage-2 独立 PSG bank

Swin-Tiny Stage-2 含六个 block。早期 anchor 对应一个只属于它的六成员 PSG bank；每个 block 后消费一次同一个 early consumer field。每个 PSG 仍为：

`Conv1×1(17→32, no bias) → GroupNorm(affine=False) → GELU → Conv1×1(32→384, no bias)`。

末投影 zero-init，调制仍为 `x·(1+0.5·tanh(delta))`。六个 bank 参数互不共享；Stage-3 的两个晚期 bank 也不与它们共享。Stage-2 gate 后的输出继续进入后续 block；最后一个 gate 仍位于 Stage-2 downsample、完整 Stage-3 与最终 pooling 之前，所以不存在 terminal dead consumer。

### 4. 双层训练与损失边界

两个层级使用相同的 e1–5 teacher、e6–10 线性 handoff、e11–120 student consumer 日程。每层 anchor 分别接受与 exp387 定义相同、权重仍为 `0.1` 的 pose supervision；总辅助项是两层 pose loss 之和。这样既有晚期 pose loss 不被减半，HT0 唯一新增对象是完整的早期 `anchor+pose supervision+PSG bank` 层级。该结构同时带来参数、辅助损失与训练成本，必须单独报告，不能把容量增量伪装成纯姿态语义贡献。

ReID loss 可更新 Swin、早期/晚期 PSG 与 head，但在两个 field 边界 detach，不能更新两个 anchor；各自 pose loss只更新对应 anchor，不更新 Swin、任一 PSG 或另一 anchor。日志分别记录 early/late pose、student fraction、reliability 与 gate magnitude。

### 5. 推理因果边界

eval 时两个 anchor 都只读取 RGB feature。外部 correct pose、batch-shuffle pose、`None` 与任何访问即抛异常的 exploding pose 均不得被索引；四种输入的 descriptor、两个 student field 与八个 gate delta 必须逐元素 exact。query/gallery 不建立 pose store。

## consumer 下游路径审计

| 层级 | source | consumer | 到最终 descriptor 的后继路径 |
|---|---|---|---|
| early | Stage-1 pre-downsample | post Stage-2 block 0–5 | 剩余 Stage-2 blocks → Stage-2 downsample → Stage-3 blocks → GAP |
| late | Stage-2 pre-downsample | post Stage-3 block 0–1 | 剩余 Stage-3 block（若有）/最终 Stage-3 output → GAP |

Stage-3 block1 后 gate 仍直接改变被 GAP 使用的最终 spatial feature，因此与 ViT 的 post-final-block CLS dead consumer 不同。门禁必须用旁路某个 gate 后 descriptor 发生变化来证明每个已学习 consumer 的可执行路径，不能只凭梯度或参数名判断。

## 启动前门禁

1. unit：继承 Gaussian/reliability/zero-field identity，并新增 early shape、六 bank 路由次数、bank 独立与 terminal-path 单元测试。
2. config-off：相对 pre-TAPF clean commit 的 state/init/RNG/forward/loss/optimizer 多步 bit-exact。
3. D0-off：新代码在 `HIERARCHICAL=False` 时与 exp387 D0 的公共 state、构造 RNG、optimizer 顺序和多步 CUDA/AMP 输出/梯度/更新 exact。
4. paired data：exp386 strict artifact、5/5 unit、32-seed RGB parity、真实 batch64/8 workers。
5. route：e1/e6/e10/e11 两层 student fraction=`0/0.2/1/1`；early 六次、late 两次消费，无遗漏或重复。
6. gradient ownership：early/late pose loss分别只更新自己的 anchor；ReID 更新两组 PSG/Swin/head而不更新 anchor。
7. state/optimizer：公共参数 exact；新增参数全部且仅一次进入 optimizer；strict save/load 后 descriptor、两 field、八 gate exact。
8. CUDA/AMP：真实 batch64 连续 24 step，记录默认 GradScaler 回退与后续有限更新。
9. overflow：人为 nonfinite 时全部 model parameter 与 optimizer state 整步 exact skip，scale 正确下降。
10. pose-free parity：correct/shuffle/None/exploding 的 descriptor、field、gate exact；query/gallery RGB-only。
11. consumer path：逐一旁路八个 learned gate，最终 descriptor 均须出现有限非零变化；任何 dead gate 均为阻塞性失败。
12. efficiency：报告参数、supported-op FLOPs、train/eval 显存与速度，并直接列 HT0−D0。
13. execution：fresh repo、exact commit/full-history bundle/config SHA、output 不存在、GPU 空闲；任一门禁失败不得启动。

用户已明确禁止 Claude。本实验只做本地/远端代码自审和可执行门禁，不创建或调用 `claude_review.md`。

## 预期结果

主比较固定为 exp389 HT0 e120 − exp387 D0 e120 的 mAP/R1/R5/R10 四项差值。只有 e120 final 与完整轨迹用于结论；不得用中途 best、单 epoch 或阈值替代。若 HT0 仅在部分指标小幅正，仍只能称 Swin-T 单 seed 条件性证据，必须与既有 Swin/ResNet/ViT hierarchical 结果共同解释。

## 风险与失败解释

1. HT0≈D0：说明 clean early level在 Swin-T 上无可分辨增量，与既有 backbone-conditional 判断一致；不追加层数、宽度或 loss 救场。
2. HT0<D0：说明早期 gate 或双 anchor supervision 干扰已训练好的晚期原子路径；保留负结果，不复活旧 runtime。
3. pose loss下降但检索不升：只能说明早期 anchor 可预测姿态，不能说明它改善身份证据。
4. 某 consumer 对 descriptor 无路径：视为实现阻塞，正式训练不得启动；禁止把 dead consumer 计入方法或参数贡献。
5. AMP/OOM：先缩小实现错误或冗余，不改 batch64；运行中不得改 config。
6. 单 seed 正增益：只作为 clean matched 描述性结果，不升级为跨架构 headline。
