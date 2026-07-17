# 实验 exp387：官方干净代码上的最小 TAPF D0

## 动机

exp384 已复现官方 Market1501 Swin-Tiny B0，final=`91.6/96.3/98.7/99.2`；exp385 在同一官方干净代码上建立 Occluded-Duke B0，final=`57.4/67.4/80.6/85.2`。exp386 又从原始 train RGB 用 fresh 官方 ViTPose-Huge 重新生成 15,618 条 COCO-17 target，并完成 manifest、paired augmentation、RGB parity 与真实 batch64/CUDA 门禁。

本实验才开始重建 TAPF。目标不是移植三百余轮实验累积的旧运行框架，而是在官方最后提交上重新写一个小型、默认关闭、因果路径清楚的 `anchor+PSG` 原子模块，直接与 exp385 matched B0 比较。

## 核心假设

训练期 target-person pose 可以把 Swin Stage-2 feature 约束成内部 pose anchor；随后由 Stage-3 Pose Spatial Gate（PSG）把该内部场用于身份特征调制。外部 pose 仅在训练时提供监督和早期交接，query/gallery 没有 pose artifact，测试 descriptor 必须完全由 RGB 产生。

主比较只解释完整 `anchor+PSG`：`D0−B0`。本轮不把 anchor、Gaussian、confidence 或 PSG 单独包装成贡献，也不恢复 hierarchical、geometry residual、GCN、part branch、adapter 或任何旧实验开关。

## 单变量对照

| 项目 | exp385 B0 | exp387 D0 |
|---|---|---|
| 官方 Swin-Tiny 与预训练 teacher | 相同 | 相同 |
| 数据、sampler、batch、seed | Occluded-Duke / identity / 64 / 1234 | 完全相同 |
| 输入与增强 | 384×128、flip/pad/crop/RE | RGB 逐随机状态 exact；仅同步 pose 几何 |
| optimizer / LR / epoch | SGD / 0.0008 / 120 | 完全相同 |
| ID/triplet/BNNeck/global descriptor | 相同 | 相同 |
| 额外变量 | 无 | 完整 anchor+PSG 与 pose loss |
| 测试输入 | RGB | 严格 RGB-only |

D0 output 固定为 `log/occluded_duke/exp387_clean_swin_tiny_d0_s1234`。不得覆盖、续训或引用旧 TAPF checkpoint。

## 数据边界

- pose artifact：`/mnt1/afrderived/exp386_occluded_duke_vitpose_huge_train`
- manifest SHA256：`cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8`
- 只允许 `bounding_box_train` lookup；query/gallery lookup 必须失败。
- 输入 target 是 paired augmentation 后的 17×2 坐标、原始 score 与 valid mask。
- 显式可靠性定义：`r = valid · clamp(score, 0, 1)`。这是预注册的 target 数值变换，不修改 exp386 artifact，也不得在 loader 中静默裁剪。

## 最小模块

### 1. Stage-2 pose anchor

Swin-Tiny 的 stage index 2 pre-downsample 输出为 `B×384×24×8`。先经该 stage 的既有 LayerNorm，再 reshape 为 NCHW。anchor 只含：

1. `1×1 Conv 384→128`；
2. `3×3 depthwise Conv`；
3. GroupNorm + GELU；
4. `1×1 Conv 128→34`，拆成 17 个 heatmap logits 与 17 个 pooled confidence logits。

pose loss 的 anchor 输入使用 `F2.detach()`，所以 pose objective 不回流 Swin。heatmap 用 `sigmoid` 得到 17 张 unit-range map；confidence 用全局池化后 `sigmoid` 得到 17 个值。student field 定义为二者逐关节乘积。

### 2. 现场 target renderer

将 paired keypoints 从 384×128 连续坐标映射到实际 anchor grid；每个有效 joint 渲染固定 `sigma=1.5` grid cell、峰值为 1 的 Gaussian。teacher field=`Gaussian·r`。

pose loss 在 FP32 计算：

- heatmap MSE 由 `r` 加权，只监督可靠关节并按有效权重归一；
- confidence BCE 的 target 为 `r`，无效 joint target=0；
- 两项相加后乘总权重 `0.1` 加入原 ID+triplet objective。

renderer 不保存 heatmap cache，不读取旧 pose 文件；所有 grid、Gaussian、归一化与空 valid 情形必须有限。

### 3. Stage-3 PSG bank

Stage-3 两个 Swin block 各有一个独立 PSG。每个 PSG 将 17 通道 field bilinear resize 到 consumer grid，再执行：

`Conv1×1(17→32, no bias) → GroupNorm(affine=False) → GELU → Conv1×1(32→768, no bias)`。

末层卷积 zero-init，输出 `delta`；调制为 `x · (1 + 0.5·tanh(delta))`。因此：

- 初始 PSG 为 identity；
- zero field 在任意训练时刻都严格产生 identity，不存在 bias 常量捷径；
- PSG 输入已经是 `[0,1]` field，内部不再 sigmoid；
- ReID loss 更新 Swin、PSG、BNNeck 与 classifier，但 student field 在 PSG 边界 detach，ReID 不更新 anchor。

## 训练期交接

- epoch 1–5：两个 PSG 只读 paired target renderer 的 teacher field；anchor 全程接受 pose loss。
- epoch 6–10：`student_fraction=(epoch−5)/5`，PSG 读取 teacher/student 的线性混合；e10 为 100% student。
- epoch 11–120：PSG 只读 detached student field；外部 target 仅用于持续 pose loss，不进入 ReID consumer。
- eval：无论传入 correct/shuffle/None/exploding pose 对象，代码都不得索引；只由 RGB→Stage-2→anchor→Stage-3 PSG→descriptor。

不因 e10、单一 eval 或阈值提前停止，正式 D0 必须自然跑满 e120，并以 e120 对 exp385 e120 计算 mAP/R1/R5/R10 四项显式差值。

## 默认关闭与构造一致性

`MODEL.TAPF.ENABLED=False` 时：

- 不实例化 anchor、renderer 或 PSG；
- 官方 dataloader、model 返回值、state keys、forward、loss、optimizer groups 与 descriptor exact；
- 不读取 manifest。

D0 先构造完整官方 backbone/BNNeck/classifier，保存构造后 CPU/CUDA RNG，再附加 TAPF 模块并恢复 RNG。由此 B0/D0 的公共 state、初始化和构造后 RNG 必须 exact；新增参数正常进入同一个官方 optimizer。

## 启动前门禁

1. CPU unit：Gaussian 坐标、score→reliability、空 valid、anchor shape、两个 PSG route、zero-field identity、无 bias 捷径。
2. config-off：官方 B0 state/init/RNG/forward/loss/optimizer 多步 bit-exact parity。
3. paired data：沿用 exp386 5/5 unit、32-seed RGB parity，并在 D0 collate 下复验 batch64/8 workers。
4. route/gradient：pose loss→anchor 有限非零、→Swin/PSG 精确为零；ReID→PSG/Swin 有限非零、→anchor 精确为零。
5. schedule：e1 teacher exact、e6 fraction=0.2、e10/e11 student exact；两个 Stage-3 block 各消费一次且 bank 参数独立。
6. state/optimizer：公共 state exact；新增参数无遗漏/重复；save/load strict roundtrip 后 descriptor、field 与统计 exact。
7. 原生 CUDA/AMP：真实 batch64 forward/backward/step 有限；默认 GradScaler 动态回退后连续有限更新。
8. overflow：人为制造真实 nonfinite，model/optimizer 参数整步不变且 scale 正确下降。
9. pose-free eval：correct/shuffle/None/exploding external pose 的 descriptor 逐元素 exact；query/gallery loader 不构造 pose store。
10. efficiency：记录新增参数、训练峰值显存、每 step 时间与 RGB-only eval 开销；不能把更多参数本身解释为 pose 收益。
11. execution：fresh 独立远端 repo、exact commit/full bundle/config SHA、输出不存在、GPU 空闲；任一门禁失败不得启动。

用户已明确禁止 Claude。本实验只做 design、代码自审、unit、真实 CUDA/AMP 与可执行不变量审计，不生成或调用 `claude_review.md`。

## 风险与失败解释

1. D0≤B0：说明旧结果未在官方干净实现与 fresh pose target 下复现；保留负结果，先检查语义门禁，不能复活旧代码救点。
2. pose loss下降但 D0≤B0：anchor 可学不等于 PSG 对检索有益；不能把 pose reconstruction 当 ReID 成功。
3. target→student handoff 不稳：这是实现/数值问题，正式训练前解决；不得运行中改 schedule。
4. score 大于 1：按预注册 `clamp(0,1)` 只在 reliability renderer 中处理，artifact 与日志保留原值。
5. PSG 学成近恒等：如实报告 gate/gradient/parameter trajectory；不临时增宽、加 bias 或叠加多 stage。
6. 单 seed 正增益：只能支撑 clean reproduction；是否补多 seed 在 e120 与效率证据闭合后另行设计。
