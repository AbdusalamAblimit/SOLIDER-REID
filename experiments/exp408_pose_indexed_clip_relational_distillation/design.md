# exp408：PICRD（Pose-Indexed CLIP Relational Distillation）

## 动机

exp402 已证明旧 `C0` 的 sample-specific semantic interface 不成立；exp404 又证明 SPK 的
final-factor 绑定有害。代码审计进一步定位到两个实现事实：旧 rich evidence 从全图 GAP 生成，且
`source_feature.detach()` 与 `hidden.detach()` 同时阻断 CLIP relation loss 对 backbone 的训练。因此
exp408 不再设计 router、donor 或推理期语义分支，而是把对象改为：**用 pose 定义的局部 CLIP 外观关系，
直接训练形成最终 global descriptor 的 Stage-2 表征**。

## 核心假设

同一解剖槽内，不同训练图像之间的 CLIP 外观关系包含遮挡下仍可迁移的身份结构。若学生 Stage-2
局部状态确实学到该结构，则它与正确 RGB 的 CLIP 关系矩阵应比 wrong-RGB、generic 和 zero 更接近，
同时梯度必须直接进入 Stage0--2；这应在不增加测试路径的情况下提高自然 ReID mAP/R1。

## 技术方案

### 1. Fresh 五槽 CLIP teacher cache

- 只读 official Occluded-Duke train RGB 与冻结 pose；不读取 exp405--407 的 cache、pair 或结果。
- taxonomy 固定为 `head / upper_torso_arms / lower_torso / upper_legs / lower_legs_feet`，与
  exp405 冻结 COCO-17 绑定一致。
- frozen CLIP ViT-L/14 从第一个 block 开始做 region-isolated visual readout，保存每图
  `[5,768]` raw normalized visual embedding 与 `[5]` geometry/readout valid。
- cache 由 official `relative_path` 唯一索引，必须完整覆盖 15,618 个 train path。
- canonical teacher view固定为：raw RGB/raw pose同步deterministic resize到`384x128`，不做
  flip/pad/crop/random erasing；随后渲染region mask并做CLIP居中letterbox。builder不得自行更换预处理。

### 2. 未 detach 的 Stage-2 逐槽状态

训练时用经过同一 flip/pad/crop 的 pose 在 Stage-2 网格重新渲染五槽 hard-owner mask，得到
`M in R^(B,5,H,W)`。对未 detach 的 `source_feature in R^(B,384,H,W)` 做 mass-normalized pooling：

`S[b,r] = sum_hw M[b,r,h,w] F[b,:,h,w] / sum_hw M[b,r,h,w]`。

cache valid 与当前增强后的 geometry valid 求交。PICRD 不引入投影头，学生 384 维与 teacher 768 维
只通过 cosine relation 比较。

### 3. 逐槽跨 batch relation 与反事实排序

禁止把 `B x 5` 展平成一个集合。对每个槽 `r`，分别构造有效样本间的 off-diagonal cosine Gram：

`G_s^r = norm(S[:,r]) norm(S[:,r])^T`，`G_t^r` 同理。

`d(T)` 是五个有效槽的 `MSE(G_s^r, G_T^r)` 均值。四个 teacher arm 为：

1. `correct`：当前 relative path 对应的正确 CLIP target；
2. `wrong-RGB`：identity sampler批内固定 cyclic offset=`4`的different-PID target；
3. `generic`：每槽有效 teacher 均值广播到各行；
4. `zero`：全零 target。

训练目标不设 temperature 或新 margin：

四臂必须共享完全相同的 `V_common=post-transform geometry valid ∩ correct cache valid ∩ wrong cache valid
∩ generic slot available`；每槽不足2行时，该槽对四臂共同排除。zero在`V_common`上仍是有效零向量，不能因
norm为零被删掉。四臂使用同一slot集合和off-diagonal pair mask。

`L_picrd = d(correct) + mean_n softplus(d(correct) - stopgrad(d(n))), n in {wrong,generic,zero}`。

负臂只提供冻结阈值，梯度不通过control主动把student推远；因此训练后的强control不会被“最大化负臂”循环构造。
`L_picrd` 加入 D0 原 pose auxiliary object，并沿既有
`POSE_LOSS_WEIGHT=0.1`进入总 loss；不调 batch、rho、scale 或既有 loss 权重。

### 4. 推理路径

eval 不加载 cache、不读取 CLIP、不读取外部 pose。PICRD 没有测试期 head/branch；最终 descriptor、PSG
和分类器路径与 clean D0 完全相同。其执行对象是被训练改变的 Stage0--2 backbone state。

## 对照与裁决

- 唯一性能对照：sealed clean D0 seed1234，自然 e120 raw=
  `57.5587756578 mAP / 67.6923076923 R1`。
- 每 e10/20/.../120 并排记录方法、D0、`delta mAP/R1`；不中途早停。
- 训练前正合同必须满足：finite、逐槽而非跨槽 Gram、different-PID wrong、
  `d(correct)<d(wrong/generic/zero)` 的可构造正例，以及 `L_picrd` 对 Stage-2 source/backbone 的非零有限梯度。
- 自然 e120 只有 mAP 与 R1 **同时严格高于** clean D0 才判性能 GO。
- cache builder同时冻结一份16 PID×4图、PID内连续排列的diagnostic relative-path manifest及SHA；e120只能在
  该确定性resize view与固定offset=4上复核四臂，不得重新随机采样。
- 首批真实 batch 记录四臂距离；若全程没有形成 correct 最小顺序，只解释为 binding 未学成，不能靠删除
  control 或改温度/权重救臂。

## 创新边界

近期查重没有发现“pose 五槽 region-isolated CLIP teacher + 逐槽跨 batch relation + 训练内强反事实排序 +
Stage-2 直传标准 global descriptor”的同构实现。最接近的 π-VL、ProFD、PAFormer、KPR、MUVA 与
CVPR 2026 Composite-Attribute ReID 已覆盖多数原子，因此本实验只按 **C 类候选** 定位。若实现退化成
per-slot cosine/KL、feature add 或测试期 part matching，机制创新门立即失败。

## 风险与失败解释

1. canonical cache RGB 与随机增强后的 student RGB存在边界；若 correct 顺序不成立，说明 teacher target
   对当前 view 不可辨识，封板 PICRD，不改 cache/augmentation 救旧臂。
2. batch relation 可能被 PID sampler 或近常量部位先验主导；共同valid支持、逐槽 off-diagonal 与
   wrong/generic/zero 是
   专门排除该 shortcut 的证据。
3. 若 relation 顺序成立但 e120 不涨点，说明 CLIP 关系可学但不改善 ReID 表征，机制仍判 NO-GO。
4. 若涨点但 correct 不胜强 control，只能记作额外训练正则收益，不能声称 pose-CLIP binding。
