# exp404 后 CLIP–TAPF 实现根因审计

## 审计边界

本审计不重开、修改或续训 exp392–404。exp404 的 `SPK_MECHANISM_NO_GO` 只封板当前
`rich evidence -> slot mean -> descriptor group product` 对象，不外推为 CLIP 或多阶段 TAPF 总体无效。

## 先给结论

当前最强证据不支持“CLIP teacher 天生不适合人体局部任务”。更直接的事实是：进入 exp401–404
正式训练的 student evidence 到 e120 仍没有学会 frozen teacher，而 production graph 又在进入最终
descriptor 前压掉了大部分槽位身份。旧路线因此同时存在 **teacher 定义偏离、student 接口欠拟合和
consumer 代数置换不敏感** 三个问题。

这不是一个可通过调温度、调 loss 权重或重跑 seed 修复的旧臂；下一机制必须重定义训练对象。

## 1. CLIP 局部能力并非整体失败

Phase 0B 的 naive dense patch teacher 的确失败：全量五类 coarse-part top-1 很低，且不满足
correct-vs-shuffle/wrong-mask/text 门。但后续只读诊断已经排除 tensor 维度、patch 顺序、坐标映射、
`ln_post+proj` 和 prompt label 顺序错误：

- OpenCLIP token readout 与官方路径逐元素一致；
- tight region crop 走受对比学习监督的 global CLS，128 图 macro top-1 达到约 `44.7%`；
- hard-owner crop ontology 在 128 图达到约 `51.6%`；
- PC-MBCLS 对 slot 内 0/25/50/75% support deletion 的 `q_visible` 单调响应通过预注册门。

所以 naive patch 失败说明的是 **raw local patch 没有被 global image-text objective直接校准到人体部位文本轴**，
不是 CLIP image encoder、文本 encoder 或 pose/RGB 几何整体接错。

独立代码红队同时指出，旧“简单 part-name crop”并不是纯净部位分类：tight crop 只用 mask 求 bbox，
没有在 bbox 内抑制非目标像素。arms 和 upper-leg 的细长/双侧 bbox 会纳入大量 torso 或相邻腿部，足以解释
arms 近 `0%`、upper-leg 约 `16–19%` 的异常类别结果。因此不能用这两个类别直接证明 CLIP 连简单人体
部位都不认识。

PC-MBCLS 也不是完整区域编码。它先让 full-image CLS 走 20 个全局 block，只在最后 4 个 block 对
`CLS-query -> patch-key` 加 region prior；patch-to-CLS 与 patch-to-patch仍保持全局，已有 CLS residual
继续携带整图语义。这解释了“局部响应存在但动态弱”，不能与从早层开始的 region-isolated readout等同。

## 2. exp394–404 的 rich teacher 已不再是双编码器 teacher

`FrozenRichClipEvidenceTeacher` 只保留 `model.visual`。其 target 为：

```text
normalize(region_cls) - normalize(global_cls)
-> subtract per-slot mean
-> shared PCA-16
-> L2 normalize
```

该类没有构造 text prototype，也没有调用 `encode_text`。因此 exp401–404 实际蒸馏的是 **image-only、
PCA-gauge-dependent 的局部残差**，不是用户原始定义的“局部视觉 feature 与全部 body-part 文本原型形成
sample-specific distribution”的双编码器 teacher。

这条偏离很关键：PCA 第 `k` 维没有固定人体语义，后续把它机械映射到 final descriptor 第 `k` 个连续
channel group，不会自动建立“关节/部位语义 -> descriptor 子空间”的对应。

此外，region-global residual 经过 slot centering/PCA 后被逐槽 L2 normalize。若 visibility/occlusion 的
主要信息体现在 residual 幅值，这一步会主动删除它。旧 scalar support teacher 的 visible/occluded文本又共享
同一个 part phrase，只相差 generic adjective；其判别只落在很小的
`t_visible - t_occluded` 方向，并使用固定 `T=0.07` 而非 checkpoint 的 native `logit_scale`。这能解释
support q 同槽动态非常弱，但不解释 part-name argmax；两类失败不能混为一谈。

## 3. student evidence 在 e120 仍近似没有学会 teacher

exp401 formal 日志给出直接证据：

| 时点 | EvidenceCos loss | 对应平均 cosine |
|---|---:|---:|
| e1 iter20 | `0.989` | 约 `0.011` |
| e30 iter200 | `0.987` | 约 `0.013` |
| e120 iter200 | `0.973` | 约 `0.027` |

`EvidenceRel` 同期长期停在约 `0.28`。相反，mask 从 `0.657` 降到 `0.163`，presence 从 `0.734`
降到 `0.036`。这说明不是整个 anchor 或 optimizer 没工作，而是 rich evidence 目标没有被 student
接口有效吸收。

因此 exp404 终审的 `correct evidence` 不能解释成“正确 CLIP teacher state”；更准确的说法是“由弱蒸馏
student head 产生的 16 维状态”。wrong/generic/NULL 反事实最终击败它，与这条欠拟合证据一致。

## 4. evidence head 的梯度和空间接口与原始候选机制不一致

当前代码为：

```python
hidden = anchor_conv(source_feature.detach())
evidence = evidence_head(GAP(hidden.detach())).view(B, 5, 16)
```

本地只读梯度探针确认，仅对 evidence cosine backward 时：

- `evidence_head.weight/bias` 有非零梯度；
- `project/depthwise/norm` 梯度均为 `None`；
- backbone 本来就被 `source_feature.detach()` 阻断。

`source_feature.detach()` 已足以保护 ReID backbone；额外的 `hidden.detach()` 又阻止 CLIP evidence loss
塑造 anchor 表征。更重要的是，五个槽全部由同一个全图 GAP vector 经不同输出行预测，根本没有执行原始
候选中的：

```text
anchor field -> pool corresponding stage ReID feature -> slot projection -> teacher distribution
```

所以 channel shuffle、wrong field 或局部内容变化不一定改变 evidence；head 可以只学习全图相关性和五组
固定输出先验。

## 5. `semantic_valid` 被错误地当成 visibility/presence

pose artifact 的 `valid` 只检查 keypoint/score finite 且坐标仍在图内，没有 score threshold，也不表示该
关节真实可见。ViTPose 对遮挡点仍会回归一个图内坐标和正 score。region renderer 随后用
“该region任一joint的 `valid * score > 0`”定义 `region_valid`，rich TAPF 又直接执行：

```text
teacher_presence = semantic_valid.float()
```

因此这里的 presence 实际是 **geometry coordinate exists**，不是 semantic visibility/support。正式检查中
五槽 hard presence 近乎全1并非偶然，而是该定义的直接结果；wrong-mask exact no-op也与此吻合。

对 exp386 全 15,618 条 train-only artifact 的只读统计进一步确认：五个region的最大score `>0` 比例均为
`100%`；即使阈值取 `0.7`，五槽比例仍为
`98.56/98.26/98.26/96.12/94.31%`，macro=`97.10%`。因此当前
`region_valid = any(score > 0)` 几乎必然产生全presence，不能承担遮挡可靠性标签。

下一版必须拆开：

- `geometry_valid`：只决定pose field/slot坐标能否构造；
- `semantic_support`：由实际student-view RGB上的局部CLIP视觉证据或明确遮挡操作连续估计；
- `consumer_presence`：不得直接复用in-bounds keypoint flag。

## 6. teacher 与 student 的可观测 RGB 不一致

paired transform 在随机擦除前保存 `teacher_rgb`，随后对 student RGB 以 `RE_PROB=0.5` 执行 Random
Erasing。训练 teacher 始终读取擦除前 RGB，student anchor 则读取擦除后 backbone feature。

对 pose geometry，这种 privileged target 尚可解释；对 sample-specific CLIP appearance/support state，
一半样本的 target 含有 student 输入中已被删除的局部证据。若任务是 visibility/support，标签甚至与实际
student view 相反；若任务是 completion，则必须显式定义 donor/original-view target和可实现的恢复目标，
不能把不可观测残差当普通逐图回归。

## 7. SPK 在代数上再次压掉槽身份

SPK 先做：

```text
pooled = sum_r evidence_r * presence_r / sum_r presence_r
factor = 16 * softmax(pooled)
```

正式审计中五槽 hard presence 对前 128 图全部为 1。此时 slot-cycle 对均值严格置换不变，wrong-mask 也
成为 no-op。随后 `factor[k]` 只缩放 final descriptor 的第 `k` 个连续 group；它既没有保留 slot index，
也没有 pose-defined gather/scatter。正式九臂 correct 与 wrong-RGB 只差约 `0.0019 mAP`，NULL/bypass
反而高约 `0.1809 mAP`，正是该压缩路径伤害排序的最终证据。

## 8. 实现错误与机制错误的区分

### 已排除的机械错误

- RGB/pose 同步 resize、flip、pad、crop；
- COCO-17 左右翻转 index；
- CLIP normalization、bicubic resize、16x16 grid；
- `ln_post/proj` 和 official global CLS readout；
- frozen teacher、checkpoint、optimizer与 eval 隔离；
- final consumer 的 forward 可达性。

### 仍成立的实现/设计缺陷

1. rich 路线从双编码器退化为 image-only PCA target；
2. student evidence 不做 pose-defined local pooling；
3. 多余 `hidden.detach()` 让 evidence loss 只能训练最后一层线性头；
4. pre-RE teacher 对 post-RE student 产生不可观测的 appearance/support target；
5. SPK 先对槽求均值，数学上消除 channel identity；
6. arbitrary PCA axis 与 arbitrary descriptor group 被直接一一绑定；
7. rich code 的逐槽 L2 normalization 删除可能承载support/occlusion的幅值；
8. PC-MBCLS只在最后4层做CLS局部重读，旧crop参考又混入bbox内非目标像素；
9. in-bounds keypoint `valid` 被直接当成semantic presence，导致五槽近乎全1。

## 9. 下一机制的冻结要求

下一条不能叫“修 exp404”，也不能只删除一个 detach 后直接 e120。它至少必须满足：

1. **可观测训练对象**：同一 student view 的 CLIP target，或显式的 original/masked/donor 三元组；
2. **双编码 teacher**：pose-conditioned CLIP visual readout与冻结 text prototype共同定义状态；
3. **slot-local student**：每个 anchor 用自己的 field 从对应 stage feature 池化，而非全图 GAP；
4. **槽身份先执行后聚合**：correct slot、wrong slot、channel cycle 必须在代数上产生不同输出；
5. **中间层 TAPF 执行**：gather/transform/scatter真实改变最终 global descriptor，禁止 terminal dead consumer；
6. **推理边界**：删除 CLIP、文本、外部 pose、donor，只保留 RGB student 和一个固定 descriptor；
7. **teacher oracle 先行**：正式训练前，correct same-slot target必须明显优于 wrong-slot、wrong-ID、generic、
   NULL 与 pose-only/image-only控制。

当前首选训练对象是 **CLIP 校准的反事实跨视角缺失槽运输**：训练时用同身份、不同相机的可见同槽 donor
为人工遮挡 recipient 提供可执行中间残差；student 从单图剩余支持预测该残差。它能同时修复“teacher target
不可观测”“槽位先被平均”“CLIP只停在辅助 loss”三项根因。正式实现前仍需完成近期代码近邻审计与
train-only teacher-forced oracle。
