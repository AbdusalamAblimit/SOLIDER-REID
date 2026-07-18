# exp392 Phase 0 零训练协议：内部语义诊断与 CLIP teacher 可辨识性

> 状态：**PHASE 0A SEALED / PHASE 0B AUTHORIZED / NO FORMAL TRAINING**
>
> 日期：2026-07-18
>
> 本文冻结审计对象、干预 seam、对照和裁决规则。用户于2026-07-18明确授权开始后，Phase 0A
> 只读封板 checkpoint 审计进入执行；仍不创建训练 config、不修改封板实现、不启动正式训练。

## 一、Phase 0 要回答的两个问题

### Phase 0A：现有 clean D0 的内部 field 是否真的具有 joint-channel 语义

exp390 只证明 D0 的 anchor、两个 PSG consumer、参数轨迹和 RGB-only 推理边界均真实；它没有回答：

> 在保持 field 数值分布和空间位置不变时，只打乱“第几个通道代表哪个关节”，最终 global descriptor
> 是否会显著变差？

如果 correct、joint-channel permutation 和 matched-wrong field 对 descriptor 的影响接近，而 PSG bypass
有明显影响，则更准确的结论是：field 是可执行条件扰动，但 joint identity 尚不可辨识。Phase 0A 是对
当前机制的冻结诊断，不是新训练实验，也不以某个中途 checkpoint 代替封板 final。

### Phase 0B：候选 CLIP 双编码 teacher 是否提供 sample-specific anatomical evidence

pose 已经规定 coarse region 的名称。CLIP 只有在同时依赖当前 RGB、当前区域 mask 和 text prototype
bank 时才提供新增信息；若输出只是由 region index 决定的近常量 one-hot，它不能解决 TAPF 的语义
不可辨识问题。

Phase 0B 必须在训练前证明：

1. frozen CLIP patch tokens 在正确 pose region 内能识别相应 coarse body part；
2. teacher distribution 会随当前实例的局部视觉证据、遮挡和可靠性变化；
3. wrong RGB、wrong mask、channel shuffle 和 wrong text 会产生方向正确的退化；
4. 水平翻转时，RGB、mask 和 left/right 语义映射满足 equivariance；
5. 结论不依赖历史 exp356 的固定水平条带实现。

## 二、历史 exp356 的可复用教训与不可复用边界

只通过 `git show` 审查历史 commit `d5273e5b3ccc9300b0d74815bdbca8d2eeae56b9`，不得导入或运行
其旧 runtime。

已确认历史 `CLIPVisualEncoder`：

1. 使用 frozen OpenCLIP ViT-L/14，hook 最后一层 transformer resblock；
2. 丢弃 CLS，对 16×16 patch tokens 做 `ln_post` 和 visual projection；
3. 重载 `train()`，父模型进入 train 时仍强制 visual encoder 保持 eval；
4. 把 384×128 ReID tensor 直接双线性拉伸到 224×224；
5. 不用 pose mask 池化 CLIP tokens，而是固定按 top-5 / middle-6 / bottom-5 patch rows 划分
   head/torso/legs；
6. pose 只决定 ReID feature map 中要 mask 的水平 region；CLIP target 与该实例的真实关节区域没有
   一一几何绑定；
7. teacher 接收的是 dataloader 已 normalize、且可能已 RandomErasing 的 student tensor，再做
   un-normalize / CLIP normalize。

历史 final：`exp356 pose-mask=57.1 mAP`、`exp356r random-mask=57.3 mAP`、直接对照
`exp341=59.8 mAP`。它封板的是“固定水平三段 CLIP completion + pose 选 mask”机制。它没有测试：

- pose mask 对 CLIP patch tokens 的实例级区域池化；
- image+text 双编码形成的 sample-specific part distribution；
- 只校准 executable anchor state、而不重建 CLIP feature；
- semantic expert router 的 geometry-semantic mismatch sensitivity。

因此 exp356 是强风险证据和必要反例，不是 exp392 的等价实现或永久否定证据。

## 三、Phase 0A：clean D0 内部 student-field frozen audit

### 3.1 固定对象与不变量

- 对象：已封板 `exp387 clean Occ-Duke D0-s1234` 的唯一 final checkpoint；执行前补录
  config/checkpoint/repo HEAD SHA，并 strict load；
- 模型全程 `eval + no_grad`，不构造 optimizer，不改参数或 buffer；
- query/gallery 仍为 RGB-only，所有 external pose 变体必须逐元素 exact；
- 每个 arm 使用同一个 strict-loaded model、同一个固定 query/gallery 顺序和同一 evaluator；
- `correct_start` 与 `correct_end` 的 descriptor、field digest 和四项检索指标必须 exact；
- 每个临时 hook 在 arm 结束后移除，模型 state SHA 前后 exact；
- 不复用 exp378 的旧 tuple output 或固定 `17×96×32` runtime contract，只借鉴短生命周期 hook、
  donor map 和 digest 审计方法。

### 3.2 clean D0 的正确干预 seam

clean D0 不是调用 `tapf.forward()`，而是：

```text
Stage-2 source -> tapf.anchor(source.detach()) -> prediction dict
prediction["field"] -> prepare() 中的 consumer_field -> 两个 Stage-3 PSG -> global descriptor
```

因此正式 Phase 0A seam 必须位于 `model.base.tapf.anchor` 的 forward output，或等价地临时包装
`tapf.prepare()` 的返回 dict；不得照搬 exp378 对旧 TAPF tuple 的 hook。

首选 anchor-output hook 契约：

1. 输入/输出仅调用一次，输出必须含 `heatmap_logits/confidence_logits/heatmaps/confidence/field`；
2. 只替换 `field`，其余四项逐元素保持；
3. 替换后的 field 与原 field shape、dtype、device、contiguous、finite 和非负边界一致；
4. eval 中 `consumer_field` 必须来自替换后的 field，且两个 PSG 均实际消费；
5. 记录原/替换 field SHA、调用次数、最大绝对差和 hook removal；
6. donor capture 与 recipient replacement 不得同时递归进入 hook。

在一批 synthetic RGB 上先做 seam 单元证明；未证明“field 已换、descriptor 路径已达、参数未变”前，
不得执行全验证集审计。

### 3.3 预注册 arms

按以下固定顺序，每个 arm 独立从同一个 sealed checkpoint 评估：

1. `correct_start`：原始 internal student field；
2. `external_correct`：传入正确 external pose；
3. `external_shuffle`：batch 内滚动 external pose；
4. `external_none`：无 external pose；
5. `external_exploding`：任何索引都会报错的 sentinel；
6. `channel_cycle`：17 通道循环移位一位，空间与全部元素 multiset 不变；
7. `left_right_channel_swap`：按 COCO 左右关节 pair 交换通道，但不做空间反翻；
8. `confidence_permutation`：每通道归一空间 shape 后，只置换 channel peak amplitude；
9. `matched_wrong_field`：同 split、不同 PID 的 donor RGB 产生 field，再替换 recipient field；
10. `spatial_constant`：每通道空间均值广播，保留 channel mean、删除 geometry；
11. `zero_field`：全零 field，验证 PSG 的 NULL 行为；
12. `psg_bypass_each`：分别 bypass 两个 PSG；
13. `psg_bypass_all`：两个 PSG 同时 exact identity；
14. `correct_end`：重复原始路径。

`matched_wrong_field` donor map 必须在运行前固化为 artifact，并满足：

- query/gallery 不跨 split；
- donor 与 recipient 不同 PID、无 fixed point；
- 优先同 camera，并按 field 总质量、有效通道数和上/中/下空间质量三元组做最近邻匹配；
- 报告未能同 camera 匹配的比例和匹配前后统计差；
- donor map 与 metadata 记录 SHA，不允许为结果临时重配。

### 3.4 指标与解释规则

每个 arm 报告：

- mAP/R1/R5/R10 及相对 `correct_start` 的四项差值；
- 每样本 descriptor cosine、L2、最大绝对差的均值/中位数/95%分位；
- field 与两个 gate delta 的 SHA、范数、非零比例和相对 correct 差异；
- bootstrap 95% CI；
- hook call/removal、state SHA、external pose access count。

Phase 0A 的语义诊断阈值固定为：

- 若 `correct - channel_cycle` 与 `correct - matched_wrong_field` 均 `<0.3 mAP`，同时
  `correct - psg_bypass_all >=0.3 mAP`，记为“consumer 有效但 joint semantics 未识别”；
- 若任一语义错配造成 `>=0.3 mAP` 且 bootstrap 方向稳定，再检查 left/right swap 和 descriptor
  扰动，只有至少两种 semantic mismatch 同方向才记为“现有 D0 已有可辨语义”；
- 若 bypass 也 `<0.3 mAP`，结论是 final descriptor 对当前 TAPF 整体效应太小，不能用语义审计
  证明或反驳 binding；
- `zero_field` 不要求 descriptor 等于 correct；但它的 descriptor 必须与 `psg_bypass_all` 逐元素
  exact，因为当前 PSG 无 bias、GroupNorm 无 affine，零 field 的数学路径就是 exact identity。

该诊断不会单独授权或阻止 exp392 训练。它只冻结当前问题是否真实，以及后续必须超过的 counterfactual
证据下限。

## 四、Phase 0B：coarse-region CLIP 双编码 teacher-only audit

### 4.1 数据边界与几何所有权

- 只读 official Occ-Duke train split 和 exp386 fresh train-only pose artifact；绝不为 query/gallery
  读取或生成 pose；
- 执行前冻结样本 manifest、图像 SHA、pose manifest SHA、CLIP 权重/tokenizer/prompt SHA；
- Phase 0B 不更新任何模型参数；CLIP image/text encoder均
  `requires_grad=False + eval + no_grad`；
- teacher RGB 与 student 必须复用同一次 resize/flip/pad/crop 参数，禁止各自重新采样几何；
- `PairedPoseTransform` 当前只返回 normalize+RandomErasing 后的 RGB。未来若进入实现，必须显式暴露
  `post_geometry_pre_erasing_rgb` 和同一几何记录，而不是从已归一 tensor 逆推“干净图”；
- 主 teacher 使用 `post_geometry_pre_erasing_rgb`；同时报告 `post_erasing_rgb` 控制，防止把收益
  偷换成 clean-view augmentation KD；
- 不得缓存旧 pose path mapping；所有 pairing 由 image relative path、image SHA 和 exp386 manifest
  三重确认。

### 4.2 CLIP token 几何方案

历史 224×224 square-stretch 不能默认为正确。Phase 0B 在 teacher-only 阶段同时报告两个预注册、
不训练的几何读法：

1. `square-stretch`：RGB 和 pose region mask 使用同一 384×128→224×224 affine stretch，保留
   16×16 patch 数但扭曲人体比例；
2. `aspect-letterbox`：等比例缩放到 224 高，左右 padding 到 224；mask 使用同一 affine+padding，
   保留人体比例但有效宽度 patch 较少。

两者均需记录 image→CLIP grid 的显式变换矩阵，并用 synthetic point/mask 做像素级对齐检查。不得只看
teacher top-1挑版本：主版本必须同时在 correct-vs-wrong mask、flip equivariance 和有效 patch coverage
上占优；若三者结论冲突，Phase 0B 不通过，先重新定义 CLIP input resolution。

### 4.3 固定 anatomical ontology

首版只用五个 coarse semantic channels，不直接做 17-joint text KD：

1. `head_face`：nose/eyes/ears及相邻 head support；
2. `torso`：shoulders/hips与 shoulder-hip torso segments；
3. `arms_hands`：shoulder-elbow-wrist chains；
4. `upper_legs`：hip-knee segments；
5. `lower_legs_feet`：knee-ankle segments。

mask 由固定 COCO-17 incidence matrix、joint Gaussians和limb segments聚合；边界 joint 可同时为相邻
region 提供 support，但 region channel顺序、sigma、segment width和归一方法在执行前一次冻结。
invalid joint不贡献，空 region 的 pooling posterior严格为零并打 invalid 标记，不用 epsilon 伪造视觉
feature。

每类 text prototype 使用同样数量、同样模板结构的冻结 prompt ensemble；不得针对结果单独改某一类
prompt。主分布固定为：

```text
v_r = L2Norm(mask-normalized pool(frozen CLIP patch tokens, region r))
t_c = L2Norm(mean(frozen CLIP text embeddings for class c))
q_r = softmax((v_r @ t_all) / 0.07)
```

`0.07` 是预注册主温度；允许把 native CLIP logit scale 和 `{0.03, 0.05, 0.10}`仅作为敏感性附表，
不得据此挑选训练版本。若 q 在主温度近饱和或近均匀，teacher 定义直接失败，先修表示而不是调温救场。

### 4.4 teacher-only arms

每个样本/region 固定报告：

1. `correct_full`：正确 RGB + 正确 mask + 正确 text bank；
2. `repeat_exact`：相同输入重复，输出必须逐元素 exact；
3. `wrong_rgb`：固定 recipient mask/text，换为同 split、不同 PID、几何统计匹配的 donor RGB；
4. `wrong_mask`：固定 RGB/text，换为面积、纵向中心和pose confidence匹配的 donor mask；
5. `channel_shuffle_mask`：同一组 masks只循环置换 semantic channel标签；
6. `wrong_text`：固定 RGB/mask，仅循环置换 prototype label解释；
7. `uniform_mask`：各 valid region使用相同person-support mask；
8. `fixed_bands`：历史 top/middle/bottom 扩展到五段的非pose空间控制；
9. `text_only_constant`：每个已知 region使用其固定 one-hot/平均prototype分布，不读 RGB；
10. `image_only_cluster`：冻结 CLIP local visual feature 的容量匹配无文本聚类，仅作强控制，不进入
    主 teacher；
11. `pre_erasing_vs_post_erasing`：同几何下比较人工擦除是否改变 teacher；
12. `horizontal_flip`：RGB、mask空间反翻并做 COCO left/right映射，输出反映射后比较。

wrong RGB/mask donor map 与 Phase 0A 一样必须先固化、不同 PID、无 fixed point并记录 SHA；不得按
teacher 分数挑 donor。

### 4.5 必报指标

1. 五类 expected top-1、macro top-1、expected-class margin、per-class confusion；
2. `correct_full` 相对 wrong-mask/channel-shuffle/uniform/fixed-bands 的 paired margin 和 top-1 差；
3. 固定 mask 时 correct-vs-wrong RGB 的 q-JSD、cosine和entropy变化；
4. 固定 RGB 时 correct-vs-wrong mask 的同组差异；
5. wrong-text 后按原 label计算的退化，以及按置换 label计算的反向一致性；
6. 同 region 跨样本 q 方差、平均 pairwise JSD、centered effective rank，排除近常量；
7. visible / low-confidence / invalid / synthetic-erased 分组的entropy、margin和可靠性；
8. pose confidence、CLIP margin与teacher reliability之间的Spearman相关及bootstrap 95% CI；
9. flip 后 region mask IoU、q-JSD、top-1一致率和left/right映射错误率；
10. 两种CLIP几何下每region有效patch数、空region率和mask泄漏到padding的比例；
11. 所有汇总的bootstrap 95% CI、样本数、invalid处理和artifact SHA。

### 4.6 Phase 0B kill-switch

以下任一成立，当前 teacher 定义记为 `NO-GO`，不得进入训练；这只否定当前粒度/预处理，不永久否定
CLIP语义校准方向：

1. `correct_full` macro top-1 的 bootstrap lower bound 不高于20%随机水平；
2. correct expected-class margin不为正，或不优于matched wrong mask/channel shuffle；
3. wrong RGB、wrong mask或wrong text中任一关键干预对 q 近乎不敏感，而 repeat_exact 已证明数值稳定；
4. 同一region跨样本 q 的 centered effective rank `<2`，且平均pairwise JSD `<0.01`，说明近常量；
5. low-confidence/invalid/synthetic-erased组没有更高entropy或更低margin，且置信区间不支持预期方向；
6. flip 反映射后 expected-class top-1一致率 `<95%`，或有效region的median q-JSD `>0.02`；
7. square-stretch 与 aspect-letterbox 均不能同时通过mask sensitivity、flip和patch coverage门禁；
8. `text_only_constant` 能复现 full teacher 的语义分组与可靠性指标，或 `image_only_cluster` 能复现
   full teacher 的sample-specific结构与后续可用信息，无法证明双编码交互；
9. pre-erasing teacher显著强于post-erasing，但这种差异完全由RandomErasing区域决定；此时必须增加
   clean-view KD强对照，不能把潜在增益归因 anatomical semantics。

Phase 0B 通过只授权 Phase 0C 单元实现，不授权正式120-epoch训练。

## 五、Phase 0C：实现前的最小单元门禁

Phase 0A/0B完成且文档封板后，才允许设计并执行 0C；当前仍不创建代码。0C 至少固定：

1. config-off 与 clean D0 forward/state/RNG/optimizer逐字节 exact；
2. frozen CLIP image/text encoder始终 eval、无梯度、无 optimizer state；
3. teacher RGB/mask共享同一几何记录，RandomErasing前后边界可审计；
4. `NULL state -> exact identity`，不得有 bias、affine norm或静态常量更新；
5. semantic loss只更新 anchor/state head，不回流 backbone；
6. ReID loss通过推理保留的 gather-transform-scatter router更新 backbone/router，但 `M/q/r` 在
   consumer前 detach；
7. correct geometry-semantic binding 与错配 binding产生不同router output；同步置换 slot枚举时输出
   保持不变；
8. generic low-rank adapter、expert-mean、static-state和bypass均有参数/FLOPs匹配控制；
9. 单stage两个consumer最终descriptor可达且无terminal dead consumer；
10. 推理删除CLIP、text、external pose后只保留RGB→student state→router→单一global descriptor。

0C全部PASS后，仍需另写正式单变量 design、fresh execution边界和最终裁决规则，才能首次启动
semantic single-stage；semantic multi-stage继续保持 `NO-START`。

## 六、当前裁决

- Phase 0A已完成并封板，裁决为
  `CONSUMER_EFFECTIVE_JOINT_SEMANTICS_NOT_IDENTIFIED`；
- channel-cycle相对correct为`+0.024 mAP`，matched-wrong为`−0.005 mAP`，均远小于`0.3`门槛；
- all-PSG bypass为`−1.359 mAP`，证明consumer有效；spatial-constant反而`+0.346 mAP`，精确空间
  geometry不是当前收益来源；
- zero-field与all-bypass逐元素exact，external pose四臂与correct repeat逐元素exact，state SHA前后
  一致；运行结束GPU恢复`2 MiB / 0%`；
- Phase 0B teacher-only实现与执行获授权；当前仍不创建训练config，不启动120-epoch正式训练。
