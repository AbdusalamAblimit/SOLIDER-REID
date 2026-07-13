# 实验 exp371：CASD（Cross-instance Anatomical Support-Advantage Distillation）

## 状态

- 当前阶段：大调研与设计冻结
- 正式训练：未启动
- 当前只允许：归因复核、缓存特征 support oracle、冻结 backbone kill-switch
- 禁止：直接完整训练、PBSR 小变体、Claude 审查、OT/MoE/slot 救场

## 动机

LGPA 与 PAFormer 的重合不能靠换 query 名称解决。本轮查新还发现：

- PAFormer 已覆盖 pose-supervised part tokens 与 pose-free inference；
- PGFL-KD/TSD 已覆盖普通 pose/parsing teacher → pose-free student；
- ProFD 已覆盖 CLIP part prompts、dense alignment 与 hybrid decoder；
- UMTS 已覆盖 multi-shot comprehensive teacher → single-shot student；
- 2022 年已有题名完全相撞的 Pose-guided Counterfactual Inference。

但仓库证据仍保留一个重要事实：LGPA 的 pose-localized local descriptors 确实稳定涨点。与此同时，`exp109` 指出遮挡 ReID 的根问题是 single-image support incomplete。

训练数据通过 P×K sampler 天然给出同一身份的多张图。某一张图不可见的部位，往往在同 ID 的另一张图中可见。CASD 因而不再问“pose 在单图哪里注入”，而问：

> 能否用 pose 在训练期组织同 ID 跨图的互补可见证据，构成 leave-one-view-out 的完整 identity support，再让单图、无姿态 student 学会吸收这种 support？

## 核心假设

相对于 same-image pose teacher，其他同 ID 视图提供了当前图真正缺失的可见解剖证据。若 support：

1. 按部位可见性聚合；
2. 严格排除 anchor 当前图；
3. 只以 identity relation/prototype 形式监督 student；

则 image-only student 可以学到对遮挡更稳健的局部身份关系，同时避免 current-view feature copy 和测试时 support/pose 依赖。

## 技术方案

### 1. Detached pose-aware teacher

保留已有 LGPA part extractor 作为训练期 teacher：

```text
T(x_i, H_i) -> {t_i,k, v_i,k}_{k=1..K}
```

- `t_i,k`：第 k 个 teacher part descriptor；
- `v_i,k`：由 pose/heatmap 得到的可见性与可靠性；
- teacher 输入 backbone features 必须 detach；
- CLIP text 只可用于兼容已有 checkpoint，不进入贡献；完成 query 归因后优先换 fixed random/learned slot ID；
- GCN、MaxSim、matching 不参与主方法。

### 2. Leave-one-view-out identity support

对 batch 中身份 `y` 的 anchor `i`，只使用其他同 ID 图像：

```text
S_i,k = {t_j,k | y_j = y_i, j != i, v_j,k >= threshold}
```

按 coverage-aware 权重聚合：

```text
c_i,k = sum_j w_j,k * t_j,k / sum_j w_j,k
m_i,k = 1[sum_j w_j,k > 0]
```

其中：

- `j != i` 是硬门禁，严禁当前图泄漏；
- 某 part 在其他同 ID 图像中均不可见时，`m_i,k=0`，该 part 不产生 distillation loss；
- 首验只使用 P×K batch 内其他 3 张同 ID 图，不引入 memory queue，避免第二变量；
- support target 全部 stop-gradient。

### 3. Image-only student

student 只接收当前图 spatial features，不读取 pose：

```text
F_i = Backbone(x_i)
{s_i,k} = StudentPartRouter(F_i)
z_i = concat(global_i, pooled_student_i, s_i,1..K)
```

第一阶段保留现有 `equal_concat` 与标准 cosine，确保只验证 CASD supervision；不做 PBSR 式 slot write-back，不用特殊 matching。

student router 可用 canonical zero-residual 初始化，降低从零学习 part routing 的风险；但 canonical/residual 本身不作为贡献。

### 4. Support-advantage distillation

不直接强迫当前图重建不可见部位像素或完整 multi-shot feature。首先同时计算：

```text
q_support_i,k = classifier_k(c_i,k)
q_self_i,k = classifier_k(t_i,k)
p_student_i,k = classifier_k(s_i,k)
```

其中 `t_i,k` 是 anchor 当前图自己的 detached teacher part。定义 support 相对 same-image teacher 的身份 margin 改善：

```text
adv_i,k = relu(margin(q_support_i,k) - margin(q_self_i,k))
```

只在 `adv_i,k>0` 且 support 存在时迁移 knowledge：

```text
L_adv = sum_i,k m_i,k * adv_i,k
        * KL(stopgrad(q_support_i,k) || p_student_i,k)
L_rel = advantage_weighted_relation({s_i,k}, batch identity supports)
L = L_id + L_tri + lambda_a * L_adv + lambda_r * L_rel
```

`L_rel` 只匹配 support 相对 self 真正修正的 batch identity ordering，而不是完整 feature MSE。这样 student 学的是其他视图带来的增量身份关系，不被迫猜测不可见衣物的具体纹理，也不退化为 UMTS 的 full multi-shot feature distillation。

### 5. Pose support 的因果门禁

support 构造必须有四种冻结控制：

```text
Scorrect  : 正确 pose/visibility
Suniform  : 每个 part 等权
Sshuffled : batch 内错配 pose
Swrong-id : 用错误身份的互补 support
```

这些干预只用于证明 pose 是否真的提高 support 纯度/覆盖度，不将“counterfactual pose inference”写成创新。

## Phase 0：训练前廉价门禁

### Gate A：LGPA query 归因

在相同 correct pose 下比较：

1. frozen CLIP queries；
2. fixed random queries；
3. learned query IDs。

若 random/learned 与 CLIP 持平，CASD 正式去 CLIP 化。

### Gate B：inference pose intervention

同一 `exp336` checkpoint 完整报告 correct/canonical/shuffled/uniform/no-pose，补齐现有训练期 shuffle 不能回答的测试干预边界。

### Gate C：support oracle

在缓存 teacher parts 上比较：

1. same-image teacher；
2. leave-one-view-out correct support；
3. uniform/shuffled/wrong-id support。

至少统计：

- part coverage；
- teacher ID CE/accuracy；
- batch-hard positive/negative margin；
- per-query teacher advantage；
- advantage 是否集中在极少数 query。

correct support 必须同时优于 same-image teacher 和最强伪 control，否则 CASD 的 support-advantage/pose-specific claim 直接 NO-GO。

### Gate D：同维压缩 oracle

在 frozen teacher descriptors 上测试 5376-D→768-D 的可复现上界。若不能保留原 LGPA 相对 global 增益的至少 80%，同维版本不与 CASD 首验捆绑。

## Phase 1：冻结 backbone kill-switch

所有 arms 使用同一缓存 spatial features、同一 student 初始化与同一 evaluator：

| Arm | 监督 | 回答的问题 |
|---|---|---|
| B0 | image-only student，无 support distill | 主控制 |
| KD0 | same-image pose teacher KD | 是否只是普通 pose KD |
| C0 | CASD correct support | 主方法 |
| Cx | CASD shuffled/uniform support | pose 组织是否必要 |
| CL | CASD correct，但允许 current-view | leave-one-out 是否必要；只作泄漏对照 |
| UM0 | correct support，但蒸馏完整 support feature | 是否只是 UMTS 式 multi-shot KD |

GO 必须同时满足：

1. `C0 - B0 >= +0.8 mAP`；
2. `C0 - max(KD0, Cx, UM0) >= +0.5 mAP`；
3. `CL` 即使更高也不计主结果，且必须显示 current-view leakage 的虚高风险；
4. pose-free student 相对 matched global 至少恢复原 LGPA `+0.9 mAP` 的 80%；
5. 测试时改换 heatmap，student descriptor 必须逐元素不变。

任一失败即停止，不加 queue、不扫温度、不改 slots。

## Phase 2：完整单 seed

只有 Phase 1 通过后，才在同一 4090、同运行时、同 seed、同 batch size、同增强与同 schedule 下做完整训练。

主门禁：

- epoch 60 `C0 - matched B0 >= +0.8～1.0 mAP`；
- `C0 - strongest non-CASD control >= +0.5 mAP`；
- student 前向与 eval 不读取 pose；
- 主结果不用 MaxSim、re-ranking 或 GCN。

## Phase 3：扩展验证

只有完整单 seed 通过后：

1. Swin-Tiny 三 seed；
2. ResNet-50 同 seed A/B；
3. 普通 ViT 同 seed A/B；
4. Occluded-Duke 主表，Market1501 与其他 PSG 原文数据集做通用性；
5. correct/uniform/shuffled/wrong-id 完整 support 因果矩阵；
6. 再评估 768-D packed descriptor；
7. 报告 teacher 仅训练期带来的训练开销与 pose-free 推理成本。

普通 ViT 必须重新选择可形成空间局部的中间层或多层 token；`exp335` 已表明直接使用末层 detached tokens 不成立。

## 必要消融

只允许四项核心消融：

1. cross-image support → same-image teacher；
2. correct pose organization → uniform/shuffled；
3. leave-one-view-out on → off；
4. support relation distill → 普通 feature KD。

不做 slot 数、mixer 深度、OT、MoE、visibility gate 或 loss 权重网格。

## 风险与失败解释

1. **跨图 support 对 unseen ID 不可迁移**：student 可能只在训练 identity 上记 prototype。必须由 test IDs 的 ReID 指标裁决。
2. **同 ID 其他图不互补**：P×K batch 中三张图可能仍缺同一 part；无 coverage 时跳过 loss，不允许用零向量伪补全。
3. **current-view leakage**：任何 support 包含 anchor 自身都会退化为 trivial self-distillation，必须硬排除。
4. **普通 KD 重合**：若 C0 不优于 KD0，跨图 support 没有独立价值，CASD 判负。
5. **UMTS 退化**：若 C0 不优于 UM0，support advantage 没有独立价值，不能把 multi-shot teacher-student 重写成新方法。
6. **pose 组织无价值**：若 correct 不优于 uniform/shuffled，不能保留 pose-specific claim。
7. **global write-back 复发**：不做 PBSR 式写回；student 继续输出 local descriptors。
8. **描述子维度问题**：首验保留 5376-D 是单变量纪律，不得冒充同维 global；768-D 版本另过门禁。

## 论文叙事草案

不再讲“在哪里注入 pose”，而讲：

> 遮挡使单图只能提供不完整身份 support，而训练集中的同 ID 多视图往往拥有互补可见部位。CASD 用训练期姿态组织 leave-one-view-out 的跨图解剖 support，并只把 support 相对当前单图真正新增的身份关系交给无姿态 student。姿态因此不是测试输入，也不是一个新的 part token，而是组织互补训练证据的 privileged variable。

GCN、CLIP 文本、matching、普通 pose KD 与 counterfactual inference 均不列为贡献。
