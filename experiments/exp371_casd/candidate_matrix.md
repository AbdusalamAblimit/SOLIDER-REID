# exp371：LGPA 改造候选矩阵

## 2026-07-13 外部查新与 Gate B/D 后最终裁决

1. **CASD 仍是唯一条件主线**，但问题新意从“首次用同 ID 多图补全单图”降为：相对 MVI²P/UMTS，严格隔离 anchor 之外真正新增、可归因且可迁移的 cross-view support gain。
2. Gate B 显示 `correct/shuffled/canonical` 只差 `0.03/0.10 mAP`，所以 `anatomical/precise-pose` 不是已成立资产；只有后续固定 feature 的 support-routing 门禁通过，才能保留 pose-organized claim。
3. train-only PCA-768 单 seed retention=`1.1158`，同维 learned packing provisional GO；它解决成本公平性，不是创新。
4. **AERC 已被 2025 NNCL 机制级覆盖，独立主创新 NO-GO**；不再作为 CASD 失败后的第二主线。
5. PELD/ACC/AEAD/ASMI 只作诊断或强对照。若 CASD Gate C 失败，正式停止 LGPA 自有化，不转 OT/MoE/slot/erasure coding 小变体。

## 评价标准

每个方案按六项判断：

1. 是否保留 LGPA 已验证的 local descriptor/extractor 资产；
2. 是否避开 PAFormer、ProFD、TSD、PGFL-KD 的直接覆盖；
3. 是否回应 `exp109` 的 single-image support incomplete；
4. 是否避开 `exp320/370` 的梯度冲突与 global write-back 负证据；
5. 是否可以先做低成本 kill-switch；
6. 是否能形成问题—机制—证据闭环。

## 候选排序

| 排名 | 方案 | 问题新意 | 机制新意 | 可验性 | 直接撞车风险 | 裁决 |
|---:|---|---:|---:|---:|---:|---|
| 1 | CASD：strict-LOO Cross-instance Support-Gain Distillation | 3/5 | 4/5 | 5/5 | 5/5 | **唯一条件主线；必须超过 MVI²P/UMTS 式控制** |
| 2 | IPER：Pose-Advantage Interventional Routing | 4/5 | 3/5 | 5/5 | 5/5 | 降为 CASD 因果门禁/辅助监督 |
| 3 | Continuous Anatomical Equivariance Field | 4/5 | 4/5 | 4/5 | 4/5 | 备查，不启动 |
| 4 | Pose-Privileged Packed Descriptor | 3/5 | 2/5 | 5/5 | 4/5 | 只作同维 oracle/后续封装 |
| 5 | Conflict-Gated Part Gradient | 3/5 | 3/5 | 3/5 | 4/5 | 否决；强系统 un-detach 已 `-6.4` |
| 6 | SoftMoE/OT/slot routing | 2～3/5 | 2～3/5 | 3/5 | 5/5 | 否决；模块拼装且 PBSR 已负 |
| 7 | pose-guided masking/transport | 3/5 | 2/5 | 3/5 | 5/5 | 否决；PTP/SPT/PGMAN/PCVT 覆盖或负证据 |
| 8 | learned matching/visibility/completion | 2/5 | 2/5 | 4/5 | 5/5 | 否决；matching 非创新且本地强负 |

## 主方案：CASD

### 问题重定义

训练集通过 P×K sampler 天然提供同一身份的多张图。遮挡使单图只含不完整 visible support；而普通 pose-part 方法仍然逐图监督和逐图编码，没有利用其他同 ID 图像中可见的互补解剖证据。

CASD 将 LGPA 降为 detached、训练期 pose-aware extractor，用它组织同 ID 的跨图 support：

```text
其他同 ID 图像的可见 part slots
        ↓ coverage-aware aggregate
leave-one-view-out identity anatomical support
        ↓ support-vs-self advantage distillation
当前单图 image-only student
```

### 为什么仍优先于 IPER

1. 它直接回应 `exp109` 的 single-image support incomplete；
2. 最近邻主要都在单图 pose supervision/teacher-student，跨实例 support 是实质训练对象变化；
3. 不依赖“correct pose 独占增量必须很大”，而利用同 ID 其他视图真正新增的可见证据；
4. IPER 的 correct/shuffled/uniform 干预可以自然成为 support-quality 控制；
5. 避开 2022 年同名 Pose-guided Counterfactual Inference 的 headline 冲突。

但 AAAI 2020 UMTS 已覆盖 multi-shot teacher → single-shot student 的大框架。因此 CASD 只有在“pose-organized part support + leave-one-view-out + support-vs-self advantage + 伪 pose 控制”四项同时成立时才保有新颖性。若只做完整 multi-shot feature KD，直接判为 UMTS 邻域，不启动。

## IPER 的新位置

IPER 不再作为方法主名。它只做两件事：

1. 在缓存 teacher features 上检查 correct pose 是否比伪 pose 组织出更纯、更完整的 support；
2. 可选地给 support part 权重加入 `correct - controls` 的可靠性权重。

若这一步无效，只能说明 pose 对 support 组织没有因果价值；CASD 的 pose-specific claim 判负。

## Packed descriptor 的位置

将 7 个 768-D 块压成总计 768-D 能解决描述子成本，但不能单独形成创新，也不能与 CASD 首验混成多变量。它先作为 frozen oracle；通过后再作为 final system 的独立封装实验。

## 最终裁决

下一步只推进 CASD 的三道廉价门：

1. correct pose 是否比 uniform/shuffled/wrong-person 构造出更好的跨图 support；
2. leave-one-view-out support 是否比 same-image teacher 提供更广泛的 identity-relation advantage；
3. 冻结 backbone 的 image-only student 是否能从 CASD 获得明确增益。

任何一道失败，停止 CASD；不转向 OT、MoE、slot 数量、温度或 loss 权重救场。
