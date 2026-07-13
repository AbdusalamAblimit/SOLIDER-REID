# exp371 CASD：文献与代码查新

## 查新结论

PAFormer 已经直接覆盖“pose heatmap 监督 learnable pose tokens，推理不再调用 pose estimator”。因此 LGPA 不能靠换掉 CLIP query、learnable part token、teacher forcing 或 pose-free inference 重新获得新颖性。

在本轮检索的 ReID 与相邻机制中，尚未发现以下完整组合的直接先例：

> 用 detached pose-aware extractor 在训练期按同一身份的互补可见部位组织跨图 anatomical support；对每个 anchor 严格 leave-one-view-out；再让单图 image-only student 学习这个跨实例、coverage-aware support 的身份关系，而不是模仿当前图自己的 pose map。

因此 CASD 若成立，可争取的创新点是 **pose-organized leave-one-view-out anatomical support advantage**。蒸馏、pose-free、multi-shot teacher、part token、cross-attention、matching 和 CLIP 均不能单独列为贡献。correct/canonical/shuffled/uniform 干预保留为验证 support 是否真的由 pose 组织得更好的因果门禁，不再作为方法 headline。

## ReID 最近邻

| 工作 | 年份/来源 | 已覆盖内容 | 对 exp371 的边界 |
|---|---|---|---|
| [Diverse Part Discovery / PAT](https://doi.org/10.1109/CVPR46437.2021.00292) | CVPR 2021 | part-aware transformer、part token 与 patch/CLS 交互 | part token、自注意力交互不是创新 |
| [PGFL-KD](https://arxiv.org/abs/2108.00139) | ACM MM 2021 | 姿态分支向主 ReID 分支蒸馏，测试去 pose | “训练 pose、测试无 pose”和普通 pose KD 已被覆盖 |
| [PFD](https://doi.org/10.1609/aaai.v36i3.20155) | AAAI 2022 | pose-guided feature disentangling、part comparison | pose-conditioned part decomposition 不是创新 |
| [Pose-guided Counterfactual Inference](https://doi.org/10.1016/j.imavis.2022.104587) | IMAVIS 2022 | 使用 pose 做 counterfactual inference | 反事实姿态只能作为 support 控制，不作为 headline |
| [SAP](https://doi.org/10.1609/aaai.v37i1.25180) | AAAI 2023 | noisy semantic teacher、attention partition、partial-consistency distillation | 语义 teacher 约束 part attention 再蒸馏不是创新 |
| [BPBreID](https://doi.org/10.1109/WACV56688.2023.00166) | WACV 2023 | body-part maps、局部表征、GiLt/可见局部比较 | part map、part loss、visibility matching 不是创新 |
| [TSD](https://arxiv.org/abs/2312.09797) | ICASSP 2024 | parsing-aware teacher decoder 到 pose-free student decoder | teacher-to-pose-free part decoder 已被直接覆盖 |
| [SPT](https://doi.org/10.1609/aaai.v38i5.28312) | AAAI 2024 | saliency-guided occluder patch transfer、双路径训练 | pose/saliency 驱动遮挡干预不再作为候选 |
| [KPR](https://arxiv.org/abs/2407.18112) | ECCV 2024 | 正/负 keypoint prompts、global/part descriptor、visibility-aware comparison | target/distractor prompt、局部描述子和 matching 都不是创新 |
| [PAFormer](https://arxiv.org/abs/2408.05918) | 2024 | heatmap supervision、learnable pose tokens、visibility predictor、test-time 无 pose locator | 与原 LGPA 改造高度重合；是最直接风险 |
| [ProFD](https://arxiv.org/abs/2409.20081) | ACM MM 2024 | part-specific CLIP prompts、dense spatial alignment、hybrid decoder、prototype memory | text part proxy、双向 attention、普通 memory 与 self-distillation 不是创新 |
| [DROP](https://arxiv.org/abs/2401.18032) | CVPR 2024 | backbone 多层特征拆分 ReID/parsing 分支，parsing 引导 pooling | “结构监督与 ReID feature 解耦”不能单独主张 |
| [PASS](https://arxiv.org/abs/2203.03931) | ECCV 2022 | part-aware SSL pretraining，local crops 与 part tokens 对齐 | 普通 part-aware contrastive pretraining 不是创新 |
| [Pose Transfer](https://doi.org/10.1109/CVPR.2018.00431) | CVPR 2018 | pose-transfer/canonical-view generation | pose-defined pseudo-view/feature transport 不是新问题 |
| [UMTS](https://arxiv.org/abs/2001.05197) | AAAI 2020 | multi-shot teacher 输入同 ID 的 K 张图，student 输入其中一张；多阶段 uncertainty-aware feature KD | **最接近 CASD 的问题先例**；“多图完整 teacher 教单图 student”绝不能作为新意 |
| [FRT](https://doi.org/10.1109/TIP.2022.3186759) | TIP 2022 | 测试时用 gallery kNN 集合恢复 query complete feature | “用其他图恢复完整特征”不是新问题；CASD 必须坚持 train-only、leave-one-view-out、无 gallery support 推理 |
| [π-VL](https://arxiv.org/abs/2308.02738) | 2023/2025 revision | parsing-guided identity-specific part prompts、层次局部视觉语言监督 | part prompt/local V-L alignment 进一步不能主张 |
| [OGFR](https://arxiv.org/abs/2507.08520) | 2025 | occlusion embedding、低质量 patch 移除/替换、holistic teacher→occluded student | 普通遮挡 teacher purification 与 patch completion 不新 |
| [DPM++](https://arxiv.org/abs/2605.06637) | 2026 | input-adaptive masked metric、CLIP semantic prior、patch transfer | masked metric、CLIP prior 与 patch transfer 不进入主线 |

## 相邻领域边界

| 工作 | 已覆盖机制 | 对 exp371 的约束 |
|---|---|---|
| [PDiscoNet](https://doi.org/10.1109/ICCV51070.2023.00179) | concentration、presence、orthogonality、equivariance 的语义一致 part discovery | 普通 part equivariance/discovery 不能作为主贡献 |
| [Invariant Slot Attention](https://arxiv.org/abs/2302.04973) | slot-centric reference frame 与等变性 | “等变 slots”本身不新 |
| [Soft MoE](https://arxiv.org/abs/2308.00951) | 可微 dispatch/combine 的 token experts | part experts/MoE 只能是实现，不是贡献 |
| [MESH](https://arxiv.org/abs/2301.13197) | Slot Attention 与 optimal transport 的联系 | Sinkhorn/OT routing 不能单独主张新颖性 |
| [PKDOT](https://arxiv.org/abs/2401.15489) | privileged multimodal optimal-transport distillation | privileged information + OT/KD 已有先例 |
| [Residual KD](https://arxiv.org/abs/2002.09168) | residual/assistant knowledge distillation | “Residual KD”名称与普通 residual distillation 不可作为贡献 |
| [Expert-exclusive Knowledge](https://arxiv.org/abs/2112.02747) | 抽取专家相对普通模型的独占差异再迁移 | exclusive difference 本身已有概念先例 |

## CASD 的最高风险：UMTS

UMTS 已在 2020 年明确提出：同 ID 多张图覆盖不同 viewpoint/pose/occlusion，multi-shot teacher 学 comprehensive feature，single-image student 在推理时单独使用。其 teacher 将 K 张 RGB 图沿通道拼接，student 输入是这 K 张中的一张，并在多阶段做 uncertainty-aware feature distillation。

因此 CASD 不能声称：

- 首次利用同 ID 多图补充单图；
- 首次 multi-shot teacher → single-shot student；
- 首次用 uncertainty/quality 控制多图蒸馏。

CASD 的有条件差异必须同时包括：

1. teacher 不是整图拼接网络，而是 LGPA 提取的 part-wise anatomical evidence；
2. support 按 pose visibility 逐部位从其他同 ID 图像选择互补证据；
3. anchor 当前图硬排除，UMTS 中 student 图属于 teacher K-shot 输入，而 CASD 严格 leave-one-view-out；
4. 不蒸馏完整 multi-shot feature，只蒸馏 support 相对 same-image teacher 真正改善的 identity relation/margin；
5. correct pose support 必须优于 uniform/shuffled/wrong-person，证明不是普通 multi-shot KD。

若实现缺少任意一项，CASD 会退化成 UMTS 的 part-wise 变体，不足以作为论文主创新。

## 被排除的改造方向

以下方向不再进入主线：

1. CLIP query 换 learnable query；
2. 普通 pose teacher → pose-free student；
3. part token / slot / cross-attention / write-back 小变体；
4. pose mask、patch transfer 或遮挡 counterfactual augmentation；
5. part equivariance、orthogonality、OT 或 MoE 的模块拼装；
6. learned matching、visibility scorer、MaxSim training；
7. GCN 或文本语义包装。

## 最重要的精确撞车

[Pose-guided counterfactual inference for occluded person re-identification](https://doi.org/10.1016/j.imavis.2022.104587) 已在 2022 年发表，而且题名与“pose counterfactual routing/inference”直接冲突。

所以：

- 反事实姿态干预只能作为 CASD 的 support-quality 门禁或辅助 loss；
- 不将 IPER、Pose Counterfactual Routing 或 Counterfactual Pose Inference 作为论文题名/核心首创 claim；
- 若需要 treatment-effect 监督，必须明确它服务于跨图 support 构造，而不是宣称首次提出 pose counterfactual inference。

## CASD 必须守住的差异

CASD 只有同时满足下列条件，查新才有效：

1. support 必须来自同一身份的其他图像，而不是当前图自己的 teacher feature；
2. 每个 part 只由其他视图中可见且可靠的同部位证据组成；
3. 对 anchor 严格 leave-one-view-out，禁止 current-view leakage/trivial copy；
4. student 只看单图 RGB，不读取 pose，也不能在测试时访问同 ID support；
5. student 学的是跨图 support 相对 same-image teacher 的正向 identity-relation advantage，而不是完整 multi-shot feature MSE；
6. correct pose support 必须显著优于 uniform/shuffled/wrong-person support；
7. 必须有 same-image teacher KD 对照，证明收益来自跨实例 support，而非普通 pose KD。

## 可主张与不可主张

若实验成立，可以主张：

- 用 pose 在训练期组织跨图、互补可见且 leave-one-view-out 的 identity anatomical support；
- 通过 leave-one-view-out 和 coverage-aware aggregation 把 support 从当前图复制中隔离；
- 只迁移 support 相对 same-image teacher 的正向 identity-relation advantage；
- 用伪 pose support 与普通同图 KD 对照证明有效变量确实是跨图结构 support。

无论结果如何，都不能主张：

- 首次提出 pose token、part query、pose teacher-student 或 pose-free ReID；
- 首次提出 counterfactual learning、residual distillation、part equivariance 或 OT；
- CLIP 文本语义已经被证明有效；
- 现阶段已经证明跨 ResNet、ViT、Swin 通用。
