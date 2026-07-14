# exp374 文献与新颖性边界审查

## 裁决

- exp374 Gate A 的**文献边界**：`PASS`，但这不代表设计、实现或执行已 PASS，也不是
  论文创新；
- 原 PSG 作为主创新：`FAIL`；
- 原 UBCFT 公式作为主创新：`FAIL`；
- 当前 reliability-bounded dense pose flow：`FAIL / NO-GO`；只有一个尚未完成数学
  定义的不确定度约束联合问题对象可继续设计，不允许训练。

## 直接先例

### PSG 与中层调制

- [Pose Guided Gated Fusion, WACV 2020](https://openaccess.thecvf.com/content_WACV_2020/html/Bhuiyan_Pose_Guided_Gated_Fusion_for_Person_Re-Identification_WACV_2020_paper.html)：
  pose+appearance 生成与中层 feature 同空间同通道的 gate，逐元素调制后继续传播；
- [SFT, CVPR 2018](https://arxiv.org/abs/1804.02815)：空间条件仿射；
- [FiLM, AAAI 2018](https://arxiv.org/abs/1709.07871)：条件 scale/shift，包含
  `gamma=1+delta_gamma` 和多 block 调制。

因此 PSG 只能作为轻量、pose-only、zero-init 的 Swin 条件调制实例。

### 图传播与遮挡补偿

- [HOReID, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_High-Order_Information_Matters_Learning_Relation_and_Topology_for_Occluded_Person_CVPR_2020_paper.html)：
  以 pose keypoint local feature 为图节点，通过 adaptive directed GCN 做动态邻接和
  message passing；
- [Pose Matters / PGGANet](https://arxiv.org/abs/2111.14411)：多层 pose mask、
  spatial/channel attention 与 learnable-adjacency graph attention；
- [RTGAT, TIP 2023](https://doi.org/10.1109/TIP.2023.3247159)：visibility-guided
  graph attention，并从同 ID holistic image 向 occluded image 传播 missing semantics。
- [Feature Recovery Transformer, TIP 2022](https://doi.org/10.1109/TIP.2022.3186759)：
  pose part、visibility-conditioned directed graph 与遮挡 query feature recovery；
- [PIRT, ACM MM 2021](https://doi.org/10.1145/3474085.3475283)：pose 支撑的
  visible/invisible part 信息混合与 confidence filtering；
- [HUPOR, ECCV 2022](https://doi.org/10.1007/978-3-031-20065-6_29)：从 visible
  joint 的 feature-level cue 推断 occluded joint。
- [RFC / Feature Completion for Occluded Person Re-Identification, TPAMI 2021](https://doi.org/10.1109/TPAMI.2021.3079910)：
  在单图 backbone 内利用 non-occluded region 的 long-range spatial context 预测/
  恢复 occluded region feature，并可插入不同 backbone 深度；
- [Neighbourhood-guided Feature Reconstruction, IJCAI 2021](https://arxiv.org/abs/2105.07345)：
  从 gallery neighbours 的 non-occluded counterpart 重建目标 occluded part，并用
  outlier-removable GNN 给 source confidence；
- [FCFormer, TMM 2024](https://doi.org/10.1109/TMM.2024.3379908)：利用邻域信息与
  Transformer decoder 恢复遮挡区域 feature。

这些工作分别并在组合上覆盖 source reliability、visible-to-occluded recovery、
pose topology 和 confidence-conditioned mixing；不是说其中任一篇已经逐式覆盖
未来完整公式。因此不能声称“首次姿态图传播”“首次非对角 pose routing”“首次
visibility-directed flow”“首次单图 backbone 内 feature completion”或“首次用可见
证据补偿遮挡区域”。

### pose-aware OT

- [Posture-Aware Robust Person Re-Identification via Optimal Transport Calibration,
  TIFS 2025](https://doi.org/10.1109/TIFS.2025.3622067)：摘要已明确使用 posture-aware
  MoE 与测试时 gallery/query 分布 OT calibration。

它与 backbone spatial-token flow 不同构，但已封住“首次 pose-aware ReID + OT”。
若候选只用 Sinkhorn 把网络输出矩阵归一化，而没有显式 transport cost、marginal 和
OT objective，方法名中不得使用 Optimal Transport。

此外：

- [UNITE, CVPR 2021](https://doi.org/10.1109/CVPR46437.2021.01478) 已在
  keypoint-map 条件输入下学习位置质量，以 entropic unbalanced OT 搬运 exemplar
  feature，并把缺失身体部位视为不应强制匹配的 outlier；
- [Self-Optimal-Transport Feature Transform](https://arxiv.org/abs/2204.03065) 已在
  person ReID 中使用对称双随机 OT plan 做 feature transform；
- [Sinkformers, ICML 2022](https://proceedings.mlr.press/v151/sander22a.html) 已覆盖
  Sinkhorn 双随机 token attention。

因此 UOT、learned mass、缺失部位、ReID feature transform、双随机或“守恒”都不能
单独承担 claim。

### pose-conditioned token attention

- [PeVL, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_PeVL_Pose-Enhanced_Vision-Language_Model_for_Fine-Grained_Human_Action_Recognition_CVPR_2024_paper.pdf)；
- [PAAB/PAAT](https://arxiv.org/abs/2306.09331)；
- [MUVA](https://arxiv.org/abs/2603.14012)。

这些工作分别覆盖 pose/body mask 修改 visual/ViT/CLIP attention、写入 QK logits、
残差更新和逐层动态 body mask 注入。不能仅凭“pose 修改 attention 权重”“多层注入”
或“Transformer 内部”建立新颖性。

### 端到端 pose + ReID

- [PABR](https://arxiv.org/abs/1804.07094)：pose 子网络由 ReID triplet loss 微调；
- [Visual Person Understanding](https://arxiv.org/abs/1906.03019)：共享 backbone 的
  pose/ReID 多任务训练与 pose semantic forgetting；
- [Pose Auxiliary VI-ReID](https://arxiv.org/abs/2201.03859)：pose heatmap loss、ID、
  triplet/KD 共同约束 pose branch，并把 pose mask 回注 ReID feature。

因此端到端、冻结后解冻、pose loss 保语义均不能单独承担贡献。

## 新近 pose-patch / restoration 边界

[Texture-Aware Transformer with Pose-Patch Mapping for Occluded Person
Re-Identification, Pattern Recognition 2026](https://doi.org/10.1016/j.patcog.2025.112341)
的本地既有全文摘录已经重新核对。TTPM 不是 source-to-sink OT：

- Multi-patch Feature Encoder 先编码 intra/inter-patch feature；
- Pose-Patch Mapping 以 Mahalanobis distance 与 cosine similarity 构造 pose-patch
  相似关系，为每个 landmark 选择最相似 patch，并用 pose confidence threshold
  过滤低置信 landmark；
- Texture-Aware Decoder 再以 mapped pose/patch relation 约束 cross-attention，强化
  target texture，另用 pose loss 拉开 human 与 non-human mapped feature。

因此 TTPM 不同构于显式 source depletion + sink receipt，但已经强力封住“首次
pose-patch mapping”“用高置信姿态筛选有效 patch”“pose-guided texture restoration”
等表述，必须作为直接对照而非未读黑箱。

仍未排除的是 [Pose-Guided Feature Restoration Transformer for Occluded Person
Re-Identification, LNCS 2026](https://doi.org/10.1007/978-981-95-7251-9_30)。当前只能
核对题名、作者与出版元数据，无法核对方法正文；在全文排除前，它与 RFC/TTPM 一起
阻止新 flow 机制训练。

## 当前只允许继续定义的问题对象

暂时未找到单篇完全同构先例的是下列完整联合对象，但“组合没有逐式同构”不足以自动
形成创新，当前仍判 FAIL：

> 校准 pose posterior 定义不确定度约束可行域；ReID utility 与姿态不确定度分别决定
> source/demand 边际；只允许骨架支持上的 directed flux；算子显式执行 source
> depletion 与 sink receipt，并验证 balance、off-diagonal flux 和姿态对应因果差异。

“分布鲁棒”只有在明确给出 ambiguity set、inner worst-case 和 robust objective 时才
能使用；当前没有这些定义，只能称不确定度约束。若最终只是 confidence-weighted
Sinkhorn layer，审稿人很容易把它解释为 HOReID/FRT/RFC + UNITE/Sinkformers 的
直接拼接。安全术语是
`token-sum-preserving update` 或 `zero-sum token flow`，不是 feature mass
conservation。若 value projection 任意改变通道，则“只移动现有证据”也不能成立。

## Gate A 与新颖性的关系

Gate A 不证明上述问题对象新，也不证明其有效。它只回答：旧 PSG checkpoint 中是否存在
足够强的 image-pose correspondence fuel。若 Gate A FAIL，继续把 PSG 复杂化没有
证据基础；若 GO，只允许进入 clean paired Gate B 设计和联合数学对象的下一轮红队，
不能直接实现 transport。
