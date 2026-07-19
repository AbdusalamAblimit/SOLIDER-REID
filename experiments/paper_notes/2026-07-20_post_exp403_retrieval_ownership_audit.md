# exp403之后：final-retrieval ownership查新记录

> 日期：2026-07-20
> 状态：`AUDIT ACTIVE / NO CANDIDATE CLEARS INNOVATION GATE / GPU NO-START`
> 目的：exp403已证明“可执行的evidence-conditioned operator”仍可在最终检索中被绕过。下一对象必须直接
> 约束最终身份排序，且不能退化为ELO-CUR换名、普通ranking loss、额外stage或retrieval-side小技巧。

## 1. 当前硬问题

exp403七臂的descriptor干预全部active，但R1/R5/R10逐项完全相同，mAP总跨度不足`0.0018 point`。
因此下一机制不能再把“descriptor变了”“operator有梯度”或训练期proxy margin当作ownership证据。最低要求是：

1. correct、matched wrong-RGB、generic、NULL和all-bypass都在设计中保留；
2. ownership目标直接作用于最终归一化检索对象或其不可绕过的组成部分；
3. shared visual trunk不能通过同步移动correct与reference来满足代理目标；
4. 不通过主动破坏wrong control制造优势；wrong evidence必须有独立、可解释的正目标；
5. 机制至少满足问题/机制/证据创新门槛中的两项，再建立新实验编号。

## 2. 近邻代码审计一：SPT（AAAI 2024）

- 论文：*Occluded Person Re-identification via Saliency-Guided Patch Transfer*
- 论文页：<https://doi.org/10.1609/aaai.v38i5.28312>
- 官方代码：<https://github.com/stone96123/SPT>
- 审计commit：`ef1e71a99bc658790d5dbbc9ab133588e849e814`

代码事实：

- `TransReID_Mask.forward_features`从第1/3/9/final transformer block拼接特征，以detach后的`mixfc`预测
  patch saliency mask；
- 第一阶段用原mask、mask与反mask三次forward，训练ID/Triplet、mask usage和两路分类正交；
- 第二阶段把二值saliency mask用于batch内patch transfer，再以普通ReID目标训练；
- 正式流程明确是先训练SPS、再训练ReID的两阶段方案。

裁决：SPT解决“哪些patch可迁移以模拟遮挡”，并不要求外部sample evidence拥有最终检索对象。它没有
matched wrong/generic/NULL/all-bypass完整执行，也没有防shared trunk绕过的所有权合同。把SPT移植为额外
augmentation或stage既不回应exp403，也违反当前“不增加stage救旧路线”的边界，故不作为下一主机制。

## 3. 近邻代码审计二：ProFD（ACM MM 2024）

- 论文/代码：*Prompt-guided Feature Disentangling for Occluded Person Re-Identification*
- 官方代码：<https://github.com/Cuixxx/ProFD>
- 审计commit：`14e47d3b04f541d2a614482848bba2071bc90cda`

代码事实：

- 训练依赖PifPaf与Mask-RCNN生成的人体解析mask；
- `PartFeatureDecoder`用prompt/part token和visual memory做双向cross-attention，生成显式part embedding；
- global、foreground、parts与concatenated parts分别接ID/metric目标，并加入pixel parsing、visibility、
  prototype memory与dissimilar loss；
- 测试通过query/gallery双方的part visibility组合pairwise distance，而非单一固定global descriptor。

裁决：ProFD证明“显式part slot + visibility-aware pairwise metric”已有完整强先例。它也提示一个真实结构方向：
让语义槽本身进入检索距离，比在global descriptor内部做小残差更难被绕过。但直接复刻会落入已有part-based
ReID与retrieval-side visibility融合，并需要当前冻结边界外的解析资产；不足以作为新贡献。

## 4. 当前筛选结论

本轮新增排除两类看似相关但不合格的方向：

1. `saliency mask + patch transfer + extra stage`：问题对象仍是增强，不是ownership；
2. `part decoder + visibility-weighted pairwise distance`：已有强先例，且回到被长期探索的retrieval-side路线。

目前没有候选通过创新门槛，故不建立exp404、不写formal config、不运行CPU/CUDA/GPU。下一轮查新收紧到：

- 对wrong evidence存在独立正目标、而非单纯被推远的counterfactual transport/equivariance；
- ownership loss只允许更新evidence-dependent终端子空间，shared identity trunk对该loss严格stop-gradient；
- 训练与最终标准欧氏检索使用同一个归一化对象，不引入pair-specific test-time scorer。

这些只是检索条件，不是已授权机制。必须继续查阅相邻领域的conditional metric、equivariant intervention与
identifiable bottleneck公开实现，确认不是普通conditional embedding/ranking loss后，才能决定是否形成新编号。
