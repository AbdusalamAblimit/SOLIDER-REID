# exp403 查新：从“route alive”转向“evidence owns the operator”

> 日期：2026-07-20
> 状态：`TARGETED LITERATURE/CODE AUDIT COMPLETE / CONDITIONAL DESIGN GO`
> 边界：本审计只授权设计与 CPU/static contract；不授权占用 GPU。

## 1. 被 exp402 精确暴露的问题

exp401 证明固定预算 route 在最终检索中不是 identity：`full-all-bypass=+0.1194214838 mAP point`。
exp402 又证明，route 的存在不能等价为 sample-specific rich evidence 的所有权：

- correct=`57.1230075595`；
- same-camera different-PID wrong-RGB=`57.1296975953`；
- zero=`57.1237039863`；
- generic expert mean=`56.9989891041`。

当前 `EvidenceBudgetRouter` 把 evidence 作为
`token_hidden + context_hidden + evidence_hidden` 的一个加性项，再交给 slot-specific static expert。
因此 static expert、token/context 与 evidence 可以互相补偿；即使 evidence 改变了 descriptor，也不要求
正确 evidence 比 wrong/zero 更有身份效用。

新问题对象不是“如何把 evidence loss 调强”，而是：

> 如何让 sample-specific evidence 对生产算子具有结构所有权，并让正确配对相对 matched wrong/NULL/
> generic 配对在完整执行效用上可辨识？

## 2. 公开近邻与代码级边界

### 2.1 Counterfactual Attention Learning（ICCV 2021）

- 论文：<https://doi.org/10.1109/ICCV48922.2021.00106>
- 官方代码：<https://github.com/raoyongming/CAL>
- 审计 commit：`0ba9d5084f2532eeb21c9ef051c23f8b339595ff`

`reid/modeling/baseline.py` 的 `BAP` 在训练时用 `uniform(0,2)` fake attention，评测时用全一
attention，随后优化真实与 counterfactual prediction 的差。它已经覆盖“反事实分支必须改变预测”这一
思想，所以 `counterfactual loss` 本身不能作为 exp403 的创新点。

它没有覆盖：同 camera 不同 PID 的 matched evidence donor、NULL identity、evidence 生成生产低秩算子，
以及 correct/wrong/generic/all-bypass 的冻结检索闭环。

### 2.2 AIM / Good Is Bad（CVPR 2023）

- 论文：<https://doi.org/10.1109/CVPR52729.2023.00148>
- 官方代码：<https://github.com/BoomShakaY/AIM-CCReID>
- 审计 commit：`c3bda2b54a3c5d81eb65ea838ae3502aecd61b67`

`train.py` 使用两个 backbone，第二路 clothing feature detach 后经 `fuse` 得到 counterfactual feature，
并对 `outputs-outputs3` 增加身份分类监督。它覆盖“用双分支/分类差消除混杂因素”，但目标是 clothing
debiasing；条件信息不生成主干生产算子，也没有 matched evidence ownership 检验。

### 2.3 UCT（TOMM 2024）

- 论文：<https://doi.org/10.1145/3674737>
- 代码页：<https://github.com/NJUPT-MCC/UCT>

论文通过 feature-adaptive 权重聚合跨模态 class prototypes，近似 backdoor adjustment。它说明
“feature-conditioned prototype adjustment”已有明确先例。当前公开仓库 README 仍主要提供测试路径，
没有可复核的完整训练实现。该工作不建立 RGB evidence→低秩生产算子，也不做遮挡 ReID 的 wrong-RGB/
NULL 检索排序。

### 2.4 其它必须承认的撞车面

- *Pose-guided counterfactual inference for occluded person re-identification*（IVC 2022，
  <https://doi.org/10.1016/j.imavis.2022.104587>）已经占用“pose-guided counterfactual inference”题名，
  因此 exp403 不以“pose counterfactual”作为 headline。
- *Counterfactual Intervention Feature Transfer for Visible-Infrared Person Re-identification*（ECCV 2022，
  <https://doi.org/10.1007/978-3-031-19809-0_22>）和 2026 的
  *Counterfactual-Guided Implicit Correspondence Prompting*（
  <https://doi.org/10.1109/JAS.2025.125432>）说明跨模态 ReID 已广泛使用 counterfactual 叙事。
- SGFNet（TIFS 2025，<https://doi.org/10.1109/TIFS.2025.3608672>）已经使用 sample-specific text、
  双语义空间注意和 semantic focal triplet；sample-specific CLIP/文本本身不能声称新。
- Dynamic Filter Networks、CondConv、Dynamic Convolution、HyperNetworks 和 LoRA 已分别覆盖输入条件
  权重、专家混合、动态卷积、权重生成与低秩参数化。exp403 不能把“动态/低秩/超网络”任一原子写成贡献。
- Instruct-ReID 公开仓库中的 legacy `DualCausalityLoss` 对 `f/f+/f-` 的正负距离施加排序，但它不是
  matched external evidence 的生产执行路径，也不提供 NULL exact identity。

## 3. 查新后的可争空间

本轮未找到以下完整同构链：

```text
RGB-only student evidence
-> shared zero-bias low-rank operator coefficients
-> local feature/evidence compatibility controls execution
-> matched correct/wrong-RGB/NULL/generic complete executions
-> counterfactual branches are stop-gradient references
-> frozen final retrieval verifies semantic ownership and route mediation
```

这不是“每个原子从未出现”的 novelty 声明。可争空间在三者同时成立：

1. **问题层面**：把 route alive 与 evidence ownership 分离；
2. **机制层面**：evidence 不再是 static expert 前的 additive bias，而是拥有共享低秩算子的逐样本系数；
3. **证据层面**：训练期 matched counterfactual utility ordering 与测试期冻结检索反事实使用同一组控制。

当前创新门槛判定为 `3/3 satisfied for design exploration`，但公开查新只能支持“未发现同构”，不能支持
绝对首创。novelty 风险评为 `6/10`：最大的风险是被归纳为 dynamic adapter + CAL loss；必须靠
NULL identity、stop-gradient comparator、防破坏约束和完整检索控制把差异做实。

## 4. 设计裁决

条件 GO 的对象命名为 **ELO-CUR**：

- `ELO`：Evidence-owned Low-rank Operator；
- `CUR`：Counterfactual Utility Ranking。

它不是 exp402 的旧臂调参：删除 slot-specific static experts，evidence 从加性 hidden 改为低秩算子系数，
并新增 matched complete-execution ordering。rho、rank、batch、seed、epoch 和 teacher 资产保持冻结。

当前仅授权 exp403 design/protocol/CPU contract。只有 contract 能同时证明 shared operator、NULL identity、
matched donor、反事实分支无梯度、正确分支梯度覆盖和负例必失败，才进入真实 batch64 CUDA preflight。
