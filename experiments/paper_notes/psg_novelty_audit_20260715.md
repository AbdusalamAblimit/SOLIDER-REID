# PSG 创新性审计（2026-07-15）

## 结论先行

以当前代码中的核心算子

\[
g=E(H),\qquad Y=X\odot(1+g)
\]

为审计对象，结论是：

1. **PSG 有稳定的实证价值，但不能再安全地作为方法级主创新。**
2. 最危险的直接先例不是宽泛的 pose-aware Transformer，而是 Bhuiyan 等人在
   WACV 2020 提出的 **Pose Guided Gated Fusion for Person Re-identification**。
   该工作已经在 ReID 中使用 pose maps 生成逐位置、逐通道 gate，在 backbone
   中层与 appearance feature 做 Hadamard product，并让调制后的特征继续向后传播。
3. 从通用条件调制看，PSG 还是 SFT/FiLM 类空间条件仿射调制的严格特例：

   \[
   \operatorname{SFT}(X\mid\gamma,\beta)=\gamma\odot X+\beta,
   \quad \gamma=1+E(H),\quad\beta=0.
   \]

4. `pose-only`、`1+g`、最后一层零初始化、换成 Swin、在多个 block/stage
   重复注入，都是有用的实现差异；但单独或组合起来仍不足以把现有算子变成新的
   方法族。
5. **诚实定位**应改为：PSG 是一个轻量、有效、适合 Swin-ReID 的
   `gamma-only / zero-initialized spatial conditional modulation` 实例。它可以保留为
   支撑组件、强基线或系统设计，不宜继续写成“首次在 backbone 表征形成阶段注入
   pose”的主贡献。

该结论只裁决“当前 PSG 本体是否新”，不否定 PSG 的涨点，也不否定以后围绕新的
问题对象或新的非对角算子继续改造。

## 一、审计对象：真实代码而不是论文概念稿

### 1.1 当前 PSG

代码：

- `model/modules/pose_spatial_gate.py`
- `model/pose_backbone_model.py`

真实实现为：

1. 输入是 17 通道 ViTPose heatmap；
2. heatmap resize 到当前 Swin feature map 大小；
3. `1x1 Conv -> ReLU -> 1x1 Conv` 生成与 feature 相同空间和通道维度的 gate；
4. 在每个启用 stage 的每个 Swin block 后执行
   `X <- X * (1 + gate)`；
5. 输出卷积零初始化，因此训练开始时严格为 identity；
6. 每个 block 使用独立 PSG encoder，启用哪些 stage 由
   `POSE_PSG_STAGES` 决定。

这不是 self-attention：它不修改 `Q/K/V` 或 attention logits，而是 block 输出后的
逐位置逐通道 feature recalibration。论文若把它称为“pose attention”，会造成算子
定义与术语不一致。

### 1.2 一个与创新无关但必须修正的语义风险

`pose_spatial_gate.py` 当前对 heatmap 再执行一次 `sigmoid`。同一模型文件另一条
pose-prompt 路径已明确注明 ViTPose heatmap 本身处于 `[0,1]`，因此这里会把数值
压到约 `[0.5,0.731]`。

这不等于 PSG 无效，但会带来两个解释边界：

- 低响应不再接近 0，不能直接把 gate 解释成严格的 keypoint confidence weighting；
- 训练完成后把 `H=0` 输入 PSG 并不等于 `no-pose`，因为 `sigmoid(0)=0.5`，此时
  encoder 通常仍产生非零输出。真正的 no-pose control 必须旁路 PSG。

零初始化只保证**训练开始时**为 identity，不能写成“任何无姿态输入都自动退化为
identity”。

## 二、决定性直接先例：WACV 2020 Pose Guided Gated Fusion

Bhuiyan et al., *Pose Guided Gated Fusion for Person Re-identification*, WACV 2020：

- [CVF 原文](https://openaccess.thecvf.com/content_WACV_2020/html/Bhuiyan_Pose_Guided_Gated_Fusion_for_Person_Re-Identification_WACV_2020_paper.html)
- [DOI](https://doi.org/10.1109/WACV45572.2020.9093370)

其方法可概括为：

\[
A^l=A^l(I),\qquad P_{S,L}=P(I),
\]

\[
G=D(E(I)),\qquad E(I)=\{A^l,P_{S,L}\},
\]

\[
f_g^l=A^l\odot g_l,qquad
\tilde f_g^l=\frac{f_g^l}{\lVert f_g^l\rVert_2}.
\]

原文明确说明：

- pose 输入是 OpenPose 的 confidence maps 与 part affinity fields；
- pose 与 backbone 中层 appearance feature 在 gate network 中拼接；
- gate network 使用 `3x3 Conv + LeakyReLU + 1x1 Conv`；
- 输出与 appearance feature 具有相同的空间尺寸和通道数；
- gate 在“pixel-wise level among all channels”缩放 appearance feature；
- Hadamard product 后的结果继续送入 backbone 后续层；
- 实验比较 C2/C3/C4/C5 和 late fusion，并认为 C3/C4 中层更合适；
- 同时在多个层做 fusion 曾导致不稳定；
- 作者还将其接到 Trinet、PCB、BOT，并主张可适配不同 feature extractor 和 loss。

### 2.1 与 PSG 的逐项比较

| 维度 | PSG | WACV 2020 Gated Fusion | 创新性后果 |
|---|---|---|---|
| 任务 | Person ReID / occluded ReID | Person ReID | 同领域直接先例 |
| 条件 | 17 个 ViTPose heatmap | 50 个 OpenPose confidence/affinity maps | pose 表示不同，不改变核心算子 |
| gate 输入 | 仅 pose：`E(H)` | pose + appearance：`D([H,X])` | PSG 是更受限的 pose-only 形式 |
| gate 维度 | 空间位置 x feature 通道 | 空间位置 x feature 通道 | 核心形态重合 |
| 注入位置 | Swin block 后 / 可多 stage | ResNet 中层 C2-C5 | backbone 与离散位置不同 |
| 调制 | `X*(1+g)` | `X*g` 后 L2 norm | 残差化和归一化不同，但仍是同类乘性调制 |
| 后续传播 | 是 | 是 | “前移到 feature extraction”已被覆盖 |
| gate 网络 | 两层 `1x1` MLP-conv | `3x3 + 1x1` conv | PSG 更轻，不是更一般的新函数族 |
| 初始化 | 最后一层零初始化 | Gaussian 初始化 | PSG 更安全的 fine-tuning 设计，但不是新问题/新算子 |

WACV 2020 的 gate 允许依赖 `[X,H]`。从函数容量看，它可以通过忽略 `X` 而退化到
pose-only gate；再令其输出 `1+E(H)`，即可覆盖 PSG 的乘法形式。因此 PSG 更像
该机制的**轻量、pose-only、identity-initialized Transformer 适配**，而不是一个与其
平行的新方法。

尤其需要停止以下旧叙事：

> “已有工作只在特征形成后使用 pose；PSG 首次把 pose 前移到 backbone 表征形成阶段。”

WACV 2020 已经把 pose-guided gate 放在 backbone 中层，并明确强调 contextual
information 向后传播。继续使用上述问题句会被直接反例否定。

## 三、通用函数类先例：SFT / FiLM / zero-initialized conditioning

### 3.1 SFT：空间条件仿射的直接上位类

Wang et al., *Recovering Realistic Texture in Image Super-resolution by Deep Spatial
Feature Transform*, CVPR 2018：

- [CVF 原文](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Recovering_Realistic_Texture_CVPR_2018_paper.html)
- [arXiv](https://arxiv.org/abs/1804.02815)

SFT 从空间概率图生成与 feature map 对齐的 `gamma,beta`：

\[
(\gamma,\beta)=M(\Psi),\qquad
\operatorname{SFT}(F\mid\gamma,\beta)=\gamma\odot F+\beta.
\]

其空间维度和通道维度均与 feature 对齐，并可在网络多个中间层重复使用条件网络。
令 `Psi=H, beta=0, gamma=1+E(H)` 就严格得到 PSG。因此 PSG 不是新的条件调制
函数类。

### 3.2 FiLM：`1+delta gamma` 和多 residual block 也已有先例

Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018：

- [arXiv](https://arxiv.org/abs/1709.07871)

FiLM 的基本形式为 `gamma(z)F+beta(z)`；补充材料还明确写出实际实现预测
`delta gamma`，并令

\[
\gamma=1+\Delta\gamma,
\]

同时为多个 residual block 预测调制参数。因此“使用 `1+gate` 保护原特征”和“在
多个 block 重复条件调制”都不能单独承担新颖性。FiLM 本身不保留空间位置，但
SFT 已补齐空间条件这一维。

### 3.3 零初始化是可靠实现，不是 PSG 独占机制

- Zhang et al., *Adding Conditional Control to Text-to-Image Diffusion Models*,
  ICCV 2023 / [arXiv](https://arxiv.org/abs/2302.05543)，使用 zero convolution
  使新增条件分支在训练开始时输出 0；
- Peebles and Xie, *Scalable Diffusion Models with Transformers*, ICCV 2023 /
  [arXiv](https://arxiv.org/abs/2212.09748)，用 adaLN-Zero 将 Transformer block
  初始化为 identity。

这些工作不属于 ReID，但足以说明“零初始化安全接入预训练 backbone”是成熟的
条件适配/残差初始化策略。它可以作为 PSG 的优点和必要实现细节，不能写成主创新。

## 四、ReID 邻近工作：分别封住哪些宽泛主张

| 工作 | 已覆盖机制 | 与 PSG 的边界 | 对 claim 的影响 |
|---|---|---|---|
| [PGFA, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Miao_Pose-Guided_Feature_Alignment_for_Occluded_Person_Re-Identification_ICCV_2019_paper.html) | landmark heatmap 与 global feature map 逐元素相乘，再 pooling/concat | 位于后置 global/part branch，不在 backbone 中层 | 封住“pose heatmap x ReID feature 的乘法是新的” |
| [PVPM, CVPR 2020](https://arxiv.org/abs/2004.00230) | keypoint heatmap/PAF 经 pose encoder 后，用 `1x1 Conv + Sigmoid` 生成 pose-only spatial attention，再对 appearance feature 加权池化 | 只生成 part-level spatial mask，不是逐通道 backbone gate | 封住“从 pose-only 路径学习空间 mask 是新的” |
| [Pose Guided Gated Fusion, WACV 2020](https://doi.org/10.1109/WACV45572.2020.9093370) | pose+appearance 生成空间 x 通道 gate，在 backbone 中层逐元素调制并继续传播 | CNN、非残差、content-adaptive | **直接封住 PSG 的主机制与旧问题句** |
| [PFD, AAAI 2022](https://arxiv.org/abs/2112.02466) | pose heatmap 映射到与 Transformer part feature 同维后逐元素相乘 | 主要用于 part aggregation/decoder | 封住“Transformer 中 pose-derived same-dimensional multiplicative conditioning” |
| [KPR, ECCV 2024](https://arxiv.org/abs/2407.18112) | keypoint heatmap tokenization，与 image token 相加后进入 Swin；解决 multi-person ambiguity | additive prompt、可含正/负 keypoint、问题定义不同 | 封住宽泛“pose 进入 Swin encoder/backbone 是新的” |
| [PAFormer, 2024](https://arxiv.org/abs/2408.05918) | pose heatmap 直接监督 pose-token/patch-token cross-attention；推理时可不使用 pose | training supervision + cross-attention，不是 PSG gate | 封住宽泛“pose-aware Transformer / pose-guided attention 是新的” |
| [FedBPrompt, arXiv 2026](https://arxiv.org/abs/2603.12912) | 在每个 ViT layer 使用 body-distribution-aware prompts 和约束 attention | 固定 body regions，不使用实例 pose heatmap，FedDG-ReID 问题不同 | 不是直接冲突，但进一步压缩“多层人体先验 prompt”的宽泛空间 |

PGFA、PFD、KPR、PAFormer 不是 PSG 的同构实现；它们的作用是说明：即使暂时忽略
WACV 2020，也不能用“pose-feature 乘法”“pose 进入 Transformer”或
“pose-conditioned attention”这些过宽表述建立新颖性。

## 五、逐项 claim 审计

| 候选 claim | 裁决 | 原因 |
|---|---|---|
| 首个 pose-conditioned gate / modulation | **不可写** | SFT/FiLM 是上位类，WACV 2020 是 ReID 直接先例 |
| 首个逐位置逐通道 pose gate | **不可写** | WACV 2020 输出与 appearance feature 同空间、同通道 gate |
| 首次在 backbone 中层注入 pose 并影响后续表征 | **不可写** | WACV 2020 明确在中层 gated fusion 并向后传播 |
| 首次把 pose 引入 Transformer/Swin | **不可写** | PFD、KPR、PAFormer 已覆盖不同形式的 Transformer pose conditioning |
| `1+g` residual gate 是新机制 | **不可写** | FiLM 已明确使用 `gamma=1+delta gamma`；仍属于 SFT |
| zero-init pose gate 是主创新 | **不可写** | 通用 zero-init conditional/residual adapter 已成熟 |
| 多 block / 多 stage pose conditioning 是新机制 | **不可写** | SFT/FiLM 已多层调制；WACV 已研究注入层位置和多层不稳定 |
| backbone-agnostic PSG | **高风险** | 换 backbone 是通用性证据；WACV 2020 已主张兼容不同 feature extractor/loss |
| PSG 建模 skeleton relation | **不可写** | 默认 `1x1` 网络只在同一位置混合 17 通道，没有骨架边或跨位置 relation operator |
| PSG 恢复/补全被遮挡内容 | **不可写** | 对角逐元素缩放不能把位置 `j` 的视觉证据传到位置 `i`，也不生成 appearance content |
| PSG 判断 appearance reliability | **不可写** | gate 只看 `H`，不看当前 appearance feature `X`；CAPSG 才看内容且实验为负 |
| 一种轻量、pose-only、zero-init 的 Swin feature recalibration 实例 | **可以写** | 是准确的实现定位，但应作为组件描述，不当作首创 claim |
| PSG 在本仓库设置下稳定有效 | **可以写但要限定证据** | 有正向实验，不等于算法新颖 |

## 六、仓库实证：有效性成立，但不自动转化为新颖性

### 6.1 目前能支持的有效性结论

旧协议的 3-seed paired 结果：

- baseline：`56.50 mAP`；
- PSG：`57.83 +/- 0.50 mAP / 67.13 +/- 0.84 R1`；
- 三个 seed 的 mAP 差均为正，均值 `+1.33`；
- 但 `n=3` 的双侧 paired test 为 `p=0.1091`，所以准确措辞是
  “稳定正向、样本量仍小”，而不是“已统计显著”。

新协议 clean stage sweep：

| Backbone | no PSG | 1 stage | 2 stages | 3 stages |
|---|---:|---:|---:|---:|
| Swin-Tiny | 59.2 | 60.2 | 60.5 | 60.5 |
| Swin-Small | 68.1 | 68.8 | 68.3 | 68.3 |

这说明：

1. Stage 3 的 PSG 在 Tiny/Small 都有正 mAP；
2. 更多 stage 不是普遍“越多越好”，Tiny 边际饱和，Small mAP 回落；
3. 该 sweep 主要是单次 run，Small 3-stage 还是 seed 41 重跑替代 seed 42 的塌缩
   run，不能把它包装成强统计规律；
4. 把 PSG 移植到 ResNet/普通 ViT 可以补通用性证据，但不会产生新颖性；尤其
   ResNet 会与 WACV 2020 的直接重合更明显。

还要区分“未来可移植”与“当前已经 backbone-agnostic”：现有
`PoseBackboneModel` 直接依赖 Swin 的 `stages/blocks` API；普通 ViT 使用单一
`blocks`，ResNet 又走不同建模路径。当前证据只有 Swin-Tiny/Small 的容量变化，
没有证明同一 PSG 模块已经可直接插拔到 ResNet、普通 ViT 或 CLIP ViT。

### 6.2 历史复杂化没有形成新的可独占机制

仓库已系统尝试多类 PSG 改造：

- multi-stage PSG：旧 exp009 与 1-stage mAP 持平；新 clean sweep 边际饱和且依赖
  backbone；
- `3x3` spatial conv：exp015 mAP 与 PSG 持平，R1 更低；
- pose channel gate：不与 PSG 叠加；
- cross-attention、attention bias、pose reconstruction、pose dropout、pose-weighted
  pooling：均未稳定超过 PSG；
- CAPSG content-adaptive gate：`57.2`，比 PSG `58.3` 低 `1.1 mAP`；
- PAA、FiLM、prompt、pooling、auxiliary loss 等大量外挂只在特定 scaffold 中出现
  局部信号，未把 PSG 本体变成新的函数类。

所以当前最强的诚实结论是“简单 pose-only PSG 是一个有效局部最优”，而不是
“失败变体很多，因此简单 PSG 自动变成创新”。实验成功回答有效性，prior art 决定
可否声称新颖；二者不能互相替代。

## 七、2025-2026 增量检索边界

本轮同时检索了 arXiv 与 OpenAlex，关键词覆盖：

- `person re-identification + pose + gate/gated fusion`；
- `pose/keypoint + feature modulation/recalibration`；
- `pose heatmap + Transformer/Swin + person re-identification`；
- `pose/body distribution + prompt/adapter + ReID`。

截至 2026-07-15，增量检索没有发现比 WACV 2020 更同构的 2025-2026 ReID
论文。近期工作更多转向 body prompts、foundation-model prompting、pose-free
training supervision 或新的 ReID 问题设定。例如 CVPR 2025
[Pose2ID](https://openaccess.thecvf.com/content/CVPR2025/html/Yuan_From_Poses_to_Identity_Training-Free_Person_Re-Identification_via_Feature_Centralization_CVPR_2025_paper.html)
研究 training-free pose-based feature centralization，2026 年 FedBPrompt 研究
body-distribution-aware ViT prompts；二者都不是 PSG 式 block-internal gate。但这不能
救回 PSG 的首创性：旧的 WACV 2020 直接先例和 SFT 上位函数类已经足够完成
NO-GO 裁决。

该检索不声称覆盖所有非英文论文、专利和未公开稿件；它是投稿前 prior-art risk
audit，不是法律意义上的穷尽检索。

## 八、对论文与下一步改造的明确建议

### 8.1 当前 PSG 如何保留

可以保留：

- 作为有效、轻量的 pose-conditioned backbone component；
- 作为与更复杂机制比较的强基线；
- 作为新方法中的条件输入接口或已有工程资产；
- 在 related work 中主动承认与 WACV 2020 Gated Fusion、SFT/FiLM 的谱系关系。

不可继续：

- 标题、摘要和贡献列表把“PSG gate 本体”列为第一主创新；
- 声称现有 pose-ReID 都是在 feature extraction 结束后才使用 pose；
- 用“换成 Transformer”“零初始化”“多 stage”“更少参数”替代算法新颖性；
- 把逐元素 feature gate 改称 self-attention、结构推理或遮挡补全。

### 8.2 什么才算真正跳出当前先例

下一版机制至少必须满足：

1. **不能写成** `gamma(H,X) * X + beta(H,X)` 的逐位置对角仿射；否则仍落回
   SFT/FiLM/PGGF 范畴；
2. 需要新的问题对象，而不是只更换 gate 网络、kernel、stage 或 backbone；
3. 必须有能区分实例 pose 与静态人体布局的 correct/shuffled/canonical/bypass
   因果控制；
4. 必须和 parameter-matched PSG/SFT、generic attention bias、KPR-style additive
   prompt 做强对照；
5. 如果声称移动/恢复视觉证据，算子必须包含真正的跨位置 off-diagonal interaction，
   并验证它没有退化成 diagonal gate。

一个可能的数学跳出方向是 **pose-conditioned conservative feature transport**：

\[
Y_i=X_i+\lambda\sum_{j\in\mathcal N_H(i)}A_{ij}(H,X)
\big(VX_j-VX_i\big),
\]

其中 pose 定义稀疏邻接，`A` 受非负/对称或双随机约束，`lambda` 从 0 启动。
它的 Jacobian 对 `X` 含 `i != j` 的 off-diagonal 项，形式上不再是 PSG/SFT 的
对角缩放，也更接近“移动已有视觉证据”而不是“把 pose 写成 appearance”。但 graph
attention、non-local transport 仍有大量先例，所以这只是下一轮独立查新的候选，
**不是本审计已经确认的新方法**。

## 最终裁决

| 问题 | 回答 |
|---|---|
| PSG 有用吗？ | **有用。** 当前证据支持稳定但幅度有限的正向收益。 |
| PSG 当前公式新吗？ | **不新。** 它可严格归约到 SFT，并有 WACV 2020 ReID 直接先例。 |
| PSG 能继续放在代码和最终系统里吗？ | **可以。** 作为组件、基线或载体。 |
| PSG 能继续单独承担论文主创新吗？ | **不建议，风险接近 NO-GO。** |
| 换到 ResNet/普通 ViT 能让它变新吗？ | **不能。** 只能增强通用性证据；ResNet 反而更贴近直接先例。 |
| 是否还有改造空间？ | **有，但必须离开逐位置仿射 gate 函数类，并先过新的 prior-art gate。** |
