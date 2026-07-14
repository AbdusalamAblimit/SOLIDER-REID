# exp373 SA 正交耦合专项查新

## 审查问题

候选机制试图把 PAA residual 投影到 PSG displacement 的正交补：

\[
b^\perp=b-\operatorname{Proj}_{\operatorname{sg}(d)}b,
\qquad y=x+d+b^\perp.
\]

审查不以“是否存在完全相同缩写”为标准，而回答：

1. 该操作是否脱离普通条件仿射调制；
2. hard orthogonal residual operator 是否已有直接先例；
3. 在 ReID 中，人体形状/姿态相关子空间与正交补身份表征是否已有直接先例；
4. 剩余差异是否足以承担论文主贡献。

## 普通 PSG+PAA 的归约

现有串联为：

\[
y=x\odot(1+g(H))+a(H).
\]

令 `gamma(H)=1+g(H)`、`beta(H)=a(H)`，即逐位置、逐通道条件仿射调制。
FiLM 已覆盖由 conditioning generator 为多个 residual block 预测 scale/shift；
SPADE 已覆盖由空间条件图生成逐位置、逐通道的 scale/shift；SSF 又覆盖在
Transformer 多个 operation 后反复执行 scale/shift。共享 pose stem、分离 head、
zero-init、交换 PSG/PAA 顺序或扩大到更多层均不改变这个判断。

直接近邻：

- FiLM：Perez et al., 2018，<https://arxiv.org/abs/1709.07871>
- SPADE：Park et al., 2019，<https://arxiv.org/abs/1903.07291>
- SSF：Lian et al., 2022，<https://arxiv.org/abs/2210.08823>
- ControlNet：Zhang et al., 2023，<https://arxiv.org/abs/2302.05543>

因此，“multiplicative PSG + additive PAA”“多层同时调制”“共享 stem 双 head”
均不得作为新颖性来源。

## 对正交版本的两种严格解释

### 解释 A：相对 pose-only gate 做投影

若投影方向仅由 pose 产生，则 `b_perp` 仍只依赖 `H`，整个输出仍可写成：

\[
y=\gamma(H)\odot x+\beta(H),
\]

只是多了 `beta` 与某个 gate direction 正交的参数约束。它是条件仿射函数族的
受约束子集，不产生新的交互对象。

### 解释 B：相对实际 PSG displacement 做投影

本实验更强的定义使用 `d=x_psg-x=x\odot g(H)`。此时 `b_perp` 同时依赖
`x,H`，不能再严格写成 pose-only `beta(H)`；但真正新增的 operator 已变成：

> 把一个 residual update 投影到另一个当前特征/更新方向的正交补，再残差写回。

这一 operator 和“主方向/正交补分别承载不同信息”的核心叙事已有高度重合先例。

## 致命直接先例

### 1. Orthogonal Residual Update

Revisiting Residual Connections: Orthogonal Updates for Stable and Efficient Deep Networks，
arXiv 2025，<https://arxiv.org/abs/2505.11881>。

该工作在 ResNet/ViT 中显式执行：

\[
f_\perp=f-\frac{\langle x,f\rangle}{\|x\|^2}x,
\qquad x'=x+f_\perp,
\]

并使用 radial/tangential、避免残差方向冗余的解释。exp373 把投影参考从输入
`x` 换成 PSG displacement `d`，属于同一 hard orthogonal residual operator 的
条件化应用，不能声称提出了新的正交更新机制。

### 2. Shape-Erased Feature Learning for Visible-Infrared Person Re-Identification

CVPR 2023，<https://arxiv.org/abs/2304.04205>。

该工作利用人体 shape/pose prior 建立 shape-related subspace，并用投影矩阵把
特征分解为相关分量与正交补分量：

\[
z_{sr}=PP^Tz,
\qquad z_{se}=(I-PP^T)z.
\]

论文目标正是让正交补承载 shape-independent identity cues。它已经覆盖了
“人体结构相关方向 + 正交补身份方向”“通过正交分解获得互补 ReID 表征”的关键叙事。

### 3. Ortho-ReID

ICML 2026 CoLoRAI Workshop，<https://arxiv.org/abs/2606.11661>。

该工作在 ReID 中学习 instance-adaptive orthonormal subspace，并约束身份与衣着
因素正交。虽然问题对象是换衣 ReID，而不是 PSG/PAA，但它进一步压缩了
“instance-adaptive orthogonal subspace 是 ReID 新机制”的可声明空间。

## 其他边界

1. 将 `b_perp` 称为 radial/tangential 并不严谨：真正 radial direction 应相对
   当前特征或实际更新定义；PSG 是逐通道 scale，`x*g(H)` 通常不与 `x` 平行；
2. stop-gradient 只改变优化耦合，不带来统计或参数可识别性；两个 head 仍可共同
   旋转、缩放或改变通道基来绕开解释；
3. `inner-product=0` 依赖通道基和尺度，不具重参数化不变性；
4. 即使旧 checkpoint 中 overlap 很高，最多说明正交正则可能有工程用途，不能
   消除上述先例；
5. 即使 virtual projection 提升检索指标，也只能把它定位为 PSG+PAA 的辅助正则。

## 新颖性裁决

**FAIL / NO-GO。**

普通 SA 是已有条件仿射；相对实际 displacement 的强版本虽然加入 `x-H`
交互，但其关键 hard orthogonal residual operator 与“人体结构子空间/正交补身份
特征”叙事均已有直接先例。剩余差异只有“把已有正交残差用到 PSG/PAA 两支之间”，
属于应用位置变化，不足以承担自有主创新。

根据 `design.md` 的预注册规则，新颖性门禁失败即停止：不运行 overlap forward、
不实现、不训练，也不转 routing、adaptive gate、content-LoRA、普通 FiLM、transport、
层数或阈值小变体。
