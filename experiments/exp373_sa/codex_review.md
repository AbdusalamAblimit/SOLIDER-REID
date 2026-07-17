# exp373 SA Gate 0：Codex 多路审查

## 审查范围

三路独立只读审查覆盖：

1. 仓库实现、历史 stage/block 消融与 matched PAA 对照；
2. FiLM/SPADE/多层 conditioning、orthogonal residual、pose/shape subspace 与
   ReID 直接先例；
3. overlap 指标、高维随机 null、double-sigmoid、counterfactual controls 与
   预注册门槛红队。

未使用 Claude，未修改训练代码，未启动推理或训练。

## 一致事实

1. 当前代码已经在每个启用 stage 的每个 block 后执行 PSG→PAA；
2. `exp073` 已做 Stage 2+3 同步 PSG+PAA，并比 Stage-3-only 低 `0.5 mAP`；
3. matched `exp251/exp254` 中，两阶段 PAA 边际为 `-0.3 mAP/-0.6 R1`；
4. clean stage sweep 不支持“PSG 越多层越好”的普遍规律；
5. 现有 heatmap 已是 `[0,1]`，PSG/PAA 再 sigmoid，zero-input 不是 no-pose；
6. 普通 scale+shift 是 FiLM/SPADE 类条件仿射，不能形成主创新。

## Fuel audit 红队结果

若新颖性门禁能够通过，合理的只读统计应使用真实中间输出：

\[
d=x_{PSG}-x,
\qquad b=x_{PAA}-x_{PSG},
\qquad
R_E=\sum\|Proj_db\|^2/\sum\|b\|^2.
\]

逐 token 单方向随机投影期望仅为 `1/C`：Stage2 约 `0.26%`，Stage3 约
`0.13%`。必须以跨图、空间、通道置换构造经验 null，并按图像 bootstrap；
zero-input、true bypass、shuffled、canonical 不能混为同一个 no-pose control。

该协议本身有效，但不是继续执行的充分理由，因为新颖性门禁是逻辑上更早的
必要条件。

## 可用 checkpoint 资产

已只读确认 exact `exp066` 资产仍存在：

- 机器：`lab-3090-d`
- repo：`/root/work/SOLIDER-REID`
- checkpoint：`log/occluded_duke/exp066_paa/transformer_120.pth`
- checkpoint SHA256：
  `a084d84995f8fcfd53eea19d8c674d1cdce07d954d9cafbd78e73a211a8903ad`
- execution commit：`8eacaf16dcd797ab8090fe19aca49f80f86bec6a`
- result：`61.6 mAP / 74.2 R1`
- 结构：Stage3 block0/block1 均有 PSG 与 PAA 参数。

资产齐备排除了“因为找不到 checkpoint 才停止”的解释；停止原因完全是新颖性
门禁失败。

## 新颖性红队结果

候选存在两种解释：

1. 对 pose-only gate 投影：仍是带正交约束的 FiLM 子集；
2. 对实际 `x*g(H)` displacement 投影：引入 content-conditioned residual，
   但 hard orthogonal residual operator 已由 arXiv 2025 Orthogonal Residual
   Update 直接覆盖；CVPR 2023 Shape-Erased VI-ReID 和 ICML 2026 CoLoRAI
   Workshop Ortho-ReID 又覆盖人体结构/外观相关子空间与正交补身份表征的
   ReID 叙事。

stop-gradient、zero-init、独立 stage mask、严格 controls 与更好的审计设计都是
良好工程/证据属性，但不能替代机制创新。

## Verdict

**新颖性 Gate FAIL，正式 NO-GO。**

- 不运行 checkpoint forward audit；
- 不实现 `POSE_PAA_STAGES` 或 orthogonal projection；
- 不占用 3090/4090；
- 不进入 e60/e120 矩阵；
- 不转 transport、routing、adaptive gate、content-LoRA、普通 FiLM、层数/阈值
  小变体。

边界：本裁决不否定 PSG 的既有增益，也不否定 PAA 在早期 `PSG+GCN` scaffold
两个 seed 上的历史正信号；它只否定把 PSG/PAA 加正交投影后作为新的论文主贡献。
