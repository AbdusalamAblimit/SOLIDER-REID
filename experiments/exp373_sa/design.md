# 实验 exp373：SA 非冗余姿态调制 Gate 0

## 当前状态

- 阶段：新颖性门禁已完成
- 训练：未启动
- 代码实现：未开始
- GPU：不占用
- 结论：`NOVELTY_GATE_FAIL / NO-GO`

## 动机

历史 PSG 在多个数据集和 backbone 上有稳定正向证据；PAA 在早期
`PSG+GCN` scaffold 的两个 seed 上也出现过正向边际作用。但是，当前实现和
历史消融不支持直接把二者包装成“新的多层 scale–shift attention”：

1. 当前代码已经在每个启用 stage 的每个 Swin block 后依次执行 PSG 和 PAA；
2. `exp073` 的 Stage 2+3 同步 PSG+PAA 低于 Stage-3-only `0.5 mAP`；
3. `exp251` 与 matched `exp254` 表明，在强 scaffold 的两阶段配置中，加入
   PAA 为 `-0.3 mAP / -0.6 R1`；
4. 普通 `x * (1 + gamma(H)) + beta(H)` 可直接归约为 FiLM/SPADE 类条件仿射；
5. 现有 PSG/PAA 对已经位于 `[0,1]` 的 ViTPose heatmap 再次 sigmoid，令
   zero-input 变成常数 `0.5`，不能解释为 no-pose 或 identity。

因此，本实验不预设“层数越多越好”，也不直接训练新模块。首先判断 PSG
实际位移与 PAA 实际残差之间是否存在足够大、与正确图像—姿态对应相关、且
不是随机高维重叠的可利用信号。

## 候选机制（仅供 Gate 0 审计）

对某个 block 的真实中间输出定义：

\[
d_l = x_l^{\mathrm{PSG}} - x_l,
\qquad
b_l = x_l^{\mathrm{PAA}} - x_l^{\mathrm{PSG}}.
\]

候选 SA 将 PAA 残差投影到 stop-gradient PSG displacement 的正交补：

\[
b_l^\perp = b_l -
\frac{\langle b_l,\operatorname{sg}(d_l)\rangle}
{\|\operatorname{sg}(d_l)\|^2+\epsilon}
\operatorname{sg}(d_l),
\qquad
y_l=x_l+d_l+b_l^\perp.
\]

该定义的目标不是通过改名声称乘法和加法天然互补，而是建立一个可逐 token
验证的非重叠约束。它是否具有足够新颖性、是否有真实燃料，均由 Gate 0 决定。

## Gate 0-A：资产与输入语义

1. 优先使用历史 `exp066` seed1234/seed42 checkpoint；若不可得，允许使用
   已定位的 4090 `Swin-Small + PSG+PAA+GCN+OA-SD` seed1234 checkpoint，
   但只能形成探索证据；
2. 记录 checkpoint SHA256、config、可恢复 commit、stage/block 数和 descriptor；
3. 对不少于 512 张图统计送入模块前 heatmap 的 min/p1/p50/p99/max、零比例，
   以及当前重复 sigmoid 后的范围；
4. 严格区分：
   - correct heatmap；
   - zero-input（旧路径会变为 0.5，不是 no-pose）；
   - true bypass（真正不执行 PSG/PAA）；
   - shuffled / canonical 解释性控制。

## Gate 0-B：重叠燃料审计

通过只读 forward hooks 捕获 `x_l`、`x_l^PSG`、`x_l^PAA`，用 float32 计算：

- `||d|| / ||x||`；
- `||b|| / ||x||`；
- token cosine 与 cosine squared；
- 主指标：

\[
R_E=\frac{\sum\|\operatorname{Proj}_{d_l}b_l\|^2}
{\sum\|b_l\|^2}.
\]

统计按图像 bootstrap，不把相关 token 当独立样本；分别报告 block/stage 和
aggregate 结果。屏蔽 `d` 或 `b` norm 的 bottom 5%，并报告保留覆盖率。

经验 null 至少包括 100 次：

1. batch 内跨图 derangement；
2. 同图空间 token 置换；
3. 固定 channel 置换。

逐 token 单方向投影的各向同性随机期望约为 `1/C`：Stage 2 约 `0.26%`，
Stage 3 约 `0.13%`。因此绝对重叠必须同时结合经验 null 判断。

## Gate 0-C：虚拟投影与新颖性门禁

在不改 checkpoint、不训练的前提下，允许 in-memory 比较：

1. 原始 PAA；
2. virtual orthogonal projection；
3. 随机方向移除相同 `b` 能量的 norm-matched controls。

同时专项查新 hard orthogonal projection、residual orthogonalization、
pose/shape subspace decomposition、conditional modulation 和 ReID 直接近邻。
如果候选可被直接归约为已有 orthogonal residual 或 pose/shape decomposition，
无论虚拟指标是否正向，都不得进入训练。

## 预注册决策

### 直接 NO-GO

- 新颖性门禁发现直接机制先例或可归约关系；
- 所有可用 checkpoint 的 aggregate `R_E < 3%`；
- 95% CI 上界不超过 `max(3%, null99 + 1pp)`；
- PAA activity 已塌缩；
- correct 相对 shuffled 和 canonical/mean-pose 均小于 `+0.3 mAP`。

### 仅允许补只读诊断

- `R_E` 位于 `3%–10%`；
- 只有单 block、单 epoch 或单 checkpoint 成立；
- virtual projection 与随机等能量移除无法区分。

### 允许进入实现

必须同时满足：

1. 至少两个独立 seed 的 aggregate `R_E > 10%`；
2. 每 seed 的 95% CI 下界至少 `3%`；
3. observed 至少为 null mean 的 5 倍，且不是低范数 token 驱动；
4. correct 相对最强 pose control 至少 `+0.3 mAP`；
5. 专项查新不能把机制直接归约为已有方法。

## Gate 0 通过后的计划（当前不执行）

1. 新增独立 `POSE_PAA_STAGES`；
2. PSG 使用 Stage 2+3，PAA 只使用 Stage 3；
3. 新路径不重复 sigmoid，并严格保证 `H=0` 时 scale/shift 都为零；
4. 所有新功能默认关闭；
5. 记录分支 norm、投影能量、projection cosine、梯度与 identity parity；
6. 先在 Swin-Tiny、固定 batch size、标准 global descriptor、无
   GCN/LGPA/OA-SD/PLBOA/特殊 matching 下跑预注册最小矩阵。

## 风险与失败解释

1. 高维空间中随机向量天然近似正交，硬投影可能几乎不改变 PAA；
2. 旧 checkpoint 已按 naive PSG+PAA 共同适配，virtual projection 负面不能单独
   否定重新训练，但能揭示高风险；
3. double-sigmoid 不妨碍 correct/shuffled 的同分布比较，却会破坏
   zero-input=no-pose 的解释；
4. exp066 含 GCN，干预 PSG/PAA 时必须保持 GCN 接收 correct pose，避免归因混淆；
5. 即使工程上有涨点，如果机制已有直接先例，仍按 NO-GO 封板。

## 最终裁决

专项查新发现：普通 PSG+PAA 是 FiLM/SPADE 类条件仿射；若对 pose-only gate
投影，正交版本仍是带约束的条件仿射子集；若对实际 `x*g(H)` displacement
投影，关键 hard orthogonal residual operator 已被 arXiv 2025 Orthogonal
Residual Update 覆盖，而 CVPR 2023 Shape-Erased VI-ReID 与 ICML 2026
CoLoRAI Workshop Ortho-ReID 已覆盖 ReID 中人体结构/外观相关子空间和正交补
身份表征的核心叙事。

因此触发预注册“新颖性门禁发现直接机制先例或可归约关系”的直接 NO-GO 条件。
checkpoint 与数据虽已定位齐全，但不再运行 overlap forward；不实现、不训练、
不扩变体。详见 `literature_novelty.md` 与 `codex_review.md`。
