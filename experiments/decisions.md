# 决策记录 — Phase 2: Pure Pose Heatmap

### [2026-03-07 19:30] 决策 #1

**上下文**: Phase 1 (33 个实验) 已充分探索了基于 ViTPose visibility 向量的所有方向。最佳训练端改进仅 +1.4%（GiLt+PCFC）。用户指示放弃 visibility 方向，转向纯 pose heatmap + mmpose 更鲁棒模型。

**选择**: 从 SOLIDER 作者原始代码重新开始，纯 pose heatmap 方向
**理由**:
1. Visibility 向量不够可靠（与 AP 相关性仅 0.237）
2. PCFC alpha suppression 限制了所有后续改进
3. 纯 pose 热图更原始、更可靠、更多论文验证过有效性
4. 从干净代码开始避免 33 个实验的代码污染

**执行计划**:
1. 先跑 baseline 确认可复现（预期 mAP ~56.6%）
2. 选择并部署 mmpose 模型进行 pose heatmap 提取
3. 设计新的 heatmap 利用方式

### [2026-03-07 19:22] 决策 #2

**上下文**: 用户补充了多条重要指导：
1. 姿态模型可以参与训练（不限于离线热图）
2. 尽可能做训练侧创新（NFC/RR 等 test-time 方法不够公平）
3. 可以大胆修改 backbone 中间层
4. 数据必须从 log 文件读取，不能凭记忆

**选择**: 以训练侧创新为核心方向
**理由**:
1. NFC/Re-ranking 等 test-time 方法所有 SOTA 都可以用，不算公平的对比
2. 训练侧创新才是论文的核心贡献
3. 用户允许 pose 模型参与训练 + 修改中间层 → 创新空间大大增加
4. 可以考虑：在线 pose 特征注入、pose-guided attention、pose 结构约束等

**潜在技术方向**:
- 冻结的 pose 模型提取特征 → 通过 cross-attention 注入 Swin 中间层
- Pose heatmap 作为 spatial attention bias 直接修改 window attention
- Pose 骨骼结构约束 part features 之间的关系

### [2026-03-09 11:55] 决策 #3

**上下文**: exp001 (Pose Part Pooling with sigmoid) 完成。结果：mAP 57.1% (+0.5%), R1 66.7% (+0.2%)。有效但提升有限。关键发现：id_part 收敛极慢（最终仍在 2.0 vs id_global 0.2），说明 sigmoid 热图在 12×4 分辨率的 soft attention pooling 不够 discriminative。

**选项**:
  A. 改进 part pooling：使用 spatial softmax 代替 sigmoid，增强热图峰值对比度
  B. 放弃 part pooling，转向 pose heatmap 作为 attention bias 注入 Swin backbone
  C. 使用更高分辨率的中间层特征（stage2: 24×8 而非 stage3: 12×4）

**选择**: 先试 A（spatial softmax 改进），如果 id_part 收敛改善但最终结果仍有限，再转 B
**理由**:
1. exp001 证明 part pooling 方向有效（+0.5% mAP），但 id_part 是瓶颈
2. Spatial softmax 是最小改动，只改一行代码就能验证 "热图对比度不够" 的假设
3. 如果 id_part 收敛问题解决，part pooling 可能有更大提升空间
4. 如果 A 验证后仍不够，则 B 是完全不同的方向，有更大的创新性

**执行结果**: exp002 结果 mAP 57.2% vs exp001 57.1%，几乎无差异。id_part 训练中期收敛更快但最终效果相同。**结论：归一化方式不是瓶颈，转向方案 B。**

### [2026-03-09 14:13] 决策 #4

**上下文**: exp001 和 exp002 结果对比完成。两种归一化方式（sigmoid vs spatial_softmax）效果几乎一致。特征模式消融发现 part-only > concat > global，说明 part 特征有效但融合方式有问题。

**关键发现**:
1. 两种 normalization 最终 part-only mAP 都是 57.5%（+0.9% vs baseline）
2. Concat 融合反而比 part-only 差（1/N scaling 稀释信号）
3. id_part 收敛慢不是 normalization 的问题，而是 12×4 分辨率下 part 区分度本身有限

**选项**:
  A. 改进 part pooling 的融合方式（如 learnable weights, attention-based fusion）
  B. 转向 pose heatmap attention bias 注入 Swin backbone 中间层
  C. 提高 part 特征图分辨率（使用 stage2 特征 24×8）

**选择**: A — 改进特征融合方式。Part 特征已被证明有效（+0.9% mAP），但融合方式拖累了整体效果。这是最直接的改进方向。

**理由**:
1. Part-only 已经超 baseline 0.9%，说明 part 学到了有用信息
2. 当前 concat 的 1/N scaling 太朴素，直接稀释了 part 信号
3. 改进融合方式是低风险高回报：不需要改 backbone，只需修改测试时的特征组合
4. 如果简单的融合改进有效，可以作为消融实验的重要证据
5. B 和 C 是更大的改动，作为备选

**具体方案**: exp003 — 移除 1/N scaling，等权拼接 global + parts；或测试只用 part-only 作为最终特征

**执行结果**: exp003 在 ep60 终止，mAP 50.2%（-6.4% vs baseline）。降低 global loss weight 严重伤害 backbone 特征质量。Part 分类器虽学得更快（id_part 2.08 vs exp001 ~3.3），但池化的 backbone 特征变差了。**结论：global 和 part 是共生关系，不能通过削弱 global 来强化 part。**

### [2026-03-09 15:32] 决策 #5

**上下文**: exp001-003 完成。核心发现：
1. Part pooling 有效（+0.9% mAP with part-only feature）
2. 归一化方式（sigmoid vs spatial_softmax）无差异
3. 融合方式：part-only > concat（1/N scaling 有害）
4. 降低 global weight 反而伤害 part（因为 backbone 质量下降）
5. id_part 收敛极慢是核心瓶颈（id_part≈2.0 vs id_global≈0.2）

**关键问题**: 如何在不削弱 backbone 的前提下增强 part 学习？

**选项**:
  A. 独立 Part BN+分类器 + 更高 LR（加速 part 收敛，不改 global loss weight）
  B. 转向 Direction B：Pose heatmap 作为 attention bias 注入 Swin backbone 中间层（全新方向）
  C. Part feature 使用更高分辨率特征图（stage2: 24×8 而非 stage3: 12×4）
  D. 在 Part head 加入额外的 self-attention 层增强 part 特征表达
  E. 改进 Part 学习信号：per-part triplet loss (GiLt) + part-specific augmentation

**选择**: E — Per-part triplet loss (GiLt)
**理由**:
1. Phase 1 中 GiLt 已证明有效（+0.5% on top of PCFC），这次在 pure heatmap 框架下重试
2. 当前 part triplet 是"所有 part 特征拼起来做一个 triplet"，每个 part 没有独立的 hard positive/negative mining
3. Per-part triplet 让每个 part 独立学习判别性特征，直接解决 id_part 收敛慢的问题
4. 最小改动：只改 loss 计算方式，不改 backbone 或 part pooling 模块
5. 如果 GiLt 有效，可以作为消融实验的重要证据（"per-part triplet vs global triplet"）

**方案 B 作为备选**：如果 GiLt + part pooling 组合无法超过 +1.5% mAP，则转向全新的 backbone attention 方向。

**执行结果**: exp004 PFM 是中性结果。mAP 与 exp001 part-only 相同（57.5%），R1 反而下降 0.8%。PFM 加速收敛但不改善最终表征。**结论：不要在同一处重复使用 pose 信息（PFM+part pooling 是冗余的）。**

### [2026-03-09 17:52] 决策 #6

**上下文**: exp001-004 已探索了当前 part pooling 架构的多个变体：
- exp001/002: 不同热图归一化（sigmoid vs spatial_softmax）→ 无差异
- exp003: 改变 loss 权重 → 负面
- exp004: 加 PFM feature modulation → 中性

当前最佳：mAP 57.5% (part-only), R1 67.1% (+0.9%/+0.6% vs baseline)

**核心瓶颈**: id_part ≈ 2.0 无法进一步降低，part 特征质量受限于 12×4 分辨率。

**选项**:
  A. 使用 stage 2 特征 (24×8, 384ch) 做 part pooling — 4× spatial resolution
  B. Part diversity loss — 惩罚 part 特征间的相似度
  C. 转向 backbone attention 注入 — 修改 Swin 中间层
  D. Part-specific data augmentation — 基于 pose 的部位级数据增强
  E. Adaptive global-part fusion — 学习动态融合权重

**选择**: A — 使用 stage 2 高分辨率特征做 part pooling

**理由**:
1. 当前 12×4 分辨率对 5 个 part 来说太粗（每个 part 只能覆盖 2-3 个 spatial position）
2. Stage 2 (24×8 = 192 positions) 提供 4× 空间分辨率，pose heatmap attention 可以更精确
3. 384 channels 虽然比 768 少，但仍然有丰富的语义信息
4. 实验简单：只需改一下 part pooling 使用的特征图来源
5. 如果分辨率是瓶颈，这个实验会看到 id_part 明显改善

**风险**: Stage 2 特征可能不够 semantic（还没经过 stage 3 的进一步抽象）

**执行结果**: exp005 明确负面。ep40 mAP 仅 37.0%（baseline 56.6%）。id_part 到 ep49 才降到 4.70（exp001 同期 ~2.0）。**确认：Stage 2 特征语义不足以支撑 part-level identity 分类。** 更高空间分辨率无法补偿语义信息的缺失。

### [2026-03-09 19:10] 决策 #7

**上下文**: exp005 证明浅层（stage 2）特征不够 semantic。exp001-005 总结：
- Part pooling 方向的上限在 +0.9% mAP（part-only mode）
- Fusion 方式（concat_scaled vs equal_concat）对最终结果影响约 0.4%
- PFM 是冗余的；stage 2 太浅
- id_part 收敛始终慢于 id_global

**核心问题**：如何突破 +0.9% 的瓶颈？

**选项**:
  A. 改进 test-time 融合（L2-norm concat）— 可能挤出 0.2-0.5%，但不需要训练
  B. 转向 backbone attention 注入 — pose 信息参与特征形成过程
  C. 多尺度 part pooling — stage 2 spatial + stage 3 semantic 的融合
  D. Part feature diversity loss — 惩罚 part 间的冗余
  E. Pose-guided token selection — 剔除无关 token 提高效率

**选择**: 先做 A（test-time L2-norm fusion，零成本验证），然后转 B（backbone attention）

**理由**:
1. A 不需要训练，5 分钟可验证，如果能把 +0.9% 提升到 +1.2%，对论文有价值
2. B 是全新方向，改变特征形成过程本身，可能突破当前 part pooling 的上限
3. C-E 仍在 part pooling 框架内优化，上限有限
4. B 的创新性更好（"pose-conditioned attention" vs "better pooling"），更适合论文

**执行结果**:
- exp006 (A): L2-norm concat 57.4% vs concat 57.2%，小改进但仍不如 part-only (57.5%)。融合方向上限已到。
- **exp007 (B): PSG backbone injection → mAP 58.3%, R1 67.9%。Phase 2 最佳结果！+1.7% mAP, +1.4% R1。超过 Phase 1 最佳 (58.0%/68.0%)。**

### [2026-03-09 21:25] 决策 #8

**上下文**: exp007 PSG 取得突破性结果 (58.3%/67.9%)。关键发现：
1. Backbone-level pose injection (+1.7%) 远优于 post-hoc part pooling (+0.9%)
2. 纯 global feature，无需 part branch，架构极简
3. 额外参数仅 102K（两个 PSG 模块），几乎不增加计算量
4. 已超过 Phase 1 最佳

**下一步方向**:
  A. PSG + Part Pooling 组合 — 让 backbone 和 part branch 同时利用 pose
  B. PSG 消融实验 — 证明 PSG 每个组件的必要性
  C. PSG 在不同 stage 注入 — Stage 2 vs Stage 3 vs 全部 stages
  D. PSG 超参数分析 — hidden_dim, 是否 sigmoid, etc.

**选择**: A — PSG + Part Pooling 组合

**理由**:
1. PSG global feat (58.3%) 和 part-only feat (57.5%) 都有各自的优势
2. PSG 改善了 backbone 特征质量 → part features 也应该受益
3. 组合后可能进一步提升（PSG backbone + enhanced part features）
4. 如果组合有效，这就是完整的方法（backbone injection + part pooling = 全方位 pose 利用）

**执行结果**: exp008 mAP 57.7%, R1 66.0%。**组合不叠加**，低于 PSG-only (58.3%/67.9%)。Part pooling 的 part_only 测试模式丢弃了 PSG 增强的 global feature，而 part features 本身无法匹配 PSG-global 的质量。**结论：backbone-level injection 是更有效的 pose 利用方式，post-hoc pooling 在 PSG 基础上没有增量价值。**

### [2026-03-09 23:35] 决策 #9

**上下文**: exp007 (PSG) 和 exp008 (PSG+Part) 的对比揭示了重要规律：
1. PSG backbone injection: mAP 58.3% (+1.7%) — 全局特征，无 part branch
2. PSG + Part Pooling: mAP 57.7% (+1.1%) — part_only 测试，丢弃 global
3. Part Pooling alone: mAP 57.5% (+0.9%) — exp001

**核心洞察**:
- PSG 的增益主要来自改善全局特征质量，而 part pooling 依赖的是局部特征
- 两种方法的增益来源有重叠：都利用 pose heatmap 做 spatial attention
- 在 part_only 测试模式下，PSG 增强的 global 特征被浪费了

**选项**:
  A. PSG + concat 融合 — 保留 PSG global + part features，不丢弃全局特征
  B. 多 stage PSG — 在 Stage 2 也注入 PSG，更早引入 pose 先验
  C. PSG 改进 — 更强的 gate 机制（如 channel attention, multi-head gate）
  D. Backbone freeze warmup — 冻结 backbone 前 5 epochs，防止随机初始化模块破坏预训练

**选择**: 先做 B（多 stage PSG），这是架构级改进，有更大创新潜力

**理由**:
1. 当前 PSG 只在 Stage 3（2 个 block）注入，信息利用有限
2. 多 stage 注入可以让 pose 信息更早参与特征形成（Stage 2 的 24×8 分辨率对 pose heatmap 更有利）
3. exp005 证明 Stage 2 特征不足以直接做 identity classification，但这不代表 Stage 2 不适合做 spatial attention（PSG 不做分类，只做 spatial gating）
4. 多 stage PSG 是论文中可以画出更好架构图的设计
5. 如果多 stage 有效，这构成了一个"层次化姿态注入"的创新点

**执行结果**: exp009 mAP 58.3%, R1 67.2%, R5 81.2%, R10 85.2%。Multi-stage PSG (Stage 2+3) 与 single-stage (Stage 3 only) mAP 持平，R1 略低（-0.7%），R5/R10 略优（+0.4%/+0.3%），但增加了 156K 额外参数。**结论：Stage 2 PSG 无显著收益，pose spatial gating 在 Stage 3 已足够。后续聚焦于改进 PSG 机制本身，而非扩展注入范围。**

### [2026-03-10 01:45] 决策 #10

**上下文**: Phase 2 九个实验的系统总结：
1. Post-hoc part pooling 上限: +0.9% mAP (exp001)
2. PSG backbone injection 最佳: +1.7% mAP (exp007)
3. PSG + Part Pooling 组合: 不叠加 (exp008)
4. Multi-stage PSG: 无额外收益 (exp009)

**核心发现**: PSG Stage 3 (2 blocks, 102K params) 是当前最优配置。进一步改进需要改变 PSG 的内部机制或训练策略。

**选项**:
  A. PSG Channel Attention — 在 spatial gate 基础上加 channel-wise attention (SE-style)
  B. Backbone Freeze Warmup — 冻结 backbone 前 N epochs，让 PSG 先学稳定的 gate 模式
  C. PSG + Global-Part Concat — concat PSG-global + part features (不是 part_only 测试)
  D. PSG 超参数搜索 — hidden_dim, sigmoid vs tanh, gate 初始化方式
  E. Pose-Conditioned Attention (PCA) — 替代简单的 channel gate，用 pose 调制 self-attention QKV

**选择**: B — Backbone Freeze Warmup

**理由**:
1. 用户曾建议冻结 backbone 前 5 epochs，这个想法值得验证
2. 当前 PSG 零初始化，但训练初期 backbone 的梯度（来自 ID loss 和 triplet loss）会同时更新 backbone 和 PSG，可能让 PSG 来不及学到好的 gate pattern 就被 backbone 适应掉了
3. 冻结 backbone warmup 让 PSG 先在"固定"的特征空间上学习 pose-to-gate mapping，之后解冻 backbone 时 PSG 已有良好初始化
4. 实现简单：只需修改训练循环，前 N epochs 冻结 backbone 参数
5. 如果有效，这个训练策略也是论文素材（"stage-wise training for pose injection"）

**执行结果**: exp010 彻底失败。ep30 mAP 仅 12.5%（exp007 ep30 ~49%），提前终止。冻结 backbone 导致：(1) 解冻后特征空间剧变使 PSG/classifier 失效 (2) PSG 学到错误的 gate pattern 产生负面干扰 (3) 训练指标看似正常但测试表现灾难性。**教训：PSG 必须和 backbone 端到端同时训练。**

### [2026-03-10 02:20] 决策 #11

**上下文**:
- exp007 PSG Stage 3: mAP 58.3% (+1.7%) — 当前最佳
- exp008 PSG+Part: 不叠加
- exp009 Multi-stage PSG: 无额外收益
- exp010 Freeze warmup: 灾难性失败
- **PSG 的基本配置（Stage 3, zero-init, 102K params）已被证明是最优的**
- 需要从不同角度改进

**选项**:
  A. PSG + Channel Attention (SE-style) — 增加 channel-wise gating 维度
  B. Pose-Conditioned Self-Attention — 让 pose 调制 Swin 的 self-attention (QKV)
  C. PSG + Global-Part Concat — 保留 PSG-global + part features 的拼接测试
  D. 更强的 gate 网络 — 增加 3x3 conv, 多层, 或加 spatial conv
  E. 训练策略改进 — 更长训练(160/200 epochs)、不同 LR schedule、不同 optimizer

**选择**: E — 更长训练 (200 epochs)

**理由**:
1. 所有 PSG 实验都在 ep100-120 还在上升，曲线没有完全收敛
2. exp007 的 mAP 轨迹：ep80 56%, ep100 58.2%, ep110 58.2%, ep120 58.3% — 后段仍有微弱上升
3. exp009 也是后发优势型：ep50 才超过 exp007 同期
4. 200 epochs 可以测试 PSG 是否有更大的后段潜力
5. 这是零风险实验：同样的模型，只是训练更久
6. 如果 120→200 epochs 带来 0.5-1% 的额外提升，说明 PSG 确实需要更长训练
7. 对论文也有价值：可以报告"our method continues to improve with longer training"

**执行结果**: exp011 最终 mAP 58.3%, R1 67.6%。与 exp007 (120ep) 完全相同的 mAP，75% 更多训练时间无收益。**结论：PSG 的性能上限由架构决定（~58.3% mAP），120ep 已足够。需要架构创新来突破。**

### [2026-03-10 05:48] 决策 #12

**上下文**: Phase 2 实验全面总结（11 个实验）：
- **有效**: PSG Stage 3 (+1.7%), Part Pooling (+0.9%)
- **无效/中性**: Multi-stage PSG (=), PFM (=), Stage 2 Parts (❌), Part-Dominant (❌)
- **有害**: Freeze Warmup (❌❌)
- **无收益**: 200ep extended training (=)

PSG Stage 3 (mAP 58.3%) 是确认的性能上限。所有尝试过的改进方向都无法突破这个上限。需要全新的方法。

**核心反思**: 当前 PSG 是一个简单的 spatial gate (17→64→768 的 1×1 conv)。它做的是"根据 pose heatmap 在空间维度上调制特征"。这个方法的局限性：
1. 1×1 conv 没有空间感受野 — 每个位置独立处理，不考虑邻域关系
2. 纯 channel-wise output — gate 对每个 channel 产生相同的 spatial weight
3. 不影响 self-attention — PSG 在 SwinBlock 之后作用，不改变 attention 计算
4. 单向信息流 — pose → feature，没有 feature → pose 的反馈

**选项**:
  A. Pose-Conditioned Self-Attention (PCA) — 将 pose 信息注入 Swin 的 self-attention 中（如 attention bias, KV 修改）
  B. PSG 3x3 conv — 在 PSG 中加 3×3 conv（depthwise），让 gate 有空间感受野
  C. Channel Attention PSG — 在 spatial gate 基础上加 channel attention (SE-style)
  D. PSG + Global-Part Concat Test — 测试时 concat PSG-global + part features（非 part_only）
  E. Label Smoothing / Stronger Augmentation — 训练技巧改进

**选择**: A — Pose-Conditioned Self-Attention (PCA)

**理由**:
1. **最高创新潜力**: 修改 self-attention 是比 post-block gate 更深层的干预，直接改变信息交互模式
2. **论文故事**: "从 post-hoc pooling → post-block gating → attention-level injection" 构成清晰的技术递进
3. 实现方式：将 pose heatmap 编码为 attention bias，加到 Swin 的 window attention 分数上
4. 这改变了 token 之间的 attention 权重，比只改变 token 值（PSG 的做法）更根本
5. 如果 PCA > PSG，这是一个强有力的消融证据

**执行结果**: exp012 最终 mAP 57.4%, R1 67.3%。PAB 有效但弱于 PSG（-0.9% mAP, -0.6% R1）。尽管只有 5.4K 参数，attention bias 的调制效果不如 feature gate。**结论：在 Swin 的 window attention 中，additive bias decomposition (bias(i,j) = val[i] + val[j]) 的表达能力有限，softmax 压缩了 bias 的效果。Feature-level gating (PSG) 仍是更有效的 pose 注入方式。**

### [2026-03-10 08:05] 决策 #13

**上下文**: 12 个实验已完成。PSG 仍是最佳方法 (mAP 58.3%)。已验证：
- Post-hoc part pooling: +0.9% (exp001)
- Backbone feature gating (PSG): +1.7% (exp007) ← BEST
- Attention-level bias (PAB): +0.8% (exp012)
- Multi-stage PSG: 无额外收益 (exp009)
- PSG+Part 组合: 不叠加 (exp008)
- 200ep: 无额外收益 (exp011)
- Freeze warmup: 灾难性 (exp010)

**关键排序**: Feature gating (PSG) > Post-hoc pooling > Attention bias (PAB)

**核心问题**: 如何突破 58.3% 的性能上限？

**选项**:
  A. PSG + PAB 组合 — 同时做 feature gating 和 attention bias，双重 pose 注入
  B. PSG + 3×3 Depthwise Conv — 给 PSG 加空间感受野
  C. Cross-Attention Pose Injection — 用 pose token 和 feature token 做 cross-attention
  D. Stronger PSG Gate — 增大 hidden_dim 或加深 gate 网络
  E. Dual-Stream PSG — 分离 body-part gate 和 structure gate

**选择**: A — PSG + PAB 组合

**理由**:
1. PSG 和 PAB 作用在不同层面：PSG 调制 feature values, PAB 调制 attention patterns
2. 两者理论上互补：PAB 让 attention 关注正确的位置，PSG 增强这些位置的特征
3. 两者加起来只有 ~108K 参数，计算开销极低
4. 如果组合有效，这构成了"多层次姿态注入"的创新点（attention + feature + pooling 三个层面）
5. 如果组合无效（像 exp008 一样），也提供了重要消融证据

**执行结果**: exp013 最终 mAP 57.6%, R1 67.2%。**组合未能超越 PSG-only (-0.7% mAP, -0.7% R1)**。与 exp008 类似的规律：两种 pose 注入方式在同一层（Stage 3）互相干扰。PAB 修改了 attention 分布导致 PSG 基于的特征分布发生变化，PSG 的 gate 不再最优。**结论：单一高质量 pose 注入点（PSG）比多个中等质量注入点更好。**

### [2026-03-10 10:15] 决策 #14

**上下文**: 13 个实验完成。Phase 2 的 backbone injection 方向已充分探索：
- PSG Stage 3 only: +1.7% mAP (exp007) ← BEST
- Multi-stage PSG: 无额外收益 (exp009)
- PAB: +0.8% mAP (exp012)
- PSG + PAB combo: +1.0% mAP (exp013，不如 PSG alone)
- PSG + Part Pooling: +1.1% mAP (exp008，不如 PSG alone)
- Extended training: 无收益 (exp011)

**核心认知**: PSG 的 58.3% 性能上限是由**pose heatmap 本身的信息量**和**Stage 3 的 2 个 block**共同决定的。在 backbone 内部叠加更多 pose 模块只会互相干扰。

**选项**:
  A. 转向训练策略创新 — 新的 loss function（如 pose-guided contrastive loss）
  B. 转向推理端改进 — pose-guided re-ranking 或 NFC
  C. PSG 内部机制改进 — 3×3 depthwise conv, channel attention
  D. 全新方向 — 如 Pose-Guided Token Pruning, Pose-aware MixUp
  E. PSG + 不同 loss 组合 — 如 per-part triplet loss (GiLt) 在 PSG backbone 上

**选择**: E — PSG + per-part triplet loss (GiLt)

**理由**:
1. PSG 只改善了特征质量，但训练信号仍然是标准的 ID+triplet loss
2. Per-part triplet loss 可以给 backbone 提供更细粒度的梯度信号，可能进一步提升 PSG 增强的特征
3. Phase 1 中 GiLt 在 PCFC 基础上额外 +0.5%，PSG 基础上可能也有类似增益
4. 这是正交方向的改进（module 级 vs loss 级），不会像 PAB 那样干扰 PSG
5. 实现简单：在 PSG 增强的特征图上做 pose-guided part pooling，然后每个 part 独立 triplet loss
6. 论文价值：训练信号层面的 pose 利用，与 backbone 层面的 PSG 互补

**执行结果**: 用户指出 exp014 与 exp008 训练完全相同，直接加载 exp008 checkpoint 做 global test 即可。结果：mAP 57.6%, R1 65.8%。**比 PSG-only 低 0.7% mAP, 2.1% R1。Part supervision 确实损害了 PSG global feature。** "loss 级正交改进"的假设不成立——part supervision 的梯度回传到 backbone 时会干扰 PSG gate 的学习。

### [2026-03-10 10:40] 决策 #15

**上下文**: 14 个实验完成。完整总结：
- **有效**: PSG Stage 3 only (+1.7% mAP) — 唯一突破
- **弱正**: PAB (+0.8%), Part Pooling (+0.9%)
- **所有组合尝试都失败**: PSG+Part, PSG+PAB, PSG+Part(global test)
- **训练策略**: Multi-stage 无效, Freeze 灾难, 200ep 无效

**关键规律**: 任何修改 PSG 训练时梯度流的改动都会降低 PSG 效果。PSG 的 58.3% 是"全局 ID+triplet + pose spatial gating"的最优解。

**战略性反思**: 继续在 PSG 基础上小修小补已经穷尽了选项。需要跳出"在 backbone 里加东西"的思路。

**选项**:
  A. PSG 机制改进 — 3×3 depthwise conv, channel attention (仍在 backbone 内部)
  B. Test-time 改进 — NFC/re-ranking (不是训练端创新)
  C. PSG + Pose-Guided Data Augmentation — 训练数据层面的 pose 利用
  D. 全新方向: Pose-Guided Feature Disentangling — 用 PSG feature 做 pose/appearance 解耦
  E. 全新方向: Pose-Conditioned Contrastive Learning — pose 相似的样本对应更严格的判别要求

**选择**: A — PSG 内部机制改进（3×3 depthwise conv），而非 C

**修正**: 重新考虑后选择 A。理由：
1. 这是对 PSG gate 本身的改进，不增加新模块或新 loss，不会像 exp008-014 那样干扰梯度流
2. 当前 1×1 conv 每个位置独立计算 gate，没有空间连贯性
3. 3×3 depthwise conv 只增加 576 参数，让 gate 考虑邻域
4. 人体部件是连续区域，相邻位置应有相似 gate 值
5. 如果无效，再转向 C (Pose-Aware Data Augmentation)

**执行结果**: exp015 训练完成。mAP 58.3% 与 exp007 完全持平, R1 67.1% 低 0.8%。3×3 depthwise conv 是冗余的，1×1 gate 已是最优。训练过程中波动极大（差距从 -2.2% 到 +2.3%），但最终收敛到相同水平。PSG 的瓶颈不在感受野。

### [2026-03-10 12:49] 决策 #16

**上下文**: exp015（PSG 空间卷积改进）完成，与原始 PSG 持平。至此：
- PSG 内部结构改进（depthwise conv）：无效
- PSG 外部组合（PAB, Part Pooling, Part Supervision）：全部有害
- PSG 训练策略（freeze, 200ep, multi-stage）：无效

**已穷尽的方向**: 在"PSG + 全局 ID/Triplet Loss"框架内的所有优化都已探索完毕。PSG 58.3% mAP 是该框架的理论上限。

**需要根本性的方向转变**: 不再在 PSG 上修修补补，需要全新的利用 pose heatmap 的方式。

**选项**:
  A. Pose-Guided Feature Disentangling — 用 pose 热图将特征解耦为 part-specific 子空间
  B. Pose-Conditioned Contrastive Learning — 基于 pose 相似度的对比学习
  C. Pose-Guided Data Augmentation — 基于 pose 的数据增强
  D. Adaptive PSG — 根据遮挡程度动态调节 PSG 强度
  E. Deformable PSG — 可变形卷积替代固定网格，对齐到关键点位置

**选择**: A → H (Pose-Guided Erasing)

**执行结果**: exp016 完成。PGE 严重有害（mAP 54.8% vs exp007 58.3%，-3.5%）。身体部件级擦除过强+削弱 PSG 输入。数据增强层面的 pose 利用方向失败。

### [2026-03-10 15:10] 决策 #17

**上下文**: exp016 PGE 失败后，16 个实验的完整总结：
- PSG 58.3% 是唯一有效方法
- 所有扩展/组合/增强/增强均失败
- 数据增强方向（PGE）也失败

**关键反思**: 已经把 PSG 本身和所有"加法"都试遍了。需要完全不同的思路。

**选项**:
  A. Pose-Conditioned Normalization (PCN) — 用 pose 调制 LayerNorm 参数
  B. 全新模型: Deformable PSG — 可变形卷积对齐关键点
  C. 全新方向: Pose Structure Token — 在 Swin 输入层添加 pose 编码
  D. 测试端优化: 基于 PSG 的 NFC/Re-ranking
  E. 跨步思考: 不再改 PSG，而是替换整个 Part Pooling 方案

**选择**: 先尝试 Pose-Conditioned Channel Gate (PCG)——在 GAP 后做通道级 pose gating，与 PSG 的空间级正交

**执行结果**: exp017 完成。PCG 与 exp007 基本持平（mAP 58.0% vs 58.3%，-0.3%）。证实了通道级正交操作不干扰 PSG（不像之前的 combo 实验那样降性能），但 GAP 后的 pose 信息太弱无法提供额外收益。

### [2026-03-10 17:25] 决策 #18

**上下文**: exp017 PCG 中性结果。17 个实验的总结：
- PSG 58.3% 仍是最佳，所有扩展都无法超越
- **新发现**: 正交维度操作不干扰 PSG（PCG 仅 -0.3%），但空间级操作干扰 PSG（PAB combo -0.7%, Part pooling -0.6%）
- 这说明 PSG 的空间 gating 格局已定，但通道维度仍有空间

**关键洞察**: PCG 失败是因为 GAP 压缩了空间信息。如果在 GAP **之前** 做通道调制（保留空间信息），可能有效。

**选项**:
  A. SE-style Pose Channel Attention (在 GAP 前) — 用 pose 热图生成通道权重，应用到 feature map 上（空间信息还在）
  B. PCG-only (无 PSG) — 验证 PCG 单独效果，排除 PSG 已经做了足够 conditioning 的可能
  C. 完全新方向：跳出 PSG 框架

**选择**: B — PCG-only 消融实验

**执行结果**: exp018 完成。**PCG-only mAP 57.8%, R1 67.7%（+1.2%/+1.2% vs baseline）。** 出乎意料，PCG 有独立效果！推翻了"PCG 无效"的初步判断。但仍低于 PSG (58.3%/-0.5%)，且 PSG+PCG (exp017) 不叠加。结论：PCG 和 PSG 各自有效但捕获相似的 pose 信号，组合不互补。
