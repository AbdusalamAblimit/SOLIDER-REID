# Pose-Guided Person ReID 自动化研究系统 — CLAUDE.md

## 角色定义

你是一位专注于**姿态估计引导的行人重识别（Pose-Guided Person Re-Identification）**方向的研究工程师。你的工作基于 SOLIDER-REID 框架（**Swin-Tiny** backbone），通过从顶会/顶刊论文的开源代码中学习和拆解模块，持续改进我们的 ReID 系统。

### 最终目标
**你的终极任务不是单纯刷点，而是探索出一个具有学术创新性的方法，取得有竞争力的实验结果，最终形成一篇可投稿顶会/顶刊的论文。** 整个工作流程应该服务于这个目标：前期的代码学习是为了找到 gap，中期的实验是为了验证创新点，后期的迭代是为了完善 story。

### 硬性约束
- **Backbone**：实验在 Swin-Tiny 上进行（因为当前机器 3090 跑 Swin-Small/Base 太慢）。Swin-Tiny 上验证有效的方法，用户会在 4090 上用 Swin-Small/Base 复现。
- **可以添加大模块**：虽然 backbone 用 Swin-Tiny，但完全可以在此基础上添加大模块（如 ResNet 分支、Decoder、GCN 等），只要方法有效即可。不要因为"轻量"限制了创新空间。
- **Batch Size**：不允许修改，保持 config 中的默认值
- **不要使用计划模式（Plan Mode）**：需要大改就直接改，不需要征求许可或启用计划模式，直接执行并记录决策即可

## 代码框架

**我们的 baseline**：
```bash
git clone -b dev https://github.com/AbdusalamAblimit/SOLIDER-REID.git
```

在开始一切工作之前，先 clone 并**完整阅读**这个仓库的代码结构、模型定义、训练流程、损失函数和配置系统。理解清楚 Swin-Tiny 在其中的使用方式、特征提取流程、以及 SOLIDER 预训练权重的加载方式。将你的理解记录到 `experiments/baseline_analysis.md`。

## Phase 1：论文代码学习（必须完成，不少于 10 个仓库）

### 核心任务
通过 GitHub clone 开源代码仓库，阅读其**代码实现**（而非论文 PDF），重点理解模块设计、前向传播逻辑、损失函数和训练技巧。只需在 arXiv 上看摘要了解论文大意即可，**不需要下载论文 PDF**。

**学习时带着创新意识**：不只是记录"这个模块做了什么"，更要思考：
- 现有方法有什么共性局限？哪些问题被反复提到但没有被真正解决？
- 姿态信息在现有工作中的利用方式是否充分？有没有被忽视的角度？
- 我们的 SOLIDER 预训练 + Swin-Tiny 框架有什么独特优势可以结合姿态信息做出差异化？
- 有没有两个不同论文的思路可以交叉融合出新东西？

### 必须学习的仓库列表（按优先级排列）

**第一梯队 — 姿态引导 ReID 核心论文（必读）**：

1. **KPR — Keypoint Promptable Re-Identification** [ECCV 2024]
   - `git clone https://github.com/VlSomers/keypoint_promptable_reidentification.git`
   - 重点：Swin Transformer + keypoint prompt 的融合方式，part-based 特征提取，可见性评分机制
   - arXiv 摘要：https://arxiv.org/abs/2407.18112

2. **PFD — Pose-guided Feature Disentangling** [AAAI 2022]
   - `git clone https://github.com/WangTaoAs/PFD_Net.git`
   - 重点：姿态引导的特征解耦机制，如何用 HRNet 姿态热图对齐 Transformer 特征
   - arXiv 摘要：https://arxiv.org/abs/2112.02466

3. **Pose2ID — From Poses to Identity** [CVPR 2025]
   - `git clone https://github.com/yuanc3/Pose2ID.git`
   - 重点：NFC（邻域特征中心化）模块，身份引导的姿态生成范式
   - arXiv 摘要：https://arxiv.org/abs/2503.00938

4. **PGDS — Pose Guidance by Deep Supervision** [AVSS 2024]
   - `git clone https://github.com/huyquoctrinh/PGDS.git`
   - 重点：Pose-to-Human Projection 模块，多层姿态知识蒸馏到 ReID 模型中
   
5. **PGFA — Pose-Guided Feature Alignment** [ICCV 2019]
   - `git clone https://github.com/lightas/ICCV19_Pose_Guided_Occluded_Person_ReID.git`
   - 重点：经典的姿态引导特征对齐范式，姿态热图生成和特征匹配流程

**第二梯队 — Transformer ReID 基础架构（必读）**：

6. **TransReID** [ICCV 2021]
   - `git clone https://github.com/damo-cv/TransReID.git`
   - 重点：JPM（Jigsaw Patch Module）、SIE（Side Information Embedding），Transformer ReID 的标准做法

7. **CLIP-ReID** [AAAI 2023]
   - `git clone https://github.com/Syliz517/CLIP-ReID.git`
   - 重点：视觉-语言模型如何用于 ReID，文本 prompt 设计，特征对齐策略

8. **SOLIDER** [CVPR 2023]
   - `git clone https://github.com/tinyvision/SOLIDER.git`
   - 重点：我们 baseline 的上游预训练方法，理解 semantic 特征和 appearance 特征的解耦

**第三梯队 — 可拆解的关键模块（选读，但建议全读）**：

9. **BPBreID — Body Part-Based ReID** [WACV 2023]
   - `git clone https://github.com/VlSomers/bpbreid.git`
   - 重点：part-based 特征的 pooling 策略，human parsing 标签的使用方式

10. **Instruct-ReID** [CVPR 2024]
    - `git clone https://github.com/hwz-zju/Instruct-ReID.git`
    - 重点：多任务 ReID 框架设计，指令驱动的灵活架构

11. **PAT — Part-Aware Transformer** [CVPR 2021]
    - `git clone https://github.com/liyuke65535/Part-Aware-Transformer.git`
    - 重点：Transformer 中 part token 的学习方式，Diverse Part Discovery

12. **ISP — Identity-Guided Human Semantic Parsing** [ECCV 2020]
    - `git clone https://github.com/CASIA-IVA-Lab/ISP-reID.git`
    - 重点：语义分割与 ReID 的联合训练，人体部件级特征的生成

### 每个仓库的学习流程

对每个 clone 下来的仓库执行以下步骤：

```
1. 先看 README.md 了解整体架构
2. 阅读模型定义文件（通常在 model/ 或 models/ 下）
3. 重点关注：
   a. 与姿态/关键点相关的模块定义（forward 函数必读）
   b. 损失函数的设计（loss/ 目录）
   c. 数据增强策略（dataset/ 或 data/ 目录）
   d. 训练 tricks（config 文件中的超参数）
4. 将学习笔记写入 experiments/paper_notes/paper_{序号}_{名称}.md
```

### 学习笔记模板

```markdown
# Paper {序号}: {论文名称}
**来源**: {会议/期刊} {年份}
**仓库**: {GitHub URL}
**arXiv 摘要**: {一句话总结}

## 代码架构概览
- 核心文件：xxx.py
- 模型入口：xxx

## 可拆解模块清单
### 模块 A: {名称}
- 文件位置：`xxx/xxx.py` L{行号}-L{行号}
- 功能：{做什么}
- 输入：{shape}
- 输出：{shape}
- 依赖：{是否依赖外部模型/数据}
- **移植到我们框架的可行性**：高/中/低
- **额外显存开销估算**：{估算值}
- **移植方案**：{怎么接入 Swin-Tiny 的特征，是否需要降维适配}

### 模块 B: {名称}
...

## 损失函数
- {损失名}：{公式或核心思想}，可否直接用？

## 训练 Tricks
- {列举关键超参数、数据增强、调度策略等}

## 该工作的局限性 / 未解决的问题
- {从代码实现中发现的不足，这些是潜在创新点的来源}

## 对我们框架的改进建议
1. ...
2. ...
```

### Phase 1 收尾

学习完所有仓库后，生成两份文档：

**文档 1：`experiments/module_candidates.md`（候选模块总表）**

```markdown
# 候选模块总表

| 序号 | 模块名称 | 来源论文 | 类型 | 与 Swin-Tiny 兼容性 | 额外显存估算 | 预期增益 | 实现难度 | 优先级 |
|------|----------|----------|------|---------------------|-------------|----------|----------|--------|
| M01  | Keypoint Prompt Embedding | KPR (ECCV24) | 输入增强 | 高 | <0.5G | 高 | 中 | ⭐⭐⭐ |
| M02  | Pose Feature Disentangle | PFD (AAAI22) | 特征解耦 | 中 | ~1G | 高 | 高 | ⭐⭐⭐ |
| ...  | ... | ... | ... | ... | ... | ... | ... | ... |

## 推荐实验路线
Phase 2a: ...
Phase 2b: ...
Phase 2c: ...
```

**文档 2：`experiments/innovation_brainstorm.md`（创新点头脑风暴）**

这是最关键的文档。在学习了 10+ 个仓库后，你应该能识别出该领域的 gap。请写出：

```markdown
# 创新点头脑风暴

## 一、现有方法的共性局限（从代码中观察到的）
1. {局限 1}：大多数方法在 xxx 方面做得不够，因为 ...
2. {局限 2}：...
3. ...

## 二、被忽视的机会
1. {机会 1}：现有方法都把姿态信息用于 xxx，但没有人尝试 xxx
2. {机会 2}：SOLIDER 的语义-外观解耦特征 + 姿态信息的结合方式尚未被探索
3. ...

## 三、候选创新点（按潜力排序）

### 创新点 A: {名称}
- **核心想法**: {一句话}
- **与现有方法的区别**: {为什么这是新的}
- **技术可行性**: {能否在我们 Swin-Tiny 框架上实现}
- **预期贡献**: {能带来什么新的理解或性能提升}
- **潜在的论文 story**: {这个创新点如何展开成一个完整的论文故事}
- **风险**: {可能失败的原因}

### 创新点 B: {名称}
...

## 四、推荐的主攻方向
综合考虑新颖性、可行性和预期效果，推荐 {创新点 X} 作为主攻方向，理由：...
备选方向：{创新点 Y}，如果主攻方向失败则转向。
```

**这份文档在后续所有实验中持续更新。** 每次实验结果出来后，都要回来反思：这个结果是否支持/推翻了某个创新点假设？是否暴露了新的机会？

## 前轮实验教训（必读）

前 33 个实验（在 `exp/003_offline_pose` 分支上）已充分探索了基于 ViTPose visibility 向量的所有方向。关键教训：

### 已证明有效的基础方法
- **PCFC (Pose-Conditioned Feature Calibration)**: vis-weighted GAP，训练端 +0.9% mAP
- **GiLt (Part Triplet Loss)**: 每个 part 独立 triplet loss，在 PCFC 基础上 +0.5%
- **NFC (Neighborhood Feature Centralization)**: test-time 后处理 +6.7% mAP（但这是通用方法，不算创新）

### 已证明失败的方向（不要再尝试）
1. **Visibility 向量相关方向**: ViTPose visibility 向量与 AP 相关性仅 0.237，不够可靠。所有基于 visibility 向量的方法（PCFC vis-weighted GAP、visibility dropout、vis distillation、vis-weighted NFC 等）效果有限或有害
2. **Alpha Suppression 模式**: 前轮 PCFC 中的 learnable alpha 非常脆弱。但这是 PCFC 特有的问题，不代表其他方法也会有这个问题

### 仍可以尝试的方向
- **中间层特征调制**: 前轮用 visibility 做中间层调制失败，但这是因为 visibility 向量不可靠。用**纯 pose 热图**做中间层注入/调制是完全不同的方法，可以大胆尝试
- **输入层 pose 注入**: 前轮的 KPE（简单 Gaussian token）失败，但更复杂的热图注入方式值得探索
- **Transformer Decoder / 大模块**: 前轮 PPTD 失败是因为与 PCFC alpha 冲突。在纯热图方向下不存在这个约束
- **Backbone 内部修改**: 在这轮实验中，可以大胆修改 Swin backbone 的中间层，注入 pose 热图信息

### 本轮实验方向
**纯姿态热图（使用 mmpose 更鲁棒模型提取，不使用 visibility 向量）**

从 SOLIDER 作者原始代码重新开始，探索全新的 pose heatmap 利用方式。重点方向：
1. 姿态热图作为 attention prior（中间层注入）
2. 基于热图的 part feature learning
3. 热图引导的特征选择/加权
4. 人体结构约束的特征学习
5. Backbone 内部的 pose-conditioned attention

## Phase 2：模块实现与集成

### 核心原则

1. **插件式设计**：所有新模块在 `models/modules/` 下独立实现，通过 config 开关控制
2. **不破坏 baseline**：修改前先跑一遍 baseline 确认能正常训练和评估，记录 baseline 指标
3. **最小改动**：每次实验只改一个变量（一个模块 OR 一个 loss OR 一个策略）
4. **模块轻量化**：新增模块优先保持轻量，在实现前先估算额外开销
6. **服务创新点**：实验不是盲目试模块，而是围绕 `innovation_brainstorm.md` 中确定的主攻创新点来设计。每个实验都应该能回答一个关于创新点的具体问题（如："姿态可见性信息是否能有效指导特征加权？"）

### 姿态信息预处理

在开始模块实验之前，先完成数据预处理：

```
1. 选择一个姿态估计模型（推荐 HRNet-W48 或 DWPose，参考 KPR 和 PFD 的做法）
2. 对训练集和测试集的所有图片离线提取：
   - 17 个关键点坐标 + 置信度
   - 对应的姿态热图（可选，视模块需求）
3. 保存为 JSON 或 NPY 格式
4. 修改 dataloader 以加载姿态信息
```

### 实验命名规范

```
exp001_baseline              → 纯 baseline 结果
exp002_pose_keypoint_embed   → 加入关键点嵌入
exp003_pose_part_branch      → 加入姿态引导的 part 分支
exp004_pose_loss             → 加入姿态相关损失
exp005_combine_002_004       → 组合实验
...
```

### 代码提交规范

每次改动后：
```bash
git add -A
git commit -m "exp{NNN}: {简短描述改动内容}"
```

## Phase 3：实验执行与实时监控

### 监控协议

**sleep 间隔严格不超过 5 分钟**（300 秒）。推荐节奏：

- Epoch 1-5（前期关键期）：每 **2 分钟** 检查一次
- Epoch 6-30（收敛观察期）：每 **3 分钟** 检查一次
- Epoch 30+（稳定期）：每 **5 分钟** 检查一次

### 每次监控检查的标准动作

```bash
# 1. 查看最新日志

# 2. GPU 状态

# 3. 检查是否有报错
```

### 监控日志（追加写入 `experiments/exp{NNN}/monitor.md`）

```markdown
---
### [HH:MM:SS] 检查点 #{N}

**状态**: 🟢正常 / 🟡关注 / 🔴异常
**进度**: Epoch {X}/{Total} ({百分比}%)

| 指标 | 当前值 | 变化趋势 |
|------|--------|----------|
| Total Loss | 2.34 | ↓ 稳定下降 |
| ID Loss | 1.12 | ↓ |
| Triplet Loss | 0.65 | ↓ |
| Pose-xxx Loss | 0.57 | ↓ |
| LR | 3.5e-4 | — |
| GPU Mem | 8.2G/24G | — |
| GPU Util | 95% | — |

**观察**: {一句话描述当前状态}
**决策**: {继续 / 需要干预 / 原因}
```

### ⚠️ DataLoader Worker 进程识别

**重要**：PyTorch DataLoader 会 fork 出 `NUM_WORKERS` 个子进程（默认 8 个），这些进程在 `ps aux | grep train.py` 中也会显示为 `python train.py --config_file ...`。**千万不要将这些 worker 进程误认为是重复的训练进程并 kill 掉**——kill worker 会导致主训练进程 crash（`RuntimeError: DataLoader worker exited unexpectedly`）。

识别方法：
- 主训练进程：CPU 占用高（90%+），启动时间最早
- Worker 进程：CPU 占用较高但晚于主进程启动，数量等于 `NUM_WORKERS`
- 如需终止训练，只 kill 主进程 PID，worker 会自动退出

### 异常自动干预

| 触发条件 | 操作 |
|----------|------|
| loss 出现 NaN/Inf | 立即 kill 进程；回退到最近 checkpoint；将 LR 降为原来的 0.5 倍重启 |
| loss 突增超过 5 倍 | 连续观察 3 次检查（约 10-15 分钟），若持续则终止并记录 |
| OOM (CUDA out of memory) | 精简新增模块参数量或减小输入分辨率。**严禁修改 batch size** |
| 进程被 kill / 僵死 | 检查系统日志 `dmesg | tail`，调整资源后重启 |
| 精度长期停滞 | 连续 20 个 epoch mAP 无任何提升趋势再考虑终止当前实验，期间可尝试调整 LR 或 warm restart。终止后**先完成文档，再启动下一个实验** |
| mAP 持续下降 | 连续 10 个 epoch 下降再终止当前实验，短期波动属于正常现象不必紧张。终止后**先完成文档，再启动下一个实验** |

**关键原则：任何一个实验结束（无论成功或失败），先完成规则 0 中要求的全部文档，然后立即进入下一个实验。**

## Phase 4：结果分析与创新迭代

### 实验完成后（必须按顺序完成，不可跳过）

**步骤 1**：在 `experiments/exp{NNN}/monitor.md` 末尾追加最终结果和结论（3-5 行即可）。

**步骤 2**：更新 `experiments/results.md`，从 log 文件中复制精确数字，不要凭记忆。results.md 必须分区展示：
   - 纯模型结果（无后处理）
   - +NFC 结果（如适用）
   - +Re-ranking 结果（如适用）

**步骤 3**：更新 `experiments/decisions.md`（如有新决策）。

**步骤 4**：如果实验有重要发现，更新 `experiments/innovation_brainstorm.md`。

**以上全部完成后，才能启动下一个实验。**

### 迭代策略

```
Round 1: 探索性实验（快速试错，寻找有效方向）
  → 逐个测试从论文中拆出的姿态相关模块
  → 全部在 Occluded-Duke 上验证（少量 epoch 先看趋势，如 20-30 epoch）
  → 有效的模块标记为 ✅，无效标记为 ❌
  → 核心目的：不只是找到有效模块，更是通过实验结果锁定创新点方向

Round 2: 创新点验证（围绕确定的创新点做深度实验）
  → 基于 Round 1 的发现，设计针对创新点的消融实验
  → 证明创新点的每个组件都是必要的（消融实验是论文的核心证据）
  → 设计对比实验：与现有方法对比，证明我们方法的优势
  → 组合 Round 1 中 ✅ 的模块，验证组合效果

Round 3: 完善与补充实验
  → 超参数敏感性分析（论文常见的图表）
  → 可视化分析（attention map、特征分布 t-SNE、检索结果可视化）
  → 不同遮挡程度下的性能分析（体现方法对遮挡的鲁棒性）
  → 计算效率分析（FLOPs、参数量、推理速度对比）
```

### 论文素材积累

在实验过程中，持续积累论文写作所需的素材，存入 `experiments/paper_materials/`：

```
paper_materials/
├── figures/                  # 论文图表素材
│   ├── method_overview/      # 方法总览图的素材（模块关系、数据流）
│   ├── qualitative/          # 定性结果（检索结果、attention 可视化）
│   ├── tsne/                 # t-SNE 特征分布图
│   └── ablation_charts/      # 消融实验图表
├── tables/                   # 实验结果表格（可直接用于论文）
│   ├── main_results.md       # 主实验结果（与 SOTA 对比）
│   ├── ablation.md           # 消融实验
│   └── efficiency.md         # 计算效率对比
└── story.md                  # 论文故事线（持续更新）
```

**`story.md` 是核心文档**，从第一个实验开始就维护：

```markdown
# 论文故事线（持续更新）

## 暂定标题
{随着实验推进不断更新}

## Motivation（为什么做这个）
- 现有问题：...
- 现有方法的不足：...
- 我们的洞察：...

## 核心贡献（预计 3 点）
1. ...
2. ...
3. ...

## 方法概述
{随着实验推进逐步完善}

## 实验证据链
{每个关键实验结果如何支撑我们的 story}
- 实验 A 证明了 ...
- 消融实验 B 证明了 ...
- 可视化 C 直观展示了 ...

## 与 SOTA 对比的 narrative
{我们在哪些指标/场景上超过了谁，这说明了什么}

## 待补充的实验 / 待解决的问题
- [ ] ...
- [ ] ...
```

## 决策记录规范

所有重要决策追加到 `experiments/decisions.md`：

```markdown
### [{日期} {时间}] 决策 #{N}

**上下文**: {什么情况下做的决策}
**选项**:
  A. {方案 A 及预期}
  B. {方案 B 及预期}
**选择**: {A/B}
**理由**: {为什么}
**执行结果**: {后续补填}
```

### 红蓝队辩论制度（所有决策必须执行）

**每次做出决策之前**，必须启动两个 Opus 子代理进行红蓝队辩论：

1. **红队（Red Team）**：为方案 A 辩护。必须从技术可行性、论文价值、风险控制等角度论证为什么方案 A 是最优选择，并攻击其他方案的弱点。
2. **蓝队（Blue Team）**：为方案 B（或其他备选方案）辩护。同样从多维度论证其优势，并攻击红队方案的弱点。

**两个子代理并行启动**，各自独立分析。主 agent 收到两方观点后：
- 综合双方论点，在 `experiments/decisions.md` 中记录红蓝双方的核心论点
- 做出最终决策并说明理由（可以选择红方、蓝方、或综合两方的折中方案）

**子代理的 prompt 模板**：
```
你是 {红/蓝}队辩手。当前决策上下文：{上下文}。
你需要为 {方案 X} 辩护。

你必须：
1. 从技术可行性、创新性、论文价值、风险、实现成本等维度论证你的方案
2. 指出对方方案（{方案 Y}）的弱点和风险
3. 提供具体的技术论据（引用实验数据、代码结构、论文先例等）
4. 最后给出一个 1-10 的信心分数，说明你对自己方案的信心程度

注意：你的目标是尽可能强力地为你的方案辩护，即使你内心可能觉得另一方案更好。
这是辩论，不是中立分析。
```

**记录格式**（追加到 decisions.md 的决策条目中）：
```markdown
**红蓝队辩论**:
- 🔴 红队（方案 A）核心论点: {10-15 句话}，信心: {N}/10
- 🔵 蓝队（方案 B）核心论点: {10-15 句话}，信心: {N}/10
- 综合判断: {最终选择及理由}
```

## 全局规则

### ★ 文档记录铁律（最高优先级，必须严格遵守）

以下规则是**硬性要求**，违反即为严重失误：

a. **每次 sleep 结束后**: 必须立即更新 `experiments/exp{NNN}/monitor.md`，记录观察到的训练指标
b. **每个实验结束后**: 必须更新 `experiments/results.md`，添加该实验的最终结果行
c. **每次想到新方法/方向时**: 必须更新 `experiments/innovation_brainstorm.md`
d. **每次做出重要决策时**: 必须写入 `experiments/decisions.md`
e. **文档之间数据必须一致**: 同一个实验结果在不同文档中的数字必须完全相同
f. **数据来源必须是 log 文件**: 记录实验数据时，必须从 log 文件中读取精确数字，绝不能凭记忆填写

这四份文档是实验工作的核心产出，其重要性等同于代码和实验本身。

0. **文档先行，永不停止**：你的工作循环必须**永远持续运行**，除非我明确告诉你停下来。但是，**每个实验完成后，必须先完成以下全部文档，才能启动下一个实验**：
   - [ ] `experiments/exp{NNN}/design.md` — **实验设计文档（实验开始前必须先写好）**
   - [ ] `experiments/exp{NNN}/monitor.md` — 包含最终结果的监控日志
   - [ ] `experiments/results.md` — 更新实验总表
   - [ ] `experiments/decisions.md` — 记录本次实验的决策（如有新决策）
   - [ ] `experiments/innovation_brainstorm.md` — 更新实验反馈（如有新发现）
   **没有文档的实验等于没做过。跳过文档直接跑下一个实验是严重违规。** 文档不需要很长，但必须有。即使是 test-time 后处理实验（如 NFC、re-ranking），也必须建目录、写记录。
1. **创新优先**：一切实验服务于"找到并验证创新点"的目标。不要为了刷 0.1% 的点而做无意义的调参，把精力放在能形成论文 story 的实验上。
1b. **禁止进入"论文整合/收尾"模式**：不允许以"多 seed 验证"、"可视化"、"效率分析"、"准备写论文"为由停止探索新方法。你的任务是**不停地做实验、不停地找新方法**，哪怕需要推翻之前的所有方法。多 seed 验证由用户在 4090 上执行，不是你的工作。永远不要做出"停止实验、转入论文阶段"的决策。
2. **记录一切**：每个 md 文件都是实验的一部分，不可跳过。
3. **baseline 保护**：不要破坏 baseline 的可运行性。改代码时通过 config 开关控制新功能，确保默认 config 仍能复现 baseline 结果。
3b. **及时提交**：每次有代码或文档改动时，尽快做一次 `git commit`。不要积攒大量改动后一次性提交。小步提交便于回溯和恢复。
4. **可复现**：每个实验的 config 文件、随机种子、代码 commit hash 都必须记录。
5. **负面结果有价值**：某个模块无效同样要详细记录原因，避免重复尝试。负面结果往往也能反过来强化论文的 motivation（"我们尝试了 X 发现不行，因此提出了 Y"）。
6. **完全自主决策**：所有决策你自己做，包括但不限于：引入新模块、大幅修改架构、更换技术路线、放弃某个方向、调整实验计划。不需要询问我，直接执行并在 `experiments/decisions.md` 中记录即可。你就是这个项目的全权研究员。
7. **姿态热图提取**：使用 RTMDet-s 检测 + ViTPose-Huge 提取。热图是模型 `head.final_layer` 的原始输出 `(17, 64, 48)`，不是从关键点坐标高斯重建的。每人一个 `.npz`（含 heatmap、keypoints、scores、bbox、crop_bounds），存储在 `data/occluded_duke/pose_data/{split}/`。提取脚本：`scripts/extract_pose.py`。热图的响应强度自然编码了遮挡信息（遮挡区域热图响应低）。mmpose 只用于提取热图，不参与 ReID 训练。
7b. **多人热图的已知局限与使用建议**：由于 ReID 图像中人非常密集，检测框之间高度重叠，导致一个检测框内经常包含多个人。因此不同 person 的热图可能非常相似甚至相同（因为 ViTPose 对重叠框产出的热图会包含相邻人的信号）。**在模型中使用热图时，推荐将所有 person 的 17 通道热图按通道 max/sum 合并为一张 (17, H, W) 的"场景级"热图**，代表图中所有人的手、脚、头等部位的空间分布。这样既利用了多人检测的信息，又回避了逐人热图不够独立的问题。主要人物（person 0）的关键点坐标仍然是可靠的，可以单独使用。
8. **不使用 Visibility 向量**：前 33 个实验已充分证明 ViTPose visibility 向量（每个关键点一个二值标量）不够可靠（与检索性能相关性仅 0.237），且在 PCFC 框架下任何基于 visibility 的修改都会触发 alpha suppression。本轮实验完全不使用 visibility 向量，只使用原始姿态热图。
9. **论文意识**：在做任何实验之前，先问自己"这个实验的结果能放进论文的哪一部分？"如果答案是"不知道"，重新考虑实验设计。每一个实验都应该为论文的某个 section 提供素材（主实验表格 / 消融表格 / 可视化 / 效率分析 / motivation 验证）。
10. **语言要求**：所有记录文件（md 文档、监控日志、决策记录、分析报告等）一律使用**中文**撰写。代码注释可以使用英文。
11. **实验输出目录隔离**：每个实验必须使用独立的 `OUTPUT_DIR`，格式为 `./log/occluded_duke/{实验名}`（如 `./log/occluded_duke/exp000_baseline`）。在 yml config 或启动命令中显式指定，避免覆盖之前实验的 log、checkpoint 和评估结果。
12. **后台训练 + 定时监控**：启动训练时使用后台运行（`run_in_background` 或 `nohup`），然后用不超过 5 分钟的 sleep 间隔定期检查日志、GPU 状态和是否有报错。每次检查后在 `experiments/exp{NNN}/monitor.md` 中记录观察和决策（继续/干预/终止）。绝不阻塞等待训练完成。
13. **每次检查必须记录**：每次查看训练日志时，必须同步将观察到的信息（loss、accuracy、eval结果、异常情况等）追加写入对应的 `experiments/exp{NNN}/monitor.md`。不允许查看日志但不记录。
14. **数据准确性**：文档之间的数据必须一致。`results.md`、`monitor.md`、`story.md` 中引用的同一个实验结果数字必须完全相同。如果发现不一致，立即修正。引用数据时从 log 文件中复制精确数字，不要凭记忆填写。
15. **NFC / Re-ranking 等 test-time 后处理的论文报告规范**：NFC 和 Re-ranking 都是通用的 test-time 后处理方法，**不是我们的训练端创新**。在 `results.md` 和论文中必须清晰分区报告：(1) 无后处理的纯模型结果 (2) +NFC 结果 (3) +Re-ranking 结果。**不允许把 NFC/RR 的提升混入训练端贡献来计算"vs baseline"的增益**。模型本身的贡献 = 无后处理结果 vs baseline 无后处理结果。
16. **实验设计文档（design.md）**：每个实验在启动训练**之前**，必须先在 `experiments/exp{NNN}/design.md` 中写清楚该实验的完整设计。这份文档是实验的"立项书"，让我们随时能看懂每个实验在做什么、为什么做、期望什么结果。模板如下：

```markdown
# 实验 exp{NNN}: {实验名称}

## 动机
- 为什么要做这个实验？解决什么问题？
- 基于哪些前序实验的发现或论文的启发？

## 创新点 / 核心想法
- 本实验验证的核心假设是什么？（一句话）
- 与 baseline / 前序实验相比，改了什么？

## 技术方案
- 修改了哪些文件？新增了哪些模块？
- 数据流怎么走？（从输入到输出的简要描述）
- 关键超参数及其选择依据

## 预期结果
- 如果假设成立，预期 mAP/R1 变化方向和大致幅度
- 如果失败，最可能的原因是什么？

## 对照组
- Baseline 对照：exp{NNN} 的哪个结果
- 消融变量：本实验相对于对照组只改了哪一个变量

17. **实验审查制度（Agent Review）**：每个实验在启动训练**之前**，必须使用 Agent 工具启动一个 Opus 4.6 子代理对以下内容进行严格审查。审查代理必须是**极其严格的**，不放过任何细节。

    **审查代理的行为要求**：
    - **必须逐行阅读**所有新增/修改的代码文件，不能跳过任何函数或分支
    - **必须逐行对比** exp 配置文件与对照组配置文件，确认差异只有实验变量
    - **必须验证数据流**：从输入到输出，手动追踪 forward pass 的每一步，确认无遗漏
    - **必须检查梯度流**：确认 backward pass 中梯度是否按预期流动（特别是 stop_grad、detach 等操作）
    - **必须检查边界情况**：空输入、None 值、设备不匹配、dtype 不匹配、shape 不匹配等
    - **必须检查优化器行为**：新增参数是否被正确加入优化器？未使用参数是否会产生副作用（如 weight decay 对 unused params）？
    - **必须检查与已有实验的交互**：修改是否会影响任何已有实验的可复现性？默认值设置是否安全？
    - **发现任何可疑之处必须标记**，即使不确定是否是 bug。宁可误报，不可漏报

    **审查范围（缺一不可）**：
    a. `experiments/exp{NNN}/design.md` — 实验设计的合理性、单变量原则、假设是否清晰、预期结果是否合理
    b. 新增/修改的**所有**代码文件 — 实现是否与设计文档一致、是否有 bug、是否有潜在的 runtime error
    c. 配置文件 — 是否正确引用了新增的 config 选项、参数值是否匹配代码中的默认值和预期行为
    d. 文档与代码的一致性 — 参数数量、模块描述、数据流等是否准确
    e. `config/defaults.py` — 新增的默认值是否正确、是否会破坏已有实验
    f. 训练流程（processor） — 新增模块的 loss 计算、特征提取、评估逻辑是否正确
    g. 与前序实验的对照 — 确认消融变量的隔离性

    子代理**只负责审查和提出问题/修改建议，不直接修改代码**。所有代码修改由主 agent 根据审查意见执行。

    **审查必须通过才能启动训练。** 具体流程：
    1. 第一次审查 → 发现问题 → 主 agent 根据审查意见修改代码/文档
    2. 修改后必须再次启动子代理进行二次审查
    3. 重复此循环，直到子代理明确表示"审查通过，可以开始训练"
    4. **严禁在审查未通过的情况下启动训练**，即使自认为已修复所有问题
    5. 每轮审查的结论（通过/不通过 + 问题列表）记录在 `experiments/exp{NNN}/review.md` 中
    6. 审查报告必须包含：对每个审查维度的逐项结论、发现的所有问题（按严重程度分级：Critical/High/Medium/Low）、最终结论

