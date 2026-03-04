# Paper 10: Instruct-ReID
**来源**: CVPR 2024
**仓库**: https://github.com/hwz-zju/Instruct-ReID
**arXiv 摘要**: 提出指令驱动的多任务行人重识别框架，通过自然语言/图像指令统一处理标准ReID、换装ReID、跨模态ReID、文本到图像ReID等多种任务。

## 代码架构概览
- 核心文件：`reid/models/pass_transformer_joint.py` — 主模型定义
- 模型入口：`PASS_Transformer_DualAttn_joint` 类（用于联合训练所有任务）
- 辅助模型：`Transformer_local` 类（用于单任务场景如换装ReID）
- Backbone：ViT-Base (768-dim, patch_size=16, 12-layer)，使用 PASS 预训练权重
- 文本编码器：BERT（来自 ALBEF 预训练模型），用于处理文本指令
- 视觉辅助编码器：`visual_encoder_m`（ALBEF 的 ViT），用于处理图像指令（如衣物模板）
- 训练入口：`examples/train_joint.py`
- 损失函数目录：`reid/loss/`（包含 adaptive_triplet, crossentropy, transloss, dual_causality_loss, adv_loss）

### 文件结构
```
reid/
├── models/
│   ├── pass_transformer_joint.py    # 核心：PASS_Transformer_DualAttn_joint
│   ├── backbone/
│   │   ├── pass_vit.py              # PASS ViT backbone
│   │   ├── vit_albef.py             # ALBEF ViT（作为指令编码器）
│   │   └── vit.py / vit_ri.py       # 其他 ViT 变体
│   ├── xbert.py                     # BERT 文本编码器
│   ├── tokenization_bert.py         # BERT tokenizer
│   └── layers/
│       ├── metric.py                # 度量学习层
│       └── gem.py                   # Generalized Mean Pooling
├── loss/
│   ├── adaptive_triplet.py          # 自适应三元组损失
│   ├── crossentropy.py              # 标签平滑交叉熵
│   ├── transloss.py                 # Translation 损失（TransLoss + SoftTripletLoss）
│   ├── dual_causality_loss.py       # 双因果损失
│   └── adv_loss.py                  # 对抗损失（换装场景）
├── datasets/
│   ├── data_builder_sc_mnt.py       # 标准 ReID 数据构建
│   ├── data_builder_cc.py           # 换装 ReID 数据
│   ├── data_builder_t2i.py          # 文本到图像 ReID 数据
│   ├── data_builder_attr.py         # 属性检索 数据
│   ├── data_builder_cross.py        # 跨模态 ReID 数据
│   └── data_builder_ctcc.py         # 换装跨相机 数据
└── evaluation/                       # 各任务评估器
```

## 可拆解模块清单

### 模块 A: Dual Attention Fusion（双注意力融合）
- 文件位置：`reid/models/pass_transformer_joint.py` L380-L389
- 功能：将视觉特征与指令特征（文本或图像）进行交叉注意力融合。核心思想是交换 CLS token：将视觉 CLS token 拼接指令 patch tokens，将指令 CLS token 拼接视觉 patch tokens，分别通过融合层获得两路互补特征。
- 输入：`bio_feats` [B, N, 768]（视觉特征），`clot_feats` [B, M, 768]（指令特征）
- 输出：`bio_fusion` [B, M, 768]（视觉增强的指令特征），`clot_fusion` [B, N, 768]（指令增强的视觉特征）
- 依赖：使用 ViT Block 作为融合层（从 backbone 的最后几层 deepcopy）
- **移植到我们框架的可行性**：中
- **额外显存开销估算**：~0.3G（fusion 层只有 1-2 个 ViT Block，参数量小）
- **移植方案**：将姿态关键点特征（或 part token）作为"指令"输入，与 Swin-Tiny 的 patch 特征做双注意力融合。需要将 Swin-Tiny 输出从窗口注意力格式转换为序列格式。但 Dual Attention 的设计初衷是融合两种不同模态（视觉+文本/衣物），对于姿态引导 ReID 的适用性一般。

### 模块 B: MaskModule（前景遮罩模块）
- 文件位置：`reid/models/pass_transformer_joint.py` L86-L127
- 功能：通过轻量卷积预测 K 个空间掩码（每个掩码对应一个语义区域），然后取最大池化得到前景掩码，加权视觉特征以抑制背景。
- 输入：feature map [B, C, H, W]（如 [B, 768, 16, 8]）
- 输出：masked_feat [B, C, H, W], mask [B, 1, H, W]
- 依赖：无外部依赖
- **移植到我们框架的可行性**：高
- **额外显存开销估算**：<0.1G（3层1x1卷积，参数极少）
- **移植方案**：直接在 Swin-Tiny 最后一个 stage 输出上应用。将 Swin-Tiny 的 [B, H*W, 768] reshape 为 [B, 768, H, W]，通过 MaskModule 得到前景掩码，用于抑制遮挡区域的特征。这是一个极其轻量的前景分割模块，可以作为辅助手段增强我们 PAMS 的 part 可见性判断。

### 模块 C: CrossAttentionLayer（交叉注意力层）
- 文件位置：`reid/models/pass_transformer_joint.py` L129-L215
- 功能：标准的交叉注意力层（query 来自一个分支，key/value 来自另一个分支），支持 DeepNorm 初始化和 deep prompt 扩展。
- 输入：tgt [N, B, C]（query），memory [M, B, C]（key/value）
- 输出：融合后的 tgt [N, B, C]
- 依赖：无
- **移植到我们框架的可行性**：高
- **额外显存开销估算**：~0.1G（单层 MultiheadAttention）
- **移植方案**：可用于将姿态关键点 embedding 与视觉特征做交叉注意力。例如，17 个关键点作为 query，Swin-Tiny patch 特征作为 key/value，实现姿态引导的特征聚合。这与 KPR 的思路类似，但实现更简洁。

### 模块 D: Momentum-based Image-Text Matching (ITM + MLM + MRTD)
- 文件位置：`reid/models/pass_transformer_joint.py` L392-L566
- 功能：ALBEF 风格的视觉-语言预训练任务集合：
  - Contrastive Learning (CL)：图像-文本对比学习（使用 momentum queue）
  - Paired Image-Text Matching (PITM)：正负样本匹配
  - Masked Language Modeling (MLM)：遮蔽语言建模
  - Momentum-based Replaced Token Detection (MRTD)：动量替换 token 检测
- **移植到我们框架的可行性**：低
- **额外显存开销估算**：>2G（需要额外的文本编码器 + 动量模型 + 大队列）
- **移植方案**：不适合直接移植，这些是多模态预训练任务，我们的 baseline 是纯视觉方案。但其中对比学习的思路（用 queue 扩充负样本）可借鉴到 part feature 的对比学习中。

### 模块 E: Translation Loss（TransLoss）
- 文件位置：`reid/loss/transloss.py` L41-L63
- 功能：基于翻译关系的三元组损失。给定 anchor、正样本、负样本的 embedding，计算 euclidean 距离并用 MarginRankingLoss 约束。区别于标准 triplet 的点在于正负样本不是通过标签挖掘，而是通过模态翻译关系确定。
- 输入：emb [B, D], emb_pos [B, D], emb_neg [B, D]
- 输出：scalar loss
- 依赖：无
- **移植到我们框架的可行性**：中
- **额外显存开销估算**：0（纯计算，无额外参数）
- **移植方案**：可用于约束 global feature 和 part feature 之间的一致性。例如让 part feature 聚合后与 global feature 保持翻译关系。

### 模块 F: Dual Causality Loss
- 文件位置：`reid/loss/dual_causality_loss.py` L11-L64
- 功能：双因果损失，用于解耦身份相关特征（f）与正向因果特征（fp，如身份+姿态）和负向因果特征（fm，如背景/遮挡）。通过约束 fp 的正样本距离小于 f，fm 的正样本距离大于 f，实现因果解耦。使用 softplus 替代 hinge 使梯度更平滑。
- 输入：s_dual = (f, fp, fm) 三元组，label [B]
- 输出：scalar loss
- 依赖：无
- **移植到我们框架的可行性**：中
- **额外显存开销估算**：0（纯计算）
- **移植方案**：可用于解耦遮挡特征。将 global feature 作为 f，前景 part 聚合特征作为 fp，被遮挡部分特征作为 fm，用 DualCausalityLoss 约束前景特征比 global 更具判别力，遮挡特征判别力更弱。这与我们 PAMS 的 visibility score 理念一致，但从损失函数角度提供额外约束。

## 损失函数
整个框架的损失组合非常丰富，取决于任务类型：

1. **标准 ReID (sc) 任务**：CE（标签平滑）+ Triplet + TransLoss + DualCausalityLoss
2. **文本到图像 (t2i) 任务**：CL（对比学习）+ PITM + MLM + MRTD
3. **换装 (cc) 任务**：CE + Triplet + 对抗损失（ClothesBasedAdversarialLoss）
4. **跨模态 (cross) 任务**：上述组合

对我们有用的：
- **CE + Triplet**：已有
- **TransLoss**：可选用于 part-global 一致性
- **DualCausalityLoss**：可选用于前景-背景解耦

## 训练 Tricks
- **Iteration-based 训练**：24000 iterations（而非 epoch-based），适合多数据集联合训练
- **WarmupMultiStepLR / WarmupCosineLR**：warmup 1000 步，milestone 在 [7000, 14000]
- **AdamW 优化器**：lr=1e-3, weight_decay=5e-4
- **混合精度训练（fp16）**：可选开启
- **多任务联合训练**：通过 data_config.yaml 配置不同任务的数据分配和 GPU 映射
- **Momentum 队列**：大小 65536，momentum=0.995，用于对比学习的负样本扩展
- **PASS ViT 预训练权重**：人体解析（PASS）预训练的 ViT-Base，自带对人体结构的理解

## 对我们框架的改进建议

1. **MaskModule 前景掩码**（优先级：高）：
   - 在 Swin-Tiny 输出上加一个极轻量的 MaskModule（3层1x1卷积），预测前景掩码
   - 用前景掩码加权 part feature，替代或辅助 PAMS 的 visibility score
   - 优势：端到端学习前景掩码，不需要额外的姿态/分割模型；显存开销 <0.1G
   - 风险：无显式监督信号时，掩码质量可能不够好，可配合 parsing loss 使用

2. **CrossAttentionLayer 用于姿态-视觉融合**（优先级：中）：
   - 用关键点 embedding 作为 query，Swin-Tiny patch 特征作为 key/value
   - 实现姿态引导的 part feature 聚合，类似 KPR 但更轻量
   - 显存开销：~0.1G

3. **DualCausalityLoss 用于前景-遮挡解耦**（优先级：中低）：
   - 将 visible part 聚合特征作为 fp，occluded part 特征作为 fm
   - 额外损失约束，增强 visibility-aware 特征学习
   - 显存开销：0

4. **不建议移植的模块**：
   - 整体 ALBEF 多模态架构（太重，需要 BERT + momentum model，显存 >2G）
   - 多任务联合训练框架（我们只关注 Occluded-Duke，无需多任务）
   - PASS ViT backbone（我们已锁定 Swin-Tiny，不可更换）
