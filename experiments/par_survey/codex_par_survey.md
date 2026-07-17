Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019f0c74-9dbf-7fa2-8ebe-7fd82714d9f0
--------
user
调研行人属性识别(Pedestrian Attribute Recognition, PAR)近三年(2023-2026)顶会现状, 给我清晰全景。
(1)主流基准(PETA / RAP / RAPv2 / PA100K / Market-1501-Attribute / UPAR)上近三年 SOTA 是哪些方法? 哪家最强? 给 mA / Accuracy / F1 数字。
(2)★最新范式: CLIP/VLM-based PAR、prompt/text-driven、LLM-MLLM-based、vision-language alignment、masked image modeling、diffusion、pure-Transformer 各自代表作 + 哪个范式最新最有潜力(2024-2026)。
(3)近三年 CVPR/ICCV/ECCV/NeurIPS/AAAI/IJCAI/ACMMM 上 PAR 代表论文, 列 6-10 篇最重要的(标题+会议+年份+一句核心机制)。
(4)PAR 当前开放问题/痛点(长尾属性 / zero-shot 新属性 / 跨域 / 属性间关系建模 / 细粒度 / 遮挡 / 标注噪声)。
(5)哪个方向/范式最值得跟进——有顶会潜力且还没卷死。
web search 查最新, 务实中文别套话别捧场。
codex
我先查近三年的论文、leaderboard 和代码仓库，重点核对每个数据集的指标口径。PAR 里同名指标很容易混用，先把可比口径厘清再给结论。
web search: 
web search: 2024 pedestrian attribute recognition PETA RAP RAPv2 PA100K SOTA mA Acc F1 CLIP
web search: 
web search: OpenPAR leaderboard PETA PA100K RAP RAPv2 Market1501 attribute results mA Acc F1
web search: 
web search: https://arxiv.org/abs/2408.09720
web search: 
web search: 'PETA' in https://arxiv.org/abs/2408.09720
web search: 
web search: https://arxiv.org/abs/2408.09720
web search: 
web search: https://arxiv.org/abs/2408.09720
web search: 
web search: https://arxiv.org/pdf/2408.09720
web search: 
web search: 'Table 3' in https://arxiv.org/pdf/2408.09720
web search: 
web search: https://arxiv.org/pdf/2408.09720
web search: 
web search: https://arxiv.org/pdf/2312.10692
web search: 
web search: https://arxiv.org/pdf/2408.09720
web search: 
web search: https://arxiv.org/pdf/2209.02522
web search: 
web search: 'Table 3' in https://arxiv.org/pdf/2209.02522
web search: 
web search: https://arxiv.org/pdf/2209.02522
web search: 
web search: Market-1501-Attribute pedestrian attribute recognition mA F1 Accuracy SOTA 2024
web search: 
web search: site:openaccess.thecvf.com pedestrian attribute recognition CVPR 2023 2024 2025
web search: 
web search: "pedestrian attribute recognition" "CVPR 2024"
web search: 
web search: https://github.com/wangxiao5791509/Pedestrian-Attribute-Recognition-Paper-List
web search: 
web search: https://github.com/wangxiao5791509/Pedestrian-Attribute-Recognition-Paper-List
web search: 
web search: https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Enhanced_Visual-Semantic_Interaction_with_Tailored_Prompts_for_Pedestrian_Attribute_Recognition_CVPR_2025_paper.html
codex
截至 2026-06-28，我的判断是：PAR 近三年已经从“CNN/attention/part/graph 小模块”转到 **CLIP/VLM prompt、LLM/MLLM 辅助、长尾重采样、跨域/开放属性 benchmark**。但榜单很乱，很多论文只报 `mA/F1`，不报 `Acc`；所以不能简单说“某一家全胜”。

**1. 主流基准近三年 SOTA**
`Acc` 指 PAR 常用的 instance-level accuracy；`-` 表示论文未报。

| 数据集 | 当前最强/最值得引用 | mA / Acc / F1 | 备注 |
|---|---:|---:|---|
| PETA | **VLM-PAR**, arXiv 2025 | **93.52 / - / 92.64** | 只报 mA/F1。若要求完整五指标，**LLM-PAR** 是 `92.25 / 84.59 / 90.39` |
| RAP/RAPv1 | **LLM-PAR** | `87.80 / 71.86 / 82.64` | mA 上 OAGCN `87.83` 略高，但 Acc/F1 低于 LLM-PAR |
| RAPv2 | **PromptPAR** | `83.14 / 69.62 / 81.00` | 我查到的公开近年完整五指标里仍是强基准；LLM-PAR 未报 RAPv2 |
| PA100K | **VLM-PAR** 按 mA；**AttriVision** 按 F1 | VLM-PAR `92.88 / - / 92.32`; AttriVision `89.80 / - / 93.10` | 完整五指标里 LLM-PAR 是 `91.09 / 84.12 / 90.41` |
| Market-1501-Attribute | **VLM-PAR** 按 mA；**AttriVision** 按 F1 | VLM-PAR `85.38 / - / 79.17`; AttriVision `83.80 / - / 88.80` | VLM-PAR 明显偏 mA，F1 反而差，说明阈值/正例校准有问题 |
| UPAR | **UPAR ConvNeXt-B + tricks** | `85.9 / - / 90.2` | UPAR 原论文/挑战体系更看重跨域和 attribute retrieval；公开表主要报 mA/F1 |

务实结论：  
**按 closed-set mA，VLM-PAR 最强；按完整指标和可复现实验，LLM-PAR/OpenPAR 线更稳；按顶会新范式，CVPR 2025 EVSITP 是最值得看的 prompt-VLM PAR。** VLM-PAR 很强但目前是 arXiv，且 Market F1 掉得很厉害，不宜无脑当最终 SOTA。

**2. 最新范式**
| 范式 | 代表作 | 核心机制 | 判断 |
|---|---|---|---|
| CLIP/VLM-based PAR | PromptPAR, AttriVision, AAPAR, VLM-PAR | 属性短语扩成文本 prompt，用 CLIP/SigLIP 图文特征做融合或对齐；PromptPAR 只调少量 prompt/classifier 参数 | 已成主流，但“再加 prompt”已经卷了 |
| Prompt/text-driven | **EVSITP, CVPR 2025** | 图像条件 dual prompt 初始化、prompt 增强正则、双模态互交互 BMIM，并新标 Celeb-PAR | 2025 最核心顶会 PAR 方向 |
| LLM/MLLM-based | **LLM-PAR / MSP60K**, AAAI 2025/arXiv 2024 | EVA-ViT-G + MEQ/Q-Former + Vicuna/OPT LoRA，生成属性描述并与分类 logits ensemble | 有潜力，但推理重、幻觉和泄漏要严控 |
| Vision-language alignment | CLEAR, ViTA-PAR, AAPAR | 显式对齐 visual token 与 attribute text/token，服务 PAR 或 attribute retrieval | 比单纯分类更有论文空间 |
| Masked image modeling | HAP, NeurIPS 2023 | human-structure-aware MIM 预训练，迁移到 human-centric/PAR | 更像 foundation pretrain，不是 PAR 专属创新 |
| Diffusion | 2025 AVSS 两篇 synthetic augmentation | 用 image-to-image 或 prompt-driven diffusion 合成长尾/zero-shot 属性样本 | 数据方向有潜力，但单独做很难顶会 |
| Pure Transformer | PARFormer, DFDT, DRFormer | Swin/ViT 做多标签分类、局部/多尺度 token 建模 | 已经成熟，不建议作为新主线 |
| 长尾/重采样 | **FRDL, ICML 2024** | label-balanced feature re-sampling + Bayesian feature augmentation，解决多标签共现导致的采样偏置 | 很务实，适合和 VLM/开放属性结合 |

**3. 近三年重要论文**
1. **Enhanced Visual-Semantic Interaction with Tailored Prompts for Pedestrian Attribute Recognition**, CVPR 2025：动态图像条件 prompt + 双模态互交互，当前最像“正统顶会 PAR”的工作。  
2. **Pedestrian Attribute Recognition as Label-balanced Multi-label Learning**, ICML 2024：把 PAR 重新定义为长尾多标签问题，用 FRDL/Bayesian feature augmentation 处理共现和稀有属性。  
3. **Pedestrian Attribute Recognition: A New Benchmark Dataset and A Large Language Model Augmented Framework**, AAAI 2025/arXiv 2024：MSP60K + LLM-PAR，强调跨域、退化、MLLM 辅助。  
4. **Selective and Orthogonal Feature Activation for Pedestrian Attribute Recognition**, AAAI 2024：SOFA，用选择性激活和正交约束减少属性干扰。  
5. **HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception**, NeurIPS 2023：人体结构感知 MIM，PAR 可作为下游任务。  
6. **Attribute-Guided Pedestrian Retrieval**, CVPR 2024：不是纯 PAR，但把属性作为 ReID/retrieval 内部变化建模信号，和 ReID 很相关。  
7. **Joint Implicit and Explicit Language Learning for Pedestrian Attribute Recognition**, AAAI 2026：显式/隐式语言学习结合，是 2026 prompt-language PAR 延续线。  
8. **PromptPAR**, TCSVT 2024/arXiv 2023：CLIP prompt + multimodal transformer，是后续 VLM-PAR 的关键起点。

**4. 当前痛点**
长尾属性仍是最大硬伤：PETA/RAP 大量属性频率极低，FRDL 明确指出 PETA 约 66%、RAP 约 57% 属性频率低于 0.1，低频属性直接拖 mA。  
跨域评估不够真实：UPAR 明确指出传统随机 split 无法衡量 OOD，才统一 PA100K/PETA/RAPv2/Market 做跨域协议。  
开放属性/zero-shot 还没解决：Prompt/VLM 能做 zero-shot，但属性定义、阈值、负样本、同义词、组合属性都不稳。  
MLLM 有幻觉：LLM-PAR 自己也展示 MiniGPT-4 会生成不存在的描述，必须做 grounding/校准。  
细粒度和遮挡仍难：眼镜、包、鞋、颜色、短袖/长袖这类局部属性受低分辨率、裁剪、遮挡、姿态影响大。  
标注噪声和伦理问题被低估：颜色、肤色、文化服饰、隐私属性容易混淆，UPAR 也专门讨论了潜在偏见风险。

**5. 最值得跟进的方向**
最值得做的是：**open-vocabulary / new-attribute PAR + VLM grounding + 长尾/跨域校准**。不要再做普通 closed-set prompt 调参。

具体说，顶会潜力更大的问题是：给定自然语言新属性或组合属性，模型能不能在未标注/少标注场景下可靠识别，并输出可校准置信度和可解释证据区域。技术上可以把 EVSITP/PromptPAR 的 VLM 对齐、FRDL 的长尾重采样、MSP60K/UPAR 的跨域协议结合起来，但贡献点应放在 **开放属性定义、少样本/零样本校准、区域 grounding、跨域鲁棒性**，而不是再堆一个 attention 模块。
tokens used
144,289
截至 2026-06-28，我的判断是：PAR 近三年已经从“CNN/attention/part/graph 小模块”转到 **CLIP/VLM prompt、LLM/MLLM 辅助、长尾重采样、跨域/开放属性 benchmark**。但榜单很乱，很多论文只报 `mA/F1`，不报 `Acc`；所以不能简单说“某一家全胜”。

**1. 主流基准近三年 SOTA**
`Acc` 指 PAR 常用的 instance-level accuracy；`-` 表示论文未报。

| 数据集 | 当前最强/最值得引用 | mA / Acc / F1 | 备注 |
|---|---:|---:|---|
| PETA | **VLM-PAR**, arXiv 2025 | **93.52 / - / 92.64** | 只报 mA/F1。若要求完整五指标，**LLM-PAR** 是 `92.25 / 84.59 / 90.39` |
| RAP/RAPv1 | **LLM-PAR** | `87.80 / 71.86 / 82.64` | mA 上 OAGCN `87.83` 略高，但 Acc/F1 低于 LLM-PAR |
| RAPv2 | **PromptPAR** | `83.14 / 69.62 / 81.00` | 我查到的公开近年完整五指标里仍是强基准；LLM-PAR 未报 RAPv2 |
| PA100K | **VLM-PAR** 按 mA；**AttriVision** 按 F1 | VLM-PAR `92.88 / - / 92.32`; AttriVision `89.80 / - / 93.10` | 完整五指标里 LLM-PAR 是 `91.09 / 84.12 / 90.41` |
| Market-1501-Attribute | **VLM-PAR** 按 mA；**AttriVision** 按 F1 | VLM-PAR `85.38 / - / 79.17`; AttriVision `83.80 / - / 88.80` | VLM-PAR 明显偏 mA，F1 反而差，说明阈值/正例校准有问题 |
| UPAR | **UPAR ConvNeXt-B + tricks** | `85.9 / - / 90.2` | UPAR 原论文/挑战体系更看重跨域和 attribute retrieval；公开表主要报 mA/F1 |

务实结论：  
**按 closed-set mA，VLM-PAR 最强；按完整指标和可复现实验，LLM-PAR/OpenPAR 线更稳；按顶会新范式，CVPR 2025 EVSITP 是最值得看的 prompt-VLM PAR。** VLM-PAR 很强但目前是 arXiv，且 Market F1 掉得很厉害，不宜无脑当最终 SOTA。

**2. 最新范式**
| 范式 | 代表作 | 核心机制 | 判断 |
|---|---|---|---|
| CLIP/VLM-based PAR | PromptPAR, AttriVision, AAPAR, VLM-PAR | 属性短语扩成文本 prompt，用 CLIP/SigLIP 图文特征做融合或对齐；PromptPAR 只调少量 prompt/classifier 参数 | 已成主流，但“再加 prompt”已经卷了 |
| Prompt/text-driven | **EVSITP, CVPR 2025** | 图像条件 dual prompt 初始化、prompt 增强正则、双模态互交互 BMIM，并新标 Celeb-PAR | 2025 最核心顶会 PAR 方向 |
| LLM/MLLM-based | **LLM-PAR / MSP60K**, AAAI 2025/arXiv 2024 | EVA-ViT-G + MEQ/Q-Former + Vicuna/OPT LoRA，生成属性描述并与分类 logits ensemble | 有潜力，但推理重、幻觉和泄漏要严控 |
| Vision-language alignment | CLEAR, ViTA-PAR, AAPAR | 显式对齐 visual token 与 attribute text/token，服务 PAR 或 attribute retrieval | 比单纯分类更有论文空间 |
| Masked image modeling | HAP, NeurIPS 2023 | human-structure-aware MIM 预训练，迁移到 human-centric/PAR | 更像 foundation pretrain，不是 PAR 专属创新 |
| Diffusion | 2025 AVSS 两篇 synthetic augmentation | 用 image-to-image 或 prompt-driven diffusion 合成长尾/zero-shot 属性样本 | 数据方向有潜力，但单独做很难顶会 |
| Pure Transformer | PARFormer, DFDT, DRFormer | Swin/ViT 做多标签分类、局部/多尺度 token 建模 | 已经成熟，不建议作为新主线 |
| 长尾/重采样 | **FRDL, ICML 2024** | label-balanced feature re-sampling + Bayesian feature augmentation，解决多标签共现导致的采样偏置 | 很务实，适合和 VLM/开放属性结合 |

**3. 近三年重要论文**
1. **Enhanced Visual-Semantic Interaction with Tailored Prompts for Pedestrian Attribute Recognition**, CVPR 2025：动态图像条件 prompt + 双模态互交互，当前最像“正统顶会 PAR”的工作。  
2. **Pedestrian Attribute Recognition as Label-balanced Multi-label Learning**, ICML 2024：把 PAR 重新定义为长尾多标签问题，用 FRDL/Bayesian feature augmentation 处理共现和稀有属性。  
3. **Pedestrian Attribute Recognition: A New Benchmark Dataset and A Large Language Model Augmented Framework**, AAAI 2025/arXiv 2024：MSP60K + LLM-PAR，强调跨域、退化、MLLM 辅助。  
4. **Selective and Orthogonal Feature Activation for Pedestrian Attribute Recognition**, AAAI 2024：SOFA，用选择性激活和正交约束减少属性干扰。  
5. **HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception**, NeurIPS 2023：人体结构感知 MIM，PAR 可作为下游任务。  
6. **Attribute-Guided Pedestrian Retrieval**, CVPR 2024：不是纯 PAR，但把属性作为 ReID/retrieval 内部变化建模信号，和 ReID 很相关。  
7. **Joint Implicit and Explicit Language Learning for Pedestrian Attribute Recognition**, AAAI 2026：显式/隐式语言学习结合，是 2026 prompt-language PAR 延续线。  
8. **PromptPAR**, TCSVT 2024/arXiv 2023：CLIP prompt + multimodal transformer，是后续 VLM-PAR 的关键起点。

**4. 当前痛点**
长尾属性仍是最大硬伤：PETA/RAP 大量属性频率极低，FRDL 明确指出 PETA 约 66%、RAP 约 57% 属性频率低于 0.1，低频属性直接拖 mA。  
跨域评估不够真实：UPAR 明确指出传统随机 split 无法衡量 OOD，才统一 PA100K/PETA/RAPv2/Market 做跨域协议。  
开放属性/zero-shot 还没解决：Prompt/VLM 能做 zero-shot，但属性定义、阈值、负样本、同义词、组合属性都不稳。  
MLLM 有幻觉：LLM-PAR 自己也展示 MiniGPT-4 会生成不存在的描述，必须做 grounding/校准。  
细粒度和遮挡仍难：眼镜、包、鞋、颜色、短袖/长袖这类局部属性受低分辨率、裁剪、遮挡、姿态影响大。  
标注噪声和伦理问题被低估：颜色、肤色、文化服饰、隐私属性容易混淆，UPAR 也专门讨论了潜在偏见风险。

**5. 最值得跟进的方向**
最值得做的是：**open-vocabulary / new-attribute PAR + VLM grounding + 长尾/跨域校准**。不要再做普通 closed-set prompt 调参。

具体说，顶会潜力更大的问题是：给定自然语言新属性或组合属性，模型能不能在未标注/少标注场景下可靠识别，并输出可校准置信度和可解释证据区域。技术上可以把 EVSITP/PromptPAR 的 VLM 对齐、FRDL 的长尾重采样、MSP60K/UPAR 的跨域协议结合起来，但贡献点应放在 **开放属性定义、少样本/零样本校准、区域 grounding、跨域鲁棒性**，而不是再堆一个 attention 模块。
