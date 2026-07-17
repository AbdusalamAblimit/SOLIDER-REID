# LGPA 创新方向分析 (GPT-5.4 + Claude 联合评估)

## 核心发现

LGPA-Detach 本身 novelty 4.5/10, 不够 CCF-B。
需要更深层创新: **Visibility-Conditional Semantic Routing (VCSR)**

## 推荐方向: VCSR

**核心主张**: Occluded ReID 失败是因为固定 part 词汇表假设完整语义支持。
遮挡下, 模型应只实例化被可见证据支持的语义组。

**机制**:
1. 使用 canonical prompt bank (CLIP text prototypes)
2. 根据 pose visibility 决定哪些语义 slot 是活跃的
3. 仅将 image tokens route 到被支持的 slots
4. 匹配时只比较共同支持的语义

**与 prior art 的区分**:
- vs ProFD: ProFD 用固定 part prompts 提取 part features. VCSR 动态选择活跃 parts.
- vs PAFormer: PAFormer 用固定 pose tokens + visibility prediction. VCSR 根据 visibility 改变 routing.
- vs KPR: KPR 用 keypoint prompting 做 target disambiguation. VCSR 建模不完整语义支持.

## 5 个候选想法评估

| 想法 | 可行性 | 新颖性 | 故事性 | 复杂度 | 推荐 |
|------|--------|--------|--------|--------|------|
| 1. Visibility-Conditional Semantic Routing | 7/10 | 7/10 | 8/10 | medium | **首选** |
| 2. Cross-Instance CLIP Completion | 6/10 | 6/10 | 7/10 | high | 备选 |
| 3. Pose-Language Contrastive Pre-training | 5/10 | 8/10 | 6/10 | high | 太间接 |
| 4. Occlusion Narration | 4/10 | 6/10 | 5/10 | high | 已有类似 |
| 5. Visibility-Conditional Prototypes | 7/10 | 5/10 | 5/10 | low | 增量 |

## 下一步

实现 VCSR: 在 LGPA 基础上增加 visibility-conditional routing
- 当 leg 被遮挡时, 不生成 leg part feature, 不在匹配中使用 leg
- 匹配时用 common visible parts 而非 fixed concat
- 这与 MaxSim 测试时策略在训练端有机结合
