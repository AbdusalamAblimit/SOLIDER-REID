# Paper 14: QPM - Quality-aware Part Models for Occluded Person Re-Identification
**来源**: 论文摘要阅读  
**论文**: https://arxiv.org/search/?query=Quality-aware+Part+Models+for+Occluded+Person+Re-identification&searchtype=all&source=header  
**代码**: 暂未找到可确认的官方实现  
**阅读日期**: 2026-03-13

## 阅读边界
- 当前只基于论文摘要做方向判断，未拿到可确认的官方代码。
- 因此它更适合作为“创新边界”参考，而不是实现模板。

## 这篇工作真正解决什么
- QPM 关注的是 **part quality / common non-occluded regions**。
- 论文不是简单做 part pool，而是显式提出：
  - 先估计各 part 的质量
  - 再从 query-gallery 的共同无遮挡区域构造 pair-specific global feature

## 对我们最重要的启发
1. **质量估计 / adaptive weighting 本身并不新。**
2. **pair-specific common visible reasoning 也不是空白地带。**
3. 因此如果我们只是做：
   - learnable fusion
   - adaptive part weighting
   - quality score 重加权
   很难形成站得住的主创新。

## 对当前代码线的约束
- `AFF` 这种“再学一个融合权重”的思路，更像是和 QPM/PAN/RGANet 同类的工程变体。
- 如果要继续做，必须把问题重新定义得更具体，例如：
  - 当前 `PSG+GCN` branch 的结构信息为什么在检索时被浪费？
  - 为什么 keypoint-level common support 比已有 part-quality 路线更适合我们的代码线？

## 当前判断
- QPM 不是给我们提供了一个要复刻的模块，而是提醒我们：
  **主线创新不能退化成“给 global/part 再配一个自适应权重”。**
