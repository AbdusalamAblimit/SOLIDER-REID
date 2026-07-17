# Paper 13: FRT - Learning Feature Recovery Transformer for Occluded Person Re-Identification
**来源**: IEEE TIP（官方 README 未给出完整代码）  
**论文**: https://arxiv.org/search/?query=Learning+Feature+Recovery+Transformer+for+Occluded+Person+Re-identification&searchtype=all&source=header  
**代码**: https://github.com/xbq1994/Feature-Recovery-Transformer  
**阅读日期**: 2026-03-13

## 阅读边界
- 官方仓库当前只有一句 “code will be uploaded soon”，**没有可审查的真实实现**。
- 因此下面结论主要来自论文摘要与官方 README，可作为方向判断，**不能当作代码级复现证据**。

## 这篇工作真正解决什么
- FRT 把 occluded ReID 定义成 **feature recovery** 问题：
  遮挡 query 的特征不完整，应该借助 gallery 中更完整的样本恢复/补全。
- 这不是普通 re-ranking，而是显式地做 **retrieval-time pairwise / gallery-assisted reasoning**。

## 从摘要能确认的机制
1. 第一步先做 visibility-guided graph matching  
   目的是先找到与当前样本最相关、且可提供补全部位证据的 gallery 邻居。
2. 第二步用 Feature Recovery Transformer  
   用邻居的完整信息去恢复当前 occluded feature。
3. 最终目标不是重新训练一个更强 backbone，而是改变遮挡样本在检索阶段的可比性。

## 对当前代码线的启发
1. **retrieval-time reasoning 可以是主问题定义，不只是 test trick。**
2. 如果 branch 已经学到了结构化局部信息，那么真正值得做的是：
   - 保留这些局部特征到检索阶段
   - 再基于 query-gallery 共同证据做恢复/重评分
3. 这和我们现在的 `equal_concat` 形成鲜明对比：
   当前做法在距离计算前已经把 branch 信息压成了单一向量。

## 对我们的风险提示
- FRT 属于更重的检索时推理路线，工程复杂度高，且当前官方代码缺失。
- 如果我们借鉴它，最好先做 **轻量诊断版**：
  先验证“共同可见关键点匹配”是否有效，再考虑更重的 feature recovery。

## 当前判断
- FRT 支持一个重要方向判断：**下一步更值得做 retrieval-time common-support reasoning，而不是继续在 branch 内部调 MLP/gate。**
