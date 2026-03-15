# exp064 PKE 审查报告

## 第一轮（Opus）
- **Critical**: concat(mu, log_sigma) + Euclidean ≠ MLS。Test feature 2304-d 与 equal_concat 不兼容
- **High**: log_sigma 缺少 min clamp
- **结论**: 不通过

## 修复
1. 改为 precision-weighted mu (mu/sigma)，output 恢复 768-d
2. 添加 log_sigma.clamp(min=-5.0, max=5.0)

## 第二轮（Opus）
- 维度验证：768-d 正确 ✅
- 精度权重数学正确（等价 Mahalanobis with diagonal cov）✅
- sigma.clamp(min=0.01) 防除零 ✅
- 初始化正确（sigma≈0.14，~7x 放大但 L2-norm 消解）✅
- 训练路径不受影响（classifier 用 raw mu）✅
- **结论**: ✅ 通过
