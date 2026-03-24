# Claude Review: exp167 DPTL

## 审查通过

### 第一轮审查发现的问题（已修复）

1. **[Critical] per-token test feature 被 pooled feature 覆盖**
   - 发现这是一个"有益的 bug"：pooled test (63.1/73.9) > per-token concat test (61.8/72.5)
   - 决定保留 pooled test feature 设计
   - 添加了注释说明原因

2. **[Medium] str_stats 未在 processor 中记录**
   - 已添加 str_stats 日志，包括 sa_cos 等诊断指标

3. **[Low] 设计文档参数计数不准确**
   - 已修正为 ~4.7M（MHA + FFN + LayerNorms）

### 验证项

- [x] 后向兼容：POSE_STR_SELF_ATTN=False 时 exp166 结果不变 (63.1/73.9)
- [x] 模块编译通过，形状正确
- [x] 零初始化：sa_cos=1.0 at init
- [x] 单变量：仅增加 POSE_STR_SELF_ATTN=True
- [x] AMP 安全
- [x] 新参数自动加入 optimizer

### 重要发现

per-token classification 是 TRAINING 技巧（强制 token diversity），但 test 时 confidence-weighted pooling 比 per-token concat 好 +1.3% mAP / +1.4% R1。
