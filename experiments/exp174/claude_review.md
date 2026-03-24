# Claude Broad Review: exp174 SupCon (Opus 4.6)

## 审查通过

### 关键验证
- SupCon 公式正确（Khosla et al. SupCon-out variant）
- 数值稳定性：max-subtraction + eps 防止 overflow/underflow
- AMP 安全：PyTorch autocast 将 log/exp 提升到 fp32
- SupCon 用 features（feat[1:]），不是 logits（score[1:]）
- Global CE 保留，仅 per-token CE 被替换
- 双重 L2 normalize 是幂等的，无害
- temperature=0.07 是标准值
- 单变量：仅增加 POSE_STR_SUPCON=True + TEMP=0.07
- 后向兼容：默认 False
