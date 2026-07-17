# exp063 PTD 审查报告

## 第一轮审查（Opus 4.6）
- **C1 (Critical)**: DecoderLayer 无 per-layer Q/K/V projections → 修复
- **C2 (Critical)**: K/V 无 per-layer norm → 修复
- **Heatmap loss**: MSE=0.000015 → 改 KL divergence，weight 10.0
- **M1**: Design doc 不一致 → 修正
- **结论**: 不通过

## 第二轮审查（Opus 4.6）
- 逐行验证维度流：Q(B,5,256) → attn(B,8,5,48) → out(B,5,256) ✅
- KL divergence 方向：KL(target_heatmap || pred_attn) ✅
- feat_proj 调用链完整 ✅
- 训练/推理路径正确（推理无 pose）✅
- 参数量 ~2.5M 正确注册到优化器 ✅
- **M1 (Medium)**: KL loss 初始可能 ~7-10，需监控 recon 值
- **L1**: design.md 描述已更新
- **L2**: use_skeleton_gcn=True hack 有维护风险（可接受）
- **结论**: ✅ 通过
