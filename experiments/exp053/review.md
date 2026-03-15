# exp053 XCAD 代码审查记录

## 第一轮审查

**审查范围**: design.md, pose_xcad.py, pose_backbone_model.py, defaults.py, config yml

### 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | LOW | design.md | 文件名写为 `pose_cross_attn_decoder.py`，实际为 `pose_xcad.py` | 接受（文档不一致） |
| 2 | LOW | pose_xcad.py | 额外的 LayerNorm 在残差输出上，GCN 没有（微小额外变量） | 接受（功能合理） |
| 3 | LOW | design.md | 参数量估计 ~7.3M 远超实际 ~1.17M（因 bottleneck 设计） | 接受（实际更高效） |
| 4 | LOW | pose_xcad.py | `kp_weight_mode` 硬编码为 `score`，非可配置。与 exp030a 默认值一致 | 接受（当前实验不影响） |

### 审查通过项

- ✅ 单变量原则：相对于 exp030a 只替换了 GCN→XCAD
- ✅ CrossAttentionEnhancer：shape 追踪全部正确 (Q: B,17,256; K,V: B,48,256; attn: B,8,17,48)
- ✅ 零初始化 out_proj 确保安全退化（初始输出 = norm(bilinear_sampled)）
- ✅ 梯度流正确：feat_map detached（与 GCN 相同），XCAD 参数正常接收梯度
- ✅ 接口兼容：forward() 返回值格式与 SkeletonGCNHead 完全一致
- ✅ 优化器自动包含 XCAD 参数
- ✅ 默认值 POSE_XCAD=False 不影响已有实验
- ✅ 边界情况处理：低置信度 keypoints、全零 scores、None pose_dict
- ✅ 设备/dtype 一致性
- ✅ 显存安全：attention map 仅 ~1.7MB@bs64，总参数 ~1.17M
- ✅ 配置文件 equal_concat 与 baseline 对照一致

### 结论

✅ **通过** — 无 Critical 或 High 级别问题。实现正确、干净、接口兼容。可以开始训练。
