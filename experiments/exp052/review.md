# exp052 KP-RPE 代码审查记录

## 第一轮审查

**审查范围**: design.md, keypoint_rpe.py, pose_backbone_model.py, swin_transformer.py, defaults.py, config yml

### 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | LOW | keypoint_rpe.py | `score_threshold` 属性在 KeypointRPE 中存储但未在 forward() 中使用（实际阈值在 compute_token_kp_distances 中处理） | 接受（功能正确） |
| 2 | LOW | keypoint_rpe.py | 归一化使用 feature map 对角线近似而非像素对角线，差异 < 0.03% | 接受 |
| 3 | MEDIUM | pose_backbone_model.py | 如果同时启用 KP-RPE 和 PAB/combo/PXA，KP-RPE bias 会被计算但不被使用（elif 链）| 接受（当前实验不会触发） |
| 4 | LOW | pose_backbone_model.py | `getattr(self, 'psg_modules_dict', {})` 冗余但安全 | 接受 |
| 5 | LOW | swin_transformer.py | `pose_bias_map` 存在时会覆盖 `extra_attn_bias`，但当前不会同时使用 | 接受 |

### 审查通过项

- ✅ 单变量原则：相对于 exp030a 只添加了 KP-RPE（POSE_KP_RPE=True）
- ✅ KeypointRPE 模块：距离计算、pairwise difference、MLP mapping 正确
- ✅ 零初始化确保安全退化
- ✅ 梯度流正确：通过 attention scores → MLP → distance computation
- ✅ 窗口分区逻辑正确：padding、cyclic shift 与 ShiftWindowMSA 一致
- ✅ 优化器自动包含 KP-RPE 参数
- ✅ 默认值 POSE_KP_RPE=False 不影响已有实验
- ✅ 边界情况处理：None pose_dict、低置信度关键点、全零距离
- ✅ 设备/dtype 一致性
- ✅ 内存安全：7.99GB peak for B=64
- ✅ 参数量 2,736 确认正确

### 结论

✅ **通过** — 无 Critical 或 High 级别问题。实现正确、干净、隔离良好。可以开始训练。
