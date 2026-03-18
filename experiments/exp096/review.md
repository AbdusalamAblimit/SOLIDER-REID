# exp096 MRKF 代码审查

## 审查轮次 1

### 发现的问题

| ID | 严重度 | 描述 | 状态 |
|----|--------|------|------|
| 1 | MEDIUM | mrkf_norm 的 learnable 参数不会收到梯度（被 detach 阻断）| ✅ 改为 elementwise_affine=False |

### 验证通过项
- ✅ Shape 正确：Stage2 (384,24,8) → sample → proj(256) → cat → fusion(768)
- ✅ grid_sample 坐标映射对不同分辨率正确（归一化到 [-1,1]）
- ✅ 训练时 Stage 2 特征正确 detach
- ✅ zero-init fusion → 初始为恒等映射
- ✅ MRKF=False 时无行为变化
- ✅ 内存增加 ~18MB（可忽略）
- ✅ Stage 2 `out` 变量捕获正确（下采样前的 pre-downsample 特征）
- ✅ 测试路径正确传递 stage2 特征
- ✅ 单变量隔离：仅 POSE_MRKF 差异

### 结论
✅ **审查通过，可以开始训练**
