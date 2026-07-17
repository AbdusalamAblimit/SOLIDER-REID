# exp143 SASA 代码审查报告

**审查日期**: 2026-03-21
**审查范围**: design.md, skeleton_attention.py, pose_backbone_model.py (SASA 部分), config/defaults.py, pose_psg_gcn_sasa.yml
**审查方法**: 逐行代码阅读 + 端到端功能验证脚本 + 形状/内存/AMP 兼容性测试

---

## 1. 设计文档 (design.md)

**结论: 通过**

设计清晰，动机合理。核心假设（基于骨架测地距离的零参数注意力偏置优于 KP-RPE 的欧式距离 MLP 偏置）有充分的理论依据。与 KP-RPE 的对比表格准确。风险和失败解释完备。

- 无问题。

---

## 2. 新模块: skeleton_attention.py

### 2.1 COCO 骨架定义

**结论: 通过**

验证结果:
- 18 条边连接 17 个关键点
- 图完全连通（BFS 从节点 0 可达所有 17 个节点）
- 包含标准 COCO 骨架连接加上 nose-to-shoulders 的额外边 (0,5) 和 (0,6)

注意: 标准 COCO 骨架通常不包含 (0,5) 和 (0,6) 这两条 nose-to-shoulder 边。标准定义是 nose 连接 left_eye 和 right_eye，shoulders 通过 shoulder-shoulder 连接。这里额外添加这两条边使 nose 直接连接 shoulders，缩短了头部到躯干的测地距离。这是一个合理的设计选择（不是 bug），因为 nose 和 shoulders 在图像中空间接近，且这个增强的骨架在多篇 pose-guided ReID 论文中有先例。

### 2.2 Floyd-Warshall 测地距离计算

**结论: 通过**

- 标准 Floyd-Warshall 三重循环实现，正确
- INF 初始值设为 `num_kp=17`（大于任何可能的最短路径长度 6），正确
- 对角线初始化为 0，正确
- 边权为 1（无向图），正确
- 归一化到 [0, 1] 通过 `geo / geo.max()`，正确（max=6）
- 注册为 buffer（不参与梯度），正确
- 验证: 最大测地距离为 6（left_ear 到 right_ankle），所有值合理

### 2.3 Token-to-Keypoint Assignment (compute_token_kp_assignments)

**结论: 通过**

- `F.interpolate(heatmaps, size=(H, W), mode='bilinear', align_corners=False)`: 正确，将 (B, 17, 96, 32) 的热图缩放到 (B, 17, 12, 4) 特征图分辨率
- `hm.argmax(dim=1)`: 正确，沿关键点维度取最大响应，返回 (B, 12, 4) int64
- AMP 兼容: F.interpolate 在 autocast 下保持 float32，argmax 返回 int64，均安全

### 2.4 compute_bias 方法

**结论: 通过（有一个 Medium 级建议）**

- 形状计算正确: (BnW, N) -> expand -> (BnW, N, N) -> indexing -> (BnW, N, N) -> unsqueeze+expand -> (BnW, num_heads, N, N)
- 使用 `self.geodesic_matrix[assign_i, assign_j]` 进行高级索引，索引张量为 int64，buffer 为 float32，结果为 float32，正确
- 负号乘法 `-self.alpha * geo_dist` 正确（距离越大，偏置越负）
- 对角线（同一关键点）值为 0（因为 geodesic_matrix 对角线为 0），即同一关键点的 token 对不受额外偏置，正确

**[Medium] M1: `.expand().contiguous()` 的内存开销**

`bias.unsqueeze(1).expand(-1, num_heads, -1, -1).contiguous()` 将 (BnW, 1, N, N) 展开到 (BnW, 24, N, N) 并分配完整内存。估算: B=64, nW=2 时，每次调用约 28 MB。由于 bias 在所有 head 间完全相同，`.contiguous()` 制造了 24 份相同数据的副本。

但是: WindowMSA 的 `attn + extra_attn_bias` 操作需要广播兼容的形状。如果去掉 `.contiguous()`，`expand()` 返回的是 view（零拷贝），PyTorch 的加法运算支持 stride=0 的 expand view，所以实际上可以不需要 `.contiguous()`。不过这不是 bug，只是内存效率建议。当前的 28 MB 开销在 24GB GPU 上可以接受。

### 2.5 参数检查

**结论: 通过**

模块确实是零参数:
- `geodesic_matrix` 注册为 buffer，不参与梯度
- `alpha` 是 Python float 属性，不是 nn.Parameter
- 没有其他 nn.Parameter 或 nn.Module 成员

---

## 3. Config 集成 (config/defaults.py)

**结论: 通过**

```python
_C.MODEL.POSE_SASA = False                  # 默认关闭
_C.MODEL.POSE_SASA_ALPHA = 0.1              # 默认 alpha
```

- 默认值 `False` 确保不影响现有实验
- alpha 默认值与 design.md 一致

---

## 4. Backbone 集成 (pose_backbone_model.py)

### 4.1 __init__ 中的 SASA 初始化

**结论: 通过**

```python
self.use_sasa = getattr(cfg.MODEL, 'POSE_SASA', False)
if self.use_sasa:
    from model.modules.skeleton_attention import SkeletonAttentionBias
    sasa_alpha = getattr(cfg.MODEL, 'POSE_SASA_ALPHA', 0.1)
    self.sasa_module = SkeletonAttentionBias(alpha=sasa_alpha)
```

- 位于 `use_kp_rpe` 检查之后，位置合理
- 使用 `getattr` 安全读取配置
- 延迟 import 避免不必要的模块加载
- `self.sasa_module` 正确挂载为子模块（SkeletonAttentionBias 继承 nn.Module）
- geodesic_matrix buffer 会正确跟随 `.to(device)` 和 `.state_dict()`

### 4.2 _compute_sasa_bias 方法

**结论: 通过**

逐步验证:

1. **Token assignment**: `compute_token_kp_assignments(scene_heatmaps, hw_shape)` -> (B, 12, 4)，正确
2. **Padding**: `F.pad(token_assign, (0, pad_r, 0, pad_b), value=0)` -> (B, 14, 7)
   - pad 参数顺序: (left, right, top, bottom) 对应 (W_left, W_right, H_top, H_bottom)
   - 对于 (B, H, W) 张量，最后一维是 W，所以 (0, pad_r) 在 W 维右侧填充，(0, pad_b) 在 H 维下方填充，正确
3. **Cyclic shift**: `torch.roll(token_assign, shifts=(-shift_size, -shift_size), dims=(1, 2))`
   - 与 ShiftWindowMSA 中对 query 的 shift 方向一致（dims=(1,2) 对应 H,W），正确
4. **Window partition**:
   - `view(B, H_pad//ws, ws, W_pad//ws, ws)` -> `permute(0, 1, 3, 2, 4)` -> `view(B*nW, ws*ws)`
   - 与 ShiftWindowMSA 中 `window_partition` 的逻辑一致（先 H 维分窗再 W 维分窗），正确
5. **Compute bias**: `self.sasa_module.compute_bias(token_assign, num_heads)` -> (B*nW, num_heads, ws*ws, ws*ws)，正确

### 4.3 SASA 在 _run_stage_with_psg 中的路由

**结论: 通过（有一个 Medium 级注意事项）**

SASA 的路由逻辑:

```python
kp_rpe_bias = None
if self.use_kp_rpe and ...:     # KP-RPE 优先
    kp_rpe_bias = ...
elif self.use_sasa and ...:      # SASA 次之
    kp_rpe_bias = self._compute_sasa_bias(...)
```

然后在下方:

```python
elif kp_rpe_bias is not None:
    x = block(x, hw_shape, extra_attn_bias=kp_rpe_bias)
    # PSG still applies after block
    if scene_heatmaps is not None and key in getattr(self, 'psg_modules_dict', {}):
        x = self.psg_modules_dict[key](x, hw_shape, scene_heatmaps)
```

关键确认:
- SASA bias 通过 `extra_attn_bias` 传入 block，正确
- PSG 在 block 之后仍然应用（line 890-891），正确
- 当 `use_combo=False`, `use_cross_attn=False`, `use_attn_bias=False` 时（本实验的情况），SASA 正确进入 `elif kp_rpe_bias is not None` 分支

**[Medium] M2: PAA/PGTM/TDPC/PCL 在 kp_rpe_bias 路径中不被应用**

当 `kp_rpe_bias is not None`（即 SASA 或 KP-RPE 激活时），代码进入 line 885-891 的分支，该分支只应用 PSG，不应用 PAA、PGTM、TDPC、PCL。这些模块只在 `else` 分支（line 892+）中应用。

对于本实验（exp143）这不是问题，因为配置中没有启用 PAA/PGTM/TDPC/PCL。但如果未来有实验想同时使用 SASA + PAA，需要修改路由逻辑。当前实现不影响 exp143 的正确性。

### 4.4 SASA 禁用时的安全性

**结论: 通过**

当 `POSE_SASA=False` 时:
- `self.use_sasa = False`
- `self.sasa_module` 不会被创建
- `_compute_sasa_bias` 不会被调用
- `kp_rpe_bias` 保持 `None`（除非 KP-RPE 也启用）
- 流程进入 `else` 分支，完全等价于原有行为

---

## 5. 配置文件 (pose_psg_gcn_sasa.yml)

### 5.1 与 exp030a 配置对比

**结论: 通过（有一个需要确认的差异）**

逐行对比 `pose_psg_gcn.yml`（exp030a）和 `pose_psg_gcn_sasa.yml`（exp143）:

| 参数 | exp030a | exp143 | 状态 |
|------|---------|--------|------|
| POSE_SASA | (不存在) | True | 新增，正确 |
| POSE_SASA_ALPHA | (不存在) | 0.1 | 新增，正确 |
| POSE_TEST_FEAT | concat_scaled | equal_concat | **差异** |
| OUTPUT_DIR | exp030a_psg_gcn | exp143_sasa | 正确 |
| 其他所有参数 | 一致 | 一致 | 正确 |

**[Medium] M3: POSE_TEST_FEAT 从 concat_scaled 改为 equal_concat**

exp030a 使用 `concat_scaled`，exp143 使用 `equal_concat`。根据 design.md，报告模式为 "equal_concat、global"，且 CLAUDE.md 中明确指出 "主汇报模式：equal_concat"。exp030a 的基准结果 `exp030a-eq` (60.73% mAP / 72.57% R1) 就是用 `equal_concat` 报告的。

因此 `equal_concat` 是正确的选择，与主汇报基线对齐。这不是配置错误，而是有意的选择。

### 5.2 其他配置验证

- PSG + GCN 均启用，与 exp030a 一致
- Batch size 64, LR 0.0008, SGD, 120 epochs，均与 exp030a 一致
- OUTPUT_DIR 使用独立路径，不会覆盖已有实验

---

## 6. AMP (混合精度) 安全性

**结论: 通过**

通过实际 CUDA 测试验证:

1. `F.interpolate` 在 autocast 下保持 float32 输入输出（不被 cast 到 float16），安全
2. `argmax` 始终返回 int64，不受 AMP 影响
3. `F.pad` 对 int64 张量正常工作
4. `torch.roll` 对 int64 张量正常工作
5. geodesic_matrix 为 float32 buffer，高级索引结果为 float32
6. 最终 bias 为 float32，与 attention scores 相加时自动广播，安全
7. 整个路径没有 float16 精度丢失风险

---

## 7. 形状一致性

**结论: 通过**

完整形状追踪:

```
输入:
  scene_heatmaps: (B, 17, 96, 32)
  hw_shape: (12, 4)
  window_size: 7

Step 1 - Token assignment:
  F.interpolate: (B, 17, 96, 32) -> (B, 17, 12, 4)
  argmax(dim=1): (B, 17, 12, 4) -> (B, 12, 4) int64

Step 2 - Padding:
  pad_r = (7 - 4%7) % 7 = 3
  pad_b = (7 - 12%7) % 7 = 2
  F.pad: (B, 12, 4) -> (B, 14, 7)

Step 3 - Cyclic shift:
  torch.roll: (B, 14, 7) -> (B, 14, 7)

Step 4 - Window partition:
  view: (B, 2, 7, 1, 7) -> permute: (B, 2, 1, 7, 7) -> view: (B*2, 49)
  nW = 2

Step 5 - Bias computation:
  assign_i: (B*2, 49, 49), assign_j: (B*2, 49, 49)
  geo_dist: (B*2, 49, 49)
  bias: (B*2, 49, 49)
  output: (B*2, 24, 49, 49) [num_heads=24 for Stage 3]

注意力加法:
  attn: (B*nW, num_heads, N, N) = (B*2, 24, 49, 49)
  extra_attn_bias: (B*2, 24, 49, 49)
  形状完全匹配 ✓
```

---

## 8. 问题汇总

### Critical 级别: 无

### High 级别: 无

### Medium 级别:

**M1: `.expand().contiguous()` 不必要的内存拷贝**
- 位置: `skeleton_attention.py` line 90
- 描述: `.contiguous()` 将零拷贝的 expand view 转为完整拷贝（约 28 MB/call），可以去掉以节省内存
- 影响: 不影响正确性，仅影响内存效率
- 建议: 可在本次实验后根据 GPU 内存情况决定是否优化

**M2: kp_rpe_bias 路径不应用 PAA/PGTM/TDPC/PCL**
- 位置: `pose_backbone_model.py` line 885-891
- 描述: SASA 激活时，PAA 等后续模块被跳过
- 影响: 对 exp143 无影响（未启用这些模块），但限制了未来组合实验
- 建议: 不影响本次实验，可后续按需修复

**M3: POSE_TEST_FEAT 从 concat_scaled 改为 equal_concat**
- 位置: `pose_psg_gcn_sasa.yml` line 23
- 描述: 与 exp030a 的 yml 文件不同，但与主汇报基线 (exp030a-eq) 对齐
- 影响: 这是正确的选择，不是 bug
- 建议: 无需修改

### Low 级别:

**L1: 填充区域的 token assignment 默认为 keypoint 0 (nose)**
- 位置: `_compute_sasa_bias` line 809, `F.pad(..., value=0)`
- 描述: 填充位置被分配给 keypoint 0 (nose)，导致填充 token 与真实 token 之间产生非零 SASA 偏置
- 影响: 实际影响极小。填充区域的 feature values 为零（来自 Swin 内部的 F.pad(query, 0)），所以即使 attention 权重被 SASA 影响，V 值接近零不会污染输出。此外，这个问题在 Swin 原始的 relative position bias 中同样存在（填充 token 使用的是 position bias table 中对应位置的值），是 Swin padding 机制的固有特征
- 建议: 可保持现状。如果后续实验需要更精确的控制，可以考虑将填充 token 的 SASA bias 设为 0

**L2: 模块未记录 COCO 骨架增强边的选择**
- 描述: 代码添加了非标准的 (0,5) 和 (0,6) 边（nose-to-shoulders），但注释只写了 "nose to shoulders" 而未说明这是对标准 COCO 骨架的增强
- 影响: 无功能影响，仅影响代码可读性
- 建议: 可添加一行注释说明

---

## 最终结论

**审查通过，可以开始训练。**

- 无 Critical 或 High 级别问题
- 3 个 Medium 级别问题均不影响实验正确性
- 2 个 Low 级别问题均为改进建议
- 代码实现与设计文档完全一致
- 端到端形状验证通过
- AMP 兼容性验证通过
- 配置文件与 exp030a 对照正确（仅新增 SASA 标志和调整 test feat 模式）
- 零参数设计确认：模块确实不引入任何可学习参数
