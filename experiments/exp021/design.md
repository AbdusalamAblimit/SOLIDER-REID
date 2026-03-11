# 实验 exp021: Content-Adaptive PSG (CAPSG)

## 动机
- 20 个实验证明 PSG 是最有效的 pose 注入方法（+1.7% mAP），但所有**外挂模块**（PXA, PCG, PAB, PRA）都无法进一步提升 PSG
- PSG 的核心局限：门控信号完全由 pose heatmap 决定（`gate = f(heatmap)`），对于相同 pose 的不同图像，gate 完全相同
- 假设：如果让 gate 同时依赖 pose 和当前特征内容（`gate = f(heatmap, features)`），可以实现更精细的空间调制

## 创新点 / 核心想法
- **Content-Adaptive PSG (CAPSG)**：在 PSG 的 gate 生成过程中引入特征内容依赖，使门控信号同时考虑"哪里有身体部位"（pose）和"当前特征在说什么"（content）
- 与标准 PSG 相比：gate 从 pose-only 变为 pose-content joint conditioning
- 保持 PSG 的逐元素乘法范式（不引入 attention），避免 PXA 的过拟合问题

## 技术方案

### 标准 PSG（当前 exp007）:
```python
pose_feat = Conv2d(17→64)(heatmap) → ReLU
gate = Conv2d(64→768)(pose_feat) → Sigmoid
output = x * (1 + gate)
```

### CAPSG（本实验）:
```python
pose_feat = Conv2d(17→64)(heatmap) → ReLU      # 来自 pose
img_feat = Conv2d(768→64)(features) → ReLU      # 来自当前特征 (NEW)
combined = pose_feat * img_feat                  # 逐元素交互 (NEW)
gate = Conv2d(64→768)(combined) → Sigmoid
output = x * (1 + gate)
```

### 修改的文件
1. `model/modules/pose_spatial_gate.py` — 新增 `ContentAdaptivePSG` 类（不修改原 `PoseSpatialGate`）
2. `model/pose_backbone_model.py` — 新增 config flag 切换 CAPSG
3. `config/defaults.py` — 新增 `POSE_PSG_CONTENT_ADAPTIVE = False`
4. `configs/occluded_duke/pose_capsg.yml` — 新 config 文件

### 关键设计决策
- `img_feat = Conv2d(768→64, 1×1)` 用 1×1 conv 投影特征到低维空间，保持轻量
- 逐元素乘法 `pose_feat * img_feat` 而非 concat 或 attention，保持 PSG 的局部操作特性
- `gate_proj = Conv2d(64→768, 1×1)` 零初始化，保证初始行为等同标准 PSG（因为 img_feat 不为零但 gate_proj 零初始化）
- 注意：需要仔细处理初始化，使 CAPSG 初始行为 ≈ PSG

### 初始化策略
- `pose_proj`: 正常初始化（与标准 PSG 相同）
- `feat_proj`: 正常初始化
- `gate_proj`: 权重零初始化，偏置零初始化
- 这样初始 gate = sigmoid(0) = 0.5，而非 PSG 的 sigmoid(conv(relu(conv(hm))))
- 问题：初始 gate 恒为 0.5 意味着所有位置被均匀放大，相当于 x * 1.5
- 更好的方案：保留 PSG 的 pose_proj → gate_proj 路径，CAPSG 作为**残差修正**：
  ```python
  pose_gate = gate_proj_pose(relu(pose_proj(hm)))  # 标准 PSG 路径
  content_mod = gate_proj_content(pose_feat * img_feat)  # CAPSG 修正项（零初始化）
  gate = sigmoid(pose_gate + content_mod)
  output = x * (1 + gate)
  ```
  这样初始行为完全等同 PSG，content 修正项逐渐学习

### 参数估算
- 标准 PSG 每个 gate: Conv(17→64) + Conv(64→768) = 17×64 + 64×768 = 1,088 + 49,152 = ~50K
- CAPSG 额外每个 gate: Conv(768→64) + Conv(64→768) = 768×64 + 64×768 = ~98K
- 总额外参数: 2 gates × 98K = ~196K
- 总参数: 标准 PSG ~102K + CAPSG ~196K = ~298K

## 预期结果
- **乐观**: mAP 58.5-59.0%（CAPSG 的 content-dependence 捕获了 PSG 遗漏的信息）
- **中性**: mAP 58.0-58.3%（CAPSG 修正项没学到有用的东西，退化为 PSG）
- **悲观**: mAP < 58.0%（CAPSG 引入了额外噪声/过拟合，类似 PXA）

## 失败的最可能原因
1. Content-dependent gating 引入过拟合（与 PXA 类似）
2. 额外的 feat_proj 计算引入无用的梯度噪声
3. 逐元素乘法 pose_feat * img_feat 无法有效编码 pose-content 交互

## 对照组
- **Baseline 对照**: exp007 PSG (mAP 58.3%, R1 67.9%)
- **消融变量**: PSG gate 是否依赖 content（唯一变化）
