# 实验 exp022: Pose-guided Dual Stream (PDS)

## 动机
- exp007 PSG 在 Stage 3 注入 pose 信息，获得 +1.7% mAP（最佳单模块结果）
- 但所有在 PSG 基础上叠加模块的尝试都失败了（exp008/013/014/017/020/021）
- 核心原因：同一个 Stage 3 上叠加多个模块导致梯度干扰
- **PDS 的解决方案**：复制独立的 Stage 3 给 Part 分支，避免梯度干扰

## 创新点 / 核心假设
**梯度隔离的双分支架构可以同时获得 PSG 的全局增益和 Part Pooling 的局部增益，两者不再互相干扰。**

与 exp008（PSG+Part Pooling 在同一 Stage 3）的关键区别：
- exp008: 共享 Stage 3 → 梯度干扰 → mAP 57.7%（低于 PSG-only 58.3%）
- PDS: 独立 Stage 3 → 无干扰 → 预期 > 58.3%

## 技术方案

### 架构
```
Input → Swin Stage 0-2 (共享)
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
  Stage 3-G           Stage 3-P (独立权重，从预训练初始化)
  + PSG               (无 PSG)
    ↓                   ↓
  norm3-G             norm3-P
    ↓                   ↓
   GAP              Pose Part Pooling (5 parts)
    ↓                   ↓
  BN → FC           5×(BN → FC)
    ↓                   ↓
 Global ID+Tri      Part ID + Part Tri
    ↓                   ↓
    └── concat (test) ──┘
```

### 修改的文件
1. **新增 `model/pose_dual_stream_model.py`** — PDS 模型主体
2. **修改 `model/make_model.py`** — 添加 PDS 分支
3. **修改 `config/defaults.py`** — 添加 PDS config 选项
4. **新增 `configs/occluded_duke/pose_pds.yml`** — 实验配置

### 数据流
1. 图像经过 Swin Stage 0-2，输出 (B, H*W, 384) tokens
2. Stage 2 downsample 输出 (B, H'*W', 768) tokens（Stage 3 输入）
3. **Global 分支**: Stage 3-G 处理 tokens，PSG 在每个 block 后注入 pose → GAP → 768-d global feat
4. **Part 分支**: Stage 3-P 处理同样的 tokens（无 PSG）→ reshape 为 (B, 768, 12, 4) → Part Pooling → 5×768-d part feats
5. 训练：分别计算 global loss 和 part loss，加权求和
6. 测试：L2-norm(global) ∥ L2-norm(parts) 拼接

### 关键超参数
- Part 分支 Stage 3：从预训练权重初始化（与 Global 分支共享初始值）
- Part loss weight: 1.0（与 global loss 等权）
- Part triplet weight: 1.0
- Heatmap norm: spatial_softmax
- Test feat: L2-norm concat (global 768 + 5×768 = 4608-d)
- 无额外 LR 调整

### 参数量估算
- Stage 3 (2 blocks × 768ch): ~6M params
- norm3-P: ~1.5K
- Part BN + classifiers: 5 × (768 + 768 × num_classes) ≈ 5 × (768 + 768 × 702) ≈ 2.7M
- PSG (已有): ~102K
- **总额外**: ~8.8M params

## 预期结果
- **如果假设成立**: mAP > 58.3%（超过 PSG-only），R1 > 67.9%
  - 理想情况: PSG 全局增益 (+1.7%) + Part 独立增益 (+0.9%) 部分叠加 → mAP ~59-60%
- **如果失败**: 最可能原因是 Part 分支共享 Stage 0-2 的梯度仍然干扰 Global 分支
  - 此时可尝试：stop_gradient 隔离、Part 分支更低 LR、或 Part 分支延迟启动

## 对照组
- Baseline: exp000 (mAP 56.6%, R1 66.5%)
- PSG-only: exp007 (mAP 58.3%, R1 67.9%)
- PSG+Part 同 Stage: exp008 (mAP 57.7%, R1 66.0%)
- 消融变量：Part 分支是否使用独立 Stage 3
