# 实验 exp023: PDS + Stop Gradient (Part → Shared)

## 动机
- exp022 PDS global-only 结果 57.9%，低于 PSG-only 58.3%（-0.4%）
- 这 -0.4% gap 来自 Part 分支通过共享 Stage 0-2 的反向传播干扰 Global 分支
- **假设**: 如果完全阻断 Part 分支到共享层的梯度，Global 分支应恢复到 58.3% 水平
- 同时 Part 分支仍可训练（独立 Stage 3 + Part classifiers），因为 Part 只需微调共享特征

## 创新点 / 核心假设
**阻断 Part → 共享层梯度后，Global 分支可恢复 PSG-only 性能，且 Part 分支仍能在 frozen 共享特征上学到有用的部件表示。**

## 技术方案

### 修改的文件
1. **修改 `config/defaults.py`** — 添加 `POSE_PART_STOP_GRAD = False`
2. **修改 `model/pose_dual_stream_model.py`** — 在 Part 分支输入处添加 `.detach()` 逻辑
3. **新增 `configs/occluded_duke/pose_pds_stopgrad.yml`** — 实验配置

### 核心代码修改
在 `forward()` 方法中，Part 分支的输入从 `shared_x.clone()` 改为 `shared_x.detach()`（当 stop_grad 开启时）：

```python
# Part branch input
if self.part_stop_grad:
    part_input = shared_x.detach()  # No gradient flows to shared stages
else:
    part_input = shared_x.clone()  # Original PDS behavior
```

### 梯度流变化
```
Before (exp022 PDS):
  Part loss → Part Stage 3 → shared_x → Stage 0-2 (updates shared weights)
  Global loss → Global Stage 3 + PSG → shared_x → Stage 0-2

After (exp023 PDS + stop_grad):
  Part loss → Part Stage 3 → STOP (detach)
  Global loss → Global Stage 3 + PSG → shared_x → Stage 0-2
```

### 关键超参数
- 与 exp022 完全相同，仅添加 `POSE_PART_STOP_GRAD: True`
- Test feat: `equal_concat`（训练时），多模式评估（训练后）

## 预期结果
- **如果假设成立**: global-only mAP ≥ 58.3%（恢复 PSG-only 水平）
  - Part-only 和 concat 结果可能更差（Part 分支缺少共享层梯度适应）
  - 但如果 global-only 与 PSG-only 持平，则证明 Part 分支干扰是可消除的
- **如果失败**: global-only 仍低于 58.3%
  - 说明 PDS 的问题不仅是梯度干扰，可能还有其他因素（如 `.clone()` 的前向计算本身影响）

## 对照组
- exp022 PDS global-only: 57.9%（直接消融对照）
- exp007 PSG-only: 58.3%（目标值）
- 消融变量：仅添加 stop_gradient，其他完全相同
