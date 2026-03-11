# 实验 exp025: PDS + Delayed StopGrad (梯度预热隔离)

## 动机
- exp023 (PDS+永久StopGrad) 取得 mAP 59.5%，但永久阻断 Part 梯度在理论上意味着 Global 分支等同于 PSG-only，+1.2% 的提升难以解释
- exp022 (PDS 无StopGrad) mAP 57.9%，Part 分支的随机初始化梯度破坏了共享层的预训练权重
- **核心矛盾**: 完全不阻断 → Part 噪声梯度干扰 Global；永久阻断 → Part 无法贡献有用梯度给共享层
- **解决思路**: 延迟释放——前 N 轮阻断（保护预训练权重），之后放开（让 Part 的已收敛梯度优化共享层）

## 创新点 / 核心假设
- **核心假设**: Part 分支在早期（分类器随机初始化时）产生噪声梯度会破坏预训练特征，但在后期（分类器基本收敛后）其梯度是有益的，能帮助共享层学习更好的多任务特征
- **与 exp022/023 的区别**:
  - exp022: 始终允许 Part 梯度流 → 早期噪声干扰
  - exp023: 永久阻断 Part 梯度流 → 后期浪费有用信号
  - **exp025**: 前 30 轮阻断，之后放开 → 两阶段优化
- **论文故事**: "Gradient Warmup for Part Branch" — 保护预训练权重的同时允许后期联合优化

## 技术方案
- 修改 `model/pose_dual_stream_model.py`:
  - 新增 `stop_grad_epochs` 配置项和 `current_epoch` 属性
  - forward 中根据 `current_epoch <= stop_grad_epochs` 决定是否 detach
- 修改 `processor/processor.py`:
  - 每个 epoch 开始时设置 `model.current_epoch = epoch`
- 修改 `config/defaults.py`:
  - 新增 `POSE_STOP_GRAD_EPOCHS = 0` (0=使用静态 POSE_PART_STOP_GRAD)

### 关键超参数
- `POSE_STOP_GRAD_EPOCHS: 30` — 前 30 轮阻断，与 warmup 期（20 轮）重合 +10 轮
- 选择 30 的依据: 从 exp022 看 Part ID loss 在 ep30 仍为 4.84 (vs Global 1.60)，说明 Part 分类器此时仍未收敛到稳定水平。但 ep30 后 Part loss 下降速度减缓，梯度信号趋于稳定

### 数据流
- Epoch 1-30: `shared_x.detach()` → Part branch (与 exp023 相同)
- Epoch 31-120: `shared_x.clone()` → Part branch (与 exp022 相同)
- Global branch 始终不受影响: `shared_x.clone()` + PSG

## 预期结果
- **最好情况**: mAP > 59.5% (超过永久 StopGrad)
  - 前 30 轮保护了预训练权重 → 后 90 轮 Part 梯度提供额外的多任务监督信号
- **中等情况**: mAP ≈ 58-59%
  - 释放后 Part 梯度有轻微干扰但不如 exp022 严重
- **最差情况**: mAP < 57.9% (低于 exp022)
  - 释放后 Part 梯度突然改变优化景观导致不稳定
- **如果失败**: 最可能原因是释放梯度时的"过渡冲击"——共享层突然接收 Part 梯度导致特征空间扰动

## 对照组
- **直接对照**: exp023 (PDS+永久StopGrad, mAP 59.5%) — 区别: 永久 vs 延迟
- **直接对照**: exp022 (PDS+无StopGrad, mAP 57.9%) — 区别: 延迟 vs 无
- **消融变量**: 仅改变梯度阻断策略 (永久 → 延迟 30 轮)

## 论文价值
1. 如果成功: "Gradient Warmup" 成为一个可解释的、有明确动机的训练策略
2. 消融表中可以展示 {无隔离, 永久隔离, 延迟隔离} 三种策略的对比
3. 论文故事从"梯度隔离"升级为"两阶段训练策略"
