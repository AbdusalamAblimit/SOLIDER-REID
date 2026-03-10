# exp010: Backbone Freeze Warmup for PSG

## 动机
exp007 (PSG Stage 3) 是当前最佳方法 (mAP 58.3%, R1 67.9%)。但 PSG 模块虽然零初始化（初始 gate=0），训练时 backbone 和 PSG 同时更新。这可能导致：
1. PSG 在训练初期还没学会有效的 gate pattern 时，backbone 已经适应了"无 gate"的状态
2. PSG 的梯度信号太弱（零初始化，初始输出为 0），被 backbone 的大梯度"淹没"
3. warmup 阶段 backbone 在快速适应新的 LR schedule，PSG 来不及"跟上"

如果先冻结 backbone 让 PSG 在固定特征空间上学会 pose-to-gate mapping，解冻后 PSG 已经有了良好的 gate 初始化，可能带来更好的最终性能。

## 核心假设
冻结 backbone 前 N epochs，只训练 PSG + classifier + BN layers，让 PSG 先学稳定的 spatial gate pattern，之后解冻 backbone 时能产生更有效的 pose-conditioned 特征调制。

## 技术方案
- 基于 exp007 的 PSG Stage 3 配置
- 新增 config: `SOLVER.FREEZE_BACKBONE_EPOCHS: 5`
- 在 `processor/processor.py` 的训练循环中，前 N epochs 冻结 backbone 参数（`requires_grad=False`），只训练 PSG、classifier、BN layers
- 第 N+1 epoch 解冻所有参数
- Config: `configs/occluded_duke/pose_psg_freeze.yml`
- Output: `./log/occluded_duke/exp010_psg_freeze`

## 实现细节
需要修改的文件：
1. `config/defaults.py`: 添加 `FREEZE_BACKBONE_EPOCHS` 参数
2. `processor/processor.py`: 在 epoch 循环中加入 freeze/unfreeze 逻辑
3. 新 config 文件: `configs/occluded_duke/pose_psg_freeze.yml`

冻结逻辑：
- 冻结 `model.base` 的所有参数（backbone）
- 保持训练：`model.psg_modules_dict`、`model.classifier`、`model.bottleneck`
- BN layers 的 running stats 仍然更新（eval mode vs freeze）

## 预期结果
- 如果假设成立：mAP 59-60%，PSG 学到更好的 gate pattern
- 如果失败：可能是因为 PSG 需要 backbone 的梯度信号来学习有用的 gate（不冻结反而让 PSG 和 backbone 协同适应更好）
- 最可能的失败原因：5 epochs 太短，PSG 还没学好；或冻结反而限制了 backbone 的适应能力

## 对照组
- Baseline 对照: exp007 (PSG Stage 3, 无 freeze, mAP 58.3%, R1 67.9%)
- 消融变量: 是否冻结 backbone 前 5 epochs
