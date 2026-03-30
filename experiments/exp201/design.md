# 实验 exp201: SupCon + OA-SD mutually exclusive 的确认 + 3-view CE 最终配置

## 动机
- exp199 (OA-RD+SupCon) 负结果：任何 EMA distillation 都与 SupCon 不兼容
- exp200 (OA-RD+CE) 负结果：OA-RD 不如 OA-SD
- 当前最佳两条互斥路线已确立：
  - SupCon 路线: exp187 = 64.9/76.6
  - OA-SD 路线: exp193 = 64.4/76.5
- **下一步应该做什么？** 需要一个不依赖 distillation 的新方向

## 核心实验
**SupCon + 3-view + PAPE + multi-stage PSG + 0.5x global loss + PLBOA**
这是 exp187 的完整配置。但 exp187 的 GLOBAL_LOSS_SCALE 是默认的还是 0.5x？让我确认。

实际上，我重新审视发现：exp187 使用的是 SupCon config (`pose_psg_stdpr_pertoken_plboa_pape_ms_supcon.yml`)，其中 GLOBAL_LOSS_SCALE 默认是 0.5。所以 exp187 已经是这个配置。

**那接下来真正需要的是一个全新的创新方向。**

## 实际方向：Curriculum Occlusion Training (COT)
从研究 agent 的建议中选择 "Curriculum Learning for Occlusion"：
- 训练前期（ep1-40）：正常 PLBOA prob=0.3（温和遮挡）
- 训练中期（ep40-80）：PLBOA prob=0.7（标准遮挡）+ heavy Random Erasing
- 训练后期（ep80-120）：PLBOA prob=0.9（强遮挡）+ 更强 RE

这不需要新模块，只需要在 processor.py 中根据 epoch 动态调整 PLBOA 概率。

## 核心假设
逐步增加训练难度（occlusion severity）可以让模型先学好基础特征，再学习遮挡鲁棒性。比一开始就用高遮挡率更有效。

## 技术方案
- 修改 `datasets/pose_dataset.py`: 添加动态 PLBOA prob 接口
- 修改 `processor/processor.py`: 每 epoch 更新 PLBOA prob
- 新 config: `POSE_CURRICULUM_OCC = True`, `POSE_COT_SCHEDULE = [0.3, 0.7, 0.9]`

## 预期结果
- 假设成立: mAP +0.5-1.0% vs exp187
- 如果中性: 证明 PLBOA 的固定概率已经够好

## 对照组
- exp187 (固定 PLBOA prob=0.7): 64.9/76.6

## 创新门槛
1. ✅ 问题层面：重新定义训练过程——从固定难度到课程式递增
2. ✅ 证据层面：可设计清晰的 3-stage vs 固定 消融
3. 🟡 机制层面：curriculum learning 不算新，但应用到 pose-guided occlusion augmentation 的 scheduling 有新意
