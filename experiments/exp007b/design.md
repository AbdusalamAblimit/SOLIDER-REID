# 实验 exp007b/c: Loss Scale 敏感性分析

## 动机
- exp007a (0.5x loss) 达到 mAP 59.5%, 比 exp007 (1.0x) 高 +1.2%
- 论文需要 loss scale 敏感性分析图表（常见论文 figure）
- 确定最优 loss scale 和改善幅度的边界

## 核心想法
- 在 PSG 基础上，仅改变 GLOBAL_LOSS_SCALE，测试 {0.25, 0.5, 0.75, 1.0} 四个值
- 与 exp007 (1.0x) 和 exp007a (0.5x) 组成完整网格
- **Config-only 变更，无代码修改**

## 技术方案
- exp007b: `GLOBAL_LOSS_SCALE: 0.25` → `pose_psg_quarter_loss.yml`
- exp007c: `GLOBAL_LOSS_SCALE: 0.75` → `pose_psg_threequarter_loss.yml`
- exp007: 1.0x (已有), exp007a: 0.5x (已有)

## 预期结果
- 预期最优在 0.5x 附近（exp007a 已验证）
- 0.25x 可能过度正则化（loss 太小, underfitting）
- 0.75x 可能中间效果
- 但由于训练方差 (~2%), 精确排序可能不可靠

## 对照组
- exp007 (1.0x): mAP 58.3%
- exp007a (0.5x): mAP 59.5%
- 消融变量: GLOBAL_LOSS_SCALE (0.25 / 0.75)

## 论文用途
- 敏感性分析图: x轴 = loss scale, y轴 = mAP/R1
- 证明方法对超参数不过度敏感
