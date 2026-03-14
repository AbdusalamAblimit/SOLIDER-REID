# 实验 exp056: PGAM 多 Stage 注入 (Stage 2 + Stage 3)

## 动机
- exp054/055 的 PGAM 仅在 Stage 3（12×4 feature map）注入，效果为正向边缘信号
- Stage 3 只有 2 个 7×7 window，每个 window 内可供 mask 的非 body token 数量有限
- Stage 2 的 feature map 是 24×8（192 tokens），window partition 为 (24/7)×(8/7) ≈ 4×2 = 8 个 window
- 更多 window + 更高分辨率 = PGAM 的 masking 效果应该更显著
- PSG 在 Stage 2 注入曾经失败（exp005），但 PSG 需要学习参数而 PGAM 无参数

## 创新点 / 核心想法
- 核心假设: 在更早的 Stage（更高分辨率）注入 PGAM，让 backbone 更早地隔离遮挡物的注意力影响
- 与 PSG Stage 2 失败的区别: PSG 在 Stage 2 修改特征值（可能破坏 SOLIDER 预训练的低级特征），PGAM 只修改注意力路由（不改变特征值本身）

## 技术方案
- 修改 config: `POSE_PSG_STAGES: [2, 3]` → PSG 仍只在 Stage 3，但 PGAM 跟随 PSG stages
- 实际上需要分离 PSG 和 PGAM 的 stage 配置
- 或者更简单：直接把 PGAM 也加到 Stage 3 中所有 block（当前就是这样）再加 Stage 2 的 block

## 预期结果
- 如果成功: Stage 2 PGAM 进一步改善特征质量
- 如果失败: Stage 2 的 masking 破坏了低级特征学习
- 如果中性: 与 Stage 3 alone 相同，说明 masking 效果主要在最后一层

## 对照组
- 对照: exp054 (PGAM Stage 3 only)
- 消融变量: 增加 Stage 2 的 PGAM
