# exp050 PAMC 代码审查记录

## 第一轮审查

**审查范围**: pamc.py, defaults.py, pose_backbone_model.py, processor.py, config yml, design.md

### 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | processor.py | Masked view 在 train 模式下运行（DropPath 激活，BN 用 batch stats），一致性目标有噪声 | ✅ 已修复 |
| 8 | HIGH | design.md | 设计文档描述对称 loss，代码实现非对称 loss（不一致） | ✅ 已修复 |
| 3 | LOW | processor.py | `from model.modules.pamc import` 在训练循环内部 | ✅ 已修复 |
| 9 | LOW | pose_backbone_model.py | `self.pamc_warmup` 和 `self.pamc_weight` 存在模型上但从未使用（死代码） | ✅ 已修复 |
| 12 | LOW | processor.py | `_prepare_pose` 被调用两次（模型内部 + processor） | 接受（计算量极小） |
| 5 | LOW | pamc.py | `pixel_mean`/`pixel_std` 构造参数未使用 | 接受（不影响功能） |
| rand | LOW | pamc.py | Python `random.sample` 不受 `torch.manual_seed` 控制 | 接受（标准增强实践） |

### 修复操作
1. processor.py: 在 masked forward 前后加 `_m.eval()` / `_m.train()` 切换
2. design.md: 完全重写，准确描述非对称设计
3. processor.py: import 移至文件顶部
4. pose_backbone_model.py: `self.pamc_warmup/weight` 改为局部变量

## 第二轮审查

**结论**: ✅ **通过**

所有 HIGH 和 LOW 问题均已修复。额外检查通过：
- eval/train 模式切换正确恢复状态
- 与 GCN branch、DDP、BatchNorm 无交互问题
- POSE_PAMC=False 时训练路径与修改前完全一致
- 特征维度一致（768-d）
- Config / defaults / design.md / code 四方一致
