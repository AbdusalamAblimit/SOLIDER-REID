# exp092 LSRM 审查报告

## 审查范围
- `experiments/exp092/design.md`
- `configs/occluded_duke/pose_psg_gcn_paa_lsrm.yml`
- `model/modules/skeleton_recovery.py`
- `model/pose_backbone_model.py`
- `processor/processor.py`
- `log/occluded_duke/exp092_lsrm/train_log.txt`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | design.md vs repo | 设计要求“测试时用 SRM 代替 naive SGCFR recovery”，但仓库里没有任何 test-time LSRM 脚本或推理接线，`lsrm` 从未进入实际检索流程 | 未修复 |
| 2 | HIGH | processor.py / pose_backbone_model.py | 当前实现只把 `lsrm_loss` 作为一个额外辅助损失加入训练，恢复后的 keypoint feature 并没有回写到 GCN 主分支、也没有参与主 ID+Triplet loss；这和 design.md 的“recovery 改变主特征”不一致 | 未修复 |
| 3 | MEDIUM | skeleton_recovery.py | 训练目标是“同 ID 可见 partner 的特征均值”，而不是 design 里写的“目标图像该 keypoint 的真实 feature”，这是一个更弱的代理任务 | 未修复 |

## 审查通过项

- `lsrm` 已注册进模型，参数会被正常保存/加载
- Cross-attention recovery 模块本身 shape/梯度路径是成立的
- 日志完整，120 epoch 结束

## 结论

❌ **不通过**

`exp092` 现在测到的不是“learned recovery module 改写检索范式”，而只是一个训练期辅助 recovery loss。现有结果不足以验证 design.md 的核心假设。
