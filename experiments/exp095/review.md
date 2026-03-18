# exp095 DPF 代码审查

## 审查轮次 1

### 审查范围
- `model/modules/skeleton_gcn.py` — 核心 DPF 实现
- `model/pose_backbone_model.py` — DPF 配置传递
- `config/defaults.py` — POSE_DPF 默认值
- `configs/occluded_duke/pose_psg_gcn_paa_dpf.yml` — 实验配置

### 发现的问题

| ID | 严重度 | 描述 | 状态 |
|----|--------|------|------|
| 1 | MEDIUM | 精度权重可极大 (max 1e6)，一个关键点可能完全主导池化特征 | ✅ 已修复: eps=1e-3 + clamp(max=1e3) |
| 2 | LOW | kp_scores 未显式用 person_mask 遮蔽，但数据管线保证安全 | 接受风险 |
| 3 | LOW | GCN 在区域池化均值上操作（非点采样特征），输入统计量不同 | 设计考虑，非 bug |
| 4 | LOW | kp_vars 在 aux_data 中但暂无测试脚本使用 | 后续扩展使用 |

### 结论
✅ **审查通过，可以开始训练**

Shape 正确性、梯度流、配置安全性、向后兼容性全部通过。单变量隔离（仅 POSE_DPF）干净。
