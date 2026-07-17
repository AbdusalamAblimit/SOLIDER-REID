# exp078 APG 代码审查记录

## 完整审查 — 通过 ✅

### 审查维度逐项结论

| 维度 | 结论 | 严重程度 |
|------|------|---------|
| a. 设计文档 | PASS | — |
| b. pose_additive_adapter.py | PASS | — |
| b. pose_backbone_model.py | PASS | — |
| b. defaults.py | PASS | — |
| c. 配置对比 (vs exp066) | PASS | 差异仅 ADAPTIVE_GATE + OUTPUT_DIR |
| d. 数据流 | PASS | hm→sigmoid→mean→gate_mlp→sigmoid→gate, shapes 全部正确 |
| e. 梯度流 | PASS | gate_mlp 在模块树中，optimizer 自动收集 |
| f. 向后兼容 | PASS | 默认 False 跳过所有新代码 |
| g. routed+gate 交互 | WARNING (Low) | 双重抑制，但 exp078 不启用 routed |
| g. ST-PAA+gate 交互 | WARNING (Medium) | 34ch 混合信号，但 exp078 不启用 ST-PAA |
| h. 边界条件 | PASS | sigmoid(0)=0.5 安全退化 |

### 关键验证
- zero-init: sigmoid(0)=0.5，PAA 初始输出减半（safe identity）
- gate shape: (B,17) → Linear(17,1) → (B,1) → unsqueeze → (B,1,1) 广播正确
- 参数增量: 2 × 18 = 36 params（忽略不计）
- 梯度: sigmoid(0) 处梯度 = 0.25（最大值），无消失风险

### 结论
审查通过，可以开始训练。
