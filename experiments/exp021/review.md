# exp021 审查记录

## 第一轮审查 — PASS

### 关键验证点:
1. ContentAdaptivePSG forward 与设计文档一致，形状正确 ✅
2. 零初始化正确保证初始行为 = 标准 PSG ✅
3. Config 正确连接（defaults.py → yml → model） ✅
4. 单变量原则：与 exp007 仅 POSE_PSG_CONTENT_ADAPTIVE 不同 ✅
5. inplace ReLU 安全（feat_proj 产生新 tensor） ✅
6. 梯度流正确（x 同时用于 gate 计算和最终乘法，无冲突） ✅
7. 参数估算准确（~300K total, ~198K extra vs PSG） ✅
8. 显存开销极小（Stage 3 H=12, W=4） ✅
