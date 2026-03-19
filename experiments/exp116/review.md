# exp116 审查报告

## 第一轮审查

### 发现的问题

| # | 严重程度 | 位置 | 描述 | 修复状态 |
|---|----------|------|------|----------|
| 1 | High | `support_complete_bank.py` | `@torch.no_grad()` 装饰器不必要（实际上无害因为 feat_map 已 detach） | ✅ 改为局部 `with torch.no_grad()` |
| 2 | High | Config | `POSE_SCKD_UPDATE_THR: 0.7` 违反单变量原则 | ✅ 改为 0.5 匹配 exp110 |
| 3 | Medium | `support_complete_bank.py` | 全遮挡样本 `vis_norm=0` 导致零向量替换 | ✅ 添加 `valid_samples` guard |
| 4 | Medium | `skeleton_gcn.py` | SGMKC 与 SCFR 潜在交互 | N/A 本实验不触发 |
| 5 | Low | `support_complete_bank.py` | `@torch.no_grad()` 误导性 | ✅ 重写 |
| 6 | Low | `skeleton_gcn.py` | `_scfr_bank` 外部赋值 | 保留（功能正确） |
| 7 | Low | `design.md` | "移除 distillation loss" 措辞不精确 | ✅ 代码注释已解释 |

### 第一轮结论：**不通过**（2 个 High 问题）

## 第二轮审查

所有第一轮问题已修复确认：
- Issue 1: 局部 `with` 块正确使用
- Issue 2: Config 已匹配对照组（UPDATE_THR=0.5）
- Issue 3: valid_samples guard 正确排除边界情况
- Issue 4-7: 已处理或记录

### 第二轮结论：**通过**

单变量确认：与 exp110 唯一差异为 `POSE_SCFR: True`
