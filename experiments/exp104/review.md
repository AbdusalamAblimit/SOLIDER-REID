# exp104 PACD 审查记录

## 审查轮次 1 (初版)
- BUG: feat_maps 在 parallel_aug 路径未捕获 → 已修复

## 审查轮次 2 (外部 Claude 发现 mask bug)
- CRITICAL: sigmoid(hm)>0.5 几乎处处为真 → mask 76% 而非 40%
- CRITICAL: GAP 未重归一化 → student 特征幅值系统性偏小
- CRITICAL: MSE 随特征范数增大而增大 → loss 不可收敛
- 已修复: 改用关键点坐标 + 归一化池化 + cosine distance

## 审查轮次 3 (完整审查修复后代码)
修复的问题:
| ID | 严重度 | 描述 | 修复 |
|----|--------|------|------|
| 1 | MEDIUM | 单点 mask 只覆盖 8-12% (太弱) | ✅ 扩展到 3×3 邻域 (~30-40%) |
| 2 | MEDIUM | Python loop .item() 慢 | ✅ 改为收集索引后批量 scatter |
| 3 | LOW | parallel_aug 路径 feat_maps 丢失 | ✅ 捕获 fm_v |

待二次审查确认。
