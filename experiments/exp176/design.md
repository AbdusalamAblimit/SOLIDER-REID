# 实验 exp176: SupCon Temperature Ablation (T=0.05)

## 动机
- exp174 SupCon T=0.07 是突破性结果 (63.6/75.3, +0.5/+1.4 vs exp166)
- Temperature 控制对比学习的"硬度"：更低 T = 更尖锐的分布 = 更关注 hard pairs
- T=0.05 可能更适合 ReID（需要区分外观相似的不同人）

## 技术方案
- 与 exp174 完全相同，仅改 POSE_STR_SUPCON_TEMP: 0.07 → 0.05
- 更低 temperature → 更强的对比信号 → 可能更好的 hard negative 分离

## 对照组
- exp174 (T=0.07): 63.6/75.3
