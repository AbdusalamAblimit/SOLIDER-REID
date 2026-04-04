# exp242: PPA + GCN 双分支 on Small

## 动机
exp241 (PPA+GCN Tiny): +0.5/-0.1 — 最佳综合结果。
需要在 Small 上验证。

## 技术方案
与 exp241 相同，换 Small backbone。ROA=False, PLBOA=0.7。

## 对照组
- exp241 (Tiny PPA+GCN): 63.7/75.3 (+0.5/-0.1)
- exp206r (Small GCN): 70.6/82.6
