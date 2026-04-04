# exp240: PPA on Small (backbone scaling validation)

## 动机
exp237 PPA on Tiny: +0.5/-0.4。需要在 Small 上验证。
如果 PPA 在 Small 上也正向，这就是 cross-backbone 验证。
Small baseline (exp206r): 70.6/82.6。

## 技术方案
与 exp237 完全相同 PPA 配置 (w=0.5)，换 Small backbone。
ROA=False, PLBOA=0.7。
注意: Small + PPA 可能 OOM — PPA 在 non-detached 上操作。
如果 OOM，用 TEST.IMS_PER_BATCH 128。

## 对照组
- exp206r (Small OA-SD GCN): 70.6/82.6
- exp237 (Tiny PPA): 63.7/75.0 (+0.5/-0.4)
