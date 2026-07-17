# exp259: Weight Decay / OA-SD 调参 on Small GCN512+2stage

## 动机
exp255 最佳 73.2/83.3。探索正则化和蒸馏强度调参。

## 变体
- exp259: WD=2e-4 (vs baseline 1e-4), 更强正则化
- exp259b: OA-SD weight=2.0 (vs baseline 1.0), 更强 teacher 蒸馏

## 对照
exp255 (WD=1e-4, OA-SD=1.0): 73.2/83.3, MaxSim 73.5/83.8
