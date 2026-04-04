# exp239: PPA with GiLt-style loss (Part triplet only, no Part CE)

## 动机
exp237 PPA = +0.5/-0.4。KPR 用 GiLt loss: global CE + part triplet-only。
当前 PPA 用 5 个 part CE + triplet，可能与 global CE 竞争。

## 技术方案
在 processor 中，当 PPA 启用时，Part branch 只用 triplet loss，不用 CE。
需要修改 loss 计算逻辑：跳过 score[1:] 的 CE，只计算 feat[1:] 的 triplet。

## 对照组
- exp237 (PPA w=0.5, full CE): 63.7/75.0
- exp191 (GCN): 63.2/75.4
