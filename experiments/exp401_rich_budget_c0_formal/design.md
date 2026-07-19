# 实验 exp401：rich-budget C0正式训练

## 当前状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / STATIC-CPU SEALED-PASS /
FORMAL RUNNING`

## 动机

exp400已同时通过baseline-relative原生GradScaler稳态和最终production接口门，result显式
`formal_training_authorized=true`。exp401不再改模型、loss、batch、rho、schedule或seed，只执行唯一fresh
rich-budget C0 e120，回答该route在正式检索上是否真实存活。

## 核心假设

Frozen rich CLIP evidence能通过两个owned residual-budget consumer形成可测的ReID增益；若最终full相对
all-router-bypass至少高`0.1 mAP`，且full mAP不低于Semantic C0 `56.9`超过`0.2`，则route alive。

## 固定方案

- source exact=`11d7a35788c4645c355d96d76a2a4ff20a9801ac`，production source/config语义与exp400 exact；
- official Occluded-Duke train/test，batch64，seed1234，Swin-Tiny，120 epochs；
- default GradScaler、cosine warmup、base LR `8e-4`、weight decay `1e-4`、pose总权重`0.1`；
- rho：e1–e5=`0`，e6–e9线性，e10+/eval=`0.08075544983148575`；
- fresh execution/output/CLIP/codebook实体；不加载checkpoint、不续训、不换seed；
- eval每10 epoch只记录，checkpoint仅e120保存一次；中间性能和GateAbs不得早停或裁决。

## final裁决

e120自然结束后先做checkpoint/state/source/asset/teacher-free/RGB-only终审，再串行运行冻结的
all-router-bypass retrieval。只有同时满足：

1. `full_mAP - all_bypass_mAP >= 0.1`；
2. `full_mAP >= 56.7`；

才判`RICH_BUDGET_ROUTE_ALIVE / PHASE-B INTERFACE GO`。否则只关闭当前rich-budget C0 route，不永久否定
Phase0E、Phase0R或CLIP–TAPF总体。

## 风险与失败解释

- NaN/Inf/OOM/runtime error：封板当前formal execution，禁止续训或改参救活；
- full不足Semantic C0底线：production route整体NO-GO；
- full达标但bypass差不足：说明route执行影响不足，不能以teacher证据替代ReID route证据；
- 单seed只支持该冻结production route裁决，不自动形成多seed论文结果。
