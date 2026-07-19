# 实验 exp401：rich-budget C0正式训练

## 当前状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / FORMAL SEALED-PASS /
RICH_BUDGET_ROUTE_ALIVE / PHASE-B INTERFACE GO`

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

## 冻结结果（2026-07-19）

唯一fresh seed1234训练自然跑满e120，final full=`57.1/67.3/80.3/84.8`。同一唯一checkpoint的冻结
all-router-bypass为`57.0/67.4/80.0/84.6`；raw full−bypass mAP=
`+0.1194214838 point`，full raw mAP=`57.1230075595`，同时通过`+0.1`差值门和`56.7`绝对门。

41项checkpoint/state/source/asset/strict reload/RGB-only/patch恢复门全部PASS；241项state finite且
teacher-free，两个router与evidence head保留，唯一checkpoint SHA=
`fe00d08a9a0f651c2c0852c0661e720995a65292459aec9797a359895aa52efc`。因此按预注册规则封板为
`RICH_BUDGET_ROUTE_ALIVE / PHASE-B INTERFACE GO`。该PASS幅度仅比差值门高`0.0194214838 point`，且
R1差为`−0.0904977322 point`；它只授权下一阶段接口与强反事实，不构成多seed或论文主结果。
