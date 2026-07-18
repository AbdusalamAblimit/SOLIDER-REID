# exp393 Phase 0E：rich CLIP evidence teacher-only协议

## 状态

`PROTOCOL FROZEN / 0E-S SYNTHETIC SEALED-PASS / 0E-C8 8-IMAGE NO-START /
0E-128/FULL NO-START`。

本阶段不构建ReID model、optimizer、训练config、output或checkpoint。teacher-only审计与Phase A的
route activation逻辑独立；Phase 0E失败只关闭当前rich evidence code并阻断Phase B，不取消已通过
自身preflight的Phase A。

## 固定输入

- official clean Occluded-Duke train：15,618图；
- exp386 fresh train-only ViTPose-H manifest：
  `cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8`；
- frozen OpenCLIP ViT-L/14 checkpoint：
  `9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e`；
- exp392已验证的hard-owner ontology、aspect-letterbox geometry、PC-MBCLS shared block0–19与
  region-conditioned block20–23 CLS readout；
- seed=`20260719`，不读取query/gallery，不使用PID优化teacher数值。

## 固定code

对每图、每个valid slot：

```text
e_raw      = normalize(region_cls) - normalize(global_cls)
mu_r       = mean_fit_pid_partition(e_raw[r])
e_centered = e_raw - mu_r
e_code     = normalize(P_shared_16 @ e_centered)
```

official train按PID hash做deterministic disjoint fit/audit划分。五个slot只分别估计`mu_r`；PCA在
全部fit slot-centered rows上拟合一个共享16维坐标系，并对每个basis vector做最大绝对坐标为正的
sign canonicalization。PCA不读取PID label值，只用PID构造无泄漏partition；门禁只在held-out PID
audit partition裁决。

## 串行审计

### 0E-S：synthetic exact

脚本：`phase0e_static_contract.py`。必须通过：

1. PID split deterministic、fit/audit identity exact-disjoint；
2. slot mean在fit rows中心化到`<=1e-12`；
3. shared PCA basis orthonormal误差`<=1e-10`且sign固定；
4. valid code unit norm、invalid code exact zero、repeat exact、全量finite；
5. hard-owner slot-cycle mask IoU与pairwise product exact zero；
6. synthetic matched-positive相对wrong RGB/wrong mask margin方向为正。

冻结结果：`PASS`。fit/audit image=`26/54`，五slot fit count均为`26`；中心化最大误差
`4.2701e-17`，basis正交最大误差`1.9984e-15`，valid code单位范数最大误差`2.2204e-16`；
wrong RGB/wrong mask synthetic margin=`+0.72342/+0.86266`，hard-owner pairwise product与
slot-cycle IoU均为exact zero，全部finite。script/result SHA256分别为
`6c1b370912f5f668ce117d4320d62b68a032549ff06821f5bee1ae020acb3dab`与
`120085ddffdea2d18adfd73a856426229bfb132218e79fef6e0dc318d49c23ac`。该PASS只验证数学与数据
契约，不证明真实CLIP code有效。

### 0E-C8：8图真实contract

只在0E-S PASS后执行。8图由official train按path hash确定，必须覆盖至少两个PID和五个slot；不足时
顺序扩展候选但最终仍固定8图。读取correct、同步水平flip、different-PID donor RGB+donor mask、
donor RGB+recipient mask、同RGB low-IoU slot-cycle mask。必须验证：

1. PC-MBCLS官方global tail parity、repeat、NULL、hard-owner overlap、RGB/mask flip exact；
2. region/global feature与raw/centered/code全finite，shape分别为`[N,5,768]`、`[N,768]`、
   `[N,5,16]`；
3. teacher、PCA、counterfactual均无optimizer/grad，CUDA peak与吞吐落盘；
4. donor不同PID、无fixed point，wrong-mask target IoU为0；
5. 8图只裁决contract，不裁决统计性能，不因单slot小样本margin关闭路线。

### 0E-128与0E-FULL

0E-C8 PASS后先做128图稳定性，再做official train held-out PID全量裁决。正式门禁：

1. 五slot held-out within-slot逐维variance非零；
2. macro entropy effective rank至少`8/16`，同时报告top singular energy fraction；
3. matched positive为correct与同步flip；negative分别为different-PID same-slot RGB、same-RGB
   low-IoU wrong mask、slot-cycle；wrong RGB与wrong mask paired margin逐slot PID-cluster 95% CI下界
   都大于0；
4. slot-zero/static mean、raw uncentered、fixed random orthogonal、global-only为强对照；
5. result、runner、script、fit-partition、PCA mean/basis分别记录SHA，异常/NaN/Inf/AMP warning为0。

full PASS只授权Phase B使用该teacher code；不授权semantic multi-stage，也不替代Phase A final裁决。

## 失败解释

- rank失败：当前region-global residual仍被CLIP各向异性或slot prior主导，只封板该code；
- wrong RGB失败：code没有绑定到当前人的局部appearance；
- wrong mask失败：code没有绑定到正确局部support；
- flip失败而其他门通过：先审查几何/ontology一致性，不能调阈值救场；
- static或random control解释主要差异：不能称为CLIP-owned evidence，Phase B不启动。
