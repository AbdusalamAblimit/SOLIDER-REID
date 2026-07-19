# 实验 exp402：Phase-B RGB-only semantic-interface counterfactual

## 当前状态

`SEALED / VALIDITY PASS / CURRENT_SEMANTIC_INTERFACE_NO-GO`

## 动机

exp401已经以正式e120 checkpoint证明fixed-budget rich route满足`full−all-bypass=+0.1194214838 mAP`
point，因而route不是retrieval identity；但该差值只比预注册门高`0.0194214838 point`，R1还低
`0.0904977322 point`。这只能授权Phase-B interface，不能证明router使用的是图像特定、slot特定的
rich evidence，而不是mask、generic residual或任意16维扰动。

exp402不训练、不改checkpoint，也不再次回答route alive。它在同一sealed checkpoint上做一次只读、
全validation、串行反事实kill-switch，直接问：correct RGB student evidence是否稳定优于
wrong/static/rotated/slot-broken/generic接口。

## 核心假设

若推理保留的`consumer_evidence[B,5,16]`确实携带可执行语义，则保持token、mask、presence、rho与模型
参数不变时，破坏图像归属、16维坐标、slot标签或mask–evidence binding应同时改变内部descriptor并降低
完整检索mAP；只把五个expert压成同一均值也应低于correct。反之，若这些控制与correct近似，则exp401的
小route贡献更可能来自generic residual或mask/context，不足以授权Phase-B formal mechanism。

## 冻结对象

- execution source=`11d7a35788c4645c355d96d76a2a4ff20a9801ac`，不得修改sealed repo；
- config SHA=`c2992bdf4321f906b19eb22dc7ec69a5678498ea0f93bf55a45a15a2e47cea84`；
- checkpoint=`transformer_120.pth`，SHA=
  `fe00d08a9a0f651c2c0852c0661e720995a65292459aec9797a359895aa52efc`；
- official Occluded-Duke query+gallery固定顺序，full reference raw=
  `57.1230075595/67.2850668430/80.2714943886/84.7511291504`；
- eval全程只读RGB；不得读取external pose、CLIP teacher、text或train target，不构造optimizer/
  scheduler/scaler，不backward，不保存checkpoint。

## 全局映射与缓存

correct arm第一次完整forward同时缓存每张validation RGB预测出的`consumer_evidence`和raw descriptor。
缓存严格按`query + gallery`的absolute dataset index与path SHA绑定，不按batch位置定义。

`wrong_rgb_evidence` donor在query与gallery各自内部、同camera组内，按固定dataset顺序寻找第一个不同PID
候选；必须满足same-split=`1`、same-camera=`1`、different-PID=`1`、fixed-point=`0`。应用时用absolute
index查全局cache，因此batch size或chunk边界变化不得改变recipient→donor映射或输出。

## 冻结arms

所有arm在同一model、同一loader顺序、同一进程内串行执行；每个arm结束立即恢复patch并核验state：

1. `correct`：原始RGB-only路径，同时建立evidence/descriptor cache；
2. `wrong_rgb_evidence`：只把evidence换成全局same-split/same-camera/different-PID donor，recipient的
   mask、presence、token和expert不变；
3. `static_zero_evidence`：evidence exact zero；Phase0E已证明centered slot-mean/global-only code为0；
4. `orthogonal_evidence`：用seed1234、局部CPU generator产生的canonical 16×16正交矩阵右乘evidence，
   保持每slot norm与pairwise cosine，破坏训练时坐标轴；
5. `evidence_slot_cycle`：只把五slot evidence循环`-1`，mask/presence/expert不变；
6. `wrong_mask_binding`：只把mask与presence循环`-1`，evidence与expert不变，并重算consumer field；
7. `generic_expert_mean`：两个router的五个expert均临时使用各自五权重均值，其他输入不变；
8. `bypass_router0`：router0 exact identity，router1保持correct；
9. `bypass_router1`：router1 exact identity，router0保持correct；
10. `all_router_bypass`：两个router均exact identity。

不加入teacher-oracle，因为正式eval不得读取teacher；不把batch内roll当wrong RGB；不调rho、loss、scale、
batch或任何parameter值来优化结果。

## 双层证据

每个arm必须记录：

- 完整mAP/R1/R5/R10；
- 相对correct的raw descriptor mean L2、max-abs、exact-equal count；
- 两个router实际call count；
- intervention count与absolute-index覆盖；
- model/checkpoint/config/source、patch、RNG、loader paths执行前后SHA/exact。

所有破坏性arm必须descriptor finite且相对correct `mean L2 > 0`、`max-abs > 0`；三种bypass必须覆盖
预期bank并产生非零descriptor差。descriptor变化只证明接口被触达，科学裁决仍以完整检索方向为准。

## 裁决门

### validity

1. correct与exp401 reference四项raw absolute error均`<=5e-8`；
2. all-bypass与exp401 final-audit reference四项raw absolute error均`<=5e-8`；
3. donor/global-index/chunk invariance、orthogonal、slot/binding、generic与bypass contract全部exact；
4. 所有arm full dataset覆盖、finite、descriptor active；
5. 241项state、checkpoint/config/source、RNG与patch执行前后exact，teacher-free、RGB-only；
6. 无NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow/AMP数值warning，进程退出且GPU空闲。

任一validity FAIL只封板当前execution；测量器runtime错误使exp402=`SEALED-INVALID`，必须新编号修正，
不得同编号补跑。

### scientific kill-switch

定义semantic controls=
`wrong_rgb_evidence/static_zero_evidence/orthogonal_evidence/evidence_slot_cycle/wrong_mask_binding/
generic_expert_mean`。只有同时满足：

1. `correct_mAP - max(semantic_control_mAP) >= +0.1 point`，即所有预注册semantic control都至少低
   `0.1 mAP`；
2. `correct_mAP - all_bypass_mAP >= +0.1 point`且correct mAP `>=56.7`，复核route alive；
3. 六个semantic control及三个bypass的descriptor差均active且方向与mAP裁决一致；
4. 两个single-router bypass都覆盖全dataset并产生独立非零descriptor差；

才判`PHASE_B_SEMANTIC_INTERFACE_IDENTIFIABLE / PHASE-B FORMAL MECHANISM DESIGN GO`。否则判
`CURRENT_SEMANTIC_INTERFACE_NO-GO`，只关闭当前student-evidence/expert解释，不永久否定Phase0E、Phase0R
或CLIP–TAPF，也不得靠调rho/loss/batch或删除失败control救活。

## 论文边界

PASS也只是同一checkpoint的干预可辨识性，不是多seed或新训练结果；它只授权下一编号formal mechanism
design/preflight。FAIL说明exp401的弱route贡献尚不能归到CLIP-owned semantic mediator。无论结果如何，
NFC、re-ranking或其他test-time trick均不得进入裁决。

## 封板结果

唯一formal full的全部validity门通过，correct与all-bypass逐项精确复现exp401 reference，10臂均完整覆盖
19,871张图且descriptor finite/active，所有patch/state/RNG/source/config/checkpoint恢复exact。route gap仍为
`+0.1194214838 mAP point`且correct=`57.1230075595 mAP`，因此exp401 route-alive结论保留。

但semantic kill-switch失败：六个control中最高的是wrong-RGB evidence=`57.1296975953 mAP`，比correct
高`0.0066900358 point`；zero evidence也高`0.0006964267 point`。预注册semantic margin实际为
`−0.0066900358 point`，远低于要求的`+0.1`。最终判定=
`CURRENT_SEMANTIC_INTERFACE_NO-GO / PHASE-B FORMAL MECHANISM DESIGN NO-START`。该结论只关闭当前
student-evidence/expert语义解释，不改写exp401 route alive，也不永久否定Phase0E、Phase0R或CLIP–TAPF。
