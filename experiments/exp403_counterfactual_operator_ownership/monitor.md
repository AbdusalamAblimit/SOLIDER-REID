# exp403 ELO-CUR 监控记录

## 2026-07-20 接手与封板复核

- local HEAD=`7d880d2e843d3bb431f87515ba245eea3526b344`，tracked clean；
- remote exp401 HEAD=`11d7a35788c4645c355d96d76a2a4ff20a9801ac`，tracked clean；
- formal config SHA=`c2992bdf4321f906b19eb22dc7ec69a5678498ea0f93bf55a45a15a2e47cea84`；
- 唯一checkpoint=`transformer_120.pth`，SHA=`fe00d08a9a0f651c2c0852c0661e720995a65292459aec9797a359895aa52efc`；
- train/eval process=`0`，GPU=`2 MiB / 0%`；
- exp401/402均不重跑、不补跑。

判断：封板状态与 heartbeat 一致，允许只读研究与本地 CPU 设计，不允许 GPU。

## 2026-07-20 targeted literature/code audit

已审计 CAL、AIM、UCT、Instruct-ReID legacy dual causality，并对照 PGMAN/CIFT、SGFNet 及通用
dynamic-filter/hypernetwork/LoRA先例。结论不是原子首创，而是当前未发现
`evidence-owned operator + matched complete-execution ordering + frozen retrieval controls`同构闭环。

创新门槛：问题/机制/证据=`3/3`，条件允许进入 exp403 design与CPU contract；novelty风险=`6/10`。

当前状态：`GPU NO-START`。

## 2026-07-20 standalone CPU/static contract

`static_cpu_contract.py` 在本地 `.venv`/uv 环境连续执行两次，均为 `26/26 PASS`，两个 result逐字节一致：

- result SHA256=`041cd6d26f1e3469478c902d443f0f211fd329e9f6a89ffc2e85fcef818b4df5`；
- source SHA256=`b0f40b015150942f12b099e54de406faf63baf787e2fd74cc80cca4706a8eefe`；
- compatibility correct/wrong/generic/NULL=`0.9877629876/0.5/0/−1`；
- ordinal gaps=`0.4877629876/0.5/1.0`，hinge=`0`；
- correct evidence grad norm=`1.8288789e-02`；
- wrong/generic/NULL reference evidence grad norm=`0/0/0`；
- `H/U/V/C/Q/K` 六组参数梯度全部finite且非零；
- NULL逐元素exact identity，correct/wrong/generic descriptor均active/distinct；
- same-camera/different-PID donor映射完整、确定且重复exact；
- evidence-ignored、aux-only、reference-not-detached三个mutant全部被抓住；
- CUDA在执行前后均未初始化。

判断：`STANDALONE STATIC-CPU PASS / PRODUCTION IMPLEMENTATION GO / CUDA NO-START`。这只证明数学和
autograd contract可执行，不证明真实 Swin/AMP/检索有效。下一步先在新 config开关下实现生产图及 off-parity/
source CPU contract；未通过前不启动4090。

## 2026-07-20 生产实现前的梯度合同澄清

复核发现，三个reference全部stop-gradient时，`wrong>generic>NULL`两个reference-reference hinge不可能提供
训练梯度；若开放它们的梯度，又会违反“不靠破坏control制造margin”。因此生产目标冻结为
`correct-max(stopgrad(wrong,generic,NULL))`单边compatibility hinge，后两段顺序只记录为诊断。该修订不
改变standalone的26项结果、final retrieval门或ELO结构，只消除一个无梯度伪目标。

判断：允许继续生产实现；GPU仍为`NO-START`。

## 2026-07-20 生产 CPU/source 门

按用户要求不再扩张重复 CPU 矩阵，只执行一次必要生产合同。结果`34/34 PASS`：默认关闭时 D0/C0相对
实现前commit=`0722176`的state、初始化RNG与输出逐tensor exact；ELO无slot expert、六组linear无bias、
NULL exact identity；mini-Swin三个Stage-3 reference完整no-grad重放，correct输出和全局RNG相对correct-only
exact；student evidence与12组共享生产参数梯度均finite/nonzero；strict reload、optimizer覆盖、teacher/generic-
free state和generic资产SHA/metadata正反校验全部通过。CUDA未初始化。

result=`production_cpu_result.json`，当前判定：`PRODUCTION CPU PASS / FRESH ASSET + CUDA PREFLIGHT GO / FORMAL NO-START`。

## 2026-07-20 generic asset-v1 运行入口无效

fresh asset-v1在导入阶段因子目录启动未注入repo root而`ModuleNotFoundError: datasets`；GPU始终
`2 MiB/0%`，official数据访问0，generic输出0。该记录封板为`ASSET-V1 INVALID`，不代表机制失败；只允许
asset-v2修正模块入口，资产内容、teacher、数据与聚合合同保持不变。

asset-v2在错误的Phase0E Python环境导入`model`时因缺OpenCV退出；同样未初始化GPU、未访问数据且输出0，
封板为`ASSET-V2 INVALID`。随后不改脚本，改用exp394–exp401冻结链的canonical OpenCLIP+ReID runtime
执行fresh asset-v3。

asset-v3自然覆盖official train=`15,618`图后PASS，五slot valid count=
`15,615/15,618/15,618/15,617/15,578`，generic mean norm=
`0.0900701/0.0885263/0.0812544/0.0593401/0.0758309`，均finite/nonzero；peak CUDA=
`1,712,272,384 bytes`，异常0，进程退出且GPU恢复`2 MiB/0%`。generic/result runner SHA256=
`dc2dfe9e1fd00b6a8b374eb4f6894f1dc6c7680df6d00540cbea37e9b5ae431d`/
`a21b1c2e3f06b687c0940f10180d72ca9a4f39b9b2c8b097340160fe669faaad`。

判断：`FRESH GENERIC ASSET PASS / ACTUAL BATCH64 CUDA/AMP PREFLIGHT GO / FORMAL NO-START`。

## 2026-07-20 CUDA/AMP preflight 与 formal 启动

真实 batch64 CUDA/AMP preflight 已自然退出，`16/16 PASS`，result 明确
`formal_training_authorized=true`。默认 GradScaler 前四次只发生自然 scale backoff，第5次成功更新；12组共享
生产参数梯度、correct evidence 梯度均 finite/nonzero，reference no-grad、RNG exact、eligible ratio=`1.0`，
rho@e6=`0.0161510899663`，CUR=`0.0499977395`，checkpoint=`0`。遵照用户要求，不再增加 CPU 或诊断矩阵。

启动前最后核对：remote HEAD=`fe854ea0808d86d37566100e59ea629e8b409d38`且 tracked clean，config SHA256=
`06a80c9d7589fe539b4d8f5820df307f08868c2005517ae6fce3226ed8a470ba`，formal output为空、runner不存在、
GPU无compute process。随后直接启动唯一 fresh seed1234/batch64/e120，main PID=`423319`、parent=`1`、
8 workers，GPU唯一PID且约`8968 MiB`，checkpoint=`0`。

e1 Iter60：Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`16.972/1.424/0.506/0.658/0.735/0.990/0.320/0.285/0.050`；eligible=`1.000`，CoeffStd=
`1.555e-01`，EffRank=`15.690`，RNGExact=`1`，Student=`0`，Reliability=`0.997`，rho=`0`，
BudgetAbs=`0`，finite。当前判定：`FORMAL RUNNING`；只按冻结协议监控，不早停、不续训、不改 recipe。

## 2026-07-20 formal heartbeat：e2

已进入e2 Iter160。Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`9.534/1.404/0.489/0.647/0.705/0.993/0.319/0.220/0.050`；eligible=`0.953`，CoeffStd=
`1.535e-01`，EffRank=`15.605`，RNGExact=`1`，Student=`0`，Reliability=`0.997`，rho=`0`，
BudgetAbs=`0`，finite。remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、
GPU唯一任务，fatal/AMP数值warning=`0/0`，checkpoint=`0`。判断：继续自然运行，不作中间裁决。

## 2026-07-20 formal heartbeat：e9

已进入e9 Iter180。Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`5.900/1.231/0.425/0.547/0.558/0.994/0.296/0.102/0.050`；Compat C/W/G/N=
`0.4998/0.5015/-0.0404/-1.0000`，eligible=`1.000`，CoeffStd=`1.512e-01`，EffRank=`15.641`，
RNGExact=`1`，Student=`0.80`，Reliability=`0.999`，rho=`0.064604360`，BudgetAbs=`3.232e-04`，finite。
remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、GPU唯一任务，fatal/AMP数值
warning=`0/0`，checkpoint=`0`。e10正式eval尚未产生；继续自然运行，不按训练期compatibility作科学裁决。

## 2026-07-20 formal heartbeat：e10 eval / e16

首个完整正式评测e10 mAP/R1/R5/R10=`34.4/44.0/60.2/67.0`，仅记录、不裁决。当前e16 Iter140：
Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`3.305/1.002/0.371/0.393/0.354/0.993/0.338/0.100/0.050`；Compat C/W/G/N=
`0.7302/0.7305/-0.1047/-1.0000`，eligible=`0.984`，CoeffStd=`1.522e-01`，EffRank=`15.565`，
RNGExact=`1`，Student=`1`，Reliability=`1`，rho=`0.080755450`，BudgetAbs=`5.681e-04`，finite。
remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、GPU唯一任务，fatal/AMP数值
warning=`0/0`，checkpoint=`0`。判断：继续自然运行，不按e10或训练期compatibility早停。

## 2026-07-20 formal heartbeat：e20 eval / e27

完整正式评测e20 mAP/R1/R5/R10=`40.8/51.0/66.9/74.5`，仅记录、不裁决。当前e27 Iter100：
Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`0.966/0.806/0.311/0.243/0.157/0.988/0.329/0.100/0.050`；Compat C/W/G/N=
`0.8434/0.8424/-0.0814/-1.0000`，eligible=`1.000`，CoeffStd=`1.475e-01`，EffRank=`15.617`，
RNGExact=`1`，Student=`1`，Reliability=`1`，rho=`0.080755450`，BudgetAbs=`3.118e-04`，finite。
remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、GPU唯一任务，fatal/AMP数值
warning=`0/0`，checkpoint=`0`。判断：继续自然运行，不按e20或训练期compatibility早停。

## 2026-07-20 formal heartbeat：e30 eval / e34

完整正式评测e30 mAP/R1/R5/R10=`45.6/54.7/70.5/76.0`，仅记录、不裁决。当前e34 Iter120：
Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`0.467/0.770/0.295/0.207/0.107/0.989/0.319/0.100/0.050`；Compat C/W/G/N=
`0.8699/0.8697/-0.0684/-1.0000`，eligible=`1.000`，CoeffStd=`1.463e-01`，EffRank=`15.621`，
RNGExact=`1`，Student=`1`，Reliability=`1`，rho=`0.080755450`，BudgetAbs=`2.307e-04`，finite。
remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、GPU唯一任务，fatal/AMP数值
warning=`0/0`，checkpoint=`0`。判断：继续自然运行，不按e30或训练期compatibility早停。

## 2026-07-20 formal heartbeat：e40 eval / e41

完整正式评测e40 mAP/R1/R5/R10=`49.5/59.8/74.6/80.8`，仅记录、不裁决。当前e41 Iter80：
Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`0.393/0.755/0.286/0.191/0.082/0.983/0.311/0.100/0.050`；Compat C/W/G/N=
`0.8882/0.8875/-0.0379/-1.0000`，eligible=`1.000`，CoeffStd=`1.454e-01`，EffRank=`15.623`，
RNGExact=`1`，Student=`1`，Reliability=`1`，rho=`0.080755450`，BudgetAbs=`1.947e-04`，finite。
remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、GPU唯一任务，fatal/AMP数值
warning=`0/0`，checkpoint=`0`。判断：继续自然运行，不按e40或训练期compatibility早停。

## 2026-07-20 formal heartbeat：e48

当前e48 Iter100。Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`0.288/0.747/0.282/0.182/0.069/0.986/0.306/0.100/0.050`；Compat C/W/G/N=
`0.8966/0.8967/-0.0036/-1.0000`，eligible=`1.000`，CoeffStd=`1.450e-01`，EffRank=`15.630`，
RNGExact=`1`，Student=`1`，Reliability=`1`，rho=`0.080755450`，BudgetAbs=`1.658e-04`，finite。
remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、GPU唯一任务，fatal/AMP数值
warning=`0/0`，checkpoint=`0`。e50正式eval尚未产生；继续自然运行，不作中间裁决。

## 2026-07-20 formal heartbeat：e50 eval / e55

完整正式评测e50 mAP/R1/R5/R10=`52.6/63.2/76.2/81.3`，仅记录、不裁决。当前e55 Iter40：
Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`0.250/0.738/0.277/0.176/0.059/0.978/0.301/0.100/0.050`；Compat C/W/G/N=
`0.9049/0.9048/0.0021/-1.0000`，eligible=`0.984`，CoeffStd=`1.449e-01`，EffRank=`15.641`，
RNGExact=`1`，Student=`1`，Reliability=`1`，rho=`0.080755450`，BudgetAbs=`1.480e-04`，finite。
remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、GPU唯一任务，fatal/AMP数值
warning=`0/0`，checkpoint=`0`。判断：继续自然运行，不按e50或训练期compatibility早停。

## 2026-07-20 formal heartbeat：e60 eval / e62

完整正式评测e60 mAP/R1/R5/R10=`53.0/63.0/76.9/82.4`，仅记录、不裁决。当前e62 Iter20：
Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`0.195/0.745/0.279/0.172/0.057/0.993/0.301/0.100/0.050`；Compat C/W/G/N=
`0.9105/0.9101/0.0057/-1.0000`，eligible=`0.969`，CoeffStd=`1.452e-01`，EffRank=`15.662`，
RNGExact=`1`，Student=`1`，Reliability=`1`，rho=`0.080755450`，BudgetAbs=`1.259e-04`，finite。
remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、GPU唯一任务，fatal/AMP数值
warning=`0/0`，checkpoint=`0`。判断：继续自然运行，不按e60或训练期compatibility早停。

## 2026-07-20 formal heartbeat：e69

当前e69 Iter60。Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`0.188/0.734/0.275/0.170/0.049/0.986/0.296/0.100/0.050`；Compat C/W/G/N=
`0.9161/0.9164/0.0126/-1.0000`，eligible=`1.000`，CoeffStd=`1.454e-01`，EffRank=`15.665`，
RNGExact=`1`，Student=`1`，Reliability=`1`，rho=`0.080755450`，BudgetAbs=`1.234e-04`，finite。
remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、GPU唯一任务，fatal/AMP数值
warning=`0/0`，checkpoint=`0`。e70正式eval尚未产生；继续自然运行，不作中间裁决。

## 2026-07-20 formal heartbeat：e70 eval / e76

完整正式评测e70 mAP/R1/R5/R10=`54.5/64.7/78.1/83.4`，仅记录、不裁决。当前e76 Iter20：
Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`0.180/0.729/0.273/0.168/0.046/0.979/0.296/0.100/0.050`；Compat C/W/G/N=
`0.9191/0.9189/0.0164/-1.0000`，eligible=`1.000`，CoeffStd=`1.455e-01`，EffRank=`15.668`，
RNGExact=`1`，Student=`1`，Reliability=`1`，rho=`0.080755450`，BudgetAbs=`1.205e-04`，finite。
remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、GPU唯一任务，fatal/AMP数值
warning=`0/0`，checkpoint=`0`。判断：继续自然运行，不按e70或训练期compatibility早停。

## 2026-07-20 formal heartbeat：e80 eval / e82

完整正式评测e80 mAP/R1/R5/R10=`55.5/65.6/78.8/84.1`，仅记录、不裁决。当前e82 Iter200：
Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`0.155/0.735/0.275/0.168/0.046/0.988/0.296/0.100/0.050`；Compat C/W/G/N=
`0.9203/0.9205/0.0171/-1.0000`，eligible=`1.000`，CoeffStd=`1.454e-01`，EffRank=`15.676`，
RNGExact=`1`，Student=`1`，Reliability=`1`，rho=`0.080755450`，BudgetAbs=`1.210e-04`，finite。
remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、GPU唯一任务，fatal/AMP数值
warning=`0/0`，checkpoint=`0`。判断：继续自然运行，不按e80或训练期compatibility早停。

## 2026-07-20 formal heartbeat：e89

当前e89 Iter200。Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`0.153/0.734/0.274/0.167/0.044/0.988/0.296/0.100/0.050`；Compat C/W/G/N=
`0.9226/0.9223/0.0263/-1.0000`，eligible=`1.000`，CoeffStd=`1.454e-01`，EffRank=`15.676`，
RNGExact=`1`，Student=`1`，Reliability=`1`，rho=`0.080755450`，BudgetAbs=`1.128e-04`，finite。
remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、GPU唯一任务，fatal/AMP数值
warning=`0/0`，checkpoint=`0`。e90正式eval尚未产生；继续自然运行，不作中间裁决。

## 2026-07-20 formal heartbeat：e90 eval / e96

完整正式评测e90 mAP/R1/R5/R10=`56.5/66.9/80.0/84.6`，仅记录、不裁决。当前e96 Iter160：
Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`0.146/0.732/0.273/0.166/0.043/0.985/0.295/0.100/0.050`；Compat C/W/G/N=
`0.9225/0.9226/0.0278/-1.0000`，eligible=`0.969`，CoeffStd=`1.457e-01`，EffRank=`15.686`，
RNGExact=`1`，Student=`1`，Reliability=`1`，rho=`0.080755450`，BudgetAbs=`1.168e-04`，finite。
remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、GPU唯一任务，fatal/AMP数值
warning=`0/0`，checkpoint=`0`。判断：继续自然运行，不按e90或训练期compatibility早停。

## 2026-07-20 formal heartbeat：e100 eval / e103

完整正式评测e100 mAP/R1/R5/R10=`56.4/66.3/78.9/84.0`，仅记录、不裁决。当前e103 Iter140：
Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`0.146/0.731/0.273/0.166/0.042/0.984/0.294/0.100/0.050`；Compat C/W/G/N=
`0.9225/0.9226/0.0356/-1.0000`，eligible=`1.000`，CoeffStd=`1.457e-01`，EffRank=`15.685`，
RNGExact=`1`，Student=`1`，Reliability=`1`，rho=`0.080755450`，BudgetAbs=`1.135e-04`，finite。
remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、GPU唯一任务，fatal/AMP数值
warning=`0/0`，checkpoint=`0`。判断：继续自然运行，不按e100或训练期compatibility早停。

## 2026-07-20 formal heartbeat：e110

当前e110 Iter140。Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`0.143/0.729/0.272/0.166/0.041/0.983/0.293/0.100/0.050`；Compat C/W/G/N=
`0.9229/0.9226/0.0343/-1.0000`，eligible=`1.000`，CoeffStd=`1.455e-01`，EffRank=`15.680`，
RNGExact=`1`，Student=`1`，Reliability=`1`，rho=`0.080755450`，BudgetAbs=`1.130e-04`，finite。
remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、GPU唯一任务，fatal/AMP数值
warning=`0/0`，checkpoint=`0`。e110正式eval尚未产生；余约10 epoch，继续自然跑满，不作中间裁决。

## 2026-07-20 formal heartbeat：e110 eval / e117

完整正式评测e110 mAP/R1/R5/R10=`56.7/66.6/79.5/83.8`，仅记录、不裁决。当前e117 Iter160：
Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`0.141/0.732/0.273/0.166/0.041/0.984/0.296/0.100/0.050`；Compat C/W/G/N=
`0.9256/0.9256/0.0350/-1.0000`，eligible=`0.969`，CoeffStd=`1.454e-01`，EffRank=`15.682`，
RNGExact=`1`，Student=`1`，Reliability=`1`，rho=`0.080755450`，BudgetAbs=`1.109e-04`，finite。
remote HEAD/config SHA/tracked-clean均exact；唯一main PID=`423319`、8 workers、GPU唯一任务，fatal/AMP数值
warning=`0/0`，checkpoint=`0`。余约3 epoch，继续自然跑满；结束前不启动反事实评测。

### final frozen counterfactual执行器就绪

复用exp402已封板的全量检索骨架，冻结exp403七臂为correct、same-split/same-camera different-PID wrong-RGB、
generic、NULL、slot-cycle、wrong-mask与all-router-bypass；不新增control。终审额外锁定ELO shared projections、
无static experts、checkpoint strict/finite/teacher-free、RGB-only、逐臂完整覆盖与state/RNG/patch恢复，并按
D0 mAP/R1 floor、correct-max(wrong/generic/NULL)和correct-all-bypass三个正式门裁决。

执行器已部署到fresh只读目录`/home/afr/reid-clean/audits/exp403_elo_cur_final_v1`，7个源文件语法、shell与
动态base patch contract均PASS；generic 5x16常量与fresh asset逐值exact。contract/result/runner/manifest均
尚未创建，GPU评测未启动。训练自然结束和退出终审PASS后才允许启动该once-only执行。

## 2026-07-20 formal e120自然完成与final audit启动

唯一fresh seed1234训练自然跑满e120；main PID=`423319`及8 workers均自然退出，GPU恢复`2 MiB/0%`。
e120 Iter200 Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Compat/CUR=
`0.135/0.734/0.274/0.166/0.042/0.988/0.296/0.100/0.050`；Compat C/W/G/N=
`0.9243/0.9247/0.0259/-1.0000`，eligible=`0.969`，CoeffStd=`1.456e-01`，EffRank=`15.679`，
RNGExact=`1`，Student/Reliability=`1/1`，rho=`0.080755450`，BudgetAbs=`1.118e-04`，finite。
e120训练内正式评测mAP/R1/R5/R10=`57.0/67.4/79.7/84.1`，仍只记录，raw由final audit裁决。

唯一checkpoint=`transformer_120.pth`，SHA256=
`c5593cd73b06cde4ec3306b3458617d8835fa529b77c545923e874e7e780cc71`。冻结checkpoint contract
`15/15 PASS`：state count=`237`、state SHA256=
`ad08a670330b48c5028359131793796934acdd0b7ae53a98eff630db162f50f9`，全部tensor finite、teacher-free，
证据头和两个ELO router完整、static experts缺失，source/config/repo exact，训练进程与GPU compute均为0。

随后串行启动唯一7臂全量RGB-only final audit；wrapper PID=`430930`，唯一GPU audit PID=`430932`，初始显存
约`7924 MiB`，strict model load成功。当前判定：`FORMAL TRAINING COMPLETE / CHECKPOINT CONTRACT PASS /
FINAL COUNTERFACTUAL RUNNING`。

## 2026-07-20 final counterfactual自然完成与封板

唯一7臂全量RGB-only审计自然退出，每臂完整覆盖`19,871`图/`78` batches。七臂raw结果为：

| arm | mAP | R1 | R5 | R10 | correct−arm mAP point | descriptor mean L2 / max-abs |
|---|---:|---:|---:|---:|---:|---:|
| correct | 56.992955931509 | 67.420816421509 | 79.728507995605 | 84.072399139404 | 0 | reference |
| wrong RGB evidence | 56.993435832959 | 67.420816421509 | 79.728507995605 | 84.072399139404 | −0.000479901450 | 0.001668183133 / 0.000701904297 |
| generic evidence | 56.993713150692 | 67.420816421509 | 79.728507995605 | 84.072399139404 | −0.000757219183 | 0.035066194832 / 0.008772373199 |
| NULL zero evidence | 56.993730466937 | 67.420816421509 | 79.728507995605 | 84.072399139404 | −0.000774535428 | 0.035825069994 / 0.008928775787 |
| evidence slot cycle | 56.991975523448 | 67.420816421509 | 79.728507995605 | 84.072399139404 | +0.000980408061 | 0.006056237034 / 0.001584291458 |
| wrong mask binding | 56.992356510393 | 67.420816421509 | 79.728507995605 | 84.072399139404 | +0.000599421116 | 0.005656813271 / 0.001407384872 |
| all-router-bypass | 56.993730466937 | 67.420816421509 | 79.728507995605 | 84.072399139404 | −0.000774535428 | 0.035825069994 / 0.008928775787 |

wrong-RGB donor严格满足same-split/same-camera/different-PID=`1/1/1`且无fixed point。所有六个干预臂
`exact_equal_rows=0`，descriptor均finite/active；两个router在每臂各执行`78` batches，逐臂state、RNG、
loader RNG、prepare patch与router patch恢复exact。strict reload、237项finite/teacher-free state、
ELO shared projections、无static experts、RGB-only、source/config/checkpoint before-after exact、
teacher/pose forbidden access=`0`均PASS。runner异常计数全部为0，audit exit=`0`，进程退出，GPU回到
`2 MiB/0%`。postflight `7/7 PASS`，所以这是有效测量，不是runtime或patch INVALID。

四个正式科学门全部失败：correct raw mAP/R1=`0.569929559315091/0.674208164215088`，低于冻结D0门
`0.575587756578/0.676923076923`；correct相对`max(wrong,generic,NULL)`的raw mAP margin为
`−7.745354277944e-06`（`−0.000774535428 point`），相对all-bypass亦为同值，均未达到
`+0.001` raw mAP（`+0.1 point`）。尤其NULL与all-bypass的raw metrics逐项完全相同，七臂R1/R5/R10也
逐项完全相同。

训练期compatibility/CUR proxy虽持续有限且活跃，却没有转化为final retrieval ownership。
descriptor干预活跃但排名近乎不变，说明当前ELO-CUR学到的是可执行的shortcut/proxy，而不是“正确图像
evidence拥有检索算子”的身份条件。

唯一checkpoint SHA256=`c5593cd73b06cde4ec3306b3458617d8835fa529b77c545923e874e7e780cc71`，
state SHA256=`ad08a670330b48c5028359131793796934acdd0b7ae53a98eff630db162f50f9`。
result/runner/checkpoint-contract/manifest SHA256=
`baf4a016d008f4a86ac26d9ff78524ecc30d46071392dad8a8d151aaf05063cb`/
`af1872d65d8dabc9d51f317465a3d60d4f141a6c79db84c2751d43cc964b26eb`/
`7af4b3bcf49d3cf33f683a385dd94fdb145b971bed7d245c7d089867cffbf217`/
`891588308683ce99a363613c0f94c724b367323579fc55ad7b922994a46d329a`。

**最终判定**：`SEALED / VALIDITY PASS / SCIENTIFIC ELO_CUR_MECHANISM_NO_GO`，
`phase_b_formal_mechanism_design_authorized=false`。exp403禁止重跑、补跑、续训、换seed或通过调
rho/loss/batch/stage、mask及删除不利control救活。exp401的route-alive窄幅边界保持；exp402关闭旧C0
student-evidence/expert semantic解释，exp403进一步关闭当前evidence-owned low-rank operator + CUR对象。
下一候选必须重新定义问题或结构对象并重新通过创新门槛，不得围绕ELO-CUR做尺度或损失变体。
