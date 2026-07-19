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
