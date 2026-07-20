# exp409 PCHM 监控

## 2026-07-21：设计冻结前审计

exp408 已按 `MECHANISM ORDER PASS / PERFORMANCE FAIL` 封板，4090 当前空闲。三个独立视角提出 PCOIR、PCHM、
PCSIC 与 PSRD；近期近邻审计认为继续 part KD/attention/router 会重复 PICRD、ProFD、PAFormer、MUVA 等对象。

当前选择 PCHM：它不再回归 CLIP embedding/relation，而是让 pose×CLIP 共同决定 final descriptor 原 triplet 的
离散 pair index，最直接回应 exp408“语义可学但未改变最终检索几何”的失败。PCOIR 暂不执行，因为 foreign-part
paste 容易退化成 pose-guided CutMix并引入 recipient 标签噪声；不得把 PCOIR 与 PCHM 组合。

代码映射智能体独立确认：exp408 的 supervision 位于 Stage-2 局部 Gram，而最终检索走 Stage-3→GAP
`global_feat`；PCHM 可在 `processor.py` 构造pair index，并由 `make_loss.py` 透传到 `TripletLoss` 对最终
distance matrix gather。默认 `None` 可保持 D0 exact。另一个智能体从实现侧否决 PCOIR：现有RGB已经
normalize/随机擦除，pose hard-owner是稀疏Gaussian区域，异PID part copy同时存在patch质量与partial-chimera
标签污染。

状态：`DESIGN FROZEN / IMPLEMENTATION NO-START / GPU IDLE`。下一步只做必要实现检查与一次独立代码盲审；
`0B/0H` 后立即构建 fresh exp409 cache并运行，不进行无穷 preflight。

首轮实现盲审发现`0B/1H`：cache整体SHA已绑定，但NPZ未内嵌逐图RGB SHA、CLIP checkpoint、pose manifest、
preprocess和source provenance，且builder未验证实际RGB SHA。已在任何cache/GPU执行前补为逐图image SHA验证、
完整metadata写入和loader严格校验；等待同一盲审者聚焦闭环。

同一独立审查者聚焦复审确认原HIGH闭环，最终=`0 BLOCKER / 0 HIGH`。rank方向、候选mask、Borda/tie-break、
pair gather梯度、default-off exact及AMP接线均未发现B/H；紧凑contract还覆盖cache provenance roundtrip和错误
image SHA拒绝。当前=`IMPLEMENTED / BLIND REVIEW 0B/0H / REMOTE MMPOSE CONTRACT NEXT / GPU IDLE`。

远端fresh隔离repo=`/home/afr/SOLIDER-REID-exp409-pchm-d2cc18f-v1`，固定source HEAD=
`07170a22f1a6fbef2f0f140106e7350c223deb29`且clean。MMPOSE-ABU contract在CUDA上PASS：batch64，
default loss/gradient bit-exact，错误positive与cache image-SHA mutant均被拒；correct相对control的pair change rate=
wrong-RGB `0.9375`、generic `0.96875`、zero/CLIP-only `0.875`、pose-shuffle `0.90625`。这证明联合miner与
五个control均active，不是D0路径漂移；下一步直接构建唯一fresh exp409 cache。

唯一fresh cache-v1已启动：remote repo=`/home/afr/SOLIDER-REID-exp409-pchm-d2cc18f-v1`，source HEAD=
`07170a22f1a6fbef2f0f140106e7350c223deb29`，asset=
`/home/afr/reid-clean/assets/exp409-pchm-cache-v1`，runner=
`/home/afr/reid-clean/train-logs/exp409-pchm-cache-v1.runner.log`，主PID=`501962`。启动前GPU无compute PID、
asset/runner均fresh；首次观测`encoded 8/15618`，GPU约`2.2 GiB/94%`且只有该任务，无异常。运行中只监控
自然完成，不改builder/source/参数；完成后核验覆盖、norm、metadata、SHA并把真实cache SHA写入config。

等待cache期间只在本地准备一次性real-batch执行器。独立盲审首轮`1B/0H`：Stage-3参数筛选误用
`layers.3`而实际注册为`base.stages.3.*`，会造成假失败；已在任何执行前精确修正，等待聚焦复审。cache运行
source与参数未修改。

real-batch执行器聚焦复审最终`0B/0H`。cache最新自然推进至`11000/15618`，主PID=`501962`，GPU约2.2GiB且
无异常；仍只监控完成，不同步或修改运行中的远端repo。

唯一fresh cache-v1已自然完成并退出，GPU恢复`2 MiB/0%/0 compute PID`。完整核验PASS：15,618图完整唯一
覆盖，shape=`[15618,5,768]` FP16，五槽valid=`[15616,15618,15618,15618,15586]`，所有feature finite，
valid feature L2 norm在冻结容差内；逐图image SHA、pose/CLIP/preprocess/source/builder/teacher provenance及
64图snapshot均通过。cache SHA=`d502a0f03fe556284fd01259ed81143dcfb171855b9b2aebaa29e3b7a682fd36`，
snapshot SHA=`2c34567396c057d65cc5cb40bc18e7001c2069b55cf2793ed0eaf5f74675bbf8`，builder/teacher
source SHA=`8fe06de77f5f8256f31a572c577c473ed699303c9b3dee2e2e4a507e6df74e59`/
`fbd3e137a729f44d3179864f9978bd8846b22e8627a3c311747b0a2541092864`。真实cache SHA现已冻结进config；
下一步同步已审real-batch执行器并执行唯一检查。

real-batch v1完成真实loader/model/forward/loss后，在reporter于`unscale_`前检查scaled Stage-3梯度处抛错；没有
调用`scaler.step/update`，optimizer update为0，GPU已释放。该次冻结为
`REAL-BATCH V1 INVALID CHECKER / MODEL SCIENCE NOT EVALUATED`，不重跑。根因是reporter违背native AMP语义，
不能据此声称PCHM或D0数值失败。

fresh v2只把测量改为：未缩放descriptor gradient、`unscale_`后参数report、default GradScaler自然skip/backoff，
固定同一真实batch最多8 attempts，第一且唯一成功update后停止。禁止覆盖scaler初值或改loss/batch/pair；当前等待
独立聚焦盲审，GPU空闲。

v2首轮盲审发现`1B/0H`：最初只统计base nonfinite，而GradScaler扫描完整optimizer，可能把classifier-only
overflow的正确skip误报为finite failure。已在执行前改成全model nonfinite并以scale下降作为native overflow
权威判据；等待聚焦闭环。

v2聚焦复审最终`0B/0H`，只授权一个fresh v2 execution；不得重跑v1。GPU空闲，立即同步固定源码并执行。
