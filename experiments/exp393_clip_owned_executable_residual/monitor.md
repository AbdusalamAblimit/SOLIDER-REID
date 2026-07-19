# exp393 监控记录

## 当前状态

- `DESIGN-ONLY / PHASE 0E NO-START / PHASE A/B FORMAL TRAINING NO-START`；
- exp392 Phase 0C/0D已封板，禁止续训、重跑或修改其repo/config/checkpoint；
- semantic multi-stage保持NO-START；
- 当前远端无计算/训练进程，4090应为`2 MiB/0%`；
- 未创建exp393训练repo、config、output、runner或checkpoint。

## 2026-07-19 设计冻结前审查

exp393把下一步拆成两道单变量门禁：

1. Phase A仅把Semantic C0的zero expert改为非零branch+zero ReZero scalar，验证route activation；
2. Phase B仅在Phase A route alive后，把scalar q执行变量换成centered rich CLIP evidence code并加入
   router必经latent的内部alignment。

该顺序避免把“路由优化修复”和“CLIP teacher信息增量”混成一个bundled差值。Phase 0E teacher-only
必须先证明rich code具有within-slot动态、有效秩及correct-vs-wrong敏感性；当前不占用4090。

下一步先做只读代码seam审计和Phase 0E synthetic/8图脚本设计。任何正式训练前必须完成独立
config/diff/gradient ownership/RNG/optimizer/checkpoint/RGB-only/CUDA/AMP preflight；禁止因用户要求
继续CLIP方向而跳过门禁。

## 2026-07-19 代码seam与远端状态复核

远端真实状态：4090=`2 MiB/0%`，无`train.py`、Phase 0或exp392/393审计进程；exp392正式repo仍为
exact HEAD=`ed5783416528be4284adce11fa192fe119e344f4`且tracked clean，唯一checkpoint仍为
`transformer_120.pth`。没有重启、续训或修改封板资产。

只读代码审查确认Phase 0D归因与实现一致：Semantic C0的support/presence/mask heads从zero logits
起步；consumer mask/support在进入router前detach；router token/context projection为非零随机初始化，
但expert exact-zero，因此首步只有expert可从ReID loss离零，token/context必须等expert非零后才有
梯度。正式执行路径的AMP optimizer step只有一次；早先合并显示出的“双step”是截断输出拼接造成的
误读，已用本地行号、git blame和远端execution HEAD三方逐行排除，不构成exp392结果bug。

设计审查发现并修正一个真实计算图问题：旧稿把`L_exec`写在expert/alpha之前，却声称它能更新
ReZero branch。冻结后的修正版改为对真实pre-alpha branch proposal做teacher relation alignment，使用
共享生产参数和detached-token重算，使梯度实际到达token/context/evidence projection与expert；alpha
仍只由ReID loss打开。Phase 0E失败现在只阻断Phase B，不再无关地否决Phase A的route activation诊断。

## 2026-07-19 Phase 0E-S synthetic exact封板

新增`phase0e_static_contract.py`并在远端`CUDA_VISIBLE_DEVICES=`纯CPU执行，未加载CLIP、ReID数据、
model或optimizer。结果`SEALED-PASS`：PID fit/audit严格disjoint且repeat exact；fit/audit image=
`26/54`；五slot中心化误差最大`4.2701e-17`，shared PCA basis正交误差最大`1.9984e-15`；invalid
code exact zero，valid unit-norm误差最大`2.2204e-16`，hard-owner pairwise product与slot-cycle IoU
均exact zero；synthetic correct-positive相对wrong RGB/wrong mask margin=`+0.72342/+0.86266`，finite。

远端冻结路径：`/home/afr/reid-clean/audits/exp393_phase0e/`。script SHA256=
`6c1b370912f5f668ce117d4320d62b68a032549ff06821f5bee1ae020acb3dab`，result SHA256=
`120085ddffdea2d18adfd73a856426229bfb132218e79fef6e0dc318d49c23ac`。执行后无残留进程，4090=
`2 MiB/0%`。该PASS只授权0E-C8真实8图contract，不裁决teacher统计有效性，不授权训练或Phase B。

## 2026-07-19 Phase 0E-C8启动前兼容性退出

首版C8脚本在进入OpenCLIP构造/forward前调用`torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))`，
远端PyTorch 2.6立即报`RuntimeError: Invalid device argument`并退出；result不存在、GPU全程
`2 MiB/0%`、无残留进程。该退出没有产生任何teacher观测，不能裁决Phase 0E。旧script SHA256=
`aeaedb9cc698ac9e5afc685e17f314c1b30bf16fc8ae2adfcc8e5ed39efe5276`，runner SHA256=
`7139124e9bcbf483ee69e976b48a5a8e09b6bbf8ebea7a8d547f214c77cf1497`。

归因是CUDA memory API在该runtime要求整数device index。修复仅把reset/max-memory/synchronize参数
规范为显式整数索引，不改变样本、teacher、mask、PCA、counterfactual或任何门禁；静态编译与SHA
复核后允许一次真实C8执行。这不是训练重跑或性能重复。

整数索引版仍在context初始化前的同一行退出，第二个runner SHA256=
`b1ae33b5c1450d62405bc75bf747aa6919e02924f6f8a249025740d97922f5b0`，仍无result或teacher观测。
随后最小runtime probe确认CUDA available、device count=`1`、current=`0`、设备为RTX 4090 D；显式
`torch.cuda.init()`后，无参`reset_peak_memory_stats()`与`max_memory_allocated()`均正常且peak=`0`。
最终兼容修正固定为先`set_device+init`，再使用当前device的无参memory/synchronize API；其余执行
图与门禁不变。

切换到实际含OpenCLIP的已存在uv环境前，`/home/afr/reid-clean/.venv`在teacher构造入口因
`ModuleNotFoundError: open_clip`退出；runner SHA256=
`bcb288e69038cd7989f38ac80a08471da430aa24b644d8a3329d990fdf84663a`，仍无模型forward/result。
只读环境审计找到`/home/afr/par2606/.venv`，版本为torch `2.6.0+cu124`、open_clip `3.3.0`，没有
安装或修改任何包。后续C8固定使用该已验证环境。

## 2026-07-19 Phase 0E-C8真实8图contract封板

0E-C8在唯一4090上自然完成并`SEALED-PASS`。固定8图来自8个不同PID，fit/audit=`5/3`；official
global tail parity max-abs=`0`，repeat、NULL output/valid exact；hard-owner pixel product和
slot-cycle wrong-mask IoU均为`0`。global/region/code shape分别为`[8,768]`、`[8,5,768]`、
`[8,5,16]`，全部finite，teacher所有参数frozen且输出无grad。donor全部不同PID且无fixed point。

描述性五slot `correct↔flip`相对donor+donor-mask margin=
`0.806/0.778/0.898/0.647/0.917`；相对donor RGB+recipient mask=
`0.847/0.886/1.047/0.718/0.889`；相对same-RGB wrong-mask=
`0.621/0.173/0.627/0.203/0.756`。这些8图值不做统计裁决，只说明real-data contract方向未立即反转。
五组PC-MBCLS累计forward=`1.049s`，peak allocation=`1,712,272,384 bytes`。

script/result/runner SHA256分别为
`ab36357174fbf2f2181bcfbaefb71d5a47d0b55de901603c3d2e475a2bd32569`、
`9a2cdd2ec69707ce325fd3bd22d47b82fe2bc869116263e5595faa16579222df`、
`4233a7c856a7c9085522f015c8c0887eb601214cbb64fa4007462bb853710d83`。严格异常/NaN/Inf/AMP
warning/OOM=`0`，execution repo exact HEAD与tracked clean，进程退出，GPU回到`2 MiB/0%`。只授权
0E-128稳定性审计；正式训练、Phase B和semantic multi-stage仍NO-START。

## 2026-07-19 Phase 0E-128实现与启动前冻结

新增独立`phase0e_rich_evidence_128.py`，不修改已封板C8脚本。样本固定为128个不同PID且全部五slot
valid，按同一PID hash选择64 fit/64 held-out audit；slot mean与shared PCA-16只在fit侧拟合，basis、
partition和paths单独写入codebook JSON并记录SHA。五组teacher arm固定为correct、同步flip、
different-PID donor RGB+donor mask、donor RGB+recipient mask、same-RGB low-IoU slot-cycle mask。

正式门禁只包括：held-out每slot 16维std都`>1e-8`、macro entropy effective rank `>=8/16`、
wrong-RGB与wrong-mask paired margin逐slot PID-cluster bootstrap 95% CI下界都`>0`，以及contract/
NULL/repeat/hard-owner/static-global exact、frozen/no-grad/finite。slot-cycle binding、raw uncentered、
fixed random orthogonal和donor-recipient只报告为强对照，不根据128图结果调门槛或换projection。

本地/远端script SHA256均为
`deae5c9308650f9f9344ab19e0e78fa78b193a53244e41ccc24d9274fbd1526a`；在已验证
`/home/afr/par2606/.venv`完成静态编译，execution repo exact HEAD/tracked clean，启动前GPU=
`2 MiB/0%`且无训练/审计进程。当前状态`0E-128 READY / NO-RESULT`，不授权训练或Phase B。

## 2026-07-19 Phase 0E-128稳定性审计封板

0E-128在唯一4090自然完成并`SEALED-PASS`。128个不同PID按同一hash严格分成64 fit/64 held-out；
official parity=`0`、repeat/NULL exact、hard-owner/wrong-mask IoU=`0`、donor不同PID且无fixed point、
teacher frozen/no-grad、全部feature/code finite。五slot各16维held-out std全`>1e-8`，实际最小std=
`0.1649/0.1480/0.1802/0.1219/0.1371`；effective rank=
`10.764/10.756/11.843/10.788/11.101`，macro=`11.050`。

五slot correct↔flip相对wrong RGB margin均值=
`0.808/0.735/0.781/0.742/0.821`，95% CI下界=
`0.709/0.639/0.709/0.655/0.733`；相对same-RGB wrong-mask均值=
`0.614/0.179/0.413/0.165/0.645`，CI下界=
`0.531/0.097/0.341/0.103/0.575`，五slot均严格正。donor RGB+recipient mask与slot-cycle binding的
五slot CI也均严格正，但不是正式门。

强对照显示slot-mean/global-only code exact zero。raw uncentered macro rank=`10.216`，但wrong-RGB
margin仅`0.211–0.261`、wrong-mask仅`0.047–0.280`，低于centered code；fixed random orthogonal
macro rank=`13.458`且同样保留强margin，说明信号属于rich local residual，不依赖PCA偶然挑轴。PCA只
作为固定压缩器，不能包装成贡献。五组arm累计PC-MBCLS forward=`12.505s`，peak allocation=
`1,712,272,384 bytes`。

script/result/codebook/runner SHA256分别为
`deae5c9308650f9f9344ab19e0e78fa78b193a53244e41ccc24d9274fbd1526a`、
`47a27631756c42bfa696f9751b604532fa9033489d67ef107126fcaa254b19dc`、
`4a671a70e0744edad88f911ce628d421650cb09453eb511a61e8d01c239269ef`、
`e8f35143a8599bfec3f3e0354b872bc71090d48420a6408fa9d517d3f46c01a3`。严格异常/AMP warning=
`0`，execution HEAD/tracked clean，进程退出，GPU=`2 MiB/0%`。仅授权0E-FULL teacher-only审计；
Phase A/B正式训练与semantic multi-stage仍NO-START。

## 2026-07-19 Phase 0E-FULL实现与启动前检查

已实现official 15,618 train的两遍流式teacher-only审计脚本
`phase0e_rich_evidence_full.py`，SHA256=
`54a1a899e634fa317eacf0caa5acf788434b4d3cc55f4d8a9a9173b557e17deb`。第一遍只缓存correct raw
evidence/valid，fit侧用分块协方差/eigh拟合shared PCA-16；第二遍仅对held-out PID编码四类
counterfactual。不会构建ReID model/optimizer/config/output/checkpoint，也不会把15,618图RGB驻留
内存。

本地和远端`/home/afr/par2606/.venv`的streaming-fit synthetic contract均PASS：center mean max=
`2.050e-16`、orthogonal max=`1.998e-15`、相对direct SVD subspace max分别为
`7.216e-16/5.933e-16`，finite及partition checks全PASS。远端脚本复制后SHA exact，`py_compile`
PASS；execution repo HEAD=`ed5783416528be4284adce11fa192fe119e344f4`且tracked clean，启动前无
训练/审计进程，GPU=`2 MiB/0%`。当前状态`0E-FULL READY / NO-RESULT`，不授权Phase A/B训练或
semantic multi-stage。

## 2026-07-19 Phase A RNG-neutral完整CUDA预检PASS

以新execution slice HEAD=`09340f76f84502f9018bee3c8eec005961b0a8cb`完整重做24步official
Torch1.13.1/OpenCLIP2.32 CUDA/AMP预检，13项gate全部PASS。candidate相对Semantic C0仅config
`SEMANTIC_REZERO False→True`与独立OUTPUT_DIR不同；非目标state mismatch=`[]`，参数只增加两个
alpha scalar，两个expert std=`0.019981/0.020003`。

初始化full/all-bypass逐tensor exact，initial与final NULL exact。前5步GradScaler exact skip；第6步为
首个finite update，两alpha grad=`1.126e-3/6.647e-4`，六类branch grad exact zero；此后连续18步
token/context/expert均有finite非零梯度与参数更新。full/bypass gap由0增长至max-abs=
`1.473e-4`、mean L2=`6.167e-4`。teacher parity/isolation、RGB-only、strict reload、state finite均PASS；
peak allocated/reserved=`7.57/8.01 GB`，吞吐=`131.16 samples/s`。

result/runner SHA256=
`6cbe7367848cf289b80793d897f27b4e37ad7dae8432c5734e01c595a7a9c08c`/
`67bb82541160227e608c5893e3bf71384dd78944673cd2bbb7c531e8bbfb2462`；config/script SHA256=
`f409cc069b6f3500e009e6d40681e8baf9547bb77b864e9f35a7ea02ca11d1a6`/
`c0b223bf83a5fd7ef8bd75539c1db4be6808588fd4bcd62d67475fd159e4148e`。GPU恢复`2 MiB/0%`。
Phase A RZ-C0 fresh e120由此获得正式启动授权；Phase B和semantic multi-stage仍NO-START。

## 2026-07-19 Phase A首次完整CUDA预检：实现审计FAIL并定位

首次正确official runtime的24步CUDA/AMP预检自然完成，19次finite更新、前5步GradScaler exact skip；
initial full/bypass exact，首个finite step两alpha梯度非零且六类branch梯度exact zero，随后18次finite
更新两router的token/context/expert梯度与参数更新全部非零；final full/bypass max-abs=
`2.522e-4`、mean L2=`6.101e-4`。NULL、teacher parity/isolation、strict reload、RGB-only、state finite、
约8.01 GB峰值等其余12项门全部PASS。

唯一FAIL是single-variable state：router-1的token/context projection与Semantic C0初始化不再exact。
根因不是route数学或训练不稳定，而是router-0新增expert random draw推进了全局CPU RNG，导致随后
router-1 projection取到不同随机数。该审计正确阻止了形式上的单变量混淆。修复冻结为expert init前后
保存/恢复CPU RNG state，使新增random expert不移位任何非目标参数；必须用新exact source与新结果
完整重做preflight，不能把当前FAIL JSON改写为PASS。

首次启动后复核：PID=`884504`持续唯一运行，第一遍已推进到`correct-pass 5,632/15,618`；GPU=
`2,402 MiB/99%`、唯一compute PID显存=`2,394 MiB`，异常关键词计数=`0`。继续自然运行，不作中间
统计裁决。

## 2026-07-19 Phase 0E-FULL自然完成与封板

PID=`884504`自然退出，GPU恢复`2 MiB/0%`，execution HEAD/tracked与脚本SHA保持exact，严格异常/
AMP warning计数=`0`。result verdict=`PHASE0E_FULL_PASS`，13项正式gate全部PASS。official
15,618图全部流式覆盖；fit/audit=`7,860/7,758`图、`361/341`个PID且strict disjoint。

五slot effective rank=`12.332/12.289/12.950/12.278/11.828`，macro=`12.335/16`；逐slot最小std=
`0.1800/0.1631/0.1684/0.1687/0.1739`且全部16维非零。wrong RGB margin mean=
`0.774/0.765/0.750/0.790/0.785`、CI low=`0.756/0.748/0.733/0.773/0.766`；same-RGB
wrong-mask mean=`0.650/0.170/0.497/0.201/0.646`、CI low=
`0.632/0.160/0.480/0.189/0.633`。donor-recipient与slot-cycle五slot CI也全正，但非正式门。

raw uncentered明显更弱；fixed random orthogonal macro rank=`14.725`且保留强margin，说明证据属于
rich residual而非PCA偶然轴。slot-mean/global-only exact zero。correct/counterfactual forward=
`305.211/610.067s`，peak=`1,712,272,384 bytes`。script/result/partition/codebook/runner SHA依次为
`54a1a899e634fa317eacf0caa5acf788434b4d3cc55f4d8a9a9173b557e17deb`、
`f2f9d2b0d03eb46091978f5c52b849eaa6b2fd5411947d959153f6d81df8828e`、
`5aed0bc67c5998d79a7ed9ccbeb8481815921728e0af57c33182939d8478da67`、
`fb87da370ea945d526f499bef78093a6b07203d87c6d84efe06b5eb6594f954a`、
`53b31edcbd5e3779782429468a41b38c57c05c763cdf4ebfa1a3ce3dce403cb7`。

裁决：`Phase 0E = SEALED-PASS`，只授权Phase B使用该teacher接口；不直接授权任何正式训练或
semantic multi-stage。下一步转入逻辑独立的Phase A RZ-C0实现/preflight。

后续heartbeat复核：PID=`884504`仍由PID 1接管且唯一，第一遍推进到
`correct-pass 9,632/15,618`；partition已自然落盘（约6.4 KiB），result/codebook尚未出现，符合两遍
执行顺序。GPU=`2,402 MiB/53%`、唯一compute PID显存=`2,394 MiB`，脚本SHA、execution HEAD与tracked
clean保持exact，异常关键词计数=`0`。继续自然运行，不作中间裁决。

## 2026-07-19 Phase 0E-FULL正式teacher-only启动

0E-FULL已在唯一4090上fresh启动，main PID=`884504`，runner=
`/home/afr/reid-clean/audits/exp393_phase0e/phase0e_full.runner.log`，result/partition/codebook/cache均为
独立新路径。启动后main由PID 1接管且唯一；首个完整chunk已到`correct-pass 32/15,618`，GPU=
`2,402 MiB/98%`、唯一compute PID=`884504`，无异常。execution repo exact HEAD=
`ed5783416528be4284adce11fa192fe119e344f4`且tracked clean；本地设计/实现提交HEAD=
`8beff997990d98babbecf2c3807aab14cd830a55`。

当前状态`0E-FULL FORMAL TEACHER AUDIT RUNNING`。不得修改运行中脚本、依赖、partition或阈值；必须
自然完成两遍流式处理后再按全部五slot门禁裁决。它不是训练，不授权并行GPU任务、Phase A/B或
semantic multi-stage。

## 2026-07-19 Phase A RZ-C0正式训练e3复核

heartbeat已从过时的0E-FULL状态更新到当前RZ-C0正式训练，仍保持`ACTIVE`且每15分钟触发。远端
execution HEAD=`09340f76f84502f9018bee3c8eec005961b0a8cb`、tracked source clean，config SHA256=
`f409cc069b6f3500e009e6d40681e8baf9547bb77b864e9f35a7ea02ca11d1a6`，均与启动冻结值exact。

main PID=`888440`、parent shell=`888439`，8个DataLoader worker；4090仅该main占用
`8,286 MiB`，总显存/利用率=`8,294 MiB/85%`。训练已自然完成e3，e3 Iter 200的Loss/Pose/
Semantic/RegionMask/Presence/Q=`8.010/1.597/0.687/0.684/0.683/0.693`，Acc=`0.147`，
Student=`0.00`，Reliability=`0.508`。e2起`GateAbs`由exact zero变为约`1e-14–1e-11`的极小非零值，
e3 Iter 200=`2.779e-13`；这只表明ReZero gate开始数值打开，不作route有效性或性能裁决。

严格异常、NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow/AMP warning=`0`；e120前
checkpoint数=`0`。状态保持`FORMAL RUNNING`，禁止修改、重启、续训或按当前GateAbs/中间性能早停。

后续轻量复核时e4已自然完成并进入e5。e4 Iter 200的Loss/Pose/Semantic/RegionMask/Presence/Q=
`7.497/1.582/0.679/0.674/0.671/0.693`，Acc=`0.216`，Reliability=`0.507`；e5 Iter 100的
`GateAbs=3.088e-09`。GateAbs相对e2–e3的`1e-14–1e-11`继续缓慢增大，只记录为ReZero轨迹，
不作中间裁决。exact HEAD/config、main+8 workers、唯一GPU任务保持不变，严格异常=`0`、checkpoint=`0`。

## 2026-07-19 Phase A RZ-C0首次e10评测

e10自然完成后的首次正式评测为mAP/R1/R5/R10=`33.8/43.6/59.6/65.5`。该值只作为冻结recipe的
轨迹记录，不与final门槛比较，也不授权按中间性能修改、重启或早停。评测后训练自然继续，heartbeat
复核时已完成e13并进入e14。

handoff后`Student=1.00`；e11–e13的`GateAbs`已稳定进入约`1.9e-07–3.6e-07`，相对e4–e5的
`1e-09`量级继续打开，说明当前route不再数值冻结，但是否形成有意义的retrieval贡献仍只能由e120
full与all-router-bypass终审决定。e13 Iter 200的Loss/Pose/Semantic/RegionMask/Presence/Q=
`4.162/1.230/0.523/0.457/0.418/0.693`，Acc=`0.431`，Reliability=`0.504`，GateAbs=
`1.871e-07`。

execution HEAD/config/tracked保持exact，main PID=`888440`及8 workers唯一；4090仅该main占用
`8,254 MiB`，严格异常=`0`，e120前checkpoint=`0`。状态保持`FORMAL RUNNING`。

## 2026-07-19 Phase A RZ-C0 e20评测

e20自然完成后的评测为mAP/R1/R5/R10=`40.9/53.2/68.0/73.7`，相对e10轨迹
`33.8/43.6/59.6/65.5`正常上升，但仍不作为final裁决或任何调参、重启、早停依据。复核时训练已
自然完成e21并进入e22。

e20 Iter 200的Loss/Pose/Semantic/RegionMask/Presence/Q=
`1.926/0.935/0.384/0.269/0.189/0.692`，Acc=`0.827`，Student=`1.00`，Reliability=
`0.509`，GateAbs=`2.567e-07`；e21–e22仍约`1.9e-07–2.5e-07`，保持finite且非零。execution
HEAD/config/tracked exact，main+8 workers与唯一4090任务正常，严格异常=`0`、e120前checkpoint=`0`。

## 2026-07-19 Phase A RZ-C0 e30评测

e30自然完成后的评测为mAP/R1/R5/R10=`43.6/52.6/68.8/74.1`。mAP相对e20的`40.9`继续
上升，R1由`53.2`波动至`52.6`；两者都只作轨迹，不按中间单项变化裁决。评测后已自然进入e31。

e30 Iter 200的Loss/Pose/Semantic/RegionMask/Presence/Q=
`0.624/0.803/0.320/0.187/0.080/0.692`，Acc=`0.970`，Student=`1.00`，Reliability=
`0.512`，GateAbs=`1.617e-07`；e29–e31约`1.4e-07–1.8e-07`，仍finite且非零。execution
HEAD/config/tracked exact，main PID=`888440`与8 workers唯一；4090仅该main占用`8,374 MiB`，
严格异常=`0`、e120前checkpoint=`0`，继续自然运行。
