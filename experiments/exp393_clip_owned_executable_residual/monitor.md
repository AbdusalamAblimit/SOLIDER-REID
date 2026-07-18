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
