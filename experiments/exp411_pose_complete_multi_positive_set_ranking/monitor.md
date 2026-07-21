# exp411 PCMPSR 监控

## 2026-07-21：对象选择与设计冻结

exp410唯一fresh correct自然e120=`45.0/56.4/71.3/76.7`，相对clean D0全面大幅下降，已永久封板为
`SEALED NO-GO / PERFORMANCE FAIL`；GPU恢复`2 MiB/0%/0 compute PID`。下一对象明确禁止固定CLIP classifier、
projection/adapter救臂、局部relation重试或单hard-pair微调。

独立候选审计推荐PCMPSR：在每个PK batch内为每个身份构造等大小leave-one-position-out三图支持集，五槽pose
coverage与同PID CLIP槽共识只离散选择slot owner；final student descriptor对全部16个身份集合做listwise排序。
learned CE、D0 pose loss、student坐标与eval保持不变。该对象同时回应exp409的R1/AP分裂与exp410的外部轴错配。

近邻审计把SupCon、lifted/listwise metric、episodic set loss、pose-aware sampling和CLIP-ReID/ProFD列为已有原子；
未发现“等支持leave-one-position-out身份集合+五槽pose×CLIP owner multiplicity+final student全身份排序”的同构
实现。问题门PASS、证据门PASS、机制门CONDITIONAL PASS，定位C类候选。当前状态=
`DESIGN/PROTOCOL FROZEN / IMPLEMENTATION NEXT / GPU IDLE`。

## 2026-07-21：实现与盲审闭环

已完成default-off PCMPSR config、fresh cache builder/strict loader、等支持set/owner构造、FP32 listwise loss、
`make_loss`与processor接线；model/eval零修改。本地synthetic PK64合同PASS：support/owner shape=
`[64,16,3]/[64,16,5]`，owner unique mean=`2.421875`，wrong-RGB/generic/pose-only owner change=
`0.096875/0.059375/0.05625`，listwise loss与final feature梯度finite/nonzero。

独立智能体盲审首轮`0B/2H`，指出pose-invisible owner与真实default-off/isolated梯度合同缺口。两轮聚焦修复后最终
`0B/0H`：正常owner严格`visibility>0 & clip_valid`，显式pose-first fallback单独报告；唯一真实PK64脚本已冻结为
同时检查D0-vs-default-off四类RNG/state/forward/loss exact、isolated PCMPSR descriptor/Stage-3/backbone梯度及
combined native AMP update。当前状态=`IMPLEMENTATION REVIEW 0B/0H / FRESH CACHE NEXT / GPU IDLE`；真实CUDA
合同仍待cache后一次执行，尚无exp411性能结果。

## 2026-07-21：fresh cache已启动

relay恢复后，从exp410 formal clean基底建立fresh远端repo=
`/home/afr/SOLIDER-REID-exp411-pcmpsr-feb56c1-v1`，显式传输目标文件并提交；运行source HEAD=
`ebf60f2b4a5c943958f7077779d8500c2855874a`，关键loss/builder/real-batch/config SHA与本地byte-exact。启动前repo
tracked/untracked均clean、CLIP checkpoint SHA=`9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e`、
GPU=`2 MiB/0%/0 compute PID`，asset与runner路径均fresh。

唯一cache主PID=`574498`，asset=`/home/afr/reid-clean/assets/exp411-pcmpsr-cache-v1`，runner=
`/home/afr/reid-clean/train-logs/exp411-pcmpsr-cache-v1.runner.log`。首次观测已编码`8/15618`，GPU约
`2186 MiB/94%`且只有该compute PID，无异常。当前=`FRESH CACHE RUNNING / SOURCE+PARAMETERS FROZEN`；只监控
自然完成，不修改builder/source/参数，不复用exp408/409/410 cache。

## 2026-07-21：fresh cache完成并核验PASS

唯一cache自然完成全15,618图并正常退出，runner异常计数0，GPU恢复`2 MiB/0%/0 compute PID`。独立strict loader
核验：samples/unique paths=`15618/15618`，shape=`[15618,5,768]` FP16，五槽valid=
`[15616,15618,15618,15618,15586]`，全finite；有效feature L2 norm min/max=
`0.9996981621/1.0002959967`。

- cache SHA=`b07576130a0c50b89194f2c59467defcf39293d96ca886616865eb198e7965d1`；
- manifest SHA=`2ea8410f27737aaf3ba12547049e9013a24b86be1d5753509c9abbff0e7194a4`；
- runner SHA=`272f8507279aeb47f4a9e086c7f8ce6e0af4635a002344384ad64b0e4b228c6c`；
- path/RGB-SHA-vector/PID-vector SHA=
  `e53ef9189f12737d6621ae152979cf2d12f8bb24cc823466a6ef11928bd99f4e`/
  `c9398b6d8fa81062e37968783ccea76252d9c4401ba36d50b8f8a1ae83bdbbf1`/
  `4d0181ec8795fcffc0d3e63785db92fc15b6f9eed232bf81e54a0d49cdc419ce`。

source HEAD=`ebf60f2b4a5c943958f7077779d8500c2855874a`，builder/teacher source SHA=
`58e47b65dc34ff5642a0f683b38f631294d14442b62b598024daa5b08bb2203c`/
`fbd3e137a729f44d3179864f9978bd8846b22e8627a3c311747b0a2541092864`。真实cache SHA已写回config。
当前=`FRESH CACHE PASS / UNIQUE REAL-PK64 CONTRACT AUTHORIZED / GPU IDLE`；该cache结果只证明输入有效，不是
ReID性能或机制GO。

## 2026-07-21：real-batch v1运行时退出，科学未评估

唯一v1 runner在CUDA/model/cache合同开始前退出：fresh远端repo由exp410 formal基底克隆，该基底未包含
`configs/occluded_duke/swin_tiny_tapf_d0.yml`，default-off对照读取时报`FileNotFoundError`。v1 runner SHA=
`0469410c044cbc15b9dacb1620670577ea3ac0943cd121b36803585b57443550`；GPU始终`2 MiB/0%/0 compute PID`，没有
forward、梯度、update或科学指标。

最终记录=`REAL-PK64-V1 SEALED RUNTIME FAILURE / SCIENCE NOT EVALUATED`，禁止覆盖或重跑v1。修正只把本地sealed
D0 config byte-exact传入远端fresh repo，不改PCMPSR core、cache、manifest、method config、contract逻辑或门槛；
随后使用fresh `exp411-pcmpsr-real-batch-v2.runner.log`执行同一合同。

## 2026-07-21：唯一有效real-batch v2合同PASS

fresh v2在固定MMPOSE-ABU中自然完成并正常退出，runner SHA=
`c60afabde025ed00c9e66ad0ef0d0a5dbb4331ed309dd402c5455d8b3c2cc3ef`，GPU恢复`2 MiB/0%/0 compute PID`。
全部冻结门PASS：

- D0与PCMPSR-default-off的同seed model state、真实forward、combined loss，以及构造/forward前后Python、NumPy、
  Torch CPU、全部CUDA RNG exact；default-off loss=`20.3135318756`；
- PK64=`16 PID×4`，correct owner unique mean=`2.3125`、fallback=`0.0`；correct相对wrong-RGB/generic/pose-only
  owner change=`0.284375/0.175/0.209375`，三轴均active；
- set loss=`1.6809189320`，positive/negative set distance=`57.9002265930/67.9471282959`，均finite；
- isolated PCMPSR对Stage-3/backbone产生`26/173`个finite nonzero梯度tensor，排除CE/pose代偿；
- combined loss/reid/pose=`8.0442008972/7.9523572922/0.9184324741`；default GradScaler前四次native overflow
  从65536回退到4096，第5次无nonfinite并真实更新
  `base.stages.3.blocks.1.attn.w_msa.qkv.weight`；combined Stage-3/backbone非零梯度=`26/181`。

当前=`IMPLEMENTATION/CACHE/REAL-PK64 PASS / UNIQUE FRESH E120 AUTHORIZED / GPU IDLE`。不再追加测试；清理本次
生成的untracked Python3.8 cache后，从fresh formal clone启动唯一correct seed1234/e120，运行中冻结source/config/cache。

## 2026-07-21：唯一fresh correct student已启动

真实合同后生成的untracked Python3.8 cache已清理且tracked runtime文件保持exact；fresh formal clone=
`/home/afr/SOLIDER-REID-exp411-pcmpsr-formal-v1`，运行source HEAD=
`0db28ecec911cf4776dcbabaf4ce0cda018dcf90`，config SHA=
`01c060d676b4f2b267d0c2c60366d70b1d244a44609c95a7b12fc38b759b4651`。启动前formal repo clean、cache SHA exact、
GPU空闲，output/runner均fresh。

唯一训练主PID=`576005`，output=`/home/afr/reid-clean/logs/exp411-pcmpsr-s1234-v1`，runner=
`/home/afr/reid-clean/train-logs/exp411-pcmpsr-s1234-v1.runner.log`。首batch set loss/positive/negative distance=
`2.204848/58.077400/66.888412`，owner unique/fallback=`2.2969/0.0`，correct相对wrong-RGB/generic/pose-only
owner change=`0.284375/0.171875/0.20625`，全部finite/active。首次观测GPU约`6990 MiB/69%`且只有该compute
PID，无异常。

当前=`UNIQUE FRESH E120 RUNNING / SOURCE+CONFIG+CACHE FROZEN`。运行中只监控自然完成，在e10/20/.../120记录与
sealed clean D0同epochmAP/R1，不修改formal repo、config、cache或参数，不按中间点早停。

## 2026-07-21：e10首次正式评测

唯一fresh训练自然完成e10评测：PCMPSR=`27.7 mAP / 36.9 R1 / 52.8 R5 / 59.6 R10`；sealed clean D0
同epoch=`33.4 mAP / 42.7 R1`，因此rounded `ΔmAP/ΔR1=-5.7/-5.8`。读取结果时训练已自然进入e12，
主PID=`576005`仍为唯一compute PID，GPU约`7072 MiB/65%`，runner中
Traceback/RuntimeError/OOM/NaN/Inf计数为0。

| epoch | PCMPSR mAP/R1 | PCMPSR R5/R10 | sealed clean D0 mAP/R1 | rounded ΔmAP/ΔR1 |
|---:|---:|---:|---:|---:|
| 10 | 27.7/36.9 | 52.8/59.6 | 33.4/42.7 | -5.7/-5.8 |
| 20 | 45.5/55.8 | 70.0/76.4 | 42.2/52.4 | +3.3/+3.4 |
| 30 | 50.2/61.8 | 75.6/80.1 | 46.6/56.2 | +3.6/+5.6 |

e10为明显不利的warmup中间点，但不改变冻结协议、不早停也不修改运行中source/config/cache；继续自然训练至e120，
最终仍只按raw mAP/R1双门裁决。

e20已自然转为双领先：PCMPSR=`45.5/55.8/70.0/76.4`，相对sealed clean D0同epoch mAP/R1=
`42.2/52.4`为`+3.3/+3.4`。读取时已进入e23，主PID仍唯一，GPU约`7080 MiB/65%`，异常计数0。该恢复说明
e10不能作为早停依据，但e20仍只属于中间轨迹，不改变自然e120唯一裁决。

e30继续双领先：PCMPSR=`50.2/61.8/75.6/80.1`，相对sealed clean D0同epoch mAP/R1=
`46.6/56.2`为`+3.6/+5.6`。读取时已进入e33，唯一compute PID，GPU约`7068 MiB/41%`，异常计数0。
当前优势强于e20，但仍不据此早停、调参或宣告GO。
