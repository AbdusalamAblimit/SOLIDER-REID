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
| 40 | 54.7/66.6 | 80.0/84.2 | 50.0/60.7 | +4.7/+5.9 |
| 50 | 54.9/66.1 | 79.1/83.6 | 52.1/62.8 | +2.8/+3.3 |
| 60 | 57.2/69.6 | 81.5/84.8 | 55.1/66.1 | +2.1/+3.5 |
| 70 | 57.4/70.2 | 81.5/84.9 | 55.4/65.2 | +2.0/+5.0 |
| 80 | 58.2/70.4 | 81.9/86.1 | 56.1/66.3 | +2.1/+4.1 |
| 90 | 58.8/70.7 | 83.0/86.3 | 57.5/67.9 | +1.3/+2.8 |
| 100 | 58.5/70.1 | 82.4/85.8 | 56.9/67.1 | +1.6/+3.0 |
| 110 | 58.6/69.9 | 82.2/85.7 | 57.4/67.4 | +1.2/+2.5 |
| 120 | 58.8/70.1 | 82.1/85.8 | 57.6/67.7 | +1.2/+2.4 |

e10为明显不利的warmup中间点，但不改变冻结协议、不早停也不修改运行中source/config/cache；继续自然训练至e120，
最终仍只按raw mAP/R1双门裁决。

e20已自然转为双领先：PCMPSR=`45.5/55.8/70.0/76.4`，相对sealed clean D0同epoch mAP/R1=
`42.2/52.4`为`+3.3/+3.4`。读取时已进入e23，主PID仍唯一，GPU约`7080 MiB/65%`，异常计数0。该恢复说明
e10不能作为早停依据，但e20仍只属于中间轨迹，不改变自然e120唯一裁决。

e30继续双领先：PCMPSR=`50.2/61.8/75.6/80.1`，相对sealed clean D0同epoch mAP/R1=
`46.6/56.2`为`+3.6/+5.6`。读取时已进入e33，唯一compute PID，GPU约`7068 MiB/41%`，异常计数0。
当前优势强于e20，但仍不据此早停、调参或宣告GO。

e40仍保持并扩大双领先：PCMPSR=`54.7/66.6/80.0/84.2`，相对sealed clean D0同epoch mAP/R1=
`50.0/60.7`为`+4.7/+5.9`。读取时已进入e43，唯一compute PID，GPU约`7072 MiB/69%`，异常计数0。
这是连续第三个双领先中间点，但性能GO仍只由自然e120 raw双门决定。

e50 PCMPSR=`54.9/66.1/79.1/83.6`，相对sealed clean D0同epoch mAP/R1=`52.1/62.8`仍双领先
`+2.8/+3.3`。相对e40自身mAP仅升`0.2`且R1回落`0.5`，领先幅度收窄，属于必须完整保留的中间不确定性；
读取时已进入e53，唯一compute PID，GPU约`7080 MiB/51%`，异常计数0，继续自然训练。

e60恢复到PCMPSR=`57.2/69.6/81.5/84.8`，相对sealed clean D0同epoch mAP/R1=`55.1/66.1`为
`+2.1/+3.5`，继续双领先。读取时已进入e63，唯一compute PID，GPU约`7072 MiB/44%`，异常计数0。e60的
rounded mAP仍低于最终raw门，但不同epoch不可代替e120裁决，因此继续自然运行。

e70 PCMPSR=`57.4/70.2/81.5/84.9`，相对sealed clean D0同epoch mAP/R1=`55.4/65.2`为
`+2.0/+5.0`，连续保持双领先。读取时已进入e74，唯一compute PID，GPU约`7064 MiB/41%`，异常计数0。
晚段R1优势较强，但e70 rounded mAP仍不能代表最终raw双门，继续按协议自然训练。

e80 PCMPSR=`58.2/70.4/81.9/86.1`，相对sealed clean D0同epoch mAP/R1=`56.1/66.3`为
`+2.1/+4.1`。读取时已进入e84，唯一compute PID，GPU约`7076 MiB/41%`，异常计数0。虽然e80 rounded数值已
高于最终raw门，但协议禁止用不同epoch替代e120，故只记录为连续双领先中间证据。

e90 PCMPSR=`58.8/70.7/83.0/86.3`，相对sealed clean D0同epoch mAP/R1=`57.5/67.9`为
`+1.3/+2.8`。读取时已进入e94，唯一compute PID，GPU约`7096 MiB/41%`，异常计数0。mAP领先幅度继续收窄，
因此不能将此前较大中间优势外推到e120；完整保留并继续自然训练。

e100 PCMPSR=`58.5/70.1/82.4/85.8`，相对sealed clean D0同epoch mAP/R1=`56.9/67.1`为
`+1.6/+3.0`。相对e90自身mAP/R1回落`0.3/0.6`，但同epoch仍双领先。relay短时不可达期间训练自然继续；恢复读取时
已进入e110，唯一compute PID，GPU约`7104 MiB/66%`，异常计数0。继续等待e110/e120，不选择性外推。

e110 PCMPSR=`58.6/69.9/82.2/85.7`，相对sealed clean D0同epoch mAP/R1=`57.4/67.4`为
`+1.2/+2.5`，仍双领先，但相对e100自身mAP仅升`0.1`、R1再降`0.2`。读取时已进入e115，唯一compute PID，
GPU约`7080 MiB/41%`，异常计数0。最后裁决只剩自然e120，继续冻结运行。

## 2026-07-21：自然e120完成，correct性能GO

唯一fresh correct arm自然完成e120并正常退出，最终=`58.8 mAP / 70.1 R1 / 82.1 R5 / 85.8 R10`；sealed
clean D0=`57.6/67.7/80.8/84.6`，rounded四项差=`+1.2/+2.4/+1.3/+1.2`。runner只打印一位小数，但
`58.8`与`70.1`各自的舍入区间下界仍严格高于预注册raw门`57.5587756578/67.6923076923`，所以性能双门
无歧义PASS。

主PID=`576005`已自然消失，GPU=`2 MiB/0%/0 compute PID`，runner异常计数0；formal HEAD仍为
`0db28ecec911cf4776dcbabaf4ce0cda018dcf90`，tracked文件无变化，config/cache SHA仍为
`01c060d676b4f2b267d0c2c60366d70b1d244a44609c95a7b12fc38b759b4651`/
`b07576130a0c50b89194f2c59467defcf39293d96ca886616865eb198e7965d1`。产物SHA：

- checkpoint=`8bd928f39bd895ddf3733ede4ff5449dff90190e5f2cddac30d186d05a92c01e`；
- train log=`55ecc3686195671f155c5df5baf9bfadaf1cdbe82f3af1fc984af509acf426f6`；
- runner=`76be6ba38c877dda7617db8bfb637283e7396a1b6b61cd45139dc26326536d5d`。

当前封板=`CORRECT ARM SEALED / PERFORMANCE GO / ATTRIBUTION PENDING`。不得重跑correct、续训或调owner/loss；
下一步只按冻结协议串行执行zero-owner与wrong-RGB matched controls。只有correct严格胜二者，才升级为pose+CLIP
科学GO与正面story。

## 2026-07-21：matched-control单变量与顺序冻结

新增`matched_controls.md`明确归因执行图。`zero_owner`只从集合距离删除五个slot-owner multiplicity，保留同一三图
support与全身份set ranking；`wrong_rgb`只把owner所用CLIP槽按固定4行different-PID shift轮换，support PID不变。
新增配置必须默认`correct`，PCMPSR关闭时D0 exact；两臂仍加载同一cache与pose并使用fresh output、seed1234、自然
e120。执行顺序固定为`zero_owner`先、`wrong_rgb`后，禁止并行或提前用中间结果裁决。当前=
`CONTROL DESIGN FROZEN / IMPLEMENTATION+CONTRACT NEXT / GPU IDLE`。

## 2026-07-21：control实现与synthetic/盲审闭环

已新增默认`correct`的显式`PCMPSR_CONTROL_MODE`，formal helper只允许`correct/zero_owner/wrong_rgb`；zero-owner在
loss内严格使用三support均值且owner term=`0`，wrong-RGB复用固定shift=4，processor按config选择正式state。两份
matched config相对sealed correct只改变mode与fresh output。

本地uv synthetic合同PASS：correct旧state与显式correct loss/set distance逐位exact，zero-owner逐项等于手工三support
均值，formal wrong owner exact等于direct wrong且非correct；correct/zero/wrong loss分别=
`0.8500004411/0.8254277110/0.8496659398`，descriptor梯度L1分别=
`2.6877369881/2.4881243706/2.6964554787`，均finite/nonzero。独立盲审首轮`0B/1H`指出formal wrong helper可能未被
合同硬绑定；只补负向合同后复审=`0B/0H`。远端4090实测空闲，correct formal tracked仍未变化。当前=
`IMPLEMENTATION/SYNTHETIC/REVIEW PASS / REAL-PK64 CONTROL CONTRACT NEXT / GPU IDLE`。

## 2026-07-21：zero-owner唯一真实PK64合同PASS

独立远端preflight repo=`/home/afr/SOLIDER-REID-exp411-pcmpsr-controls-preflight-v1`，source HEAD=
`f98ab2daafa294dd0db004e10519363025a45488`；七个关键文件与本地SHA byte-exact，zero-owner config SHA=
`f418b0433c13208e9f844a249be3089fac3bc38c06ea173471f39ce41774f6c5`。固定MMPOSE-ABU synthetic复现PASS后，
唯一fresh真实PK64合同自然完成并退出：

- control mode=`zero_owner`，owner term=`0`，correct owner unique/fallback=`2.3125/0`；
- wrong-RGB/generic/pose-only owner change=`0.284375/0.175/0.209375`；
- default-off state/forward/combined loss及Python/NumPy/Torch CPU/all-CUDA RNG exact；
- zero-owner set loss/positive/negative distance=`1.5925421715/57.8170280457/67.6581573486`；
- isolated set loss对Stage-3/backbone产生`26/173`个finite nonzero梯度tensor；
- combined loss/reid/pose=`7.8512554169/7.7594122887/0.9184324741`，default GradScaler第5次取得真实
  Stage-3 update，combined Stage-3/backbone非零梯度=`26/181`。

runner SHA=`28731e86899bac6f7a9444bef7f4a896822b942c51d73d9672ca833bd6d3b3ba`，异常计数0；运行后GPU=
`2 MiB/0%/0 compute PID`，preflight tracked clean，sealed correct formal HEAD/tracked仍未变化。当前=
`ZERO-OWNER REAL-PK64 PASS / FRESH FORMAL E120 AUTHORIZED / GPU IDLE`。不追加测试；只从已提交preflight
HEAD建立fresh zero-owner formal并自然运行到e120。

## 2026-07-21：唯一fresh zero-owner student已启动

从已提交preflight HEAD建立fresh formal=
`/home/afr/SOLIDER-REID-exp411-pcmpsr-zero-owner-formal-v1`，source HEAD=
`f98ab2daafa294dd0db004e10519363025a45488`，clone后tracked/untracked均clean。zero-owner config/loss/processor/cache
SHA分别为`f418b0433c13208e9f844a249be3089fac3bc38c06ea173471f39ce41774f6c5`/
`a17518f8e986d7f4a1bb6b1c75d5eb71672e6e1d61a8f30f72d3c848d18f03fb`/
`b3d47788ab1f4836111df6ed85e90e062024f5e13a6ace76c93fc6bf86ac0baa`/
`b07576130a0c50b89194f2c59467defcf39293d96ca886616865eb198e7965d1`；output/runner启动前均不存在。

唯一训练主PID=`642339`，output=`/home/afr/reid-clean/logs/exp411-pcmpsr-zero-owner-s1234-v1`，runner=
`/home/afr/reid-clean/train-logs/exp411-pcmpsr-zero-owner-s1234-v1.runner.log`。首batch set
loss/positive/negative=`1.970790/58.300308/66.560944`，owner term/unique/fallback=`0/2.2969/0`，formal-vs-correct
owner change=`0`，wrong-RGB/generic/pose-only change=`0.284375/0.171875/0.20625`；与冻结zero-owner合同一致。
首次稳定观测已到e1 iter60，loss finite，GPU约`7000 MiB/41%`且只有该compute PID，严格异常模式计数0。

当前=`UNIQUE ZERO-OWNER E120 RUNNING / SOURCE+CONFIG+CACHE FROZEN / WRONG-RGB NO-START`。只在e10/20/.../120
记录zero-owner、correct与sealed clean D0同epoch轨迹；不按中间结果早停，不修改formal或补跑correct。

## 2026-07-21：zero-owner e10首次正式评测

zero-owner自然完成e10=`28.4 mAP / 38.1 R1 / 53.9 R5 / 60.8 R10`；同epoch sealed correct=
`27.7/36.9/52.8/59.6`，sealed clean D0=`33.4/42.7/59.8/65.2`。rounded四项差为：

| epoch | zero-owner mAP/R1/R5/R10 | correct同epoch | clean D0同epoch | zero−correct | zero−D0 |
|---:|---:|---:|---:|---:|---:|
| 10 | 28.4/38.1/53.9/60.8 | 27.7/36.9/52.8/59.6 | 33.4/42.7/59.8/65.2 | +0.7/+1.2/+1.1/+1.2 | -5.0/-4.6/-5.9/-4.4 |

e10时zero-owner四项均高于correct，却仍全面低于D0；这对owner必要性是暂时不利的warmup证据，但不能代替自然e120
归因裁决。读取时训练已自然进入e11，主PID=`642339`仍为唯一compute PID，GPU约`7,070 MiB/41%`，formal HEAD
仍为`f98ab2daafa294dd0db004e10519363025a45488`且tracked source未变化，严格异常计数0。继续冻结运行，不早停或
修改任何运行内容，wrong-RGB继续`NO-START`。

## 2026-07-21：zero-owner e20正式评测

zero-owner自然完成e20=`45.6 mAP / 55.0 R1 / 70.6 R5 / 75.8 R10`；同epoch sealed correct=
`45.5/55.8/70.0/76.4`，sealed clean D0=`42.2/52.4/67.6/74.0`。截至当前的完整rounded归因轨迹为：

| epoch | zero-owner mAP/R1/R5/R10 | correct同epoch | clean D0同epoch | zero−correct | zero−D0 |
|---:|---:|---:|---:|---:|---:|
| 10 | 28.4/38.1/53.9/60.8 | 27.7/36.9/52.8/59.6 | 33.4/42.7/59.8/65.2 | +0.7/+1.2/+1.1/+1.2 | -5.0/-4.6/-5.9/-4.4 |
| 20 | 45.6/55.0/70.6/75.8 | 45.5/55.8/70.0/76.4 | 42.2/52.4/67.6/74.0 | +0.1/-0.8/+0.6/-0.6 | +3.4/+2.6/+3.0/+1.8 |

e20时zero-owner相对correct呈混合结果：mAP/R5分别高`0.1/0.6`，R1/R10分别低`0.8/0.6`；相对clean D0则
四项均领先。该中间点既不支持提前确认owner必要性，也不支持提前否定，最终仍只按自然e120的mAP与R1严格比较
裁决。读取时训练已进入e21，主PID=`642339`仍为唯一compute PID，GPU约`7,080 MiB/41%`，formal HEAD/tracked
source未变化，严格异常计数0。继续冻结运行，wrong-RGB保持`NO-START`。

## 2026-07-21：zero-owner e30正式评测

zero-owner自然完成e30=`49.2 mAP / 60.3 R1 / 75.0 R5 / 80.0 R10`；同epoch sealed correct=
`50.2/61.8/75.6/80.1`，sealed clean D0=`46.6/56.2/71.3/76.4`。截至当前的完整rounded归因轨迹为：

| epoch | zero-owner mAP/R1/R5/R10 | correct同epoch | clean D0同epoch | zero−correct | zero−D0 |
|---:|---:|---:|---:|---:|---:|
| 10 | 28.4/38.1/53.9/60.8 | 27.7/36.9/52.8/59.6 | 33.4/42.7/59.8/65.2 | +0.7/+1.2/+1.1/+1.2 | -5.0/-4.6/-5.9/-4.4 |
| 20 | 45.6/55.0/70.6/75.8 | 45.5/55.8/70.0/76.4 | 42.2/52.4/67.6/74.0 | +0.1/-0.8/+0.6/-0.6 | +3.4/+2.6/+3.0/+1.8 |
| 30 | 49.2/60.3/75.0/80.0 | 50.2/61.8/75.6/80.1 | 46.6/56.2/71.3/76.4 | -1.0/-1.5/-0.6/-0.1 | +2.6/+4.1/+3.7/+3.6 |

e30时correct四项均高于zero-owner，差为`+1.0/+1.5/+0.6/+0.1`，首次形成一致有利于owner的中间证据；但
zero-owner本身仍全面领先clean D0，且e30不能代替自然e120的mAP/R1严格归因门。读取时训练已进入e31，主PID=
`642339`仍为唯一compute PID，GPU约`7,064 MiB/42%`，formal HEAD/tracked source未变化，严格异常计数0。
继续冻结运行，不早停或修改，wrong-RGB保持`NO-START`。

## 2026-07-21：zero-owner e40正式评测

zero-owner自然完成e40=`55.0 mAP / 66.2 R1 / 79.8 R5 / 84.4 R10`；同epoch sealed correct=
`54.7/66.6/80.0/84.2`，sealed clean D0=`50.0/60.7/76.2/81.0`。截至当前的完整rounded归因轨迹为：

| epoch | zero-owner mAP/R1/R5/R10 | correct同epoch | clean D0同epoch | zero−correct | zero−D0 |
|---:|---:|---:|---:|---:|---:|
| 10 | 28.4/38.1/53.9/60.8 | 27.7/36.9/52.8/59.6 | 33.4/42.7/59.8/65.2 | +0.7/+1.2/+1.1/+1.2 | -5.0/-4.6/-5.9/-4.4 |
| 20 | 45.6/55.0/70.6/75.8 | 45.5/55.8/70.0/76.4 | 42.2/52.4/67.6/74.0 | +0.1/-0.8/+0.6/-0.6 | +3.4/+2.6/+3.0/+1.8 |
| 30 | 49.2/60.3/75.0/80.0 | 50.2/61.8/75.6/80.1 | 46.6/56.2/71.3/76.4 | -1.0/-1.5/-0.6/-0.1 | +2.6/+4.1/+3.7/+3.6 |
| 40 | 55.0/66.2/79.8/84.4 | 54.7/66.6/80.0/84.2 | 50.0/60.7/76.2/81.0 | +0.3/-0.4/-0.2/+0.2 | +5.0/+5.5/+3.6/+3.4 |

e40时zero-owner相对correct再次呈混合结果：mAP/R10分别高`0.3/0.2`，R1/R5分别低`0.4/0.2`，并继续
全面领先clean D0。e30的一致顺序没有延续，说明owner对中间排序指标的作用尚不稳定；不得选择性解释或外推，
最终仍只按自然e120 mAP/R1严格门裁决。读取时训练已进入e41，主PID=`642339`仍为唯一compute PID，GPU约
`7,084 MiB/41%`，formal HEAD/tracked source未变化，严格异常计数0。继续冻结运行，wrong-RGB保持`NO-START`。

## 2026-07-21：zero-owner e50正式评测

zero-owner自然完成e50=`55.1 mAP / 66.1 R1 / 80.0 R5 / 83.7 R10`；同epoch sealed correct=
`54.9/66.1/79.1/83.6`，sealed clean D0=`52.1/62.8/77.0/81.9`。截至当前的完整rounded归因轨迹为：

| epoch | zero-owner mAP/R1/R5/R10 | correct同epoch | clean D0同epoch | zero−correct | zero−D0 |
|---:|---:|---:|---:|---:|---:|
| 10 | 28.4/38.1/53.9/60.8 | 27.7/36.9/52.8/59.6 | 33.4/42.7/59.8/65.2 | +0.7/+1.2/+1.1/+1.2 | -5.0/-4.6/-5.9/-4.4 |
| 20 | 45.6/55.0/70.6/75.8 | 45.5/55.8/70.0/76.4 | 42.2/52.4/67.6/74.0 | +0.1/-0.8/+0.6/-0.6 | +3.4/+2.6/+3.0/+1.8 |
| 30 | 49.2/60.3/75.0/80.0 | 50.2/61.8/75.6/80.1 | 46.6/56.2/71.3/76.4 | -1.0/-1.5/-0.6/-0.1 | +2.6/+4.1/+3.7/+3.6 |
| 40 | 55.0/66.2/79.8/84.4 | 54.7/66.6/80.0/84.2 | 50.0/60.7/76.2/81.0 | +0.3/-0.4/-0.2/+0.2 | +5.0/+5.5/+3.6/+3.4 |
| 50 | 55.1/66.1/80.0/83.7 | 54.9/66.1/79.1/83.6 | 52.1/62.8/77.0/81.9 | +0.2/0.0/+0.9/+0.1 | +3.0/+3.3/+3.0/+1.8 |

e50时zero-owner与correct的R1持平，mAP/R5/R10分别高`0.2/0.9/0.1`，是当前对owner必要性不利的中间
证据；zero-owner仍全面领先clean D0。该点不能代替e120严格双指标归因门，也不授权早停或修改。读取时训练已
进入e51，主PID=`642339`仍为唯一compute PID，GPU约`7,076 MiB/41%`，formal HEAD/tracked source未变化，
严格异常计数0。继续冻结运行，wrong-RGB保持`NO-START`。

## 2026-07-21：zero-owner e60正式评测

zero-owner自然完成e60=`57.6 mAP / 70.3 R1 / 81.0 R5 / 85.2 R10`；同epoch sealed correct=
`57.2/69.6/81.5/84.8`，sealed clean D0=`55.1/66.1/79.0/83.3`。截至当前的完整rounded归因轨迹为：

| epoch | zero-owner mAP/R1/R5/R10 | correct同epoch | clean D0同epoch | zero−correct | zero−D0 |
|---:|---:|---:|---:|---:|---:|
| 10 | 28.4/38.1/53.9/60.8 | 27.7/36.9/52.8/59.6 | 33.4/42.7/59.8/65.2 | +0.7/+1.2/+1.1/+1.2 | -5.0/-4.6/-5.9/-4.4 |
| 20 | 45.6/55.0/70.6/75.8 | 45.5/55.8/70.0/76.4 | 42.2/52.4/67.6/74.0 | +0.1/-0.8/+0.6/-0.6 | +3.4/+2.6/+3.0/+1.8 |
| 30 | 49.2/60.3/75.0/80.0 | 50.2/61.8/75.6/80.1 | 46.6/56.2/71.3/76.4 | -1.0/-1.5/-0.6/-0.1 | +2.6/+4.1/+3.7/+3.6 |
| 40 | 55.0/66.2/79.8/84.4 | 54.7/66.6/80.0/84.2 | 50.0/60.7/76.2/81.0 | +0.3/-0.4/-0.2/+0.2 | +5.0/+5.5/+3.6/+3.4 |
| 50 | 55.1/66.1/80.0/83.7 | 54.9/66.1/79.1/83.6 | 52.1/62.8/77.0/81.9 | +0.2/0.0/+0.9/+0.1 | +3.0/+3.3/+3.0/+1.8 |
| 60 | 57.6/70.3/81.0/85.2 | 57.2/69.6/81.5/84.8 | 55.1/66.1/79.0/83.3 | +0.4/+0.7/-0.5/+0.4 | +2.5/+4.2/+2.0/+1.9 |

e60时zero-owner相对correct的mAP/R1分别高`0.4/0.7`，R10高`0.4`、R5低`0.5`；这使预注册的最终
mAP/R1 owner必要性门在当前中间点双双不成立，归因风险继续上升。zero-owner同时仍全面领先clean D0，说明
三support集合排序本身可能已承载主要性能收益；但上述均不能代替自然e120裁决。读取时训练已进入e62，主PID=
`642339`仍为唯一compute PID，GPU约`7,060 MiB/41%`，formal HEAD/tracked source未变化，严格异常计数0。
继续冻结运行，wrong-RGB保持`NO-START`。

## 2026-07-21：zero-owner e70正式评测

zero-owner自然完成e70=`57.8 mAP / 70.2 R1 / 81.7 R5 / 85.5 R10`；同epoch sealed correct=
`57.4/70.2/81.5/84.9`，sealed clean D0=`55.4/65.2/79.5/83.6`。截至当前的完整rounded归因轨迹为：

| epoch | zero-owner mAP/R1/R5/R10 | correct同epoch | clean D0同epoch | zero−correct | zero−D0 |
|---:|---:|---:|---:|---:|---:|
| 10 | 28.4/38.1/53.9/60.8 | 27.7/36.9/52.8/59.6 | 33.4/42.7/59.8/65.2 | +0.7/+1.2/+1.1/+1.2 | -5.0/-4.6/-5.9/-4.4 |
| 20 | 45.6/55.0/70.6/75.8 | 45.5/55.8/70.0/76.4 | 42.2/52.4/67.6/74.0 | +0.1/-0.8/+0.6/-0.6 | +3.4/+2.6/+3.0/+1.8 |
| 30 | 49.2/60.3/75.0/80.0 | 50.2/61.8/75.6/80.1 | 46.6/56.2/71.3/76.4 | -1.0/-1.5/-0.6/-0.1 | +2.6/+4.1/+3.7/+3.6 |
| 40 | 55.0/66.2/79.8/84.4 | 54.7/66.6/80.0/84.2 | 50.0/60.7/76.2/81.0 | +0.3/-0.4/-0.2/+0.2 | +5.0/+5.5/+3.6/+3.4 |
| 50 | 55.1/66.1/80.0/83.7 | 54.9/66.1/79.1/83.6 | 52.1/62.8/77.0/81.9 | +0.2/0.0/+0.9/+0.1 | +3.0/+3.3/+3.0/+1.8 |
| 60 | 57.6/70.3/81.0/85.2 | 57.2/69.6/81.5/84.8 | 55.1/66.1/79.0/83.3 | +0.4/+0.7/-0.5/+0.4 | +2.5/+4.2/+2.0/+1.9 |
| 70 | 57.8/70.2/81.7/85.5 | 57.4/70.2/81.5/84.9 | 55.4/65.2/79.5/83.6 | +0.4/0.0/+0.2/+0.6 | +2.4/+5.0/+2.2/+1.9 |

e70时zero-owner与correct的R1持平，mAP/R5/R10分别高`0.4/0.2/0.6`，延续对owner必要性不利的中间
证据；zero-owner也继续全面领先clean D0。e50、e60、e70连续显示三support集合排序单独保持强性能，但最终
归因仍只看自然e120的mAP/R1严格比较，不得早停。读取时训练已进入e71，主PID=`642339`仍为唯一compute PID，
GPU约`7,088 MiB/72%`，formal HEAD/tracked source未变化，严格异常计数0。继续冻结运行，wrong-RGB保持
`NO-START`。
