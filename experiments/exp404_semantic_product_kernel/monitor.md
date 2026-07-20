# exp404 SPK 监控记录

> 当前：`FORMAL E120 RUNNING / MAIN PID 436043 / NO EARLY STOP`

## 2026-07-20：目标降为C类后的机制准入

用户明确要求不要继续按B类主贡献标准过度筛选，目标为C类会议。研究底线与sealed纪律不变；创新准入调整为
“问题/证据明确 + 一个可执行的适度结构贡献”，不再要求张量积或对比学习原子本身首创。

针对性代码审计：

- ICLR 2023 *Identifiability Results for Multimodal Contrastive Learning*官方commit=
  `c1b361f277deeff645c15aa3c3002de8d275003c`。代码用图像/文本独立encoder和双向对称InfoNCE恢复paired
  modalities的共享factor；它没有open-set ReID final-descriptor ownership与random-key controls。
- CITRIS/iCITRIS官方commit=`95f6c90b9ff769ef0250d3a5434b9352853f4302`。代码与README明确依赖temporal
  sequence、已知intervention target及合成可观测causal variables；当前official ReID数据不满足其可识别性前提。

因此exp404只借用“paired shared factor”作为训练直觉，不声称理论可识别；机制对象冻结为无参数SPK final
descriptor，证据对象冻结为wrong/generic/NULL/bypass/random-key/random-cluster完整终审。按C类门槛，
问题/机制/证据=`PASS/PASS-moderate/PASS`，允许进入static CPU，不授权CUDA/GPU。

## 2026-07-20：standalone static CPU正反合同

冻结source SHA=`9739ad1d8388b45922f2ccdb3fec91ffa77c12d6a2e333e75e43c144c10e9e05`，通过仓库
uv环境、`CUDA_VISIBLE_DEVICES=''`连续执行两次。两次均exit `0`，stdout SHA均为
`6b2ca7d88669238cc9f7bebd04ff21567fe5b7f61a0a3f1dbfa549a909b19a64`且byte-exact。

结果为`17/17 PASS`：

- SPK参数数=`0`，固定`12 -> 4 x 3`toy映射exact；
- NULL factor逐元素为1，NULL输出逐元素exact等于global feature；
- correct相对wrong/generic/NULL/random-key/random-cluster的最小positive utility margin=`0.7641115299`；
- global feature/correct evidence梯度范数=`0.6707680955/0.4743046689`，均finite/nonzero；
- unique random-key保持逐样本norm与绝对值多重集；
- random-cluster为8簇、每簇48 sample、48 PID、双camera；
- donor same-camera/different-PID/no-fixed-point；
- evidence-ignored、auxiliary-only、additive-bypass三个mutant全部被抓；
- CUDA前后均未初始化，official data/pose/cache/checkpoint访问0。

result SHA=`6b2ca7d88669238cc9f7bebd04ff21567fe5b7f61a0a3f1dbfa549a909b19a64`。判定：
`STATIC CPU PASS / PRODUCTION IMPLEMENTATION AUTHORIZED / CUDA NO-START / GPU NO-START`。toy utility不作为
ReID性能或semantic ownership结果。

## 2026-07-20：production实现与CPU/source正反合同

生产实现只修改默认关闭的四个目标文件：config增加`SPK_ENABLED=False/SPK_GROUPS=16`；TAPF图使用rich RGB
student evidence与原D0 `PoseSpatialGate`，不含C0 static expert或ELO-CUR router；无参数SPK在BNNeck、
classifier、triplet返回与eval descriptor之前绑定final global feature。构建期新增`SPK_GROUPS==16`显式门，
SPK与ELO-CUR互斥。

production CPU v1真实动态路径及40项门均通过，但源码顺序reporter用全文件`string.index`，误命中构造函数中更早
出现的BNNeck文本，得到`40/41`与`PRODUCTION_CPU_FAIL`。这不是模型失败，但按测量纪律保留v1记录，不覆盖：

- v1 contract SHA=`0716dd5db1521d0b4ecf2ea072c7970aa4e3bb89d06344fa8ce43e053a59a26c`；
- v1 result SHA=`086b627d89052ff21e68878f3636e2fc8c1f96fc0b0d051df605e08365ea1f0c`。

fresh v2仅把该reporter改为AST限定`build_transformer.forward`；绑定行`355`严格早于BNNeck行`364`。连续两次
均为`41/41 PASS`且result byte-exact：

- D0/C0相对preimplementation commit=`07ca01c`的state、初始化RNG、output逐tensor exact；
- SPK参数/缓存数=`0/0`，NULL factor全1且float16 descriptor逐元素identity；
- train classifier、triplet返回、eval before/after BNNeck全部读取同一个bound descriptor；
- direct global/evidence梯度范数=`0.1973796189/0.1360884756`，真实forward shell为
  `0.4505997896/0.2918346226`，均finite/nonzero；
- 两个consumer均为D0 `PoseSpatialGate`，旧C0/ELO router type与state key为0；
- strict reload、optimizer覆盖、teacher/generic-free state与evidence-ignored/aux-only/additive-bypass mutant全部PASS；
- CUDA前后均未初始化。

v2 contract/result SHA=
`766ef5ad65e0ee8cbc2643e320fb5c1f4b247664ce459a2ea834a818a3fe78dd`/
`829fcaad9b9aa88f596b4b3ca51180e6e42ce50d488542ae0f8ebdcc27a4f6c8`。

判定：`PRODUCTION CPU PASS / CUDA PREFLIGHT AUTHORIZED / FORMAL TRAINING NO-START / GPU NO-START`。下一步只
允许创建fresh config与必要CUDA/AMP preflight；尚无ReID性能或semantic ownership结论。

## 2026-07-20：CUDA/AMP preflight静态冻结门

fresh formal config固定Swin-Tiny、seed1234、e120、batch64、workers8与原optimizer/pose监督；只启用
`SPK_ENABLED=True/SPK_GROUPS=16`并关闭ELO-CUR，pretrain/CLIP/codebook均指向fresh exp404 asset目录，
official数据与pose仍分别只读`/mnt1/afrdata`和`/mnt1/afrderived`。

静态contract连续两次`33/33 PASS`且byte-exact：默认GradScaler、独占4090、真实batch64、16组feature/factor
梯度、NULL/random intervention、RGB-only eval、fresh output、no-resume/no-mAP-early-stop等源码门全部通过，
CUDA前后未初始化。

config/preflight/contract/result SHA=
`2bd191ef96da0158a57f917831ea70627f1fef163397219ce1168e3e30bb297d`/
`fb0a21168bef619a561bb77da0a2e5fe9216fde114ea7c34705c3fec544b7fe7`/
`7d8c95896d3c97068060f7bc7795b7b8bc70bf2627d8915a5bafdc996c67e46a`/
`65d8caf2b8e64c7fa6608eaf5842407b67f06ba3872bbccdbcfcfb135631df46`。

远端只读探测显示RTX 4090显存`2 MiB`、利用率`0%`且无compute PID。判定：
`CUDA PREFLIGHT STATIC PASS / CUDA EXECUTION AUTHORIZED / FORMAL TRAINING NO-START / GPU NO-START`。

## 2026-07-20：actual CUDA preflight v1封板与joint-field修复

fresh远端repo commit=`6dc3a034d4eb93b45d7d5fd77ae5574bdb40a359`，关键source/config SHA与本地冻结值
exact；pretrain/CLIP/codebook均复制为fresh link-count=1、mode444资产。fresh uv Python路径为
`/home/afr/reid-clean/runtimes/exp404-spk-py310/bin/python`，56包freeze与
`runtime_requirements.txt` byte-exact，SHA=`3d38c99c7f06502d8b40467d2674c966723e5c913d2edf962c5a7088ec60cddb`。

actual batch64 preflight v1在首个production forward、第一处D0 `PoseSpatialGate`自然失败：gate权重要求17通道，
实际`consumer_field`为5-slot region mask。这是production接线缺陷，不是AMP或数据失败；正式e120未启动，结果文件
未生成，GPU postflight=`2 MiB/0%/0 compute PID`。v1封板记录SHA=
`9958ec661fcaaea20499be04e0450085d76ec3ec5094e8df03179ccff426b498`，判定
`SEALED_INVALID_RUNTIME / V1 NO-RERUN`。

修复只恢复设计中冻结的D0 pose对象：rich anchor同时保留17通道`student_joint_field`，训练期按原D0 handoff混合
teacher/student joint field；SPK子类只把该`consumer_joint_field`送入两个D0 gate，5-slot region mask/evidence仍
用于原监督与最终SPK。

production v3合同连续两次`49/49 PASS`且byte-exact：train/eval joint field shape=`[4,17,4,2]`，region/evidence
仍为`[4,5,4,2]/[4,5,16]`，两个gate真实执行，5通道mutant被抓；旧D0/C0 off-parity及v2全部门保持PASS。
v3 contract/result SHA=
`ce85da278b551a66cacaddd14b3fda79bff356fcee4f7aeff717a927710534ef`/
`56dc8a29957674034c9fb53b0894e686dfbc861c6c7668c3bffda2feed274603`。

fresh CUDA preflight v2 wrapper只委托冻结v1 core并标记新execution；其静态合同连续两次`11/11 PASS`且
byte-exact。wrapper/contract/result SHA=
`2f581913753cc2fc91f02308316433cbe061b16718be1323f8800b744d151b51`/
`224303abf880b670cf8cd694b214d14cdd085b826fdb1443dbedf6249f060fcc`/
`d32a0df0ccbec3c303937c7d4057a542ac0d3adc2b50258d0bc18a600f92a17c`。

判定：`V1 SEALED-INVALID / PRODUCTION V3 PASS / CUDA PREFLIGHT V2 EXECUTION AUTHORIZED / FORMAL
TRAINING NO-START / GPU IDLE`。

## 2026-07-20：actual CUDA v2封板与default-GradScaler v3合同

actual v2完整执行4次AMP attempt，scale按默认GradScaler自然序列
`65536 -> 32768 -> 16384 -> 8192 -> 4096`下降；四次的student evidence、16组bound feature与16组factor
梯度全部finite/nonzero，但每次均有其他参数overflow，optimizer step被跳过，所以`all_evidence_head_updated=false`。
结果`15/26 PASS / CUDA_AMP_PREFLIGHT_FAIL / formal_training_authorized=false`，GPU postflight=`2 MiB/0%/0 PID`。
v2 result SHA=`d49e9421052675193eacb91828918033cbeefcd60a6702d2b31aad82c3a20c29`；禁止同编号重跑。

该序列与sealed exp403 preflight记录一致：默认GradScaler前4次backoff，第5次首次成功更新。因此v3不设置
`init_scale`，不改loss/rho/batch/model/config，也不放宽“必须实际更新”门；只把fresh执行的自然观察窗口冻结为
最多8次，并禁止CLI覆盖。

v3 wrapper/static contract连续两次`14/14 PASS`且byte-exact；同时冻结v2完整FAIL记录、四次scale序列、每次目标
梯度finite/nonzero、production v3 `49/49 PASS`与default `amp.GradScaler()`源码。wrapper/contract/result SHA=
`f4175e3552b06c875144769989fead232dcfd823fd8157bdd4e07561a0a40c87`/
`7930ffeaf4758b0fc677176e71a4663a796a1ad70940ddbd501e1769d8cd3361`/
`10709e126d5187b3331b0b96a3738173e4778936952b5e506e66b8ee4275c245`。

判定：`V2 SEALED-FAIL / DEFAULT-GRADSCALER V3 EXECUTION AUTHORIZED / FORMAL TRAINING NO-START /
GPU IDLE`。

## 2026-07-20：actual CUDA v3与formal prelaunch

fresh v3在第5次attempt成功：前4次scale自然backoff，attempt5保持`4096 -> 4096`并发生evidence-head weight/bias
真实更新。actual结果`26/26 PASS / formal_training_authorized=true`，峰值显存=`6,237,267,456 bytes`。

关键测量：SPK factor mean/std/min/max=`1.0/0.1424809247/0.7198756337/1.3562411070`；correct-vs-NULL/random
descriptor mean-abs=`0.2054978013/0.2366053164`；student evidence/bound feature/factor梯度范数=
`2113.571533/1211.839111/7530.110840`；16个feature/factor group梯度均finite/nonzero；NULL factor全1且
descriptor exact raw，rho exact 0，RGB-only/none-vs-exploding pose exact，state teacher-free，无checkpoint。
result SHA=`70566973f0387d0b335040ff20fe2c1f091563cc18f4a65370b25aac303d58bf`。

formal once-only wrapper冻结fresh output/runner/launch/lock、独占GPU、remote clean、source/config/runtime/preflight
SHA及no-resume。prelaunch static连续两次`15/15 PASS`且byte-exact；wrapper/contract/result SHA=
`5e500fcd67a1ed408141b112d480a5fec2cffe3df5bd545122902f3e53597d86`/
`512dacb626b50d64b80ba7b6e02c15891d1a5f02d96e507bf58330e9048d1750`/
`e2ded956ff1f741b2a4f51bf38bca31234a716f67a21b464f7729d2570ff26c4`。

判定：`CUDA V3 PASS / FORMAL PRELAUNCH PASS / UNIQUE FRESH SEED1234 E120 AUTHORIZED / GPU IDLE`。

## 2026-07-20：唯一formal e120已启动

远端fresh repo提交为`1e40e9a9d1717139b06d09f55821c7f0e68143c7`，启动前repo clean，正式
output/runner/launch/lock均不存在，GPU=`2 MiB/0%/0 compute PID`。远端fresh static结果再次
`15/15 PASS`且与本地两次结果byte-exact，SHA=
`e2ded956ff1f741b2a4f51bf38bca31234a716f67a21b464f7729d2570ff26c4`。

冻结wrapper于`2026-07-20T04:52:15Z`启动唯一fresh seed1234/e120，main PID=`436043`；launch/wrapper SHA=
`78b49f6971c42d73e8bdf4ee5dc0394a9d05090f1213e571b9cad10eea3758e4`/
`5e500fcd67a1ed408141b112d480a5fec2cffe3df5bd545122902f3e53597d86`。启动快照runner SHA=
`723e612b259e3692160a7991cfde97c5ac979eab299374c940dd5531e8d64f8f`；该文件后续持续增长，只作为启动快照。

首次健康检查：唯一compute PID=`436043`，GPU=`8,134 MiB/99%`；训练已到epoch1 iter20/227，loss finite，
SPK factor mean/std/min/max=`1.0000/0.2041/0.5980/1.5249`，descriptor delta abs=`2.868e-01`，无异常。
这里只证明执行健康与SPK active，不作性能判断。

判定：`FORMAL E120 RUNNING / GPU EXCLUSIVE / CONTINUE NATURALLY / NO EARLY STOP`。

## 2026-07-20T05:11Z：formal健康检查

main PID=`436043`存活且仍是唯一compute PID，GPU=`8,134 MiB/98%`。epoch9已自然完成，单epoch约
`125.215 s`、`107.8 samples/s`；检查时进入epoch10 iter20/227。当前loss=`5.804`、所有分项finite，
SPK factor mean/std/min/max=`1.0000/0.0998/0.7807/1.2410`，descriptor delta abs=`1.080e-01`，干预仍active。
日志异常扫描无Traceback、RuntimeError、OOM、NaN或Inf；尚无正式eval/performance结果。

判定：`CONTINUE / EXECUTION HEALTHY / NO PERFORMANCE JUDGMENT`。

## 2026-07-20T05:26Z：formal健康检查

main PID=`436043`仍是唯一compute PID，GPU=`8,104 MiB/99%`。epoch16自然完成，单epoch=`124.440 s`、
`108.0 samples/s`；检查时进入epoch17 iter20/227。当前loss=`3.100`、所有分项finite，SPK factor
mean/std/min/max=`1.0000/0.1003/0.8054/1.2406`，descriptor delta abs=`9.862e-02`，干预保持active。
日志异常扫描继续无Traceback、RuntimeError、OOM、NaN或Inf；尚无正式eval/performance结果。

判定：`CONTINUE / EXECUTION HEALTHY / NO PERFORMANCE JUDGMENT`。

## 2026-07-20T05:41Z：formal健康检查

main PID=`436043`继续独占GPU，显存/利用率=`8,116 MiB/99%`。epoch23自然完成，单epoch=`124.927 s`、
`108.1 samples/s`；检查时进入epoch24 iter20/227。当前loss=`1.339`、所有分项finite，SPK factor
mean/std/min/max=`1.0000/0.1036/0.8207/1.2126`，descriptor delta abs=`9.483e-02`，干预仍active。
日志异常扫描无Traceback、RuntimeError、OOM、NaN或Inf；正式性能仍未产生。

判定：`CONTINUE / EXECUTION HEALTHY / NO PERFORMANCE JUDGMENT`。

## 2026-07-20T05:56Z：formal健康检查

main PID=`436043`仍为唯一compute PID，GPU=`8,102 MiB/98%`。epoch30自然完成，单epoch=`125.380 s`、
`108.2 samples/s`；检查时进入epoch31 iter20/227。当前loss=`0.677`、所有分项finite，SPK factor
mean/std/min/max=`1.0000/0.1062/0.8094/1.2329`，descriptor delta abs=`9.861e-02`，干预继续active。
epoch30冻结中间eval=`47.0 mAP / 57.4 R1 / 71.6 R5 / 77.3 R10`；该数值不用于早停、best-pick或正式
科学裁决。日志异常扫描无Traceback、RuntimeError、OOM、NaN或Inf。

判定：`CONTINUE TO E120 / INTERMEDIATE EVAL NON-DECISIVE / NO EARLY STOP`。

## 2026-07-20T06:11Z：formal健康检查

main PID=`436043`为唯一compute PID，GPU=`8,104 MiB/99%`。epoch37自然完成，单epoch=`124.982 s`、
`108.0 samples/s`；检查时为epoch38 iter60/227。当前loss=`0.415`、所有分项finite，SPK factor
mean/std/min/max=`1.0000/0.1097/0.8184/1.2422`，descriptor delta abs=`1.049e-01`，干预active。
无新增eval；日志异常扫描继续无Traceback、RuntimeError、OOM、NaN或Inf。

判定：`CONTINUE / EXECUTION HEALTHY / NO PERFORMANCE JUDGMENT`。

## 2026-07-20T06:24Z：中间eval显式汇总与健康检查

以下数字逐项读取formal runner log、sealed exp387 clean D0 monitor与sealed exp401 monitor。三者均为
seed1234/e120并在相同epoch评测；差值严格按同epoch计算，后续heartbeat固定报告最新同epoch mAP/R1对比：

| epoch | exp404 mAP/R1 | clean D0同epoch | exp404-D0 | exp401同epoch | exp404-exp401 |
|---:|---:|---:|---:|---:|---:|
| 10 | 32.9/42.3 | 33.4/42.7 | -0.5/-0.4 | 34.4/43.3 | -1.5/-1.0 |
| 20 | 44.9/55.8 | 42.2/52.4 | +2.7/+3.4 | 42.4/54.7 | +2.5/+1.1 |
| 30 | 47.0/57.4 | 46.6/56.2 | +0.4/+1.2 | 45.6/55.9 | +1.4/+1.5 |
| 40 | 50.0/61.1 | 50.0/60.7 | 0.0/+0.4 | 48.6/59.2 | +1.4/+1.9 |

检查时main PID=`436043`仍是唯一compute PID，GPU显存=`8,106 MiB`；训练到epoch44 iter60/227，
loss=`0.360`、SPK factor std=`0.1071`、descriptor delta abs=`1.054e-01`，均finite/active。最新正式中间
eval为epoch40的`50.0 mAP / 61.1 R1`；同epoch相对clean D0为`0.0/+0.4`，相对exp401为
`+1.4/+1.9`。只记录趋势，不用于早停、best-pick或最终裁决。

判定：`CONTINUE TO E120 / E40 VS D0 0.0 mAP +0.4 R1 / NO EARLY STOP`。

## 2026-07-20T06:44Z：e50同epoch对照与健康检查

三条对照均使用总计120 epoch并在e50评测，不以其他epoch替代：

| arm@e50 | mAP | R1 | 相对exp404 |
|---|---:|---:|---:|
| exp404 SPK | 52.5 | 62.5 | 0.0/0.0 |
| sealed clean D0（exp387） | 52.1 | 62.8 | exp404-D0=`+0.4/-0.3` |
| sealed rich route（exp401） | 53.5 | 65.0 | exp404-exp401=`-1.0/-2.5` |

exp404 e50完整R5/R10=`75.9/80.9`。检查时main PID=`436043`仍为唯一compute PID，GPU=
`8,106 MiB/82%`；训练到epoch53 iter160/227，loss=`0.246`，SPK factor std=`0.1068`、descriptor
delta abs=`1.072e-01`，均finite/active。异常扫描无Traceback、RuntimeError、OOM、NaN或Inf。

判定：`CONTINUE TO E120 / E50 VS D0 +0.4 mAP -0.3 R1 / VS EXP401 -1.0 -2.5 / NO EARLY STOP`。

## 2026-07-20T07:00Z：e60同epoch对照与健康检查

三条对照均为总计120 epoch并在e60评测：

| arm@e60 | mAP | R1 | 相对exp404 |
|---|---:|---:|---:|
| exp404 SPK | 54.5 | 64.3 | 0.0/0.0 |
| sealed clean D0（exp387） | 55.1 | 66.1 | exp404-D0=`-0.6/-1.8` |
| sealed rich route（exp401） | 53.5 | 64.8 | exp404-exp401=`+1.0/-0.5` |

exp404 e60完整R5/R10=`78.3/83.0`。评测后已自然进入epoch61 iter20/227，main PID=`436043`仍为
唯一compute PID，显存=`8,126 MiB`；loss=`0.228`，SPK factor std=`0.1074`、descriptor delta abs=
`1.098e-01`，均finite/active。异常扫描无Traceback、RuntimeError、OOM、NaN或Inf。

判定：`CONTINUE TO E120 / E60 VS D0 -0.6 mAP -1.8 R1 / VS EXP401 +1.0 -0.5 / NO EARLY STOP`。

## 2026-07-20T07:29Z：e70同epoch对照与健康检查

三条对照均为总计120 epoch并在e70评测：

| arm@e70 | mAP | R1 | 相对exp404 |
|---|---:|---:|---:|
| exp404 SPK | 56.2 | 67.0 | 0.0/0.0 |
| sealed clean D0（exp387） | 55.4 | 65.2 | exp404-D0=`+0.8/+1.8` |
| sealed rich route（exp401） | 55.2 | 66.1 | exp404-exp401=`+1.0/+0.9` |

exp404 e70完整R5/R10=`79.4/84.3`。检查时训练到epoch74 iter200/227，main PID=`436043`仍为唯一
compute PID，GPU=`8,102 MiB/98%`；loss=`0.177`，SPK factor std=`0.1083`、descriptor delta abs=
`1.078e-01`，均finite/active。异常扫描无Traceback、RuntimeError、OOM、NaN或Inf。

判定：`CONTINUE TO E120 / E70 VS D0 +0.8 mAP +1.8 R1 / VS EXP401 +1.0 +0.9 / NO BEST-PICK`。
