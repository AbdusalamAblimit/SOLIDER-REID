# exp408 PICRD 监控

> 当前：`STUDENT RUNNING / FRESH SEED1234 E120 / DO NOT MODIFY OR EARLY-STOP`

## 2026-07-21：从 CAVT 切到直接训练对象

exp407 已按 `FORMAL SEALED-FAIL / VALIDITY FAILURE / SCIENCE NOT EVALUATED`封板，4090空闲。CAVT
不作科学否定，但连续三次消耗在测量合同后从活动主线移除；exp408禁止继续修donor/matcher。

代码审计确认旧rich路线的relation监督被`source_feature.detach()`和`hidden.detach()`双重阻断，且旧evidence
由全图GAP生成。近期论文/开源审计未发现PICRD整体同构实现，但pose+CLIP、part KD和batch relation原子高度拥挤，
故只定位为C类候选。最终冻结机制为逐槽跨batch relation、correct-vs-wrong/generic/zero训练内排序和Stage-2
直接反传；若退化为普通part KD则不运行。

下一步只实现必要代码和最小正反/梯度/CUDA-AMP检查，然后进入一次独立盲审。当前未创建cache、未启动GPU，
没有新增mAP/R1。

## 2026-07-21：设计盲审首轮1B/2H并修正

首轮审查发现四臂若各自使用valid集合，关系排序可能由缺失模式产生；另有负臂反传会造成循环论证、canonical
cache和e120 batch未冻结两项HIGH。设计现已修正为四臂共同`V_common`、control距离stop-gradient、固定
wrong offset=4、deterministic resize cache及16 PID×4图诊断manifest。等待代码实现后的独立盲审；GPU仍NO-START。

独立代码盲审首轮为`0B/1H`：processor外层AMP会让声明为FP32的Gram matmul/einsum重新落到FP16。已把
PICRD mask/pooling/relation整个块显式置于`autocast(enabled=False)`并把source/teacher转FP32；等待同一审查者闭环。

同一独立审查者已闭环为`0 BLOCKER / 0 HIGH`。本地语法/config merge、逐槽shape、共同valid、构造正例四臂顺序
和source非零有限梯度均PASS；没有启动GPU。下一步只做一次固定MMPOSE-ABU真实model CUDA/AMP update，然后立即
fresh生成cache并冻结SHA，不追加其它static。

固定MMPOSE-ABU真实batch64 CUDA/AMP检查PASS：PICRD块为FP32，`stage_grad_tensors=142`，首个
`base.patch_embed.projection.weight`梯度绝对和=`49.140007`，默认GradScaler=`65536`且一次optimizer真实更新。
四臂初值correct/wrong/generic/zero=`0.727728/0.727611/0.037120/0.725649`，generic明显更近，说明control
不是装饰性弱臂；训练必须实际扭转顺序。检查后GPU=`2 MiB/0%`。

首次cache-v1后台调用在顶层`from datasets...`立即因repo根未进入`sys.path`退出；log仅237 bytes，未读official/
pose、未初始化CUDA、未创建cache/diagnostic。v1目录冻结不复用。修复只在import前加入脚本解析出的repo root，
cache/config改用fresh `exp408-picrd-cache-v2`；等待同一独立审查者聚焦闭环后立即执行。

入口修复聚焦复审=`0B/0H`。fresh cache-v2现已启动，运行仓库=
`/home/afr/SOLIDER-REID-exp408-picrd-0700703-v3`，远端HEAD=`021aa359e7a28dd7d814382e9fd4ca1386b91558`，
asset/output=`/home/afr/reid-clean/assets/exp408-picrd-cache-v2`，主PID=`465720`。首次有效观测已从
`8/15,618`推进到`1,000/15,618`，GPU=`2,186 MiB/99%`，无异常。当前只监控自然完成，不修改运行中源码；
cache发布并核验SHA后立即写入config并启动student。

cache-v2已自然完成并通过固定loader核验：official train `15,618`张路径完整唯一覆盖，embedding为
`[15618,5,768]` FP16且finite；五槽valid计数=`[15616,15618,15618,15618,15586]`。冻结64图diagnostic由
`16 PID x 4`连续排列组成，offset `4`全部为different-PID，四臂共同valid五槽均为`64/64`。cache SHA256=
`80db6448a38745a7846bbb1ffb63d868b4efcda8851bc069cd8166dc311cebee`，diagnostic manifest SHA256=
`8ef842f98a1172d7c8c197828cb3d4fda2006ced52062c9608569da5be62cff8`；核验后GPU=`2 MiB/0%/0 compute PID`。
因此不再追加preflight，冻结config SHA并授权唯一fresh seed1234 student自然运行至e120。

唯一student已于远端fresh output启动：repo=`/home/afr/SOLIDER-REID-exp408-picrd-0700703-v3`，source HEAD=
`86496f0062d7553062567e7d2bbcb371a24ef500`，output=`/home/afr/reid-clean/logs/exp408-picrd-s1234-v1`，
runner log=`/home/afr/reid-clean/train-logs/exp408-picrd-s1234-v1.runner.log`，主PID=`466984`。启动前GPU无compute
PID、output/runner均fresh；固定MMPOSE-ABU实际加载15,618样本cache及冻结SHA，batch=`64`。首batch
loss/correct/wrong/generic/zero/rank=`0.615141/0.022584/0.023626/0.042036/0.727439/0.592558`，
shift=`4`，common-valid=`1.0000`，均finite；首观测GPU约`7,000 MiB/89%`且只有该主PID。epoch 1自然完成，
无异常并进入epoch 2；当前只监控自然e120和e10/20/...评测，不修改运行中源码/config，不按中间性能早停。

首个正式中间评测已自然完成：e10 PICRD=`32.0 mAP / 42.8 R1 / 57.7 R5 / 63.9 R10`；sealed clean D0
同epoch=`33.4/42.7` mAP/R1，因此rounded差=`-1.4 mAP / +0.1 R1`。检查时主PID仍唯一、GPU约
`7,002 MiB/92%`，runner中Traceback/RuntimeError/OOM/NaN/Inf计数为0，训练已继续进入e11。该点只记录
轨迹，不作早停或机制裁决；最终仍只看自然e120双门与冻结diagnostic顺序。

## 2026-07-21：e20--e40同epoch轨迹

| epoch | PICRD mAP/R1 | sealed clean D0同epoch | PICRD-D0 |
|---:|---:|---:|---:|
| 10 | 32.0/42.8 | 33.4/42.7 | -1.4/+0.1 |
| 20 | 44.0/55.1 | 42.2/52.4 | +1.8/+2.7 |
| 30 | 47.3/57.4 | 46.6/56.2 | +0.7/+1.2 |
| 40 | 50.6/61.8 | 50.0/60.7 | +0.6/+1.1 |

e20/30/40完整R5/R10分别为`70.7/76.7`、`71.9/76.9`、`76.4/81.4`。检查时训练已进入e42，主PID
`466984`仍为唯一compute PID，GPU约`7,112 MiB/88%`，异常计数仍为0。首batch逐epoch的correct距离通常略低于
wrong-RGB，但差值仍小，generic/zero明显更远；只记为objective active，不把中间顺序或性能写成科学GO。
继续自然运行至e120，不修改源码/config或按当前领先早停。

## 2026-07-21：e50--e70同epoch轨迹

| epoch | PICRD mAP/R1 | sealed clean D0同epoch | PICRD-D0 |
|---:|---:|---:|---:|
| 50 | 53.0/63.0 | 52.1/62.8 | +0.9/+0.2 |
| 60 | 53.2/63.8 | 55.1/66.1 | -1.9/-2.3 |
| 70 | 55.6/66.2 | 55.4/65.2 | +0.2/+1.0 |

e50/60/70完整R5/R10分别为`76.8/81.5`、`77.5/82.0`、`79.5/84.0`。e60出现中间回落，e70又恢复
双指标领先，说明单点波动不能用于早停或改机制。检查时已进入e75，主PID仍唯一，GPU约`7,070 MiB/90%`，
异常计数为0；PICRD四臂和common-valid持续finite/active。继续自然e120。
