# 实验 exp397：matched native GradScaler动态轨迹门

## 当前状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / STATIC-CPU SEALED-PASS /
CUDA NATIVE-PARITY SEALED-FAIL / NATIVE_GRADSCALER_PARITY_FAIL /
FORMAL NO-START`。

## 动机

exp396完整matrix证明exp394首步non-finite并非rich auxiliary特有：matched D0与rich的`reid/total`只在
backbone产生完全相同的368 NaN、3,753 `+Inf`和4,183 `-Inf`，其余D0 pose与rich全部auxiliary均finite。
因此绝对要求default GradScaler初始scale的第一步finite，会同样拒绝clean D0，不能用来判断rich是否
额外不稳定。

GradScaler的canonical语义本来就是：若unscale后检测到non-finite，则跳过optimizer step并在
`update()`中自然降低scale；后续finite时才执行更新。下一门不能手工降低initial scale或删除loss，而应
比较D0与rich从同一默认状态出发时的原生skip/scale/成功更新轨迹。

## 核心假设

若exp394 production新增图没有额外AMP不稳定性，则在同一12个official batch、同一初始common state、
相同step RNG和default GradScaler下：

1. D0与rich的skip/success及scale-before/after轨迹应完全一致；
2. 允许共享ReID backbone造成相同的初始skip，但rich不得增加skip或延迟首个成功update；
3. rich-specific evidence/head/router/expert组在所有attempt中都必须finite；
4. e6 handoff阶段必须全部finite并成功更新，证明非零rho没有引入新overflow。

该门只判断relative AMP execution parity，不做检索评测，也不产生可续训checkpoint。

## 冻结对象

- source commit=`11d7a35788c4645c355d96d76a2a4ff20a9801ac`；
- canonical runtime、CLIP、full codebook与source/config SHA沿用exp396；
- exp396 chunk-safe reporter dependency SHA=
  `6e8b2b67efcda7cfaca8527fb0ae1dd4c6aedcebef3fec6ded2e5ba6ddab8164`；
- official first 12 batch64由同一个rich loader一次性materialize，D0/rich复用逐tensor clone；
- schedule固定attempt 1–6=`tapf_epoch=1`，attempt 7–12=`tapf_epoch=6`；
- 每个arm fresh model、fresh optimizer、fresh default `torch.amp.GradScaler("cuda")`；
- 不设置`init_scale/growth_factor/backoff_factor/growth_interval`，不直接修改scaler内部state；
- 两arm每attempt恢复同一预生成CPU/CUDA/NumPy/Python RNG state；
- 每attempt只执行一次`scale -> backward -> unscale -> group capture -> scaler.step -> scaler.update`；
- 不调用scheduler/EMA，不加载/保存checkpoint。

## 更新与记录

与exp396 zero-update不同，exp397必须允许GradScaler在finite attempt调用optimizer step，否则无法验证原生
动态语义。每attempt记录：

- total及全部component scalar；
- scale before/after；
- 15组unscaled finite/NaN/±Inf/range；
- parameter `_version`与optimizer state before/after；
- optimizer step是否真实发生、skip是否与non-finite一致；
- model/optimizer state SHA；
- epoch、batch manifest SHA、teacher target SHA；
- scratch清理与peak memory。

成功update只存在内存中；结束后不写checkpoint，不能续训或转正式臂。

## 预注册有效门

固定12 attempts、每arm独立运行。actual只有同时满足以下条件才PASS：

1. source/runtime/assets/config/batch/teacher与initial common state exact；
2. D0/rich各12行完整，scale初值均exact `65536`；
3. 每行`had_nonfinite == optimizer_skipped`；skip时scale只由native update减半，success时在固定12步内保持；
4. D0/rich `skip/success + scale before/after`逐attempt完全一致；
5. 两arm成功optimizer update均至少10/12，首个成功attempt相同且不晚于3；
6. 首个成功update之后不再出现任何group non-finite；
7. rich-specific 11组（mask/presence/evidence及两个router T/C/E/Expert）全程finite；
8. e6六步两arm全部success且所有group finite；
9. teacher/codebook state与version exact，step RNG入口逐arm一致；
10. checkpoint=`0`、scratch=`0`、进程退出、GPU空闲、result/runner/manifest SHA封板。

若轨迹不一致，FAIL只关闭当前production dynamic parity；若两者一致，则只授权新的production CUDA
preflight设计，不直接授权e120。

## static/CPU门

CUDA前必须证明：

- source/exp396 dependency SHA exact；
- AST中default scaler无任何override，`step/update`每attempt各一次；
- 无scheduler/checkpoint/formal GO；
- 12-batch、1×6→6×6 schedule、15组与两arm顺序exact；
- synthetic trajectory evaluator对matched PASS、rich extra skip FAIL、late success FAIL、handoff FAIL、
  rich-specific non-finite FAIL的裁决exact；
- CUDA隐藏且before/after未初始化；
- 两遍result/runner逐字节一致。

## 风险与失败解释

本门有真实optimizer update，但只限12-step内存preflight且不保存权重。D0/rich在成功update后因auxiliary
不同而参数轨迹分叉是预期，不要求state相等；只要求数值skip/scale parity和rich-specific finite。
若native trajectory PASS，只说明旧绝对首步门过严且rich不比D0差，不证明final retrieval有效。

## static/CPU封板

production脚本与独立contract连续两遍21/21 PASS且result/runner逐字节一致。AST确认12 attempts、
e1×6→e6×6、default GradScaler无任何override、每attempt只调用一次`step/update`、unscale/report顺序、
单一materialized loader、matched RNG、无scheduler/checkpoint/formal GO。synthetic matched轨迹PASS；
rich extra skip、首个成功过晚、handoff失败、rich-specific non-finite四类反例全部按冻结规则FAIL。
CUDA initialized=`false/false`。

implementation/static/result SHA256=
`4ad2c40a8d679e8dd52619d9216016aaecdc0fd6530d7ca679e0bb16b7cfa9ba`/
`99ad9a0d34db4bcbc0816ecd05c62d361322f47d214bca21c9927f92738269dd`/
`82d52315d1472e996fc50f330d332853c2e025ecf1c333651aca6cd7385f06eb`；runner与repeat同SHA。

裁决=`STATIC_CPU_SEALED_PASS / CUDA NATIVE-PARITY FRESH-EXECUTION GO`。提交后按用户持续授权直接执行
唯一actual；formal训练仍`NO-START`。

## actual封板

唯一fresh CUDA actual完成全部D0/rich各12行。两臂轨迹exact：e1前五步从默认`65,536`连续backoff，
attempt 6在`2,048`首次成功；切到e6后attempt 7再次matched skip到`1,024`，其余五步成功。两臂均仅
`6/12`次update、首个成功为attempt 6，违反预注册的`>=10/12`、`<=3`和e6全success门；因此status=
`FAIL`。rich-specific 11组始终finite，所有non-finite均为shared backbone且发生在相同attempt，这只
限制失败解释，不能推翻冻结裁决。

result/runner/manifest SHA256=
`eef02328fb4026459fa28a7095d8d5c7b5703834e25ba950a78d9a3f1978faa2`/
`eef02328fb4026459fa28a7095d8d5c7b5703834e25ba950a78d9a3f1978faa2`/
`a9c2acd912f57a7020e129c59c4b24b615d1e4065f0bf482f71ad631eb7b3c51`。最终裁决=
`CUDA NATIVE-PARITY SEALED-FAIL / FORMAL NO-START`；禁止重跑、补步或调整exp397门。
