# 实验 exp398：production-shaped baseline-relative AMP稳态门

## 当前状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / STATIC-CPU SEALED-PASS /
CUDA EXECUTION SEALED-INVALID / GROUP_STATE_REPORTER_RUNTIME_FAIL /
FORMAL NO-START`

## 动机

exp397必须按预注册绝对门判FAIL：D0与rich均只有`6/12`次update、首个成功为attempt 6，且e6首步再次
skip。但它同时给出两个不能丢失的事实：两臂skip/scale轨迹逐项exact，rich-specific 11组全程finite。
因此不能通过放宽exp397的`>=10/12`或`<=3`门来救活，也不能把同一现象归为rich-specific不稳定。

正式trainer使用一个default GradScaler贯穿大量batch，真正需要的新证据不是“默认scale在前三步就成功”，
而是：在不手工改scale的情况下，每个production阶段能否自然进入连续稳态；rich是否相对matched D0增加
任何skip；nonzero-rho阶段的生产组是否真实更新且保持finite。

## 核心假设

若rich production图没有引入额外AMP不稳定性，则在同一32个official batch64、相同初始common state、
相同step RNG和各自fresh default GradScaler下：

1. e1和e6各16步窗口的最后8步，两臂都应连续success且全部group finite；
2. 任一D0成功的attempt，rich不得额外skip；rich总成功数不得少于D0；
3. rich发生non-finite时，其group必须是同attempt D0也non-finite的shared group；
4. rich-specific 11组全程finite，并在e6成功步中有非零梯度且最终state相对initial改变；
5. scaler只能通过原生`step/update`适应，不得改initial scale或补步。

该门不重判exp397，也不直接启动e120。PASS只授权另立最终production CUDA preflight；FAIL只关闭当前
rich production baseline-relative稳态接口。

## 冻结对象

- source commit=`11d7a35788c4645c355d96d76a2a4ff20a9801ac`；
- source/config、canonical runtime、CLIP、full codebook与exp397 exact；
- exp396 chunk-safe reporter SHA=
  `6e8b2b67efcda7cfaca8527fb0ae1dd4c6aedcebef3fec6ded2e5ba6ddab8164`；
- `ATTEMPTS=32`，attempts 1–16=`tapf_epoch=1`、17–32=`tapf_epoch=6`；
- 每阶段稳态尾窗固定`8`步，不能按actual结果修改；
- 单一rich loader一次materialize前32个official batch，D0/rich复用CPU clone；
- D0/rich各自fresh model、optimizer与`torch.amp.GradScaler("cuda")`；
- 不传`init_scale/growth/backoff`参数，不修改scaler内部state；
- 每attempt严格一次`scale→backward→unscale→report→step→update`；
- 允许native skip，不允许retry、补step、gradient clipping、scheduler、checkpoint load/save。

## 预注册PASS门

1. source/runtime/assets/config/batch/teacher/common init、RNG和parameter coverage exact；
2. D0/rich各32行完整，initial scale=`65,536`，native skip/backoff语义exact；
3. e1最后8步与e6最后8步，两臂全部success且所有15组finite；
4. `rich_extra_skip_on_d0_success=0`，rich successful updates不少于D0；
5. rich任一non-finite group都必须在同attempt D0 non-finite group集合内；
6. rich-specific 11组全程finite；每组在至少一个e6成功步中有非零梯度，initial/final state SHA不同；
7. teacher/codebook state/version exact，checkpoint=`0`、scratch=`0`；
8. process自然退出、GPU空闲、result/runner/manifest SHA封板。

任一FAIL即封板停止，不调参、不重跑。PASS=`BASELINE_RELATIVE_STEADY_STATE_PASS`，仅授权新编号的最终
production preflight，formal训练仍`NO-START`。

## 风险与失败解释

32步不是retrieval实验，不能证明最终性能。尾窗门不要求D0/rich在参数已分叉后逐step轨迹exact，而要求
rich不在D0成功处新增skip，并保留shared-only non-finite归属。这样既不把exp397阈值事后降到刚好能过，
也不把baseline自身的default-scale适应误记为rich缺陷。

## static/CPU封板

production脚本与独立contract连续两遍24/24 PASS且result/runner逐字节一致。AST确认32步、e1/e6各16、
尾窗8、default GradScaler无override、单一materialized loader、matched RNG、每attempt一次step/update、
15组与11个rich-specific组、group state前后SHA、无scheduler/checkpoint。synthetic shared warm-up PASS、
rich better PASS；extra skip、persistent tail skip、rich-only non-finite和rich-specific inactive均准确FAIL。
CUDA initialized before/after=`false/false`。

implementation/static/result SHA256=
`8bc599e94264eb3fb89b3cdc94810c483f4c4a037ebe86311c30a741011aeac9`/
`b137fadaf0463ae51eb2e552945cf87923a2c788582dbf0a4aaf00e296829414`/
`d7efa6894411f7b7433c8819422e14c2495110484544d86b9b21083d0bb24317`；四份result/runner同SHA。
裁决=`STATIC_CPU_SEALED_PASS / CUDA BASELINE-RELATIVE FRESH-EXECUTION GO`。

## actual封板

唯一fresh actual通过source/runtime/assets、official 32-batch materialize与teacher target前置阶段，但在
D0 arm进入首个forward前，新增`parameter_group_state()`把reporter返回的`(name, parameter)`元组当成
tensor调用`.detach()`，触发`AttributeError`。异常发生在任何backward/scaler.step之前，optimizer
update=`0`、checkpoint=`0`、scratch=`0`；没有产生D0/rich轨迹，不能回答稳态假设。

result/runner/manifest/stdout SHA=
`71a943e6a233999549f69c1ece2ce1c2c3e507c69d9e99364272442d9b6ac998`/
`71a943e6a233999549f69c1ece2ce1c2c3e507c69d9e99364272442d9b6ac998`/
`b719b3acdec3746dae8f602fc526564a08047ae5ad1a9e2c3a3865a973c2b12e`/
`745ca9f364324f091100eab01d76ccff8d1f44fa276fe554dbee079ef92ba43a`。裁决=
`CUDA EXECUTION SEALED-INVALID / GROUP_STATE_REPORTER_RUNTIME_FAIL / FORMAL NO-START`；禁止修补或重跑
exp398。
