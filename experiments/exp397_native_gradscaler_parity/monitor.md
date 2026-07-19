# exp397 matched native GradScaler动态轨迹监控

## 状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / STATIC-CPU SEALED-PASS /
CUDA NATIVE-PARITY SEALED-FAIL / NATIVE_GRADSCALER_PARITY_FAIL /
FORMAL NO-START`

## 2026-07-19 接手

- exp396完整矩阵status=`PASS`，outcome=`SHARED_D0_OR_RUNTIME_NONFINITE`；
- D0/rich `reid/total`仅backbone共享`368/3,753/4,183` NaN/+Inf/-Inf，其余aux finite；
- 这不推翻exp394原门FAIL，但证明绝对首步finite缺少matched D0校准；
- exp397唯一变量是让default GradScaler按原生`step/update`运行并比较D0/rich轨迹；
- 当前只冻结design/protocol，尚未实现，GPU空闲，formal训练`NO-START`。

## 2026-07-19 implementation与static/CPU

- 固定12 attempts、e1×6→e6×6、单一materialized rich loader、matched per-step RNG；
- D0/rich各自fresh default GradScaler，不传任何scale/growth/backoff参数；
- 每attempt唯一`scale/backward/unscale/report/step/update`，无scheduler/checkpoint；
- static首遍/repeat均21/21 PASS，四份result/runner逐字节exact；
- matched synthetic=`11/12`成功、attempt2首个成功、trajectory PASS；
- rich extra skip、late first success、e6 handoff failure、rich-specific non-finite均准确判FAIL；
- CUDA initialized before/after=`false/false`。

冻结SHA256：implementation=
`4ad2c40a8d679e8dd52619d9216016aaecdc0fd6530d7ca679e0bb16b7cfa9ba`，static=
`99ad9a0d34db4bcbc0816ecd05c62d361322f47d214bca21c9927f92738269dd`，result/runner/repeat=
`82d52315d1472e996fc50f330d332853c2e025ecf1c333651aca6cd7385f06eb`。

裁决：`STATIC_CPU_SEALED_PASS / CUDA NATIVE-PARITY FRESH-EXECUTION GO`；直接推进fresh actual。

## 2026-07-19 唯一CUDA native parity actual

- fresh execution repo=
  `/home/afr/SOLIDER-REID-exp397-native-gradscaler-fresh-11d7a35`，HEAD=
  `11d7a35788c4645c355d96d76a2a4ff20a9801ac`且tracked clean、无alternate；
- production脚本、exp396 reporter、fresh regular CLIP/codebook、runtime freeze SHA逐项exact；启动前GPU=
  `2 MiB/0%`且无compute process；
- D0/rich轨迹逐attempt完全一致：attempt 1–5依次把scale从`65,536`原生backoff到`2,048`且均skip，
  attempt 6首次成功；e6的attempt 7在`2,048`再次matched skip并降到`1,024`，attempt 8–12成功；
- 两臂均只有`6/12`次optimizer update，首个成功均为attempt 6；因此冻结的`>=10/12`、首个成功
  `<=3`、首次成功后全finite和e6六步全success四项门失败；
- `matched_scale_skip_trajectory`、native step语义、12行完整和rich-specific 11组全程finite均PASS；所有
  non-finite只出现在两臂相同attempt的shared backbone；
- source/runtime/assets、common initial state、parameter coverage、target、matched RNG、teacher/codebook
  state/version、scratch cleanup与checkpoint zero全部PASS；
- elapsed=`20.892324913293123 s`，peak memory=`7,907,269,120 bytes`，checkpoint=`0`、scratch=`0`；
- result/runner/manifest/stdout SHA256=
  `eef02328fb4026459fa28a7095d8d5c7b5703834e25ba950a78d9a3f1978faa2`/
  `eef02328fb4026459fa28a7095d8d5c7b5703834e25ba950a78d9a3f1978faa2`/
  `a9c2acd912f57a7020e129c59c4b24b615d1e4065f0bf482f71ad631eb7b3c51`/
  `e23e8ea0a23c7217cefc3f526dba7829f1b563ab62dcc23596aac6118404746e`；
- 进程自然退出，GPU恢复`2 MiB/0%`且无compute process；没有Traceback、RuntimeError、OOM或AMP
  warning。

裁决：`CUDA NATIVE-PARITY SEALED-FAIL / NATIVE_GRADSCALER_PARITY_FAIL / FORMAL NO-START`。该FAIL
不得按matched事实事后改判，也不得补跑或放宽exp397门；它同时不能被解释成rich-specific不稳定，因为
D0与rich skip/scale轨迹exact且rich-specific组全finite。后续只能另立新编号、按production-shaped
baseline-relative语义预注册新门。
