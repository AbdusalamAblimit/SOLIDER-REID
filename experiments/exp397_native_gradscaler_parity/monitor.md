# exp397 matched native GradScaler动态轨迹监控

## 状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / STATIC-CPU SEALED-PASS /
CUDA NATIVE-PARITY FRESH-EXECUTION GO / FORMAL NO-START`

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
