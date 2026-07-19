# exp398 production-shaped baseline-relative AMP稳态监控

## 状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / STATIC-CPU SEALED-PASS /
CUDA EXECUTION SEALED-INVALID / GROUP_STATE_REPORTER_RUNTIME_FAIL /
FORMAL NO-START`

## 2026-07-19 接手

- exp397按冻结绝对门`SEALED-FAIL`，禁止改判或重跑；
- exp397同时证明D0/rich动态轨迹exact、rich-specific 11组全程finite；
- exp398只测试更长production-shaped阶段的baseline-relative稳态，不调scale/loss/batch；
- 当前仅完成design/protocol冻结，尚未实现，GPU=`2 MiB/0%`且无compute process。

## 2026-07-19 implementation与static/CPU

- 固定32 attempts、e1×16→e6×16、每阶段8步稳态尾窗；
- D0/rich各自fresh default scaler，baseline-relative extra-skip/shared-subset判定；
- rich-specific 11组增加e6非零梯度和initial/final state变化门；
- static首遍/repeat均24/24 PASS，四份result/runner逐字节exact；
- shared warm-up与rich-better synthetic PASS；extra skip、persistent tail、rich-only non-finite、inactive组
  四类反例准确FAIL；
- CUDA initialized before/after=`false/false`，checkpoint=`0`。

冻结SHA256：implementation=
`8bc599e94264eb3fb89b3cdc94810c483f4c4a037ebe86311c30a741011aeac9`，static=
`b137fadaf0463ae51eb2e552945cf87923a2c788582dbf0a4aaf00e296829414`，result/runner/repeat=
`d7efa6894411f7b7433c8819422e14c2495110484544d86b9b21083d0bb24317`。

裁决：`STATIC_CPU_SEALED_PASS / CUDA BASELINE-RELATIVE FRESH-EXECUTION GO`；按持续授权直接推进actual。

## 2026-07-19 唯一CUDA actual

- fresh repo=`/home/afr/SOLIDER-REID-exp398-baseline-relative-fresh-11d7a35`，HEAD exact且tracked clean；
- production/reporter/CLIP/codebook/runtime SHA exact，启动前GPU=`2 MiB/0%`；
- official 32 batches与teacher targets完成前置物化；
- D0 `run_dynamic_arm`在首个forward前计算initial group state时，parameter列表元素实际为
  `(parameter_name, parameter)`，新增hasher直接调用tuple的`.detach()`而失败；
- exception=`AttributeError: 'tuple' object has no attribute 'detach'`；没有backward、scaler.step或
  optimizer update，checkpoint=`0`、scratch=`0`；
- result/runner/manifest/stdout SHA=
  `71a943e6a233999549f69c1ece2ce1c2c3e507c69d9e99364272442d9b6ac998`/
  `71a943e6a233999549f69c1ece2ce1c2c3e507c69d9e99364272442d9b6ac998`/
  `b719b3acdec3746dae8f602fc526564a08047ae5ad1a9e2c3a3865a973c2b12e`/
  `745ca9f364324f091100eab01d76ccff8d1f44fa276fe554dbee079ef92ba43a`；
- 进程自然退出，GPU恢复`2 MiB/0%`且无compute process。

裁决：`CUDA EXECUTION SEALED-INVALID / GROUP_STATE_REPORTER_RUNTIME_FAIL / FORMAL NO-START`。这只否定
exp398测量器的真实parameter container契约，不提供AMP稳态或rich相对D0证据。禁止修补/重跑；若继续
只能另立新编号，并在static/CPU中加入named-parameter tuple exact测试。
