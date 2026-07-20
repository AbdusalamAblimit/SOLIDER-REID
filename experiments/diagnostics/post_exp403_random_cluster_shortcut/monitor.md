# 频率匹配random-cluster CPU诊断监控

> 当前：`EXECUTION V1 SEALED / DIAGNOSTIC INCONCLUSIVE / GPU NO-START`

- 新诊断独立于已封板random source-key v1/v2，不重跑、不修改旧脚本或结果；
- 先完成stdlib-only source与static contract，再允许唯一fresh CPU执行；
- 执行后记录uv/Python、CUDA环境、正反合同、result与source SHA；
- 任何runtime错误只封板本次execution，不就地修补结果。

## 2026-07-20T02:43Z：唯一execution v1封板

- 仓库uv环境、`CUDA_VISIBLE_DEVICES=''`、stdlib-only static contract通过；
- source SHA=`a08d17c35e78c91d96c5b36f2e532ebe151162a2a16db3560be0e41cfb3b4c21`；
- torch未导入，official data/pose/cache/checkpoint/GPU访问均为0；远端GPU保持`2 MiB / 0% / 0 process`。

原始随机簇出现correct/wrong/generic/NULL mAP=
`1.000000000000/0.750006477921/0.040268546661/0.033666906739`，两个margin均通过；频率保持置换也出现
`1.000000000000/0.719609964287/0.040268546661/0.033666906739`。mutant四臂完全相同并被抓住。

但原始assignment的`cluster_7`只覆盖`38`个PID，低于冻结门`>=40`；因此original
`all_gates_pass=false`，正式裁决只能是`DIAGNOSTIC_INCONCLUSIVE`。上述arm顺序不形成科学结论，不降低门槛、
不换seed、不补跑、不建立v2或exp404。唯一执行按冻结科学门自然以exit code `1`退出，不属于runtime错误；
result SHA=`84df7233bbd09e59ac1d1eb27c7ad6c73866c1bbcb6567ac4209636f5c570b17`。
