# exp399 named-parameter state reporter exact监控

## 状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / IMPLEMENTATION STATIC SEALED-PASS /
STATIC-CPU SEALED-PASS / CUDA FRESH-EXECUTION GO / FORMAL NO-START`

## 2026-07-19 接手

- exp398 actual在首个forward前因tuple reporter INVALID，update=`0`、checkpoint=`0`；
- exp399仅修named-parameter state SHA及真实container static contract；
- 32-step科学门、source/runtime/assets和所有阈值不变；GPU=`2 MiB/0%`、无compute process。

## static初次复现性FAIL

- 两遍各35项逻辑gate均PASS、CUDA未初始化；
- 但tiny real-container model未固定CPU seed，导致两遍initial/final group SHA不同，逐字节复现门FAIL；
- 归因明确，只在static contract入口固定`torch.manual_seed(3991234)`；production脚本与科学门未改；
- 初始两遍result/runner保留，不覆盖；修正后重新执行正式static v1/repeat。

## 正式static/CPU

- 固定CPU seed后v1/repeat均35/35 PASS，四份result/runner逐字节exact；
- 真实15组named tuple、optimizer coverage和单group mutation exact；
- name/order/value binding、空组、duplicate/empty/bare/wrong-length/nonparameter反例全部PASS；
- exp398的32-step AST与六类trajectory synthetic门保持PASS；CUDA未初始化。

SHA：implementation=
`b9da4346b0d74d13b537bd7fa3f5eff1e65b0b6e512014026800506807723907`，static=
`7948845f1600141302285cee12c025cbf0ba50faa1af01d1fb298bd3aa558810`，result/runner=
`32adc18d2b6dc06c0d3ea37ca6003d749a2ff2540efefdbca0e35e1fba2f0d98`。

裁决：`STATIC_CPU_SEALED_PASS / CUDA FRESH-EXECUTION GO`；直接推进唯一actual。
