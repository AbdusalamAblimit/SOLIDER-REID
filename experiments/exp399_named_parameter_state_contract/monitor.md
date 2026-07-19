# exp399 named-parameter state reporter exact监控

## 状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / STATIC-CPU SEALED-PASS /
CUDA BASELINE-RELATIVE SEALED-PASS / PRODUCTION PREFLIGHT GO /
FORMAL NO-START`

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

## 唯一CUDA actual

- fresh repo=`/home/afr/SOLIDER-REID-exp399-named-state-fresh-11d7a35`，HEAD exact、tracked clean、无
  alternate；production/reporter/CLIP/codebook/runtime SHA exact；
- D0/rich各32行完整，status=`PASS`，outcome=`BASELINE_RELATIVE_STEADY_STATE_PASS`；
- attempts 1–5与7仅shared backbone skip，两臂scale轨迹exact；attempts 8–32连续success；
- D0/rich均`26/32` update、first success=6；extra rich skip=`0`、rich-only non-finite=`0`；
- `d0/rich × e1/e6`四个tail8均success/finite；
- rich-specific 11组全程finite、e6 nonzero active、initial/final state changed全部PASS；
- source/runtime/assets、common init、coverage、target、matched RNG、teacher/codebook state/version、scratch、
  checkpoint全部PASS；
- elapsed=`57.138093701563776 s`，peak memory=`7,901,594,112 bytes`，checkpoint=`0`、scratch=`0`；
- result/runner/manifest/stdout SHA=
  `d5255fced4553c6d4669ce11a1644e1495340a590ee76e54f22139f547cb9cca`/
  `d5255fced4553c6d4669ce11a1644e1495340a590ee76e54f22139f547cb9cca`/
  `b719b3acdec3746dae8f602fc526564a08047ae5ad1a9e2c3a3865a973c2b12e`/
  `8a0f6e13433382f61bf0f1f510a636b3ae9f2f0c9bbe26d67698e1b26644e6c8`；
- 进程自然退出，GPU=`2 MiB/0%`、compute process=`0`，异常扫描无Traceback/RuntimeError/OOM/AMP
  warning。

裁决：`CUDA BASELINE-RELATIVE SEALED-PASS / PRODUCTION PREFLIGHT GO / FORMAL NO-START`。它证明default
GradScaler自然适应后rich不劣于D0且生产组真实更新；尚未完成strict reload、RGB-only、bypass或检索门。
