# exp396 chunk-safe exact AMP归因监控

## 状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / PHASE 0Q STATIC-CPU SEALED-PASS /
CUDA ATTRIBUTION FRESH-EXECUTION GO / FORMAL NO-START`

## 2026-07-19 接手

- exp394保持`CUDA_AMP_PREFLIGHT_SEALED_FAIL`；
- exp395唯一actual保持`CUDA_ATTRIBUTION_EXECUTION_SEALED_INVALID / REPORTER_RUNTIME_FAIL`，禁止重跑；
- exp395失败位置为第一行D0 `reid` scaled capture中的大输入`torch.quantile`，发生在unscale前；
- exp395 update=`0`、checkpoint=`0`、GPU已恢复空闲，result/runner/manifest SHA已封板；
- 用户已给出持续自主授权，不再等待逐次CUDA确认，但仍执行design→static→fresh actual串行门；
- 当前只完成exp396设计与协议冻结，尚未实现或运行static，CUDA保持`NO-START`。

## 本轮唯一变量

只替换gradient finite range reporter为固定chunk双遍扫描与temporary memmap exact sort。不得改变D0/rich
loss矩阵、15组、batch64、seed1234、default GradScaler、source/runtime/assets或zero-update边界。

## 2026-07-19 Phase 0Q implementation与static/CPU

- production reporter使用`1,048,576`元素chunk，两遍只读扫描，temporary FP64 regular memmap原地sort，
  linear order statistic P50/P95/P99；
- production源码不含`torch.quantile`或全量`torch.cat`，每格finally删除memmap，顶层
  `TemporaryDirectory`在成功/异常均清理；
- 首遍与repeat均33/33 PASS，四份result/runner逐字节exact；
- 小张量NaN/±Inf/count/range与reference一致，multi-parameter/multi-chunk一致；
- 超大case=`16,777,217`元素，P50/P95/P99=
  `8,388,608 / 15,938,355.2 / 16,609,443.84`，与解析值exact；
- input SHA前后exact；success、full-report、multi-chunk、oversize与注入异常scratch全部为空；
- CUDA initialized before/after=`false/false`，optimizer update=`0`、checkpoint=`0`。

冻结SHA256：

- CUDA implementation：`6e8b2b67efcda7cfaca8527fb0ae1dd4c6aedcebef3fec6ded2e5ba6ddab8164`；
- static contract：`f3a2ee3ccafa4caa1606b92b93b86177cc0b5ef6cfe7ac2b6f0d31fa195c415b`；
- v1/repeat result/runner：
  `e5d68df7731042a98f440f43acc45c9cf11b70aa7df25e09397ff6375f355394`。

裁决：`PHASE0Q_STATIC_CPU_SEALED_PASS / CUDA ATTRIBUTION FRESH-EXECUTION GO`。用户持续授权已写入
heartbeat，提交后直接推进fresh actual；formal训练继续`NO-START`。

## 2026-07-19 canonical runtime CPU-only复核

- 在远端canonical Python 3.10 / Torch 2.6.0+cu124 / NumPy 2.2.6下隐藏CUDA执行；
- production script SHA保持
  `6e8b2b67efcda7cfaca8527fb0ae1dd4c6aedcebef3fec6ded2e5ba6ddab8164`；
- 同一`16,777,217`元素case的P50/P95/P99再次得到
  `8,388,608 / 15,938,355.2 / 16,609,443.84`，6/6 gate PASS；
- per-cell与root scratch均为空，CUDA initialized before/after=`false/false`；
- canonical smoke script SHA=
  `2081b3c755aaa8175a63d491de584348670600fdbf4969b6c84ae3fa4f3e75c9`。

该复核不读取official batch、不占用4090，只排除exp395曾出现的目标runtime大输入统计限制；CUDA
actual GO不变。
