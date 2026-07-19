# exp400 rich-budget final production preflight监控

## 状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / IMPLEMENTATION STATIC SEALED-PASS /
STATIC-CPU SEALED-PASS / CUDA FRESH-EXECUTION GO / FORMAL NO-START`

## 2026-07-19 接手

- exp399 CUDA full matrix PASS，D0/rich各`26/32` update、tail8×4 PASS；
- rich-specific 11组finite/active/updated，teacher/codebook与退出终审PASS；
- exp400只补terminal strict reload/RGB-only/rho/bypass/state contract；
- 当前尚未实现，GPU=`2 MiB/0%`且无compute process。

## 2026-07-19 static/CPU封板

- final CUDA脚本保持exp399的32-step动态门，并新增final state finite、teacher-free state、strict reload、
  eval RGB-only、epoch1 rho=0 identity、epoch6 all/bypass0/bypass1三类非零差、diagnostic state/RNG恢复；
- 所有descriptor变体在`finally`恢复state、RNG和`apply_gate`实例覆盖状态；失败路径不得授权formal；
- static/CPU连续两遍`48/48 PASS`，CUDA初始化前后均为`false`；toy双router验证rho0 exact identity、
  consumer0/consumer1各自非零、strict reload、ExplodingPose零访问和patch/state恢复；
- 两遍result/runner逐字节一致。CUDA/static/result SHA256=
  `1f069614fd789f7c3a6ca1d5666239d7ce91769a502087cc379769bd1cceb797`/
  `e4019edf3df23a675c9ee0b1c2da1006f28bf184a97db036b88cb2d67888b33e`/
  `501b12b4a926e8bc0b9de88995a939beca465ea05fef8b8d96410ef9074c3f02`；
- 当前裁决=`STATIC-CPU SEALED-PASS / CUDA FRESH-EXECUTION GO / FORMAL NO-START`；下一步直接
  建立fresh远端execution与regular exp400 CLIP/codebook资产，逐SHA后运行唯一actual。
