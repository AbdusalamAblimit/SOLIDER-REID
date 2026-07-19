# exp400 rich-budget final production preflight监控

## 状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / STATIC-CPU SEALED-PASS /
CUDA FINAL-PRODUCTION SEALED-PASS / FORMAL E120 GO`

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

## 2026-07-19 fresh execution前置审计

- 首个`git clone --no-local`在任何实验进程前因上游历史缺失Git blob中止；该半成品目录原样保留，
  GPU始终空闲，不计actual；
- 随后从已完成exp399的exact clean working tree逐文件复制到新的fresh2目录；新repo=
  `/home/afr/SOLIDER-REID-exp400-final-production-fresh2-11d7a35`，HEAD exact、tracked/all status clean、
  无alternate；所有工作树文件均为新实体；
- fresh exp400 CLIP/codebook均为regular非symlink且inode与exp399不同；script/reporter/CLIP/codebook/
  runtime SHA逐项exact；actual启动前GPU=`2 MiB/0%`、compute process=`0`。

## 2026-07-19 唯一CUDA actual封板

- 唯一PID=`403580`自然退出；D0/rich各32行完整，skip均仅attempts 1–5，first success均为6，
  各`27/32` optimizer update；rich extra skip=`0`、rich-only non-finite=`0`；
- D0/rich的e1/e6四个tail8全部连续success/finite；11个rich-specific组全程finite、e6 active、
  initial/final state全部改变；
- terminal `31/31 PASS`：state 241项全部finite且teacher-free，strict reload与descriptor exact，eval
  correct/shuffle/None/ExplodingPose逐元素exact且访问数0；
- epoch1 rho=`0`且full/all-bypass exact；epoch6 rho=`0.01615108996629715`，all-bypass max-abs/
  mean-L2=`0.1353397369/0.4205047190`，bypass0/bypass1 max-abs=
  `0.0727601051/0.0865910053`，全部finite非零；
- diagnostic state/RNG/apply_gate、teacher/codebook/source/assets/tracked均exact；checkpoint=`0`、scratch=`0`；
- elapsed=`58.02608146890998 s`，peak memory=`7,901,594,112 bytes`；进程退出后GPU=`2 MiB/0%`、
  compute process=`0`；
- result/runner/manifest/stdout SHA=
  `3935eb6df97ae832770316eff27cbfc757e4d2bd305b789d0b9b97835659a02f`/
  `3935eb6df97ae832770316eff27cbfc757e4d2bd305b789d0b9b97835659a02f`/
  `b719b3acdec3746dae8f602fc526564a08047ae5ad1a9e2c3a3865a973c2b12e`/
  `e91ffd6c4732387b90fe4f49dc31b41eb1c35ca831ac65c30975048979d4e620`；
- result=`FINAL_PRODUCTION_PREFLIGHT_PASS`且`formal_training_authorized=true`。

裁决：`CUDA FINAL-PRODUCTION SEALED-PASS / FORMAL E120 GO`。按冻结协议直接建立并启动唯一fresh
rich-budget C0 seed1234 e120，exp400不得重跑。
