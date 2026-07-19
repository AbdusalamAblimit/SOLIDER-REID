# exp399 named-parameter state reporter exact协议

## 状态

`PROTOCOL-FROZEN / STATIC-CPU SEALED-PASS /
CUDA BASELINE-RELATIVE SEALED-PASS / PRODUCTION PREFLIGHT GO /
FORMAL NO-START`

## 上游边界

- exp398=`CUDA EXECUTION SEALED-INVALID / GROUP_STATE_REPORTER_RUNTIME_FAIL`；
- exp398不得修改或重跑；其result/runner SHA=
  `71a943e6a233999549f69c1ece2ce1c2c3e507c69d9e99364272442d9b6ac998`；
- exp399不改变科学轨迹门，只修复真实parameter container测量契约。

## static/CPU exact

必须覆盖：真实exp396 `parameter_groups()` tuple输出、15组与coverage exact、name/order/value绑定、空组、
重复名、空名、裸Parameter、错误tuple和非Parameter反例；同时复跑exp398全部32-step AST与六类trajectory
synthetic门。CUDA必须未初始化，两遍result/runner逐字节一致。

## CUDA actual

fresh repo/assets、同32 batch64、e1/e6各16步、tail8、两arm fresh default GradScaler。PASS/FAIL公式与
exp398冻结协议相同；不手工scale、不补步、不改loss/batch、不保存checkpoint。PASS只授权final
production preflight，formal仍`NO-START`。

## static封板

正式两遍35/35 PASS且result/runner逐字节一致。implementation/static/result SHA=
`b9da4346b0d74d13b537bd7fa3f5eff1e65b0b6e512014026800506807723907`/
`7948845f1600141302285cee12c025cbf0ba50faa1af01d1fb298bd3aa558810`/
`32adc18d2b6dc06c0d3ea37ca6003d749a2ff2540efefdbca0e35e1fba2f0d98`。该PASS授权唯一fresh CUDA
actual；不授权formal。

## actual封板

D0/rich各32行完整、skip/scale轨迹exact，各`26/32`成功；最后25步连续success，e1/e6四个tail8全部
success/finite。rich extra skip=0，rich-only non-finite=0，11个rich-specific组finite/active/updated
全部PASS。result/runner/manifest SHA=
`d5255fced4553c6d4669ce11a1644e1495340a590ee76e54f22139f547cb9cca`/
`d5255fced4553c6d4669ce11a1644e1495340a590ee76e54f22139f547cb9cca`/
`b719b3acdec3746dae8f602fc526564a08047ae5ad1a9e2c3a3865a973c2b12e`。PASS只授权另立final
production preflight；formal仍`NO-START`。
