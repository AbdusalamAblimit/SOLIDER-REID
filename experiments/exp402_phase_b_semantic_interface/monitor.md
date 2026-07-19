# exp402 Phase-B semantic-interface监控

## 当前状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / STATIC-CPU RUN1 SEALED-FAIL / FIX PENDING VALIDATION / GPU NO-START`

## 2026-07-20 接手与设计冻结

- 上游exp401=`RICH_BUDGET_ROUTE_ALIVE / PHASE-B INTERFACE GO`，full−all-bypass=
  `+0.1194214838 mAP point`，但R1差为`−0.0904977322 point`；
- exp402只做同checkpoint RGB-only只读语义接口kill-switch，不修改sealed repo/config/checkpoint，不训练；
- 冻结10个串行arm：correct、五类evidence/slot/binding破坏、generic expert mean、router0/1/all bypass；
- wrong RGB使用same-split/same-camera/different-PID的dataset-global absolute-index donor，不允许batch roll；
- scientific GO要求所有六个semantic controls相对correct至少低`0.1 mAP`，并复核route gap与两个consumer；
- 当前远端无GPU任务，正式脚本尚未实现或传输；下一步仅实现CPU/static正反contract。

## initial static CPU run1：SEALED-FAIL

- fresh run1完成38项CPU/static contract，GPU始终未初始化；36项PASS，仅
  `orthogonal_norm=false`、`orthogonal_cosine=false`；
- canonical正交矩阵本身Gram max-abs=`9.9920072216e-16`，已通过`<=1e-12`门；但double
  synthetic evidence在干预实现中被强制降为float32，导致norm max-abs=`6.1642145166e-08`、
  cosine max-abs=`1.2328428656e-07`；
- run1 core/static/result/runner SHA256=
  `8b536ec1e03d6ac17253900462af182cc11f65757fb35ef134b038644e71dfd2`/
  `039f0b9cf3ed45b7416772be0acaafbc6c7ecb8bbbcb8be59d8f7d75e1963778`/
  `41f2e84b5cc2ea2f165417ecb4822442472b4b6f993afb03dad651a7657b16dd`/
  `41f2e84b5cc2ea2f165417ecb4822442472b4b6f993afb03dad651a7657b16dd`；
- 判定=`EXP402_STATIC_CPU_FAIL`。这是通用contract的dtype语义错误，不是正式模型数值失败；正式
  evidence原本为float32。run1永久保留，不覆盖、不补跑；修复仅让右乘矩阵跟随evidence dtype，随后必须
  用fresh run2/run3路径重新完成两遍38/38与byte-exact门，GPU继续`NO-START`。
