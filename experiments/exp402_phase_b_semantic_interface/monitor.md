# exp402 Phase-B semantic-interface监控

## 当前状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / STATIC-CPU NO-START / GPU NO-START`

## 2026-07-20 接手与设计冻结

- 上游exp401=`RICH_BUDGET_ROUTE_ALIVE / PHASE-B INTERFACE GO`，full−all-bypass=
  `+0.1194214838 mAP point`，但R1差为`−0.0904977322 point`；
- exp402只做同checkpoint RGB-only只读语义接口kill-switch，不修改sealed repo/config/checkpoint，不训练；
- 冻结10个串行arm：correct、五类evidence/slot/binding破坏、generic expert mean、router0/1/all bypass；
- wrong RGB使用same-split/same-camera/different-PID的dataset-global absolute-index donor，不允许batch roll；
- scientific GO要求所有六个semantic controls相对correct至少低`0.1 mAP`，并复核route gap与两个consumer；
- 当前远端无GPU任务，正式脚本尚未实现或传输；下一步仅实现CPU/static正反contract。
