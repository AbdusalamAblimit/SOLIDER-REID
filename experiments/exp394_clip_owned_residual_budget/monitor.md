# exp394 监控记录

## 2026-07-19 立项与NO-START边界

直接前因是exp393 RZ-C0 `ROUTE-ALIVE-FAIL`：e120 full=`56.8/66.8/79.6/83.9`，all-bypass
同为`56.8/66.8/79.6/83.9`，raw full−bypass=`-0.000249709 mAP point`。两个alpha最终仅
`-1.843e-4/-1.363e-4`，尽管token/context/expert/alpha参数轨迹和strict finite全部PASS。

exp394不重跑RZ、不调alpha，也不把Phase 0E rich teacher PASS作训练成功。当前只冻结新的问题定义：
rich evidence拥有production branch方向，执行幅度改为train-only D0能量匹配的固定有界预算，并以
wrong/static/generic/all-bypass证明是否真正属于CLIP语义。

当前远端无训练/审计进程，4090=`2 MiB/0%`。下一任务严格串行为：

1. 只读复核clean D0 checkpoint与固定128 train image来源；
2. 写Phase 0R-S synthetic/CPU contract；
3. contract PASS后才实现Phase 0R-128预算冻结审计；
4. 所有门禁、`rho_star`与SHA冻结前，远端production model/config实现、CUDA preflight和正式训练均
   `NO-START`；只允许本地独立Phase 0R-S contract与后续只读预算审计。

禁止用验证/测试性能选择预算，禁止恢复自由alpha，禁止并行GPU任务。
