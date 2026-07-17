# exp387 监控：官方干净代码上的最小 TAPF D0

## 当前状态

- 状态：DESIGN
- 直接对照：exp385 official clean B0 e120=`57.4/67.4/80.6/85.2`
- pose provenance：exp386 final manifest SHA256=`cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8`
- exp386 extraction/loader/paired augmentation/RGB parity/DataLoader CUDA：PASS
- 4090：空闲
- 正式 D0：未实现、未启动

下一步严格按 design 实现独立 TAPF 模块、默认关闭接线与可执行门禁。所有 Gate PASS 前不得创建正式 output 或启动训练。

## 实现阶段

- 本地实现提交：`9b0eb2b`
- 远端实现提交：`8dbf9e90dfd5ddb656db3d05b572151bda0357ad`
- D0 config SHA256：`510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b`
- `model/tapf.py` SHA256：`8d97ccf8f3d18f7efe45335f2968943d538253b08d63ad8786442eb1c60231ea`
- unit SHA256：`ce8fcc32e93fd14dd70e4fe5c0dda673466f690e9e57868209ec0611cf5f9db9`

已实现范围仅为：FP32 Gaussian/reliability renderer、Stage-2 小型 anchor、两个无 bias/无 affine 常量捷径的 Stage-3 PSG、e1–e10 handoff、默认关闭 dataloader/model/processor 接线与独立 config。未恢复任何旧 TAPF runtime 或其它历史模块，未创建正式 output。

## 当前门禁证据

### 纯模块 unit

原生 torch1.13.1：5/5 PASS：

- 坐标 renderer、`score→clamp(0,1)` 显式可靠性与 empty-valid finite；
- schedule e1/e5/e6/e9/e10/e11/e120；
- zero field 在 PSG 参数随机化后仍逐元素 exact identity；
- e1 teacher exact、e6 0.2 student、e10 student exact；
- pose loss 只到 anchor，ReID gate loss 只到 PSG/tokens；
- eval 传入 exploding pose 不被读取；两个 PSG bank 参数独立。

### config-off 官方路径 parity

以远端 `d4fa227` 建 detached 官方-clean worktree，与新代码 `TAPF.ENABLED=False` 分别运行同一 seed、teacher、模型、SGD、合成 batch 和 10-step CUDA/AMP：完整 JSON 指纹 exact。

- common state=211 tensors，initial SHA256=`a9e84220705e2021d3441663ad287c5f5522b045810f7a3576cfde46516865c1`；
- 构造后 CPU/CUDA RNG、state key/order、optimizer group/order/hyperparameter exact；
- 10 次 loss、output SHA、173 参数 gradient SHA 全部 exact；
- 动态 GradScaler 最终 scale=512，产生 173 个 momentum buffers；final state 与 momentum SHA exact。

这项门禁包含真实有限 optimizer update，不只是 forward 或 overflow-only 对比。

### B0/D0 构造不变量

`EXP387_MODEL_INVARIANTS_PASS`：

- B0/D0 公共 211 state 逐 tensor exact，构造后 CPU/CUDA RNG exact；
- 公共 optimizer 179 参数的过滤后顺序与超参数 exact；
- TAPF 12 个可训练 parameter tensor 全部进入同一 optimizer，无遗漏或重复；
- 两个 PSG bank 不共享参数，末投影 zero-init；
- B0/D0 参数量=`28,074,042/28,179,484`，新增 `105,442`，overhead=`0.375585%`。

### 首次完整 Swin 路由

合成 batch2 CUDA/AMP train+backward/eval PASS：Stage-2 student/teacher=`2×17×24×8`，两个 Stage-3 gate delta 均为 `2×48×768`，descriptor=`2×768`；score/feature/四级 featmap/pose loss/gate 全有限。首次运行发现 autocast 下 probability BCE 不安全，已在正式训练前改为 FP32 `binary_cross_entropy_with_logits` 并复跑 unit/full route 通过。

## 当前结论

实现、config-off parity、构造不变量与小批量完整路由通过；正式训练仍为 NO-START。下一步必须完成真实 paired batch64 的梯度/AMP/动态 scale/overflow、strict roundtrip、pose-free descriptor 与效率门禁。
