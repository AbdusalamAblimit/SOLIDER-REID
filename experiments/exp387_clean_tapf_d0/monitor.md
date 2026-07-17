# exp387 监控：官方干净代码上的最小 TAPF D0

## 当前状态

- 状态：FORMAL D0 RUNNING
- 直接对照：exp385 official clean B0 e120=`57.4/67.4/80.6/85.2`
- pose provenance：exp386 final manifest SHA256=`cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8`
- exp386 extraction/loader/paired augmentation/RGB parity/DataLoader CUDA：PASS
- 4090：唯一正式 D0 运行中
- 正式 D0：fresh 启动，main PID=`1013560`

实现与全部启动前 Gate 已通过；fresh execution 门禁也已通过。当前只允许该 D0 自然运行至 e120，不因单 epoch、阈值或 best checkpoint 提前停止。

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

实现、config-off parity、构造不变量、小批量路由、真实 batch64、full-model 因果路径、overflow、strict state、pose-free parity 与 matched efficiency 全部通过。正式训练门禁由 `NO-START` 更新为 `GO`，但本记录时点尚未创建正式 output 或启动训练。

## 真实 paired batch64 / CUDA / AMP

脚本：`preflight_cuda.py`，SHA256=`791c4b27f69f194dbd06697298266ed5b02a8af6dfcb1379d3b510a2848a1cab`。使用正式 D0 config、真实 Occluded-Duke identity sampler、batch64、8 workers、paired pose target、SGD 与默认 GradScaler 连续运行 24 step，结果 `EXP387_REAL_BATCH64_CUDA_AMP_PASS`。

- 默认 scale：`65536→1024`；共 6 个 overflow step，均未改变 Swin/anchor/PSG/head 参数 probe；
- step6 在 scale=2048 首次有限更新，step7 再次 overflow 后降至 1024，step8–24 连续 17 步有限更新；
- 最终全部 model parameter 与 185 个 optimizer tensor 有限；
- 发生改变的 parameter tensor：Swin `171/193`、anchor `8/8`、PSG `4/4`、head `2/3`；
- 训练峰值 allocated/reserved=`6,488,881,664/6,796,869,632` bytes；
- JSON SHA256=`d10876a2f595e1691717e71d8360aae11e495d45b43fe5339592e6577f2e36a7`；
- runner SHA256=`3b2801bfd98c7cf10f8b5b5eeb5877608b6a88c326782f65aed7f71fc67f61a2`。

这复现了 exp385 观察到的官方动态 scale 回退，而不是通过修改初始 scale、loss 或训练逻辑规避 overflow。

## Full-model 路由、梯度与 state 门禁

脚本：`preflight_semantics.py`，SHA256=`f4eea3b096d5e659ee7f29f0c6180cf49f2d7a013757f4f5487d32ee4a0a7379`，结果 `EXP387_FULL_SEMANTICS_PASS`。

### 路由与因果隔离

- e1：`student_fraction=0`，consumer 与 teacher 逐元素 exact；
- e6：`student_fraction=0.2`，consumer 与 `0.8·teacher+0.2·student` 逐元素 exact；
- e10/e11：`student_fraction=1`，consumer 与 student 逐元素 exact；
- 四个边界 epoch 中两个独立 PSG bank 每次均各调用一次；
- 只反传 pose loss：anchor `8/8` parameter tensor 有非零梯度，Swin/PSG/head 精确为零或无梯度；
- 只反传 ReID loss：Swin `171`、PSG `2/4`、head `2/2` tensor 有非零梯度，anchor 精确为零或无梯度。PSG 仅末投影在 zero-init 首步有非零梯度，符合预注册初始化。

### 人为 nonfinite overflow

先用 scale=1 做一次有限 step 建立 momentum，再注入 nonfinite loss：GradScaler `found_inf=1`、scale=`1→0.5`；208 个 model parameter tensor 与 185 项 optimizer state 在该整步前后逐元素 exact，证明真实 skip，不是只看 scale 日志。

### strict roundtrip 与 pose-free eval

- state tensor=`223`，strict load missing/unexpected=`0/0`，全量浮点 tensor 有限；
- roundtrip 后 descriptor、student field 与两个 gate delta 逐元素 exact；
- eval 的 correct pose、batch-shuffle pose、`None` 与会在任何索引时抛异常的 exploding pose，四者 descriptor/student field/gate delta 逐元素 exact；
- query+gallery 仍为 `ImageDataset`，没有 `pose_store`；RGB normal-train evaluator 同样没有 `pose_store`；
- preflight checkpoint SHA256=`d1a4b7a1c8743df0679a91ede8d4db2c631570d05e0714a610171dd73882cec9`；
- JSON SHA256=`1a7d14047c8f83c8ba38271b6e78d44bf470a9ff1657248a1042dc788f088954`；
- runner SHA256=`19512f3d33db99fb76bcdf8320831337aa000fd6b70d6b381d5618700b4ac91e`。

## Matched efficiency

脚本：`preflight_efficiency.py`，SHA256=`4a68f257c3cc41154d82311c2694cba09625e03fd7220ea5356e270dc47de446`，结果 `EXP387_MATCHED_EFFICIENCY_PASS`。B0/D0 复用一份冻结的真实 paired train batch 与一份真实 RGB-only validation batch；训练测 forward+backward+SGD，eval 按官方 FP32 RGB-only forward。最初未截断高斯合成 RGB 会令 B0 AMP loss 非有限，因此该不代表正式数据分布的 harness 被废弃，改用真实 batch 后两臂均有限且 scale=1 不回退。

| 项目 | B0 | D0 | D0−B0 |
|---|---:|---:|---:|
| 参数 | 28,074,042 | 28,179,484 | +105,442 / +0.375585% |
| MMEngine supported-op FLOPs / image | 5,535,368,448 | 5,548,787,520 | +13,419,072 / +0.242424% |
| train batch64 mean step | 100.860 ms | 102.839 ms | +1.979 ms / +1.96% |
| train peak allocated | 6,042,440,704 B | 6,185,095,168 B | +142,654,464 B / +2.36% |
| eval batch256 mean step | 225.004 ms | 228.682 ms | +3.678 ms / +1.64% |
| eval peak allocated | 4,725,194,240 B | 4,725,158,400 B | −35,840 B（测量分配噪声，实质持平） |

FLOPs 只报告 analyzer 支持的算子；两臂共同未计入的 elementwise/normalization 类算子不被伪装成完整理论 FLOPs。eval 两臂都显式 `pose_input=None`。

- 固定 train RGB SHA256=`9682bf9b5deb0a5fed3fe809a5892a90805086163128181f39138090780b086f`；
- 固定 eval RGB SHA256=`64263ed38d2f07d303a6db100c420ce980a6c8828299c35936aee49d9ba1df22`；
- JSON SHA256=`753b4f3f761e7b07bcbb487412292e334ff766021493e990980744146ffe368a`；
- runner SHA256=`bee5f8674f31a0d2b5b6549ba5a843546c3998c86c87373c352354de794fcb82`。

## 正式执行

- fresh repo：`/home/afr/SOLIDER-REID-exp387-d0-0d1822a`；
- exact execution commit：`0d1822a07dda8daac0210b68916035b1886d5d99`；
- full-history bundle：`/home/afr/reid-clean/bundles/exp387_clean_d0_0d1822a.bundle`；
- bundle SHA256：`f10fcce50129c71663a0db835e9de9c5e5313d42c413f1c0f868c4f722aeacb1`；
- config SHA256：`510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b`；
- output：`log/occluded_duke/exp387_clean_swin_tiny_d0_s1234`；
- runner：`/home/afr/train-logs/exp387_clean_d0_s1234.runner.log`；
- main PID：`1013560`。

启动前 fresh repo 为 detached exact HEAD、tracked clean、正式 output 不存在、GPU `2 MiB/0%`，并在 fresh repo 原生环境复跑 TAPF unit `5/5 PASS`。正式 output 只在训练命令启动后创建。

首次健康检查：唯一 main+8 DataLoader workers，GPU 约 `6,994 MiB`、利用率 `96%`；e1 已到 iter120/227，`Loss=14.477`、`Pose=0.918`、`Student=0`、`Reliability=0.850`、`GateAbs=7.895e-05`，日志中的 NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow 严格命中为 0。继续运行，不作早期效果裁决。

### e10

- mAP / R1 / R5 / R10：`33.4 / 42.7 / 59.8 / 65.2`；
- 同 epoch exp385 B0：`33.2 / 43.2 / 59.0 / 65.2`；
- D0−B0：`+0.2 / −0.5 / +0.8 / +0.0`；
- 此时 handoff 刚到 `Student=1`，只记录完整轨迹，不用单点判断方法成败。

### e20

- mAP / R1 / R5 / R10：`42.2 / 52.4 / 67.6 / 74.0`；
- 同 epoch exp385 B0：`39.8 / 51.1 / 66.3 / 72.9`；
- D0−B0：`+2.4 / +1.3 / +1.3 / +1.1`；
- e20 训练末尾 `Pose=0.564`、`Student=1`、`GateAbs=1.809e-02`，anchor supervision 与两个 PSG consumer 均保持 active；
- 评测后自然进入 e21/e22。检查时 exact HEAD/config 与 tracked source clean，唯一 main+8 workers，GPU 约 `7.16 GiB`；runner/train log 的严格异常及 AMP warning 均为 0。

e10→e20 的变化不替代 e120 final；不得据此提前停止或挑选中途节点。

### e30

- mAP / R1 / R5 / R10：`46.6 / 56.2 / 71.3 / 76.4`；
- 同 epoch exp385 B0：`45.4 / 55.0 / 70.5 / 76.2`；
- D0−B0：`+1.2 / +1.2 / +0.8 / +0.2`；
- e30 末尾 `Pose=0.491`、`Student=1`、`GateAbs=2.275e-02`。

### e40

- mAP / R1 / R5 / R10：`50.0 / 60.7 / 76.2 / 81.0`；
- 同 epoch exp385 B0：`49.8 / 61.2 / 76.0 / 81.1`；
- D0−B0：`+0.2 / −0.5 / +0.2 / −0.1`；
- e40 末尾 `Pose=0.474`、`Student=1`、`GateAbs=2.346e-02`。

### e50

- mAP / R1 / R5 / R10：`52.1 / 62.8 / 77.0 / 81.9`；
- 同 epoch exp385 B0：`52.7 / 63.1 / 77.1 / 82.2`；
- D0−B0：`−0.6 / −0.3 / −0.1 / −0.3`；
- e50 末尾 `Pose=0.468`、`Student=1`、`GateAbs=2.371e-02`。

### e60

- mAP / R1 / R5 / R10：`55.1 / 66.1 / 79.0 / 83.3`；
- 同 epoch exp385 B0：`54.7 / 65.0 / 79.6 / 83.5`；
- D0−B0：`+0.4 / +1.1 / −0.6 / −0.2`；
- e60 末尾 `Pose=0.467`、`Student=1`、`GateAbs=2.404e-02`；
- 评测后自然进入 e61。唯一 main+8 workers，GPU 约 `7.08 GiB`；exact HEAD/config、tracked source clean，runner/train log 严格异常与 AMP warning 均为 0，尚无 checkpoint。

e30–e60 的正负波动只记录训练轨迹，不选择局部节点，也不改变必须自然跑满 e120 的协议。

### e70

- mAP / R1 / R5 / R10：`55.4 / 65.2 / 79.5 / 83.6`；
- 同 epoch exp385 B0：`55.2 / 66.1 / 79.8 / 84.3`；
- D0−B0：`+0.2 / −0.9 / −0.3 / −0.7`；
- e70 末尾 `Pose=0.464`、`Student=1`、`GateAbs=2.379e-02`。

### e80

- mAP / R1 / R5 / R10：`56.1 / 66.3 / 79.5 / 84.0`；
- 同 epoch exp385 B0：`56.4 / 66.9 / 80.4 / 85.6`；
- D0−B0：`−0.3 / −0.6 / −0.9 / −1.6`；
- e80 末尾 `Pose=0.462`、`Student=1`、`GateAbs=2.379e-02`。

### e90

- mAP / R1 / R5 / R10：`57.5 / 67.9 / 81.2 / 85.3`；
- 同 epoch exp385 B0：`57.0 / 67.5 / 80.7 / 85.8`；
- D0−B0：`+0.5 / +0.4 / +0.5 / −0.5`；
- e90 末尾 `Pose=0.463`、`Student=1`、`GateAbs=2.410e-02`；
- 评测后训练自然推进，最新现场检查已完成 e97。唯一 main+8 workers，GPU 约 `7.10 GiB`；exact HEAD/config、tracked source clean，runner/train log 用边界词重算的 NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow/AMP warning 严格命中均为 0，尚无 checkpoint。

e70–e90 仍然只用于完整轨迹审计，不以正负单点、局部 best 或阈值裁决；继续自然运行至 e120。

## 官方 Swin 只读审计

训练期间按用户要求只读核查官方 `semantic_weight` 与 `with_cp`，完整证据见 `official_swin_audit.md`，可执行结果为 `EXP387_OFFICIAL_SWIN_AUDIT_PASS`。结论是：`with_cp` block 核心 exact，但官方 defaults/builder 未接线；semantic stage0–2 有效，terminal stage3 对 descriptor 为 dead path；另有硬编码 `.cuda()` 与 backbone `.train()/.eval()` 返回 `None` 的 API 边界。当前 B0/D0 共享这些官方行为，不构成 D0−B0 混淆，也不修改运行中代码/config。
