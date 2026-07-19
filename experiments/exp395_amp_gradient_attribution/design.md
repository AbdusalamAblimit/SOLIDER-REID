# 实验 exp395：AMP 首步梯度归属与动态范围只读门

## 当前状态

`DESIGN-FROZEN / PHASE 0S STATIC-CPU SEALED-PASS /
CUDA ATTRIBUTION IMPLEMENTATION STATIC SEALED-PASS / CUDA EXECUTION NO-START /
FORMAL NO-START`。

本实验是独立的数值归因门，不是 exp394 修补或重跑。exp394 保持
`CUDA_AMP_PREFLIGHT_SEALED_FAIL`：actual batch64 的 teacher target 前置门通过，step 1 在
`scaled backward -> unscale` 后出现 model gradient non-finite，并在 `scaler.step` 前退出；成功
optimizer update=`0/24`、checkpoint=`0`。现有 sealed result 没有保存逐 loss 或逐 parameter-group
归属，因此不得预判失败来自某个 loss、head、router 或 normalization。

## 动机

exp394 已经把问题收紧到一个明确事实：CPU 图上的梯度所有权正确，并不保证 canonical AMP 下首步
数值有限。下一步不能靠修改 GradScaler 初值、loss 权重、rho、batch 或补步来“救活”原臂，而应先回答：

1. non-finite 是 shared D0/ReID 图也会出现，还是只出现在 rich production 图；
2. rich 图中，哪个**隔离 loss 的反向图**首先产生非有限值；
3. 非有限值落在哪些**预注册参数组**，其余组的有限动态范围是多少；
4. 各组在 scaled 与 unscaled 两个时点的范围是否与固定 scale 一致。

这些答案只用于建立后续机制设计的证据边界，不授权修改 exp394 或启动训练。

## 核心假设

若在同一 official batch、同一 canonical runtime 和相同初始 rich state 上，把每个 loss 独立重算、独立
backward，并在 unscale 前后保存完整分组统计，则可以把 exp394 的“model gradient non-finite”收紧到
可复核的 loss×parameter-group 支持集合，同时避免梯度累加、optimizer 更新或事后调参混淆。

## 诊断对象

### Arm A：matched D0 baseline

- source 仍来自 exp394 production exact commit `11d7a35788c4645c355d96d76a2a4ff20a9801ac`；
- config 使用 sealed clean D0 `configs/occluded_duke/swin_tiny_tapf_d0.yml`；
- 与 rich arm 共享同一个由 rich dataloader 产生的 official batch64 图像、PID、camera、view 与 pose target；
- 不构建 CLIP teacher，不读取 rich codebook；
- 只审计 `ReID / heatmap / confidence / pose / total`；baseline 不是新的训练对照，也不产生更新。

### Arm B：exp394-rich graph attribution

- 使用 exp394 exact rich config、fresh model、fresh external frozen teacher target；
- 保持默认 GradScaler initial scale，不覆盖 scale；
- 逐 loss 独立 forward/backward；每行开始前恢复同一 model state 与 RNG，清空 gradient；
- 不调用任何 optimizer/scaler update 路径。

### Arm C：within-rich aggregate attribution

Arm B 的隔离项与 `pose/total` 聚合项在同一初始 state 上比较。若单项均有限而聚合项非有限，只能说明
组合尺度/图交互；若某个单项及其支持组非有限，只能把问题定位到该反向子图，不能据此声称该 loss 是
唯一根因。

## 冻结 loss 矩阵

rich arm 必须按固定顺序审计以下 11 项：

1. `reid`；
2. `heatmap`；
3. `confidence`；
4. `mask`；
5. `presence`；
6. `evidence_cosine`；
7. `evidence_relation`；
8. `exec_consumer0`；
9. `exec_consumer1`；
10. `pose`；
11. `total = reid + 0.1 * pose`。

其中冻结公式为：

```text
exec_mean = mean(exec_consumer0, exec_consumer1)
semantic = mean(mask, presence, evidence_cosine, evidence_relation, exec_mean)
pose = heatmap + confidence + semantic
total = reid + 0.1 * pose
```

不得把两个 `L_exec` 只记录为均值；不得新增、删除或重加权 loss。

## 冻结 parameter-group 矩阵

每个 loss 同时报告以下 15 个互斥组：

1. `backbone`；
2. `anchor_trunk`；
3. `pose_head`；
4. `mask_head`；
5. `presence_head`；
6. `evidence_head`；
7. `id_head`；
8. `router0_token_projection`；
9. `router0_context_projection`；
10. `router0_evidence_projection`；
11. `router0_experts`；
12. `router1_token_projection`；
13. `router1_context_projection`；
14. `router1_evidence_projection`；
15. `router1_experts`。

另报告 optimizer 中未被以上组覆盖或被重复覆盖的 parameter 名；任一非零数量均使诊断无效并停止。

## 每格必须记录的统计

unscale 前的 `scaled` 与 `scaler.unscale_(optimizer)` 后的 `unscaled` 各保存：

- parameter tensor 总数、grad present/absent、nonzero/zero tensor 数；
- gradient element 总数；
- finite、NaN、`+Inf`、`-Inf` element 数；
- 有限元素的 `abs-max / L2 / abs-P50 / abs-P95 / abs-P99`；
- loss scalar、GradScaler scale、autocast dtype、参数组名称与明确的 parameter name 列表。

分位数和 L2 只在有限元素上计算，非有限元素单独计数；空集合写 `null`，禁止把 NaN/Inf 直接写进
JSON。所有统计从 gradient clone 读取，不得在 capture 过程中修改原 gradient。

## 零更新与可复现边界

1. 每行只允许 `zero_grad -> fresh forward -> scale(loss).backward -> capture scaled -> unscale -> capture
   unscaled -> discard grads`；
2. 禁止调用 `optimizer.step`、`scaler.step`、`scaler.update`、scheduler 或 EMA；
3. model state、optimizer state、teacher state、source SHA、asset SHA 与 CPU/CUDA/Python/NumPy RNG 在审计
   前后必须 exact；
4. optimizer update 必须 exact `0`，checkpoint 必须 `0`；
5. 每个 loss 使用相同 batch manifest、相同 teacher target SHA 与同一初始 model state；不得用
   `retain_graph=True` 复用已 backward 的图；
6. 结果必须包含异常词扫描、进程退出和 GPU 恢复空闲证据。

## 归因规则

- baseline total 已非有限：只能优先归到 shared runtime/D0 图，不能称 rich-specific；
- baseline 有限、rich `reid` 非有限：定位到 rich model 在 ReID 反向下的 production 图，不归到 teacher
  auxiliary loss；
- rich `reid` 有限、某隔离 auxiliary 非有限：定位到该 loss 的反向支持子图；共享参数组意味着仍可能有
  上游激活/算子交互，不能声称唯一参数根因；
- 隔离项全有限而 `pose` 或 `total` 非有限：定位为聚合尺度/组合图问题；
- 全部有限但未复现 exp394：记录为 fresh exp395 未复现，exp394 sealed FAIL 不被推翻，也不得重复抽样；
- 任一结果都只授权下一步设计一个新的 AMP-stable mechanism，绝不自动授权 e120。

## Phase 0S：static/CPU contract

在任何 CUDA 占用前，独立 CPU contract 必须证明：

1. exp394 source/config/preflight SHA exact，未修改 sealed 资产；
2. source 暴露两个独立 `exec_losses`，loss 公式和 parameter seam 可静态识别；
3. gradient reporter 精确区分 absent、zero、finite、NaN、`+Inf`、`-Inf`；
4. synthetic 11-loss×15-group 所有权矩阵 exact；
5. scaled/unscaled 有限统计满足固定 power-of-two scale 比例；
6. semantic/pose/total 公式 exact；
7. 参数 state 与 RNG 前后 exact，optimizer update exact 0；
8. contract 进程从未初始化 CUDA。

CPU PASS 只封板诊断器数学与静态 seam，不说明 canonical CUDA 下哪一项失败，也不授权 GPU 执行。

## 停止边界

- static/CPU 任一 gate FAIL：先保留失败 result/runner、归因 contract，再修改 exp395 自身；不得碰 exp394；
- 未封板 protocol 与 CPU contract 前：CUDA `NO-START`；
- 后续若获独立授权，CUDA 诊断中 teacher/source/asset/batch 前置门任一 FAIL，必须在 backward 前停止；
- OOM、RuntimeError、state/RNG 漂移、parameter 覆盖不完整时立即停止，禁止减 batch、改 scale 或补跑；
- 诊断完成后仍为 optimizer update=`0`、checkpoint=`0`、formal training 与 semantic multi-stage
  `NO-START`。

## 风险与失败解释

逐 loss backward 改变了“所有 loss 同时反向”的执行图，因此隔离结果只能定位支持集合，不能完全替代
原 total backward；这正是同时保留 `pose/total` 两行的原因。D0 与 rich 的 TAPF 参数结构不同，baseline
只用于判断 shared runtime/standard loss 是否同样非有限，不用于性能或逐参数幅值一一配对。任何
AMP-stable 修复都必须另立后续实验并预注册，不能回写为 exp394 的 continuation。

## Phase 0S 封板

独立 static/CPU contract 已连续两遍逐字节 exact PASS。13项gate全部通过：exp394 frozen
source/config/preflight SHA exact；11-loss×15-group synthetic ownership exact；scaled/unscaled五项范围
统计均满足固定`65536`比例；sentinel正确区分absent、zero、finite、NaN、`+Inf`、`-Inf`；loss公式、
model state、RNG、zero-update全部exact。两遍均未初始化CUDA，optimizer update=`0`、checkpoint=`0`。

script/result/runner SHA256=
`d4c6d67b082e4e4f68ff215de3e7cf1f2a2ac1c4c59e17ceb265353b8810083a`/
`89afc893409957ee5ad356e0e2d5789683b36bcce449076d26a7dec3d3bed91c`/
`89afc893409957ee5ad356e0e2d5789683b36bcce449076d26a7dec3d3bed91c`；repeat result/runner SHA相同。

裁决只冻结归因器数学与protocol seam；没有读取official batch、CLIP或pose资产，也没有产生任何AMP
根因证据。下一步只允许设计独立CUDA attribution implementation；没有新的明确授权前，4090继续
`NO-START`，exp394、formal e120与semantic multi-stage边界不变。

## CUDA attribution implementation静态封板

独立`cuda_amp_attribution.py`已经按本设计实现但从未执行。它固定D0 baseline 5行、rich 11行、15组
完整覆盖、每行fresh GradScaler默认`65536`、scaled capture→unscale→unscaled capture、state/buffer/
RNG恢复、fresh asset SHA、canonical runtime版本和post-exit GPU审计。源码没有optimizer/scaler/
scheduler step/update、checkpoint load、retain_graph或训练授权路径。

CPU-only AST/static contract连续两遍29/29 PASS且result/runner逐字节一致。CUDA implementation/static
script/result SHA256=
`64840b710db587720aa8807571212b246af3eabb54306bd5aa1bbf692f5ea08b`/
`345d26309043dd8d14119316a7ca186e1cf9faea2e666bd01d652ded50663c1b`/
`30b7b7ae06ff2bd3153208fe4384e11e06a097608c6ce876d6c254c079f2e314`。

该PASS只说明**未执行脚本**满足冻结静态协议；没有复制fresh远端资产、没有初始化CUDA、没有产生
actual loss/gradient matrix。CUDA execution仍需新的明确授权，formal训练与semantic multi-stage继续
`NO-START`。
