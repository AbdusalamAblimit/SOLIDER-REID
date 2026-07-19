# exp395 AMP 首步梯度归属与动态范围协议

## 当前状态

`PROTOCOL-FROZEN / PHASE 0S STATIC-CPU SEALED-PASS /
CUDA ATTRIBUTION IMPLEMENTATION STATIC SEALED-PASS /
CUDA ATTRIBUTION EXECUTION SEALED-INVALID / REPORTER RUNTIME FAIL /
FORMAL NO-START`。

本协议只定义只读归因门。它不修改、不重跑 exp394，不调用 optimizer update，也不产生训练 checkpoint。

## 冻结上游证据

- exp394 source commit：`11d7a35788c4645c355d96d76a2a4ff20a9801ac`；
- exp394 sealed failure commit：`be6844ee13d2da031c229a376c4c877861c8d4b8`；
- exp394 actual script/result/runner SHA256：
  - `bae2210bc606048371b4750f85919595c0b8fdbd1e11681abac59fe9727ea4f0`；
  - `3897d76fd6b6aeb0d9ed2a27e527053874f6cdf32b56cc80d5bc2f12e584b152`；
  - `c76e9285a41f65f0e9333dda2ef10a75bd1a17bf85538019ac3871d000b0c879`；
- canonical runtime freeze SHA256：
  `3d38c99c7f06502d8b40467d2674c966723e5c913d2edf962c5a7088ec60cddb`；
- CLIP checkpoint SHA256：
  `9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e`；
- rich codebook SHA256：
  `fb87da370ea945d526f499bef78093a6b07203d87c6d84efe06b5eb6594f954a`。

exp394 的唯一已知失败点是 total scaled backward 后、unscale 后的 model gradient non-finite。sealed
result 没有 parameter attribution，故本协议不带任何 loss/head 根因先验。

## source/config 静态冻结

| 文件 | SHA256 |
|---|---|
| `model/tapf.py` | `95c5d0ff80bf9e4529589a5f31819e7aad5db644b88e2a33d6af07c9ffc42886` |
| `model/clip_semantic_teacher.py` | `c648fa768b178d153258c46eee69679cbc0b90a11db918800323ab5c5c6054d5` |
| `model/make_model.py` | `6bc7d9c83a2f4d12b78dd2c09335d366ce568107ddce5dded3abfe7ca8538f03` |
| `processor/processor.py` | `be1c19ea5af19534e3855eb2a5914e0dc9a5643c63a39cfa508c81f89660eac1` |
| `config/defaults.py` | `a13e5f6df0e8c770c254c115d6d55208baac7938cffbec6f208ba9caa24dd7c5` |
| `model/backbones/swin_transformer.py` | `b389b7243e204d851ed365c986c8c4077d7fa86ce79e6cbb0be6fc4a1ba58eef` |
| `datasets/pose_dataset.py` | `d04e74908d18eaf8105f9b85c66287cac6980ddf5ffe8132e855c7d5a9f61bbc` |
| rich config | `e0413a497976ad6dbf4c74cf13b55c86c169d659bab6d967455e87c592e47f4e` |
| D0 config | `510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b` |
| exp394 CUDA preflight | `bae2210bc606048371b4750f85919595c0b8fdbd1e11681abac59fe9727ea4f0` |

上述文件在 exp395 中只读；任何 blob 改变都直接使 protocol FAIL。

## fresh execution 与资产边界

后续只有另行获准 CUDA 诊断时，才允许：

1. 从 source commit `11d7a35...` 建立新的 exp395 execution repo；不得在 exp394 execution repo 内运行；
2. 把 CLIP checkpoint 和 codebook 复制成 exp395 名称的 canonical regular file，禁止 symlink、旧路径
   mapping 或 cache；复制前后 SHA 必须与本协议相同；
3. 使用新的 exp395 result、runner、manifest 路径，路径在启动前必须不存在；
4. canonical runtime 只能是 freeze SHA exact 的 exp394 clean runtime；不得安装、升级或切换依赖；
5. GPU 启动前必须确认唯一 4090 compute process 为 0，结束后恢复空闲。

资产复制只为隔离 provenance，不表示改变 CLIP/codebook 内容，也不授权训练。

## official batch 与 arm 顺序

- batch 固定为 rich config、seed1234、RandomIdentitySampler 的 fresh first batch64；
- rich loader 产生的 normalized image、PID、camera、view、pose 和 pre-RE `teacher_rgb` 作为唯一 batch；
- batch manifest 必须保存 relative path、RGB SHA、PID、camera、view；
- teacher target 只计算一次，保存 target tensor SHA、shape、finite、valid count、invalid-zero 与 valid norm；
- 所有 loss 行复用相同 input/target clone，不重新抽样；
- 顺序固定：D0 baseline 行在前，rich 11 行在后；arm 内按 design.md 的 loss 顺序执行。

## baseline 构造

D0 baseline 与 rich model 都 fresh 构造，均不加载训练 checkpoint。baseline 接受同一 rich batch，但忽略
`teacher_rgb` 与 rich target。baseline 只报告 standard D0 的 `reid/heatmap/confidence/pose/total`。

为了防止将不同 model 初始化误写成逐参数 matched 比较：

- baseline-vs-rich 只比较 finite/non-finite 支持、数量级和公共组，不比较性能；
- rich model 内 `reid` 对其他 rich auxiliary 的比较才是同 state 归因；
- 必须分别保存两个 model initial-state SHA；禁止把一个 model 的参数复制进另一个 model。

## loss 隔离执行

每一行都从 arm 的 initial model state 与 RNG exact 恢复，重新 forward 并构建该 loss：

```text
zero_grad(set_to_none=True)
fresh forward under canonical autocast
assert selected loss scalar and all frozen targets are finite
scaled_loss = scaler.scale(selected_loss)
scaled_loss.backward()
capture scaled gradients
scaler.unscale_(optimizer)
capture unscaled gradients
discard gradients
restore model/optimizer/RNG
```

每行使用 fresh `GradScaler`，initial scale 必须等于 canonical default；禁止通过参数覆盖。只允许
`scale/backward/unscale_`，源码与运行时调用计数必须证明：

- `optimizer.step = 0`；
- `scaler.step = 0`；
- `scaler.update = 0`；
- scheduler/EMA update=`0`。

不得使用 `retain_graph=True`，不得从上一行复用 gradient 或 optimizer state。

## loss 提取 seam

rich model forward 必须保留并读取：

- `reid_loss`；
- `aux["heatmap_loss"]`；
- `aux["confidence_loss"]`；
- `aux["region_mask_loss"]`；
- `aux["presence_loss"]`；
- `aux["evidence_cos_loss"]`；
- `aux["evidence_relation_loss"]`；
- `aux["exec_losses"][0]` 与 `[1]`；
- `aux["pose_loss"]`；
- `reid_loss + 0.1 * aux["pose_loss"]`。

必须先静态和 runtime 验证 `exec_losses` 长度 exact 2，且 `aux["exec_loss"]` 逐元素等于两项 mean。

## parameter group 定义

组必须由 `named_parameters()` 名称构建并保存 name list，不允许只按 module object 临时拼接。15 组定义为：

```text
backbone
anchor_trunk
pose_head
mask_head
presence_head
evidence_head
id_head
router{0,1}_token_projection
router{0,1}_context_projection
router{0,1}_evidence_projection
router{0,1}_experts
```

optimizer 中全部 requires-grad parameter 必须恰好落入一个组。`uncovered/duplicate` 名称必须完整写入
result；任一非空立即停止 backward。D0 不存在的 rich-only 组写 `not_applicable`，不得伪装成 zero grad。
D0的每个旧PSG没有T/C/E分解，因此其完整`input_projection/norm/output_projection`只映射到对应
`router{k}_experts` baseline bucket；三个projection bucket保持`not_applicable`。该映射只用于覆盖D0
optimizer与判断shared finite，不与rich router做逐参数幅值配对。

parameter name list在每个arm顶层只保存一次，loss行引用同一冻结group schema，避免在16行中重复写入
大段名称；每行仍完整保存15组统计。

## gradient 统计 schema

每个 loss×group×stage 保存：

```text
parameter_tensors
grad_present_tensors / grad_absent_tensors
grad_nonzero_tensors / grad_zero_tensors
elements
finite_elements / nan_elements / posinf_elements / neginf_elements
all_finite
finite_abs_max / finite_l2 / finite_abs_p50 / finite_abs_p95 / finite_abs_p99
```

统计基于所有 present gradients flatten 后的 FP64 CPU clone。分位数采用 deterministic linear
interpolation；只有 finite 子集参与范围统计。若 finite 子集为空，范围字段为 JSON `null`。

同一格还保存 scaled/unscaled 非有限计数是否一致；当两者全 finite 时保存
`scaled_stat ~= unscaled_stat * scale` 的相对误差。该比例是审计项，不是调 scale 的依据。

## state、RNG 与隔离终审

执行前后必须逐项 exact：

- rich/D0 model state SHA；
- optimizer state SHA 与每个 parameter `_version`；
- frozen teacher parameter `_version`、codebook tensor `_version` 与 state SHA；
- torch CPU、全部 CUDA、NumPy、Python RNG；
- source/config/runtime/asset SHA；
- batch/teacher target SHA。

不得写 model checkpoint；result/runner/manifest 是唯一允许输出。异常扫描至少包括
`NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow/AMP warning`，并区分预期记录到 gradient matrix
的 NaN/Inf 计数与未捕获异常。

## 诊断有效性 gate

本实验的 PASS 只表示 attribution matrix 完整可信，不表示所有 gradient finite。有效门为：

1. source/runtime/asset/batch/teacher target 全部 exact；
2. baseline 5 行与 rich 11 行完整；
3. 15 组覆盖无遗漏/重复；
4. 每行 scaled/unscaled 报告完整；
5. model/optimizer/teacher/RNG 前后 exact；
6. optimizer/scaler/scheduler update exact 0；
7. checkpoint 0、进程退出、GPU 空闲；
8. result/runner/manifest SHA 冻结。

gradient 是否 finite 是输出分类，不得通过早停、调 scale 或删除 loss 让“有效性 gate”表面通过。

## Phase 0S static/CPU contract

独立脚本 `static_cpu_contract.py` 在工作目录 uv 环境、`CUDA_VISIBLE_DEVICES=''` 下执行。它只读取 frozen
source，使用 synthetic CPU 图验证 reporter、所有权、loss 公式、scale 比例、zero-update、state/RNG
exact；不导入 dataset，不读取 official image/pose/CLIP/codebook，不构建 CUDA context。

CPU contract PASS 后只允许把本协议状态改为 frozen，并设计独立 CUDA attribution script。没有新的明确
CUDA 授权前，仍不得占用 4090。

## 停止与裁决

- static/CPU FAIL：保留失败资产，只修 exp395 contract；
- CUDA 前置 provenance FAIL：在 teacher/forward/backward 前停止；
- parameter 覆盖 FAIL：在 backward 前停止；
- 某隔离 loss 产生 non-finite：记录完整 matrix，清理 gradient，继续下一预注册 loss；不得更新参数；
- 未捕获 RuntimeError/OOM/state/RNG 漂移：立即停止整个诊断；
- 无论结果如何，exp394 仍 sealed，exp395 optimizer update=`0`、checkpoint=`0`、formal e120 与 semantic
  multi-stage=`NO-START`。

## Phase 0S static/CPU封板

`static_cpu_contract.py`在工作目录uv环境、`CUDA_VISIBLE_DEVICES=''`下连续执行两遍，13/13 gates均
PASS，且两份result与runner逐字节一致：

- 11个冻结loss与15个参数组的synthetic ownership全部exact；
- scaled/unscaled的abs-max、L2、P50/P95/P99均满足固定`65536`比例；
- sentinel对absent、zero、finite、NaN、`+Inf`、`-Inf`的tensor/element计数与范围统计exact；
- exp394 source/config/preflight SHA与静态loss seam exact；
- semantic、pose、total与两个consumer mean公式exact；
- model state与torch/Python RNG前后exact，optimizer update=`0`、checkpoint=`0`；
- CUDA初始化before/after均为`false`。

script/result/runner SHA256=
`d4c6d67b082e4e4f68ff215de3e7cf1f2a2ac1c4c59e17ceb265353b8810083a`/
`89afc893409957ee5ad356e0e2d5789683b36bcce449076d26a7dec3d3bed91c`/
`89afc893409957ee5ad356e0e2d5789683b36bcce449076d26a7dec3d3bed91c`；repeat SHA相同。

裁决=`PHASE0S_STATIC_CPU_SEALED_PASS`。该PASS不包含actual AMP、official batch或根因归属；只允许
下一步实现独立CUDA attribution script，且仍需新的明确CUDA授权。

## CUDA attribution implementation static封板

实现文件=`cuda_amp_attribution.py`，只读静态审计文件=`cuda_attribution_static_contract.py`。静态
contract连续两遍29/29 PASS，确认：

- D0 baseline 5行与rich 11行顺序exact，15组顺序exact；
- parameter覆盖检查发生在任何backward之前；
- 每行fresh默认GradScaler，不允许`init_scale`覆盖；
- scaled统计在`unscale_`之前、unscaled统计在之后，并保存非有限计数一致性和range比例误差；
- 两个consumer `L_exec`独立暴露；D0旧PSG映射与rich-only `not_applicable`语义明确；
- canonical Torch/OpenCLIP/OpenCV/timm版本、fresh exp395 regular asset名称与SHA均为执行前硬门；
- source不存在optimizer/scaler/scheduler step/update、retain_graph、checkpoint load或formal training GO；
- output只允许result/runner/manifest，且明确要求进程退出后的GPU空闲外部审计。

CUDA implementation/static script/result/runner SHA256=
`64840b710db587720aa8807571212b246af3eabb54306bd5aa1bbf692f5ea08b`/
`345d26309043dd8d14119316a7ca186e1cf9faea2e666bd01d652ded50663c1b`/
`30b7b7ae06ff2bd3153208fe4384e11e06a097608c6ce876d6c254c079f2e314`/
`30b7b7ae06ff2bd3153208fe4384e11e06a097608c6ce876d6c254c079f2e314`；repeat SHA相同。

裁决=`CUDA_ATTRIBUTION_IMPLEMENTATION_STATIC_SEALED_PASS / CUDA EXECUTION NO-START`。该封板未读取
official数据或teacher资产、未复制fresh远端文件、未初始化CUDA；执行仍需新的明确授权。

## CUDA actual执行与停止门落地

唯一actual在fresh source/regular assets/canonical runtime/GPU空闲门通过后启动。official batch64与
teacher target已经越过前置门；第一行D0 `reid`完成scaled backward后，scaled reporter在调用
`torch.quantile`处理backbone组全量元素时触发`RuntimeError: quantile() input tensor is too large`。
异常发生在`scaler.unscale_`之前，故D0五行、rich十一行与十五组双时点矩阵均不完整。

依照本协议“未捕获RuntimeError立即停止整个诊断”，本次执行只可封板为`INVALID`：不得重跑exp395，
不得把已执行的backward误写成finite/non-finite证据，也不得修改reporter后沿用同一实验编号。外部终审
确认optimizer/scaler update=`0`、checkpoint=`0`、进程退出、GPU空闲、两端source tracked clean；
进程在中途退出，因而完整model/optimizer/teacher/RNG after-exact gate没有产出，必须明确记为未证明，
不能由控制流推断替代。

封板SHA256：

- actual script：`64840b710db587720aa8807571212b246af3eabb54306bd5aa1bbf692f5ea08b`；
- result/runner：`cdffff60b1b6e04e6bb0b13bb54e12518380421675c59c2f2c785f1b7a5adb75`；
- manifest：`3a0ef5d98dd6387b330958bbfb1e9d893e60745e8857237bbbbe375778886c64`。

裁决=`CUDA_ATTRIBUTION_EXECUTION_SEALED_INVALID / REPORTER_RUNTIME_FAIL`。只允许另立新的reporter协议；
exp394与正式训练边界不变。
