# exp378 TAPF 修复后最终实现复审

## 审查边界

本轮只读复审修复后的 TAPF 生产实现、配置、单测与 CUDA preflight，没有修改生产代码、配置或测试，也没有启动训练。唯一写入是本报告。

审查覆盖：

- `model/modules/task_adaptive_pose_field.py`
- `model/pose_backbone_model.py`
- `model/modules/pose_spatial_gate.py`
- `processor/processor.py`
- `config/defaults.py`
- `configs/occluded_duke/exp378_*.yml`
- `experiments/exp378_e2e_pose/test_task_adaptive_pose_field.py`
- `experiments/exp378_e2e_pose/cuda_preflight.py`
- `experiments/exp378_e2e_pose/design.md`

## 最终裁决

**生产核心修复：PASS。**

上一轮两个 P0 实现错误已经修正：

1. P0/F0 在 bootstrap 后、以及所有 TAPF eval，现在会在 `_prepare_pose()` 之前关闭 external pose 路径；audit override 同样被忽略。
2. confidence target 已改为 person-0 的 `scores[:, 0] * person_mask[:, 0:1]`，并由独立 confidence logits 正确进入 `binary_cross_entropy_with_logits`。

**正式 Gate 0：`CHANGES_REQUIRED / 仍不可启动正式训练`。**

原因不是 TAPF 主体拓扑仍然错误，而是现有正式门禁脚本含一个确定会误报的数值对照，且上一轮要求的完整梯度、default-OFF 与全流水线 teacher-off 回归仍未被测试锁定。修正门禁后再做 CUDA batch64 preflight 即可，不需要推倒 TAPF 实现。

## 已确认通过

### 1. teacher 生命周期的模型级修复

`PoseBackboneModel.forward()` 现在先计算 `tapf_teacher_required`：

- bootstrap 期四臂读取 teacher；
- D0/J0 在 epoch 11 后继续读取 teacher以构造 pose loss；
- P0/F0 在 epoch 11 后不索引、不校验、不变换、不传递 `pose_dict`；
- 所有 TAPF eval 都走 predicted-only；
- `scene_heatmaps_override` 在 predicted-only 路径也不会被读取。

从静态控制流看，正确 pose、随机 pose、非法 pose和 `None` 都不能影响 P0/F0 post-bootstrap 或任一 TAPF eval descriptor。CUDA preflight 已准备 `_ExplodingPoseDict` 来验证“不发生字典索引”，但尚未实际执行。

### 2. person-0 score confidence target 与 BCE

顶层传给 TAPF 的 target 为：

```python
pose_dict['scores'][:, 0] * pose_dict['person_mask'][:, 0:1]
```

TAPF 的 shape target仍来自 person-0 heatmap，confidence target独立来自上述 score；confidence head返回 raw logits，loss 使用 `binary_cross_entropy_with_logits(logits, target)`，没有把 sigmoid 后 confidence错误当 logits。

额外只读探针固定 heatmap、只把 score从全 0 改为全 1，报告的 `teacher_confidence` 从 `0.0` 变为 `1.0`，confidence loss同步变化，证明当前实现真正读取 score target，而不是退回 heatmap peak。

### 3. 四臂实际梯度拓扑

用彼此独立的 `torch.autograd.grad(..., allow_unused=True)` 探针复核 epoch 11，当前生产模块得到：

| arm | pose→anchor | pose→adapter | ReID field→anchor | ReID field→adapter |
|---|---:|---:|---:|---:|
| F0 | 0 | 0 | 0 | 0 |
| D0 | 1 | 0 | 0 | 0 |
| P0 | 0 | 0 | 0 | 1 |
| J0 | 1 | 0 | 0 | 1 |

因此 detach位置正确：pose loss不进 adapter/backbone，ReID field不进 anchor/F2，P0/J0只允许 ReID写入共享 `17×4` geometry adapter。

### 4. schedule、低带宽写入与单次 forward

- e1–5 student fraction=`0`；
- e6–10依次为 `0.2/0.4/0.6/0.8/1.0`；
- e11后固定 student；
- adapter只能输出每关节 `dx/dy/dlog_sigma_x/dlog_sigma_y`；
- anchor probability、confidence、joint pooling输入均正确 detach；
- 正式 field只能由有界均值/对角尺度重新渲染 Gaussian；
- Stage-2产生 field，Stage-3消费；没有第二次 PatchEmbed或 backbone forward。

实际参数量为 `85,542`：anchor `59,874`，geometry adapter `25,668`，满足 `<0.5M`。

### 5. PSG 数值链的结构

结构上没有把空间 field先 sigmoid 后又二次 sigmoid：

1. teacher raw heatmap或 `confidence × Gaussian` student field作为 PSG raw输入；
2. `PoseSpatialGate.forward()` resize后统一执行一次 `torch.sigmoid(hm)`；
3. sigmoid结果进入 PSG encoder。

confidence head自身的 sigmoid只定义关节可靠性 `c_j`，不是对最终空间 field预先执行 PSG sigmoid。

### 6. 配置与 H0 叙述

- 六份 YAML 均通过当前 schema加载；
- F0/D0/P0/J0去掉注释、`POSE_TAPF_MODE` 与 `OUTPUT_DIR` 后内容 SHA256 完全一致：
  `794878d3b6e7017713fcf55bff521d1a5b753307e91ff6e58742836aee828aa2`；
- 六臂统一 batch64、Swin-Tiny、seed1234、AMP scale1024、`RE_PROB=0`；
- TAPF构造器已 fail-fast 检查 target-person heatmap、pose dropout与 legacy shuffle；
- PSG与 TAPF构造均保存/恢复 RNG，四个 TAPF arm初始化路径一致；
- `design.md` 已准确把当前最小 LiteHR-style anchor与后续 `H0` 分开：`H0` 只是 bootstrap质量失败时的轻量 anchor容量对照，不是主方法创新、不得进入 Gate A，也不得用来挽救已判负的 P0机制；完整 HRNet被排除。

## 剩余 blocker

### [P0] CUDA preflight 的 raw→PSG 对照比较了不同分辨率，正确实现也会失败

`cuda_preflight.py:157-211` hook到的是 PSG 在 Stage-3 尺寸完成 bilinear resize并 sigmoid后的真实输入；但它拿该值与 TAPF 在原始 `96×32` field上计算的 `field_sigmoid_mean/min/max` 比较，并要求误差 `<=2e-4`。

两者不是同一个张量：

```text
hook:  sigmoid(interpolate(field, stage3_size))
stats: sigmoid(field)  # 96×32
```

bilinear resize与 sigmoid不交换，极值也必然随下采样改变。用当前生产模块的只读 probe缩放到 Stage-3 `12×4`，观察到：

| 统计 | hook语义 | 当前 reported语义 | 绝对差 |
|---|---:|---:|---:|
| mean | 0.543318 | 0.542924 | 3.95e-4 |
| min | 0.504771 | 0.501490 | 3.28e-3 |
| max | 0.616904 | 0.629393 | 1.25e-2 |

三项均有至少一项超过 `2e-4`；因此当前 preflight 会把正确的单次 sigmoid链误判为失败。

**必须修正**：要么在 hook处直接与 `torch.sigmoid(F.interpolate(field, stage3_size, ...))` 比较；要么让生产统计额外报告 PSG resize后的值。应做逐元素对照，而不是只比 mean/min/max。

### [P0] 正式 2×2 回归测试仍会用累积梯度掩盖交叉路径

`test_post_boot_2x2_gradient_semantics()` 仍先 backward pose loss，再 backward field，最后只问某组参数是否“曾有非零梯度”。因此 J0 中以下两类错误仍可通过：

- ReID错误进入 anchor，会被已有 pose→anchor grad掩盖；
- pose错误进入 adapter，会被随后 ReID→adapter grad掩盖。

本轮独立只读探针表明当前实现逻辑是正确的，但仓库正式测试没有锁住这一性质；后续重构可无声破坏。现有 CUDA preflight也只对“pose+ReID总 loss”做一次 backward，J0同样不能排除交叉污染。

**必须修正**：把 pose objective与真实 ReID objective分别 forward/求 `autograd.grad`，每次独立检查 anchor、adapter和至少一个 F2/backbone参数；optimizer delta再单独执行。由于 adapter末层零初始化，首个 P0 step只能要求末层更新；再做第二个独立 step后检查上游 adapter层获得非零梯度。

### [P1] 强 teacher-off 声明尚未覆盖训练 processor

模型 forward 已不读取 P0/F0 post-bootstrap pose，但 `processor.py:597-600` 仍在每个 iteration无条件执行：

```python
pose_dict = _pose_to_device(pose_dict, device)
```

也就是说 full training pipeline仍会遍历并把 external teacher tensor搬到 GPU，然后模型再忽略它。它不会造成 descriptor数值泄漏，但不满足上一轮验收条件中的“teacher tensor不再读取/搬运”强定义，也使损坏的 pose batch仍可能在进入模型前失败。

**必须二选一**：

- 对 P0/F0 epoch 11后在 processor中令 `pose_dict=None`，不再搬运；D0/J0保持原逻辑；或
- 把 claim明确收窄为“模型表示与损失不读取 external pose”，不再声称完整训练流水线关闭 teacher tensor。

### [P1] default-OFF、paired RNG、strict reload仍没有可执行回归

当前代码的条件分支与 RNG保存/恢复设计合理，但没有测试证明：

- `POSE_TAPF=False` 相对 exact legacy B0 的 state-dict keys、descriptor、final featmap逐元素一致；
- B0/R0/P0在相同 seed下共享参数与 PSG初值逐键一致；
- F0/D0/P0/J0在 e1–10逐步一致；
- TAPF checkpoint strict save/load后 descriptor与 optimizer state一致。

`cuda_preflight.py` 明确拒绝 `POSE_TAPF=False`，也没有 checkpoint roundtrip，所以不能覆盖这些门禁。

### [P1] CUDA/AMP 门禁脚本尚未执行，且需先修正上述误报

当前只完成 CPU standalone测试；没有 4090/3090 batch64 AMP、GradScaler、显存峰值和 optimizer step的实际 PASS记录。preflight脚本已经覆盖 production dataloader、ID+triplet、teacher score、模型级 teacher-off、梯度 norm和参数 delta的大部分必要路径，但必须先修复分辨率对照，并补独立交叉梯度/default-OFF/strict reload。

## 本轮实际执行结果

- `uv run --active python -m pytest -q experiments/exp378_e2e_pose/test_task_adaptive_pose_field.py`：
  `5 passed in 1.05s`；
- 六份 exp378 YAML schema load：全部 PASS；
- `git diff --check`：PASS；
- 独立四臂梯度探针：拓扑 PASS；
- confidence score target探针：PASS；
- Stage-3 resize后 PSG统计一致性探针：确认当前 preflight存在确定性误报；
- 未运行 CUDA、未启动训练、未修改生产代码/config/test、未提交 Git。

## 启动许可

当前不给正式训练许可。最短修复路径是：

1. 修正 preflight 的 Stage-3 resize后逐元素 PSG输入对照；
2. 把正式 2×2测试改成 pose/ReID独立梯度；
3. 明确并实现 full-pipeline teacher-off，或收窄 claim；
4. 补 default-OFF/paired-init/strict-reload测试；
5. 在 4090先跑四臂关键 epoch的 batch64 AMP preflight，3090复核至少 P0 e11；全部 PASS后再启动 Gate A。

TAPF主实现无需重写；剩余工作是让门禁准确验证它已经具备的语义，并补齐尚未证明的 production integration。

---

## 第二次修复快速复核（2026-07-16）

本节复核上一版报告之后的新修复；若与上文“剩余 blocker”冲突，以本节为准。本轮仍未修改生产代码、配置或测试，也未启动训练。

### 已解除的旧 blocker

1. **full-pipeline teacher-off 已修复。** `processor.py` 现在仅在 bootstrap期或 D0/J0长期 pose supervision需要 teacher时搬运 pose；P0/F0 e11+直接令 `pose_dict=None`。训练内 eval与独立 inference对所有 TAPF arm同样不再搬运 external pose。`flip_batch(..., None)` 能正确返回 flipped RGB与 `None` pose，因此 flip-test路径兼容。
2. **PSG 数值对照已修复。** preflight分别 hook TAPF送入 PSG的 raw field与 PSG encoder真正收到的 tensor，再按 Stage-3尺寸执行相同 bilinear resize和一次 sigmoid，最后逐元素 exact比较。它不再拿 `96×32`统计与 Stage-3 tensor极值错误比较；当前对照方向正确，不能由二次 sigmoid或错序 resize蒙混通过。
3. **正式 2×2 standalone测试已改为独立 autograd。** pose objective与 ReID proxy分别对 anchor、adapter、feature求 `autograd.grad`，不会留下累计 `.grad`，因此上一版指出的 J0交叉路径掩盖问题已经解除。
4. **四臂初始化与 roundtrip测试已加入。** F0/D0/P0/J0 standalone state dict在相同 seed下逐键 exact；TAPF module及其 SGD momentum state能 strict保存/恢复，恢复后 field exact一致。

本轮实际执行：

- standalone单测：`7 passed in 1.05s`；
- `git diff --check`：PASS；
- processor eval/inference与 `flip_batch(None)` 静态接线：PASS。

### 仍需修正：[P0] warm priming只能证明最末层连通，当前仍可能假 PASS

`cuda_preflight.py:154-185` 的想法是合理的：fresh process中的 PSG末层和 geometry adapter末层均为零初始化，所以先做一个 e10 step打开 PSG，再为 P0/J0做一个 e11 step打开 adapter，随后审计完整路径。这比直接在 fresh zero-init模型上要求上游 adapter梯度正确。

但当前实现还没有验证两个 priming step是否真的生效：

- `optimizer_step_at()` 不检查 loss/grad finite、GradScaler是否跳步或 priming参数 delta；
- e10应打开的 PSG final projection没有 delta断言；
- P0/J0 e11应打开的 geometry adapter最后一层没有 delta断言；
- 最终 `reid_to_adapter` 把整个 adapter所有参数合成一个 norm。即使上游 `LayerNorm/Linear` 被错误 detach，只要最后一个 `Linear(64→4)` 自身有梯度，总 norm仍大于零并通过。

因此存在具体假 PASS路径：e11 priming因 AMP overflow被跳过，或 adapter上游被错误断开；最终 audited forward仍可能只给零初始化输出层非零梯度，`reid_to_adapter > 0` 与一次 adapter delta检查都会通过，却没有证明完整 MLP已经由真实 ReID loss训练。

**最小修正**：

1. e10 priming前后分别记录 `s3_b0` PSG final projection，要求 finite且 delta `>0`，同时确认 scaler没有跳步；
2. P0/J0 e11 priming前后记录 geometry adapter最后一层，要求 finite且 delta `>0`，同时确认 scaler没有跳步；
3. 最终独立梯度把 adapter拆成 output layer与 upstream参数两组；P0/J0 warm后两组都必须 `>0`，F0/D0两组都必须为 `0`；
4. 将注释中的“reproduce the real warm state”改为“minimal connectivity priming”。一个 e10 step并不等价于真实训练十个 epoch；它只能验证连接与 AMP step，真实 warm-state数值质量仍要由正式 e1–10运行日志证明。

独立 pose/ReID autograd本身实现正确，不会因 total-loss累计梯度产生假 PASS；剩余风险集中在 priming是否真正打开完整 adapter。

### 仍未解除：[P1] 新测试不是 production default-OFF / paired-model / full checkpoint gate

新增 exact-init与 roundtrip都只实例化 standalone `TaskAdaptivePoseField`：

- 没有构造 `POSE_TAPF=False` 的 production B0并与 legacy forward/state keys比较；
- 没有逐键比较 production B0/R0/P0共享 backbone/classifier与 R0/P0 PSG初值；
- 没有 strict恢复完整 `PoseBackboneModel + production optimizer + GradScaler/scheduler`。

所以它们准确证明了“TAPF模块四臂同构、模块checkpoint可恢复”，但不能把上文 default-OFF与 production checkpoint门禁勾为已完成。

### 更新裁决

**本轮三项主要修复方向均正确，TAPF生产核心继续 PASS；正式 Gate 0仍为 `CHANGES_REQUIRED`。**

训练前只剩两类工作：先给 warm priming增加不可跳步的分层 delta/gradient断言并实际跑 CUDA batch64；同时补一次 production default-OFF/paired-model/full-checkpoint审计。完成并落盘 PASS后，本实现复审可转为正式 GO。

---

## 最终 Gate 0 裁决（2026-07-16）

本节基于修复后的生产代码、4090/3090原始日志与本地 SHA复核作最终裁决；若与上文阶段性 `CHANGES_REQUIRED` 冲突，以本节为准。

### 最终结论

**`GATE 0 PASS / GO_FOR_EXECUTION`。**

TAPF 的代码、配置、teacher生命周期、四臂梯度语义、AMP数值链、paired construction、初始 identity、strict reload与两机 batch64 production preflight均已通过。允许下一步冻结干净 exact execution commit并启动预注册 Gate A；本轮复审没有启动训练。

### 最后一项数值修复

真实 AMP preflight暴露了一个仅在生产数值精度下出现的问题：geometry adapter零初始化输出层的首个有效更新约为 `4.6e-8`，原 autocast路径会把该微小信号量化回零，使上游 MLP长期打不开。

修复后：

- Swin主干与 PSG继续使用 AMP；
- `17×384 -> 17×4` 的小型 geometry MLP及矩/渲染计算成为显式 FP32数值岛；
- 最终 field再转换回 feature dtype进入 PSG；
- 没有改变 batch、模型语义、loss、课程或四臂定义。

4090与3090 P0 e11均观测到：adapter output与 upstream梯度同时有限非零，optimizer delta非零；F0/D0仍保持 adapter严格不更新。因此该修复解决的是 AMP下的可训练性 bug，不是实验中途改机制。

### production model invariants

4090原始证据：

- `remote_artifacts/exp378_gate0_20260716/4090/model_invariants.log`
- SHA256：`171c0c1716410ba4e4dafa9d6556202e39e67d079afc275486097ab3a9097cfb`
- 结论：`TAPF_MODEL_INVARIANTS_PASS`
- production B0 keys=`211`，P0 keys=`259`，descriptor=`(2,768)`，featmaps=`4`。

该测试确认：

1. B0/P0所有 shared state逐键 exact；
2. F0/D0/P0/J0完整 TAPF state逐键 exact初始化；
3. 零初始化 PSG下，B0/P0初始 descriptor与全部四级 featmap逐元素 exact；
4. production P0 state strict reload后 descriptor与 featmaps逐元素 exact。

standalone测试另确认 TAPF module与 SGD momentum optimizer state strict roundtrip；本地最终执行为 `7 passed in 3.38s`。

### 4090 batch64 AMP 梯度矩阵

所有日志均报告 `TAPF_CUDA_PREFLIGHT_PASS`，GradScaler最终 scale=`1024.0`，raw field按 Stage-3尺寸 resize后只执行一次 sigmoid，teacher score逐元素等于 person-0 `scores × person_mask`。

| arm/epoch | pose loss | pose→anchor | pose→adapter/backbone | ReID→anchor | ReID→adapter output/upstream | anchor delta | adapter delta | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P0 e1 | 2.888818 | 1.387959 | 0 / 0 | 0 | 0 / 0 | 1.102e-3 | 0 | PASS |
| F0 e11 | None | 0 | 0 / 0 | 0 | 0 / 0 | 0 | 0 | PASS |
| D0 e11 | 2.882910 | 0.983022 | 0 / 0 | 0 | 0 / 0 | 1.389e-3 | 0 | PASS |
| P0 e11 | None | 0 | 0 / 0 | 0 | 1.081e-3 / 2.520e-10 | 0 | 3.746e-6 | PASS |
| J0 e11 | 2.877904 | 0.822319 | 0 / 0 | 0 | 1.103e-3 / 2.591e-10 | 1.554e-3 | 3.750e-6 | PASS |

原始日志及 SHA256：

- P0 e1：`remote_artifacts/exp378_gate0_20260716/4090/p0_reid_only_geometry_e1.log`，`295f4c7ae5087b58d56df0d1916fdb7807b057da5a6f092dfd2e20c42306d837`；
- F0 e11：`remote_artifacts/exp378_gate0_20260716/4090/f0_frozen_anchor_e11.log`，`65d0c7fb8b83600a22e90800b72c1bb68d87a81c7bc234d75452d7d5e9dfc021`；
- D0 e11：`remote_artifacts/exp378_gate0_20260716/4090/d0_continued_pose_e11.log`，`e3a2a97e56199a412b725603fbd61d11547f1a4de1972504b52a85baa03732e0`；
- P0 e11：`remote_artifacts/exp378_gate0_20260716/4090/p0_reid_only_geometry_e11.log`，`8d804aec15426b5e3a30140d7803b5ce5b33e9d84414c09f67e55c50bd3d6953`；
- J0 e11：`remote_artifacts/exp378_gate0_20260716/4090/j0_joint_control_e11.log`，`08982a699763616086b98cb847153f4edd24b06ee8c315f58ff50ae1bc440872`。

warm priming现已同时要求 e10 PSG final projection delta非零、P0/J0首个 ReID step的 adapter output weight delta非零；最终审计再把 adapter output与 upstream参数分组求独立 ReID梯度。GradScaler跳步、仅末层连通或 pose/ReID交叉污染均不能再假 PASS。

### 3090跨机复核

- 原始日志：`remote_artifacts/exp378_gate0_20260716/3090/p0_reid_only_geometry_e11.log`
- SHA256：`557ca6a6a2de3f886be3e603cb7b78d9afc7a4df398cd7224c48aba162fc111c`
- 结论：`TAPF_CUDA_PREFLIGHT_PASS`
- P0 e11：pose=None，anchor grad/delta=`0/0`，adapter output/upstream ReID grad=`1.092e-3 / 2.340e-10`，adapter delta=`3.749e-6`。

这证明 FP32 geometry island与梯度隔离不是4090单机偶然行为。

### 证据完整性

本地重新计算并核对两个 `SHA256SUMS`：4090六份日志、3090一份日志全部 PASS；六份 YAML schema load、`git diff --check`与 standalone `7/7`单测全部 PASS。

### 启动边界

Gate 0 PASS只授权执行已经预注册的 Gate A，不提前支持方法有效或创新成立。正式执行仍必须：

1. 冻结并记录 exact execution commit/archive SHA；
2. batch固定64、Swin-Tiny、`RE_PROB=0`，不得运行中修改机制；
3. 逐次记录所有 eval的 mAP/R1/R5/R10；
4. 按设计执行 P0/D0首轮趋势与同机 B0/R0/F0/J0归因；
5. e10→e120继续审计 anchor函数漂移、field统计与几何 adapter有效更新。

**最终实现审查状态：PASS。**
