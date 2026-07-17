# Claude Broad Review — iso 双分支 trunk-undersupervision 修复 (`--airl_iso_trunk_recce`)

**结论**: 审查通过（PASS）
**Date**: 2026-06-23
**Review round**: 第 1 轮（Opus 子代理广范围审查；全范围逐行）
**范围**: `experiments/afd_reid/afd_model.py`、`experiments/cargo_cvpb/afd_train.py`、`model/backbones/swin_transformer.py`、`experiments/cargo_cvpb/smoke_airl_iso.py`

## 背景：这次改的是什么

`--airl_dualbranch_iso` 把 f_rec 做成独立晚期 Swin stage（在 `iso_stage` 输入处从共享 trunk 分叉）。
原实现 fork 喂入**总是 detach**，于是 f_rec 的 clean ID-CE **和** degradation-consistency 梯度
**都**被从 trunk 切断 → trunk 少了一份身份监督，f_full 反而偏弱（诊断: iso 的 f_full ep20=45.56 <
baseline 48.98 < 全共享版 f_rec 47.39）。

修复（新 flag `--airl_iso_trunk_recce`，默认 on）：
- **clean forward**（`rec_only=False`）+ `trunk_recce=True`：fork 喂入**不 detach** → f_rec 的
  **clean ID-CE 梯度回流共享 trunk**（补身份监督，强化 f_full）。
- **degraded forward**（`rec_only=True`，consistency pass）：fork 喂入**始终 detach** →
  consistency 梯度保持 trunk 隔离。
- `trunk_recce=False`：clean fork 也 detach（原全隔离版，消融对照）。

## 五条关键不变量（全部成立）

1. **consistency 梯度绝不回流 trunk（两个 `trunk_recce` 取值都成立）**——双重切断：
   (i) degraded pass 用 `rec_only=True`，`detach_fork = bool(rec_only) or (not iso_trunk_recce)`
   恒为 True → `fork_x = x.detach()`；(ii) `airl_consistency_loss` 在 kl 与 feat 两种 mode 下
   都把 clean/original 端 `.detach()` 当作目标（kl: `p_o`/`log_p_o` detach；feat: `zo` detach），
   所以即使 `trunk_recce=True` 时 clean 端带活图，它也**只作为 detached 目标**进入 loss，贡献 0 consistency 梯度。consistency 只经 degraded `out_d`（detach fork）流动。
2. **`trunk_recce=True` 时 clean f_rec ID-CE 回流 trunk**：clean pass `detach_fork=False` →
   `fork_x = x`（活引用）。`iso_stage=3` 时末 stage `downsample=None`，rec map 由 block `out`
   （= 由 fork_x 计算的 block 输出）经 `rec_norm` 回传到 fork_x → trunk。`loss_ce_rec` 加进总 loss。
3. **f_full 路径 0 consistency 梯度**：degraded `rec_only=True` 只返回 `{bn_feat_rec, logits_rec}`，
   f_full 的 pool+BN+classifier 不在 degraded 图上跑；consistency loss 只读 `_rec` 张量。
4. **`--airl_dualbranch_iso` off 字节级复现 baseline**：`iso_branch=False` → 整个 iso 构造块跳过，
   forward 走 `not (iso_branch and return_rec)` 单图路径；`iso_trunk_recce` 无任何作用。
5. **forward 数值对 `trunk_recce` on/off 一致**：唯一差异是 `x.detach()` vs `x`，`.detach()` 共享
   storage/数值，只差 `requires_grad`/图；Swin stage `x = block(x,...)` 重新赋值不原地改输入
   （`SwinBlockSequence.forward`），所以 f_full/f_rec 数值逐位相同，仅反向图不同。

## 逐项检查（a–i）

- **(a) detach_fork 真值表**：`(rec_only=F, recce=T)→F`（clean 回流）；`(F,F)→T`（clean detach，消融）；
  `(T,T)→T`、`(T,F)→T`（degraded 恒 detach）。完全符合"degraded 恒隔离、clean 仅在 recce off 时隔离"。
- **(b) fork 捕获 + 非原地**：`fork_x` 捕获的是 stage `iso_stage` **输入**（stage 跑之前）；
  `swin_transformer.py` 的 stage/ block forward 全部重新绑定 `x`，无原地 mutate，非 detach 引用图有效。
- **(c) semantic embed 不阻断 trunk 梯度**：rec semantic Linear 冻结（`requires_grad=False`，且
  deepcopy 后重新置 False）；rec map 取自 `out`（semantic-embed 之前），与 `SwinTransformer.forward`
  完全镜像；`x*softplus(sw)+sb` 只改延续流且保 x 图；`semantic_weight.detach()` 是冻结常量，
  不阻断经 fork_x 的梯度。
- **(d) consistency 仅经 degraded out_d**：见不变量 1，两 mode 都 detach clean 端。
- **(e) eval 路径**：`airl_dualbranch_eval.extract` 在 `@torch.no_grad()` 下；eval+`return_dual=True`
  时 fork 非 detach 但 no_grad 不建图 → 无害，数值与 baseline f_full 一致（I1b/I17）。
  额外加固：`want_iso` 加入 `or rec_only`，使 `model.eval(); model(x, rec_only=True)` 也遵守
  rec-only 契约（Codex Low 项）。
- **(f) 优化器/LR 分组未变**：优化器从 `model.parameters()` 建，已含全部 rec 参数；本修复**不加参数**，
  只改梯度路径。`backbone_swin` 内的 rec_stages/rec_norm 进缩放 Swin 组，bottleneck_rec/classifier_rec
  进 full-LR 组，断言仍成立。
- **(g) off 路径**：`airl_dualbranch_iso=False` 时 iso 块不构造，`_forward_swin_split` 不可达，
  `iso_trunk_recce` 零作用。
- **(h) AMP/dtype/shape/None**：rec 路径与 f_full 同一 autocast；degraded forward 在
  `autocast(enabled=not no_amp)`，consistency 在 `autocast(enabled=False)` fp32，与 `--airl` 一致；
  无新 None/shape 分支。
- **(i) flag 端到端**：argparse `--airl_iso_trunk_recce`(int default=1) → `bool()` cast →
  `build_model` getattr → `AFDModel.__init__(airl_iso_trunk_recce=)` →
  `SwinBackboneReID(iso_trunk_recce=)` → `self.iso_trunk_recce`。完整贯通。

## Smoke 证据（smoke_airl_iso.py，18/18 PASS）

- **I14（核心，分解）**：同一 iso 模型上分别 backward 两个梯度源——(a) clean ID-CE → trunk `early` 非零 + rec stage；(b) consistency → trunk `early/patch_embed/shared_last` 全 None（零）、f_full head None、rec stage/bnrec 非零。**直接证明"clean 回流、degraded 隔离"**。
- **I4**：纯 consistency step → trunk None、f_full head None（修复为默认后隔离仍成立）。
- **I9/I12**：clean ID-CE 回流 pre-fork trunk；f_full 自己的 forked stage 保持零。
- **I16**：`trunk_recce=0` → trunk None（恢复原全隔离）→ flag 正确切换。
- **I17**：on/off 数值（eval+train）逐位相同 → 纯反向路由开关。

## 结论

修复正确：`trunk_recce=True` 把 clean f_rec ID-CE 回流 trunk 强化 f_full，degradation-consistency
经"degraded 恒 detach fork" + "consistency loss 的 clean 端 detach 目标"双重保证仍 trunk 隔离；
不增参数；iso off 时退回 baseline。审查通过。
