# Codex Review — AIRL 梯度隔离单模型双分支 (`--airl_dualbranch_iso`)

**Verdict**: approve
**Date**: 2026-06-23 (round 2, 修复后重审)
**Review round**: 第 2 轮(round 1 的 Medium + 2 Low 修完重审)

## Round 1 Findings（已全部修复）

- **Medium — BN running-stat leak**: degraded forward `model(deg_imgs)` 走 train-mode iso 路径仍会 `self._embed(full_map)` 更新 f_full BNNeck running stats（degraded ground 图污染 f_full eval 头）。
  - **修复**: 训练环 degraded forward 改为 `model(deg_imgs, rec_only=True)`；`AFDModel.forward` 的 want_iso 路径在 rec_only=True 时只返回 `{bn_feat_rec, logits_rec}`，绝不 pool full_map → `self.bottleneck` 不在 degraded 图上做 BN forward。`_forward_swin_split(rec_only=True)` 额外 detach full_map。
- **Low — DropPath RNG 非忠实**: 原 split 在 f_full 共享 stage 之前 interleave 跑 rec stages，训练时 DropPath 消耗 `torch.rand` → 改变 f_full 看到的 stochastic-depth RNG 序列。
  - **修复**: `_forward_swin_split` 改为先跑完整 f_full stage 循环（捕获 detach 的 fork tensor），rec 拷贝在循环之后跑。
- **Low — smoke 未覆盖 iso_stage=2 / BN-stat**: 已补 smoke I4(rec_only BN-stat clean + keys)、I12(iso_stage=2 隔离)、I13(train-mode f_full RNG 忠实)。

## Round 2 Findings

None.

## Confirmed

- `rec_only=True` 对 f_full BNNeck 干净：iso 路径在 `_embed(full_map)` 调到 `self.bottleneck` 之前就返回 rec-only dict；trainer degraded pass 用 `rec_only=True` → f_full `running_mean/var` 不被 degraded 图更新。
- 梯度隔离对 iso_stage=3 与 iso_stage=2 都成立：fork = `x.detach()` at stage iso_stage 输入；rec 分支只跑拷贝的 rec_stages；iso_stage=2 正确拷贝/跑 stages [2,3]，rec CE 无法回传共享 trunk stage2。
- DropPath 顺序已修：`_forward_swin_split` 先跑原 Swin stage 循环（对齐 `SwinTransformer.forward`），rec 拷贝在之后跑 → train-mode f_full RNG 等价原 forward。eval legacy 路径仍走原 `self.swin(x)`，仅 return_dual=True 才走 split。
- 优化器/LR 分组正确：`backbone_swin.parameters()` 把拷贝的 rec_stages/rec_norm 放进缩放 Swin 组，bottleneck_rec/classifier_rec 留在 full-LR 头组；scheduler 用 per-group base_lrs，warmup/cosine 保比例。`airl_lambda_eff` 已含 `args.airl_dualbranch_iso`。
- OFF 行为保持：`airl_dualbranch_iso=False` 不构造 rec stages/head，forward/eval 走既有单特征路径。

## 结论

codex 审查通过。
