# Codex Review — iso 双分支 trunk-undersupervision 修复 (`--airl_iso_trunk_recce`)

**Verdict**: approve
**Date**: 2026-06-23
**Review round**: 第 1 轮（`codex --search exec -s read-only`，联网核对范式新颖性）

> 改动文件未 git-track（无 working-tree diff），故 Codex 直接读 4 个目标文件审查。

## Findings

- **Low** — `experiments/afd_reid/afd_model.py` (`want_iso`): `rec_only=True` 在
  `model.eval()` + `return_dual=False` 下会被忽略（`want_iso = airl_dualbranch_iso and
  (training or return_dual)`），eval caller 调 `model(x, rec_only=True)` 会静默拿到 f_full eval
  张量而非 rec-only dict。训练安全（degraded pass 在 `model.train()` 下），但契约脆弱。
  - **已修复**：`want_iso` 加入 `or rec_only`。
- **Low** — 多处旧注释/help 仍称 `--airl_dualbranch_iso` 下 f_rec ID-CE 永不回流 trunk（与默认
  `trunk_recce=1` 矛盾，clean ID-CE 现在有意回流）。涉及 `afd_train.py` 的 `--airl_dualbranch_iso`
  help、`afd_model.py` `_run_rec_stages` docstring、`AFDModel.forward` iso-path 注释。
  - **已修复**：三处注释/help 已改为"degradation-consistency 隔离 + clean ID-CE 按 trunk_recce 回流"。

无 Critical / High / Medium 功能性问题。

## Verified（Codex 确认）

- detach_fork 真值表：`(F,T)→F`、`(F,F)→T`、`(T,T)→T`、`(T,F)→T`。
- consistency 不泄漏到 clean trunk：degraded pass `rec_only=True`（afd_train.py:2212），
  clean 目标在 feat/kl 两 mode 都 detach（afd_train.py:1212 / 1220）。
- `trunk_recce=True` clean f_rec CE 经非 detach fork 回流 trunk（afd_model.py:281 / 292）；
  Swin stage 重新赋值 `x` 不原地改（swin_transformer.py:1088）。
- f_full head 在 degraded rec-only 不跑，consistency 只读 rec logits/feat（afd_model.py:760 /
  afd_train.py:2217）。
- flag 端到端贯通：argparse/cast/build/model/backbone（afd_train.py:1531/1544、
  afd_model.py:883/561/620）。
- `trunk_recce` 不引入新参数，纯 bool 路由开关；同权重同 RNG 下 on/off forward 数值一致
  （`detach()` 值保持）。

## 范式新颖性（联网核对）

Codex web search 确认这是**标准 stop-target 模式**，非"新但坏"的梯度技巧：clean 任务损失回流共享
表示、对辅助 consistency/蒸馏目标做 stop-gradient（detach 目标网络）。先例：Mean Teacher
（官方 PyTorch 实现对 EMA teacher 参数/目标 detach）、FixMatch（强增广对 weak-augmentation 伪标签
目标训练）。设计本身成立。

## 结论

codex 审查通过（approve）。两个 Low 项已全部修复（`want_iso` 加固 + 三处注释/help 去漂移），
重新 py_compile + smoke 18/18 仍 PASS。
