# Codex Review — AG-ReID.v2 接线(--dataset agreid_v2)

**Verdict**: approve
**Date**: 2026-06-23
**Review round**: 1
**Command**: `codex --search exec -s read-only --skip-git-repo-check`(联网,xhigh,~120k tokens）

## Findings
- Critical: none
- High: none
- Medium: none
- Low: none

## Checks(codex 原文摘录)
- **Direction recovery is correct.** combined query/gallery layout + `view` filtering in
  `filter_by_view` / `run_cross_view_eval` 给出:
  - A→G = exp1 query Aerial vs exp1 gallery Ground
  - G→A = exp4 query Ground vs exp4 gallery Aerial
  - exp4 aerial gallery cannot pollute A→G because A→G uses `gallery` filtered to `Ground`.
- **Shared test pid relabel is correct.** combined adapter 用一个 `test_pid2label` 覆盖两协议
  query/gallery,matching folder-name identities 两方向共享同一 int pid。
- **`eval_market` junk removal 不会误删正样本。** 只删 same `(pid, camid)`;exp1/exp4 用 C0 vs
  C3 camid,跨平台 match 永不同 cam。
- **Cargo default path 行为未变。** `--dataset` 仍默认 `cargo`(L1241),首分支仍实例化 `CARGO`
  (L1747),新 AG-ReID.v2 路径隔离在 `elif args.dataset == 'agreid_v2'` 后。

## 结论
codex 审查通过(approve,0 findings,4 项正确性主张独立确认)。
