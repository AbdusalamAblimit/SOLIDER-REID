# Codex Review — exp359 LM-ReID

**Verdict**: approve
**Date**: 2026-06-25
**Review round**: 2（round 1 = needs-attention，5 findings 全修 → round 2 = approve）

## Round 1 Findings（全部已修，逐条对应）
- **High-1** [line 173]: 每图 RNG `seed+idx` → 每 epoch 同一组 lattice variants（无多样性，废掉 lattice-invariance 学习）。**修**: `LatticeTrainSet.set_epoch(ep)` 每 epoch 调 + seed 混 epoch（`seed = cli.seed*1000003 + epoch*9973 + idx*2654435761`）；DataLoader persistent_workers 默认 False → worker 每 epoch 重 fork 看新 epoch。
- **High-2** [line 48/118]: M=2+`j%3` 只训 canonical+bbox（没训 phase/zoom/kernel）。**修**: `make_lattice_variants(rand_mode=True)` 每非 canonical variant 随机 axis+kernel；返回 (variants, axes)；M 默认→3。
- **Medium-3** [GRL]: λ 双重缩放 → 梯度 λ²。**修**: `GradReverse.apply(z, adv_lamb)` + loss 加 L_adv 权重 1.0（single-lambda DANN）。
- **Medium-4** [L_id triplet over B*M]: 同图 variants 当正样本。**修**: per-slot `mean_m batch_hard_triplet(gf_bm[:,m], y)`。
- **L_adv 真 axis**: disc 输出→4（axes 0=canon/1=phase/2=bbox/3=zoom）；`axis_lbl=axb.reshape(B*M)` image-major 对齐 gf。

## Round 2 Verdict（codex 原文）

**Verdict: approve** — Round-1 fixes are correctly resolved. No new shape/alignment/dtype/runtime bug found.

Confirmations:
- High-1 resolved: `ds.set_epoch(ep)` runs before `iter(loader)` each epoch; persistent_workers default False → workers recreated per epoch see updated epoch.
- High-2 resolved: default M=3; rand_mode=True; non-canonical variants sample real mode+kernel; axes returned.
- Medium-3 resolved: standard single-lambda DANN (disc CE added once; feature grad reversed+scaled once by adv_lamb).
- Medium-4 resolved: triplet per lattice slot over [B,D] with labels y, averaged over M.
- L_adv real-axis alignment correct: `xb.view(B*M,...)` and `axb.reshape(B*M)` both image-major `[i*M+m]`.

Findings: None. Only non-blocking stale comments (line 15/308 还写 variant/slot)。

## 结论
codex 审查通过（round 2 approve，0 findings，仅非阻塞注释）。双审（Claude 自审 + Codex 独立审）均通过，可进 smoke + 训练。
