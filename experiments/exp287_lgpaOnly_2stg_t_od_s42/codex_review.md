# Codex Review — exp287_lgpaOnly_2stg_t_od_s42

**Verdict**: approve
**Date**: 2026-04-20 23:30
**Review round**: 1

## Findings

零代码改动。相对 exp261 单变量改: `MODEL.POSE_SKELETON_GCN` True → False (PSG stages 保持 default `[-2,-1]`)。

**最简洁的 Phase 3 override**,仅 1 个布尔标志。和 exp286 配对构成 Phase 3-C 2×2 矩阵的 "2-stage" 列,和 exp261 (GCN on) 构成 "GCN on/off" 对照。

srvC auto-chain from exp286 via queue_on_ckpt daemon (默认 python3,srvC 无需 conda env 特殊处理)。

## 结论

codex 审查通过。
