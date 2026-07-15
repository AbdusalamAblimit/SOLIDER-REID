# exp375 PRSM 监控记录

## 2026-07-15 — 设计与实现启动

exp374 已正式 `COMPLETE / NO_GO`，因此本实验独立于 PSG gate 小变体。三路并行查新、仓库
接入检查与审稿人红队已完成：直接空间仍存在，但简单 `pose -> Delta/B/C`、解剖扫描顺序
和 skeleton state fusion 都已有强先例。主线收敛为 pose-routed part-state write/retain。

当前状态：`IMPLEMENTATION_IN_PROGRESS`。下一步直接完成模块、默认关闭配置、CPU
forward/backward 与反事实单测；随后做多路 Codex 代码/科学审查。审查通过后立即在 GPU
做单 batch AMP smoke，并启动 B0/M0/P0，不再增加形式性门禁。

## 2026-07-15 — 实现审查与本地机制测试 PASS

PRSM、默认关闭配置、B0/M0/P0 三份隔离 config、模块单测与真实 Swin 集成 smoke 已完成
编码。三路静态审查在修复 YAML、canonical cache、pre-write read 和 target-only 集成覆盖
后全部 PASS。

仓库 uv 隔离环境执行：

```text
.venv-exp374/bin/python tests/test_pose_routed_selective_memory.py
PRSM mechanism checks: PASS
```

该测试覆盖 shape/finite、关键参数与输入梯度、zero-pose exact identity、correct/shuffle
敏感性、双向纵翻转等变性、uniform 对 pose bitwise 不敏感及 CPU bfloat16 autocast。当前
状态升级为 `PASS_FOR_REAL_MODEL_GPU_SMOKE`；下一步只做 4090 production
B0/M0/P0 forward/backward/reload smoke，通过后直接启动训练。
