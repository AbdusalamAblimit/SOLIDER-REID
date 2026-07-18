# exp393 监控记录

## 当前状态

- `DESIGN-ONLY / PHASE 0E NO-START / PHASE A/B FORMAL TRAINING NO-START`；
- exp392 Phase 0C/0D已封板，禁止续训、重跑或修改其repo/config/checkpoint；
- semantic multi-stage保持NO-START；
- 当前远端无计算/训练进程，4090应为`2 MiB/0%`；
- 未创建exp393训练repo、config、output、runner或checkpoint。

## 2026-07-19 设计冻结前审查

exp393把下一步拆成两道单变量门禁：

1. Phase A仅把Semantic C0的zero expert改为非零branch+zero ReZero scalar，验证route activation；
2. Phase B仅在Phase A route alive后，把scalar q执行变量换成centered rich CLIP evidence code并加入
   router必经latent的内部alignment。

该顺序避免把“路由优化修复”和“CLIP teacher信息增量”混成一个bundled差值。Phase 0E teacher-only
必须先证明rich code具有within-slot动态、有效秩及correct-vs-wrong敏感性；当前不占用4090。

下一步先做只读代码seam审计和Phase 0E synthetic/8图脚本设计。任何正式训练前必须完成独立
config/diff/gradient ownership/RNG/optimizer/checkpoint/RGB-only/CUDA/AMP preflight；禁止因用户要求
继续CLIP方向而跳过门禁。
