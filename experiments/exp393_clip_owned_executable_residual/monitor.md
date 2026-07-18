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

## 2026-07-19 代码seam与远端状态复核

远端真实状态：4090=`2 MiB/0%`，无`train.py`、Phase 0或exp392/393审计进程；exp392正式repo仍为
exact HEAD=`ed5783416528be4284adce11fa192fe119e344f4`且tracked clean，唯一checkpoint仍为
`transformer_120.pth`。没有重启、续训或修改封板资产。

只读代码审查确认Phase 0D归因与实现一致：Semantic C0的support/presence/mask heads从zero logits
起步；consumer mask/support在进入router前detach；router token/context projection为非零随机初始化，
但expert exact-zero，因此首步只有expert可从ReID loss离零，token/context必须等expert非零后才有
梯度。正式执行路径的AMP optimizer step只有一次；早先合并显示出的“双step”是截断输出拼接造成的
误读，已用本地行号、git blame和远端execution HEAD三方逐行排除，不构成exp392结果bug。

设计审查发现并修正一个真实计算图问题：旧稿把`L_exec`写在expert/alpha之前，却声称它能更新
ReZero branch。冻结后的修正版改为对真实pre-alpha branch proposal做teacher relation alignment，使用
共享生产参数和detached-token重算，使梯度实际到达token/context/evidence projection与expert；alpha
仍只由ReID loss打开。Phase 0E失败现在只阻断Phase B，不再无关地否决Phase A的route activation诊断。
