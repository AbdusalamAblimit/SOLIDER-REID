# exp374 训练/评测前 Codex 交叉审查

## 审查方式

用户要求任何训练或正式测试开始前完成充分审查，并明确禁止 Claude。本实验以多路
独立 Codex 红队替代旧流程中的 Claude 审查，覆盖：

1. PSG 与端到端 pose 新颖性；
2. transport/graph/attention 直接先例；
3. 数学可归约性与计算可行性；
4. 反事实设计与统计门槛；
5. checkpoint provenance、实现入口与资源安全。

## 当前总裁决

- 设计：`PASS_FOR_AUDIT_SCRIPT_DESIGN`
- unit/synthetic tests：`NO_GO_FOR_TESTS`
- 正式 Gate A/训练：`NO_GO_FOR_EXECUTION`

统计、工程与独立总红队已分别完成第三轮只读签字；该 PASS 只授权编写 audit-only
脚本。正式脚本尚不存在，provenance 也只能支持 legacy screen，因此当前仍禁止运行
测试、训练、正式评测或用旧 flag 拼出近似干预。

## 分项裁决

| 审查项 | 状态 | 裁决 |
|---|---|---|
| 原 PSG 新颖性 | 完成 | FAIL：WACV 2020 + SFT/FiLM 直接覆盖 |
| joint pose+ReID 新颖性 | 完成 | FAIL：PABR/VPU/VI-ReID 已覆盖 |
| 原 UBCFT 数学新颖性 | 完成 | FAIL：可归约 residual attention/GAT/Laplacian |
| 原 heatmap W2 trust region | 完成 | FAIL：计算量与约束漏洞阻塞 |
| source/demand + skeleton + Sinkhorn 改写 | 完成 | FAIL：差分不足，易被解释为 HOReID/FRT/RFC + UNITE/SOT/Sinkformers 拼接 |
| Gate A 对应依赖协议 | 完成 | 三路第三轮均签 `PASS_FOR_AUDIT_SCRIPT_DESIGN` |
| checkpoint 文件完整性 | 完成 | PASS |
| exact execution provenance | 完成 | FAIL：目录复用、文档与当前 checkpoint 日志错代、无 Git SHA |
| true bypass 语义 | 完成 | PASS：同模型传 `pose_dict=None` |
| matched donor/centroid 实现 | 未实现 | 设计已唯一化；脚本、静态审查和 synthetic test 均未开始 |
| per-query/层级 bootstrap | 设计完成 | 两 primary contrasts 统计设计 PASS；实现仍未开始 |
| 资源安全 | 设计完成 | 492 passes、4.25–4.5h、矩阵 hash 后释放、80GB 门槛；执行 preflight 未做 |
| 不确定度约束联合 transport 数学对象 | 未完成 | BLOCKED：当前只有问题描述，没有清晰联合目标与可行域 |
| 2026 TTPM / Pose-Guided Feature Restoration 全文边界 | 部分完成 | TTPM 已核；后一篇仍 BLOCKED，阻止新机制训练 |

## 已否决的执行捷径

1. 不用 `heatmap=0` 冒充 no-pose；
2. 不用 `POSE_ENABLED=False` 做同 checkpoint bypass；
3. 不用 training-only `POSE_SHUFFLE` 做正式评测；
4. 不用 LGPA 的 fixed-bands flag 干预 PSG；
5. 不把 exp371 的 LGPA correct/shuffle 结果外推到 PSG；
6. 不在全量 gallery 上构造稠密 Hungarian cost；
7. 不在看到指标后修改 donor、centroid、阈值或解剖组；
8. 不因旧 checkpoint provenance 不足就把 legacy screen 包装成正式复现。

## 代码实现前置要求

### 第三轮签字摘要

- 统计红队：`PASS_FOR_AUDIT_SCRIPT_DESIGN`；
- 工程红队：`PASS_FOR_AUDIT_SCRIPT_DESIGN`；
- 独立总红队：`PASS_FOR_AUDIT_SCRIPT_DESIGN`。

三路均明确：该签字不授权测试或正式评测。

设计交叉审查通过后，audit-only 脚本至少必须实现并静态证明：

- checkpoint/config/path/content SHA manifest；
- no-flip correct parity 与历史 flat 日志逐 seed 对齐；
- query/gallery split-local、exact-person-count 的 sparse bijective donor matching；
- PID 不同、person count 相同、无 fixed point；
- 冻结 nuisance/cost/k/Gumbel/solver/tie-break/20 mapping 与 1,000 baseline seeds；
- 在两个 PSG block 实际消费的 sigmoid-resized tensor 上通过弱干预门禁；
- train-only final-scene centroid fitting、zero-padded translation 与输入能量保持；
- flip 只作 secondary，并对同一受控 scene bundle 同步执行；
- audit-only 三态 override、strict state load、active-module inventory 与同模型 true bypass；
- correct、20 shuffle、centroid secondary、bypass、七组 group-bundle corruption sensitivity；
- per-query AP/R1/margin；
- 三 seed 固定 paired blocks + 同步 PID-cluster bootstrap；
- shuffle/bypass 两个 primary contrasts 的 one-sided simultaneous max-deviation intervals；
- 七组只作 secondary simultaneous sensitivity，不触发 GO；
- create-exclusive output、atomic publish、逐项 hash resume 和异常 fail closed；
- 逐臂释放 feature/distance matrix，启动前目标卷至少 80 GB。

## 下一轮审查顺序

1. 编写 audit-only 脚本，不运行；
2. 多路代码静态审查；
3. 代码审查 PASS 后才运行 unit/synthetic tests；
4. 测试结果与资源 preflight 再审；
5. 总裁决改为 `PASS_FOR_LEGACY_GATE_A` 后，才允许正式评测。
