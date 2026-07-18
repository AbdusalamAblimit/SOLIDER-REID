# 实验 exp390：官方干净 TAPF matched 多 seed

## 动机

官方最后代码上的 clean seed1234 结果为：Occluded-Duke B0=`57.4/67.4/80.6/85.2`，
D0=`57.6/67.7/80.8/84.6`，D0−B0=`+0.2/+0.3/+0.2/−0.6`。Market 同 seed 的
D0−B0=`+0.4/+0.2/+0.1/+0.1`。两个训练域的 mAP 都是小幅正差，但只有单 seed，无法判断是否
超过运行方差。clean hierarchical 已明确为负，不再追加结构变体。

本实验只补最必要的稳健性证据：在 Occluded-Duke 上增加 seed4321 与 seed2025 的 matched B0/D0，
与 seed1234 合并为三 seed。它不引入新方法、超参数或数据变量。

## 核心假设

若完整 `anchor+PSG` 的 clean 增益可重复，则新增 seed 的 D0−B0 mAP 应总体保持非负，三 seed
均值应为正，且四项差值与标准差必须完整报告。若均值不为正或方向高度不稳定，当前 MMAsia
headline 必须降级；不得挑选正 seed、best checkpoint 或中途 epoch。

## 单变量与执行顺序

每个 seed 内唯一方法变量为 `MODEL.TAPF.ENABLED`：

| 顺序 | arm | seed | output |
|---|---|---:|---|
| 1 | official B0 | 4321 | `log/occluded_duke/exp390_clean_swin_tiny_b0_s4321` |
| 2 | clean D0 | 4321 | `log/occluded_duke/exp390_clean_swin_tiny_d0_s4321` |
| 3 | official B0 | 2025 | `log/occluded_duke/exp390_clean_swin_tiny_b0_s2025` |
| 4 | clean D0 | 2025 | `log/occluded_duke/exp390_clean_swin_tiny_d0_s2025` |

严格串行，上一 arm 完整 e120 终审后才启动下一 arm。每臂均 fresh，不加载 B0/D0 final，不续训。

## Matched recipe

- SOLIDER 官方最后代码与同一 official Swin-T teacher；
- Occluded-Duke 原始 RGB split=`15618/2210/17661`；
- D0 只使用 exp386 fresh ViTPose-H train-only artifact，query/gallery 无 pose；
- batch64、120 epoch、SGD、lr=`0.0008`、semantic weight=`0.2`；
- identity sampler、384×128、flip/pad/crop/Random Erasing、eval10、checkpoint120；
- B0 与 D0 在同 seed 内使用完全相同的 seed、数据、增强、sampler、optimizer 与训练长度；
- D0 保持 exp387 的单层 Stage-2 anchor→Stage-3 两个独立 PSG，`HIERARCHICAL=False`。

D0 新增 config 相对 seed1234 canonical config 只允许修改 `SOLVER.SEED` 与 `OUTPUT_DIR`。B0 的
official-clean 有效启动边界不是 YAML 单文件：exp385 正式执行在 canonical YAML 之外还通过 CLI
固定 `MODEL.PRETRAIN_CHOICE=self` 与 official teacher path。exp390 的 B0 config 必须把这两个既有
固定覆盖显式收进 YAML，形成自包含配置；将这两个 official 覆盖归一化后，B0 相对 seed1234 的
唯一变化仍只能是 seed/output。

## 启动前门禁

1. D0 config 的结构化 diff 仅含 seed/output；B0 config 相对“canonical YAML + exp385 official
   teacher CLI 覆盖”的有效配置仅含 seed/output，四个 SHA 均固化；
2. exact execution commit、full-history bundle、teacher/data/pose manifest 不变；
3. clean TAPF unit 6/6、pose data unit 5/5；
4. B0 config-off 与 official clean 路径保持 exact；D0 `HIERARCHICAL=False` 与 exp387 路径 exact；
5. 对新增 seed 复核 B0/D0 公共 state、构造 RNG、optimizer 成员与超参数；
6. D0 复跑真实 paired batch64/8-worker CUDA/AMP、route/gradient、overflow、strict state 与
   correct/shuffle/None/exploding pose-free 门禁；
7. 每臂启动前确认 fresh repo、exact commit/config、output/runner 不存在、GPU 空闲且无其他训练。

代码与数据门禁在四臂间共享；每次只重做 seed/config/fresh execution 边界，不把相同代码重复实现。

## 结果与裁决

每个 arm 必须自然跑满 e120，以 final 而非 best 报告 mAP/R1/R5/R10。完成后报告：

1. 每个 seed 的 B0、D0 与 D0−B0 四项差值；
2. 三 seed B0/D0 mean±std；
3. 三 seed paired D0−B0 mean±std；
4. 三 seed 中正/负方向计数；
5. 每个 arm 的 PID/GPU/checkpoint/SHA/strict finite/异常终审，D0 另做参数轨迹与 pose-free exact。

不预注册基于单点的停止阈值；四臂均须完整结束。只有执行异常允许停止并修复，不能因性能负向停掉
剩余 seed。

## 风险与失败解释

1. paired 均值接近 0 或 std 大于均值：clean 效应不稳定，论文必须如实降级；
2. 某 seed 为负：仍完成全部 arm，不追加超参数救场；
3. B0 跨 seed 波动大：说明单 seed `+0.2` 不具判别力，不能用 Market 单点抵消；
4. D0 参数真实学习但性能不稳：结论是结构可训练而效果不稳，不归因于 dead branch；
5. Video、hierarchical、更多 stage/width/loss 均不作为失败后的即时替代实验。
