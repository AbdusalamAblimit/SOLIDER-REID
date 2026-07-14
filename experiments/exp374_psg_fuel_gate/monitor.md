# exp374 PSG 图像—姿态对应依赖门禁 — 监控记录

## 2026-07-15 — Goal 激活

- 状态：`PRE_EXECUTION_REVIEW_ACTIVE`
- 训练进程：无
- 推理/正式评测进程：无
- GPU：本实验未占用 3090/4090
- 当前动作：只读文献、数学、资产、因果协议与资源审查
- 约束：禁止 Claude；改用多路 Codex 红队交叉审查

## 2026-07-15 — PSG 与端到端 pose 查新边界

已完成两份专项审计：

- `experiments/paper_notes/psg_novelty_audit_20260715.md`；
- `experiments/paper_notes/end_to_end_pose_reid_novelty_audit_20260715.md`。

结论：

1. 原 PSG 是 WACV 2020 gated fusion 和 SFT/FiLM 类条件调制的受限实例；
2. PABR、VPU 与 VI-ReID pose auxiliary 已覆盖 pose branch 由 ReID loss 微调、
   联合 pose/ReID 训练和 pose loss 保语义；
3. freeze-then-unfreeze 只能是训练策略，不能承担主创新。

## 2026-07-15 — 数学/因果红队

原 UBCFT 公式与 trust region 触发两个阻塞项：

1. `Y=X+lambda(T-I)VX` 可归约为 residual attention/GAT 或图拉普拉斯扩散；
2. 全分辨率 heatmap 离散 W2 在计算和约束上均不可接受；batch 平均预算允许单个
   样本/关节严重漂移。

这一版 Gate 0 口径后来被正式复审替代，不能继续作为当前执行条件。当前 Gate A 只把
matched-shuffle 与 true-bypass 作为两个 primary contrasts：三 seed 均同号、最小均值
至少 `+0.30 mAP`、两个 simultaneous mAP LCB 均大于 0，且两个 R1 LCB 均不低于
`-0.50 pp`。scene-centroid 与七个解剖组均为 secondary，不触发 GO/NO-GO。

## 2026-07-15 — 文献红队

以下宽泛 claim 均已被封住：

- pose-aware ReID + OT；
- pose-guided graph/message passing；
- multi-stage pose attention/modulation；
- 从可见证据向遮挡区域传播；
- joint pose estimator + ReID。

第三路 source-to-sink 专项查新又确认，HOReID/FRT/PIRT/HUPOR/RFC 等工作分别并
在组合上覆盖 source reliability、visible-to-occluded recovery、pose topology 与
confidence-conditioned mixing；UNITE 已覆盖 keypoint-map 条件下对缺失身体部位
做 learned-mass UOT feature transport，SOT 与 Sinkformers 已覆盖 ReID OT feature
transform 和双随机 attention。因此，完整四元组合“未逐式同构”也不足以支撑主创新，
当前新机制正式为 FAIL/NO-GO。只有先形成差分足够清楚的不确定度约束联合优化问题，
才允许重新审查。TTPM 全文现已核为 pose-patch matching、confidence filtering 与
texture decoder 的强邻居；Pose-Guided Feature Restoration Transformer 正文仍是
额外阻塞项。

## 2026-07-15 — 资产与实现预审

4090 上三枚 exp007 checkpoint 的文件和 SHA 完整，GPU 空闲；但发现：

1. 同一输出目录存在两代测试日志；
2. 当前 checkpoint 对应 flat 日志，而文档引用旧 nested 日志；
3. 训练日志无 Git SHA；
4. 现有 shuffle/fixed-bands/zero-input 均不能直接实现 PSG Gate A；
5. 没有现成脚本同时满足 matched shuffle、scene-centroid、true bypass、
   per-query statistics 和哈希门禁。

当前裁决：

- 资产完整性：PASS；
- exact execution provenance：FAIL；
- 正式 Gate A 立即执行：NO-GO；
- 下一步：冻结设计，编写 audit-only 脚本，再做静态、合成、统计和资源复审。

## 2026-07-15 — 第一轮设计复审 FAIL，协议重写

统计红队否决了初稿中的 composite max-control percentile bootstrap、seed bootstrap、
未定义的 20-shuffle 聚合和把解剖组破坏作为 GO 条件。工程红队又证明
`camera × person_count` 硬分层存在 singleton，完美匹配数学上不可行；person-level
canonical 也无法保证 PSG 最终接收的 max-merged scene 输入只改变几何。

设计已按下列方向重写，并完成第二轮与独立总红队复审：

1. 两个 primary controls 分别定义 paired contrast，使用同步 one-sided simultaneous interval；
2. 三个 seed 固定为 paired blocks，只对 query PID cluster 重采样；
3. 20 mappings 先在 per-query AP 上等权平均，不平均 descriptor/distance；
4. 解剖组降为 group-bundle corruption secondary，不触发 GO；
5. donor 硬约束缩为 split/exact-person-count/different-PID/bijection，其余全进 soft cost；
6. scene-centroid 直接在 final scene heatmap 上做 zero-padded integer translation，
   但因多人峰间结构仍保留而降为 secondary；
7. 历史主口径固定 no-flip，correct 必须先复现 flat 日志；
8. Occluded-Duke 明确锁为 development，Partial-REID 锁为方法冻结后一次性确认。

第二轮统计主体 PASS；工程与独立总红队仍判 FAIL，指出 matching cost、Gumbel、弱干预、
centroid 边界、final-scene override、R1 聚合、反向 NO-GO、manifest/恢复和资源预算尚未
冻结。设计已逐项补齐，目前等待第三轮只读签字；签字前仍禁止实现脚本。

## 2026-07-15 — 第二轮/独立总红队后的设计修订

- matched-shuffle 的 95 维 nuisance、robust scaling、cost、soft weights、稀疏 k 序列、
  20 mapping seeds、Gumbel scale、solver tie-break 与 1,000 baseline seeds 已冻结；
- exact person-count 的 Hall/稀疏完美匹配失败仍 fail closed，不允许事后分箱；
- 在 PSG 真正消费的 `sigmoid(bilinear_resize(H))` 上新增干预强度门禁；
- final-scene override 三态入口、strict state load、UNSET/override seam parity、active
  module inventory 与 correct 前后 parity 已写成硬约束；
- primary 只剩 shuffle/bypass；R1 的 20-mapping query-level 聚合、bootstrap RNG、
  quantile 方法和反向 NO-GO 已唯一化；
- 492 passes 约 4.25–4.5 小时，大矩阵只留 SHA 后释放，目标卷启动前至少 80 GB；
- 三路第三轮只读签字均为 `PASS_FOR_AUDIT_SCRIPT_DESIGN`；只授权编写 audit-only
  脚本。unit/synthetic tests 仍 `NO_GO_FOR_TESTS`，正式 Gate A/训练仍
  `NO_GO_FOR_EXECUTION`。

## 下一检查点

1. 实现 audit-only 脚本但不运行；
2. 脚本实现后先静态审查，不立即运行；
3. 单元/合成测试也要在代码审查 PASS 后才启动；
4. 所有单元测试和 preflight PASS 后，才允许 4090 顺序执行 Gate A。
