# exp371 CASD 监控

## 当前状态

- 阶段：大调研完成，候选收敛，设计已写；尚未启动训练
- 主方案：CASD（Cross-instance Anatomical Support-Advantage Distillation）
- IPER 位置：仅作为 support-quality 因果门禁/辅助权重，不作为 headline
- 当前训练进程：无
- 当前 GPU 占用：未因 exp371 启动任何任务

## 已完成

- [x] 审计 `exp109/148/335/336/337/340/353/357/358/370` 的真实证据边界
- [x] 核对 LGPA descriptor 真实维度和 test-time pose 依赖
- [x] 核对 PAFormer、PGFL-KD、TSD、PFD、BPBreID、KPR、PAT、SAP、ProFD、DROP、PASS、SPT、PGMAN
- [x] 核对 PDiscoNet、Invariant Slot Attention、SoftMoE、OT/privileged KD、residual/exclusive KD 邻域
- [x] 二次查新发现 AAAI 2020 UMTS 已覆盖 multi-shot teacher → single-shot student，并据此把 CASD 收紧为 part-wise leave-one-view-out support advantage
- [x] 发现 2022 `Pose-guided counterfactual inference` 精确撞名并降低 IPER 优先级
- [x] 排除 CLIP query 换皮、普通 pose KD、slot/write-back、OT/MoE、masking、matching 路线
- [x] 冻结唯一主方案与停止规则
- [x] 创建长期 Goal；其总目标仍是保留 LGPA 涨点并改造成自有创新
- [ ] 替换 Goal 的旧 IPER 主方案正文：工具不支持原地编辑 active Goal，需清空/结束旧 Goal 后按 CASD 正文重建；在此之前以本设计为执行真值

## 尚未执行

- [ ] Gate A：CLIP/random/learned query 归因
- [ ] Gate B：exp336 checkpoint inference intervention 矩阵
- [ ] Gate C：same-image / correct cross-image /伪 support 的 identity-relation advantage oracle
- [ ] Gate D：5376-D→768-D frozen oracle
- [ ] Phase 1：缓存特征 CASD 六臂 kill-switch

## 当前判断

**允许继续做廉价门禁，不允许直接开完整训练。**

原因：UMTS 已证明普通 multi-shot teacher-student 不是新意。CASD 的新颖性依赖“pose-organized part support + leave-one-view-out + support-vs-self advantage”相对 same-image KD、full multi-shot KD 和伪 pose support 都有独立价值；这尚未被数据证明。先在缓存 teacher parts 上验证 support coverage、identity margin 与 controls，能避免再投入一次机制工作但身份指标不动的完整训练。

## 保护事项

- `experiments/decisions.md` 当前包含用户未提交的 #99/#100 改动；本阶段不修改、不暂存该文件。
- 不修改现有 tracked 模型/config，不启动 3090/4090 训练。
- 后续若使用 Python，必须先在工作目录通过 `uv` 建立环境。
- 禁止 Claude；正式训练前的审查改由 Codex 与可复现机制测试完成。
