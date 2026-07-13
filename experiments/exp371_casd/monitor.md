# exp371 CASD 监控

## 当前状态

- 阶段：外部查新仍在收尾；Gate B / Gate D 单 seed 已完成；尚未启动训练
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
- [x] 在 4090 找回 exp340/340c 的原始 checkpoint、train/test logs 与 SHA；canonical fixed-random `59.9/68.7` 高于 CLIP `59.5/68.1`，共同 global `58.8/67.8`
- [x] 实现 query mode 枚举；random-frozen/random-learned 初值逐 bit 相同，仅差 3072 个可训练参数
- [x] 实现 Gate B 五臂评测与缓存脚本；shuffled 为 query/gallery 内异 PID 双射，uniform 为 common-body-support
- [x] 实现 Gate D train-only JL/PCA-768 oracle 与 paired-gain retention
- [x] 本地 uv 环境 11 项单元测试通过，Python compile 与 `git diff --check` 通过
- [x] 3090 完整模型 query 接线 smoke 通过；4090 execution 的 11 项测试通过
- [x] Gate B correct parity 通过：`59.8357 / 67.6018`，复现 exp336 s0 `59.9 / 67.6`
- [x] Gate B 五臂完成；五臂 global SHA 完全一致，descriptor 均为 `7×768=5376-D`
- [x] Gate D 单 seed 完成：train-only PCA-768 为 `59.9336 / 67.8733`，paired-gain retention=`1.1158`；固定 JL-768 失败

## Gate B / Gate D 单 seed 结果

| arm | mAP | R1 | 相对 global mAP | 解释边界 |
|---|---:|---:|---:|---|
| global | 58.9908 | 67.3756 | — | 同一 checkpoint 的共同 global |
| correct | 59.8357 | 67.6018 | +0.8449 | 原 exp336 scene-merged pose |
| canonical | 59.7374 | 67.6471 | +0.7465 | 固定 canonical 只比 correct 低 0.0984 |
| shuffled | 59.8037 | 67.7376 | +0.8129 | 异 PID 双射 donor pose 只比 correct 低 0.0320 |
| uniform | 59.3689 | 66.8326 | +0.3781 | 删除通道特异结构但保留 foreground support |
| no-pose | 59.4014 | 66.6063 | +0.4106 | 同一 pose-trained head 的推理干预，不等于 exp337 重训 |

五臂的共同 `global_sha256` 为：

```text
e5c3a041d6fe930c4c074ee3d7bdec1bea984503ff1c184f8f5cbf7ddfc0d310
```

单 seed packing：

| method | dim | mAP | R1 | retention | 判断 |
|---|---:|---:|---:|---:|---|
| full equal-concat | 5376 | 59.8357 | 67.6018 | 1.0000 | reference |
| fixed JL | 768 | 58.8011 | 67.5566 | -0.2245 | NO-GO |
| train-only PCA | 768 | 59.9336 | 67.8733 | 1.1158 | provisional GO |

PCA 只在 `train_loader_normal` 上拟合，train/eval path overlap=`0`；该结果只说明“线性 learned packing 可行”，不说明任意随机压缩可行。最终同维 claim 仍需三 seed paired 验证。

Gate B 的机制结论必须收紧：LGPA 的局部融合增益真实存在，但当前图精确姿态只解释很小部分；更可靠的资产是**结构化局部分解**，不是实例级精确姿态对齐。后续在 target-only / support-routing 门禁通过前，不把 `anatomical pose support` 当作已成立事实。

## 尚未执行

- [ ] Gate A：canonical CLIP/random 已闭合；待 correct-pose random-frozen/random-learned paired run
- [x] Gate B：exp336 checkpoint inference intervention 矩阵（s0）
- [ ] Gate C：same-image / correct cross-image /伪 support 的 identity-relation advantage oracle
- [x] Gate D：5376-D→768-D frozen oracle（s0 provisional；三 seed 待补）
- [ ] Phase 1：缓存特征 CASD 六臂 kill-switch

## 当前判断

**允许继续做廉价门禁，不允许直接开完整训练。**

内部 `exp120/123/125/129/130` 是 CASD 必须超越的强对照，不是外部 prior，也不自动否定论文创新。若 CASD 能用 strict LOO、part-structured support 与 support advantage 解决旧实验“teacher 有新增关系但 student 无法兑现”的失败，它们反而构成完整的机制动机。正式新颖性只由外部查新裁决。

下一执行顺序：先完成外部查新与内部前驱差分；随后把 Gate C 重写为 target-only、strict-path LOO、class-free、shared-mask、loss-matched 的 frozen support oracle，并直接加入 identity-only、slot permutation 与 exp123-style relational teacher。AERC 只作为正交备份，先做专项 ECOC/erasure-coding 查新与 frozen codec oracle。Gate A correct-pose learned query 已降为低优先级归因，不占用主线训练资源。

原因：UMTS 已证明普通 multi-shot teacher-student 不是新意。CASD 的新颖性依赖“pose-organized part support + leave-one-view-out + support-vs-self advantage”相对 same-image KD、full multi-shot KD 和伪 pose support 都有独立价值；这尚未被数据证明。先在缓存 teacher parts 上验证 support coverage、identity margin 与 controls，能避免再投入一次机制工作但身份指标不动的完整训练。

## 保护事项

- `experiments/decisions.md` 当前包含用户未提交的 #99/#100 改动；本阶段不修改、不暂存该文件。
- 不修改现有 tracked 模型/config，不启动 3090/4090 训练。
- 后续若使用 Python，必须先在工作目录通过 `uv` 建立环境。
- 禁止 Claude；正式训练前的审查改由 Codex 与可复现机制测试完成。
