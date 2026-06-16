# 实验 exp329: Burstiness 抑制 / democratic part-set 聚合（范式 import）

## 动机
- PRCV 已投后探索新创新点。强 SOTA 栈(exp255, 75.2)上一切 in-domain 训练端机制(backdoor 去混淆/TopoFR 拓扑/UCE 校准)全被压没 headroom(数据驱动证负)。FM-import 整条线证负(判别性-互补性张力)。
- 用户洞察: "SOLIDER 太强就换 TransReID" —— 弱 baseline(TransReID vit_base ~59, 离 SOTA 16 mAP)是有 headroom 的合法验证场。
- 夜间范式调研 agent(扫 gait/face/video/CC-ReID/vehicle CCF-B+)唯一过审强 bet = **burstiness suppression**, 搬自:
  - Jégou et al., On the Burstiness of Visual Elements, CVPR 2009
  - VLAD-BuFF: Burst-aware Fast Feature Aggregation for VPR, ECCV 2024 (arXiv 2409.19293)
  - On the Burstiness of Faces in Set, arXiv 2506.20312 (2025-06)

## 核心假设
遮挡 ReID 中, 少数**可见但过表达**的区域(大块近匀质躯干 patch、重复背景泄漏纹理)在很多 patch token 上重复出现; 检索对 token 求和/取 max → 这些 bursty token **抬高匹配分却不加身份信息**, 挤掉稀有可辨细节(包带/鞋)。按 self-similarity 反比降权(democratic 聚合)应提升遮挡检索, 且**与 visibility 加权正交**(可见 ≠ 不冗余)。

## 为何不是已关方向（红队自检）
- **非 visibility 小变体**: visibility 问"是否被遮挡"(二值/遮挡驱动); burstiness 问"这个**可见**特征是否被过度计数"(可见集内冗余统计)。一个 part 可完全可见且高 visibility 却 bursty。
- **非 feature completion**: 不重建/不补任何特征, 只对已有 token 重加权。
- **非 retrieval-side scorer 微变体**: 改的是**特征集本身的聚合/归一化**(VLAD-BuFF/GMP 形式化), train+test 一致, 不是排序侧 context 微调。
- **非 uncertainty 加权**: 无 per-feature 置信; 是集内 self-similarity 冗余测度。
- **Novelty 已核(agent + web)**: 无 occluded-ReID 做 burst-aware/democratic over part/patch。最近 cousin = Self-similarity guided probabilistic embedding matching (ESWA 2023) 用 self-sim **去噪/校验**(相反符号干预)。SDALF(CVPR'10) 是对称性非 burst 抑制。

## 阶段 1：0-GPU kill-switch（已搭好, e120 自动触发）
`scripts/burstiness_probe.py`（已 staged 到 hyy `/hy-tmp/transreid/scripts/`, smoke test 全 pipeline 跑通）。
- **复用 TransReID 自己的** make_dataloader / make_model / R1_mAP_eval → mAP 与训练日志直接可比。
- hook `model.base.norm` 取全 token 序列 (B, 129, 768)（JPM off 时 base(x) 只返回 cls, 需从 trunk 末 LayerNorm 取序列）。
- 每图: patch token L2 归一 → `w_i = 1/Σ_j sim(f_i,f_j)`(VLAD-BuFF 闭式, bursty token row_sum 大→w 小) → burst 加权 mean。
- 对比四种 descriptor 检索 mAP: `cls`(训练特征基线) / `uniform_patch` / `burst_patch` / `cls+burst`。
- **PRIMARY 判据 = burst_patch − uniform_patch**(隔离 burstiness 机制, 两者同用 patch token 仅加权差异); cls 仅作 context(patch-pool 能否抗衡训练特征)。
- **决策**: burst−uniform ≥ +1.0 → REAL, 升级阶段 2; < +0.3 → KILL; 0.3–1.0 → GREY(看诊断 + part-MaxSim 版本再定)。
- **诊断**: query(遮挡探针) vs gallery(整体) 的集内平均 self-similarity。occluded 更 bursty(正)→机制有理由存在。
- **smoke(pretrained)**: burst−uniform=+0.02→KILL(未训练无真 burst 结构, 证 metric 不虚高)。
- **真实数据**: e120 收敛弱 baseline 自动触发(monitor bc4m6btrv)。

## 阶段 2：全量方法（仅 kill-switch 通过才设计 + 双审 + 训练）
若 burst−uniform ≥ +1.0(或 GREY 但 part-MaxSim 版本显著)：
- 把 burst-aware 加权接入 **part-MaxSim 匹配**(不止 pooled descriptor): 匹配时对 gallery/query token 按 burst 权重折扣。
- 训练端: democratic 聚合作为 part 特征的可微聚合(替代/补充 mean/GeM)。
- 单变量对照: 同 backbone/config, 仅开关 burst 加权。先弱 baseline(TransReID)验 headroom, 再上 SOLIDER 强栈看是否噪声淹没。
- **该阶段才写完整 design + Claude review + Codex review + hook 双门**, 现在不预设(避免 kill-switch 没过就堆方法)。

## 预期结果
- kill-switch 通过(弱 baseline burst−uniform ≥+1.0 且 occluded 更 bursty): burstiness 是真机制, 全量方法预期弱 baseline +1~3 mAP。
- 失败最可能原因: ReID 训练已把 cls/global 学得对 bursty 区域鲁棒(part token 冗余但 cls 不受影响) → burst-pool 不及 cls, 机制在 pooled 层面无效。则降级到 part-MaxSim 层面再验一次, 仍无则 KILL。

## 对照组
- baseline = uniform_patch(同特征同流程, 无 burst 加权)。
- 二级对照 = cls(训练特征), 测 patch-pool 可用性。
- backbone: TransReID vit_base 弱 baseline(主, 有 headroom); SOLIDER exp255 强栈(次, 验是否被压没)。

---

## ⛔ 阶段1 KILL-SWITCH 判决 (2026-06-17, 弱 baseline TransReID e120=53.5)

| descriptor | mAP | R1 |
|---|---|---|
| cls(训练检索特征) | **53.53** | 60.59 |
| uniform_patch | 43.14 | 48.28 |
| burst_patch | 42.85 | 47.96 |
| cls+burst | 49.81 | 55.79 |
| partmaxsim_uniform | 42.80 | 47.47 |
| partmaxsim_burst | 42.56 | 47.24 |

- **PRIMARY pooled burst−uniform = −0.29 → KILL**；**part-MaxSim burst−uniform = −0.25 → KILL**（双判据一致，排除 pooled 误杀）。
- cls+burst(49.81) < cls(53.53)：加 burst 反伤训练特征(−3.73)。
- **前提 FAIL**：query(遮挡) intra-sim 0.689 < gallery(整体) 0.704 = −0.0154（遮挡更不 bursty）。前提 frozen DINO 成立(+0.0206)、训练后翻负。
- **阶段2(全量方法)取消**（kill-switch 未过）。
- **结论**：burstiness 在训练好的 ReID 模型上无 headroom，即便弱 baseline。ReID 训练已隐式吸收遮挡-burstiness 结构。归入"in-domain 特征机制 frozen-promising / trained-absorbed" pattern。
