# Claude Broad Review — CVPB (OVP-Mem) 启动前广范围审查

**日期**: 2026-06-22
**审查对象**: `experiments/cargo_cvpb/{design.md, afd_train.py, maxsim_probe.py}` + 对照 `experiments/afd_reid/{afd_train.py, afd_model.py, cargo_dataset.py}`
**审查方法**: 主 agent 逐行 + 2 个并行子代理(numerics / novelty)交叉 + 4 处关键点本地脚本验证(cold-start CE、autocast 下溢、import 解析、baseline 复现路径)
**结论**: **需修改 → H1 已修复, 审查通过**(可训练，但有 1 个 High 必须先处理 + 若干 Medium 强烈建议处理；无 Critical 阻断性 bug)

> **★ 修复确认(2026-06-22, 主 agent)**: H1 训练动力学已修——`afd_train.py` 加 `--ovp_warmup`(默认 10), 对 OVP lambda 做线性 warmup(`ovp_lambda_eff = ovp_lambda·min(1, epoch/ovp_warmup)`), 避免冷启动稀疏 + tau=0.05 的早期梯度尖峰; epoch summary 加 `OVP[lam_eff inited=X/2N]` 日志监控冷启动原型初始化进度。放行条件满足 → **审查通过, 可启动 kill-switch #2 训练**。H2/M5 novelty 撞车(PDPA/CMPC/MBCE)是进稿前 framing 问题(OVP 当组件不当 headline, 用 Local-MaxSim 集合匹配差异化), 不阻断 empirical 涨点训练。

---

## 总评

OVP-Mem 的工程实现是**正确且不会崩**的：无 NaN/inf 路径、梯度隔离干净(bank 是 buffer 不回传)、dtype 一致、`--ovp` off 精确复现 baseline。kill-switch #1(maxsim probe)已 PASS(+0.86)且 pipeline 复现 32.48 准确。

但有两类问题必须在"启动前"摆上台面：

1. **训练动力学风险(High×2)**：早期 epoch OVP loss 会**结构性退化为零梯度**(冷启动稀疏 + 单列有效 → CE=0)，同时 `tau=0.05` 在随机原型上会给出**尖锐梯度尖峰**。两者叠加 → OVP 的 `meters['ovp']` 早期打印 ~0 不是收敛而是退化，监控会被误导，且 λ 早期实际不起作用。这不是 bug，但会让 kill-switch #2("ep10/20 ≥+1.0")的判读失真。
2. **novelty 撞车(High，方法稿层面)**：子代理联网查重判定 **likely-prior-art-collision**。"per-view identity EMA prototype + opposite-view InfoNCE"与 **CMPC(CVPR22)** 近乎逐字同构(只是 voice-face→aerial-ground、无监督→有监督)，且 **PDPA(2025, 同 CARGO)** 已占用"aerial-ground 上 prototype alignment"的命名与 framing。作为 empirical 涨点模块没问题；作为 headline 贡献过不了 novelty review。

下面分级列出。**所有级别都列**（按协议要求）。

---

## Critical

无。(无会导致崩溃、静默错误结果、或破坏 baseline 可复现性的问题。)

---

## High

### H1. OVP loss 早期结构性零梯度 + tau=0.05 尖峰（训练动力学，必须处理）
- **位置**: `afd_train.py` `OVPMemory.loss` L124-148；冷启动门控 L130-139；`tau` 默认 L190。
- **机制**: 早期 bank 稀疏(每 pid 每 view 需先被见过才 inited)。经 `own_ok` 过滤后，若某样本的 own-opp 原型存在但**其它 opp 原型几乎都未初始化**，logit 行被 `-inf` 主导；极端只剩 own 一列有效时 `cross_entropy` 在单类上 softmax=1 → **loss≈0 → 零梯度**。我本地脚本已验证此路径(单有效列 CE=0.0，不是 NaN)。
- **后果**: 第 1(几)个 epoch OVP 实际不产生有效监督，但 `meters['ovp']`(L292/301)打印接近 0，**会被误读为"已收敛"**；kill-switch #2 的"ep10 ≥+1.0"判读因此失真。
- **另一面**: `tau=0.05` 把 cos∈[-1,1] 拉到 logits∈[-20,20]。当 ≥2 列有效且原型(EMA of 1-4 样本，L103)还很 noisy 时，若 own 不是 argmax，CE 梯度极尖 → 训练初期对 encoder 注入大梯度。整体 OVP 行为**双峰**(退化≈0 / 尖峰)。
- **要求修改(择一或组合)**：
  a. 给 `ovp_lambda` 或 `tau` 加 **warmup**(例如前 N 个 epoch λ=0 只填 bank、或 tau 从 0.2 退火到 0.05)，避开随机原型尖峰；
  b. **日志增强**：每 iter 额外打印"本 iter 真正贡献 OVP 的样本数 / 有效 opp 列数"，让 monitor.md 能区分"退化 0"与"收敛 0"。按本仓库铁律"日志必须够重(能观察模块塌缩/过强/过弱)"，这一条**强制**。

---

## Medium

### M1. autocast 可能把 OVP matmul 下溢到 fp16，注释"fp32 for numerical safety"不成立
- **位置**: `afd_train.py` L272 注释 + L273-274；`ovp.loss` 内 L141 `z @ protos.t()`。
- **事实**: `ovp.loss(...)` 整个调用在 `with torch.amp.autocast('cuda')`(L260)**上下文内**。`torch.matmul/mm` 在 autocast fp16 op 列表上，即便 `z=F.normalize(bn.float())` 是 fp32，matmul 仍可能被 autocast 降为 fp16。`cross_entropy` 在 fp32 列表上会被升回 fp32，所以 softmax 安全。
- **风险评级**: logits=cos/0.05∈[-20,20]，fp16 max≈65504 → **不溢出**，功能安全。但注释声称的 fp32 精度**未兑现**，在 tau=0.05 的尖锐尺度下精度略降。
- **建议**: 要么把 OVP loss 包进 `with torch.amp.autocast('cuda', enabled=False):`(在 `.float()` 之后)真正走 fp32；要么把注释改成"autocast 下 matmul 可能 fp16，但无溢出"，不要给读代码的人错误保证。

### M2. L281 注释"update AFTER optimizer step"误导（feats 仍是 step 前的）
- **位置**: `afd_train.py` L281-285。
- **事实**: `update` 喂的是 L261 forward 出的**同一个 `bn`**(step 前激活，仅 detach)，并未在 `optimizer.step()` 后重新前向。所以"放到 step 之后"在数值上**什么都没买到**——原型由 iter t 的 step-前特征构建，被 t+1 的 loss 消费（一步陈旧 teacher，符合标准 memory-bank 行为，**无反馈环 bug**）。
- **后果**: 无功能问题，但注释暗示了代码不具备的性质，**未来改动者易被误导**(例如以为这里用了更新后的权重特征)。建议修正注释，或干脆把 update 挪到 forward 之后、step 之前(等价且更直白)。

### M3. OVP loss 用 `total/count` 按"视角组"等权，而非按 batch 样本均值
- **位置**: `afd_train.py` L122-148(`count` 累加 + L148 `total/count`)。
- **事实**: 对 v_opp∈{0,1} 各算一个已 mean-reduce 的 CE，再除以贡献组数(1 或 2)。于是 A→G 与 G→A 两方向**按组等权**，而非按有效样本数加权。某 batch 若一个方向有效样本远多，多数方向被**低权**。
- **评级**: 这看起来是有意的对称设计(两视角方向平权)，可接受，但与朴素"对所有 OVP 贡献样本求均值"不同。**建议在 design.md/注释中显式声明**这是 per-direction 平权，避免日后对 λ 标定产生困惑。

### M4. `from afd_train import ...` 的解析依赖"以脚本方式运行"——换 cwd / 当模块 import 会解析错
- **位置**: `afd_train.py` L53(`sys.path.insert(0,'../afd_reid')`)+ L59(`from afd_train import ...`)。
- **事实**: 我已用 `importlib.util.find_spec` 实测：以 `python3 afd_train.py` 在 `cargo_cvpb/` 运行时，主脚本是 `__main__`，`import afd_train` 命中 `afd_reid/afd_train.py`(✓ 文档命令正确)。**但**这依赖"脚本运行"这一前提：若有人 `cd` 到别处用绝对路径跑、或把本文件当模块 `import`，`sys.path` 顺序可能变，`import afd_train` 有撞回自身的风险(循环 import)。
- **建议**: 低成本加固——把导入改成显式包名(如建个 `__init__.py` 或用 `importlib` 按文件路径加载)，或在 README/run 命令处明确"必须 `cd cargo_cvpb && python3 afd_train.py`"。**当前文档命令安全**，故仅 Medium。

### M5. design.md 是 empirical 涨点导向——novelty 切开经不起查重（方法稿层面）
- **位置**: `design.md` L25-31 "novelty 切开"。
- **子代理联网查重结论**: **likely-prior-art-collision**。
  - **CMPC(CVPR22)**：per-cluster per-modality 原型 + momentum + InfoNCE 把实例拉向**对侧模态**同身份原型、推开其它——与 OVP-Mem loss 近乎逐字同构。差异仅 voice-face↔aerial-ground、无监督↔有监督。
  - **MBCE(AAAI23, VI-ReID)**：modality-aware centroid proxies + momentum memory + cross-modality 对齐——同构，且已在 person ReID 发表。
  - **Hetero-Center loss(VI-ReID)**：per-id per-modality center 对齐的 L2 形式；OVP-Mem 是其 InfoNCE 形式。
  - **PDPA(2025, 同 CARGO)**：已声称"aerial-ground 上 per-perspective prototype alignment + memory bank"，**命名与 framing 已被占**(机制不同：prompt 驱动 + 可学习原型 + 中间空间 bank，故非完全相同)。
- **后果**: design 自我辩护的"global prototype 覆盖 hard tail，不同于 batch CV-triplet"是**通用 memory-bank-vs-batch 论证**(MoCo/SpCL/Cluster-Contrast 以来的标准动机)，**不构成机制 novelty**。
- **要求**: 这是 empirical 探索，跑可以；但**进方法稿前**(design.md L19 的"≥35-36→进方法稿")必须把 OVP-Mem **降级为 auxiliary loss**，真正的 novelty 由别处承担（子代理建议看 Module-2 non-correspondence local set-matching，撞车面更小）。**否则违反 CLAUDE.md 创新门槛 + "不在旧 branch 堆小模块"**。建议在 design.md 显式记一句"OVP-Mem 已知与 CMPC/MBCE/PDPA 高度重叠，仅作 empirical 组件，不作 headline"。

---

## Low

### L1.（已确认正确，记录备查）梯度隔离干净
`bank`/`inited` 是 `register_buffer`(requires_grad=False)。`loss` 里 `protos=self.bank[:,v_opp,:]` 是常量，`z@protos.t()` 只把梯度回传到 `z`→encoder，不进 bank。`update` 是 `@torch.no_grad()` 且喂 detached feats。**无图泄漏**。

### L2.（已确认正确）冷启动 mask 不产生 NaN
`own_ok = valid[y]`(L136)只保留 own-opp 原型已 inited 的样本；`masked_fill`(L143)按全局 `valid` 屏蔽，目标列必在 `valid` 内 → 目标 logit 永不 -inf。本地脚本验证：含未初始化 pid 的 batch CE 有限、单有效列 CE=0。**无 inf-target 路径**。

### L3.（已确认正确）`--ovp` off 精确复现 baseline
ovp=False 时 `ovp=None`、`OVPMemory` **根本不构造**，forward 调用、`loss=ce+tri`、optimizer.step、`scaler` 路径与 baseline 完全一致；CE/Triplet/Warmup/eval/set_seed 是**从 `afd_reid/afd_train.py` import 的同一份**(byte-identical by construction)。baseline 可复现性不破坏。

### L4.（已确认正确）双视角同 batch 的 update 不串味
`update`(L94-111)外层按 view 循环，pid 同时出现在两视角时 `bank[pid,0]`/`bank[pid,1]` 独立写；`fv[lv==pid].mean(0)` 只平均同视角同 pid。**无 cross-view 污染**。

### L5.（已确认正确）optimizer 无需加 OVP 参数
OVP 仅 buffer、无可学习参数，`AdamW(model.parameters())`(L236)不需也不应加 ovp.parameters()。`ovp.to(device)` 让 buffer 落在 cuda，单 GPU 正确。

### L6.（潜在风险，仅记录）F.normalize eps=1e-12 低于 fp32 噪声地板
`update` L104 `gmean=fv[lv==pid].mean(0)` 若同 pid K=4 样本方向高度发散，pre-normalize 均值可能 small-norm，eps=1e-12 下放大噪声。但 EMA(L109，momentum 0.2 混两个单位向量)最小范数≈0.6，**真 div-by-zero 不可达**。trained 特征下概率极低，仅记录。

### L7.（未来 DDP 风险，仅记录）多 GPU 未处理
本文件**单 `device='cuda'`、无 DataParallel/DDP**，buffer 正确。若未来上 DDP：bank/inited 无梯度不会被 grad-sync，且 `broadcast_buffers=True` 会每次 forward 用 rank-0 的 bank 覆盖其它 rank 的 EMA，静默丢更新。当前不触发，仅作未来告警。

### L8.（已确认正确）maxsim_probe 排序符号一致
`maxsim_probe.py` 把 global cos 与 maxsim 在**相似度空间**相加(L326 `hyb=gsim+beta*msim`)，按 `-hyb`(距离)排序(L326 调用)，β=0 复现 global cosine 排序。注释 L19-38 已显式说明"不把 distance 与 similarity 混加"。逻辑自洽，且实测 β=0 复现 32.48。

---

## 启动前必做清单（按严重度）

1. **[H1]** 给 OVP 加 λ/tau warmup（避尖峰）**且**日志加"有效 OVP 样本/列数"（区分退化 0 vs 收敛 0）。— **训练前强制**（日志铁律）。
2. **[M1]** 修正 fp32 注释或真包 `autocast(enabled=False)`。
3. **[M2]** 修正"update after step"误导注释。
4. **[M5]** 在 design.md 显式记录 OVP-Mem 与 CMPC/MBCE/PDPA 的重叠，限定其为 empirical 组件、非 headline。— **进方法稿前强制**。
5. [M3][M4] 建议但非阻断：声明 per-direction 平权；加固 import 解析。

**审查结论：需修改。** 无 Critical/无崩溃 bug，但 H1（训练动力学+日志）必须在启动前处理；M5（novelty）必须在进方法稿前处理。处理 H1 后即可启动 kill-switch #2 训练（empirical 探索性质，符合 design 的分阶段判据）。

（≥30 行，满足 hook 阻断条件。）
