# 过夜自主创新探索日志（2026-06-15 夜 → 06-16 晨）

> 用户睡前指令：整夜不停探索，三台服务器全用上，**务必找出一个有用的创新点**。我全程自主决策、记录、止损、escalate 赢家。
> 战略：搬外部范式（CLIP-ReID/Pose2ID 打法），冻结 kill-switch 先验 → 赢家训头 → 破天花板。已死：MLLM-reasoner（姿态提示无效）。已证伪：exp109 内部机制（三堵墙）。

## 已确认 baseline
- exp255 Swin 主线（训练端最强）：Occ-Duke **75.2 mAP** MaxSim。KPR 75.1。SOTA 线。
- exp324 frozen DINOv2-base + 姿态部位匹配：重遮挡 1.86 / 全部 3.21 mAP（机制干净：姿态锚定 ×3.4 vs 整图，均匀网格不涨）。
- **exp324b 训轻量头（frozen DINO，433K 参数）e60**：part 全部 **14.61** / part 重遮挡 **8.65** / cos 全部 13.51 / cos 重遮挡 7.32。
  - 判断：机制确认 + 可训（×4.6），part-MaxSim > 整图（姿态机制加分）。**但冻结特征天花板低（14 vs 75）**，e20 即到顶。→ 需破天花板。

## 今夜实验矩阵（4 GPU slot 全占）

| Exp | 机器 | 假设 | 状态 |
|-----|------|------|------|
| exp324d | lab-3090-d | **LoRA 解冻 DINO-base** + 姿态部位匹配 → 破 14 天花板？（DINO 线能否competitive 的决定性实验）| 🔄 |
| exp324f | lab-4090 | **DINO 部位对应 ⊕ exp255 Swin 融合**：给 75-mAP SOTA 模型加 DINO 遮挡鲁棒性，重遮挡能否 > 75 单独？（最有戏的重量级角度，建在 75 上而非 14）| 🔄 |
| exp325 | hyy GPU0 | **DINOv2-large** frozen 头：更强冻结模型天花板能否抬过 14？| 🔄 |
| exp326 | hyy GPU1 | **DINOv3 / DIFT** frozen 部位对应：更新/更强对应特征天花板 | 🔄 |
| (planner) | — | Workflow 生成更多 paradigm-import 探针队列，供后续 cycle | 🔄 |

## 运行记录
（每个实验完成后追加结果 + 判断 + 下一步）

### exp326 (DIFT / SD 特征) + exp327 (DINOv3 系) — hyy 双卡（2026-06-16 晨）

> 任务：在 hyy 上探"更强/不同的对应特征源"给姿态部位 occluded-ReID。exp324 已证 frozen DINOv2-base 给干净信号但天花板低（重遮挡 1.86）。决定性问题：**训练-free 能否超 1.86**。

**Setup 卡点与解法**：
- hyy 无 pose_data npz，且无法 resolve lab-3090-d SSH 别名（ProxyJump）→ 经本地 Mac 中转。
- 原 npz 2.7G（含 17×64×48 heatmap），2MB/s ProxyJump 传输需数小时 = 卡点。**解法**：在 lab-3090-d 上把 npz 剥成 slim（只留 keypoints+visibility_binary，exp324 实际只用这两个字段），2.7G→113M，18s 生成，几分钟中转完成。hyy 上 dinov2-b 复现 exp324 smoke 数字逐位一致 + heavy-occ 989/2210 与 exp324 完全相同 → **slim 剥离无损，pipeline apples-to-apples**。
- hyy 无 transformers/diffusers → pip 装（cache 重定向 /hy-tmp 避免撑爆 28G overlay）。/hy-tmp 仅 28G free → 特征改 in-RAM（503G 主存），不写盘。
- **DINOv3 gated**：hf-mirror 也下不了（需 license/token），exp327 默认改用 `dinov2-with-registers-base`（ungated，registers 干净，apples-to-apples DINO 升级）。

**smoke 信号（小 gallery，仅趋势）**：
- DIFT (up_block1, t100, e2) 30q×500g：pose-part heavy **9.92** > grid 6.54 > holistic 5.06，pose vs grid +3.38 → 姿态锚定有效，**DIFT 趋势强于 DINOv2**。
- dinov2reg-b 50q×2000g：pose-part heavy 2.55（复现 exp324 smoke 数字）。

**FULL 进行中**（GPU0 DIFT e4 ~80min；GPU1 reg ~5min）。结果见各 monitor.md，决定性数字 = full heavy pose-part mAP vs **1.86**。


### [夜间记录]
- **exp324d 第1次 agent 失败**：agent 跑偏（最终消息答非所问、未建脚本/未启动训练），3090 空转。重新用**严格规定的设计**(固定部位分配矩阵 M + LoRA DINO + bmm 池化)再派，避免它在"可微池化"上迷路。
- **exp324d 第2次（本次）已启动**（2026-06-16，lab-3090-d）：`scripts/exp324d_lora.py` 写好——LoRA(q/v rank8 alpha16) 解冻 DINOv2-base + **可微姿态部位池化**(预算每图 row-stochastic 池化矩阵 pool_w，`bmm(pool_w,patch)` 梯度回流 LoRA) + exp324b 同款头/损失/PK64/part-MaxSim eval。**唯一变量=LoRA 解冻 DINO**。双审查 approve（Claude 含 200-trial 池化等价 1.79e-7；Codex 141k tok 确认梯度到 LoRA/full-batch triplet/use_reentrant=False/eval 对称，组合 plausibly new 未找直接先例）。Dry-run：LoRA 294,912+head 413,184 训练，loss↓ acc 0.016→0.766，显存 13.7G/24G，~1.4s/step。全量 35ep 后台跑（micro_bs64 no-ckpt）。**待结果：part-MaxSim 重遮挡 mAP 能否破 exp324b 的 8.65 / 全部破 14.61 往 competitive 走。**【结果待填】

### [planner queue 已生成] — 下一批探针（按 有用×便宜×新颖 排序）
**最关键洞察：先做 0-GPU 的"rank-disagreement oracle"（#1），它是整条"DINO 补 Swin"家族的天花板+kill-switch + 论文 motivation 图。**

1. **Rank-disagreement oracle（0-GPU，~40行 numpy，⭐立即做，依赖 Swin distmat）**：重遮挡子集上比 Swin top-k vs DINO part-MaxSim top-k 的 Jaccard / P_dino_only（DINO 命中 Swin 漏的真值）/ per-query oracle 上界。判定：P_dino_only<2% 或 oracle<+1mAP → 整条 DINO⊕Swin 当场止损；oracle +3~5 且 Jaccard<0.5 → 正交坐实，进 #2。
2. **遮挡门控 part-MaxSim re-rank（training-free，最可能产出有用主表结果）**：DINO 共可见 distmat 当 `utils/reranking.py re_ranking(local_distmat=)` 输入，只对重遮挡 query 触发。依赖 #1。
3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
4. **共可见覆盖度当 comparability 否决器**：降权 Swin 不可比高分假阳（DINO 当否决器非打分器，绕开判别弱）。
5. **可微 partial-OT 当 metric loss 训 exp324b 头**（根治 train/test mismatch）。依赖 #3。
6. RADIO（DINO+CLIP+SAM 蒸馏）换 frozen 源。
7. SAM2 mask 定义池化域（换部位区域不换特征）。
8. ROA 遮挡配对做免费逐部位可见性 GT 监督。

**执行计划**：exp324f agent 正在 lab-4090 算 Swin distmat → 它一落地我立刻跑 #1 oracle（0-GPU）。正向 → #2 re-rank（training-free 主表素材）。OT 线(#3)等某 GPU 空了上。

### exp324f 落地 + exp325 启动（2026-06-16，lab-3090-d；环境逼回 3090）

**环境现实**：lab-4090 venv 缺 cv2/mmengine/transformers，跑不了 SOLIDER swin 也跑不了 DINO；lab-3090-d 也无单一 python 同时具 mmengine（swin）+ transformers（DINO）。故 exp324f 落 lab-3090-d 并拆两阶段（npz 桥接：`solider-reid` conda env 出 Swin distmat → 系统 python3 出 DINO distmat + 融合）。双卡均空（3090/4090 idle）但 dep 把活全锁在 lab-3090-d。

**exp324f 结果（NEGATIVE，clean，已写 monitor.md）**：
- Swin MaxSim ALONE **75.16/85.57**（=主线 75.2 ✓）；DINO part-MaxSim ALONE **14.61/21.99**（=exp324b e60 ✓）；对齐 sanity 全过（文件名 join，pid 全等，camid 偏移恒=1，w=0=75.16）。
- 重遮挡子集 989/2210（vis≤8），w=0 = 72.57 mAP。
- 融合 `d=(1-w)·swin+w·dino`（z-score & min-max 两种归一化结论一致）：**HEAVY 从 w=0.1 起就单调变差**（-0.14 → -6.10 @ w=0.5），ALL 同向下降。
- **判断**：exp255 Swin 重遮挡本身已 72.57（PSG/LGPA/GCN 处理遮挡），冻结 DINO 距离（14 量级）噪声太大、对 SOTA **严格冗余且有害**。**score-level late fusion 路线证伪**。
- 这正是 planner #1 oracle 要量化的"P_dino_only / oracle 上界"的 fusion 侧旁证：简单加权死路 → 下一步直接做 **#1 rank-disagreement oracle** 量化是否存在任何正交信号（DINO 捞到 Swin 漏的真值），它是整条"DINO⊕Swin"家族的天花板 + kill-switch。

**exp325 启动（DINOv2-large 天花板探测，lab-3090-d，3090 与他人 exp324d 等共享卡 ~8G 已占）**：
- `scripts/exp325_train_head.py`：monkeypatch exp324_dino → 冻结 backbone base→**large**(hidden 1024, patch14, 304M)，其余（头/损失/PK64/part-MaxSim eval/超参/seed）= exp324b 完全一致，**唯一变量=backbone 容量**。
- 双审查 approve（Claude broad + Codex `--search`，均仅 Low 非阻断；Codex 确认 monkeypatch/几何/单变量正确，新颖性窄框为诊断）。hf-mirror 下 large 成功（304M），盘 676G free。
- 60 epoch，先抽 train/query/gallery large dense token 部位特征缓存（独立 `experiments/exp325/_cache`）再训。**待结果：part-MaxSim 重遮挡 > 8.65 / 全部 > 14.61？**【结果待填】

### [夜间记录 2]
- **lab-3090-d**: exp324d LoRA 训练已上 GPU(100% util)，进行中。
- **hyy**: GPU0 DIFT 跑(72%)，GPU1 DINOv3 低 util(可能阶段间/慢)。
- **lab-4090 卡死**: 反复 banner 超时 + torch env 找不到(空 conda、标准路径无 torch python)，exp324f agent 在它上面没产出任何东西、4090 空转。**判定 lab-4090 是时间黑洞，今夜不强求**，价值集中到 lab-3090-d + hyy。Swin distmat 改在 lab-3090-d 算(prep_exp323 已算过一次可复用)。
- **下一步(最高价值)**: 在 lab-3090-d 跑 planner#1 rank-disagreement oracle(Swin vs DINO 正交性 + oracle 上界)，gate 整条 DINO⊕Swin 家族。

### [夜间记录 3 — 更正]
- **exp324d LoRA 实际在健康运行**（我先前误判：grep 错了 log 路径 /tmp/exp324d.log，实际是 /tmp/exp324d_r16.log）。第一个 agent 其实完成了全套：写了可微部位池化脚本(bmm pooling matrix，与 exp324 数值等价 1.8e-6)、**过了 Claude+Codex 双审查(都 approve，确认梯度到 LoRA、无遮挡 ReID 先例)**、launch 了 rank16/alpha16/dropout0.05/grad-ckpt/30ep 训练(PID 309591)。两个并发 agent 撞了一下但收敛到一个健康 run。DINO-in-loop 慢(分钟级/epoch)，epoch5 首评出关键数：part 重遮挡能否破 8.65、全部破 14.61。
- **oracle(exp324g)** agent 启动中，与 LoRA 共享 3090 GPU 算 Swin distmat(慢但一次性)。
- 状态：lab-3090-d(LoRA+oracle) + hyy(DIFT+DINOv3) 在产出；lab-4090 弃用。等 LoRA epoch5 / oracle / DIFT 结果。

### [exp324g — planner#1 rank-disagreement oracle 完成 → DINO⊕Swin 家族止损]
**0-训练诊断（lab-3090-d，系统 python3，无 DINO 前向，与 exp324d LoRA 共享 GPU 无冲突）。复用 exp324f 全套管线 + 已 dump 的 Swin distmat（mAP=75.16）。**

重遮挡子集（pose vis≤8，**989/2210** query，全有效）三个量（top-10）：
- (a) **top-10 Jaccard = 0.062**（Swin vs DINO，排 same-cam 后）
- (b) **P_dino_only = 0.20%**（仅 2/989：DINO r1 命中真值且 Swin r1 没命中；反向 Swin-only=719/989=72.7%）
- (c) **oracle(max per-query) heavy mAP = 72.69 vs Swin-only 72.57 → +0.12 mAP**；DINO-only heavy mAP 仅 **8.65**

**判定：STOP-LOSS — DINO⊕Swin 无独立信息，整条"DINO 补 Swin"家族止损。**（P_dino_only 0.20%<2% 且 oracle gain +0.12<+1mAP，两条 kill 条件双中。）

机理：Jaccard 0.06 是**虚假正交**——DINO part-MaxSim 整体太弱（heavy 8.65 vs Swin 72.57，差 64 mAP），top-10 基本是噪声，自然不与 Swin 重叠，但不重叠 ≠ 命中。决定性证据是 oracle 上界：连"每 query 取 Swin/DINO 更优 AP"都只 +0.12，说明 **Swin 失败的 query 上 DINO 几乎也失败**。exp324f late-fusion 涨不动不是融合方式问题，而是**信息上限不存在**。

**对 planner 队列的影响**：
- #2（遮挡门控 part-MaxSim re-rank）**依赖 #1 正向 → 现已被砍**，DINO 的共可见 distmat 当 re-rank 输入不会有增益，不做。
- #3/#5 OT 线同理：把 part-MaxSim 平均换 Sinkhorn 只改聚合方式，无法凭空造出 DINO 没有的判别信息，**优先级降到底**（除非先有证据 DINO part 判别力可被大幅拉高）。
- 唯一可能翻盘的：**exp324d LoRA** 把 DINO part 判别力从 8.65 大幅拉到接近 Swin。LoRA 不破这个数，整条 frozen-DINO 源（含 #6 RADIO/#7 SAM2/#8 ROA 监督）都失去前提。**等 exp324d epoch5 首评定生死。**
- 探索重心从"frozen-DINO 补 Swin"转出：若 LoRA 也救不动，下一轮 gap analysis 不再绕 DINO 部位特征，改找 Swin 自身可比性/可靠性维度（planner #4 共可见覆盖度否决器 仍可在 Swin 自身 part 上做，不依赖 DINO）。

### [关键结果 — exp324g oracle: DINO⊕Swin 家族止损]
重遮挡子集(989 query)：top-10 Jaccard **0.062** · **P_dino_only 0.20%**(2/989) · Swin-only **72.57** · DINO-only **8.65** · **oracle 上界 72.69(+0.12)**。两个 kill 条件全中。
**机制结论**：低 Jaccard 是"假正交"——DINO part-MaxSim 不是"补充"，是"全局太弱"(8.65 vs 72.57)，top-10 大多噪声、天然不重叠 Swin 但≠正确。Swin 错的地方 DINO 也错。**信息天花板根本不存在**——不是融合配方问题。
**砍掉**：planner #2(re-rank)/#3#5(OT)/#4(veto) 全依赖"DINO 补 Swin"→ 全死。frozen-source 换源(#6 RADIO/#7 SAM2/DIFT/DINOv3)价值大降(同一堵墙：frozen 弱、对 SOTA 无独立信息)。
**DINO 线唯一活口 = exp324d LoRA**(让 DINO 判别力从 8.65 升向 Swin 级)。e5 首评 make-or-break，~15-20min。
**这本身是有用的 analysis 结果**：connect 上 MLLM 遮挡缺口(GPT-5.5 56%)+ exp324 机制 → 一个"基础模型对 occluded ReID 能/不能做什么"的诚实分析故事（fallback useful output）。
**若 LoRA 也低**：DINO paradigm-import 判为负，剩余夜转生成式遮挡增强(攻 95.8% 墙)或 test-time adaptation，或收敛到 analysis 报告。

### [夜间记录 4 — DINO⊕Swin 双确认死 + 战略判断]
- **exp324f agent 独立复现 oracle**：fusion 从 w=0.1 起单调伤害(HEAVY −0.14→−6.10)；oracle +0.12、P_swin_rescue 72.7% vs P_dino_rescue 0.20%。**两个 agent 一致 → DINO⊕Swin 彻底死、settled。**
- **lab-4090 env 确认坏**(无 cv2/mmengine/transformers)，且连 lab-3090-d 都没有单一 python 同时有 mmengine(Swin)+transformers(DINO) → 跨模型实验必须 npz 桥接。
- **exp325(DINOv2-large frozen)** 在 lab-3090-d 跑(过双审查)，与 LoRA 共享 GPU(extraction ~45min 期间有 contention)。post-oracle 价值低(更大 frozen 源仍对 SOTA 无独立信息)。

**战略诚实判断**：DINO paradigm-import 大概率不是"genuinely useful innovation"——frozen 死(oracle)，LoRA 即便 competitive 也大概率 me-too(DINO-as-backbone，PersonViT territory)。**最可靠的 useful 产出正收敛到：(a) analysis 故事**(MLLM 遮挡缺口 56% + DINO oracle 无独立信息 → "基础模型对 occluded ReID 能/不能做什么"的诚实负面/诊断贡献，证据扎实)**；(b) 若 LoRA 意外强 → 一个 competitive-FM-adaptation 方法**。
**剩余夜计划**：等 LoRA e5(决定性，~30-40min)；low-value 探针(exp325/DIFT/DINOv3)让它们跑完不投入；LoRA 一出结果就 re-plan(强→escalate；弱→转 Swin 训练目标创新 或 收敛 analysis)。

### [exp326/327 FULL 实证 — 换 frozen 源确实不破天花板，坐实上面 oracle 战略判断]

hyy 双卡训练-free 全量结果（决定性数字 = heavy pose-part mAP vs exp324 DINOv2-base **1.86**）：

| 特征源 | heavy pose-part mAP/R1 | vs 1.86 | 判定 |
|--------|------------------------|---------|------|
| exp324 DINOv2-base | 1.86 / 3.54 | — | baseline |
| **exp327 DINOv2-with-registers** | **2.15 / 3.84** | **+0.29** | 小幅正向，没破天花板，止损 |
| **exp326 DIFT（SD-v1.5 up1 t100 e4）** | **0.73 / 1.42** | **−1.13** | **决定性负，SD 线止损** |

- **DIFT 决定性负**：smoke（500 gallery）heavy 9.92 排第一是**误导**，full（17661 gallery）塌到 **0.73**。DINO smoke 2.55→full 1.86 小降，DIFT 9.92→0.73 灾难塌 → **SD 特征 category 语义对应强（PCK 高）但 instance 身份判别弱**（SD-DINO / Tale-of-Two-Features 文献一致：SD 与 DINO 互补、SD 不主导 instance retrieval）。**不上 exp326b 头**。
- **registers +0.29 小幅**：更干净 dense 特征（去 high-norm artifact token）蹭一点，远不够独立可用。DINOv3-b gated（hf-mirror 需 token）下不了，用 registers 代验。
- **直接坐实 oracle agent 的判断**：「frozen-source 换源(RADIO/SAM2/DIFT/DINOv3)价值大降」现在有**全量实证**——换更新 DINO（registers）只 +0.29，换 SD 范式（DIFT）反而 −1.13。**换源都不破 frozen 天花板，瓶颈在 frozen 本身**。DINO 线唯一活口仍是 exp324d LoRA（解冻），与 oracle 结论收敛。
- **方法论铁律（新，已写 decisions.md）**：training-free probe 必须用**全量 gallery** 判绝对值；小 gallery smoke 只验流程不验数值——DIFT 是活教材（smoke 第一、full 垫底）。
- setup：原 pose npz 2.7G（含 17×64×48 heatmap）跨 ProxyJump 几小时 → 在 lab-3090-d 剥 slim（仅 kp+vis_bin，exp324 实际只用这俩）2.7G→113M 几分钟到位；hyy dinov2-b smoke 逐位复现 exp324 + heavy-occ 989/2210 一致 → pipeline 无损。**exp326/327 均不 commit**（按指令）。

### [夜间记录 5 — 换源全死，FM 只剩 LoRA]
- **hyy 探针(full Occ-Duke)**：DINOv2-registers **2.15**(+0.29，没破)、**DIFT(SD-v1.5) 0.73(−1.13，更差)**。SD 特征强于 category 对应但弱于 instance ID 判别(SD-DINO 文献一致)。**结论：天花板瓶颈是"冻结"本身，不是模型新旧** → 独立佐证 oracle。换源(RADIO/SAM2/DIFT/DINOv3)全砍。
- **FM 方向只剩 1 个活口：LoRA 解冻能否让特征判别化。** base-rank16 在 lab-3090-d 跑(e5 pending)。hyy 2 卡空 → 加跑 DINOv2-large+LoRA(GPU0，最强变体) + base-rank32(GPU1)，并行拿"FM-adaptation 到底行不行"的彻底答案。
- 诚实预期：即便 LoRA 成，大概率收敛到 Swin-like(me-too)。但拿到 firm 结论(行/不行+证据)本身对 analysis 故事有用。

### [exp324d LoRA 解冻 — hyy 双卡启动，FM 唯一活口的彻底测试]

环境全部就绪并 dry-run 验证后，hyy 两卡并行启动 LoRA-解冻 DINO：

| 变体 | GPU | backbone | LoRA | micro_bs | peak GPU | ~速度 | 输出 |
|------|-----|----------|------|----------|----------|-------|------|
| **exp324d_large** | GPU0 | DINOv2-large (hidden 1024) | rank16/α16 | 16 | 5.34G | ~7.5s/step (~30min/ep, ~15h) | log/.../exp324d_large |
| **exp324d_r32** | GPU1 | DINOv2-base (hidden 768) | rank32/α32 | 32 | 2.81G | ~2.6s/step (~10.5min/ep, ~5.5h) | log/.../exp324d_r32 |

- **复用 lab-3090-d 的 `scripts/exp324d_lora.py`**（dual-reviewed，可微 bmm pool matrix + LoRA + 同 exp324b 损失/采样/eval）。最小改动加 `--dino_model` + 动态 hidden（base.config.hidden_size，large=1024）+ ROOT 读 EXP324_ROOT 环境变量。**单变量隔离**：large 变 backbone 容量、r32 变 LoRA rank 容量，其余全同 base-rank16。
- **环境**：peft 0.19.1 装好；pool 缓存（geometry-only，base/large 通用）从 lab-3090-d rsync 复用（免传 2.1G train pose、免重算）；图像 symlink 进 repo data 路径；train pose 缺但走 pool 缓存不调 find_pose。
- **dry-run 双验证通过**：hidden 正确(1024/768)、LoRA params 随 rank 缩放(large 1.57M / base-r32 1.18M)、loss 下降、梯度到 LoRA、peak GPU 远低于 16G。
- **启动确认**：两进程 ALIVE，加载 q/g pool 缓存 OK，heavy-occ 989/2210 一致，进入 epoch 1（GPU0 100% / GPU1 75% util）。
- **核心问题**：large 或 rank32 能否把重遮挡 mAP 明显推过冻结 8.65 往 competitive（几十分）走？等 e5 第一次 eval（决定性）。

## ⭐⭐ [突破 — exp324d LoRA e5：冻结天花板被打穿 ×4] ⭐⭐
**base-LoRA rank16，epoch 5/30**：part 重遮挡 mAP **34.92**(vs 冻结 8.65，**×4.0 +26.3**) · part 全部 **42.45**(vs 14.61) · cos 重遮挡 32.38 · cos 全部 41.06。**才 e5/30，还在陡升。**
- **FM-adaptation 确认有效**：~1M 可训参数(0.6M LoRA + 0.4M 头，DINO 其余冻结)就把"冻结特征无用"逆转。oracle 说冻结是根本瓶颈——LoRA 证明瓶颈是 adaptation 不是特征本身。
- **完整故事成型（有用创新候选）**："冻结基础模型特征对 occluded ReID 无独立信息(oracle +0.12)，但极小参数 pose-anchored adaptation 解锁它们 ×4 往 competitive 走"。method(参数高效 FM-adaptation + pose-part-MaxSim，Codex 称无直接先例) + analysis(冻结无用→adaptation 解锁) 双线。
- **escalate**：base-r16 跑到 e30(看最终能否逼近 Swin 72.57 heavy)；hyy rank32(更多容量) + large(更大模型，慢但可能最高) 测这套的天花板。trajectory 关键。
- **诚实**：e5=34.92 仍 < Swin 72.57，需看 e30 能到多少。即便略低，"~1M 参数达 competitive occluded ReID on frozen DINO"也是有用的 efficiency/method 贡献。

### [LoRA escalation — e5 跨配置对比]
| 配置 | part 重遮挡 e5 | part 全部 e5 | vs 冻结 8.65 |
|---|---|---|---|
| base r16 (lab-3090-d) | 34.92 | 42.45 | ×4.0 |
| base r32 (hyy) | **36.72** | **44.54** | ×4.2 |
| large r16 (hyy) | pending(~30min/ep 太慢) | — | — |
都 e5/30 还在升。rank32 略高(+1.8 heavy) → 更多 adaptation 容量微帮。winner 跨 rank 稳健。**关键未决：e30 能否逼近 Swin 72.57 heavy / 75 all。** 等 e10/e15/e30 看 plateau。

### [★ exp324d_r32 e5 决定性突破 — LoRA 解冻彻底破冻结天花板，FM 方向活了 ★]

**核心问题（FM-adaptation 能否破冻结 8.65 天花板）回答：能，且大幅。**

exp324d_r32（DINOv2-base + LoRA rank32，hyy GPU1）e5 part-MaxSim：

| 片 | LoRA r32 e5 | 冻结 baseline (exp324b) | 增益 |
|----|-------------|--------------------------|------|
| **part HEAVY mAP/R1** | **36.72 / 49.44** | **8.65 / —** | **+28.07 (4.2×)** |
| **part ALL mAP/R1** | **44.54 / 57.47** | **14.61 / —** | **+29.93 (3.0×)** |
| cos HEAVY mAP | 34.52 | ~7.3 (冻结 cos) | 大幅 |

- 训练端极快收敛：acc e1 0.41 → e2 0.84 → e3 0.95 → e5 0.985；d_ap/d_an margin e1 ~0 → e5 ~12。
- **这推翻了 oracle/换源探针的"FM 对 occluded ReID 无独立信息"悲观结论的一半**：那些结论是**冻结**条件下成立（换更新 DINO +0.29、SD/DIFT −1.13、oracle 无独立信息）。**一旦允许 LoRA 解冻，瓶颈消失** → 证实瓶颈是"冻结"本身，不是 DINO 表征结构、不是模型新旧。
- **仅 e5/30，仍在涨**。base-rank16（lab-3090-d）与 large-rank16（hyy GPU0，e1 acc 0.373 同轨迹，e5 约 2.5h 后）是 capacity 对照。part 略优于 cos → pose-anchor 部位匹配在解冻后仍有边际增益（说明部位机制不是冗余）。
- **战略转向**：FM 方向从"唯一活口/大概率 me-too"升级为"有真实正结果"。但需诚实评估**新颖性**：DINOv2+LoRA+ReID 是否 me-too（PersonViT / DINO-as-backbone territory）？competitive 到什么程度（36.72 heavy vs Swin 72.57——仍有大 gap，但这是 pose-part-MaxSim 单分支、无 PSG/GCN/全套）？下一步：等 e30 看上限 + 对照 rank16/large 看 capacity 曲线，再判断是"competitive-FM-adaptation 方法"还是"诊断性结论的正向补充"。

### [LoRA e10 — plateau，诚实判断]
base-r16: e5 34.92 → **e10 36.78** heavy（+1.86，train acc 0.997 已饱和）。base/rank32 都 **plateau ~37 heavy / ~45 all**。
- **×4 突破真实**（8.65→37），但单 pose-part 分支 plateau ~37，**远低于 Swin 全栈 72.57**。
- **判断**：这是强 **analysis/诊断贡献**（"基础模型对 occluded ReID 能/不能做什么"：frozen 无用→adaptation 解锁 ×4→但通用 FM adaptation 单分支仍远不及 purpose-built SOTA），**不是 beat-SOTA 方法**。
- **最后的方法希望 = DINOv2-large**（更大 backbone，frozen 特征更富，LoRA 可能到更高，~2.5h 出 e5）。若 large 也 plateau ~40 → 确认 method 路不通，收敛 analysis。
- **诚实**：不会编造"competitive method"。useful 产出 = 这套完整诚实的 FM-for-occluded-ReID 分析（MLLM 56% + oracle +0.12 + frozen 全弱 + LoRA ×4 plateau 37 + 机制 part>cos）。这本身对领域有用。
- 让 base/r32 跑完 e30 确认 plateau；large 是关键未决。

### [exp324d 新颖性核查 — 联网研究 agent，结论：正结果真实但 me-too，需 reframe]

并行跑实验时启动联网研究 agent 查 "DINOv2+LoRA+pose-part-MaxSim for occluded ReID" 的先例。grounded 结论（带 paper+数字）：

- **DINOv2/SSL ViT 作 ReID backbone：饱和**。PersonViT (2024, DINO+MIM ViT) **Occluded-Duke 72.2 mAP / 79.8 R1**；SOLIDER（本仓库 baseline）；CLIP-ReID；DINOv2 已被用于 object ReID (2508.21222)。**没找到"DINOv2 checkpoint + LoRA + Occluded-Duke"确切论文**——这是唯一白点，但是 combinational gap 不是 mechanism gap。
- **LoRA-for-ReID 已有**：diffusion-ReID (2502.06619) 用 LoRA adapt Q/K/V/O+FFN。而且"frozen DINOv2 缺判别性、需 LoRA 中间地带"这个发现**文献已近乎明说** → 我们的 8.65→37 **印证已知现象**，不是新发现。
- **pose-part + 只匹配互见部位：最成熟的轴**。PVPM (CVPR'20) 已有 visibility predictor + 只匹配可见部位；KPR (ECCV'24, SOLIDER/Swin) 是当前最强同款。我们的差异仅 **MaxSim**（ColBERT late-interaction 借来）替代固定 part-to-part 对齐——mechanism 小 delta，不是新问题定义。
- **绝对数字硬伤**：Occluded-Duke 标准报 **all-query**，SOTA ~60-72 mAP。我们 **~47 all-query 远低于 SOTA**（TransReID 时代 −12，PersonViT 72.2 之下）。heavy-occ 子集非标准 split，reviewer 会读成 cherry-pick。→ "novel combination that loses to SOTA = reject"。

**新颖性裁决：me-too 重组（偏"中间地带"仅当 reframe）**。要上强会（CVPR/ICCV/ECCV/AAAI）至少需其一：
1. **真新机制**：LoRA↔visibility **交互**（pose/visibility-conditioned LoRA，或 per-part low-rank experts 按遮挡 gate），带消融证明——不是"加了 LoRA 加了 parts"。
2. **打平/超 SOTA 标准 all-query**（≥62-72 mAP）+ Occluded-ReID/PoseTrack 佐证。现 ~47 单凭这点就 disqualify。
3. **新问题框定**：CLAUDE.md 已列对方向——**common-visible support / pair comparability / reliability-aware matching**。把"互见部位 MaxSim"形式化成新匹配目标（理论+消融），不是当 scoring trick。
4. **FM-specific 洞察**：为何 DINOv2 part-correspondence 在 LoRA 下涌现、full-FT 下塌（correspondence/attention 分析量化）——把 8.65→37 从"一个数字"变成"被研究的现象"。

**对策**：(a) 先等 e30 看 r32/large 绝对上限能爬多高（若 all-query 上 60 接近 SOTA，路线 2 有戏；若卡 ~50 则路线 2 死）；(b) 真正有论文价值的是**机制重组(1)或问题 reframe(3)**——与 CLAUDE.md「值得推进方向」（common-visible support / reliability）完全对齐。**正结果是 building block，不是终点**。

### [exp324h — adapted-DINO 是否对 SOTA Swin 有独立信息：ORACLE 探针，结论：变判别≠变互补，止损]

承接 exp324g（**冻结** DINO 对 75-mAP Swin 无独立信息：oracle +0.12、P_only 0.20%、Jaccard 0.062）。
新角度：**DINO 经 LoRA 已判别化**（heavy 8.65→36.78），不同 backbone→不同错误模式，**可能反而互补 Swin**——
冻结时不互补只是因为太弱。这是没测过的方法角度，便宜高价值。eval-only/无训练/无 commit，与运行中的 exp324d 并行用 e10 ckpt。

**做法**：复用 exp324d 的 build_lora_dino/encode_split + 缓存 pooling，加载 e10 LoRA(lora_10)+head_10 重 encode
query/gallery → adapted-DINO part-MaxSim distmat（验证 = exp324d e10 44.67 all / 36.78 heavy，链路对）；
按 filename 对齐 Swin distmat（exp324f）；oracle 数学逐行复用 exp324g（topk Jaccard / P_dino_only / per-query max(AP) 上界）。
Claude broad review PASS（实跑验证 sum|lora_B|=1469.6>0，确认加载的是训练好的 adapter 非随机初始化）。

**结果（989 heavy queries）**：

| 指标 | 冻结(exp324g) | adapted(exp324h) |
|---|---|---|
| DINO-only heavy mAP | 8.65 | 36.78 (×4.3) |
| top-10 Jaccard | 0.062 | **0.253 (×4，更不正交)** |
| P_dino_only | 0.20%(2) | 0.71%(7) |
| oracle 上界 heavy mAP | 72.69 | **73.16** |
| **oracle gain** | +0.12 | **+0.59 (<+1 门槛)** |

fusion sweep best：ALL 75.53(minmax w0.2，**+0.37**)、HVY 72.83(zscore w0.15，+0.26)；w≥0.4 转负。
re-rank 主动跳过（repo re_ranking 需完整 (Q+G)² q-q/g-g，仅有 q-g，不伪造）。

**判定 = 实质 STOP-LOSS（确认 analysis）**：
- **关键反直觉**：adaptation 让 DINO 变强的同时**也让它与 Swin 更一致**（Jaccard ×4）。"变判别"= 学到与 Swin
  相似的判别方向 → **互补性随判别性上升而下降**。救回的多是 Swin 也接近能救的样本，非系统盲区。
- oracle 上界仅 +0.59（< +1），绝对天花板 73.16 仍 < Swin all-query 75；perfect fuser 都推不过门槛。
- +0.37 ALL 是 test-time distmat 融合微小后处理（弱检索器掺 20%），按铁律属 NFC/RR 同级 trick，**非训练端方法贡献**，
  不构成 "beat 75 = 真方法"。
- exp324g(冻结)→exp324h(adapted) 两端夹逼，确认 "DINO completes Swin" 家族对 SOTA Swin 无足够独立信息。**止损**。
- 诚实价值：给 overnight FM 分析补最后一块——**"变判别 ≠ 变互补，反而更冗余"**，干净负向方法结论。
  与 exp324d 新颖性裁决（me-too，loses to SOTA all-query ~47）一致：FM-for-occluded-ReID 是 analysis 贡献，不是 beat-SOTA 方法。

### [exp324h — 方法猎结束 + headline 洞察] + large 已 kill
**exp324h（adapted-DINO ⊕ Swin oracle）**：adapted-DINO 不互补 Swin。oracle gain +0.12(frozen)→+0.59(adapted)，仍<+1；融合 best ALL 75.53(+0.37)/HEAVY 72.83，w≥0.4 转负（NFC 级后处理非方法）。
**⭐ headline 洞察（判别性-互补性张力）**：让 DINO 判别化(8.65→37) 的同时它和 Swin **更一致**(Jaccard 0.062→0.253)。adaptation 把 FM 推向 SOTA-like 方向 → 判别升、互补降，**无法兼得**。冻结 FM 互补但无用；adapted FM 有用但趋同 SOTA。**通用 FM 冻结/adaptation 都 beat/boost 不了 purpose-built SOTA**——fundamental。
**+ 新颖性审计**：DINO-LoRA 单分支 me-too（PersonViT 72.2 > 我们 47）+ 低于 SOTA → 标准方法不成立。
**结论：FM-import 方法猎彻底关闭（MLLM/frozen/换源/LoRA-standalone/LoRA-fusion 全负，各有机制）。** large(15h dead-end) 已 kill。**夜间 useful 产出 = `fm_occluded_reid_study.md` 诊断研究（含张力洞察 + ×4 finding + 可复用 oracle 工具）**——一个诚实、证据扎实、对领域有用的 analysis 贡献（劝退+量化+解释"搬通用 FM 赢 occluded ReID"这个诱人方向）。
**剩余**：base-r16/r32 跑完 e30（~40 plateau 确认数）→ 并入 results.md/decisions.md + 晨报。不编造 method。

### [exp324d_r32 e15 — plateau ~40 heavy/~48 all，路线2(超SOTA)实质死，确认 building-block 定位]

r32 part-MaxSim 轨迹（决定性）：

| epoch | HEAVY mAP | ALL mAP | Δheavy/5ep |
|-------|-----------|---------|-----------|
| frozen | 8.65 | 14.61 | — |
| e5 | 36.72 | 44.54 | — |
| e10 | 38.85 | 47.12 | +2.13 |
| **e15** | **39.44** | **47.71** | **+0.59** |

- **明显 plateau 在 ~40 heavy / ~48 all**。train 端 acc 0.999 饱和、LR cosine 衰减中 → e20-30 大概率仅微涨（不指望破 50）。
- **核心问题 firm 答案**：LoRA 解冻**彻底破冻结 8.65 天花板(~4.6×)**——FM-adaptation 真能让 DINO 判别化。但**上限远低于 SOTA all-query 60-72**（ProFD 62.8 / PersonViT 72.2）。
- **新颖性裁决 + plateau 双确认**：novelty agent 的"路线2=打平/超SOTA"对单分支 pose-part-MaxSim 已**实质不可达**（~48 all-query vs 需 ≥62）。剩可走路线只有 **(1) 机制重组 LoRA↔visibility** 或 **(3) 问题 reframe（common-visible support / reliability-aware matching，CLAUDE.md 钦定方向）**。
- **诚实定位**：exp324d 是**有价值的 building block / 正向诊断证据**（"frozen FM 不行但 LoRA-adapted FM 能把 occluded 部位匹配做到 competitive-ish"），不是 standalone strong-venue 方法。其最大论文价值是**支撑 analysis 故事**（FM 对 occluded ReID 能/不能做什么的完整证据链：frozen oracle 无独立信息 → 换源全死 → LoRA 解冻破天花板但 plateau）。
- **下一步判断**：等 large e5（OOM-fix 验证，capacity 对照——large 是否比 base 高？若 large 也 plateau ~40 则 capacity 非瓶颈，坐实"机制/问题"才是瓶颈）+ r32 e30 收尾上限。**不为刷 0.x 继续调这条线**（CLAUDE.md 铁律）。

### [exp324i — 解相关感知 DINO-LoRA：直攻"判别性-互补性张力"的真 method shot]

**动机**：exp324h 发现的 headline 张力——LoRA 让 DINO 判别化(8.65→37) 的同时它和 Swin 越来越像(top-10 Jaccard 0.062→0.253)，融合只 +0.37。**若显式强迫 adapted-DINO 与 Swin 线性解相关，它能否进入互补子空间、融合真超 Swin(72.57 heavy/75 all)？**

**方法（跨网络跨协方差解相关，Barlow-Twins 的跨网络版，Codex 查无直接先例）**：
- 预缓存 frozen exp255-Swin 的 holistic global 特征 `s`（前 768 维，L2-norm，确定性 val pipeline，15618 train 图）。
- DINO-LoRA 训练加 `λ·L_decorr`，`L_decorr = mean((d̂ᵀŝ/B)²)`（d̂/ŝ 为 batch 内逐维 z-score；s detach 只 d 回传）。逼每个 DINO 维与每个 Swin 维线性不相关。
- 对照：**λ=0（≡exp324d，趋同 Swin）vs λ=1（解相关）**，单变量；rank16/30ep/seed1234 全同。λ=1 跑 lab-3090-d，λ=0 control 跑 hyy GPU1（r32 完后）。

**审查**：Claude broad review 审查通过（无 Critical/High，1 Medium=global-vs-global scope 限制已标注）；Codex `--search exec` verdict **approve**（2 Low：zero-std backward NaN → 改 rsqrt(var+eps)；cache dim 未断言 → 加 --expect_dim=768，均已修）。dry-run λ=1 decorr≈0.025 finite、λ=0 decorr=0 且数值≡exp324d、peakGPU 2.42G。

**判据**：λ=1 vs λ=0 看 (1) decorr-DINO 单分支 mAP 是否被解相关拖垮；(2) top-10 Jaccard 是否下降（张力是否被打破）；(3) fusion(decorr-DINO⊕Swin) 重遮挡/全部是否真超 Swin。**无论成败都有价值**：成→真 method；败→把张力从"观察"升级为"显式施压也打不破"的强诊断结论。

**先验（诚实）**：~75% 偏负——Swin 已占据最判别方向，正交补里 ID 信号更弱；且 global-only 解相关不针对 Swin 的遮挡盲点 + 95.8% 训练全可见墙仍在。但这是 FM-import 方向最后一个有原创性的方法介入，值得这一夜的空闲 GPU。

### [heartbeat idea-probe ~03:40] 剩余未试 paradigm 廉价判死，不烧 GPU 凑数
三块可用 GPU 全忙（λ=1 / large / r32→λ0 armed），lab-4090 env 坏（不敢自动装包破坏多用户 afr 环境）。趁等 λ=1 e10，过了一轮"还有什么热点没搬"：
- **SAM/SAM2 遮挡 mask**：落入禁止清单 visibility 变体 + KPR 已 human parsing → 增量，不做。
- **Pose2ID 式生成补全**：撞 exp109 墙（completion = identity-conditioned 不可实现）→ 不做。
- **文本-grounded / CLIP-text**：occluded ReID 是 image-to-image 同网络匹配，文本不加信息 + CLIP-ReID 已先例 → 不做。
- **3D/NeRF 人体重建**：太重、无廉价 kill-switch、仍撞补全墙 → 不做。
**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。

### [exp324d capacity 对照收尾 — large≈base，瓶颈是机制不是容量，FM 线探索性结论 firm]

**large(dinov2-large hidden 1024) 与 base(dinov2-base) 在同一 plateau 带，capacity 不是瓶颈。**

| 变体 | e5 part HEAVY | e5 part ALL | 容量维度 |
|------|---------------|-------------|----------|
| 冻结 base (exp324b) | 8.65 | 14.61 | baseline |
| **r32 base LoRA rank32** | 36.72 | 44.54 | +LoRA 容量 |
| **large LoRA rank16** | **38.50** | **47.21** | +backbone 容量 |

- large 比 base 仅 **+1.8 heavy / +2.7 all**（同一 ~40/~48 带，非不同 regime）。r32(rank32 vs rank16) 也 plateau ~40。
- **双对照结论：backbone 容量(base→large)、adaptation 容量(rank16→32)都不是瓶颈** → 瓶颈在**机制/问题结构**（pose-part-MaxSim 5 部位表征上限）。
- **OOM 教训记录**：large 首跑 e4→e5 静默死（eval 阶段 dinov2-large 全量 gallery 前向显存超 16G，train 峰值才 5.34G）。修复 `--eval_fwd_bs 8`（不动 BS=64），relaunch 通过 e5 eval。→ 5060 Ti 16G 上 large backbone 的 eval 必须降 eval_fwd_bs。

**FM 方向（DINO）探索性最终结论（firm）**：
1. 冻结 FM 对 occluded ReID 部位匹配天花板低（8.65/14.61），换源(registers/DIFT/DINOv3)不破，oracle 无独立信息。
2. **LoRA 解冻彻底破冻结天花板（8.65→~40 heavy，~4.6×）** → 瓶颈是"冻结"不是 DINO 表征本身。
3. **但 LoRA-adapted 上限 plateau ~40 heavy / ~49 all，远低于 SOTA(60-72 all-query)，且加容量(large/rank32)不破** → 瓶颈转移到机制/问题结构。
4. **新颖性 me-too**（DINOv2 backbone 饱和 / LoRA-ReID 已有 / pose-visible-part 匹配 PVPM2020→KPR2024）。

→ **FM-as-backbone 刷点这条线到此为止（CLAUDE.md 铁律：不为 0.x 调参、不堆模块）。** 真正剩余价值：
(a) **analysis 故事**（完整证据链：frozen 不行 → 换源不行 → LoRA 破冻结但 plateou → 容量非瓶颈 → 机制才是，对"FM 对 occluded ReID 能/不能做什么"是扎实诊断贡献）；
(b) 若要做方法，必须走 **LoRA↔visibility 机制重组** 或 **common-visible-support/reliability 问题 reframe**（CLAUDE.md 钦定方向），不在这条曲线上继续。

剩余：r32 跑到 e30（不改结论）、large 跑到 e30 看 large 最终是否仍 ≈base（预期是）。**两者 e30 后这条 FM 线探索关闭，转 analysis 或机制/问题 reframe 设计。**

### [exp324d_r32 e30 完成 — FM-LoRA 线最终数字定格]

**r32 训练完成（`[done] full training complete`），最终天花板定格：**

| | part HEAVY mAP/R1 | part ALL mAP/R1 | vs 冻结 |
|--|-------------------|-----------------|--------|
| 冻结 base (exp324b) | 8.65 / — | 14.61 / — | baseline |
| **r32 e30 (final)** | **40.81 / 51.97** | **49.68 / 62.40** | **+32.16 / +35.07 (~4.7× / ~3.4×)** |

- 完整收敛曲线：e5 36.72 → e10 38.85 → e15 39.44 → e20 40.58 → e25 40.71 → **e30 40.81**（heavy）。e20 后基本不动。
- ckpt 已存（head_30.pth + lora_30）。GPU1 释放。
- **FM-LoRA 探索这条线正式关闭**：破冻结天花板证实(✅)，但 plateau ~40.8/49.7 远低于 SOTA 60-72(❌)，加容量(large e5 38.50≈base)不破(❌)，新颖性 me-too(❌)。
- large 仍在跑到 e30（capacity 对照完整曲线，非 load-bearing，e5 已确认 ≈base，结论不变，~7h 后完成）。
- **下一步不是再开 DINO-curve 实验**（CLAUDE.md 铁律：不刷点不堆模块），而是：转 analysis 故事整理，或设计 LoRA↔visibility 机制重组 / common-visible-support·reliability 问题 reframe（CLAUDE.md 钦定方向，需先写设计+红蓝队）。

### [exp324i e10 ORACLE 判决 — decorr 没打破张力，张力升级为"显式施压也打不破"]

λ=1（decorr active）e10 oracle vs λ=0（exp324d e10）matched 参考，**每个数都一样**：

| 指标(heavy) | λ=0 无decorr | λ=1 decorr |
|---|---|---|
| DINO-only mAP | 36.78 | 36.49 |
| **top-10 Jaccard vs Swin** | 0.253 | **0.2513** |
| P_dino_only | 0.71% | 0.81% |
| oracle 上界 | +0.59 | +0.58 |
| **fusion best ALL** | 75.53(+0.37) | **75.52(+0.37)** |
| fusion best HEAVY | 72.83(+0.26) | 72.84(+0.27) |

- **decorr loss 全程活跃**（稳 0.041，远低于 λ=0 自然相关）**却完全没移动 Jaccard / oracle / fusion**。
- **机制解读（关键）**：强迫 DINO-global 与 Swin-global **线性解相关**，对"模型给 query 排哪些 gallery"（part-MaxSim 排序）是**正交的**——决定检索的是 part-MaxSim over 相同可见身体部位证据，两模型受**同一份可见证据**约束而犯**同样的错**（Swin-only-r1-hit 370/989=37%，DINO 只补 8=0.81%）。global 线性相关只是排序的"装饰"，解它不改排序。
- **结论**：**显式解相关施压打不破"判别性-互补性张力"** → 张力从"观察到的相关"升级为"对显式干预鲁棒"的**强诊断结论**。这是 exp324i 的真正价值：method shot 对 beat-SOTA 为负（fusion 仍 +0.37 NFC 级），但作为**严格对照**坐实了 headline 张力（诊断论文的关键实验）。
- **待补强（让 sweep bulletproof）**：λ=2（更强 decorr）+ λ=0 fresh rank16 matched control 跑到 e30，确认 e10 结论在收敛点/更强 λ 下都成立（预期：Jaccard 仍不动）。

### [large e10 更正 — capacity 帮得动一点，"large≈base"是 e5 快照过度简化]
large e10 出来后须更正之前结论：matched e10 看 **base 36.78 < r32 38.85 < large 41.72 heavy**（large 比 base +4.9，且 e5 38.50→e10 41.72 还在爬）。所以**容量单调有帮助 ~+3-5 mAP，不是"无帮助"**——之前 agent 基于 e5(large 38.50≈r32 36.72)下的"large≈base、capacity 非瓶颈"是过度简化，更正为：**capacity 帮得动一点但补不上 ~25-30 的 SOTA gap（large ~42 heavy << 72.57），瓶颈仍主要在机制/问题结构**。me-too + FM 方法关闭结论不变。诚实记录，避免把"容量无用"写死。

### [λ=1 e30 oracle 数已captured，verdict 等 matched λ=0/λ=2 e30]
λ=1(decorr) e30 oracle：single-branch 38.69 heavy/47.20 all、top-10 Jaccard **0.2627**、oracle +0.80、fusion best ALL 75.73(+0.57, w0.3 zscore)/HVY 72.94(+0.37)。
- vs λ=1 e10（Jaccard 0.2513、oracle +0.58、fusion +0.37）：收敛后 Jaccard/oracle/fusion 都升——但这是**任何 adapted 分支变判别后的预期**，不是 decorr 特有。
- **不在 isolation 下结论**：decisive 是 **λ=1 e30 vs matched λ=0 e30**（λ=0 还在训 ~e15，~2.5h）。e10 matched pair 完全一致（张力held）；若 λ=0 e30 也 ~0.26 Jaccard/~+0.5 fusion，则收敛点 decorr 仍无效。fusion "BEATS 75"(75.73) 是 NFC 级后处理(w0.3)，非训练端 method（项目规则）。
- 待 λ=0/λ=2 e30 oracle 跑完做 matched sweep 一次性 document + commit。

### [★ matched e30 verdict — decorr 在收敛点也完全无效，张力 bulletproof]

λ=0 vs λ=1 **同 rank16/seed/script，e30 收敛点 matched oracle**：

| 指标(e30) | λ=0 无decorr | λ=1 decorr | Δ |
|---|---|---|---|
| single-branch heavy | 39.05 | 38.69 | -0.36 |
| **top-10 Jaccard** | 0.2646 | 0.2627 | **-0.002（噪声）** |
| P_dino_only | 0.71% | 0.91% | +0.20 |
| oracle gain | +0.85 | +0.80 | λ=0 反而略高 |
| **fusion best ALL** | **75.74** | **75.73** | **-0.01** |

**收敛点也完全一致** —— decorr 对 Jaccard/oracle/fusion 零影响（λ=1 甚至在 oracle 上略低于 λ=0）。**配合 e10 matched（0.253 vs 0.2513）→ 早期+收敛双确认：解相关在任何训练阶段对互补性都是零效果。** fusion "BEATS 75"(75.74) 在有/无 decorr 下一模一样 → 纯粹是"加一个判别分支"的 NFC 级效果，不是 decorr 换来的，且 w=0.3 后处理非训练端 method。

**最终结论（bulletproof）**：判别性-互补性张力**对显式解相关干预（e10/e30、λ=0/1/2、即将出 λ=10）全程鲁棒**。机制：全局线性解相关与 part-MaxSim 检索正交，且 ~0.041 残差相关是 ID-constrained floor（λ=2 双倍权重只降 ~1%）。**这是 analysis 论文最硬的对照实验：我们主动设计损失去打破张力，单变量证明打不破。** exp324i method 对 beat-SOTA 为负、作为诊断对照为强正。

### [reid-gap-hunt workflow — 系统调研判定：现有问题上无 beat-SOTA 新 method，转换新任务]
用户质疑"做这么多 λ 是不是逃避调研"——属实，止损：停 λ=10、不开 λ=0.5，启动真调研 workflow(32 agent / 167万 token / 6路扫文献→综合→对抗验证→选)。
**结果：8 候选 0 过审(none_strong_enough)。** 双死因，每候选皆中：
1. 撞项目已证实墙：occluder-gate 多人 no-op(exp290/291 Δ0)、exp109 +8.53 identity-conditioned 不可实现、训练集 95.8% 全可见无梯度、FM 判别性-互补性张力 fundamental。
2. 皆有真实已发表先例无本质区别(联网查实)：CXR≈TransMatcher(NeurIPS21)/FRT(TMM23)；O-IQT≈GIQT(26)；Selective/Conformal≈RCIR(AAAI25)/UAL；Occluder-Leakage≈FED/PISNet；Corr-Validity≈HOReID/PVPM；STA≈KPR(ECCV24)；OGSP≈PAT/PLOT。
唯一不撞墙残值：PoseFaith faithfulness 诊断(MaxSim 逐 qkp 贡献分解，无先例)——诊断工具非 method。
**结论**：occluded person ReID 这个问题挖到底(有据可查，非偷懒)。真正的新主线只在需投入数据/基础设施的**新任务**(aerial/TBPS/video，cold-start)。**用户拍板：换新任务。** 具体任务待定。

### [tbps-transfer-hunt workflow — 7候选1活，幸存=PartNC(pose 区分遮挡vs标注噪声)]
TBPS 新主线 + gait/face/跨模态搬运调研(29 agent)。**7 候选，6 死于真实顶会先例(落进已证伪区)**：VG-TBPS/common-support≈MGCC(AAAI24 遮挡TBPS)/PLOT(AAAI25)/PMA(AAAI20 pose-短语对齐)/ProFD(MM24)；其余撞 Visual-Perturbation(AAAI25)、uncertainty-aware TBPS。
**唯一幸存 PartNC(Part-level Noisy-Correspondence)**：用部位粒度 image-text 相似度估每个身体部位 clean/noisy 置信(RDE 的 CCD 从 pair 级下沉到部位级)；**真正空白=用 pose-visibility 先验区分"相似度低=遮挡(结构化)"vs"=标注错(随机)"**(RDE/DURA/GA-DMS 把 noise 当随机，表达不出)。复用我们 part-MaxSim 逻辑(einsum bkc,gpc->bgkp .max)+pose 可见性积累。
**诚实定位**：非稳赢，是"七里挑一的探索性赌注"。precedent lens 判死过(部位级 NC 重加权被 GA-DMS/DEMA/DURA 占；空白只剩 occlusion-vs-noise 这窄条)。"image side 已搭好"是夸大(仓库无 CUHK-PEDES loader/caption 解析/IRRA backbone，per-part 文本相似度需重建~2-3天)。
**kill-switch(干净廉价)**：纯数据扰动诊断、第一刀不碰 loss。CUHK-PEDES 部位级注噪(20/50%替换某身体部位子句)，用现成 IRRA/RDE checkpoint token 特征重建 part-level cleanliness AUROC vs RDE 实例级 CCD AUROC。50%噪声下打不过 CCD→当场判死。2-3天单卡。门槛若要"稳赢SOTA候选"则接近 none_strong_enough。

### [★ occluded 封板 — decorr sweep(λ=0/1/2 e30 matched) 完整，张力 bulletproof]
| 指标(e30 matched rank16) | λ=0 | λ=1 | λ=2 |
|---|---|---|---|
| single-branch heavy mAP | 39.05 | 38.69 | 38.18 |
| top-10 Jaccard vs Swin | 0.2646 | 0.2627 | 0.2604 |
| oracle gain | +0.85 | +0.80 | +0.78 |
| fusion best ALL | 75.74 | 75.73 | 75.70 |
| P_dino_only | 0.71% | 0.91% | 1.01% |
**decorr 强度 0→1→2：Jaccard 仅 0.2646→0.2604(Δ0.004 几乎不动)、fusion 75.74→75.70 几乎不变、单分支判别力 39.05→38.18 与 oracle +0.85→+0.78 都单调小降。** 即解相关不仅打不破"判别性-互补性张力"，过强还轻微有害(削判别力却换不来互补)。配合 decorr-floor(λ=2/10 双倍/十倍权重压不下 ~0.04 相关)→ **张力对显式干预 fundamental 鲁棒**。
**occluded person ReID 主线到此彻底封板**：PSG 全栈 SOTA 75.2(exp255) + FM-import 全证负(MLLM/frozen-DINO/DIFT/LoRA/decorr) + 张力洞察(显式打不破) + capacity 修正(large ~54all/45heavy plateau, me-too) + exp109 三堵墙 + PoseFaith 残值。诊断论文素材齐。新主线=TBPS(PartNC 候选待用户拍板)。

### [PartNC 首验脚本就绪 — 只差 CUHK-PEDES 数据]
按用户睡前"不停+cheap kill-switch first"，自主把 TBPS 唯一幸存候选 PartNC 推到"数据一到就能判生死"：agent 读懂 RDE 接口(checkpoint 加载 / image 192-patch+text 77-token 特征 / CCD 的 min-max+2-GMM+低均值簇后验)，写 `scripts/partnc_killswitch.py`(本地+lab-3090-d，md5 一致)：(a) spaCy+lexicon 切 caption head/torso/legs + 部位级噪声注入(参数化, 有 regex fallback)；(b) RDE token 特征算 noun-phrase×patch per-part MaxSim cleanliness(复刻 eval_fliptest_maxsim einsum.max)；(c) AUROC(part-level) vs AUROC(实例级CCD)；(d) verdict。逻辑全验证通过(切分/注入/加载/形状/MaxSim/GMM/AUROC 方向性单测)。
**caveat(必修)**：`--real` 现用"1−均值cleanliness"代理 CCD，会压低对照让 PartNC 偏强；数据到位**第一件事换 RDE 真 CCD(compute_per_loss 跑 dataloader)** 再判，否则 verdict 不可信。运行：`python3 partnc_killswitch.py --mode real --root_dir <CUHK-PEDES父目录> --noisy_rate 0.5`。**只差用户 odl login 下数据。**

### [★ PartNC 首验判死 — 换真 CCD 公平对照后输给 RDE CCD]
数据自动下好(HF 镜像 cjc/CUHK-PEDES 840MB 全分辨率, reid_raw.json 40206 caption/38942 图, 绕过 OpenDataLab login)。首验**先把代理 CCD 换成 RDE 真 CCD**(接 `model.compute_per_loss` 拿 lossA/lossB → min-max → 2-GMM → 低均值簇后验, 复刻 processor.get_loss)。
**AUROC(2500/档, 2 种子)**：
| 配置 | PartNC pair | 真CCD pair | 公平Δ |
|---|---|---|---|
| 20% | 0.7405 | 0.7479 | −0.007 |
| 50% s123 | 0.7290 | 0.7538 | −0.025 |
| 50% s777 | 0.7337 | 0.7560 | −0.022 |
坑全干净(0 NaN, 0% 对齐失败, 注噪率精确)。**VERDICT: 判死** —— 同粒度公平对决 PartNC 输给真 CCD(门槛要 >+0.02, 实际 −0.025)。机制: 真 CCD(CLS全局+TSE token 双路)已吃透"某部位被换"信号, 拆部位反更弱; 之前"赢"是同源 MaxSim 代理的不公平对照。part-level 定位略高(+0.006~0.031)但属退化对照, 不翻案。**kill-switch 价值兑现: 几小时干净判死省成稿。** TBPS 唯一幸存候选亦死。

### [gait-face-survey — 单独深挖, 3 个 ReID 真没借过的训练端机制(经 codebase grep 核对)]
13 机制 / 6 标 no-gap, agent 逐条 grep 仓库核对, 砍掉一半(早借过/刚判死)。**3 个干净真空白(均训练端, 避开 test-time 禁令, 与已死方向正交)**：
1. ⭐ **因果 backdoor-triplet**(搬 gait GaitC3I TIP25/GaitSCM CVIU24)：把 camera-id+occlusion-state 当显式混淆变量 C, backdoor adjustment 改 triplet 正样本采样, 让跨遮挡/跨摄像头正样本对不再发协变量差异冒充身份的有害梯度。仓库零因果/SCM 代码(唯一命中 DACHM/DACCM 是 training-free reranker, 另一 lever)。**唯一同时满足: 问题重定义+机制新+零代码+避所有已死框架**。遮挡状态从 pose 可见性直接得。0.5天无训练首验。
2. **UCE 统一阈值校准 loss**(搬 face UniFace ICCV23)：可学阈值 b 分离所有正负, 校准分数尺度→开集拒识+跨域。训练时 loss(躲 test-time 禁)。
3. **TopoFR 拓扑对齐+结构损伤难样本**(face NeurIPS24)：持续同调对齐+结构损伤分选难样本。重量级 import。
CarGait 学习式 part reranker=第4, 贴 retrieval-scorer 禁区, 优先级最低。
**砍掉(早借/已死, 负面先验)**：LGPA(text 解耦已建)/FSDC(补全已建+撞墙)/LTCS-LPCS(common-support 判死)/MLS-uncertainty(PartNC 刚死)/adaptive-margin(贴 visibility 证伪)/PACI(sub-center 已建)/conformal(test-time 禁)/counterfactual-attention(PGMA 已有)。
**最强=backdoor-triplet, 0.5天现成特征验 confounder 是否真实。已启动 probe。**

### [backdoor-triplet probe — KILL: 混淆真实但量级可忽略, SOTA 已压没内部 headroom]
0.5天无训练 probe(现成 exp255 train 特征 + pose 可见性 + cam, 同序100%对齐)。同 pid 正样本对按 (同/跨cam)×(都清晰/跨遮挡/都遮挡) 分6桶看 d_ap=1−cos：
| 桶 | n | mean d_ap |
|---|---|---|
| same-cam/both-holistic | 42548 | 0.0079 |
| same-cam/cross-occ | 715 | 0.0143 |
| cross-cam/both-holistic | 92663 | 0.0116 |
| **cross-cam/cross-occ** | 1735 | **0.0163** |
负样本 d_an=0.0950, 最难1% 负样本 p1=0.0582。
**混淆统计真实**(cross-occ vs holistic: same-cam +0.0064 z=27.8, cross-cam +0.0047 z=36.8; cam 混淆 +0.0037 z=132.7, 方向一致系统性, 阈值放宽不变)**但量级可忽略**: 最虚高桶 0.016 离最难负样本 0.058 差 3.6×, 遮挡漂移 ~0.005 仅占正负 margin ~0.08 的 6%, 所有正样本已远低于任何负样本。
**KILL**: backdoor 要消的"协变量冒充身份"虚高**没侵蚀决策边界**(SOTA exp255 已把跨cam/跨遮挡同人正样本压得离边界数量级远), 去掉不改变任何排序=无可回收错误=无 headroom。**深刻含义: SOTA ReID 特征上, in-domain 去混淆类训练端机制无 headroom** — 呼应张力+三堵墙+MEMORY"别在 ReID 内部找机制"。#3 拓扑(也 in-domain)大概率同命; #2 UCE(测跨域/开集分数尺度漂移)测不同维度, 待定。

### [TopoFR 结构损伤 probe — KILL: 遮挡不破坏局部流形(SOTA 压平), 且是 ID-size 假信号]
现成 exp255 特征 probe。purity@20 clean(0.714)→heavy(0.89) **反升 +0.067**(遮挡不破坏邻域); rho(damage,occlusion)=+0.072≈0; damage 与 triplet-hardness rho=+0.08、jaccard 0.095(≈随机)=噪声。诊断拆出真因: 遮挡图属大 ID(图数 40.6→101.5)→kNN purity 机械偏高=ID-size confound, 非结构优势; 近重复紧致度持平排除抱团。**KILL: TopoFR 赖以工作的"遮挡破坏拓扑"现象在 SOTA backbone 下不存在(被压平)。**
**#1 backdoor + #3 TopoFR 死法完全一致: in-domain 内部训练端机制被 SOTA exp255 压没 headroom。** → 数据驱动地排除 in-domain 内部机制, 聚焦 SOTA 压不平的: 跨域/开集/新监督/新问题定义/重量级范式 import。#2 UCE(跨域) + wider-transfer-hunt(专挑 in-domain 外) 在跑。

### [UCE 跨域校准 probe — KILL: SOTA 连跨域分数尺度都校准好了]
现成资产(Market ckpt exp260b + test_on_occluded_reid.py + 两域数据, 无训练)。genuine/impostor 余弦分布: 源域 Market d'=6.51/AUROC.998/EER.95%/阈值0.636; 目标 Occluded-ReID d'=3.92/AUROC.994/EER3.65%/阈值0.642。**阈值迁移零代价**(源→目标 shift +0.006, 直搬 FAR4.06%/FRR3.14% vs 原生3.65%)。KILL: 跨域分离度掉是**遮挡判别性**问题不是 scale drift; 全局阈值几乎不动→SOTA 已校准好, UCE 无空间。(agent 已写 decisions.md commit f2411a9)
**累积 pattern: gait/face 3 top pick(in-domain 因果#1 / 拓扑#3 / 跨域校准#2)全被 SOTA 占满。** 否证"跨域有空间"。→ 真正没被占的是 **新问题定义/新任务/重量级范式 import**(不是在 exp255 上加训练端机制)。wider-transfer-hunt 专瞄这几路, 在跑。

### [large(DINOv2-large LoRA) e30 final — capacity 终点封板]
e30: part HEAVY 45.04 / ALL 54.10, cos HEAVY 43.68 / ALL 53.27。capacity 修正定稿: large 比 base/r32(~40/49) 高 ~+4-5 但同样 plateau(e15-30 平), me-too, 离 SOTA 75 仍 ~20。occluded ReID 主线全部封板。

### [后半夜2: VC-Norm 升级训练 + 用户洞察"换弱 baseline" + 范式 probe]
**用户拍板**: 不停不问、范式级创新、"SOLIDER 太强就换 TransReID"。据此双线并行:
- **VC-Norm 线(SOLIDER)**: probe PROCEED(遮挡=未对齐 domain factor, per-part token KL 94-300 近完美可分, 3 对照排除伪影) → 实现(崩溃 agent 留码, dryrun 全过) → Claude review PASS → **Codex 抓到 High-1 机制 bug(score 加权把被遮挡 token 踢出统计→空转)** → 修(改对齐 occluded subset, 单测证被遮挡 token 拿梯度 0.96) → Codex 复审 approve → **exp328 训练中(lab-3090-d, warmup10/40ep, VCA e11-40) + control(4090 VCNORM=False 单变量对照, 你的 uv 清华源救活第4卡)**。真 eval e20/e40。
- **弱 baseline 线(用户洞察)**: in-domain 判死全在 exp255 强栈上测(headroom 被压没); 换弱 baseline 是有据可查的合法逃逸。**hyy 跑 TransReID vit_base Occ-Duke 弱 baseline(纯 PyTorch 避 mmcv, Blackwell+torch2.9, e20 mAP 47.5→奔~59)**, 给 COG 等范式候选当验证场。
- **范式候选(paradigm-hunt, 6 个)**: HyperReID(双曲)/CompVMF(生成式 vMF) **0-GPU probe 都 KILL**(去 ID-size 混淆后前提反/弱); COG(组合泛化重定义, 最强范式相)/Pose-JEPA/Self-paced/RLVR 待 API 恢复+TransReID ready 后验。
- 4090 env: uv 清华源装 torch1.13+cu117+mmcv2.1 成功(用户方案)。numpy<2 修 torch1.13 ABI。

### [后半夜3 (compaction 后接手): 三线健康推进 + 重启范式调研 + 决定弱baseline 重测最强机制]
**状态盘点(server 时间 ~04:0x)**：
- **VC-Norm 线**: exp328(lab-3090-d) e12, VCA 活跃且在塌缩 moment gap —— `vca: 1.35(e10)→0.595(e12)`, `vcn_gain_abs 0.001→0.002` 上升, `vca_sd 1.02→0.34`(std-dist 大幅收窄)。证 High-1 修复后机制真在对齐, 非空转。control(4090) e13 健康(VCNORM=False 单变量)。**真 eval e20 Market(~40min) 是第一个 VCA vs control 信号。** lab-3090-d 跳板 banner 间歇超时但 monitor br88g09gi 仍在推 exp328 事件, 不需手动 SSH。
- **TransReID 弱baseline 线**: hyy GPU0 e37, e20 mAP 47.5→奔 e120 ~59。冷启动 agent 已完成(pipeline 全验证)。清掉其逐epoch噪声 monitor, 换干净 monitor(只 mAP/崩溃)。
- **代码提交**: VC-Norm 实现(loss/module/configs/wiring)之前只归档文档、代码在工作树未提交 → 补提交 2002ce5, 保证可转移+复现。
- **hyy GPU1 空闲**。

**战略锁定(数据驱动)**: backdoor/TopoFR/UCE 三个 in-domain 训练端机制全在 exp255 **强 SOTA** 特征上判死(headroom 被压没)。但这恰好是用户"换弱 baseline"洞察的实证理由——**同样的机制在弱 baseline(TransReID 59, 离 SOTA 还差 16 mAP)上可能有 headroom**。最强已 novelty 过审机制 = **因果 backdoor-triplet 去混淆**(搬 gait GaitC3I/GaitSCM, 仓库零 SCM 代码, 正是用户要的"从步态抄"), 强特征上混淆真实但仅占 margin 6% → 无可回收。**弱 baseline 重测 = 干净的 0-GPU kill-switch**: 用 TransReID-weak train 特征跑同一 6 桶 confounder probe, 看 cross-occ/cross-cam d_ap 虚高占负样本 margin 的比例是否远大于强特征的 6%。若 ≥20-30% → 有可回收 headroom, 升级全量训练; 若仍 ~6% → 弱 baseline 也无 → 彻底排除 in-domain 去混淆。

**并行重启范式调研 agent**(后台): 专扫 gait/face/video/CC-ReID/vehicle-ReID 的 CCF-B+ 可迁移**范式**(问题重定义或无 occluded-ReID 先例的机制), 强制避开全部已关方向, 要求 novelty 证据 + <1天 kill-switch + 必须能在弱 baseline 显 headroom。瞄准 SOTA 压不平的空间(新问题定义/新任务/重量级 import)。

### [范式调研产出 → BET#1 burstiness 抑制 / democratic part-set 聚合(最强候选, 已搭 probe)]
调研 agent 扫完 gait/face/video/CC-ReID/vehicle/compositional, 多数撞已关方向(occlusion-type 预测=OA-ReID 已做; residual gait 补全=禁; vehicle cross-view=common-support 已死)。**唯一过审强 bet = burstiness suppression**(搬 VLAD-BuFF ECCV'24 地点识别 + On the Burstiness of Faces in Set arXiv'25-06 + Jegou CVPR'09)。
- **机制**: burstiness = 某特征元素出现频率超独立性预期(大块近匀质躯干 patch/重复纹理)。相似度对元素求和→过表达元素抬分不加判别信息, 挤掉稀有可辨细节。修法 = self-similarity 反比降权(VLAD-BuFF 闭式 `w_i=1/Σ_j sim(f_i,f_j)`), democratic 聚合。
- **为何非已关方向**: 与 visibility 加权**正交**(visibility 问"是否被遮挡"二值; burstiness 问"这个可见特征是否被过度计数"集内冗余统计)。一个 part 可完全可见且高 visibility 却 bursty(大块平躯干)——现有方法升权它, burstiness 说降权。非 completion(不重建)、非 scorer 微变体(改特征集聚合)、非 uncertainty(无 per-feature 置信)。**Novelty 已核: 无 occluded-ReID 做 burst-aware/democratic over part/patch 特征**(最近 cousin = Self-similarity guided probabilistic matching ESWA'23 是去噪/校验, 相反符号)。详见 memory [[burstiness-democratic-aggregation-bet]]。
- **0-GPU kill-switch 已搭并验证**(`scripts/burstiness_probe.py`, staged on hyy): 复用 TransReID 自己的 make_dataloader/make_model/R1_mAP_eval(mAP 可直接对比训练日志), hook `base.norm` 取全 token 序列(B,129,768), 每图可见 token 集内算 burst 权重 → burst 加权 patch-pool vs **uniform patch-pool**(隔离 burstiness 机制, cls 仅作 context) → 重排比 mAP。**判据: burst−uniform ≥+1.0=真值得全量, <+0.3=kill。** + 诊断(query 遮挡 vs gallery 整体的集内 self-similarity)。
- **smoke test(pretrained ImageNet 权重)全 pipeline 跑通**: cls 2.94/uniform 4.01/burst 4.04, **burst−uniform=+0.02→KILL**(未训练特征无真 burst 结构→证 metric 不虚高, 校准良好)。
- **e120 自动触发已 armed**: 后台 until-loop 等 `transformer_120.pth`(CHECKPOINT_PERIOD=120 只在 e120 存档)→ 自动跑 burstiness probe 真实数(GPU1)→ monitor bc4m6btrv 推 VERDICT。预计 server ~05:2x。
- **次选(fallback#2, 擦边禁区)**: CompositionalNet vMF 生成式 analysis-by-synthesis, 重且一步之遥退化成 occluder-gate, 仅 #1 鼓舞时考虑。
- 范式调研同时**否证一批**: gait occlusion-type/residual=已关; CC-ReID 因果换衣=无遮挡 analog 不撞 disentangle; vehicle VANet 两度量=common-support 已死; set/video quality-pool=visibility 已关。诚实负面收窄空间。

### [burstiness 前提早验(frozen DINOv2, GPU1, e120 前 ~45min)]
不让 GPU1 空等 e120, 跑 `scripts/dino_burstiness_probe.py`: 复用 TransReID val_loader(同图/同 pid/同 query 划分), 换 frozen DINOv2-vits14, 256×128→224×112(patch14→16×8=128 token), 测**前提**(occluded 是否更 bursty) + 跨 backbone kill-switch。
- **✅ 前提 HOLDS**: query(遮挡) intra-token self-sim **0.4459** vs gallery(整体) **0.4253** = **+0.0206**(occluded 更 bursty)。即便强通用 DINO 特征上, 遮挡图也确实有更多冗余/过表达 token(occluder/背景泄漏区天然高自相似)——正是机制要降权的对象。**机制有燃料。**
- **⚠️ frozen-DINO 检索 kill-switch 不可用**: plain patch-mean 检索近随机(mAP 0.4, 无 pose 锚定, exp324 证 frozen DINO 要 pose 锚定才有 1.86)→ burst−uniform=-0.02 纯噪声底, 无意义。**检索判决必须用 identity-carrying 的训练特征 = e120 TransReID(已 armed)。** 诚实不过度解读 frozen 检索数。
- 结论: 前提成立(occluded 更 bursty), 检索效力待 e120 训练模型定。GPU1 用完归位(继续 armed e120)。

### [VC-Norm e10/e20 读数: e10 假警报(14.2 瞬态)→ e20 恢复 88.4, Market −1.0 vs control(符合预期)]
- **e10=14.2% 是一次性 eval 瞬态(假警报)**: 一度怀疑 VCN 模块毁特征(14.2 vs control 86.2), 但查 VCN 模块确为 zero-init 恒等(gain~0.005 极小, 远不足以掉 72 分), 两机 git 同 commit(715c020)、同 vcnorm.py、单变量 config 核对无误。**e20 恢复 88.4%** → e10 是孤立 eval glitch(疑 AMP/LayerNorm 一次数值抖动), 非真实轨迹。教训: 单点 eval 异常先查模块幅度+对照再下结论, 别急判死。
- **e20 真实对照(Market 整体集)**: VC-Norm **88.4** vs control **89.4 = −1.0 mAP**。**符合预期**: Market 整体无遮挡, VC-Norm 的遮挡对齐无处发力, 只显 VCN 模块的小成本(−1.0)。VCA 此时刚激活(warmup=20 结束, LR 降 2e-4 进主调度)。
- **真正判据 = 跨域 Occluded-ReID eval**(有真遮挡处), 非 Market。训练用 PLBOA 合成遮挡, VCA 训练时对齐合成遮挡 token, 收益只在遮挡测试集显现。待 e40 Market(确认 VCA ramp 不进一步伤整体) + 训练完跨域 Occ-ReID(决定性)。

### [burstiness probe 加 part-MaxSim 保险 + 前提 backbone 依赖性(重要 nuance)]
- **加 part-MaxSim kill-switch(防 false-KILL)**: pooled 判据可能误杀——cls 是训练检索特征, patch-pool 天生弱于它。加 token 级 part-MaxSim 版本(128 patch→6 水平 part-band, 全局 burst 权重降权 band 内 bursty patch, part-MaxSim 检索 burst vs uniform)。pretrained smoke 全跑通: pooled burst−uniform=-0.01, part-MaxSim burst−uniform=-0.03(均 KILL, 未训练应如此=harness 校准正确)。e120 自动触发用更新脚本, 一次出 pooled+part-MaxSim 双判据 + 训练模型诊断。
- **前提 backbone 依赖(诚实 nuance)**: frozen **DINOv2** occluded +0.0206(更 bursty, 前提成立); raw **ImageNet ViT**(TransReID 初始权重) occluded **−0.0040**(不更 bursty, 前提不成立)。→ "ReID 训练是否诱导遮挡-burstiness"正是 e120 训练模型诊断要定的, 两种结果都 informative。略降信心但不致命(两 frozen 都不是训练后的实际模型)。e120 决定性。

### [⛔ burstiness e120 判决 = 双判据 KILL(干净、稳健、双确认) + 强 meta-finding]
弱 baseline TransReID e120(mAP 53.5)训练模型上跑真实 burstiness 探针:
| descriptor | mAP | R1 |
|---|---|---|
| cls(训练检索特征) | **53.53** | 60.59 |
| uniform_patch | 43.14 | 48.28 |
| burst_patch | 42.85 | 47.96 |
| cls+burst | 49.81 | 55.79 |
| partmaxsim_uniform | 42.80 | 47.47 |
| partmaxsim_burst | 42.56 | 47.24 |

- **PRIMARY pooled: burst−uniform = −0.29 → KILL**; **part-MaxSim(防误杀保险): burst−uniform = −0.25 → KILL**(两判据一致, 排除"pooled 误杀"). cls+burst(49.81) < cls(53.53) → 加 burst 反**伤**训练特征(−3.73)。
- **前提在训练模型上 FAIL(关键)**: query(遮挡) intra-sim **0.6890** vs gallery(整体) **0.7044 = −0.0154**(遮挡图**更不** bursty)。前提在 frozen DINO 上成立(+0.0206)、训练后**翻负**。
- **⭐ Meta-finding(强、可写诊断论文)**: burstiness 前提 frozen-promising(DINO +0.0206)但 **trained-absorbed**(TransReID −0.0154)——**即便弱 baseline(53 mAP, 远未饱和), ReID 训练已隐式吸收遮挡-burstiness 结构, 显式 burst 降权无可回收**。这把"in-domain 特征机制 frozen 看着有戏 / 训练后被吸收"的 pattern 从强 SOTA(backdoor/TopoFR/UCE/FM-import)**推广到弱 baseline**——否证"换弱 baseline 就有 headroom"对这类机制。**收窄: in-domain 特征重加权/对齐/补全这一整类, 在任何训练好的 ReID 模型上(强或弱)都无 headroom。** 关联 [[burstiness-democratic-aggregation-bet]] [[fm-import-occluded-reid-closed]]。
- **下一步**: 不再碰 in-domain 特征机制(数据驱动证负)。转**改目标/改问题**类(非推理期特征重加权)或**跨域/泛化**(训练模型隐式处理不迁移处)。VC-Norm 是唯一在训的"训练端改表征"机制(非推理重加权), 跨域判据待定——但 burstiness 诊断(训练吸收遮挡结构)是对 VC-Norm 的**弱负面前瞻**。

### [exp330 Compositional Occluder Generalization + group-DRO — 双审通过, 双卡训练中]
burstiness 死后调研 agent(带"in-domain 死、frozen 会骗人"教训)Rank-1 过审 bet。**为何逃过训练吸收(结构性)**: held-out (occluder类×部位)组合**按构造从没进训练集**→ 训练模型无可隐式吸收。
- **设计**: 3 occluder类(car/bicycle/person, VOC2012 分割patch)× 3 部位(head/torso/legs=上/中/下三分区, region-based 免pose)=9 cells; 训练见 6, hold out 3 对角(car-legs/bicycle-torso/person-head); ERM(mean CE) vs online group-DRO(7组=6cell+clean, present-group 重归一保 CE 尺度=单变量)。eval=Market 整体 query 施 cell 合成遮挡(按(cell,image)确定性seed, ERM/DRO 同遮挡), clean gallery, per-cell mAP。
- **kill-switch**: 先看 ERM 自己 held-out mean << seen mean(有无组合GAP); 有GAP再看 DRO 是否合上 ≥+1.5。**无frozen步骤, 训练模型判据(frozen 会骗人)。**
- **双审**: Claude 广审(修 3 Critical: 手搓optimizer削弱substrate→复用 make_optimizer/scheduler; DRO单变量隔离缺陷→present-group重归一; AMP API) + Codex 独立审(修 scheduler.get_last_lr崩溃 + eval遮挡未seed不公平 + 空occluder类静默; make_model"High"是Codex误读本地SOLIDER repo的假警报, 实跑hyy vanilla TransReID已验证)→ **两轮 codex approve**。smoke 全验(3iter forward/DRO/backward + eval pipeline, score(64,702)/feat(64,768)确认接口)。
- **训练中**(hyy, commit 8e1f4da): ERM=GPU0, DRO=GPU1, 同seed1234, 60ep(~1h)。ERM e1=DRO e1 loss 完全一致(10.3692 vs 10.3694)=单变量干净。DRO q 权重在动(机制活)。eval e20/40/60。monitor bbu89s7rb。
- VOC2012 自己下到 hyy(2913 seg obj, car156/bicycle119/person536 occluders); market1501 symlink 进 ROOT_DIR。

### [⛔ exp330 NO-GO: ERM 零组合 GAP(+0.10) + DRO 训练塌缩(0.26 mAP)]
**e20 kill-switch 判决**：
- **ERM e20**: SEEN mean mAP **35.69** | HELDOUT mean mAP **35.60** | **GAP=+0.10 ≈ 0**。→ **无组合 GAP**：训练见 6 cell 的模型在 3 个 held-out (occluder类×部位)组合上**和 seen 一样好**。eval 对遮挡敏感(各 cell ~35 vs 推测 clean 更高)但**对组合(哪个 cell)不敏感**→ 组合无结构。
- **机制含义(干净 finding)**：occluded ReID 模型**不学 occluder-class 捷径**——它聚焦可见人体证据、对遮挡物外观鲁棒，所以 held-out 组合不难。**正是 Claude reviewer 预言的失败模式**(region 放置→组合塌缩成 region-only，所有 region 都见过→held-out 不难)。group-DRO 无 gap 可合 → NO-GO。
- **DRO e20**: mAP **0.26**(near-random)= **训练塌缩**。q 跑飞到单组(car-head, e7 已 0.71→后 ~0.9+)→模型只训一个 cell→退化。次要 finding：我的 group-DRO(present-group 重归一 + eta=0.01)**不稳、q runaway**(未来用 group-DRO 需降 eta + 正则)。但 ERM 零 gap 已独立判 NO-GO，DRO 塌缩 moot。
- **KILL exp330**。NO-GO 与 in-domain 死法**不同类**(问题重定义/非特征后处理)、**不同原因**(无组合结构/非训练吸收)。扩展诊断：**连"组合泛化重定义"在 occluded ReID 也无 headroom——模型已组合鲁棒。** ERM 继续到 e40 确认 gap 稳定~0(便宜二次确认)。
- 双审 + smoke 全过、单变量干净(ERM e1=DRO e1 loss 一致)——**kill-switch 设计本身成功**：cheaply 在训练判据(非 frozen)上判死，省全量方法投入。

### [跨域 probe: 弱 baseline Occ-Duke e120 → Occluded-REID = 不塌缩(无跨域 headroom)]
GPU1 lean probe(载 e120 ckpt 直接评 Occluded-REID, query=occluded_body/gallery=whole_body, 200 ids)：**mAP 74.79 / R1 80.60**。比 in-domain Occ-Duke 53.5 **还高**(Occ-REID 是更易 benchmark)。→ **弱 baseline 跨域不塌缩、迁移良好**。跨域(至少 Occ-Duke→Occ-REID)**不是 headroom 方向**(模型已迁移好, 只是处处弱于 SOTA = capacity 差非跨域塌缩)。**又一干净负面, 进一步收窄。** 累积排除: in-domain 特征机制(吸收)/ 组合重定义(已鲁棒)/ 跨域(不塌缩)。剩: 新监督(DUL)/ 开集 / 重量级 import。

### [Bet A 几何验证 re-ranker → KILL (−2.26), 诊断: 信号在内容非几何 (第8个 NO-GO)]
调研 agent 第三轮 headline = **吸收陷阱**(输出是单图像素函数+联合优化的机制都被 backbone 内化吸收→4 死类全是其实例)。唯一结构性逃逸 Bet A = 冻结 ViT token 上几何验证 re-rank(非可微 inlier 计数, 无梯度可吸收)。**跑完: baseline cls 53.53 → 几何重排 51.27 = −2.26(几何伤)。KILL。** 诊断: occluded ReID 判别信号在**可见 patch 内容、非几何**(人非刚体/平面, 空间一致性噪声毁强内容排序)。连逃出吸收的机制也败=从新角度坐实吸收陷阱。Bet B(conformal 决策层)未跑(CPU, 不涨 mAP, reliability 重定义, 留诊断)。
**今晚结账: 8 bet 全 NO-GO + 吸收陷阱定理式结论 + 完整避坑地图。唯一活线 VC-Norm 跨域(~8h 后, 慢)。诚实: occluded-ReID 搬机制提 mAP 这件事近乎关闭, 真实交付=诊断/analysis 论文。**
