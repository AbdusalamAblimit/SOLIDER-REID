# 实验 exp374：PSG 图像—姿态对应依赖门禁

## 当前状态

- 阶段：`REMOTE_FULL_PREFLIGHT_PASS_PREPARE_ONLY`
- 训练：未启动
- 正式评测：未启动
- 代码实现：audit-only runner、协议层、模型三态 seam 已编写；20 个 formal preflight
  tests 与原 85 个纯 CPU/synthetic regression tests 全部 PASS
- GPU：未占用
- 当前执行许可：只允许在外层独占锁、前置可用空间至少 100 GiB 的条件下运行一次
  `prepare`，并在 `PREPARED_ONLY` 后强制停住；prepare 可按冻结协议加载 checkpoint、
  解析历史 flat parity 指标、读取 RGB/pose 资产并用 GPU 构造 matching，但仍禁止
  Gate A `run`、当前 arm/per-query 指标生成、`summarize` 或训练

## 动机

当前 PSG 的性能价值与方法新颖性必须分开判断。

旧协议中，三组 seed 的 no-PSG 与 PSG 均值分别为 `56.50` 和 `57.83 mAP`，
说明 PSG 在当时设置下具有稳定正向趋势；但是 WACV 2020 的 pose-guided gated
fusion、SFT 和 FiLM 已覆盖 backbone 中层的 pose-conditioned spatial/channel
调制。因此，原始

\[
Y=X\odot(1+E(H))
\]

不能继续承担论文主创新。

在设计更复杂的新算子前，必须先回答一个更基础的问题：PSG 的收益是否真的依赖
**当前图像与其正确实例姿态的对应关系**，还是来自额外参数、固定人体位置模板、
训练正则或测试分布外干预造成的假象。exp371 在 LGPA 上得到的
`correct-shuffled=+0.0320 mAP` 不能替代这个检查，因为它干预的是 LGPA，而不是
纯 PSG checkpoint。

## 核心假设

若 PSG 确实利用实例姿态，固定同一 RGB、同一 checkpoint 和同一评测流程后，
正确姿态应稳定优于：

1. 协变量匹配但身份错误的完整姿态；
2. 真正跳过 PSG 的 bypass。

训练集导出的 scene-channel centroid control 和七个解剖组局部破坏只作 secondary
敏感性分析。特别是，多人 scene 的 max-merged channel 可能含多个峰，整体平移不能
消除峰间相对结构，所以它不能被称为固定 canonical 姿态，也不能触发 GO/NO-GO。

只有该假设在三组 seed 上通过预注册门槛，才允许为 PSG 设计下一阶段干净训练门禁。
冻结 checkpoint 的反事实只能证明“已训练模型依赖对应姿态”，不能单独证明 PSG
相对 no-pose 的训练因果增益，更不能证明未来新机制有效。本文中的 Gate A 结论只称
`matched counterfactual dependency/fuel screen`，不称总体因果效应。

## 两级门禁

### Gate A：历史 checkpoint 冻结燃料筛查

Gate A 不训练，只使用 4090 上现存的三枚 PSG-only checkpoint：

| seed | checkpoint SHA256 | 当前 flat 日志 mAP/R1 |
|---:|---|---:|
| 1234 | `51c37c49537119deb38bce08702fb5a3ea7fc2b4bc251f1b8f4eebd9ddf6ec69` | `58.3 / 68.1` |
| 42 | `174e8f9316f60219cbeca292457bf976e73cc88df6fddf9d83f94a89280d2a75` | `57.5 / 66.7` |
| 2024 | `c525e9c1ba90d896b703f6eca9a117ba1a97cd08fbab02618021bf20efd09f3d` | `58.0 / 68.4` |

这些目录被复用过：nested 日志是另一代结果，现有训练日志没有 execution commit。
因此 Gate A 无论结果多好都只能叫 `LEGACY_FROZEN_FUEL_SCREEN`，不能直接进入论文
主表或作为最终可复现因果证据。

### Gate B：干净配对训练门禁

只有 Gate A 明确 GO 后，才另写训练设计并重新进行完整审查。Gate B 至少要求：

1. exact commit、archive SHA、run manifest 和独立 `OUTPUT_DIR`；
2. 同一 `PoseBackboneModel`、同一初始化、同一 sampler、同一 batch size；
3. `correct-pose train`、`matched-shuffle train` 与 `true-bypass train`；
4. 三组 paired seed；
5. 训练前验证共享参数逐项 hash 一致，避免 PSG 模块初始化消耗 RNG 后改变
   classifier/backbone 初始化；
6. 形成 train intervention × eval intervention 的配对矩阵。

Gate B 的代码、配置和资源方案必须另行 PASS，Gate A 通过本身不授权训练。

## Gate A 干预臂

所有臂固定相同 RGB、checkpoint、resize、descriptor 定义/归一化和距离计算，只改变
送入 PSG 的最终 `scene_heatmaps`；descriptor 数值本身正是干预结果，不可能固定。
历史主口径固定为 no-flip；flip-test 只允许作为 secondary。

### A0 correct

使用数据集中原始 pose bundle，经原 `_prepare_pose` 得到最终 scene heatmap。

在运行任何 control 前，correct-only 必须按以下历史口径做 parity：

- `TEST.FLIP_TEST=False`；
- `TEST.RE_RANKING=False`；
- `TEST.NFC=False`；
- `TEST.POWER_NORM=0`；
- `TEST.NECK_FEAT='before'`；
- `MODEL.POSE_USE_TARGET_HEATMAP=False`；
- `MODEL.POSE_PSG_STAGES=[-1]`。

三个 checkpoint 的 correct mAP/R1 都必须复现各自 flat 日志到打印精度，即四舍五入
到 `0.1 pp` 后完全一致；否则 Gate A 整体 `INVALID`。若补 flip secondary，必须先
固定受控 scene bundle，再同步翻转 RGB 与该 bundle，禁止重新抽 donor。

### A1 matched-shuffled（primary）

query 与 gallery 分别生成 20 份固定、无 fixed point、严格一一的 donor map：

- donor PID 必须不同；
- 禁止 query/gallery 跨 split；
- different path/content 同时约束 RGB path/content SHA 与 pose path/content SHA；
- 完整 final scene heatmap 一起替换，不逐关节拼接；
- person count 必须完全相同；
- camera、framing 与连续 pose summary 只进入 soft cost，不作硬 strata；
- nuisance 只允许使用 pose score、heatmap L1/peak/entropy/support、skeleton bbox
  中心/尺度/aspect、border-touch/crop 程度；
- 禁止使用 ReID embedding、feature norm、AP、排序或任何评测结果挑 donor；
- 所有 mapping 和 RNG seed 必须在读取指标前落盘并哈希。

全量 gallery 不允许构造稠密 `N×N` Hungarian cost。实现应使用分层的稀疏
候选图和 minimum-weight full bipartite matching；若任一分层不存在完美匹配，
协议标记 `INVALID`，不得静默退化成随机乱序。

#### 唯一 nuisance 与 cost 定义

对每个 final scene bundle 唯一定义连续向量 `u_i`：

1. 17 维 `scene_scores`；
2. 每个 joint channel 的 `log1p(L1)`、peak、normalized spatial entropy 和 support
   fraction，共 `17 × 4` 维；support 定义为大于该 channel `0.10 × peak` 的像素比例；
3. scene support-union bbox 的 normalized `cx, cy, log(w/W_hm), log(h/H_hm),
   log(w/h)`、四边 border-touch indicator 和 crop degree，共 10 维；这里 `H_hm,W_hm`
   是 heatmap 高宽，crop degree 是四个 border-touch indicator 的均值。

所有 heatmap 值必须非负；零 channel 的 L1、peak、entropy、support 均定义为 0；
normalized entropy 唯一定义为

\[
-\sum_p \pi_p\log(\pi_p)/\log(H_{hm}W_{hm}),\qquad
\pi_p=H_p/\sum_pH_p.
\]

任一非有限值使对应 split 整体 `INVALID`。每一连续维在 query/gallery 各自 split 内用
`z=(u-median)/(1.4826×MAD)` 标准化，再 winsorize 到 `[-5,5]`；`MAD<1e-8` 的维置 0
并在 manifest 中记录为 constant。定义

\[
d_{cont}(i,j)=\frac{1}{d}\sum_{m=1}^{d}\min(|z_{im}-z_{jm}|,5),
\]

\[
C_{base}(i,j)=d_{cont}(i,j)+0.25\,\mathbb{1}[camera_i\ne camera_j]
+0.25\,\mathbb{1}[frame_i\ne frame_j].
\]

`frame` 是四个 border-touch bit 按 `top,bottom,left,right` 顺序组成的 4-bit 类别；不再
存在其它人工 framing bin 或可调权重。person count 仍是硬约束，不进入 cost。

#### 稀疏 matching 与固定随机化

匹配实现与门槛固定如下：

1. continuous nuisance 在每个 split 内用 median/MAD robust z-score；
2. 硬约束只有 split-local、exact person count、different PID、different path/content、
   bijection 与 no fixed point；
3. 每个 `split × person_count` 二部图的候选数序列唯一为
   `K(n)=sorted(unique({min(n-1,k): k in [8,16,32,64,128,256]}))`；每个 anchor 实际
   保留按 `C_base` 排序的 `min(k, eligible_count)` 条边。每次扩图先用 Hopcroft-Karp
   检查 `matched=N, unmatched=0`，取第一个存在完美匹配的 k；k=256 仍失败即
   `INVALID`，禁止构造或退化成稠密 `N×N` 图；
4. 第 `r=1..20` 份 mapping 使用固定 seed `374000+r`。在选定稀疏图上定义
   `C_r=C_base+eta_g×Gumbel(seed,edge)`；`eta_g` 在每个已选定的
   `split × person_count` sparse edge set 上单独固定为 `0.25×IQR(C_base)`，若 IQR
   小于 `1e-8` 则固定为 `0.01`。这称 seeded edge perturbation，不称 tie-only
   perturbation。20 个 seed、候选边、`C_base/C_r` 与 mapping 必须先落盘并哈希；
5. 任意两份 donor map 的 Hamming distance 必须在 query maps 和 gallery maps **分别**
   `>=0.90`；否则报告 effective unique count 并将 Gate A 标记 `INVALID`；
6. 每份 mapping 的全体 donor 必须是同 split 的完整排列，因此每个 nuisance 的
   marginal SMD/KS 理论上应为 0；这里仅作 permutation sanity，不当作 pair-quality
   证据，浮点容差固定为 `1e-10`；
7. pair quality 唯一用未加 Gumbel 的 `C_base` 审计：每个连续维的 paired
   `median(|z_i-z_j|)` 必须 `<=0.50`，每份 mapping 的 `P95(C_base)<=1.25`；mapping
   mean cost 还必须不高于同一稀疏候选图和硬约束下 1,000 个随机化 full matching
   的 median mean cost 的 `0.75` 倍。baseline seed 唯一定义为 `475000+b`，
   `b=0..999`。baseline 不优化 cost：在同一稀疏候选图上，以 PCG64DXSM(seed)
   分别打乱 anchor 顺序和各 anchor adjacency，再用固定 Hopcroft-Karp augmenting
   order 求 full matching；任一 baseline 非 full matching 即实现错误并 fail closed。
   任一 pair-quality 门槛失败即 `INVALID`；
8. final scene-merged heatmap 另保存四个 report-only summaries：
   `total_L1=sum_j L1_j`、`mean_confidence=mean_j scene_score_j`、
   `visible_joint_count=sum_j 1[L1_j>1e-8 and peak_j>1e-6]`、
   `scene_entropy=mean_j channel_entropy_j`。它们只用于透明报告，不扩展 95 维 `u_i`，
   不进入 `d_cont/C_base` 或额外 `INVALID` threshold；禁止实现者二次选择。

若任一 scene 的 support union 为空，95 维 bbox 项没有定义，对应 matching split 必须
直接 `INVALID`，禁止把 bbox 项零填或换用整图 bbox。

所有 arms 必须使用同一 query 集合；禁止删除难匹配样本或在结果后放宽门槛。

20 份 query/gallery mappings 在三枚 checkpoint 间完全复用，mapping index `r` 唯一
绑定 `(query_map_r, gallery_map_r)`，不得交叉重配。候选 edge 先按
`(C_base, donor_path)` 排序；Gumbel 使用 NumPy `Generator(PCG64DXSM(seed))`，按
`(anchor_path, donor_path)` 词典序消费随机数；assignment cost 最后加
`1e-12 × edge_lexicographic_rank/(E+1)` 作确定性 tie-break。solver 名称、版本和输出
mapping SHA 必须进 manifest。

#### 实际 PSG 输入上的干预强度门禁

禁止仅审计原始 heatmap。只读 forward hook 必须捕获两个 PSG block 在 encoder 前实际
消费的 float32、同-device tensor；其语义必须与
`S(H)=sigmoid(F.interpolate(H, block_hw, mode='bilinear', align_corners=False))` bitwise
一致。在读取 ReID 指标前计算：

\[
D_{rel}(i,j)=\frac{\|S(H_i)-S(H_j)\|_1}
{0.5(\|S(H_i)-0.5\|_1+\|S(H_j)-0.5\|_1)+10^{-12}}.
\]

另对 `R(H)=S(H)-0.5` 的 17 个 channel 计算 centroid displacement，并除以 block 网格
对角线。每个 block/channel 唯一以 `mass=sum R >1e-8` 且 `peak(R)>1e-6` 判为有效，
centroid 权重为 `R/mass`；两边有效则用 centroid 欧氏距离，仅一边有效则记为 1，
两边均无响应则记为 0，最后对 17 channels 等权平均。每份 query map 与 gallery map
分别要求：全部 sample
tensor content SHA 不同、`median(D_rel)>=0.10`、`P10(D_rel)>=0.01`、median normalized
centroid displacement `>=0.03`。两个 block 若 shape 相同，审计值必须逐值一致；若不同
则分别过门槛。失败称 `INVALID_WEAK_INTERVENTION`，禁止据此作 NO-GO。

### A2 train-derived scene-channel centroid control（secondary）

该 control 直接作用于 PSG 真正接收的 final scene heatmap，而不在 person-level
处理后重新 max-merge，从而避免把 nonlinear owner/overlap 变化混进 treatment。
但它只平移每个 max-merged channel 的整体质心；多人 scene 的峰间相对结构仍被保留。
因此它只称 centroid control，不称 canonical pose，且不进入 primary contrast、
`theta_min` 或 GO/NO-GO。

唯一算法固定如下：

1. 用训练集 deterministic、无增强的原 `_prepare_pose` scene heatmap 拟合；
   训练集与 query/gallery 的 RGB path、RGB content SHA、pose path 和 pose content SHA
   必须完全无交集，否则该 secondary arm `INVALID`；
2. joint channel 有效条件始终统一为 L1 `>1e-8` 且 peak `>1e-6`；scene bbox 定义为所有
   有效 channels 中大于各自 `0.10 × peak` 的 support union；有效 channel 少于 2 个
   时：0 个有效 channel 必须输出原 all-zero scene；恰有 1 个有效 channel 则以该
   channel support 作为 scene bbox，并按该 joint 的训练集 median 平移；任何非零但
   不满足有效谓词的 weak channel、空 union、或缺失训练 median 都使整个 arm `INVALID`；
3. 对每个有效 channel 计算 heatmap centroid，在 scene bbox 中归一化；训练集
   centroid target 是该 joint 有效样本 normalized centroid 的逐坐标 median；
4. 测试时以测试 scene 自身 bbox 恢复目标 centroid，只对原 channel 做 zero-padded
   integer translation；整数取整固定为 half-away-from-zero。禁止 wrap-around、插值
   变形或重新生成 Gaussian；target 与实际输出 centroid 的误差在平移和裁剪后计算，
   必须 `<=0.75` heatmap pixel；
5. 该操作只平移 anchor 自身 channel，因此原 L1、peak、shape 和零通道应保持；任何
   边界裁剪都由下述门禁 fail closed，不能逐样本修补或删除。

对 correct 中不满足统一有效谓词的 channel，只有原 all-zero channel 可保持全零；
其它情况已按上文 fail closed。所有有效 channel 要求：

- 100% 数值 finite；
- 100% 的 sample-channel L1 ratio 位于 `[0.95,1.05]`；
- 100% 的 sample-channel peak ratio 位于 `[0.95,1.05]`；
- normalized spatial entropy 绝对差 100% 不超过 `0.01`；
- 不允许删除违规样本；任一比例门槛失败，整个 centroid arm `INVALID`。

这些审计直接发生在 final PSG input 上。若边界导致任何非零 channel 超门槛，整个
secondary centroid arm `INVALID`，但不使 primary Gate A 失效；只能报告该 secondary
control 不可用，禁止据此修改 primary 结论。

### A3 true bypass

保持同一个 `PoseBackboneModel` 和同一 checkpoint，向 forward 传
`pose_dict=None`。此时 `scene_heatmaps=None`，两个 PSG block 的 guard 均跳过。

以下方式明确禁止解释为 bypass：

- `MODEL.POSE_ENABLED=False`：会切换成另一个模型类；
- `heatmap=0`：PSG 内部 `sigmoid(0)=0.5`；
- 现有 `POSE_DROPOUT_P`：同样只是 zero-response 输入。

### Audit-only final-scene override 入口

实现必须给 `PoseBackboneModel.forward` 增加默认关闭、显式三态的 audit-only keyword：

- `UNSET`：保持现有 `pose_dict -> _prepare_pose` 完整路径；
- tensor：跳过 `_prepare_pose`，把该 tensor 作为唯一 `scene_heatmaps` 输入；
- explicit `None`：true bypass。

A0 correct 也必须先由原 `_prepare_pose` 离线生成 final scene tensor，再通过 tensor
override 进入模型；A1/A2/A4–A10 只改同一类 tensor；A3 使用 explicit `None`。禁止把
final scene 伪装成单人 `pose_dict`。运行时必须断言：eval mode、恰有两个 PSG block、
`POSE_PSG_STAGES=[-1]`、descriptor 为 768 维、PAA/LGPA/GCN/PPA/VCSR/PBSR/part branch/
pose prompt/pose patch embedding 等其它 pose 机制全部关闭。A0 必须在全部 arms 前后各
运行一次，两次 descriptor SHA 和 mAP/R1 必须完全一致。每枚 checkpoint 必须 strict
state-dict load，任何 missing/unexpected key 均 `INVALID`；同一 batch 的 normal UNSET
forward 与离线 correct tensor override 的 descriptor 必须 bitwise identical，先过该
seam parity 才允许生成 control。

### A4–A10 解剖组局部破坏敏感性（secondary）

预先固定七个左右对称解剖组：

1. head：nose/eyes/ears；
2. shoulder；
3. elbow；
4. wrist；
5. hip；
6. knee；
7. ankle。

每次只把一个组的 heatmap channels 替换为 matched donor 的对应 channels，其余
channels 保持 correct。每组使用与 primary shuffle 完全相同的 20 个 mapping index，
先对 mapping 等权平均，再报告全部七组。

该操作会破坏 recipient 相邻关节之间的骨长和拓扑一致性，因此只能称
`matched-donor local-channel corruption sensitivity`，不称 joint drop、关节效用或
因果移除。它不进入 GO/NO-GO 硬条件；即使显著下降，也只能说明 checkpoint 对该类
局部 OOD 破坏敏感。禁止用零通道表示“移除”，也禁止看完结果后只挑最好的一组。

### Secondary controls

- 项目现有的无协变量 wrong-PID bijection 只作 secondary；
- zero-response 只作 sigmoid 语义诊断；
- query-only 干预作敏感性分析，主结果仍对 query/gallery 都施加固定干预；
- exp007 只有 Stage 3 的两个 block，block leave-one-out 只能作 secondary，不能
  宣称跨 stage 规律。

## 指标与统计

### 原始输出

每个 seed、每个臂必须保存：

- 完整 mAP、R1、R5、R10；
- 每 query AP；
- 每 query R1 indicator；
- 每 query retrieval margin（最近负样本距离减最近正样本距离）；
- descriptor、distance matrix、donor map、centroid 参数与运行 manifest 的 SHA256；
- query/gallery 文件路径和内容哈希无交叉的断言结果。

retrieval margin 必须在 official junk removal 后定义为
`min(valid negative distance) - min(valid positive distance)`；不存在 valid positive 或
negative 的 query 使该 arm `INVALID`。descriptor 与 distance matrix 逐臂生成、校验、
计算 per-query 结果并记录 SHA/shape/dtype 后删除，不作长期持久化；长期保存的是输入、
mapping、per-query 输出、汇总和上述内容哈希。

### Fail-closed manifest、恢复与资源

`execution_sha` 唯一定义为不含输出路径和任何结果的 frozen pre-metric manifest 的
SHA256。新 execution 必须以 `mkdir(exist_ok=False)` create-exclusive 地创建
`/home/afr/exp374_artifacts/gate_a_<execution_sha>`；普通启动遇到已存在目录即拒绝，
禁止覆盖。只有显式 `--resume <exact_execution_dir>` 可打开已有目录，并且必须满足：
存在 frozen manifest、没有 execution `COMPLETE` marker、所有输入 SHA 完全一致；已
atomic-published 的 arm 只读复用，残留临时 arm 目录必须删除后从该 arm 重做，禁止
覆盖已发布 arm。manifest 在读取任何 ReID 指标前冻结并至少包含：

1. audit code commit、dirty diff SHA 或 clean archive SHA；
2. 三枚 checkpoint、config、flat log、train log 的路径与 SHA；
3. RGB 顺序与内容 SHA、pose index/NPZ SHA、query/gallery 顺序、`num_query`；
4. nuisance scaler、constant dims、cost 公式版本、k、Gumbel scale、20 mapping seeds、
   1,000 baseline seeds、candidate edges、mappings 和 centroid 参数 SHA；
5. Python/PyTorch/CUDA/cuDNN/GPU、determinism flags 与 package lock；
6. checkpoint 中 PSG canonical keys 与兼容 alias keys 的 shape、内容 SHA 和逐值一致性。

每个 arm 只可写临时目录，全部文件 fsync、SHA 与断言 PASS 后 atomic rename；失败保留
失败 manifest 但不得发布半成品。恢复前必须逐项重算上述 SHA，任何路径、顺序、shape、
NaN/Inf、版本或 hash 不一致都拒绝恢复并把 execution 标成 `INVALID`。每个 arm 后必须
释放 descriptor/distance/GPU cache，且任一时刻只运行一个 arm、一个 seed、一个评测
进程。

按 `2 correct（全臂前后各一次）+ 20 shuffle + 1 centroid + 1 bypass + 7×20 group`
估算为每 seed 164 passes、三 seed 492 passes，历史 no-flip 速度约需 4.25–4.5 小时。
Secondary controls 中的 unmatched wrong-PID、query-only、zero、flip 和 block-LOO 不在
这 492 次核心执行内；若以后执行，必须另做设计与资源复审。4090 根卷当前
只读预审约有 217 GB 可用；由于大矩阵只保留 hash，启动前仍要求目标卷可用空间至少
80 GB，低于门槛直接拒绝。任何实现若改为持久化全部 descriptor/distmat，门槛必须
提高到 150 GB 并重新做资源审查。

### Primary estimands

所有 AP/R1 差值以 percentage point（pp）报告。对 seed `s`、query `q`、第 `r`
份 mapping，先唯一地定义：

\[
AP_{shuffle}(s,q)=\frac{1}{20}\sum_{r=1}^{20} AP(s,q,r).
\]

禁止先平均 descriptor 或 distance matrix 再计算 AP。20 份 mapping 是预注册的固定
Monte-Carlo nuisance ensemble，不作为 20 个独立数据集扩充样本量，也不在 primary
bootstrap 中重采样。另报告 leave-one-mapping-out sensitivity 与 Monte-Carlo SE。
R1 同样先按 query 定义
`R1_shuffle(s,q)=mean_r R1_indicator(s,q,r)`，再进入 seed contrast；禁止先对 20 次
整体 R1 求均值后伪装成 query-level 数据。per-query Monte-Carlo SE 定义为 20 个 mapping
值的 sample SD（`ddof=1`）除以 `sqrt(20)`，并汇总其 median/P95。leave-one-mapping-out
必须报告删去每个 mapping 后两个 primary `theta_c` 的 min/max，不据此删除 mapping。

对两个 primary controls

\[
c\in\{shuffle,bypass\}
\]

分别定义配对 contrast：

\[
\theta_{s,c}=100\times\operatorname{mean}_{q}
[AP_{correct}(s,q)-AP_c(s,q)],
\]

\[
\theta_c=\frac{1}{3}\sum_s\theta_{s,c},\qquad
\theta_{min}=\min_c\theta_c,
\]

以及每 seed 的

\[
\theta_{min,s}=\min_c\theta_{s,c}.
\]

`theta_min` 只作为保守点估计；不对 `correct-max(control)` 做普通 percentile CI。
两个 `theta_c` 必须始终单列报告。centroid control 只作为 secondary 单列，不进入
`theta_min`。

Gate B 另定义：

\[
\Delta_{train}=mAP(correct\text{-}pose\ train)-mAP(true\text{-}bypass\ train).
\]

### 固定重采样协议

三个 seed 只有三个，固定为 paired blocks，不对 seed 做 bootstrap，也不声称泛化到
随机 seed 总体。执行 10,000 次 one-sided 95% PID-cluster bootstrap。唯一 bootstrap
RNG 为 NumPy `Generator(PCG64DXSM(374900))`；PID 按数值升序形成抽样 universe，
quantile 固定使用 NumPy `method='higher'`：

1. 每个 replicate 只对 query PID clusters 有放回抽样；抽中 PID 时保留其全部 query；
2. 同一组 PID multiplicities 同步应用到三个 seed、全部 arms 和全部 20 mappings；
3. 每个 replicate 内先计算各 seed contrast，再对三个固定 seed 等权平均；
4. gallery、checkpoint 与 20 份 donor mapping 固定。

对两个 mAP contrasts 使用同一 bootstrap replicate 构造 one-sided simultaneous
max-deviation intervals：

\[
q_L=\max\left(0,Q_{0.95}\left(\max_c[\theta_c-\theta_c^{(b)}]\right)\right),\quad
LCB_c=\theta_c-q_L,
\]

\[
q_U=\max\left(0,Q_{0.95}\left(\max_c[\theta_c^{(b)}-\theta_c]\right)\right),\quad
UCB_c=\theta_c+q_U.
\]

R1 indicator 对两个 controls 单独定义 `theta^R1_c`，用同样方法构造另一个两对照
simultaneous interval family。不得先按 mAP 选择 control 再检查 R1。

七个解剖组只作 secondary sensitivity；使用固定 20 mappings 和相同 PID replicate，
可报告 one-sided seven-group simultaneous max-deviation intervals，但不触发 Gate A GO。

上述区间只表示：给定 official gallery、checkpoint 和固定 donor mappings 时，对 query
identity 采样的条件不确定性。gallery 干预会共同影响全部 query，当前 bootstrap 不支持
gallery 总体 ATE 或一般 SUTVA 因果解释。

## 预注册决策

### Gate A GO

必须全部满足：

1. 三个固定 seed 的 `theta_min,s` 均大于 0；
2. `theta_min >= +0.30 pp`；
3. 两个 mAP contrasts 的 one-sided 95% simultaneous `LCB_c` 全部大于 0；
4. 两个 R1 contrasts 的 one-sided 95% simultaneous LCB 全部大于 `-0.50 pp`；
5. donor、路径哈希、协变量匹配和能量审计全部 PASS。secondary centroid arm 即使
   `INVALID` 也必须原样报告，但不改变 primary GO/NO-GO。

Gate A GO 只授权 Gate B 的干净配对训练设计与审查，不授权新机制训练。

### Gate A NO-GO

满足任一项即停止 PSG 自有化：

- 至少两个 seed 的 `theta_min,s <= 0`；
- 任一 primary mAP contrast 的 simultaneous `UCB_c < +0.30 pp`，使 GO 的最小
  实用效应不可能成立；
- 任一 primary `theta_c <= -0.30 pp`，且同一 simultaneous family 的 `UCB_c < 0`。
  这唯一表示 control 显著优于 correct；不再另造未定义的“反向 LCB”。

### INVALID / INCONCLUSIVE

- donor map 非双射、跨 split、PID 碰撞、路径/内容交叉、匹配门槛或能量门槛失败：
  `INVALID`，只允许修协议后重跑；
- 任一 matching 数值门槛失败均使 primary Gate A `INVALID`；secondary centroid arm
  数值门槛失败只使该 arm `INVALID`。禁止逐样本删除，所有可比较 arms 使用同一
  query 集合；
- 其余未达到 GO/NO-GO 的灰区：`INCONCLUSIVE`，不得直接开大训练或放宽门槛。

Gate A NO-GO 是本项目停止复杂化 PSG 的管理决策，不构成“所有 pose 方法均无因果
价值”的科学证明。

## Development / confirmation 数据锁

Occluded-Duke official test 已在本项目中被长期用于方法、阈值和 stop-rule 选择，
因此 Gate A/B 的全部结果从一开始就是 development evidence，不能作为最终
confirmatory claim。

在读取 Gate A 指标前固定 `Partial-REID` 为一次性独立确认数据集：只允许在 Gate A
GO、Gate B 完成且方法/超参数/epoch 全部冻结后评测一次。在此之前只允许核对数据
许可、文件完整性和 pose 资产可生成性，不读取任何方法指标。若 Partial-REID 资产或
许可不可获得，则本项目放弃 confirmatory claim，不在看到结果后改选其他数据集。

## 新机制边界

原候选 `Y=X+lambda(T-I)VX` 已被数学红队判 FAIL：它可精确归约为 residual
attention/GAT 或图拉普拉斯扩散；若双随机更新位于末 block 后并立刻 GAP，空间和
保持还会使其对 global descriptor 严格零作用。全分辨率 heatmap 离散 W2 与
batch-average trust region 也因计算量和单关节漂移漏洞判 FAIL。

第三路专项查新进一步确认：即便改成显式 source/demand，已有工作也已分别并在组合
上覆盖 source reliability、visible-to-occluded recovery、pose topology、
confidence-conditioned mixing、UOT 处理缺失部位、ReID 中双随机 OT feature
transform 和 Sinkhorn 双随机 attention。直接邻居包括 HOReID、FRT、PIRT、
HUPOR、RFC、UNITE、SOT 与 Sinkformers。因此，“把 GCN 换成 Sinkhorn”或“把
confidence 换成 posterior entropy”的差分不足，容易被解释为已有模块拼接，仍然
直接 NO-GO。

当前只保留一个待查候选**问题对象**，不保留方法 claim：

> 校准 pose posterior 是否能定义一个不确定度约束可行域，并与 ReID utility 共同决定
> source/demand 边际，在骨架支持上求解显式 source depletion + sink receipt。

即使 Gate GO，该候选仍必须先形成不能被解释为 confidence-weighted Sinkhorn
拼接的联合优化问题，再正面对照 HOReID、FRT、RFC、PGGANet、RTGAT、UNITE、
SOT、普通 GAT、row-stochastic attention、2026 TTPM 与 2026 Pose-Guided
Feature Restoration Transformer。TTPM 正文已核，属于 pose-patch matching +
confidence filtering + texture decoder 的强邻居；没有完成联合公式并排除后一篇
restoration 正文前，
不得启动该机制训练；
不得使用“首次 pose-aware OT”“首次姿态图传播”或“首次从可见区域补偿遮挡区域”
等宽泛表述。

## 风险与失败解释

1. 历史 checkpoint 缺 exact execution provenance，Gate A 只能筛燃料；
2. correct/shuffle/bypass 都是训练后反事实，其中后两者可能 OOD；centroid 与解剖组
   corruption 的 OOD 更强，所以只作 secondary；
3. official Occluded-Duke test 已被本项目长期用于选择机制，当前就是 development
   set；Gate A/B 只能做开发决策，最终 claim 必须遵守上面的 Partial-REID 一次性锁；
4. PSG 当前对 `[0,1]` ViTPose heatmap 重复 sigmoid，限制了可解释性；
5. Gate A 成功不证明任何候选新模块成功，失败则足以停止继续复杂化 PSG。
