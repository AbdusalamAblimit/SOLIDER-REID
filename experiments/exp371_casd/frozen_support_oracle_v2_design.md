# exp371 CASD：Frozen Support Oracle v2 预注册设计

## 状态

> **DESIGN ONLY — REAL CACHE DEFAULTS TO DRY-RUN**

- 第一版 `frozen_support_oracle_design.md` 与其脚本是 rejected prototype，不得用于正式 GO/NO-GO。
- v2 只筛选 frozen support routing geometry；不训练 student，不裁决 MVI²P、UMTS、LCR²S 或 `exp123` 的最终胜负。
- 真实 cache 的 CLI 默认只做 provenance、duplicate、K-fold、camera 与 eligibility dry-run；只有显式 `--execute-frozen-oracle` 才计算 retrieval 指标。
- formal oracle 强制 `max_queries=0`、`cross-camera`、显式 device、预注册 seeds/2000 bootstrap/768 block dim 且 target/canonical/scene 三份 cache 齐全；manifest 与 results 必须记录 device、distance batch、expected block dim、max queries 和全部 seeds。已有 `manifest.json/dry_run.json/results.json` 的 output dir 一律拒绝覆盖。
- 本文件不授权运行真实 cache 或启动训练。实现与 synthetic tests 通过后仍需一次只读审查。
- CASD 暂作工作名。pose-routing 门禁通过前，不能把 `anatomical` 当成已证实属性；中性展开为 **Cross-instance Allocation Support Distillation**。

## 一、为什么重写 Gate C

同一 `exp336` seed-1234 checkpoint 的 frozen 指标为：

| arm | mAP / R1 |
|---|---:|
| global | 58.9908 / 67.3756 |
| correct | 59.8357 / 67.6018 |
| target-only | 59.8121 / 67.5113 |
| canonical | 59.7374 / 67.6471 |
| shuffled | 59.8037 / 67.7376 |
| uniform | 59.3689 / 66.8326 |
| no-pose | 59.4014 / 66.6063 |

这些结果只支持：

1. `global + pooled + 5 slots` 的结构化局部描述子稳定保留约 `+0.8～0.9 mAP`；
2. target-only 与 scene-merged 基本一致；
3. correct、shuffled、canonical 仅差 `0.03～0.10 mAP`，精确当前图姿态不是已成立的主增益；
4. uniform/no-pose 较弱，说明空间结构有作用，但 raw response 尚不是 calibrated visibility；
5. train-only PCA-768 单 seed retention=`1.1158` 允许后续同维公平首验，但不是创新。

第一版 oracle 的致命问题是：

- donor 与 evaluation reference 同取原 gallery；
- 正 endpoint 的 GT PID 决定 query descriptor 如何重构；
- 同一 query 对不同 endpoint 使用不同 descriptor，mAP/R1 不再是标准 retrieval；
- 在同一 val relations 上用 `gain_CASD>0` 生成 hindsight mask；
- `CASD-LIKE` 只用 feature agreement，没有 raw pose response；
- 两 donor 时 `AGREE` 数学上退化为 `PART-EQUAL`。

v2 的硬目标是同时消除 endpoint leakage、val hindsight 和 extraction/routing 混淆。

## 二、prior-art 硬边界

### MVI²P

- [Multi-view Information Integration and Propagation for Occluded Person Re-identification](https://arxiv.org/abs/2311.03828)，Information Fusion 2023/2024，[DOI](https://doi.org/10.1016/j.inffus.2023.102201)，[代码](https://github.com/nengdong96/MVIIP)。
- 已覆盖同 ID 多图互补、CAM/reliability 加权、comprehensive representation、完整 feature→单图 L2、测试单图分支。
- CASD 不能声称首次多图补单图。剩余空间只能是 support 分支排除 anchor、同 slot raw-response routing 与后续 support-vs-self gain transfer。

### UMTS

- [Uncertainty-aware Multi-shot Knowledge Distillation for Image-based Object Re-identification](https://arxiv.org/abs/2001.05197)，AAAI 2020，[DOI](https://doi.org/10.1609/aaai.v34i07.6774)。
- 已覆盖 multi-shot comprehensive teacher→其中一张 single-shot student、多 stage uncertainty KD、单图推理。
- 不能把 multi-shot→single-shot、quality weighting 或 pose-free student 单列贡献。

### LCR²S

- [Learning Comprehensive Representations with Richer Self for Text-to-Image Person Re-Identification](https://arxiv.org/abs/2310.11210)，ACM MM 2023，[DOI](https://doi.org/10.1145/3581783.3611832)。
- 已明确为 current sample 用同 ID 其他视图构造 `support set`，MHAF 融合 current+support，蒸馏 enriched feature 与 relation matrix 给单输入 student。
- 不能声称首次 same-ID support set、other-view support、single-view student 或 relation distillation。

### PVPM / KPR / PAFormer

- [PVPM](https://arxiv.org/abs/2004.00230)（CVPR 2020）已用同 ID 正对的同部位 node/edge matching 自挖 correspondence pseudo-label并训练 part visibility。
- [KPR](https://arxiv.org/abs/2407.18112)（ECCV 2024）及 KPRTrack 已覆盖 keypoint-prompted parts、共同可见部位比较和 tracklet 同部位 moving average。
- [PAFormer](https://arxiv.org/abs/2408.05918) 已覆盖 pose-supervised part tokens、visibility predictor 与 pose-free inference。
- 因此“跨图同部位”“可见部位聚合”“pose-free pose student”均不能单独首创。

### NNCL

- [Neural Network Coding Layer](https://doi.org/10.1109/ACCESS.2025.3610080)（IEEE Access 2025，[代码](https://github.com/quarry0226/NNCL)）已覆盖 fixed/learnable coding matrix、structured redundancy、feature erasure 与 pseudo-inverse recovery。
- AERC 已独立 NO-GO；CASD 失败后不得转 erasure coding 小变体。

若最终能成立，最窄的联合差分只能是：

> support 分支 strict LOO 不含 anchor + target-only raw pose-response 的逐 slot donor allocation + 只在训练身份定义的 support-vs-self 增量关系迁移。

其中任一单项都不能声称首创。

## 三、合法命名

本阶段称：

> **classifier-free, GT-supported episodic support-geometry oracle**

它不用 classifier，但使用 GT PID 构造 support 和计算 positives/negatives，所以不能简称普通 `class-free retrieval`，也不是部署协议。

缓存信号统一叫：

- `raw pose response`：归一化前每 slot 非负响应；
- `pose-response allocation`：同一图五 slot 间相对分配；
- 禁止叫 absolute visibility、occlusion probability 或 calibrated reliability。

## 四、cache 与 provenance

主输入只能是 target-only cache，至少 hard assert：

```text
schema_version == exp371_target_support_cache_v1
mode == target_only_correct
pose_source contains target_person
split == val
features.shape == [N,7,D] or [N,7D]
raw_pose_response.shape == [N,5]
target_person_valid.shape == [N]
block order == [global, pooled, slot1..slot5]
```

还必须绑定：

- PID、camid、ordered normalized path；
- checkpoint/config/extractor/cache SHA；
- raw response、relative allocation 与 tensor SHA；
- `target_person_valid/person_count`；
- 每图 `content_sha256`；
- 若有，tracklet/frame metadata。

当前大 cache 不重写。content SHA 使用 sidecar，sidecar 至少包含：

```text
schema_version
source_cache_path
source_cache_file_sha256
ordered_paths_sha256
sample_count
ordered content_sha256
unique_content_count
duplicate_content_group_count
duplicate_content_sample_count
```

v2 loader 必须同时校验 source cache 文件 SHA、紧凑 JSON 编码的 `json_sha256(ordered normalized paths)`、sample count、unique content count、duplicate group/sample count；任一不一致 hard fail。

canonical/scene cache 若参与，必须与 target cache 在 path、PID、camid、content、num_query、block_dim、checkpoint 和 fold assignment 上逐项一致。

## 五、PID 内 deterministic K-fold

预注册 `K=5`。对原 gallery 每个 PID：

1. 用 `SHA256(split_seed:path)` 排序；
2. round-robin 分到 folds `0..4`；
3. fold `f` 为 evaluation reference `R_f`；
4. 其余 folds 为 support pool `S_f`。

每 fold 必须满足：

```text
S_f ∩ R_f = ∅  （path/content；有元数据时还包括 tracklet）
```

原 query set 始终作 query。每个 `(query, arm, fold)` 只构造一次 descriptor，然后用它对该 fold 全部 reference 排序。禁止 pair-specific endpoint correction，禁止根据 endpoint PID/path 重构 descriptor。

reference 继续执行标准 same-PID/same-camera exclusion；support/reference 隔离不能替代标准 ReID 过滤。

## 六、duplicate、camera 与 eligibility

### duplicate/content

- exact duplicate normalized gallery path 必须为 0；
- path→PID/camid 映射必须唯一；
- content SHA 重复默认 hard fail；唯一白名单是“恰好一个 query + 一个 gallery、PID/CAM 完全相同”的标准 evaluator overlap。该 gallery endpoint 必须被 same-PID/same-CAM 规则排除，若落入 support 还必须按 query content SHA 排除；
- q-q、g-g、超过两成员、异 PID/CAM 的同内容组全部 hard fail；
- content SHA 不得跨 support/reference；
- 有 tracklet/frame 时必须做同 tracklet strict sensitivity；没有则明确“不能回答近重复/同轨相邻帧泄漏”。

### camera

主协议为 **cross-camera support-only**：只保留 `donor_cam != query_cam`。

- 少于 3 个 unique-path support donors 时 query/fold ineligible；
- 禁止 fallback 到 same-camera；
- unrestricted-camera 只能作为 sensitivity。

### donor budget

主协议必须与后续 `P×K, K=4` student 的 support budget 一致。每个
query/fold 从合格 cross-camera pool 中按固定、feature/pose 无关的
`SHA256(seed:query_path:donor_path)` 排序，**恰好选择 3 个 donors**。全部
arms 共享同一三张图，并同时报告选择前的 available donor count。禁止在
frozen oracle 中使用某身份的全部图像，把 multi-shot 上界冒充为训练时可兑现的机制。

### eligibility

每 query/fold 必须：

1. 至少 3 个 unique-path、content-disjoint、cross-camera donors，并从中按预注册 hash 恰好选择 3 个；
2. 至少一个经过标准 exclusion 的 positive reference；
3. target person 有效；
4. feature/raw response finite；
5. support/reference path/content/可用时 tracklet 完全 disjoint。

每 fold eligible query ratio 和 eligible PID ratio 都必须 `>=70%`。同时报告 removed reason、retained/removed SELF AP与距离、donor count、camera composition、active slot count，防止只留下容易 query。

## 七、block 与 common active-slot mask

冻结 block：

```text
F_i = [g_i, p_i, s_i1, s_i2, s_i3, s_i4, s_i5]
```

所有 part arms 保留 query 自身 `g_q/p_q`，只替换五个 individual slots。

预先固定 feature-slot derangement `pi_j` 与 response-slot derangement `rho_j`。common active mask 必须对全部结构 arms 一致：

```text
a_k = 1[sum_j r_jk > eps]
      AND 1[sum_j r_j,rho_j(k) > eps]
eps = 1e-12
```

`a_k=0` 时 `ID-MEAN/PART-EQUAL/SLOT-PERM/AGREE/POSE-SCALAR/POSE-RESP/RESP-PERM` 全部保留 SELF slot。禁止某 arm denominator 为 0 时偷偷 fallback 到 equal mean。

## 八、十二个预注册 arms

### `SELF`

原 query 七块，不使用 support。

### `ID-MEAN`

将所有 donor 的五个 slots 混成 identity-local bag mean，再复制到五 slots。保留 same-ID evidence，删除 slot correspondence。

### `ID-GLOBAL`

只对 donor global blocks 求 identity mean，再复制到五 slots；query global/pooled 仍保持自身。它是 non-local identity support control，用于排除“任意 same-ID global prototype + 扩维”解释，并进入 strongest non-pose control 集。

### `PART-EQUAL`

同 slot donor 等权均值。回答 slot correspondence 本身是否有效。

### `SLOT-PERM`

每 donor 独立 deterministic full derangement feature slots；保持每 donor feature multiset、donor 数、global/pooled与维度。

### `AGREE`

同 slot donor-consensus cosine 加权，至少 3 donors。它是 non-pose reliability control；不能替代主 arm。

### `POSE-SCALAR`

先把每个 donor 的五槽 raw response 求和为单一标量，再把同一个 donor 权重用于全部 slots：

```text
u_j = sum_k r_jk
w_j = u_j / sum_j u_j
slot_k = sum_j w_j s_jk
```

它保留人体尺度、pose detector 总响应或图像质量信号，但删除 response-slot allocation。`POSE-RESP` 必须明确超过它，才能排除 raw response 只是通用 donor-quality scalar 的解释。

### `POSE-RESP`

同 slot features 不动，只在 donor 维归一化 target-only raw pose response：

```text
w_jk = r_jk / sum_j r_jk
slot_k = sum_j w_jk s_jk
```

只称 raw pose-response allocation。

### `RESP-PERM`

features 与 slot correspondence不动，每 donor 独立 derange response slot assignment。保持 response multiset，是 `POSE-RESP` 的直接因果 control。

### `FULL-INCL`

query anchor + support 的完整七块综合 target，标定 MVI²P/UMTS/LCR²S 式 anchor-inclusive geometry boundary。

### `FULL-LOO`

只对 support 的完整七块求 mean，不含 anchor，区分 strict-support 与普通 comprehensive teacher。

`FULL-INCL/FULL-LOO` 只作 geometry boundary，不能在 frozen 阶段声称复现或优于 MVI²P/UMTS/LCR²S。

### `WRONG-ID`

从其他 PID 的 support pool 中按固定 hash 选择与 correct support 数量相同的 donors，用同 slot equal mean 替换 active slots。它只作 fail-safe corruption diagnostic：必须弱于 `POSE-RESP`，但不进入 strongest non-pose control max，也不被包装成方法消融。

## 九、`E × R` extraction-routing 正交矩阵

必须报告：

| extraction `E` | `R=EQUAL` | `R=POSE-SCALAR` | `R=POSE-RESP` | `R=RESP-PERM` |
|---|---|---|---|---|
| target-only correct blocks | `PART-EQUAL` | donor-quality scalar control | 主 routing | response-slot control |
| canonical blocks | `PART-EQUAL` | target total-response control | target raw-response routing | response-slot control |

要求：

1. 同一 extraction 行内四个 arms 的 feature tensors 逐 bit 相同，只改变 donor weights；
2. canonical 行仍复用 paired target-only cache 的 raw response；
3. 两行 metadata、fold、eligibility 与 common mask 完全相同；
4. 不从 canonical 行挑 threshold/arm；
5. scene-merged 是 paired sensitivity，不替代 `2×4` 矩阵。

这张矩阵只区分“提取到什么”和“如何路由”，不单独触发最终 GO。

## 十、指标与统计

每 arm/fold：

- mAP、R1、R5、R10；
- positive/negative distance、class-balanced margin；
- per-query AP 与相对 SELF gain；
- eligible query/PID、donor/active-slot/camera/duplicate统计。

汇总：

- equal-fold mean、min、std；
- 不能按 query 数加权隐藏低覆盖 fold；
- PID-grouped bootstrap 95% CI，PID 为重采样单位；
- bootstrap seed/replicates 在运行前冻结。

禁止：

- val threshold/mask/arm/temperature selection；
- `gain_POSE_RESP>0` hindsight mask；
- own-arm advantage weighting 后自证；
- classifier CE/accuracy 自证；
- frozen geometry 阶段裁决 advantage selector vs `exp123`；
- 把 scalar `sum(weights)=1` 冒充实际 loss/gradient matched。

## 十一、frozen routing GO 门禁

target-only 主协议必须全部满足：

1. `POSE-RESP - max(ID-GLOBAL, ID-MEAN, PART-EQUAL, SLOT-PERM, AGREE, POSE-SCALAR, RESP-PERM) >=0.5 pp` equal-fold mean；
2. 每个 fold 都必须对该 fold 自己的最强 non-pose control 为正，不能只对“全局均值最强的某一个 control”为正；
3. 对七个 controls 分别做 paired PID-grouped bootstrap，所有 95% CI lower 都必须 `>0`；
4. `PART-EQUAL - SLOT-PERM >=0.3 pp`；
5. `POSE-RESP - PART-EQUAL >=0.3 pp`；
6. `POSE-RESP - POSE-SCALAR >=0.3 pp`；
7. `POSE-RESP - RESP-PERM >=0.5 pp`；
8. 每 fold eligible query/PID ratios 均 `>=70%`；
9. retained/removed SELF 难度已报告；
10. target-only 与 scene-merged 关键方向不冲突；scene 必须对完整七个 controls 逐 fold 比较，不能只看 equal/response-perm；
11. path/content leakage 为 0；有 tracklet 时 strict sensitivity 同向；
12. canonical extraction 的 `2×4` routing matrix 完整落盘；
13. `WRONG-ID` fail-safe 弱于 `POSE-RESP`，且 retained/removed SELF 难度或不可评分原因完整落盘。

出现下列任一情况立即 NO-GO，不扫温度、queue、slot 数或 response threshold：

- `ID-MEAN` 最强：只是 same-ID prototype/denoising；
- `PART-EQUAL≈SLOT-PERM`：slot correspondence 无价值；
- `POSE-RESP≈PART-EQUAL`：raw response allocation 无价值；
- `POSE-RESP≈POSE-SCALAR`：收益只来自人体尺度或通用 donor quality，不是逐 slot allocation；
- `POSE-RESP≈RESP-PERM`：response-slot 对应无价值；
- `AGREE>=POSE-RESP`：non-pose consensus 已解释 routing；
- 某 fold 方向反转或 coverage 不过；
- 只有 scene-merged 成立；
- duplicate/content/tracklet leakage 无法消除。

这里的 `all_pass` 只能叫 **frozen routing-screen GO**。此前“frozen Gate C 必须超过 full-feature/full-relation control”的表述正式撤回：`FULL-INCL/FULL-LOO` 改变完整七块 target geometry，LCR²S/exp123 relation control 还涉及实际 student loss 与梯度，不能用 teacher-side frozen mAP 诚实裁决。它们全部迁移到 matched student 强门；frozen 只报告 boundary 数值且明确 `enters_routing_gate=false`。

## 十二、后续 matched frozen-student 矩阵

只有 frozen routing 全过，另写 student design/review 后才可执行：

| Arm | 含义 |
|---|---|
| `B0` | 无 KD |
| `KD0` | same-image LGPA teacher |
| `ID0-G / ID0-M` | strict-LOO identity-only support；分别对应 `ID-GLOBAL / ID-MEAN`，两臂都运行，不在 val 上挑一个 |
| `P0-S / P0-R` | strict-LOO permutation controls；分别破坏 feature slot / response slot correspondence，保持其余 supervision protocol |
| `PS0` | strict-LOO `POSE-SCALAR`；排除总 pose-response/通用 donor-quality scalar |
| `R0` | 与 CASD 使用完全相同的 strict-LOO `POSE-RESP` target，但不做 support-gain selector；单独归因 selector |
| `AI-ADV` | 与 CASD 相同 routing/transfer，但 target 显式包含 anchor；与 strict LOO 做 matched 对照 |
| `MV-INCL` | anchor-inclusive multi-view full-feature target；覆盖 MVI²P/UMTS feature KD 边界 |
| `LR-INCL` | current+support full feature + inter-sample relation KD；显式覆盖 LCR²S 边界 |
| `LR-LOO` | strict-support full feature + relation KD，不含 anchor、不做 gain selector |
| `EXP123` | strict-support full relational target，不做 advantage filtering；内部强对照 |
| `CASD` | strict-LOO `POSE-RESP` support；support-gain selector 只在训练身份上定义 |

除上述逐臂强对照外，还必须冻结 `routing × transfer` 的 `2×2` 因子矩阵：

| | full relation transfer | support-vs-self increment transfer |
|---|---|---|
| `PART-EQUAL` | `PE-FULL` | `PE-ADV` |
| `POSE-RESP` | `PR-FULL` | `PR-ADV`（CASD） |

三 seed 交互项 `PR-ADV - PR-FULL - PE-ADV + PE-FULL` 的 mean 必须 `>=0.3 mAP`，且 seed-paired 95% CI lower `>0`。否则 routing 与增量迁移只是两个可替换的已知组件，不能作为联合机制贡献。

公平要求：

- 相同 frozen features、student 初始化/输出维度、优化步数、data order；
- 相同主任务 loss；
- 审计 actual supervision mass、per-query mass、positive/negative mass、effective sample size、max weight 与梯度 norm；
- val/test 不参与 selector；
- 测试 student 只输入 RGB，不读 pose/support/PID/gallery context。

单 seed provisional gate：

```text
CASD - B0 >= 0.8 mAP
CASD - max(KD0, ID0-G, ID0-M, P0-S, P0-R, PS0, R0, AI-ADV,
           MV-INCL, LR-INCL, LR-LOO, EXP123,
           PE-FULL, PE-ADV, PR-FULL) >= 0.5 mAP
```

最终必须三 seed paired mean 为正、每 seed 同向，不能由单 seed 驱动。`R0` 用于证明收益不是普通 POSE-RESP target，`PS0` 排除 pose 总质量，`AI-ADV` 隔离 strict LOO，`2×2` 交互证明 routing 与 increment transfer 的联合不可替换；`ID0/P0` 用于证明不是 identity prototype 或 permutation-insensitive support。只有 student 再超过 `MV-INCL/LR-INCL/LR-LOO/EXP123`，才能讨论跨过 MVI²P/UMTS/LCR²S/exp123 边界。

## 十三、执行顺序

1. metadata-only feasibility 与 content sidecar 绑定；
2. target/canonical/scene provenance 和 paired metadata；
3. deterministic synthetic tests：fold disjoint、唯一 descriptor、derangement、multiset、inactive SELF、camera no-fallback、sidecar SHA；
4. 真实 cache 默认 dry-run，只报告 coverage/duplicates；
5. 只读实现审查；
6. 显式授权后一次性完整 frozen oracle；
7. routing GO 后才设计 frozen student；
8. student 主门过后再做三 seed、跨数据集，以及 ResNet/普通 ViT/Swin 通用性矩阵。

## 十四、允许的最强结论

frozen routing 通过后最多可说：

> 在 target-only frozen LGPA blocks 与 disjoint support/reference episodes 中，raw pose-response allocation 比 identity averaging、equal same-slot averaging、slot/response permutation 与 feature-consensus controls 更有效地组织了跨图 support geometry。

仍不能说：

- CASD 已优于 MVI²P、UMTS、LCR²S 或 `exp123`；
- student 已保留 LGPA 涨点；
- raw response 是 visibility；
- 方法已跨 backbone/数据集通用；
- CASD 已是可投稿主创新。

这些结论只能由 matched RGB-only student、三 seed paired、强邻居对照与扩展实验共同获得。
