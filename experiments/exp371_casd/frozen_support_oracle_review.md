# exp371 frozen support oracle 只读审查

## 审查结论

**当前裁决：不通过正式 Gate C 审查。脚本可用于小型 synthetic/smoke 验证，但不应在真实 val cache 上生成 GO/NO-GO。**

主要原因不是 endpoint exclusion 漏写，而是当前 protocol 本身仍使用 GT relation 改写每一个 query→gallery 距离；同时 advantage mask 在同一 val relations 上按 `gain_CASD > 0` 回看选择，再用这些 relations 证明 advantage 优于 full relation。两者都会产生结构性虚高。

此外，当前 `CASD-LIKE` 完全没有使用 pose response 或 visibility，只是同 ID 多图 slot agreement weighting。它更接近 MVI²P 式可靠多视图聚合，而不是预注册的 pose-organized CASD。即使该 arm 通过，也不能让 CASD 进入训练。

## 阻塞问题总表

| 级别 | 问题 | 当前实现 | 后果 |
|---|---|---|---|
| Critical | GT gallery support 泄漏 | donor 与 evaluation reference 同取原 gallery；正 endpoint 按 GT PID 被逐 pair 排除 | 每个正 pair 使用不同 query descriptor，mAP 不再是标准检索指标 |
| Critical | val outcome 回看选择 | `shared_mask = valid & (casd_gain > 0)` | advantage 对 FULL-REL 的“胜利”接近定义性结论 |
| Critical | CASD arm 不含 pose | 只用 donor-slot consensus cosine | 不能回答 pose organization，退化为可靠多视图 part averaging |
| Critical | target-only 未被强制 | 单个 `--val-cache`，无 pose provenance schema/paired aggregation | scene-merged 单跑也可能输出 `final_provisional_go=true` |
| High | loss-matched 仅匹配标量 mass | shared arm 与 FULL-REL coverage/权重集中度不同 | 不能等价为同一训练预算或同一梯度预算 |
| High | donor 数不足使 CASD 退化 | eligible 只要求 2 donors | 两个单位向量对自身 mean 的 agreement 完全相等，CASD=PART-EQUAL |
| High | MVI²P/UMTS/exp123 强对照不完整 | 无 anchor-inclusive/full-vector arm；EXP123-FULL 只模拟静态 geometry | 不能裁决外部/内部最近邻边界 |
| High | 内存和 Python 复杂度过高 | 同时保留 5 个 Q×G distance；endpoint 内层扫描整个 gallery | Occluded-Duke 全量存在数 GB 峰值与近十亿级 Python 比较风险 |
| Medium | exact duplicate 只报告不阻断 | val gallery 保留重复 path | mAP 和 donor/reference 计数可能重复加权 |
| Medium | “class-free”命名过强 | 不用 classifier，但 support/mask/sign 全用 GT PID | 应称 classifier-free GT-supported episodic oracle |

## 一、GT gallery support 泄漏与 endpoint exclusion

### 1. endpoint exclusion 的代码动作本身基本正确

`build_episode` 先按 `(pid,path)` 构造 donor；query path 被排除。`compute_arm_distances` 对每个 valid positive endpoint：

1. 从该 query 的 donors 中删除 endpoint path；
2. 重新构造 support descriptor；
3. 只覆盖该 positive path 对应的 distance；
4. 对 remaining donors 再检查 endpoint path 不存在。

因此，若只问“计算 q→positive-g 时，g 本身是否还在 support 中”，当前 exact-path 实现是正确的。测试也覆盖了两 donor 时 `q0→p0a` 只用 `p0b`。

### 2. 但推断协议仍然无效

问题在于 query descriptor 不再唯一：

- 对 negative endpoint，query descriptor 使用该 PID 的全部 gallery donors；
- 对 positive endpoint `g_j`，query descriptor 使用“全部同 PID donor 减去 `g_j`”；
- 对另一个 positive endpoint `g_k`，又换成另一份 descriptor。

也就是说，脚本先用 GT PID 知道当前 endpoint 是正样本，再决定是否以及如何重构 query。最终一行 distance 虽可排序，却不是同一个 query embedding 对 gallery 的距离。此时 mAP/R1 不再是标准 ReID retrieval metric，也不能作为“未见身份 class-free geometry”证据。

更严重的是，donor pool 还包含：

- 其他正在被评估的 same-ID positive gallery；
- 标准 ReID 因 same-PID/same-camera 而从 reference 排除、但仍留在 donor pool 的 gallery；
- 可能存在的近重复帧。

所以 endpoint exclusion 只避免了最直接的 self-inclusion，未消除 GT gallery support 的 transductive oracle 泄漏。

### 必须改为 disjoint support/reference episodic protocol

对原 gallery 做 PID 内 deterministic K-fold：

```text
原 query Q
原 gallery -> support pool S + evaluation reference R
S ∩ R = ∅（path/content/tracklet 均不重合）
```

每个 fold 中：

1. 每个 query 只构造一次 descriptor，始终使用其 PID 的 `S_y`；
2. 所有 query 都对同一个 `R` 排序；
3. support images 不得作为任何 query 的 evaluation endpoint；
4. 多 fold 轮换 S/R，报告 fold mean、min 和方差；
5. 若 camera 条件允许，优先用 cross-camera support，并单独报告 same-camera arm；
6. 不再需要 pair-specific endpoint correction。

这仍是 GT-supported oracle，不是部署协议，但至少不会用当前 relation label 改写当前 distance。

## 二、advantage shared mask 存在同集回看与定义性获胜

当前代码在 val 上执行：

```text
casd_gain = signed_relation_gain(CASD, SELF, GT positive/negative)
shared_mask = valid & (casd_gain > 0)
shared_weight ∝ casd_gain
```

然后又在同一组 val relations 上检查：

```text
CASD shared signed gain > controls
CASD advantage-selected gain > EXP123-FULL all-relation gain
```

这不是独立验证。CASD mask 只保留 CASD 已经改善的 pair，并按改善幅度加权；在这一 mask 上 CASD 的 signed gain 必然为正，而且大幅改善 pair 会获得更大权重。FULL-REL 则必须承受全部有益与有害 pair。两者即使 total mass 都是 1，也不能据此证明 advantage selection 比 full relation 更好。

### 合法的处理方式

三种可选协议，优先级从高到低：

1. **训练—验证分离**：只在 train identity relations 上定义 selector/阈值；val 只评估一个不读取 val GT gain 的可观测 selector；
2. **cross-fitting**：在 fold A 的 relations 上产生 selector，在 disjoint fold B 上评估，再交换；
3. **只作上界统计**：保留当前 own-positive mask，但删除所有 GO 条件和“优于 FULL-REL”的解释，明确它只是 hindsight positive-change upper bound。

如果实际 CASD 训练确实要用训练标签判断 support-vs-self relation gain，那么它的价值只能由 frozen student/full training 在未见身份 val 上裁决，不能由 val hindsight oracle 自证。

## 三、五臂是否真正 loss-matched

### 结论

**SELF/ID-MEAN/PART-EQUAL/PART-PERM/CASD-LIKE 在同一个 shared mask 上确实复用了逐 relation weight；但整体仍不能称为“与 EXP123-FULL loss-matched”。**

已经成立的部分：

- 五个 support arms 共享同一 mask SHA 和 weight SHA；
- 正/负 scalar mass 分别为 0.5；
- total scalar mass 为 1.0。

未匹配的部分：

1. shared advantage 只覆盖 CASD hindsight-positive relations，FULL 覆盖全部 valid relations；
2. 两者 per-query mass 不同，gallery 多或 gain 大的 query 会支配总 loss；
3. 两者 nonzero relation 数/effective sample size 不同；
4. raw weight 分布不同：CASD 用 gain，FULL 用 `[1,2]` focus；
5. 相同 `sum(weights)=1` 不等于相同梯度范数、方差或优化难度；
6. 当前脚本没有 student，因此所谓 loss 只是 teacher geometry 上的加权统计。

至少应报告并匹配：

- 每 query 正/负 mass；
- effective sample size `1/sum(w^2)`；
- 最大 relation/query weight；
- nonzero pair 数；
- target delta 分布；
- 若比较训练 objective，必须使用同一 student 初始化与实际 loss 后的梯度 norm。

推荐先按 query 归一，再按正/负归一，避免多 gallery PID 主导。若 FULL 与 sparse advantage coverage 本来就不同，则不要称 exact loss-matched；称 **class-mass normalized** 更准确。真正的 exp123/CASD 优劣留给 frozen-student 单变量训练。

## 四、CASD-LIKE 实际上不是 pose-organized CASD

当前 cache 只有 `7×D features`，没有：

- target-only raw pose response；
- 五个 slot 的 absolute/relative reliability；
- target person validity；
- person count/distractor response；
- slot coverage。

因此代码用 `cos(donor_slot, donor_slot_mean)` 代替 pose reliability。这只能说明跨图 feature agreement，不能说明 pose 组织了可见部位。它与 MVI²P 的“对同 ID 多图做可靠性加权综合”在机制上更接近。

### 两 donor 的数学退化

对两个单位向量 `a,b`：

```text
cos(a, norm(a+b)) = cos(b, norm(a+b))
```

所以两个 donors 时 agreement 权重严格相等，`CASD-LIKE == PART-EQUAL`。当前 endpoint protocol 中：

- 原来只有 2 donors：negative relations 也是等权；positive endpoint 排除后只剩 1 donor；
- 原来有 3 donors：positive endpoint 排除后剩 2 donors，仍严格等权；
- 只有原来至少 4 donors，positive relation 上 reliability weighting 才可能与 PART-EQUAL 不同。

当前 eligibility 只要求 2 donors，因此 CASD 核心变量可能在大多数正 relations 上根本没有生效。

### 必须扩展 cache schema

下一版至少保存：

```text
schema_version
block_names = [global, pooled, slot1...slot5]
features[N,7,D]
slot_pose_response_raw[N,5]
slot_pose_allocation_relative[N,5]
target_person_valid[N]
person_count[N]
pids/camids/paths
content_sha256 或 perceptual hash
tracklet/frame（若数据可得）
pose_source = target_only | scene_merged | canonical
checkpoint/config/dataset-index/extractor-script SHA
```

若只有归一化 `kp_weights`，必须称 `pose-response allocation`，不能称 absolute visibility。

### 2×3 extraction-routing 正交门

同一组 frozen features 上只改变 routing：

- extraction `E ∈ {target-correct, canonical}`；
- routing `R ∈ {target raw-pose response, equal, independent slot permutation}`。

其中同一 `E` 下三个 `R` 必须复用逐 bit 相同的 slot tensors，只改变 donor/slot weights。agreement weighting 可以保留为额外 `AGREE` arm，但不能替代主 `POSE-RESPONSE` arm，也不能决定 CASD GO。

## 五、target-only 约束没有被实现

设计要求 target-only 与 scene-merged 方向不冲突，但脚本：

- 只接收一个 `--val-cache`；
- 不检查 `cache["mode"] == "correct"`；
- 不检查 `split`；
- schema 没有 `pose_source/target_assignment`；
- `results.json` 可在单个 scene-merged cache 上直接给 `final_provisional_go=true`；
- 没有跨两个 cache 的 paired query/path/order/SHA 检查与联合 gate。

因此这项预注册目前只存在于文字里。必须把 target-only 作为主输入硬断言，scene-merged 只做 paired diagnostic；或者由一个 aggregator 同时读取两份 cache 并在最终 gate 中强制方向一致。

## 六、PART-PERM 公平性

### 做得正确的部分

- 每个 donor 使用 deterministic nonzero cyclic shift；
- 五个 individual slot 是严格双射、无 fixed point；
- donor/PID/feature value multiset/slot 数/descriptor 维度保持；
- pooled block 不变；
- endpoint exclusion 后仍使用同样的 path-derived permutation。

所以它是一个合理的 **individual-slot correspondence control**。

### 仍需补的边界

1. pooled support 始终保持正确聚合，因此 PART-PERM 只破坏 5 个 individual slots，不破坏全部 local organization；论文必须按这个范围解释；
2. 只跑一个 seed/path permutation 可能偶然有利或不利，应至少报告多个预注册 permutation seeds 的 mean/min；
3. 当前 min structure gap 只有 `0.001`（0.1 pp），不足以支撑强 slot claim；
4. 若 donor 少，单个 cyclic shift 会把整个 identity 的 slot 系统一致旋转，建议同时报告 per-donor independent full derangement；
5. 需额外断言每个 output slot 接收到相同 donor 数、全局 feature multiset hash 不变。

PART-PERM 本身不是当前最大漏洞；最大漏洞是 CASD 没有 pose routing、评测又是 endpoint-dependent。

## 七、class-free metric 是否可计算

### 数值上可计算

`retrieval_metrics`、AP、Rank-k、positive/negative distance 和 signed gain 的 tensor 计算没有明显公式错误。global/local blocks 都先单位化，flatten 后再整体单位化，相当于七块等权 cosine。

### 但“class-free retrieval”命名不准确

脚本不加载 classifier，所以是 classifier-free；但它仍使用 GT PID：

- 构造 same-ID donor pool；
- 定义 positive/negative；
- 对 positive endpoint 做专属 descriptor correction；
- 定义 signed gain 的方向；
- 生成 advantage mask。

建议改名：

> classifier-free, GT-supported episodic relation oracle

在改成 disjoint S/R、每 query 唯一 descriptor 前，mAP/R1 只能是 pair-dependent oracle statistic，不能写 retrieval performance。

## 八、exact-path 与近重复排除

当前 exact normalized path exclusion 有实现，但仍有四个缺口：

1. `normpath` 不处理 symlink、大小写、不同根目录下的同一文件；
2. val gallery duplicate path 只计数，不报错、不去重，仍参与 mAP；
3. 没有断言一个 normalized path 只对应一个 PID/camid；
4. 没有 content hash、perceptual hash、tracklet/frame 排除。

正式 Gate 至少应：

- exact duplicate gallery paths 必须为 0，否则 hard fail 或先确定性去重；
- path→PID/cam 映射必须一致；
- content SHA 相同 hard exclude；
- 感知近重复与同 tracklet 近邻单独报告并做 strict arm；
- 报告过滤前后 donor 数、eligible query、PID/camera 分布。

设计已承认 near-duplicate 未解决，因此当前只能叫 strict-exact-path，不能叫 strict-view LOO。

## 九、MVI²P、UMTS、exp123 强对照边界

### MVI²P / UMTS

当前五臂没有：

- anchor-inclusive multi-view teacher；
- full feature/map integration；
- comprehensive teacher→single student 的实际 KD；
- MVI²P 类 identity-confidence/CAM reliability；
- UMTS 式 multi-shot full target。

`ID-MEAN` 把六个 local blocks 混成一个向量再复制，既不是 MVI²P，也不是 UMTS；`PART-EQUAL` 是 part-wise LOO mean，也不是 anchor-inclusive comprehensive teacher。

frozen oracle 至少可加两个 geometry controls：

1. `FULL-INCL`：anchor + same-ID donors 的完整 descriptor/feature integration，对应 MVI²P/UMTS anchor-included teacher 上界；
2. `FULL-LOO`：严格排除 anchor 的 full-vector integration，区分 LOO 与普通 comprehensive target。

但真正“CASD 是否优于 MVI²P/UMTS”必须由 matched frozen student/full training 比较，不能由 frozen teacher geometry 宣布。

### exp123

当前 `EXP123-FULL` 只保留了：

```text
focus = 1 + abs(d_CASD-d_SELF)/row_max
```

它没有复现：

- exp120/123 的 online support-complete bank；
- teacher/student softmax distribution；
- 同一 focus 同时作用 teacher/student softmax；
- student optimization；
- exp123 的实际 relation set 与温度。

设计虽声明“只模拟 geometry”，但 gate 又用它判定 advantage selection 是否独立，超出了这个模拟的证据能力。`advantage_beats_loss_matched_exp123_full` 应从 frozen oracle GO 中删除；必须在同一 frozen student 上实现真实 full-relation 与 advantage-relation 两臂。

## 十、复杂度与内存

以 Occluded-Duke 约 `Q≈2.2k, G≈17.7k, D=5376` 粗估：

- 单个 float32 `Q×G` distance 约 156 MB；五臂同时保留约 780 MB；
- feature cache 约 430 MB；
- `donor_by_pid` 按 path 复制 block tensor，再接近一份 cache；
- advanced-index gallery matrix、gallery normalization、gain/delta/focus 与三个 bool mask 会再增加数百 MB；
- CPU 峰值容易超过 2–3 GB，CUDA 还会复制完整 gallery。

时间上，endpoint correction 对每个 query/positive path 都执行：

```text
scan zip(all gallery PID, all gallery paths)
```

复杂度接近 `O(arms × Q × positives × G)` 的 Python 循环。若每 query 有约 10–20 个 positives，可能达到数亿到十亿级 Python 比较，远慢于矩阵距离本身。

建议：

1. 首选 disjoint S/R，彻底删除 pair-specific correction；
2. 预计算 `(pid,path)->gallery_positions`，禁止内层扫描 G；
3. gallery normalized descriptor 只构造一次；
4. arm 逐个计算诊断并释放 distance，最多保留 SELF/CASD；
5. donor pool 存 indices，不复制每 path 7×D tensor；
6. 先在 metadata-only pass 统计 eligibility/donor counts，再决定是否加载全部 features；
7. manifest 记录估算/实际 peak RSS 与 wall time。

## 十一、数值 kill-switch 需要调整

当前阈值：

- CASD vs strongest `>=0.5 pp`：幅度本身尚可；
- PART-EQUAL vs PART-PERM `>=0.1 pp`：过低；
- CASD vs PART-EQUAL `>=0.1 pp`：过低；
- eligible query ratio `>=50%`：可能只保留容易身份；
- donor count `>=2`：不能激活 agreement weighting。

建议 revised oracle 的预注册门槛：

1. target-only、disjoint-S/R K-fold 主协议；
2. 每 fold 每 query 至少 3 个 support donors；若使用当前 agreement-to-mean，至少 3 donors 固定支持集；
3. eligible query ratio 至少 70%，并报告 retained/removed query 的 SELF 难度差；
4. `POSE-RESPONSE - max(ID-MEAN, PART-EQUAL, PART-PERM, AGREE) >=0.5 pp` 的 fold mean；
5. 每个 fold 方向为正，不能由单 fold 驱动；
6. `PART-EQUAL - PART-PERM >=0.3~0.5 pp` 才保留 slot correspondence 强 claim；
7. target-only 与 scene-merged 结论同向；
8. near-duplicate strict arm 同向；
9. advantage/full-relation 不在 frozen geometry 阶段裁决，移到 matched frozen student；
10. 若 pose-response 不优于 agreement/equal，CASD pose-specific 主线 NO-GO，即使 agreement arm mAP 更高也不能改名挽救。

## 十二、可直接实现的修订协议

### 输入

- target-only 与 canonical/scene paired caches；
- 固定 schema version、block labels、pose provenance；
- raw/relative slot pose response；
- exact/content/near-duplicate元数据；
- checkpoint/config/dataset index SHA 一致。

### episode

1. gallery PID 内按 hash 做 K-fold support/reference disjoint split；
2. support 不进入 evaluation gallery；
3. 每 query 每 fold 只有一份 descriptor；
4. cross-camera 与 unrestricted 分开报告；
5. exact/content duplicate hard fail，near-duplicate strict arm。

### frozen arms

- `SELF`；
- `ID-MEAN`；
- `PART-EQUAL`；
- `PART-PERM`（多个 seeds）；
- `AGREE`（当前 CASD-LIKE，降为 non-pose control）；
- `POSE-RESPONSE`（主 CASD routing）；
- `FULL-INCL/FULL-LOO`（MVI²P/UMTS geometry boundary）。

### 指标

- episodic mAP/R1 与 class-balanced signed gain；
- per-query、per-fold gain 分布与 bootstrap CI；
- donor count/eligible/camera/duplicate审计；
- 所有 arm 使用同一 S/R split；
- 不在 val 上按 own gain 选择主 mask。

### 后续训练门

只有 `POSE-RESPONSE` 通过 frozen geometry 门，才启动同初始化 frozen student：

- B0；
- same-image KD；
- MVI²P/UMTS-style full target；
- exp123-style full relation；
- CASD shared-mask advantage relation。

只有 actual student 上 CASD 优于后三个强对照，才能越过 MVI²P/UMTS/exp123 边界。

## 最终裁决

现有实现有不少工程防护是认真且正确的：cache SHA、train/val path overlap、block finite、strict exact-path donor 去重、PART-PERM derangement、正负 scalar mass 断言都值得保留。

但它们没有解决两个最核心的统计问题：

1. GT endpoint 决定 pair-specific query descriptor；
2. val CASD gain 决定 val advantage mask。

因此当前 `final_provisional_go` 没有有效含义。先改成 support/reference disjoint episodic protocol、去掉 val hindsight mask，并把 agreement arm降为 control、补真正 pose-response arm；完成这些修改后再做一次只读审查。
