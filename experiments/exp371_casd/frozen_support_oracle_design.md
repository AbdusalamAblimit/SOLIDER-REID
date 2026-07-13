# exp371 Gate C：class-free frozen support oracle 设计

> **状态：REJECTED PROTOTYPE — NOT APPROVED FOR REAL-CACHE GO/NO-GO。**
>
> 第二轮 red-team 已确认当前原型只能用于 synthetic wiring/smoke，禁止把真实 cache 输出解释为 CASD 证据：
>
> 1. support descriptor 随当前 positive relation endpoint 改变，是 GT endpoint-dependent oracle，不对应可部署或可蒸馏的固定 teacher descriptor；
> 2. shared advantage mask 直接在 val relation label 与 CASD signed gain 上回看生成，属于 val hindsight selection；
> 3. `7×768` cache 没有 raw pose response/absolute visibility，`CASD-like` 只能做 feature-consensus，不能识别 pose organization；
> 4. 只有两个 strict-path donors 时，consensus weighting 退化或接近 PART-EQUAL，无法识别可靠性加权；
> 5. 因此下文预注册门槛已作废，不允许据此启动 frozen student 或完整训练。
>
> `frozen_support_oracle.py` 与其小型测试仅作为被拒原型保留，记录已经排除的实现方式；截至拒绝时只完成 synthetic 测试，没有生成任何真实 cache 结果。

## 先写证据边界

本 oracle 不是新的 ReID 结果，也不是 student 训练。它只在冻结的 `7×768` 描述子上做一个带 GT identity support 的诊断：

> 在严格排除 anchor 与当前 relation endpoint 路径后，同 ID 其他图像形成的 slot-structured support，是否比普通 identity averaging、等权 part aggregation 和 slot permutation 更改善未见身份的 class-free retrieval geometry？

主结论只读取 `val_correct.pt` 或 target-only 的 `val_correct.pt`。这些 val PID 没有参与 exp336 classifier 训练；脚本不加载、不拟合、也不调用任何 ID classifier。`train_correct.pt` 只用于一个独立的 deterministic pseudo-query/gallery 诊断和输入完整性检查，不能覆盖 val 结论。

## 最新事实对本门禁的约束

Gate B s0：

| arm | mAP | 相对 global |
|---|---:|---:|
| global | 58.9908 | — |
| correct | 59.8357 | +0.8449 |
| canonical | 59.7374 | +0.7465 |
| shuffled | 59.8037 | +0.8129 |
| uniform | 59.3689 | +0.3781 |
| no-pose | 59.4014 | +0.4106 |

target-only 为 `59.8121`；Gate D train-only PCA-768 为 `59.9336`。

因此已知：

1. LGPA 的局部结构资产成立；
2. 精确当前图 pose 相对 shuffled/canonical 的独占量极小；
3. 5376-D 不是该增益的必要解释；
4. Gate C 不能从 Gate B 直接继承“anatomical/visibility 已成立”的前提。

本 oracle 的 `CASD-like` 只表示 **slot correspondence + class-free cross-view agreement reliability**。现有 cache 不含 raw pose response 或绝对 visibility，所以它不能回答 pose visibility weighting 是否有效。

## 可以回答什么

1. strict-path LOO 后是否仍存在跨图 support headroom；
2. headroom 是否只来自 same-ID averaging；
3. 保留 local slot correspondence 是否优于把 slots 打乱；
4. parameter-free 的 cross-view slot agreement weighting 是否优于 slot-wise equal mean；
5. support teacher 相对 SELF 的正向 relation changes 是否足够密集；
6. advantage-selected relation target 相对 exp123-style full relational target，是否拥有更高的 class-balanced signed gain；
7. 所有 support construction controls 在完全相同 shared mask、pair 集和 loss mass 下的相对质量。

## 不能回答什么

1. image-only student 能否学到 support target；
2. 训练后能否保留 `+0.9 mAP`；
3. target support 是否对应真实可见的目标人体部位；
4. pose estimator 或 precise pose correspondence 是否必要；
5. train-time GT same-ID support 在部署时是否可用；
6. 该机制是否跨数据集、跨骨干、跨 seed；
7. CASD 是否相对 UMTS、exp120/123 或其他工作具有论文新颖性；
8. exact-path 不重复是否等于无近重复帧。cache 没有感知 hash/tracklet 信息，故只能执行 exact normalized path exclusion；
9. oracle 的绝对 mAP 是否可作为正式主表数字。它回看 GT PID 构造 support，只能用于机制筛选。

## 输入协议

### cache 要求

- `features`: `[N, 7×D]`，正式运行 `D=768`；
- `pids/camids/paths` 与 feature 数量一致；
- `val.num_query > 0`，顺序保持原 query/gallery 协议；
- `train.num_query == 0`；
- train/val checkpoint SHA 完全相同；
- train/val normalized path overlap 必须为 0；
- feature 必须 finite；
- 每个 block 的 norm 统计落盘，但不因少量浮点误差静默修改原 cache。

### val 主协议

- query 使用原 val query；
- reference 与 donor pool 都来自原 gallery；
- 对每个 query，必须至少有两个不同路径的 same-PID gallery donors；
- query 自身路径从 donor pool 硬排除；
- 计算 query→gallery 某个正 relation 时，当前 gallery endpoint 的路径也从 donor pool 排除；
- exact duplicate path 只计一个 donor，不允许重复路径增加权重；
- 标准评测中的 same-PID/same-camera reference 继续排除。

这一 pair-specific endpoint exclusion 防止 support target 直接包含正在被比较的 positive gallery feature。

### train 辅助协议

对每个拥有至少四个不同路径的训练 PID，按 `SHA256(seed:path)` 排序，取一个 path 作为 pseudo-query，其余为 gallery。它只检查机制在训练身份上的形态与 val 是否方向冲突；所有 GO/NO-GO 以 val 为准。

## 五种 frozen support construction

所有 arm 使用同一 frozen cache、同一 query、同一 donor paths、同一 gallery references。global block 始终保留 anchor SELF global，只干预六个 local blocks，避免普通 ID global prototype 偷走主解释。

### 1. SELF

```text
z_self(i) = [g_i, pooled_i, slot_i,1 ... slot_i,5]
```

不读取任何 support，是关系基线。

### 2. ID-MEAN

把 strict-LOO donors 的全部六个 local blocks 不分 slot 放入一个 identity-local bag，求一个均值后复制到六个 local blocks：

```text
u_id(i) = norm(mean_{j,b in local blocks} t_j,b)
z_id(i) = [g_i, u_id, u_id, u_id, u_id, u_id, u_id]
```

它保留相同的同 ID donor 内容与总维度，但不保留 part correspondence。它直接检验“类中心降噪是否已经解释全部收益”。

### 3. PART-EQUAL

按原 slot 一一对应，对 strict-LOO donors 做逐 slot 等权平均并逐 block normalize：

```text
c_i,b = norm(mean_j t_j,b)
z_equal(i) = [g_i, c_i,pooled, c_i,1 ... c_i,5]
```

它保留 slot correspondence，但没有 visibility/reliability selection。

### 4. PART-PERM

对每个 donor 的五个 individual slots 使用由 path SHA 决定的非零 cyclic shift，再按 PART-EQUAL 聚合。global 与 pooled block 不变：

```text
pi_j(k) = (k + 1 + SHA(path_j) mod 4) mod 5
```

它保持 donor 数、identity、feature values、norm、全局/pooled 内容与 loss 预算，只破坏 individual slot correspondence。

### 5. CASD-LIKE SLOT SUPPORT

pooled block 仍等权。对五个 individual slots，先计算 donor slot mean 作为 class-free consensus，再用每个 donor 与 consensus 的非负 cosine agreement 做 parameter-free 权重：

```text
r_j,k = max(cos(t_j,k, norm(mean_l t_l,k)), 0) + eps
w_j,k = r_j,k / sum_l r_l,k
c_i,k = norm(sum_j w_j,k t_j,k)
```

它不使用 PID classifier、CE、learned scorer 或 val 调参。由于没有 raw visibility，它只能检验“slot correspondence + cross-view agreement reliability”，不能称为 pose-visible support。

## class-free relation 与检索指标

每个 arm 形成 query→gallery 的 pair-dependent squared normalized Euclidean distance：

```text
d(i,j) = 2 - 2 cos(z_arm(i; exclude path_i,path_j), gallery_self_j)
```

报告：

- class-free retrieval mAP/R1/R5/R10；
- positive/negative mean distance；
- class-balanced mean margin；
- 相对 SELF 的 signed relation gain：

```text
gain(i,j) = d_self - d_arm,  same PID
gain(i,j) = d_arm - d_self,  different PID
```

不使用 classifier logits、训练类 prototype CE 或线性 probe。

## shared-mask 与 loss-mass

### CASD shared advantage protocol

shared mask 只由 CASD-like arm 相对 SELF 的正 signed gain 产生：

```text
M_shared = valid_relation AND gain_CASD > 0
```

shared raw weight 为 `gain_CASD`。正、负 relations 各归一到总 mass 的一半；总 loss mass 固定为 `1.0`。SELF、ID-MEAN、PART-EQUAL、PART-PERM、CASD-like 全部使用完全相同的：

- query/gallery relation set；
- shared mask；
- per-relation weight；
- positive mass `0.5`；
- negative mass `0.5`；
- total mass `1.0`。

每臂只能替换 target distance，不能生成自己的主比较 mask。own-positive-gain coverage 只作安全性统计。

### exp123-style FULL-REL control

以 CASD-like distance 作为 support-complete teacher，以 SELF distance 作为 base：

```text
delta = abs(d_CASD - d_SELF)
focus = 1 + delta / row_max(delta)
```

FULL-REL 使用全部 valid relations，但同样做 class balance，并归一到 positive `0.5`、negative `0.5`、total `1.0`。因此它与 advantage protocol 的 supervision coverage 不同，但 **loss mass 完全相同**。报告：

- coverage；
- class-balanced signed gain；
- weighted SmoothL1(`d_SELF`, `d_CASD`)；
- pair delta 与 focus；
- total/positive/negative mass 断言。

这只模拟 exp123 的“完整 teacher relation + delta focus”target geometry，不声称复现 exp123 的 online bank 或 student optimization。

## 预注册判断

### CASD-like 继续进入 frozen student 的必要条件

val 主协议同时满足：

1. `CASD-like > max(ID-MEAN, PART-EQUAL, PART-PERM)`；
2. 相对最强 control 的 mAP 至少 `+0.5` 个百分点；
3. shared-mask class-balanced signed gain 高于所有 controls；
4. PART-EQUAL 明确优于 PART-PERM；否则 slot correspondence 没有价值；
5. CASD-like 明确优于 PART-EQUAL；否则 agreement weighting 没有独立价值；
6. advantage-selected signed gain 明确高于 loss-mass matched FULL-REL；否则 support-vs-self selection 没有独立价值；
7. exact-path LOO violation 为 0，且 eligible query 不是极小子集；
8. target-only cache 与 scene-merged cache 的结论方向不矛盾。若只能在 scene-merged 成立，删除 anatomical identity support claim。

### NO-GO

出现任一项即停止 CASD 主机制，不扫温度、queue、slot 数或 loss weight：

- ID-MEAN 最强：收益只是 same-ID prototype/denoising；
- PART-EQUAL≈PART-PERM：slot correspondence 无价值；
- CASD-like≈PART-EQUAL：reliability organization 无价值；
- FULL-REL 与 advantage protocol 等价或更好：CASD advantage selection 不构成新变量；
- strict-path LOO 后大部分 query 无 support；
- val 未见身份上无正 signal；
- target-only 与 scene-merged 方向相反。

## 输出与复现

脚本只写指定 `output-dir`：

- `manifest.json`：cache SHA、checkpoint SHA、path overlap、protocol 与参数；
- `train_results.json`：训练身份辅助诊断；
- `val_results.json`：未见身份主结果；
- `results.json`：裁决摘要；
- stdout：逐 arm 指标与 COMPLETE。

脚本不修改模型、config、cache 或 tracked 文档；不启动训练。
