# 实验 exp409：PCHM（Pose×CLIP Hard-Pair Metric）

## 动机

exp408 已经把 pose-indexed CLIP 局部关系稳定写入 Stage-2，冻结诊断也得到
`correct < wrong-RGB < generic < zero`，但自然 e120 仍比 clean D0 低 `0.5 mAP point`。这说明继续改变
局部蒸馏形式不会自动改善最终身份检索几何。exp409 因此不再增加 CLIP teacher loss、part head、router 或
推理分支，而是直接改变现有 final descriptor triplet 的正负样本对象。

普通 batch-hard 完全根据当前 student descriptor 选择最远同 ID 和最近异 ID。它不知道同 ID 正对是否提供了
互补身体覆盖，也不知道异 ID 负对是否同时具有相似姿态和相似外观。PCHM 用增强后 pose 描述“身体覆盖关系”，
用冻结 CLIP visual descriptor 描述“外观混淆”，让二者共同决定现有 triplet 实际使用的 pair index。

## 核心假设

对每个 anchor，最有价值的正对是“同 ID、pose 覆盖互补、CLIP 外观仍一致”的真实视图；最有价值的负对是
“异 ID、pose 覆盖相似、CLIP 外观也相似”的身份混淆者。若这些 pair 直接进入最终 global descriptor 的原有
soft-margin triplet，模型会比纯 student batch-hard 更直接地压缩跨遮挡类内距离并拉开外观相近的异 ID，进而
同时提高 mAP 和 R1。

## 技术方案

### 1. 冻结输入

- backbone、TAPF D0、CE、pose auxiliary、triplet 形式与全部 loss weight 保持 clean D0 不变；
- 为 official train 的每张图在 fresh exp409 asset 中保存五个 L2-normalized frozen CLIP region-isolated
  visual descriptor及其validity；cache 必须完整唯一覆盖 15,618 张 train 图并绑定 SHA256；
- CLIP 只参与离散 pair selection，不作为回归 target，不接收梯度，eval 完全不加载 cache/CLIP/外部 pose；
- 当前增强后的 COCO-17 pose 生成五槽 soft visibility：每槽取该槽 joints 的
  `valid * clamp(score, 0, 1)` 均值。槽定义沿用冻结的 head、upper-torso/arms、lower-torso、upper-legs、
  lower-legs/feet ontology。

### 2. 无连续权重的联合选对

对 anchor `i` 与候选 `j` 定义：

- pose coverage distance：`q(i,j) = mean_r |v_i[r] - v_j[r]|`；
- CLIP appearance similarity：`a(i,j)` 为两图 cache-valid 槽上的逐槽 cosine 均值；没有共同 valid 槽的候选
  不进入选对。该量不跨槽比较，也不回归给 student；

候选集合内分别把两个量变为确定性 ordinal rank，再以无权 Borda rank sum 选 pair；相同 rank sum 按较高
CLIP rank、再按较小 batch index 破同。这里没有可调温度、margin、连续加权系数或 top-k。

- 正对：只在 `same PID, j != i` 中，最大化 `rank(q) + rank(a)`，即覆盖更互补且外观仍一致；
- 负对：只在 `different PID` 中，最大化 `rank(-q) + rank(a)`，即覆盖更匹配且外观更混淆。

选出的 `p_i/n_i` 直接索引 final global descriptor distance，代入 clean D0 原 soft-margin triplet：

`L_tri = SoftMargin(d(f_i,f_n_i) - d(f_i,f_p_i), 1)`。

PCHM 不给 triplet 乘权、不改 margin、不改 loss weight，也不把当前 student feature 用于 pair selection。若实现
退化成 loss reweighting、额外 auxiliary loss 或仍先用 student batch-hard 再调分，则不属于本设计。

### 3. 强反事实与 D0

所有诊断 arm 共享 anchor、PID 候选支持、batch、pose/CLIP数值和 tie-break，只改变一个语义绑定：

- `correct`：正确 pose 与正确 CLIP descriptor 联合选对；
- `wrong-RGB`：CLIP descriptor 按固定 different-PID cyclic shift 置换；
- `generic`：每张图的 CLIP descriptor 替换为当前 batch 均值；
- `zero`：pose visibility 清零；
- `pose-shuffle`：pose visibility 按固定 different-PID cyclic shift 置换；
- `CLIP-only`：只按 CLIP rank 选对；
- `D0`：原 final descriptor batch-hard。

首轮只训练 `correct` 和既有 sealed clean D0 比较。反事实 arm 先在同一冻结 batch snapshot 上验证标签、索引、
覆盖与外观统计；只有 `correct` 性能 GO 后才串行补最关键的 `pose-shuffle` 与 `CLIP-only` matched e120，避免
为已失败机制消耗多条 4090 训练。

## 对照组

- 主性能对照：sealed clean D0 seed1234/e120，raw
  `57.5587756578 mAP / 67.6923076923 R1`；
- 机制对照：D0 batch-hard、pose-shuffle、CLIP-only；
- 归因诊断：wrong-RGB、generic、zero；不删除任何不利 control。

## 预期结果与裁决

启动前必须证明：

1. 每个 anchor 的正负 index 标签合法，正对排除 self，负对全部 different-PID；
2. correct 的正负 index 都至少有一个不同于 D0 batch-hard；
3. pose-shuffle 与 CLIP-only 都至少改变一个 correct index，证明两个输入轴都 active；
4. default-off 与 clean D0 loss/forward 逐元素一致；
5. batch64 CUDA/AMP 下 loss finite，final descriptor/backbone 获得非零有限梯度。

唯一 fresh seed1234 自然训练到 e120。只有 e120 raw mAP 和 R1 **同时严格超过** clean D0 raw 门，才判
`PCHM PERFORMANCE GO`并进入 matched control；任一未过即 `EXP409 SEALED NO-GO`，不得通过调 margin、loss、
batch、rank fusion、cache 或 pose 定义救旧臂。

## 风险与失败解释

1. **静态 miner 过难**：exogenous pair 可能比随 student 演化的 batch-hard 更噪，若最终不涨点，说明
   pose×CLIP 的静态困难性不能替代 student geometry，封板后换结构对象。
2. **同 ID K=4 支持过小**：正候选只有三张，联合 rank 可能频繁退化；若两个输入轴不 active，GPU 不启动。
3. **CLIP 偏向衣着而非身份**：hard negative 可能形成伪混淆；由 CLIP-only control 与 selected-pair 统计解释。
4. **只有性能、没有机制证据**：即便 correct 涨点，若 pose-shuffle/CLIP-only matched control 不能支持联合选择，
   也只记为工程收益，不把 PCHM 写成论文主贡献。
5. **创新边界**：PCHM 只能主张“pose×CLIP 联合离散选对直接作用 final ReID metric”；普通 batch-hard、
   pose-guided hard positive、CLIP hard negative 和 loss weighting 本身均不是新贡献。
