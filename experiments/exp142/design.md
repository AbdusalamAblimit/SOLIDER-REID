# 实验 exp142: SKC（Support-Conditioned Keypoint Completion）

## 动机

- `exp109` 的 oracle support bank 已明确说明：当前真正的 headroom 不在简单 rerank，而在 **single-image support incomplete**
- 从 `exp119` 到 `exp140` 的一系列 `LPCS / pair correction` 实验说明：
  - pair-specific correction 方向不是完全错误
  - 但它始终停留在“距离层修正”
  - 这类方法很容易变成：
    - 学一个标量权重
    - 学一个 residual score
    - 学一个 context-aware scorer
  - 最终更像检索层微调，而不是把“缺失的人体证据”真正补回表征
- 用户已明确要求：本地不要继续围绕同一个小点消耗时间，而应进入 **更大的方法级改动**

因此，本地下一条线不再是 `LPCS` 家族，而是回到 `exp109` 的核心发现本身：

**既然问题是 keypoint-level support incomplete，那么就不要只在距离层修正，而要在特征层显式补全缺失的 pose-aligned evidence。**

## 核心假设

1. 当前 `exp030a` 的 `kp_feats` 已经具备明确的人体关键点语义，但低置信关键点对应的特征支持不完整
2. 之前的 `SCRC` 之所以不够强，不是因为 support bank 无用，而是因为它只做了：
   - `gate * (proto - feat)` 的逐点残差替换
   - 缺少跨关键点结构交互
   - 也没有显式利用“高置信自证据 + support prototype + skeleton topology”三者的联合信息
3. 如果加入一个更大的 **Support-Conditioned Keypoint Completion** 模块，只对低置信关键点做 pose-aware token completion，那么：
   - `global` 和 `equal_concat` 都应优于 `exp030a`
   - 收益应主要体现在低可见 / 重遮挡样本
   - 这条线比 `LPCS` 更像论文主方法，因为它直接回答了 `exp109` 暴露出的根问题

## 技术方案

### 1. 插入位置

- 位置：`SkeletonGCNHead` 内，位于已有 `kp_feats` 生成之后、pooling / fusion 之前
- 默认关闭，通过 config 开关控制，不能破坏 baseline

### 2. 输入信号

对每个样本、每个 keypoint，构造三类输入：

1. **当前样本的 keypoint token**
   - `kp_feat_j`
2. **support-complete token**
   - 来自现有 `support_complete_bank`
   - 包括：
     - `support_proto_j`
     - `proto_conf_j`
     - `proto_count_j`
3. **pose reliability / structure token**
   - 当前关键点 score
   - low/high mask
   - skeleton 邻接关系

### 3. Completion 核心机制

不是简单的逐点 gate，而是一个更大的 completion block：

1. **低置信关键点作为 query**
   - 只对 `score < low_thr` 的关键点执行 completion
2. **高置信关键点作为 self evidence**
   - 当前图中高置信关节点提供样本内结构上下文
3. **support prototype 作为 cross-image evidence**
   - 同 ID 的 support-complete prototype 提供跨图补全证据
4. **结构化交互**
   - 低置信关节点同时与：
     - 当前图高置信关节点
     - 对应 support prototype
     - skeleton 邻接节点
   做交互，输出 completed token

一句话概括：

**不是把 prototype 直接抄给当前 keypoint，而是让低置信关键点在“自图高置信证据 + support-complete prototype + 身体结构”三者之间做条件补全。**

### 4. 输出形式

- 得到 `kp_completed`
- 只在 low-confidence joints 上写回：
  - `kp_out = kp_feat + low_mask * delta_completed`
- high-confidence joints 保持原始特征为主，避免过度覆盖

### 5. 监督方式

第一版坚持单变量，不引入复杂新损失堆叠：

1. 主监督保持 `exp030a` 原有：
   - ID loss
   - Triplet loss
2. 新增一个轻量 completion consistency loss：
   - 只对 low-confidence joints 计算
   - target 为 `support_proto`
   - 由 `proto_conf` 加权
3. 若第一版已经能转正，再考虑后续做更强的结构一致性或 teacher-student 版本

### 6. 与旧线的本质区别

相对 `SCRC / SCKD / LPCS`：

- `SCRC`
  - 只是逐 keypoint 的残差替换
- `SCKD`
  - 只是把 support-complete prototype 当蒸馏 teacher
- `LPCS`
  - 只在 pair distance 层做 correction

而 `SKC` 是：

- 在 **特征层**
- 对 **低置信关节点**
- 做 **support-conditioned, structure-aware completion**

这条线的目标不是“把距离修得更聪明”，而是**直接修复单图表征本身的不完整**。

## 对照组

- 唯一正式基线：`exp030a`
- 报告模式：
  - `equal_concat`
  - `global`
- `cvk_hybrid` 只作为机制辅助观察，不把 test-time trick 记成训练创新

## 预期结果

如果这条线成立，预期应出现以下现象：

1. `global` 首次出现比 `exp030a-global` 更清楚的正收益
2. `equal_concat` 至少不弱于当前最强 supporting 线
3. 低可见 / 重遮挡样本收益更明显
4. completion 模块的统计应表现出：
   - 只在 low-confidence joints 上工作
   - support reliability 高时补全更强
   - 不会像 `exp140` 那样快速塌成常数 gate

## 日志与止损设计

这条线不允许只看：

- `loss`
- `mAP / R1`

必须同时把“模块是否真的在工作、是否在退化”打进日志，便于尽早止损。

第一版计划至少记录以下统计：

1. `skc_lmr`
   - low-confidence joints ratio
2. `skc_spr`
   - 低置信 joints 中，拿到 support prototype 的比例
3. `skc_arr`
   - 实际执行 completion 写回的 joints 比例
4. `skc_gm / skc_gs`
   - completion gate 的均值 / 标准差
5. `skc_dn`
   - completion delta 的平均范数
6. `skc_pc / skc_pcnt`
   - prototype confidence / prototype count 的均值
7. `skc_cl`
   - completion consistency loss
8. `skc_pre / skc_post`
   - 写回前后 low-confidence token 与 support target 的距离

希望这些日志直接回答下面几个问题：

1. 模块有没有真正作用在 low-confidence joints 上
2. 它是在保守跳过，还是在过度覆盖
3. 它是在做有条件的 completion，还是迅速塌成常数 gate
4. 它到底有没有把 low-confidence token 往 support-complete target 拉近

初步止损规则也要提前写清楚：

1. 若 `epoch 21+` 后 `skc_arr` 长期接近 `0`
   - 判定为模块学会了整体跳过
2. 若 `skc_gm` 很快接近 `1` 且 `skc_gs` 很低
   - 判定为退化成恒定强覆盖
3. 若 `skc_pre / skc_post` 没有明显改善
   - 判定为 completion 没有真正兑现到 token 层
4. 若 `ep30/40` 已明显落后 `exp030a`
   - 在机制日志同时负面的情况下，优先早停

## 风险与失败解释

1. **训练转负**
   - 说明 feature-level completion 太激进，破坏了当前已稳定的 pose branch 表征
2. **只有 `equal_concat` 涨，`global` 不涨**
   - 说明补全更多是在帮助 fusion，而没有真正改善单图表征
3. **completion 很快退化成恒等映射**
   - 说明 bank reliability 或 block capacity 不足，模块学会了保守跳过
4. **completion 很快退化成恒定强覆盖**
   - 说明 support prototype 过强，模块在抄 bank，而不是做条件补全
5. **结果仍只是不痛不痒的小涨**
   - 说明 `exp109` 的 headroom 不能仅靠 feature completion 兑现，后续需要更进一步做 retrieval-time structured reasoning

## 当前执行边界

1. 先写设计，不改代码，不启动训练
2. 设计确认后，再实现最小可运行版 `SKC`
3. 实现完成后：
   - 本地自检
   - 全面 Claude 审查
   - 用户确认 review 结束
   - 才允许启动训练

## Claude 审查范围

这条线的 Claude 审查不能只查“我改了哪些文件”，而要查整个方法是否站得住。默认审查范围包括：

1. **想法合理性**
   - 这条线相对 `exp109 -> exp141` 的已有证据是否真的构成“大改动”
   - 是否只是把旧的 `SCRC/SCKD` 换个名字再做一遍
2. **单变量隔离**
   - 相对 `exp030a` 到底只改了一个核心机制没有
3. **代码正确性**
   - shape / mask / 索引 / dtype / device / autograd
   - AMP 安全
   - 默认路径是否完全不受影响
4. **训练/测试一致性**
   - train 阶段的 completion 逻辑与 test 阶段是否一致
   - 是否存在 train-only support leakage
5. **日志正确性**
   - 新增统计是否真的反映模块行为，而不是伪统计
6. **失败模式**
   - 恒定跳过
   - 恒定强覆盖
   - 只抄 prototype
   - completion 只改了日志没改到 downstream 表征
