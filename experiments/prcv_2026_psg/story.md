# 当前论文故事（PRCV / PSG 线）

## 一句话故事

现有 pose-guided occluded ReID 大多在特征形成之后再使用 pose 信息；我们提出 `PSG`，将 pose 先验前移到 backbone 表征学习阶段，并在最终系统中引入 `GCN` 结构分支做显式 skeleton relational reasoning，形成 semantic-structural complementary evidence。

## 主张层级

1. **主创新**
   - `PSG`
   - 写法：在 backbone 中间 stage 之间注入 pose-guided spatial gating

2. **最终实现**
   - `2-stage PSG`
   - 写法：最终采用 two-stage instantiation
   - 不要在标题、摘要、引言里把它单独抬成新的主术语

3. **结构补充**
   - `GCN`
   - 写法：structural pose branch / explicit skeleton relational evidence
   - 作用是补充 `PSG`，不是与 `PSG` 并列的第二主创新

4. **系统资产**
   - `LGPA-D`
   - `OA-SD`
   - `PLBOA`

5. **附加评测资产**
   - `MaxSim`
   - `POT`
   - `flip`

6. **可选补充 benchmark**
   - `Occluded-PoseTrack-ReID`
   - 作用：作为多人物遮挡场景下的补充泛化评测，不替代 `Occluded-Duke` 主战场

## 结果汇报协议

从当前 PRCV 写作开始，结果表统一采用下面的汇报口径：

1. **默认测试协议包含 flip-test**
   - `flip-test` 视为默认测试增强
   - 不把它单独拔成主创新或单独贡献点

2. **`MaxSim` 必须单独占一行**
   - 因为它改变的是测试时匹配函数
   - 它不能和默认测试协议混成同一行主结果

3. **主结果表建议写法**
   - `Ours`：默认测试协议结果（包含 flip-test）
   - `Ours + MaxSim`：单独一行
   - 如有需要，再加 `Ours + POT`

4. **文字叙述口径**
   - 主模型结果按默认测试协议汇报
   - `MaxSim` 写成 additional inference-time matching module

5. **当前限制**
   - 若历史实验尚未记录默认协议下的 `equal_concat + flip-test` 数字，不在文档中凭空补写
   - 后续如用于论文主表，需补测并单独记录

## 当前最硬的证据

1. `PSG` 本体成立
   - `exp007`: `58.3 / 67.9`
   - 3-seed mean: `57.83 / 67.13`

2. `GCN` 作为结构分支值得写进最终方法
   - `exp246` / `exp249` 已说明 `LGPA-D + GCN` 具备互补性
   - 但更适合写成 structural branch，而不是单独主创新

3. 当前训练端最强系统是 `exp255`
   - `Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA`
   - `FINAL = 73.2 / 83.3`

4. `2-stage PSG` 的当前最强依据来自 `exp255 vs exp255b`
   - `exp255`: `GCN512 + 2-stage PSG = 73.2 / 83.3`
   - `exp255b`: `GCN512 + 1-stage PSG = 71.5 / 81.9`
   - 解释：最终采用 `2-stage PSG` 是有依据的，尤其在高容量结构分支上

## 写作口径

### 标题 / 摘要 / 引言

- 只讲 `PSG`
- 可以写：我们在 backbone 中间 stage 之间注入 pose 信息
- 最多补一句：最终实现采用 two-stage instantiation

### 方法部分

- 先把 `PSG` 定义成一个通用的 pose-guided spatial gating 机制
- 再说明：实际实验中采用 `2-stage PSG` 作为最终配置
- `GCN` 写成 structural pose branch

### 消融部分

- 回答为什么最终选 `2-stage`
- 用 `1-stage / 2-stage / 3-stage` 小表说明选择依据
- 不要把全文写成 “1-stage vs 2-stage” 的主战场
- 测试表中默认主行使用包含 `flip-test` 的协议，`MaxSim` 单独列为附加行

## 与 KPR 的关键边界

`KPR` 是当前必须正面比较的最近邻工作，但两者主问题并不相同。

1. **问题定义不同**
   - `KPR` 主要解决 multi-person ambiguity
   - 我们当前主线解决的是标准 occluded ReID 设定下的 prompt-free pose-guided representation learning

2. **输入假设不同**
   - `KPR` 在测试时可输入 positive / negative keypoint prompts
   - 我们的方法不要求额外 prompt 输入

3. **pose 使用方式不同**
   - `KPR` 把 keypoint heatmaps tokenized 后与 image tokens 在编码前融合
   - 我们用 `PSG` 在 backbone 中间 stage 做 gated injection，并用 `GCN` 做显式 skeleton relational reasoning

4. **表示与匹配方式不同**
   - `KPR` 是 part-based + visibility-aware retrieval
   - 我们是 global + semantic/structural complementary branches，并可配合 `equal_concat` 或 `MaxSim`

5. **训练依赖不同**
   - `KPR` 依赖 prompt 设计、part prediction head、human parsing supervision 和 BIPO augmentation
   - 我们当前主线不依赖 promptable 输入设定

### 当前最安全的 related work 区分句

> KPR addresses promptable target disambiguation under multi-person ambiguity, whereas our method focuses on prompt-free pose-guided representation learning under the standard occluded ReID setting.

## Occ-PTrack 的使用口径

`Occluded-PoseTrack-ReID` 可以作为补充 benchmark 加入论文，但定位必须收紧：

1. 它不是当前主 benchmark
   - 主 benchmark 仍然是 `Occluded-Duke`
   - `Occ-PTrack` 只用于补充说明方法在多人物遮挡场景下的泛化性

2. 最公平的对标对象是 `KPR w/o prompt`
   - 因为我们的当前主线是 prompt-free
   - 不应把 `KPR with prompt` 当作最低门槛

3. 结果解释优先级
   - 第一目标：超过 `KPRSOL w/o prompt`
   - 第二目标：至少超过 `SOLIDER / BPBreID`
   - 若连这两个 baseline 都不能超过，则不建议放入主文

4. 写法要求
   - `Occ-PTrack` 结果只能写成 secondary benchmark / supplementary benchmark
   - 不把它写成主卖点
   - 不为它额外展开一整套新消融

## 推荐贡献点

1. 提出 `PSG`，在 backbone 内进行 pose-guided spatial gating，而不是在特征形成后再做 pose-aware pooling 或 filtering
2. 构建 semantic-structural complementary occluded ReID system，其中 `GCN` 提供显式 skeleton relational evidence，`LGPA-D` 提供语义 part evidence，与 `PSG` 形成互补
3. 在 Occluded-Duke 上系统验证该框架，并采用 `2-stage PSG` 作为最终实现；实验表明该设计更适合支撑高容量结构分支，最终在 Swin-Small 上得到当前最佳训练端结果之一

## 摘要骨架

1. **问题句**
   - 现有 pose-guided occluded ReID 往往在特征提取完成后才利用 pose，因而对表征学习阶段的结构先验注入不足。

2. **方法句**
   - 我们提出 `PSG`，在 backbone 中间层通过 pose-conditioned spatial gating 直接调制特征形成过程。

3. **扩展句**
   - 在此基础上，我们结合 `GCN` 结构分支，以显式建模 skeleton relational evidence，并在最终实现中采用 `2-stage PSG` 作为具体配置，从而形成 semantic-structural complementary representation。

4. **结果句**
   - 在 Occluded-Duke 等基准上，该框架取得了当前项目最优结果之一；在默认测试协议与附加 `MaxSim` 匹配下，均表现出稳定竞争力，且消融进一步表明，最终采用的 `2-stage PSG` 更适合支撑高容量结构分支。
