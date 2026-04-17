# 我们与 KPR 的区别

## 为什么必须单独写清楚

`KPR` 是当前与我们最接近的近邻工作之一：
- 它也使用 keypoint / pose 信息
- 它也使用 Swin backbone
- 它也关注 occluded ReID
- 它也明确讨论“pose 信息是否应前移到编码阶段”

因此，后续论文不能把它模糊带过，必须写清楚边界。

## KPR 到底在做什么

`KPR` 的核心不是普通意义上的“pose-guided occluded ReID”，而是：

1. **它主打的问题是 Multi-Person Ambiguity (MPA)**
   - 一个 bbox 里可能出现多个人
   - 需要明确告诉模型到底检索哪一个目标人

2. **它的方法是 keypoint-promptable ReID**
   - 输入不仅有 image
   - 还可以输入 positive / negative semantic keypoints prompts

3. **它的方法重心是 prompt-aware encoding + part-based retrieval**
   - keypoint heatmaps 先 tokenization
   - 再与 image tokens 在编码前融合
   - 输出的是 body-part embeddings 和 visibility scores

4. **它依赖 promptable 训练与对应增强**
   - 使用 part prediction head
   - 使用 human parsing supervision
   - 使用 BIPO 生成人工 inter-person occlusion 来强迫模型学习 prompt

## 我们当前这条线在做什么

我们当前 PRCV 主线的核心是：

1. **主问题**
   - 标准 occluded ReID 设定下，如何在不改变任务输入形式的前提下，把 pose 先验注入表征学习阶段

2. **主机制**
   - `PSG`: 在 backbone 中间 stage 做 pose-guided spatial gating

3. **结构补充**
   - `GCN`: 显式建模 skeleton relational evidence

4. **系统形态**
   - prompt-free
   - 不要求 inference 时额外 keypoint prompts
   - 最终系统是 semantic-structural complementary representation，而不是纯 promptable part-based retrieval

## 两者的关键区别

### 1. 问题定义不同

- `KPR`: 解决 **谁是目标人** 的问题
- 我们: 解决 **如何在标准设定下利用 pose 改善表征学习** 的问题

更具体地说：
- `KPR` 面向的是 explicit target disambiguation
- 我们面向的是 prompt-free representation enhancement

### 2. 输入假设不同

- `KPR`: 允许并鼓励在测试时输入 positive / negative prompts
- 我们: 不引入新的测试输入接口

这点很关键，因为：
- `KPR` 的额外能力来自额外输入信息
- 我们的方法保持标准 ReID pipeline，不改变部署接口

### 3. pose 使用位置不同

- `KPR`: 在 encoder 之前融合 prompt tokens 与 image tokens
- 我们: 在 backbone 中间 stage 通过 `PSG` 做 gated injection

也就是说：
- `KPR` 是 **prompt-aware encoder**
- 我们是 **intermediate-stage pose-guided gating**

### 4. 表示与匹配不同

- `KPR`: body-part embeddings + visibility-aware part matching
- 我们: global + semantic branch + structural branch

进一步说：
- `KPR` 的最终检索核心是 mutually visible parts 的 part-based distance
- 我们当前系统更强调 semantic-structural complementary evidence 的组合

### 5. 训练监督不同

- `KPR`: part prediction supervision + prompt依赖训练 + BIPO
- 我们: `PSG` 主机制不依赖 prompt supervision

这意味着：
- `KPR` 更像一个完整的 promptable part-based framework
- 我们更像一个可插入标准 occluded ReID pipeline 的 pose-guided representation module + structural branch

## 论文里最该怎么写

### 最安全的一句话区分

> KPR addresses promptable target disambiguation under multi-person ambiguity, whereas our method focuses on prompt-free pose-guided representation learning under the standard occluded ReID setting.

### 中文对应句

> KPR 解决的是多人物遮挡下“目标人是谁”的 promptable 判定问题；而我们的方法关注的是标准 occluded ReID 设定下、无需额外 prompt 的 pose-guided representation learning。

### 再展开一层的 related work 写法

可以按下面逻辑写：

1. `KPR` 说明：把 keypoint 信息前移到编码阶段是有效的
2. 但 `KPR` 的目标是 promptable target disambiguation，依赖额外 keypoint prompts
3. 我们不改变标准 ReID 输入设定，而是在 backbone 内部通过 `PSG` 注入 pose 先验
4. 同时再通过 `GCN` 补充 explicit structural reasoning

## 我们不能再写的说法

下面这些说法现在都不安全：

1. “现有方法都只在编码后使用 pose”
   - 错，因为 `KPR` 明确把 prompts 前移到了编码前

2. “我们是首个把 pose 注入编码过程的 occluded ReID 方法”
   - 也不安全，因为 `KPR` 已经明确 claim prompt-aware encoding

3. “KPR 和我们本质一样，只是它用了 prompts”
   - 这会把它的主问题定义抹掉，也会把我们的边界说乱

## 我们仍然可以安全 claim 的东西

1. 我们的方法是 **prompt-free** 的
2. 我们的方法保持标准 ReID 输入与部署形式
3. 我们的核心机制是 **intermediate-stage PSG**
4. 我们还有 `GCN` 作为 explicit structural branch
5. 我们的 two-stage 选择来自结构分支依赖性的实验依据

## 对当前 story 的直接影响

这件事带来的直接结论是：

1. 论文不能再用“所有前人都只做 post-hoc pose usage”这种大话
2. 更准确的写法应该是：
   - 多数 prior work 在编码后利用 pose
   - `KPR` 是 prompt-aware encoding 的例外
   - 我们进一步关注标准设定下的 prompt-free intermediate-stage pose injection

3. 如果主线继续写 `PSG`，最稳的落点是：
   - 不依赖 prompts
   - 不改任务输入形式
   - 通过中间层 gating 注入 pose
   - 通过 `GCN` 补充显式结构证据

## 参考来源

- KPR 论文：<https://arxiv.org/abs/2407.18112>
- KPR HTML：<https://ar5iv.labs.arxiv.org/html/2407.18112v1>
- KPR 代码与 README：<https://github.com/VlSomers/keypoint_promptable_reidentification>
- BPBreID：<https://arxiv.org/abs/2211.03679>
