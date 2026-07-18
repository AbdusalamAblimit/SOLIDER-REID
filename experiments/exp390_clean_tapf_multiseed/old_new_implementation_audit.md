# 旧 exp378 TAPF 与官方 clean TAPF 的实现差异审计

## 审计范围与口径

本审计只回答两个问题：旧 exp378 D0 与 exp387/exp390 clean D0 在实现和训练配方上有哪些可验证
差异，以及为什么旧结果呈现更大的 `D0−B0`。审计不修改正在运行的 exp390 代码、config 或训练，
不复用旧 `pose_data`，也不把相关性写成因果结论。

- 旧实现：exp378 exact commit `ca62c475b43f17564bb09ede90de6eed53dd2d88`，matched
  `RE_PROB=0`；
- clean 实现：exp387 exact execution commit
  `0d1822a07dda8daac0210b68916035b1886d5d99`，exp390 继续使用其
  `HIERARCHICAL=False` D0 路径；
- 指标均为 Occluded-Duke、Swin-Tiny、seed1234、自然 e120 final，不取 best。

## 先校正“旧实现涨得更多”的含义

| 体系 | B0 mAP/R1/R5/R10 | D0 mAP/R1/R5/R10 | D0−B0 |
|---|---:|---:|---:|
| exp378 旧配方 | `55.1/66.7/79.5/83.8` | `56.2/67.6/79.8/83.4` | `+1.1/+0.9/+0.3/−0.4` |
| exp385/387 official clean | `57.4/67.4/80.6/85.2` | `57.6/67.7/80.8/84.6` | `+0.2/+0.3/+0.2/−0.6` |

clean D0 的绝对 mAP 比旧 D0 高 `+1.4`，并不是旧 D0 本身更强。旧体系的增量更大，首先来自其
B0 比 official clean B0 低 `2.3 mAP`。因此当前证据只支持“旧配方下的相对增量更大”，不支持
“旧模块达到更高绝对性能”。

## 已证实的实现与配方差异

| 维度 | exp378 旧实现 | exp387/390 clean 实现 | 可验证影响边界 |
|---|---|---|---|
| Random Erasing | `RE_PROB=0.0` | `RE_PROB=0.5` | B0 与 D0 在各自体系内 matched，但两体系不是同一增强配方 |
| B0 loader | 为保持 sampler parity，旧 B0 仍设置 `POSE_ENABLED=True` 并走 pose-aware loader，只是 PSG stage 为空 | clean B0 走 official RGB `ImageDataset`；D0 才使用 paired clean pose loader | 两体系内部 matched，但跨体系还包含 loader 实现差异 |
| teacher 来源 | 旧 pose cache 中的 target-person ViTPose dense heatmap+score，经旧 paired transform 后使用 | exp386 从原始 train RGB fresh 生成的 ViTPose-H COCO-17 坐标+score，训练现场渲染固定 Gaussian | teacher 的空间信息、数值域和不确定性表达不同 |
| teacher/anchor field | `17×96×32`；dense posterior，经 moments 后重渲染 Gaussian | Stage-2 原生约 `17×24×8`；坐标直接渲染固定 `sigma=1.5` Gaussian | clean field 带宽更低，并丢弃原始 dense heatmap 形状 |
| anchor | hidden64 LiteHR-style 双分辨率 decoder，多次上采样和 residual refine；spatial softmax posterior+独立 confidence head | hidden128 的 `1×1 project→depthwise 3×3→GN→GELU→head`；sigmoid heatmap×confidence | 结构不同；旧 anchor 参数 `59,874`，clean anchor约 `55,202` |
| Stage-3 PSG | 每个 bank `17→64→768`，带 bias、ReLU；输入再做 sigmoid；输出 gate 无界，`x·(1+gate)` | 每个 bank `17→32→768`，无 bias，GN/GELU；不二次 sigmoid；`x·(1+0.5·tanh(delta))` | 旧两 bank `102,144` 参数，clean 两 bank `50,240`；clean 调制显式有界 |
| pose objective | spatial KL + `0.25×confidence BCE`，外层权重 `1.0` | Gaussian heatmap MSE + confidence BCE，外层权重 `0.1` | 旧 anchor 的持续姿态约束更强，目标也不同，loss 数值不能直接横比 |
| TAPF 总附加参数 | TAPF anchor+未激活的 geometry adapter=`85,542`，另有 PSG=`102,144`；D0 的 adapter 无输出路径且不更新 | D0 总新增=`105,442` | 旧 D0 的有效 anchor+PSG 容量约 `162,018`，明显大于 clean 的 `105,442`；dead adapter 不能解释增益 |
| AMP | 显式初始 scale `1024` | 默认动态 scale `65536`，正式 paired gate/训练中自动回退 | 会改变数值轨迹，但没有证据表明是主要增益来源 |
| eval protocol | eval batch64；旧 defaults 令 `TEST.FLIP_TEST=True`，processor 实际做原图/翻转双前向平均 | eval batch256；official clean processor 没有 flip-test 路径 | 各体系内部 B0/D0 matched，但跨体系绝对值和增量又多一层 protocol 差异 |
| checkpoint | checkpoint10 | checkpoint120 | 不改变自然 e120 训练目标，不能解释训练端相对增益 |

两版也有重要共同点：都是 Stage-2 生成 RGB 内生场，随后在 Stage-3 两个 Swin block 后分别经过
两个独立 PSG；都是 e1--5 teacher、e6--10 handoff、e11--120 student；D0 都持续 pose
supervision，ReID loss 都不写入 anchor；测试期都只由 RGB 生成 descriptor。official teacher 权重也已做
逐 tensor 对照，不是两体系的差异来源。

## 为什么旧体系呈现更大增量

### 1. 最强解释：official clean B0 更强，且 Random Erasing 与 TAPF 可能存在收益重叠

这是唯一直接由 final 数字确定的部分：clean 配方先把 B0 从 `55.1` 提到 `57.4 mAP`，而 clean D0
达到 `57.6`。旧实验为避免 RGB erase 与旧 dense pose cache 错位，整组关闭 RE；official clean
paired pipeline 则允许 B0/D0 都使用 `RE_PROB=0.5`。TAPF/PSG 可能提供的遮挡正则化或空间重标定，
在强 RE baseline 上边际收益被压缩。该机制解释目前只是高优先级假设；没有 official clean runtime 上
`RE on/off × B0/D0` 的完整 2×2，就不能写成因果结论。

### 2. 旧 teacher/anchor 保留了更高空间带宽

旧路径消费 `96×32` dense ViTPose heatmap，其峰形、扩散和多峰结构仍可进入 posterior/moments；clean
路径只保存 COCO-17 坐标和 score，并在约 `24×8` field 上固定 Gaussian 渲染。clean 目标更干净、
provenance 更可信，但表达更窄。它可能减少 PSG 可利用的空间统计，也可能只是删掉旧 cache 中的
nuisance；现有结果无法区分两者。

### 3. 旧 PSG 容量更大、输出无界，可能更像可学习的通用空间重标定器

旧 PSG hidden64、带 bias，参数约为 clean PSG 的两倍，且 gate 不受 `tanh` 上界约束。旧 D0 冻结
checkpoint 的语义干预显示：correct、shuffle、None、joint/confidence permutation、constant/zero field
与 correct 的 mAP 差都小于 `0.1` 个百分点；但真正 bypass PSG 会下降约 `2.6829 mAP`。因此旧
`+1.1 mAP` 不能主要归因于正确关节语义，更符合“PSG 容量、优化正则化或通用重标定产生收益”的
解释。旧两个 PSG 的参数量是 clean 的约 `2.03×`；旧 PSG 的 bias 又按共同
`BIAS_LR_FACTOR=2` 规则使用 2× base LR，而 clean PSG 没有 bias。若把 dead adapter也计入模型，
旧总新增约 `187,686`，是 clean `105,442` 的约 `1.78×`。clean PSG 刻意减宽且有界，换来了
更严格的机制可解释性，也可能削弱这种非语义容量收益。

### 4. 旧 pose loss 更强，可能提供更强辅助正则化

旧外层 pose 权重为 `1.0`，clean 为 `0.1`，而且 KL 与 MSE 的目标不同。旧 anchor 持续拟合 dense
teacher 的训练作用可能更强。不过两种 loss 的原始尺度不同，不能仅凭 `1.0 vs 0.1` 推导十倍梯度；
并且旧 D0 相对同实现 hard F0 只有约 `+0.3 mAP`，F0/N0/R0/RG0 也普遍落在相近的
`+0.8～+1.1 mAP` 区间。这说明旧体系的大部分收益是各 arm 共有的 anchor/PSG/训练效应，而不是
D0 独有的持续 pose supervision。该项优先级低于 baseline/RE 与 PSG 容量解释。

### 5. 旧 flip-test TTA 是额外 protocol 差异，不应误写成训练机制

旧 exp378 config 未显式关闭 `TEST.FLIP_TEST`，旧 defaults 为 `True`，训练中每个 e10 eval 和 final
都会平均原图/水平翻转 descriptor；official clean processor 没有这条双前向路径。旧 B0/D0 都使用
同一 TTA，clean B0/D0 也各自 matched，因此它不能单独证明旧 `D0−B0` 从何而来；但 TTA 对两臂
可能有非线性交互，进一步禁止把两套 delta 当成只隔离实现的直接对照。

### 6. 当前没有 RGB 内容差异证据

旧资产审计与 clean 数据都指向 canonical `/mnt1/afrdata/Occluded_Duke`，split/count 一致。两次
manifest/aggregate 算法没有做跨时点逐文件 exact 对齐，所以不能声称已经逐文件证明相同；但也没有
证据支持“旧数据更容易”或“clean 数据变了”的解释。预训练 teacher 已有 213/213 state 逐 tensor
exact，因此 teacher 权重差异明确排除。

## 当前裁决与后续最小因果对照

1. 不能恢复旧 runtime、旧 pose cache 或旧 path mapping来“救点”；它们不满足 official clean
   provenance 边界。
2. 不能说 clean 实现失败：它的绝对 D0 更高，并已通过 config-off、state/RNG/optimizer、CUDA/AMP、
   overflow、strict state、pose-free 与参数轨迹门禁；当前问题是 paired 边际效应是否稳定。
3. 先自然完成 exp390 的三 seed paired 证据。只有在四臂全部封板后，才考虑预注册 official clean
   runtime 上的 `RE on/off × B0/D0`；这是区分“强 baseline/增强重叠”与“实现表达不足”的最小因果
   对照，不能插队或修改当前 arm。
4. 若 RE 2×2 不能解释差异，下一层才应在 fresh pose artifact 与 clean runtime 内逐个隔离 PSG 宽度/
   有界性、field resolution/renderer、pose objective，仍禁止一次恢复多个旧变量。

## 独立审查记录

本轮另启用只读独立子 agent，明确禁止 Claude、GPU、文件修改、提交和训练。独立审查同意上述主
排序，并补出了旧 flip-test TTA、旧 PSG bias 的 2× LR、旧 B0 pose-aware loader三项主审遗漏；
没有发现能支持“旧 D0 绝对更强”或“teacher 权重不同”的反例。exp387 启动前文档只有 executable
gates 与主审记录，没有充分证据证明当时已经完成同等级独立静态审查，因此不能追溯性声称此前
已经使用；本轮是在等待 exp390 时补齐该审查层。
