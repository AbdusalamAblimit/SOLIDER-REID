# 实验 exp411：PCMPSR（Pose-Complete Multi-Positive Set Ranking）

## 动机

exp408证明pose-indexed CLIP局部关系可以进入Stage-2，但最终mAP不涨；exp409把pose×CLIP直接用于一个hard
positive/negative，得到R1上升而mAP下降；exp410用跨图五槽CLIP support构造冻结classifier，又因外部CLIP轴与
ReID student空间严重失配而大幅退化。三者共同要求新对象同时满足：不回归CLIP坐标、不只选择单边、保留student
自组织身份空间，并让同PID多视图完整支持与所有负身份集合直接进入最终排序梯度。

PCMPSR把标准PK batch视为一个leave-one-position-out身份集合任务。pose与CLIP不提供连续loss尺度，而只在每个
身份的同大小支持集中，按五个解剖槽离散选择最可信的owner；final student descriptor到整个身份集合的距离包含
所有支持图一次，再把五个slot owner各计一次。这样外部证据只决定同PID支持的解剖multiplicity，不规定特征轴。

## 核心假设

遮挡query的AP依赖全部同ID视图相对全部异ID身份集合的排序，而不是单个最难pair或冻结类别中心。若每个身份集合
先保留全部视图，再由pose coverage与同PID CLIP槽共识强调不同解剖槽的可靠owner，student可以在自身global
descriptor空间学习“完整身份支持集”距离；这应同时改善完整检索排序与首位命中，并避免PC²P的外部轴错配。

## 技术方案

### 1. 固定训练与fresh证据

- 保持Swin-Tiny、batch64、`P×K=16×4`、seed1234、learned classifier CE、D0 pose loss、backbone和eval不变；
- 为official train 15,618张图fresh生成五槽region-isolated frozen CLIP visual feature，schema、asset与SHA均为
  exp411独立版本；CLIP只用于owner选择，测试不读cache/CLIP/外部pose；
- 当前增强后的COCO-17 pose生成五槽soft visibility `v_jr`；CLIP槽feature只在cache-valid时参与共识；
- PCMPSR与PICRD/PCHM/PC²P互斥，默认关闭时forward、loss、state与RNG保持clean D0 exact。

### 2. 等支持数的leave-one-position-out集合

每个PK batch内按sampler顺序给每个样本确定类内位置`q∈{0,1,2,3}`。对anchor `i`及batch内每个身份`z`，
定义支持集`S(z,q)`为该身份的四张图中排除同一位置`q`后的三张图。正身份因此排除anchor本身，15个负身份也各
排除一个位置，保证所有身份集合严格同为`K-1=3`个支持，避免负集合凭样本数获得距离优势。

对每个`S(z,q)`与槽`r`，先计算该集合valid CLIP槽feature的归一化均值`mu(z,q,r)`，再离散选择：

`o(z,q,r) = argmax_{j in S(z,q)} v[j,r] * cosine(c[j,r], mu[z,q,r])`。

只有pose-visible且CLIP-valid的候选可成为owner；并列按较小batch index确定性破同。这里没有可学习head、
temperature、top-k、rank fusion或连续loss weight。owner score只用于argmax并stop-gradient。

### 3. student空间中的完整集合距离与全身份排序

令`g_i`为最终global descriptor，距离使用与原triplet相同的L2/可选`TRP_L2`设置。定义：

`D(i,z) = [sum_{j in S(z,q)} d(g_i,g_j) + sum_{r=1}^5 d(g_i,g_o(z,q,r))] / 8`。

每个身份的三张支持图至少出现一次；五个owner通过离散multiplicity额外强调其可靠解剖支持。同一图可拥有多个槽，
这正是集合结构而非伪造feature。PCMPSR以温度和margin均为零的稳定listwise surrogate替换原batch-hard triplet：

`L_set = log(1 + mean_{z != y_i} exp(D(i,y_i) - D(i,z)))`，再对anchor取均值。

实现使用`logmeanexp`与`softplus`避免溢出；在所有身份距离相同时loss为`log(2)`，与原soft-margin triplet量级对齐。
learned CE与D0 pose loss及既有loss weight不变，模型与eval代码不新增分支。

### 4. 强反事实

所有control共享anchor、PID、等大小支持集、student feature与tie-break，只改变owner evidence：

- `correct`：正确五槽pose coverage × 正确同PID CLIP槽共识；
- `zero-owner`：删除五个owner multiplicity，只保留每个身份全部三张support的纯set ranking；
- `generic`：五个槽均改用同一full-image/generic CLIP medoid owner，关闭解剖分槽；
- `wrong-RGB`：owner选择所用CLIP槽按固定different-PID循环置换，但所有support仍严格保持原PID；
- `pose-only`：owner只按pose visibility选，不使用CLIP同PID共识；
- `D0`：sealed clean D0原batch-hard triplet。

首轮只训练`correct`。control先在共享真实batch上证明owner与集合距离确实改变；只有correct自然e120性能GO后，才
串行训练最关键的`zero-owner`和`wrong-RGB` matched controls，避免为已失败机制消耗GPU。

## 对照组

- 主性能对照：sealed clean D0 seed1234/e120 raw mAP/R1=
  `57.5587756578/67.6923076923`；
- 对象边界：exp409 PCHM `57.0/68.6`（单pair导致R1/AP分裂）；
- 空间边界：exp410 PC²P `45.0/56.4`（冻结外部classifier轴失配）；
- 机制对照：zero-owner、generic、wrong-RGB、pose-only及D0。

## 预期结果与裁决

启动前只做必要门：

1. 每个anchor的16个身份支持集均为3张，正集排除self且15个负集严格different-PID；
2. owner只来自对应PID支持集，pose/CLIP无效槽有确定性fallback，所有index finite且可复现；
3. correct平均slot-owner unique数大于1，且相对zero/generic/wrong-RGB/pose-only至少各有非零owner或集合距离变化；
4. default-off与clean D0同RNG真实forward/loss exact；
5. 真实PK64 CUDA/AMP下`L_set` finite，final descriptor、Stage-3和backbone梯度finite/nonzero并取得真实update。

唯一fresh correct seed1234自然训练到e120。只有e120 raw mAP与R1同时严格超过clean D0 raw双门才判
`PCMPSR PERFORMANCE GO`；任一失败立即`EXP411 SEALED NO-GO`，不得调loss、batch、owner公式或支持数救旧臂。
性能GO后，correct还必须胜zero-owner与wrong-RGB，才允许形成pose+CLIP归因和C类主方法候选。

## 风险与失败解释

1. `K-1=3`支持过小，五槽owner可能坍缩到同一图；若真实batch机制门不active，不启动GPU训练。
2. train图整体可见度较高，pose owner multiplicity可能只重复普通medoid；由zero/generic/pose-only直接解释。
3. set ranking可能弱化最近负样本，R1下降；若mAP/R1任一不过门，说明当前全身份集合对象仍不能替代D0 triplet。
4. wrong-RGB若不改变owner，CLIP槽共识没有独立作用，机制不得声称pose+CLIP。
5. 普通SupCon、lifted/listwise metric、episodic set loss、pose sampling均为已知原子；创新仅限“等支持
   leave-one-position-out身份集合 + pose×CLIP五槽owner multiplicity + final student全身份排序”的整体。
