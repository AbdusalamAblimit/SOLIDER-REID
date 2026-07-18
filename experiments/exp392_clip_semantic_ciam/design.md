# 实验 exp392：CLIP 语义校准的可辨识解剖中介路由

> 状态：`DESIGN-ONLY / RESEARCH-ONLY / NO-START`
>
> 当前不实现代码、不创建config/output、不启动GPU任务。exp391的纯结构NO-GO继续封板，但不永久
> 否定本实验中语义校准后的多阶段版本。

## 动机

official clean TAPF在Occluded-Duke三seed上得到`+0.47±0.31 mAP`，说明完整`anchor+PSG`并非完全
无效；但rank均值不正，旧内部field审计又显示correct、joint-channel shuffle与wrong field近似不可分。
这意味着当前anchor可能只提供一般条件扰动，PSG不必知道“哪个通道对应哪个人体部位”。

exp391把两层pose loss从sum改为mean后恢复了HT0，并证明early route非dead，但final仍比单层D0低
`0.4 mAP`。因此本实验不再继续复制PSG或调loss，而是重写问题对象：让内部anatomical state同时
满足语义可辨识、NULL identity、真实下游执行和可冻结反事实干预。

## 核心假设

1. 冻结CLIP双编码器可以从pose限定的局部RGB patch中提供sample-specific visual-support语义，
   但该假设必须先通过teacher-only门禁，不能预设成立；
2. 只有把语义绑定写入consumer的gather-transform-scatter算子，才能避免自由`17→32`卷积吸收
   joint identity；
3. 单层semantic mediator成立后，consumer-balanced的多stage direct anchors可能获得额外增益；
   exp391只否定无语义校准的`6/2`纯结构链，不否定该新机制。

## 技术方案

### 1. 冻结双编码器teacher

```text
aligned pedestrian RGB -> frozen CLIP image encoder -> patch tokens V
body-part prompt bank   -> frozen CLIP text encoder  -> prototypes T
pose region mask H_j pools V -> local visual feature v_j
q_j = softmax(cos(v_j, all T) / tau)
```

- 两个encoder永久`eval/no_grad/requires_grad=False`，不进optimizer；
- RGB与student使用同一几何增强参数，CLIP normalization单独执行；
- 17-joint pose继续负责geometry，CLIP首版只校准
  `head/torso/arms/upper-legs/lower-legs`的visual support；
- text prototype使用固定、多模板平均prompt bank；不得学习ID prompt；
- CLIP不蒸馏final ReID descriptor，不建立language retrieval或part-token matching分支；
- 推理删除两个CLIP encoder、文本原型和external pose。

### 2. Ordered-Coarse CIAM state

每个stage anchor直接预测绝对状态，不读取上一field、不预测offset：

```text
A_s = {M_s,c, r_s,c}, c in coarse anatomical regions
```

- `M`是空间support，`pi=Normalize(M)`只用于gather；
- `r`是单独的visual-support/reliability amplitude；
- fixed pose incidence matrix定义17 joints到coarse regions的语义映射；
- CLIP只校准sample-specific support/reliability，不重复声称发现已知part name；
- `M=0,r=0`时consumer必须逐元素identity。

### 3. 可执行semantic router

用受限低秩router替换原field-only PSG：

```text
z_c       = Wz Pool(F_cons, stopgrad(pi_c))
u_c(p)    = U_c * sigma(V F_cons(p) + C z_c)
DeltaF(p) = sum_c r_c * M_c(p) * u_c(p)
F'(p)     = F(p) + alpha * tanh(DeltaF(p))
```

- `V/C/Wz`跨regions共享，`U_c`为语义specific低秩expert；
- `alpha`零初始化并有固定上界；所有执行模块推理保留；
- state进入router前detach，ReID loss不把anchor改造成自由identity code；
- pose/semantic loss读取`stopgrad(F_src)`，不得回流backbone；
- 不允许per-joint bias或独立高容量projector形成捷径。

### 4. CLIP校准损失的最小无projector版本

首版优先使用：

```text
q_teacher = text_logits(pool(pose_region_mask, frozen_CLIP_patch_tokens))
q_anchor  = text_logits(pool(predicted_region_mask, frozen_CLIP_patch_tokens))
L_sem     = KL(q_teacher || q_anchor)
```

梯度只能通过predicted mask回到anchor。若后续增加stage ReID feature projector，必须作为独立变量，
并加入projector-only/static control。

### 5. semantic multi-stage

只有single-stage CIAM通过全部语义与检索门禁后，才允许比较：

```text
Stage-0 direct state -> Stage-1 one router
Stage-1 direct state -> Stage-2 one router
Stage-2 direct state -> Stage-3 one router
```

三个anchor参数独立、都直接预测绝对状态；每层一个主要consumer，均有真实descriptor下游路径。
多阶段直接对照是同一CLIP teacher、同一router数学、同一loss budget的semantic single-stage，而不是
原始D0或exp391 H2-M。

## 对照组与固定顺序

### Phase 0：零训练门禁

1. `0A`：已封板clean D0内部student field frozen audit；
2. `0B`：coarse-region CLIP teacher-only audit；
3. `0C`：CPU/CUDA/AMP route、NULL identity、梯度所有权和counterfactual unit gate。

Phase 0完成前不得实现正式训练config。

### Phase 1：单层机制归因

严格逐臂、fresh、final-only：

1. current clean D0；
2. parameter-matched Generic-LR-Adapter；
3. Router17：原17-channel pose field + 新router数学；
4. Ordered-Coarse-CIAM：fixed pose semantics，无CLIP；
5. Ordered-Coarse-CIAM + CLIP support calibration；
6. Expert-mean、static-state与train-only CLIP visual K-means ontology强对照。

每一步只改变一个核心变量；前一步未通过时，后一步保持`NO-START`。

### Phase 2：重新尝试多阶段

只在Phase 1 full通过后启动：

1. semantic single-stage；
2. parameter/loss/consumer matched semantic multi-stage direct anchors；
3. 逐stage frozen bypass与correct/mismatched state干预。

## 零训练teacher门禁

必须只读train split并报告：

1. coarse-region expected top-1、margin、per-class confusion和bootstrap 95% CI；
2. correct mask相对channel-shuffle、uniform、random、matched-wrong mask；
3. 固定mask换wrong RGB，固定RGB换wrong mask，固定visual feature置乱text labels；
4. teacher distribution跨样本方差、JSD与有效秩，排除固定常量；
5. visible/low-confidence/invalid/合成遮挡分组的entropy与margin；
6. 水平翻转后的空间反翻与left/right映射equivariance；
7. CLIP权重、tokenizer、prompt bank、预处理和artifact SHA。

以下任一成立即停止当前teacher定义，但不永久否定换粒度后的CLIP方向：

- correct mask不优于random/shuffle；
- full teacher对wrong RGB或wrong text不敏感；
- distribution近似常量；
- entropy与pose confidence/遮挡分组无可辨关系；
- 左右细关节不高于随机时，禁止升级到17-joint CLIP KD，保留coarse region审查。

## 训练与因果门禁

所有正式arm必须120 epoch、batch64、SGD、lr0.0008、eval10、checkpoint120、fresh串行、final-only，
不得按中途best裁决。至少满足：

1. correct internal state相对joint/region permutation与matched-wrong state `>= +0.3 mAP`；
2. NULL state严格identity，router bypass有限非零；
3. Generic-LR-Adapter、Expert-mean、static state和CLIP visual cluster不能复现full增益；
4. single-stage CIAM相对current D0期望`>= +1.0 mAP`且R1不下降，才作为主方法GO；
5. semantic multi-stage必须相对semantic single-stage再有清楚final增益，且至少两个stage frozen bypass
   各自`> +0.1 mAP`，才把多阶段升为贡献；
6. correct/shuffle/None/exploding external pose仍严格exact，query/gallery RGB-only；
7. 所有anchor、router、expert与consumer有参数轨迹、strict finite和最终descriptor可达性。

语义敏感性成立但retrieval不升时，只能说明诊断被修复，不能包装成性能方法。性能上涨但语义干预
失败时，只能称generic router/CLIP KD，不能称anatomical mediator。

## 预期结果

成功情形不是“加CLIP涨点”，而是同时看到：

- teacher确实依赖当前RGB、pose region和文本原型；
- consumer对正确geometry-semantic binding敏感；
- single-stage超过D0且强generic controls；
- semantic multi-stage在相同机制下进一步超过semantic single-stage；
- 推理仍只有RGB与单一global descriptor。

## 风险与失败解释

1. CLIP patch token对细人体部位分辨率不足：退回coarse regions，不强行17 joints；
2. pose mask已经决定part label，CLIP distribution退化为常量one-hot：CLIP无新增信息，保留pose-only
   CIAM对照；
3. expert collapse：收益来自generic adapter，不得归因人体语义；
4. router提升但CLIP无增量：consumer重构成立，CLIP不是贡献；
5. single-stage成立、多stage不增益：保留单层主方法，多阶段进入负消融；
6. multi-stage在CLIP后成立：这是新semantic mechanism的证据，不回写成exp391成功。

## 当前裁决

`exp392 = DESIGN-ONLY / RESEARCH-ONLY / NO-START`。当前只允许继续文献、公开代码和机制审查；
不得创建训练config/output或占用4090。下一可执行动作是Phase 0A/0B的审计设计复核，而不是正式训练。
