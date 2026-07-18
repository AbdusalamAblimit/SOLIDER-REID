# 实验 exp393：CLIP-Owned Executable Residual（COER）

## 当前状态

`DESIGN-ONLY / PHASE 0 TEACHER AUDIT NO-START / FORMAL TRAINING NO-START`。

exp393不是exp392续训，也不是exp391 semantic multi-stage。它使用fresh repo、fresh optimizer、唯一final
checkpoint和完整120-epoch recipe；任何正式arm一旦启动必须自然跑满，不挑best、不续训、不换seed
救场。当前不创建训练config/output，不占用4090。

## 动机

exp392 Phase 0D已把失败定位到执行接口，而不是简单的“CLIP teacher不好”：

1. B2-SI证明PC-MBCLS五slot readout对局部遮挡有sample-specific单调响应；
2. Semantic C0 final比clean D0低`0.7 mAP`；
3. final checkpoint中五slot同slot跨图q std只有`0.00009–0.00029`；
4. static-q、q-one、spatial-constant、slot-cycle、expert-mean与router all-bypass的`|ΔmAP|<0.0007`，
   所有rank差严格为0；
5. 当前CLIP loss只监督anchor；state进入router前detach，router只靠global ReID loss；
6. router expert exact-zero初始化，初始时token/context projection被zero expert乘掉，形成双线性冷启动。

因此当前结构只是“拓扑上把CLIP state接入router”，没有让CLIP证据拥有可执行残差。新实验要回答：

> 若先消除identity-safe router的zero-expert冷启动，再让CLIP局部视觉证据直接成为router必经的
> 低维执行系数，semantic route能否获得可辨识的检索贡献？

## 核心假设

### H-route：优化断点假设

Semantic C0的route失活至少部分来自`zero expert × random latent`的冷启动，而非ReID完全不需要局部
路由。使用非零branch与zero ReZero scalar可以保持初始化严格identity，同时让scalar在首个finite step
获得梯度，随后branch全部参数获得梯度。

### H-CLIP：执行证据假设

0.5附近的单标量q丢失了局部视觉方向和相对关系。CLIP region CLS减去同图global/slot prior后的
centered rich evidence code若具有足够within-slot variance、wrong-RGB/mask敏感性和有效秩，并直接
控制router latent，而不是只监督terminal anchor head，则可能成为有检索价值的内部中介。

两个假设必须串行分开验证。Phase A失败只关闭当前ReZero route parameterization；Phase B失败只关闭
当前rich-code teacher/alignment接口。任一失败都不允许扩大成“CLIP–TAPF永久无效”。

## 总体实验链

```text
Phase 0E  teacher-only rich evidence audit
    |
    +-- FAIL: 重构teacher code，不启动训练
    |
    +-- PASS
          |
Phase A   RZ-C0：只修router初始化/scale，保留Semantic C0的M/presence/q与loss
          |
          +-- route仍retrieval-inert：封板RZ接口，Phase B不使用该接口
          |
          +-- route alive
                |
Phase B   COER：以RZ-C0为直接对照，只把scalar q执行变量换成rich CLIP evidence code
                |
                +-- single-stage语义因果与final性能成立后，才另开semantic multi-stage
```

## Phase 0E：rich CLIP evidence teacher-only审计

### Teacher输入与readout

保持B2-SI已验证的frozen OpenCLIP ViT-L/14 PC-MBCLS路径、hard-owner五slot ontology、与student同几何
RGB和训练期pose mask。每个slot读取region-conditioned CLS feature `v_r`，同时读取同图global CLS
`v_g`。不再把feature压成单个q：

```text
e_raw(r, x) = normalize(v_r) - normalize(v_g)
e_centered  = e_raw - mu_r
e_teacher   = normalize(P_frozen(e_centered))
```

- `mu_r`只由official train预审集合按slot估计并冻结；
- `P_frozen`首版使用teacher-only训练集协方差的top-K PCA basis，`K=16`，只在teacher侧离线拟合，
  不进入student optimizer；
- 不使用batch covariance，不用query/gallery拟合，不用identity label拟合PCA；
- text encoder只用于已有slot ontology/审计，不把part-name分类准确率当code目标；
- teacher仍`eval + no_grad + requires_grad=False`。

PCA不是创新点，只防止把768维CLIP各向异性直接灌入小router。若PCA审计失败，不换seed或偷偷挑K；
只允许在独立设计中比较预注册的fixed random orthogonal control。

### Teacher门禁

先做synthetic exact、8图contract，再做128图，最后才允许official train全量teacher-only审计。至少报告：

1. 每slot centered code的逐维std、effective rank、top singular value占比；
2. correct vs wrong RGB、wrong mask、same-mask different-PID、slot-cycle的cosine/InfoNCE margin；
3. 同图target region相对non-target region的局部遮挡响应；
4. horizontal flip在arms/legs合并ontology下的一致性；
5. text-only、slot-mean、random orthogonal、raw uncentered code强对照；
6. matched GPU吞吐与显存。

预注册最低门禁：五slot都必须具有非零within-slot variance；macro centered effective rank至少`8/16`；
correct相对wrong RGB与wrong mask的paired margin逐slot PID-cluster CI均大于0；slot-mean不能解释主要
variance。任一失败先归因并封板当前teacher code，不启动Phase A/B训练。

## Phase A：RZ-C0 route activation control

### 唯一变量

保持Semantic C0的PC-MBCLS `M/presence/q`、anchor heads、semantic loss、两个consumer位置、rank16、
参数预算、batch64、seed1234、SGD、lr、handoff和120-epoch recipe不变。只替换router初始化/scale：

```text
old: expert = 0
     F' = F + 0.5 * tanh(delta(expert, token, context))

new: expert ~ small nonzero variance
     alpha_logit = 0
     F' = F + 0.5 * tanh(alpha_logit) * tanh(delta(expert, token, context))
```

初始化仍严格`F'=F`，NULL mask/q仍严格identity；但首个finite step应有非零`alpha_logit`梯度。alpha打开
后，token projection、context projection和expert都必须获得finite非零梯度。禁止同时加入rich code、
新loss、调q温度或改变route位置。

### 参数匹配

新增每router一个scalar；为保持参数matched，可从unused诊断常量中扣除不影响执行的两个scalar，或
明确报告`+2`参数且FLOPs变化为0。不得通过扩大rank或expert数补偿。

### Phase A门禁

CUDA/AMP preflight必须证明：

1. 初始化full与all-bypass descriptor逐tensor exact；
2. NULL mask/q exact identity；
3. 第一个finite step alpha梯度非零，随后连续8个finite step三类router参数都更新；
4. teacher/anchor/router/backbone/ID head梯度所有权与Semantic C0一致；
5. 24步内correct与all-bypass descriptor gap从0增长且finite；
6. checkpoint strict、RGB-only、state finite、teacher隔离PASS。

正式e120后，除final性能外必须做完整all-router-bypass。只有final full−all-bypass `>=+0.1 mAP`且
full不比Semantic C0低超过`0.2 mAP`，才称`RZ route alive`并授权Phase B使用该直接对照。未达门槛只
封板当前RZ parameterization，不否定另一种CLIP-owned route。

## Phase B：CLIP-Owned Executable Residual

### 单变量边界

Phase B以sealed RZ-C0为唯一直接对照，backbone/router位置、ReZero parameterization、参数规模和recipe
不变。核心变量只有：

> 把`scalar q support`替换为`K=16 centered CLIP evidence code`，并让该code成为router hidden的
> 必经执行系数。

mask与presence仍由hard-owner pose/anchor监督。学生从detached anchor source feature预测
`e_student[B,5,16]`，用teacher code做cosine+relation/ranking监督；不预测自由identity code，不读取PID。

### 执行router

```text
M_r       = learned coarse mask * presence
pi_r      = normalize(M_r)
z_r       = Pool(F_consumer, stopgrad(pi_r))
c_r       = context_projection(z_r)
e_r       = stopgrad(e_student_r)
h_r(p)    = GELU(token_projection(F_p) + c_r + evidence_projection(e_r))
delta_r   = expert_r(h_r)
DeltaF(p) = sum_r M_r(p) * delta_r(p)
F'        = F + bounded_alpha * tanh(DeltaF)
```

`e_student`进入router前detach，ReID loss不能把它改写成identity code；但同一`e_student`由CLIP evidence
loss训练并直接参与router hidden，不存在terminal projector。`evidence_projection`是推理保留的执行
参数，不是loss-only head。

### Internal alignment

为避免CLIP监督再次停在anchor，增加只作用于router必经latent的内部relation objective：

```text
L_evidence = 1 - cosine(e_student, e_teacher)
L_relation = KL(softmax(sim(e_student)/tau_s), softmax(sim(e_teacher)/tau_t))
L_exec     = 1 - cosine(normalize(c_r + evidence_projection(e_student)),
                        stopgrad(e_teacher))
```

`L_exec`只更新router的context/evidence projections和ReZero branch，不更新backbone、final descriptor或
teacher。若维度不同，只允许使用同一个推理保留的evidence projection，禁止另接可删除projector吸收loss。

三项loss权重必须在design冻结后通过尺度等价而非性能搜索确定；首个正式arm只用一个预注册组合，
不得在训练中调权。若static/CPU/CUDA梯度所有权无法做到严格隔离，Phase B不启动。

## 强对照与反事实

Phase B完整final至少包含：

1. correct evidence；
2. slot-mean/static evidence；
3. wrong RGB evidence；
4. wrong mask binding；
5. synchronized slot-cycle与只错配evidence/slot expert；
6. random orthogonal code；
7. generic router（expert-mean）；
8. router0/1/all bypass；
9. RGB-only correct/shuffle/None/exploding exact；
10. teacher-oracle只作冻结上界，不作为部署指标。

correct相对wrong/static/generic必须在内部descriptor和完整检索上均有可辨识差，不能只看evidence loss。

## 梯度所有权

- CLIP image/text encoder、PCA basis、teacher code：永久frozen/no-grad；
- anchor输入：`stopgrad(F_source)`；pose/mask/presence/evidence loss不回流backbone；
- `e_student`由evidence loss更新，进入router前detach；
- `L_exec`只更新推理保留的router context/evidence projections与branch；
- ReID loss更新backbone、router、BNNeck/classifier，但不更新anchor/e_student head；
- final descriptor不接受CLIP feature KD、text KD或part descriptor supervision；
- 分loss backward记录anchor/router/backbone/head的梯度范数和余弦。

## 推理边界

推理只保留RGB backbone、五slot student mask/presence/evidence heads、两个COER router和唯一global
descriptor。删除CLIP image/text encoder、PCA、pose、teacher targets、teacher matching与所有纯诊断
对象。query/gallery路径不得读取任何pose/CLIP artifact。

## 正式训练recipe

- dataset：official clean Occluded-Duke；
- backbone：Swin-Tiny；
- batch：64；
- seed：1234；
- optimizer：SGD；lr=`0.0008`；
- epoch：120；eval every 10；checkpoint only 120；
- fresh、不续训、不重复、不挑best；
- 每次只允许一个正式arm占用4090。

## 成功与失败解释

### Phase A

- all-bypass贡献成立：zero-expert冷启动确为当前主要执行断点之一；
- route alive但final不涨：router可执行但当前q/state仍无性能燃料；继续Phase B仍有逻辑；
- route仍inert：只关闭RZ-C0接口，重新设计route ownership，不扩张到CLIP总体。

### Phase B

最低GO要求：

1. final相对RZ-C0 `>=+0.3 mAP`且R1不下降；
2. final相对clean D0不低，目标为`>=+0.5 mAP`；
3. full−all-bypass `>=+0.2 mAP`；
4. correct−static/wrong/generic至少一个关键mAP差`>=+0.1`，且方向与内部descriptor证据一致；
5. within-slot evidence非退化、两个consumer均有独立贡献、RGB-only/strict finite全部PASS。

未达门槛只封板当前rich-code/alignment定义，不把单seed包装为统计显著，也不永久否定CLIP–TAPF。
达到门槛也只授权必要拆因与matched seed，不直接声称论文结论；semantic multi-stage仍需独立设计。

## Novelty边界

不能声称首次CLIP+pose、局部CLIP KD、inference-free teacher或part-language alignment。可争对象仅为：

> frozen CLIP的centered局部视觉证据不监督最终retrieval descriptor，而是拥有一个推理保留、
> 可反事实干预的内部residual route；方法成功以correct evidence与all-route mediation同时成立定义。

该差分仍需与RegionCLIP、ALADIN、π-VL、PAFormer、ProFD的真实代码路径持续核对；若最终只能归纳为
普通local CLIP KD加adapter，则不能作为主创新。
