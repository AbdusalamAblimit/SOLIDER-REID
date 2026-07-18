# exp392 Phase 0D：Semantic C0 final checkpoint冻结拆因协议

## 目的

Phase 0C只证明当前`teacher + mask/presence/q student + two routers` bundled组合未超过clean D0，
不能回答失败主要来自哪一层。Phase 0D不训练、不改checkpoint、不选best，只在Semantic C0唯一e120
checkpoint的真实推理seam做冻结反事实，区分四个候选来源：

1. sample-specific q support；
2. 五slot空间mask/presence；
3. slot-specific expert identity；
4. generic low-rank router transform。

本审计只做机制归因，不把冻结干预结果当新的训练方法性能，也不以一次FAIL否定CLIP–TAPF。

## 固定资产

- repo：`/home/afr/SOLIDER-REID-exp392-semantic-c0-ed57834`；
- execution HEAD：`ed5783416528be4284adce11fa192fe119e344f4`；
- config SHA256：`ecf3403c3e3d61af575f49420f21247f93029785364d32a23711eab66458d39c`；
- checkpoint SHA256：`8f8e4a8af1280f17f736053a3068dfae0384ac54915f9c68fb0c779350c3638e`；
- checkpoint full：`56.9/67.1/80.6/85.0`；
- query/gallery不读取external pose、CLIP、text或teacher。

## 执行seam

只在`CleanSemanticTapfC0.prepare(..., training=False)`生成student state后、两个
`SemanticSpatialRouter`消费前临时替换`consumer_mask/consumer_support`，或在router参数的内存副本上
做slot expert变换。每个arm结束后恢复原tensor/参数；checkpoint文件和正式repo保持只读。

所有arm必须：

- 同一dataloader顺序、同一batch、同一evaluator；
- 先做correct-start，全部arm后再做correct-end，descriptor与四项指标必须exact；
- 模型始终`eval + no_grad`；
- state SHA前后exact；
- 单进程、单4090、无训练model/optimizer/checkpoint输出。

## 冻结arms

### A0 correct

production RGB-only state：learned mask、learned presence、learned sample-specific q、slot-specific experts。

### A1 static-slot-q

先在同一query/gallery顺序的独立只读pass收集五slot q，对每个slot取全验证集均值；正式评测时每图均
使用该五维常量。保留slot prior与整体尺度，只删除sample-specific q动态。该常量只用于冻结解释，
不是可部署方法或test-time调参。

### A2 q-one（mask/presence-only）

把所有有效slot support固定为1，保留每图learned mask/presence。它回答coarse空间state在不依赖
CLIP q幅度时是否已解释主要router效应。

### A3 spatial-constant-mask

对每图每slot把learned mask替换为其空间均值常量，保留该图q与presence。它删除空间geometry，
同时精确保留每图每slotmask mass与q动态。

### A4 slot-cycle

对mask、q和presence同步做五slot循环置换，但router expert保持原slot顺序。它破坏state↔expert
解剖绑定，同时保留同图全部数值和空间统计。

### A5 expert-mean

把五个slot expert替换为其均值的五份复制，保留correct state和token/context projections。它保留
generic low-rank feature transform容量，删除slot-specific expert identity。

### A6 router-0-bypass / A7 router-1-bypass / A8 all-router-bypass

分别把对应router的expert临时置零。该组量化完整检索指标上的consumer条件性贡献，不替代与B0/D0
的matched训练比较。

## 预注册解释规则

所有差值均为`arm − correct`，只在完整query/gallery上计算：

1. 若A1绝对mAP变化`<0.1`，则当前sample-specific q对final没有可辨识边际贡献；若A1更高，q动态
   在当前模型中更像噪声；若A1降低至少`0.1`，才保留“q动态被使用”的描述性证据。
2. 若A3绝对mAP变化`<0.1`，则learned mask的精确空间geometry没有可辨识边际贡献；A3更高表示
   低频/mass解释再次优于geometry。
3. 若A4与A5均绝对变化`<0.1 mAP`，不得声称五slot identity已成为检索因果变量。
4. A6--A8只判断router路径依赖；即使all-bypass下降，也不能证明q、geometry或slot identity有效。
5. 单checkpoint小于`0.1 mAP`差值视为近零描述，不声称统计显著；不调阈值、不重复、不挑arm。

## 终审与后续

审计结束必须记录script/result/runner SHA、进程自然退出、GPU空闲、strict异常0、correct start/end exact、
state SHA exact。根据拆因结果只允许选择一个下一机制变量：

- q动态近零：优先重构为centered residual/ranking support，不再调BCE温度；
- geometry近零：不继续加更细mask，改为局部相对视觉证据；
- slot identity近零：router必须建立显式state↔expert匹配约束；
- 仅generic router有效：CLIP语义主张继续NO-GO。

任何新训练仍需独立design与完整preflight；single-stage语义因果和final性能未成立前，semantic
multi-stage保持NO-START。
