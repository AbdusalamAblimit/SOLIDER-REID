# exp392 监控记录

## 当前状态

- `DESIGN-ONLY / RESEARCH-ONLY / NO-START`；
- exp390与exp391均已封板，禁止重启、续训或把本实验记为exp391 Phase B/C；
- 当前GPU训练任务：无；
- 当前仅完成文献、公开代码、当前TAPF执行路径与机制审查；
- 未创建config、output、runner、checkpoint，未启动任何GPU任务。

## 已确认边界

1. current PSG是field-only自由`17→32`空间门控，结构上不要求joint-channel语义可辨识；
2. exp391 H2-M相对D0为`−0.4 mAP`，但early-bypass仍有`+0.141 mAP`，说明route可达而纯结构
   topology不足；
3. RegionCLIP、π-VL、PAFormer、MUVA、ProFD、ALADIN等已覆盖CLIP局部语义、pose-aware part、
   multi-level guidance或inference-free KD的主要构件；
4. 可争对象必须是counterfactually identifiable executable anatomical mediator，而不是“首次CLIP+pose”；
5. exp391只封板无语义校准的纯结构链；semantic single-stage通过后允许重新验证semantic multi-stage。

## 2026-07-18 Phase 0 协议冻结

新增 `phase0_protocol.md`，当前仍为文档审查，不执行脚本、不占GPU。已冻结：

1. **Phase 0A clean D0 内部field seam**：clean runtime不是旧TAPF tuple；干预点必须是
   `model.base.tapf.anchor`输出dict中的`field`（或等价的`prepare()`返回dict），确保替换发生在
   student field形成后、两个PSG消费前。预注册channel-cycle、left/right channel swap、
   confidence permutation、matched-wrong、spatial-constant、zero与逐consumer/all bypass；外部
   correct/shuffle/None/exploding仍需exact。
2. **Phase 0B 双编码teacher门禁**：teacher RGB必须与student复用同一次resize/flip/pad/crop，主teacher
   读取RandomErasing前RGB，并用post-erasing作为clean-view KD混淆控制。square-stretch与
   aspect-letterbox只做预注册teacher-only比较，mask必须使用完全相同的RGB→CLIP grid变换。
3. **五类coarse ontology**：head/torso/arms/upper-legs/lower-legs；固定incidence、limb segments与
   prompt ensemble。主teacher用pose-region池化frozen CLIP patch tokens，再与全部frozen text
   prototypes形成sample-specific分布；不允许text-only常量冒充双编码teacher。
4. **强反事实与kill-switch**：wrong RGB/mask/text、channel shuffle、uniform/fixed bands、
   text-only、image-only cluster、flip equivariance、样本方差/JSD/effective-rank和遮挡分组均为
   必报项。任何关键输入不敏感、分布近常量或双编码不优于单编码控制时，当前teacher定义NO-GO。
5. **历史exp356边界**：已只读定位其ViT-L/14末层patch hook、强制eval、224方形拉伸和固定
   top-5/mid-6/bottom-5池化。历史pose-mask 57.1、random-mask 57.3只封板“固定水平条带completion
   + pose选mask”，没有实现pose-mask池化CLIP tokens或可执行语义中介，不能等价否定exp392。

协议同时记录未来Phase 0C的NULL identity、梯度所有权、config-off exact、semantic mismatch和
generic adapter强控制；Phase 0A/0B通过也只授权0C，不直接授权120-epoch训练。

## 2026-07-18 Phase 0A 开始执行

用户明确授权开始。当前只执行封板checkpoint的frozen audit，不修改exp387 repo/config/checkpoint，
不启动正式训练。

执行边界：

- exact repo：`/home/afr/SOLIDER-REID-exp387-d0-0d1822a`；
- execution HEAD：`0d1822a07dda8daac0210b68916035b1886d5d99`；
- config SHA256：`510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b`；
- checkpoint SHA256：`59017755d61370754aa2e852a487d8e242fcee8814685f77f5388ba3a430e069`；
- 审计脚本：`phase0a_internal_field_audit.py`，SHA256=
  `e3343e96b2b6b4202cabcdeedf5300da3704cff1bb60947babb17cddd9d21682`；
- 审计输出独立位于`/home/afr/reid-clean/audits/exp392_phase0a/`；
- 启动前GPU=`2 MiB / 0%`，无其他compute process。

真实首批256图route smoke=`EXP392_PHASE0A_ROUTE_SMOKE_PASS`，JSON SHA256=
`b54389b729e4e2c57ad34cc298a0ae54caa01d8a7e4329483444fdf4c0dfe8c7`。确认：

1. external correct/shuffle/None/exploding descriptor逐元素exact；
2. `zero_field == psg_bypass_all` descriptor逐元素exact；
3. descriptor最大绝对变化：channel-cycle `0.0423`、left/right swap `0.0619`、
   confidence permutation `0.0729`、matched-wrong `0.2884`、spatial-constant `0.3439`、
   PSG0 bypass `0.7897`、PSG1 bypass `3.0584`、all bypass/zero `3.9160`；
4. 所有预注册内部干预均非dead，hook seam与NULL identity可进入全验证集审计；
5. smoke只验证执行路径，不用于mAP裁决。

## 下一步

完成Phase 0A全验证集frozen audit、donor-map SHA、四项指标与paired bootstrap；随后再进入Phase 0B
teacher-only实现。正式训练保持`NO-START`。
