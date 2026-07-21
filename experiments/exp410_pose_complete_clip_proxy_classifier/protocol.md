# exp410 PC²P 协议

## 冻结边界

- student固定Swin-Tiny、batch64、seed1234、120 epoch、official Occluded-Duke、fresh OUTPUT_DIR；
- D0 TAPF pose loss、CE权重、triplet、margin、优化器、学习率和eval descriptor全部不变；
- PC²P与PICRD/PCHM/semantic/SPK/ELO-CUR/hierarchical互斥；
- 无 `Q`、projection、adapter、temperature、额外scale、新loss或测试分支；
- train-only bank只读自fresh exp410 asset；eval不读取bank/CLIP/外部pose；
- 任何时刻只允许一个4090任务，不重跑sealed编号、不续训。

## Fresh bank 合同

1. 输入cache SHA必须等于冻结exp409 per-image五槽cache：
   `d502a0f03fe556284fd01259ed81143dcfb171855b9b2aebaa29e3b7a682fd36`；
2. 路径集合与official train 15,618图完整唯一一致，逐图RGB SHA和pose/CLIP provenance必须透传并验证；
3. 702个relabel PID逐行绑定original PID，不能缺失、重复或重排；
4. 每个PID五槽支持计数均大于0；`proxy=[702,768]` FP32、finite、行L2 norm在冻结容差内且无重复行；
5. fresh bank、manifest和builder SHA写入config后冻结；错误cache/bank/PID mapping/SHA mutant必须fail closed。

## 必要实现检查

只执行以下必要项，随后一次独立智能体代码盲审；最终 `0B/0H` 即进入fresh运行：

1. default-off初始化、state、forward、loss与D0 exact；
2. PC²P开启时原classifier不被调用，其梯度为None/0；bank无梯度且不进optimizer/state；
3. 真实PK batch64的proxy logits=`[64,702]` FP32、finite、非恒定；记录BN norm、logit mean/std/abs-max和CE；
4. CE-only backward必须让BN/global_feat、`base.norm3`、`base.stages.3.*`产生finite/nonzero梯度；
5. CE+原triplet+D0 pose loss用default GradScaler取得一次真实optimizer update；
6. 无bank、CLIP、外部pose的eval返回原768维global descriptor。

只修盲审BLOCKER/HIGH；不根据logit统计增加scale或其它超参，不追加无穷static/CPU测试。

## 唯一正式arm

- execution/output/runner必须fresh；启动前remote repo clean、GPU无compute PID；
- 只运行`correct` PC²P seed1234/e120；运行中冻结源码/config/bank/参数；
- e10/20/.../120记录PC²P与sealed clean D0同epoch mAP/R1/Δ，不按中间点早停；
- 自然e120同时严格超过raw `57.5587756578/67.6923076923`才判性能GO。

## 后续与封板

- 性能FAIL：`SEALED NO-GO`，不补controls，不调旧臂，下一编号换对象；
- 性能GO：再串行执行`wrong-RGB`与`generic` matched e120；必要时才补zero/random-code；
- correct不胜wrong-RGB：PID–CLIP语义绑定FAIL；correct不胜generic：pose-completion归因FAIL；
- 所有完成/失败均更新monitor/results/decisions/innovation/story并显式提交目标文件。
