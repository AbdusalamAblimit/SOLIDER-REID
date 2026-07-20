# exp409 PCHM 独立代码盲审

## 审查范围

- `design.md / protocol.md`；
- exp409 config、cache builder、contract；
- `loss/pose_clip_hard_mining.py`、`loss/triplet_loss.py`、`loss/make_loss.py`；
- `processor/processor.py`与default-off路径。

审查者未参与实现，只读检查联合选边、rank方向、候选mask、tie-break、pair index梯度、cache来源、fresh边界、
训练/AMP接线和default-off exact。

## 首轮：0 BLOCKER / 1 HIGH

HIGH：最初cache只有整体文件SHA及`schema/relative_paths/features/valid`，builder使用
`verify_image_sha=False`。这只能证明训练读取同一份字节，不能证明它来自冻结RGB、CLIP checkpoint、pose
manifest、preprocess及源码，违反协议中的输入绑定。

启动任何cache/GPU前完成最小修复：

1. builder逐图`verify_image_sha=True`；
2. NPZ写入逐图image SHA、pose manifest SHA、CLIP checkpoint SHA、preprocess ID、source HEAD、builder SHA、
   teacher source SHA；
3. loader严格验证字段集合、shape/dtype/hex、expected pose+CLIP SHA；
4. processor保留每个训练batch的image SHA，并与cache path对应SHA逐项比对；
5. contract增加cache provenance roundtrip和错误image SHA拒绝mutant。

## 聚焦复审：0 BLOCKER / 0 HIGH

原HIGH完整闭环。其余检查结论：

- positive/negative rank方向、候选mask和Borda/tie-break与设计一致；
- positive同PID且非self，negative严格异PID；
- 外部pair index只选择distance matrix元素，不截断被选final descriptor梯度；
- legacy batch-hard index显式传入时，loss与gradient bit-exact；default-off保留原路径；
- 未发现必现runtime、AMP或训练接线B/H。

最终结论：`0B/0H / FRESH CACHE AND ONE REAL BATCH CUDA-AMP AUTHORIZED`。

## real-batch执行器追加盲审

首轮发现`1 BLOCKER / 0 HIGH`：脚本用`layers.3`筛Stage-3梯度，但当前Swin参数注册为
`base.stages.3.*`，会把正常梯度误报为空。已在执行前改为精确`name.startswith("base.stages.3.")`；其余
config merge、center CUDA0、GradScaler、pair/data SHA、forward/loss与真实optimizer update检查未见B/H。
同一审查者聚焦复审确认修复精确匹配且语法PASS，未引入新问题。real-batch执行器最终=
`0 BLOCKER / 0 HIGH`。

real-batch v1实际执行随后暴露新的reporter问题：脚本在`GradScaler.unscale_`之前把scaled参数梯度的非有限值
直接当失败，既没有让原生scaler执行skip/backoff，也没有产生optimizer update。本次v1冻结为
`INVALID CHECKER / MODEL SCIENCE NOT EVALUATED`，禁止重跑。

v2只修AMP测量语义：final descriptor用未缩放`autograd.grad`检查；参数梯度在`unscale_`后报告；默认
GradScaler不覆盖初值，最多8个固定batch native attempts，只允许overflow skip/backoff，得到第一且唯一成功
optimizer update即停止。没有手调scale、loss、batch、pair或模型；等待追加盲审后才能fresh执行v2。

v2首轮盲审发现`1 BLOCKER / 0 HIGH`：reporter只汇总base参数的nonfinite，但GradScaler扫描完整optimizer；若
classifier/bottleneck单独overflow，会把正确native skip误报成“finite未更新”。已改为对全部model参数汇总
nonfinite，并以`scale_after < scale_before`作为native overflow权威判据；等待聚焦复审。

同一审查者聚焦复审确认全model nonfinite覆盖完整optimizer，native overflow、skip、finite update与梯度门顺序
正确。fresh real-batch v2最终=`0 BLOCKER / 0 HIGH / EXECUTION AUTHORIZED`。
