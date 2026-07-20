# exp408 PICRD 执行协议

## 冻结对象

- backbone=`Swin-Tiny`，dataset=`Occluded-Duke`，batch=`64`，seed=`1234`，epoch=`120`；
- baseline config=`configs/occluded_duke/swin_tiny_tapf_d0.yml`；
- official RGB只读`/mnt1/afrdata`，pose只读`/mnt1/afrderived`；
- runtime=`/usr/local/anaconda3/envs/mmpose-abu/bin/python`；
- fresh cache、repo、output只写`/home/afr`；
- sealed clean D0 e120=`57.5587756578/67.6923076923` mAP/R1。

## 串行执行

1. 冻结 design/protocol 与文献差分；
2. 实现独立 `PICRD_ENABLED` 开关、fresh cache builder/loader、Stage-2 slot pooling、逐槽 relation 与
   counterfactual ranking；旧 `SEMANTIC_ENABLED` 必须保持 false；
3. 只做语法、cache schema/coverage、shape/finite、config-off exact、relation正反例、backbone梯度和一次
   MMPOSE-ABU CUDA/AMP step；
4. 独立智能体盲审 design/config/diff/关键数据路径，只修 BLOCKER/HIGH；`0B/0H`后立即运行；
5. fresh 生成 15,618 图 cache和冻结diagnostic path manifest，核验 path唯一/覆盖/finite/valid/ontology后发布SHA；
6. GPU空闲且无其他4090任务时，启动唯一fresh seed1234 e120；自然完成，不续训、不按中间性能早停；
7. e10/20/.../120记录方法与sealed D0同epoch mAP/R1/差值；e120封板后更新全局文档并进入下一机制。

## 必须通过的实现门

- `relative_paths` 从 collate 到 processor 无丢失，并严格索引 official manifest；
- train进程加载cache，但eval forward不得访问cache；config-off不改变D0初始化、forward、loss；
- teacher cache=`[15618,5,768]`，valid=`[15618,5]`，路径完整唯一；不读exp407 cache/output；
- cache与训练mask使用完全相同五槽 COCO-17 ontology；增强后valid与cache valid求交；
- relation按slot在batch维计算，FP32、排除对角；任一槽有效数少于2时返回connected zero；
- 四臂冻结到同一`V_common`、同一有效槽和同一off-diagonal pair mask；zero不得因零norm失效；
- wrong-RGB固定cyclic offset=`4`（等于`NUM_INSTANCE`）且每行different-PID，否则fail closed；
- counterfactual距离在ranking中stop-gradient，只允许correct距离向下训练；
- `correct/wrong/generic/zero`全部finite并记录，反事实不得在日志或实现中被静默删除；
- `L_picrd`对Stage-2 source及至少一个Stage0--2参数产生非零有限梯度；不得detach source；
- AMP一步真实optimizer update，默认GradScaler，不调scale。

## 科学门

机制顺序：训练batch诊断的 `d(correct)` 应低于 `d(wrong-RGB)`、`d(generic)`、`d(zero)`；完整训练后只用
builder发布的16 PID×4图deterministic-resize manifest、固定顺序/offset复核。性能门：自然e120 mAP与R1同时
严格超过clean D0。两门都过才记为
`PICRD GO`；任一失败如实`SEALED NO-GO`并换训练/结构对象。

## 一次性与失败纪律

cache execution与student execution各使用fresh ID/output。取得started seal后的runtime失败封板该execution；
修复必须新实验编号。exp405--407的cache、pair、MAD、recipient、donor或运行输出均不得成为输入。
