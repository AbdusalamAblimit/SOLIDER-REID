# 随机source-key CPU诊断监控

> 当前：`V1 SEALED-INVALID / V2 VALIDITY PASS / RANDOM SOURCE-KEY FALSE OWNERSHIP DEMONSTRATED / GPU NO-START`

- 只允许一次冻结CPU执行；runtime错误封板该次记录，不就地改结果。
- 执行后记录uv/Python、CUDA环境可见性、正反合同和result SHA。

## 2026-07-20T02:12:14Z：execution v1 SEALED-INVALID

- uv语法检查PASS；冻结CPU execution v1在任何descriptor/metric产生前退出；
- 原因：`donor_map`把query `q_0`末段PID误当replica slot，查找`(camera=0,pid=1,slot=0)`失败；
- 异常：`KeyError: (0, 1, '0')`；
- torch未导入，CUDA不可见，official data/pose/cache/checkpoint访问0，GPU任务0；
- v1脚本不修改、不补跑，状态`SEALED-INVALID / NO SCIENTIFIC RESULT`。

已建立fresh `execution v2`：只把query slot定义为常量`q`，gallery仍使用replica编号；其余设计、seed、门槛与
正反合同完全冻结。GPU保持NO-START。

## 2026-07-20T02:13:57Z：execution v2完成

冻结v2以仓库uv环境、`CUDA_VISIBLE_DEVICES=''`执行；torch未导入，数据/pose/cache/checkpoint/GPU访问均为0。

原始随机key：

- correct/wrong/generic/NULL mAP=`1.000000000000/0.608134449011/0.039242546015/0.030194547250`；
- R1=`1.000000/0.664062/0.023438/0.000000`；
- correct−wrong=`0.391865550989` raw mAP；
- wrong−max(generic,NULL)=`0.568891902996` raw mAP；
- same/different-PID key cosine mean=`-0.009694111550/0.039808033682`，绝对gap=`0.049502145232`；
- donor contract、unit norm与全部冻结门PASS。

语义盲key置换后仍PASS：correct/wrong/generic/NULL mAP=
`1.000000000000/0.592976695646/0.021800154060/0.028283562338`，两个margin分别为
`0.407023304354/0.564693133308`。constant-quota mutant没有通过correct floor与wrong−low门，成功被抓住。

**裁决**：`RANDOM_SOURCE_KEY_FALSE_OWNERSHIP_DEMONSTRATED`。该toy证明强arm顺序存在非语义source-key
假阳性，不证明exp402/403实际采用此shortcut，也不授权任何模型实验。未来候选必须新增semantic-blind
random-key或等价null-semantic control；exp404与GPU保持NO-START。

SHA256：

- v1 source=`7a5495f2ae5e4b19624600fe1c4a9753f0596002508bf8631c4f23988231f2fc`
- v2 source=`0f7cf17b03d0f419fd5aa7e5300af98d4c58057d6166d7dc04a518716d7d23a6`
- v1 result=`299b0a604918db2b20b8c5062a58291b4678b43e60da9055668938058abfb3bd`
- v2 result=`516875803462d4f38d987791889ca2e5f2164905b6a8bd10d3cc4d8c0dc95630`
