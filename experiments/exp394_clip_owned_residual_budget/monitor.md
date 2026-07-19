# exp394 监控记录

## 2026-07-19 立项与NO-START边界

直接前因是exp393 RZ-C0 `ROUTE-ALIVE-FAIL`：e120 full=`56.8/66.8/79.6/83.9`，all-bypass
同为`56.8/66.8/79.6/83.9`，raw full−bypass=`-0.000249709 mAP point`。两个alpha最终仅
`-1.843e-4/-1.363e-4`，尽管token/context/expert/alpha参数轨迹和strict finite全部PASS。

exp394不重跑RZ、不调alpha，也不把Phase 0E rich teacher PASS作训练成功。当前只冻结新的问题定义：
rich evidence拥有production branch方向，执行幅度改为train-only D0能量匹配的固定有界预算，并以
wrong/static/generic/all-bypass证明是否真正属于CLIP语义。

当前远端无训练/审计进程，4090=`2 MiB/0%`。下一任务严格串行为：

1. 只读复核clean D0 checkpoint与固定128 train image来源；
2. 写Phase 0R-S synthetic/CPU contract；
3. contract PASS后才实现Phase 0R-128预算冻结审计；
4. 所有门禁、`rho_star`与SHA冻结前，远端production model/config实现、CUDA preflight和正式训练均
   `NO-START`；只允许本地独立Phase 0R-S contract与后续只读预算审计。

禁止用验证/测试性能选择预算，禁止恢复自由alpha，禁止并行GPU任务。

## 2026-07-19 clean D0与固定128图只读来源冻结

只读复核确认预算审计的唯一D0来源为exp387官方干净seed1234 final：远端repo=
`/home/afr/SOLIDER-REID-exp387-d0-0d1822a`，exact HEAD=
`0d1822a07dda8daac0210b68916035b1886d5d99`且tracked clean；output内只有一个checkpoint
`transformer_120.pth`，SHA256=
`59017755d61370754aa2e852a487d8e242fcee8814685f77f5388ba3a430e069`。不得引用Semantic C0、
RZ-C0或其它历史checkpoint替代。

固定128图不重新抽样，直接复用exp393 0E-128 sealed codebook：codebook SHA256=
`4a671a70e0744edad88f911ce628d421650cb09453eb511a61e8d01c239269ef`，内部selection SHA256存储值
与canonical JSON重算值均为
`7f3f7626c84553416f39c72be0c15ab430458aa7b201c4bf64461990bbdf15e3`。selector seed=
`20260719`；128条path和128个PID均唯一，fit/audit=`64/64`，全部path以
`bounding_box_train/`开头，真实RGB文件存在=`128/128`。该冻结来源不读取query/gallery，也不允许
根据预算分布更换样本。复核期间4090保持`2 MiB/0%`且无任务。

## 2026-07-19 Phase 0R-S synthetic/CPU contract封板

新增本地独立`phase0r_static_contract.py`，不导入、不修改远端sealed repo、production model或config。
首次执行按门禁返回`PHASE0R_S_FAIL`：15项梯度所有权检查全部通过，唯一FAIL是把batch内两条相同
输入的proposal要求bit-exact。失败script/result/runner SHA256分别为
`53d8d3e29e128e79734e8fdc4942916799d044ede88d41a8ac83eb1ef9bc18e7`、
`872b6294418b2e13e1010456b9dafecf1f42d851ab4ad6ec337dc950462a863b`、
`1c11280aeae0070be61be4c94ff96953aa9f1a4a742c090d46f1eeddd0fd0e4c`，原始资产保留。

第二次只增加诊断量，确认两条输入的token/mask/static evidence最大差均exact `0`，proposal差仅
`7.450580596923828e-08`，归因为CPU线性代数kernel对不同batch row的舍入，而不是static code产生
样本信号。诊断FAIL的script/result/runner SHA256分别为
`ce261eb1918555baa6a0a55dc20d37815892b98df520d3fc4ee7f20cb161bffc`、
`48a21f40af1c84d6b020d34236bc40df31b40d17e57ea28223421fd50cfe4967`、
`928c054f86d3e4115b113a029078dc180d8702993e11c5219a8de2e55faacc56`。没有按观察值设置容差；正式
contract改为同一单样本与static code两次独立forward必须bit-exact，同时保留batch-row差作描述量。

最终CPU contract连续两次result与runner逐SHA一致，verdict=`PHASE0R_S_PASS`。19项check全部PASS：
`rho=0` full/bypass exact、NULL mask/presence identity与applied delta exact zero、epoch-only schedule
repeat exact且teacher阶段为零、handoff后固定、rho不在parameter/buffer、RMS normalization finite、
zero-mass slot exact zero、correct/wrong proposal max-abs=`0.0645029`、static单样本repeat exact。
teacher/`L_exec`/ReID三类loss的15项梯度所有权全部exact：`L_exec`更新evidence head与真实
token/context/evidence/expert，且不回流backbone、anchor或ID head；ReID不更新anchor/evidence head。

最终script/result/runner SHA256分别为
`29aedae0b409f3b96aa0cf20413124d0206c6c129c63140ba3aa5f592a2f4912`、
`fad574aed5e0fb04e77fadc0f0b2ab4bf4506f0f9e0f49fd7bacbeda6fa464a8`、
`0fe10057d84da52ee3c8735fdb2789de0462922da745a5ccbcc430a1bf6dbc6f`。该PASS只授权下一步实现并执行
Phase 0R-128 train-only只读预算冻结审计；远端production model/config、CUDA preflight、正式训练与
semantic multi-stage仍`NO-START`。

## 2026-07-19 Phase 0R-128协议冻结

新增`phase0r_128_protocol.md`并在读取delta结果前冻结production seam与唯一预算公式。审计只hook
exp387 D0两个Stage-3 `PoseSpatialGate`的真实input/output，定义
`a_k=y_k-x_k`、`r_k=sqrt(mean_channel(a_k^2))`；`rho_star`固定为两个bank全部
`2×128×48` token RMS的pooled median，包含零值且不按图/PID/bank截尾。协议同时冻结两遍
None/exploding-pose逐tensor exact、state/checkpoint SHA、strict finite、吞吐/显存和异常门。

当前为`PROTOCOL-FROZEN / AUDIT NO-START / FORMAL NO-START`。下一步只允许实现独立只读脚本；
不得在看到分布后改公式、重选128图或启动production/CUDA训练preflight。
