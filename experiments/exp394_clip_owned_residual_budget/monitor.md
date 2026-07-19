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

## 2026-07-19 Phase 0R-128实现与启动前冻结

新增外置只读`phase0r_128_budget_audit.py`，script SHA256=
`628ce2f88a868ccb2a14f5c0a3204099332253e392bf8c271dd53301057222a3`。脚本不构建optimizer，先验
校验HEAD/tracked/config/checkpoint/codebook/selection与128条official train映射；用原生RGB-only
eval transform冻结CPU input，strict load后只以`eval+no_grad`注册两个短生命周期consumer hook，
两遍分别传`None`和exploding pose并要求descriptor/applied delta exact。

本地uv Python静态编译PASS；远端`/home/afr/par2606/.venv` Python 3.10静态编译与`--help` PASS，
本地/远端script SHA exact。远端脚本仅落在
`/home/afr/reid-clean/audits/exp394_phase0r/`，未修改exp387 sealed repo；启动前4090=`2 MiB/0%`、
无compute PID。当前状态`PHASE0R-128 READY / NO-RESULT / FORMAL NO-START`。下一步只允许一次正式
只读审计，不得并行启动其它GPU任务。

首次正式入口使用`/home/afr/par2606/.venv`，在import sealed model时即因
`ModuleNotFoundError: cv2`退出；尚未构建model、加载checkpoint或初始化CUDA，GPU始终
`2 MiB/0%`。该FAIL归因为选错runtime，不是数据、checkpoint或预算门失败，不修改script/protocol。
失败result/runner SHA256分别为
`d25126a76f575b51fff91102e914417fd3323b12f89e30ee310494f4f6dbc937`、
`cc91a4084ab48c4d179d05958d34e0f433f5cf3460282aba53edfa1011d32195`，本地与远端原样保留。

只读环境核查确认exp387原生clean runtime为`/home/afr/reid-clean/.venv`，Python/Torch/CUDA/OpenCV/
timm=`3.10.12/2.6.0+cu124/4.13.0/1.0.27`，依赖import PASS。禁止向错误runtime补装包；下一次只允许
在该原生环境重做同一冻结script的一次完整审计。

## 2026-07-19 Phase 0R-128预算冻结审计封板

同一script在exp387原生clean runtime自然完成，verdict=`PHASE0R_128_PASS`。official train=
`15,618`、固定selected=`128`图/`128` PID、fit/audit=`64/64`、4 batch；两个consumer各覆盖
`6,144=128×48`个token，pooled=`12,288`。bank0/bank1/pooled的median分别为
`0.0376448072/0.1204396486/0.0807554498`，P95分别为
`0.1426717915/0.3115715981/0.2651471898`，nonzero fraction均=`1.0`。按预注册pooled median公式
冻结唯一`rho_star=0.08075544983148575`，没有按性能、bank或样本筛选。

16项gate全部PASS：exact HEAD/tracked/config/checkpoint/codebook/selection、official覆盖、strict
223-state finite、两个consumer每遍4次调用、hook removed、None/exploding-pose两遍descriptor与
两个bank applied delta逐tensor exact、exploding pose访问0、rho finite positive、无optimizer、
state/checkpoint SHA前后exact。state SHA前后均为
`c75e9d2e26f83255ae122a6c84b1717bc9474493453c7e04d95163da3cea96a3`；RGB manifest/input tensor
SHA分别为`e7416534abe4489d256eacefec050379bfab443acabdd49c77ce457d0aaec5e7`/
`f0e793478c65e1e30ff999560f4081f6e34b4ffa39f20bd583b1593f967235b2`。

两遍model-only吞吐=`407.09/1069.66 img/s`，耗时=`0.3144/0.1197s`，peak allocated=
`770,461,184 bytes`。进程自然退出，4090恢复`2 MiB/0%`，runner严格异常词命中0，exp387 tracked
source与checkpoint SHA保持不变。script/result/runner SHA256分别为
`628ce2f88a868ccb2a14f5c0a3204099332253e392bf8c271dd53301057222a3`、
`4f20bef4539129d0e2a9250262b7a09ee7feee03a80fbd2c5491e3450e0d1715`、
`7142cdb1cfd194262ef7daf6c4e3e9823bf561080ec8227143e55408d464887d`。

裁决=`PHASE0R_128 SEALED-PASS / RHO FROZEN / PRODUCTION NO-START / FORMAL NO-START`。该结果只
证明D0能量基准可复现并冻结，不证明exp394 route或CLIP方向有效。下一步仅允许production实现前的
static设计/代码seam审查；不得直接启动CUDA训练preflight、正式训练或semantic multi-stage。

## 2026-07-19 Phase 0P static source-seam封板

已冻结`production_protocol.md`，明确rich teacher code、student/production detach边界、同权重
`L_exec`重算、relation loss、两个独立consumer、RMS normalization、`rho` schedule、默认off和十项
同checkpoint反事实。特别将generic arm收紧为同checkpoint `generic-context-only` evidence ablation；
真正matched generic adapter只能在候选PASS后另做fresh训练，不能用当前反事实冒充。

随后独立`phase0p_source_contract.py`在本地uv CPU连续两次输出逐SHA exact，19项全部PASS。它确认八个
local source/config blob与远端RZ-C0 sealed repo exact，Stage-3两个consumer、external teacher、
no-grad target、model-only checkpoint、RGB-only eval、source/state detach和默认off seam均成立；同时
确认exp394 production flag/config当前不存在，避免把source可行性误写成实现完成。

script/result/runner SHA256分别为
`27859a2ae0b5a1020b9a68cda5777ad332e05701c372940d584728eb5d60fae1`、
`496630c3e7ba1d76d4e49b6347f8741f6d799b56ca5321283792bda92fd4cb8d`、
`0e5070db2ec733e76139d7b41bd55cba724cf4cafba9dc5e23845e7c85be5eb5`。

裁决=`PHASE0P_SOURCE_PASS / FRESH IMPLEMENTATION GO / CUDA NO-START / FORMAL NO-START`。下一步
只允许从`09340f7`创建fresh独立repo实现冻结协议并先做static/CPU exact；不得直接占用GPU或训练。

## 2026-07-19 fresh production实现与CPU contract封板

按冻结范围新增rich image-only PC-MBCLS teacher、从`hidden.detach()`产生的`[B,5,16]` evidence head、
两个独立T/C/E/Expert router、FP32 per-token channel RMS与Python-float rho schedule；训练processor只在
rich开关打开时构建外置teacher，checkpoint/eval仍只接model与RGB。新config把CLIP checkpoint改为
待复制的新clean canonical实体路径，未引用旧`/home/afr/SOLIDER-REID`；full codebook与两个SHA固定。

CPU contract依次保留四类非最终入口：v1缺repo-root import、v2包级import触发本地缺`cv2`、v3错误
要求proposal 768维与teacher code 16维相同、v4暴露新anchor零初始化使独立mask/presence loss首步不回
trunk；均未初始化CUDA。v3按冻结定义修正为“只共享`[B,5]`、分别形成cosine Gram”；v4移除新anchor
不必要的mask/presence零初始化，因为e1–e5 identity已由rho exact zero保证。v5已全PASS但runner含一条
诊断转标量warning，只对诊断读数detach后重做；随后又增加rich-on/semantic-off非法组合显式FAIL保护。

最终contract连续两次逐SHA一致，三组总gate及全部production子项PASS。D0/HT0/Semantic C0/RZ-C0
相对`f5de340`的state/forward exact；rho schedule、NULL identity、relation loss、两个consumer、strict
reload、teacher-free state、config-off与四类梯度所有权全部PASS。handoff/fixed synthetic descriptor
max-abs gap=`0.0405087/0.132013`；correct−wrong/static proposal max-abs=`0.0304661/0.0193796`；RMS距1
最大`0.000876129`，全部finite。runner异常词中除JSON键`teacher_inference_mode`的字面`inf`外无
warning/NaN/Traceback/RuntimeError/OOM/nonfinite/overflow。

最终source SHA256：tapf/teacher/make_model/processor/defaults/config=
`95c5d0ff80bf9e4529589a5f31819e7aad5db644b88e2a33d6af07c9ffc42886`/
`c648fa768b178d153258c46eee69679cbc0b90a11db918800323ab5c5c6054d5`/
`6bc7d9c83a2f4d12b78dd2c09335d366ce568107ddce5dded3abfe7ca8538f03`/
`be1c19ea5af19534e3855eb2a5914e0dc9a5643c63a39cfa508c81f89660eac1`/
`a13e5f6df0e8c770c254c115d6d55208baac7938cffbec6f208ba9caa24dd7c5`/
`e0413a497976ad6dbf4c74cf13b55c86c169d659bab6d967455e87c592e47f4e`。
contract/result/runner SHA256=`5be2980eb6a666f791ba5e3cd87bbabb7a0b9934bb44724e091cbbb7e4545cd1`/
`658ac1fd261ec09db618e9d658ae00fa3f0f7d7887b87e8716c601adbc8b0636`/
`658ac1fd261ec09db618e9d658ae00fa3f0f7d7887b87e8716c601adbc8b0636`。

裁决=`PRODUCTION_STATIC_CPU_SEALED_PASS / CUDA PREFLIGHT DESIGN GO / CUDA NO-START /
FORMAL NO-START`。本轮未占用4090；下一步只能设计并冻结至少24步真实batch64 CUDA/AMP门禁和clean
canonical资产复制，不得直接启动CUDA、正式训练或semantic multi-stage。

## 2026-07-19 clean canonical teacher资产落盘

本地production与CPU contract完成后，才从已封板源复制CLIP checkpoint到新clean canonical实体：
`/home/afr/reid-clean/weights/exp394_clip_l14_openclip_9ce2e8a8.safetensors`。远端复核为regular file、
非symlink，SHA256=`9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e`；full codebook
SHA256=`fb87da370ea945d526f499bef78093a6b07203d87c6d84efe06b5eb6594f954a`。config所写路径/SHA与实体
exact。复制未加载模型或初始化CUDA；sealed RZ-C0 repo仍为exact `09340f7`且tracked clean，4090=
`2 MiB/0%`、无compute PID。

## 2026-07-19 CUDA/AMP preflight协议冻结

在production static/CPU commit `11d7a35`之后新增独立CUDA协议，尚未实现或执行脚本。协议冻结24个
official actual batch64更新：step1–12只用epoch1验证rho=0 exact identity但`L_exec`真实更新branch；
step13–24只用epoch6的`rho_star/5`验证handoff非零descriptor路径。禁止用epoch10放大gap、补跑overflow
step、换batch或按loss筛样本。

协议同时冻结actual-batch四次隔离backward、correct/wrong/static proposal可分、NULL exact、两个
consumer、strict reload、teacher/optimizer/checkpoint/eval隔离、RGB-only四变体、显存`<22 GiB`与
异常词0。该阶段裁决=`CUDA_PROTOCOL_FROZEN / IMPLEMENTATION NO-START / CUDA NO-START / FORMAL
NO-START`；等待后续heartbeat明确授权后才可实现外置preflight脚本，当时未启动4090。

## 2026-07-19 CUDA/AMP preflight实现与静态门

用户后续指令已解除preflight实现/执行的过时等待边界，但没有授权正式e120。新增独立外置
`cuda_amp_preflight.py`，只读取exact implementation commit `11d7a35788c4645c355d96d76a2a4ff20a9801ac`、
official train与两份canonical teacher资产；脚本不保存model/optimizer/scaler/checkpoint，也不修改
production repo或config。它冻结24次actual batch64 AMP update、四类actual-batch隔离backward、两个
consumer反事实、strict reload、RGB-only、teacher/state/asset/hook隔离和`<22 GiB`门。

本地uv与远端clean runtime的`py_compile`/`--help`均PASS；远端static入口使用
`CUDA_VISIBLE_DEVICES=`，4090全程保持`2 MiB/0%`且无compute PID。远端外置script位于
`/home/afr/reid-clean/audits/exp394_cuda_amp_preflight/cuda_amp_preflight.py`，本地/远端SHA256 exact=
`bae2210bc606048371b4750f85919595c0b8fdbd1e11681abac59fe9727ea4f0`。sealed RZ-C0 repo保持exact
`09340f7`与tracked clean；canonical CLIP/codebook存在。

裁决=`CUDA_PREFLIGHT_IMPLEMENTATION_STATIC_PASS / ONE EXACT CUDA PREFLIGHT GO / FORMAL NO-START`。
下一步只能从implementation commit建立fresh execution repo并执行这一次冻结preflight；不得启动e120、
semantic multi-stage或任何并行GPU任务。

## 2026-07-19 fresh execution repo与runtime入口归因

远端fresh sparse execution repo已建立为
`/home/afr/SOLIDER-REID-exp394-rich-budget-c0-11d7a35`，detached HEAD exact=
`11d7a35788c4645c355d96d76a2a4ff20a9801ac`、tracked clean；sparse只排除与执行无关且包含1.9 GB模型的
`experiments/`目录，不改变commit/tree。六项production SHA与两个保护blob均和协议exact。首次直接传
完整bundle因两条历史不共祖膨胀到1.7 GB，发现后终止未完成传输并保留`.partial.failed`归因；随后用
commit/tree pack、runtime blob pack与sparse checkout得到相同exact HEAD，没有修改source。

首次CUDA入口误用`/home/afr/reid-clean/.venv`，在model/optimizer构建后、teacher import时因
`ModuleNotFoundError: open_clip`退出，成功optimizer update=`0/24`，未读取actual batch、未产生
checkpoint，GPU恢复`2 MiB/0%`。失败script/result/runner SHA256=
`bae2210bc606048371b4750f85919595c0b8fdbd1e11681abac59fe9727ea4f0`/
`7bf30f836c063c37035c51f44156daf407ed62cc3c9765fd6cb383ca488006af`/
`a61677899013df3cb4822a33f262e328dc0afce7e6d16e3dae6cafc1f4e6b898`。该FAIL只关闭缺OpenCLIP的
runtime入口，不修改script/source/rho/loss/batch/门槛。

只读环境审计确认Phase0E runtime有OpenCLIP但缺OpenCV，clean ReID runtime有OpenCV但缺OpenCLIP；
禁止用`PYTHONPATH`混合二者。于是建立独立canonical实体runtime=
`/home/afr/reid-clean/runtimes/exp394-openclip-reid-py310`：复制clean ReID环境后只安装公开精确版本
`open-clip-torch==3.3.0`，没有指回旧环境的symlink。CPU-only依赖与repo import PASS，版本
Torch/OpenCLIP/OpenCV/timm=`2.6.0+cu124/3.3.0/4.13.0/1.0.27`；完整freeze SHA256=
`3d38c99c7f06502d8b40467d2674c966723e5c913d2edf962c5a7088ec60cddb`。构建与import全程GPU=
`2 MiB/0%`且无compute PID。

裁决=`WRONG-RUNTIME-ENTRY SEALED-FAIL / CANONICAL-RUNTIME STATIC-PASS / ACTUAL 24-STEP
PREFLIGHT STILL NO-START / ONE EXACT ACTUAL PREFLIGHT GO / FORMAL NO-START`。只有因为成功更新exact 0，
才允许同一冻结script从step 0进入实际24-step门；实际更新开始后不得重跑或补步。
