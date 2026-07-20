# exp405 CAVT 监控记录

> 当前：`BROAD NOVELTY NO-GO / NARROW PHASE-0 CONDITIONAL GO / REAL-TEACHER STATIC V10 REMOTE PASS /
> MMPOSE-ABU FROZEN / THREE-WAY BLIND REVIEW PASS / CUDA PREFLIGHT AUTHORIZED /
> FORMAL P0B NO-START / GPU IDLE`

## 2026-07-20：exp404后根因边界

exp404已封板为`VALIDITY PASS / SPK MECHANISM NO-GO`。本实验不修改或重跑SPK。根因审计确认旧
student evidence到e120仍基本未学会teacher，且presence定义、slot-local readout、双编码teacher与SPK聚合
均偏离用户原始候选。

两路独立审查均反对把失败扩大为CLIP–TAPF总否定，同时也反对直接把“CLIP选槽 + same-ID donor + TAPF
搬运”包装为新方法。新设计因此冻结identity轴 x slot轴二维反事实、可观察original/deleted/donor target与
donor-free同路径student三项必要条件。

## 2026-07-20：train pair可行性只读统计

远端official Occluded-Duke train目录只读统计：

- train images=`15,618`；PID=`702`；
- `702/702` PID具有至少两个camera；
- `15,618/15,618`图可找到same-ID/different-camera donor；
- `510` PID覆盖至少3个camera。

exp386 artifact全量score只读统计：五region的`max(score)>0`比例均为`100%`；阈值`0.7`时分别为
`98.56/98.26/98.26/96.12/94.31%`。这进一步证明旧`semantic_valid`不是visibility。

这些统计只授权继续设计pair map与teacher oracle，不授权CLIP/GPU执行或正式训练。下一步等待近期公开代码
审计闭合后修订新颖性边界，再实现Phase 0 static合同。

## 2026-07-20：根因驱动的teacher/readout拆分

设计现已把CLIP失败拆成两个独立对象：`P0B-PATCH`严格验证原始pose-pooled patch-token teacher；若其失败，
只封板该readout。`P0B-ISO`作为预注册的独立后继，使用从第一层开始的region-isolated frozen image readout，
专门排除旧PC-MBCLS前20层全局CLS残差泄漏。二者均使用frozen image+text双编码器、checkpoint原生
`logit_scale`、actual-view RGB和全body-part prototype分布，不再使用image-only PCA target。

transport oracle新增self-slot restore测量上界和frequency-matched random-cluster强控制；CLIP必须相对
pose-only产生独立收益，且correct必须同时击败wrong-slot与wrong-ID。该修订仍是设计，不创建config、output、
runner，不占用GPU。

## 2026-07-20：近期近邻与第二轮红队收窄授权

MVI²P已覆盖同ID多视图teacher到单图student；RegionCLIP/ProFD/KPR/MUVA已覆盖局部CLIP、part prompt、
可见性和局部表示；FLaN-Net、Composite-Attribute ReID、DPM++与VLCDC进一步覆盖细粒度语言引导、pose槽、
身份/属性关系、动态masked metric与patch transfer。因此宽CAVT机制创新判NO-GO。

只保留最小Phase 0：唯一五槽taxonomy、region-isolated双编码readout、50% primary deletion、identity×slot
双干预、MVI²P/pose-part/attribute-relation/generic-transport近邻，以及held-out PID donor-free可预测性门。
正式config与e120继续NO-CREATE/NO-START。远端只读检查为RTX4090 `2 MiB / 0% / 0 compute PID`。

## 2026-07-20：Phase 0 static v14最终授权

v11--v13均保留为历史、不授权结果，不覆盖、不重判。v14只修复盲审指出的科研正确性合同：完整Torch
RECORD/site-packages闭包、payload/receipt统一异常清理、逐target-slot/PID独立latent、极大/次正规有限值、
feature/mask设备一致性、sample/pixel/slot/seed联合avalanche删除以及camera-aware mAP/R1稳定tie。

唯一v14源码SHA为core=`29ddd00c...f8b`、contract=`13aff524...f60`、bootstrap=`1b9cacf0...17da`。
两次fresh CPU execution均为`56/56 PASS`，canonical payload逐字节相同，SHA256=
`6d073b72894c65236a53ee52d8e1d868e8492c60cc69ade139d57d0560130ee3`；receipt分别绑定各自output名。

最终独立七文件盲审结论为`BLOCKER=0 / HIGH=0 / MEDIUM=0 / LOW=0`。这只授权立即实现真实
region-isolated frozen image+text teacher measurement；不授权直接运行、CUDA、正式训练或e120。静态启动器与
供应链威胁模型到此封口，不再扩审；下一次盲审只针对真实CLIP/pose/data/metric实现是否污染科学结论。

## 2026-07-20：真实teacher measurement v1--v8与最终盲审

真实measurement实现期间没有访问official图像、pose资产、CUDA或GPU。v1/v2的早期`5/5 PASS`尚未覆盖统计与
once-only状态；v3以`7/8 FAIL`抓到旧签名/统计合同错误并永久保留；v4--v7虽分别通过当时合同，但后续盲审继续
发现geometry/readout混淆、recipient被复用为donor、non-torso PID universe污染、top64假FAIL、全候选metadata
内存膨胀和终态窄窗口，故全部只作历史、不授权，不覆盖、不删除。

v8冻结修复为：pose geometry/readout/analysis三轴；从block 1开始的CLIP region isolation与token pooling；固定
512图preflight和每槽4个recipient；所有recipient从donor reserve全局排除；same-camera/different-PID、MAD
caliper与一对一增广匹配按`64 -> 128 -> 256 -> full`扩展；排序持久化为紧凑`int32`且只为最终donor生成
metadata；non-torso在槽`{0,3,4}`上按全局PID共同bootstrap；preflight仅裁决机械有效性；formal以validity x
science四象限裁决；seal前拒绝只读，seal后错误FAILED，COMPLETE不可逆；runtime缺失必要origin/RECORD时
fail closed。

fresh v8 CPU合同连续两次`8/8 PASS`且byte-exact，结果SHA256均为
`45413c3323f7af7636e1e2f9e581b4a9c5fe15c44d4b0a6e47aa987c0ef9f8ca`。源码SHA为
measurement=`f489a6679c57387be49cac4b088d8db49bdb145000e80b2be22aa64dff965981`、
teacher=`af255cbbb6eafca2024f7882023deda50445f9a01c1df0b28422a24e23cc35a0`、
contract=`1146c1e5bf49bbfb040c0467a86173cbb7ec8d1936e122278d997c0c43bcfb19`、
core=`29ddd00ce03ed73b6d1c7ab722de88490e2490638bc83b192e215c6ab4bb0f8b`。

同一固定v8快照的代码、once-only/repro、统计/matching三路只读盲审分别为`0B/0H`、`0B/0H`、
`0B/0H/0M`。该门只授权提交、同步并运行唯一512图CUDA preflight；不授权formal P0B、transport、student、
config或e120。preflight必须自然完成，且只产生机械PASS/FAIL，不得产生scientific GO。

## 2026-07-20：按用户指定冻结MMPOSE-ABU与static v9

用户明确指定远端Conda `MMPOSE-ABU`。只读探针确认其为Python `3.8.20`、Torch `1.13.1+cu117`、
torchvision `0.14.1`、OpenCLIP `2.32.0`、CUDA available且探针后`cuda_initialized=false`；
`VisionTransformer._embeds/_pool`与block `attn_mask`接口存在。v8最初只接受wheel `RECORD`，会拒绝Conda安装的
torch/torchvision；v9只扩展runtime字节绑定：无RECORD时必须在当前`sys.prefix/conda-meta`按包名和精确版本
唯一命中JSON，并同时绑定manifest路径/SHA和实际module origin路径/SHA。Python3.8无`BaseException.add_note`
时也保持主异常优先。matching、bootstrap、scientific gate和once-only语义均未改变。

fresh v9 CPU合同连续两次`8/8 PASS`且byte-exact，结果SHA256均为
`a2e66de37bc4cfbe9ed37dabd9d45761b9590fc5441f08d07877c440fb32f4f4`；measurement/contract SHA=
`52ee00f1eaf817877807ffbd691c09aafdd89288b5c87b56747f99f8695a2648`/
`5422fb34dce954c809c3c28daefa7bb62e4aeafbe71c7e5b7c7ec6bd4242d4ca`。三路只读复审均为`0B/0H`。
此前新建但未完成、未使用的exp405 venv安装已停止并清理；CUDA preflight将只使用用户指定MMPOSE-ABU。

## 2026-07-20：MMPOSE-ABU远端v9兼容失败与v10修复

远端MMPOSE-ABU首次CPU static在取得fixed preflight seal、读取official图像/pose或初始化CUDA之前，以
`AttributeError: module 'ast' has no attribute 'unparse'`退出。根因是合同脚本用了Python 3.9才提供的
`ast.unparse`，而MMPOSE-ABU固定为Python 3.8.20；这不是MMPose、Torch、OpenCLIP或CUDA不兼容，也不消耗
once-only preflight。该次v9远端static如实保留为FAIL。

v10只把禁止训练调用的AST读取改为Python 3.8可用的`ast.Attribute.attr / ast.Name.id`，measurement、teacher、
matching、bootstrap、scientific gate与运行环境均未改变。源码SHA为contract=
`5d15dd73d56714b2dbe725e88a157f889d5f22866126c1076a7fc59a5e351399`、measurement=
`52ee00f1eaf817877807ffbd691c09aafdd89288b5c87b56747f99f8695a2648`、teacher=
`af255cbbb6eafca2024f7882023deda50445f9a01c1df0b28422a24e23cc35a0`。本地两次fresh结果均`8/8 PASS`且
byte-exact，SHA256均为`15ae43641d2e13afd487978033b61b8f83d1702fbfc74972d95a3f733230723c`；代码、复现与
终审三路固定快照均为`0B/0H`。

当时远端SSH在banner阶段连续超时，因此没有把本地v10 PASS误写成远端PASS，也未启动CUDA/GPU。网络恢复后
必须先用`/usr/local/anaconda3/envs/mmpose-abu/bin/python`完成两次远端v10 static并核对byte-exact、CUDA未
初始化、GPU独占和fresh output；全部通过后才允许唯一512图preflight。

网络恢复后已在远端固定MMPOSE-ABU中完成两次v10 static，均为`8/8 PASS`且byte-exact，SHA256均为
`07eeb98692e6d8f54f7bc25dee3fc21803434f4d83ee8e9d33a01a44101123ce`。provenance明确记录Python `3.8.20`、
Torch `1.13.1`和四份冻结源码SHA；两次结果前后`cuda_initialized=false`。复核时4090为`2 MiB / 0% / 0
compute PID`，fixed preflight output、started seal与failed receipt均不存在，远端tracked worktree clean。

该结果把门5闭合并授权唯一512图CUDA preflight；它仍不是科学GO，不含mAP/R1，也不授权formal P0B或student。

## 2026-07-21：唯一512图MMPOSE-ABU CUDA preflight启动

远端冻结HEAD=`ded9ebde1378e9b82e70f5639b7d6df9c731a507`，prelaunch再次确认runner/core/teacher/protocol、
CLIP checkpoint、pose manifest哈希一致，tracked clean，唯一4090空闲且fixed output/seal/FAILED全fresh。随后
使用`/usr/local/anaconda3/envs/mmpose-abu/bin/python`、`CUDA_VISIBLE_DEVICES=0`、batch2、CLIP microbatch1、
workers4启动唯一`exp405-p0b-preflight-v1`。

进程PID=`455109`，已取得immutable started seal并写入`started.json`；首个original batch已进入，GPU观测为
`2362 MiB / 94%`，FAILED receipt不存在。当前判断=`CONTINUE`：只监控自然完成，不修改远端运行中源码、
protocol或参数，不按中间结果早停。该preflight只裁决机械有效性，不计算科学GO或mAP/R1。
