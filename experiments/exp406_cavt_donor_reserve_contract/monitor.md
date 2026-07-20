# exp406 donor-reserve合同监控

> 当前：`EXP405 SEALED-FAIL PRESERVED / STATIC V1 BLIND-REVIEW BLOCKED /
> LOCAL STATIC V2 17/17 PASS BYTE-EXACT / FINAL V2 BLIND REVIEW PENDING /
> GPU NO-START / FORMAL NO-START / STUDENT NO-START`

## 2026-07-21：建立新编号

exp405唯一preflight在512图core内找不到满足冻结caliper的wrong-mask donor，权威failure receipt已封板；科学未
评估。两路独立只读盲审确认失败不能外推到15,618图formal universe，并共同要求使用新编号、fresh namespace、
不读取exp405缓存。

初始“preflight删除MAD/caliper、仅保留机械hash donor”的建议因可能降低既有门槛而拒绝。exp406改用固定core
MAD与caliper `8.0`的单调donor-only扩池，完整保留same-camera/different-PID、recipient排除、donor唯一和一对一
匹配；formal科学合同不变。

第三路代码根因审查确认失败发生在caliper edge构建、尚未进入top64/Hall/一对一；现有receipt不能诚实判断
是哪一轴主导，near-zero support MAD仅是风险而非已证实根因。审查提出“在512内先看edge再选recipient”，但
这会改变已冻结recipient选择并可能使preflight更容易，因此未采用；采用更保守的固定recipient、固定core MAD、
单调扩donor-only pool方案。

代码审查同时指出旧CPU合同没有覆盖零边recipient、近零MAD、recipient/donor全局互斥和Hall无解。上述项已加入
exp406强反合同和失败诊断字段。当前三路对“exp405不可重跑、科学未评估、新编号前GPU NO-START”一致；下一步
实现独立CPU/static正反合同。在两次byte-exact PASS和三路新代码盲审`0B/0H`前，不创建远端execution/output/
assets，不占用GPU。

## 2026-07-21：production/static v1本地通过

exp406自包含复制冻结core/teacher并保持其SHA与exp405 exact；新runner只改变exp406 namespace、fresh asset
binding、preflight full-train original donor metadata和固定core尺度的单调prefix matcher。为避免动态DataLoader/
cache状态成为新故障面，preflight固定一次性original编码全15,618图，阶段只控制matching可见pool；scientific
仍为false，diagnostic仍20对。

CPU/static正反合同连续两次`13/13 PASS`且byte-exact，结果SHA256均为
`5b3d7fd6d3ac5ff99302807b85e5753077b2eb30dff4837047d9ccc03348ad3a`。synthetic正例在512阶段零edge、1024阶段
完整匹配；subset-only、caliper `8.1`、preference `128`与Hall无解mutant均被抓。零MAD的`1e-6` floor、20个
recipient/donor不重叠、same-camera/different-PID、donor唯一、formal函数/常量投影、fresh namespace/cache隔离、
failure诊断与CUDA未初始化全部PASS。

源码SHA：runner=`ba0a49b5ad17906511525761b68c1828691b67f20b5ed1a54e16f4e85a312700`，donor module=
`d2a2425303ec1bd56d9ffa67c39afa612263ea0ffa52ef6b261ce557ae99129d`，contract=
`0eb560dcc2c6cb9333baead4e8889ba56d05a9cfbfd0ef0e2d61b137ec727ecc`，core/teacher仍为
`29ddd00ce03ed73b6d1c7ab722de88490e2490638bc83b192e215c6ab4bb0f8b`/
`af255cbbb6eafca2024f7882023deda50445f9a01c1df0b28422a24e23cc35a0`。

当前只授权提交固定快照并开始三路新代码盲审；fresh远端仓库/assets/MMPOSE static和GPU execution仍NO-START。

## 2026-07-21：v1三路盲审阻塞与v2最小修复

固定commit `afe9d490deb56e7c6014d1eed8c70cd95b893419`的三路只读盲审没有授权GPU。合并后的已知
BLOCKER/HIGH为：旧`13/13`合同只实际构造四类mutant；Hall全caliper失败丢失逐层attempt和冲突见证；runner
没有硬绑定MMPOSE-ABU解释器、exp406 core/teacher和冻结pose；CLIP实际字节在started seal与CUDA之后才校验；
formal receipt没有绑定started seal/started.json；static结果未绑定contract自身字节。

v2只修这些已知问题，没有修改teacher科学函数、formal matcher、caliper、MAD、controls或阈值：

- 固定解释器`/usr/local/anaconda3/envs/mmpose-abu/bin/python`；
- core/teacher必须来自当前exp406目录，pose固定为
  `/mnt1/afrderived/exp386_occluded_duke_vitpose_huge_train`；
- pose manifest SHA固定
  `cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8`，CLIP SHA固定
  `9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e`；
- CLIP/pose regular-file与SHA检查前移到seal/CUDA之前；
- COMPLETE绑定started.json和started seal，formal validator验证整条provenance；
- donor plan在public matcher内验证完整单调prefix、stage/hash和full universe；
- Hall失败记录每层limit/status/witness，并由production validator检查真实failure diagnostics；
- static绑定contract自身及全部start/end source SHA，并增加invocation、receipt、plan、core-recipient、固定MAD、
  eligibility与真实public Hall failure场景。

static v2第一次fresh输出真实为`15/17 FAIL`，SHA=
`f5e1be6d8cbb75b6eb22182691f9c0e58e68f43c6c9ba8ec1fdeb6c06f0cb0b2`；失败来自测试构造：same-PID decoy
可被同槽其他PID使用，以及formal AST verifier未识别attribute call。该次不删除、不改判。修正测试合同后两次fresh
输出均为`17/17 PASS`；随后把v2边界补入protocol后重新执行两次最终source-bound合同，仍为`17/17 PASS`，
byte-exact SHA=`9c3286e4ec911c25f09c5797806f8d4ba5ed01a3def81162846612e1473d4061`，CUDA前后均未初始化。

v2源码SHA：runner=`545e07bbb05c763f6ae70be39a92da08e57a35c157c25e5df88d3e7cea5e3f80`，donor=
`59edc1248031d24098c6eebb03f9c7fe1bc235fa19f9751536727ca05ef7a5b1`，contract=
`50ec0010273bf63669aee47104158cedb82f1eaae38e96ee0223d294408a305d`；core/teacher仍与exp405冻结版本
byte-exact。当前只授权固定提交和一次最终三路盲审；0B/0H前仍不创建远端execution/assets，不占用GPU。
