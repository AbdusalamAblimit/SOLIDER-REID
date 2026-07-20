# exp406 独立盲审记录

## 审查边界

三路只读审查均未编辑文件、未启动GPU，也未使用Claude。材料包括exp405冻结源码、started/failure/console、
once-only路径和exp406候选池修复提案。

## 代码根因审查

初始计数=`1 BLOCKER / 1 HIGH`。

- BLOCKER：exp405先按PID/hash冻结20 recipients，再在仅512图的池内要求正式caliper edge；至少一个recipient
  有same-camera/different-PID/analysis-valid候选，但全部四轴MAD距离`>8.0`。异常发生在preference构建，尚未
  进入top64、Hall或一对一assignment。
- HIGH：旧CPU合同只覆盖刻意可匹配的小表，没有覆盖零边recipient、零/近零MAD、全局recipient/donor互斥、
  Hall无解和完整`choose_diagnostic_subset -> choose_wrong_masks`组合路径。

审查提出先看edge再选机械recipient。主审拒绝该方案，因为它改变已冻结recipient选择并可能降低preflight难度；
exp406改用固定recipient与尺度、只单调扩donor pool。HIGH要求的mutant和诊断字段全部纳入design/protocol。

## 统计与matching审查

初始建议删除preflight MAD/caliper，只保留机械配对；因heartbeat禁止降低既有门槛，该建议被退回。修订计数=
`1 BLOCKER / 0 HIGH`，提出固定core MAD的单调前缀：

`512 -> 1024 -> 2048 -> 4096 -> 8192 -> full train`。

recipient、尺度、caliper `8.0`、same-camera/different-PID、recipient排除、donor唯一和一对一全部不变；新增图只作
donor，每图最多编码一次，feasible edge集合单调增加。full train仍无完整解则FAIL。formal独立全train重算MAD，
不复用preflight任何scale/pair/cache。

## 复现与once-only审查

exp405终态计数=`0 BLOCKER / 0 HIGH`：started seal、started.json、failure.json完整，result/complete/cache在异常
之后才会写，因此不存在授权产物；formal validator必拒绝，同编号不可重跑。

立即启动任何后继的计数=`1 BLOCKER / 0 HIGH`：exp406 design、fresh namespace/assets、static正反合同和新代码
尚未闭合。审查冻结新execution/output，并要求exp405输入拒绝、formal投影exact、asset/runtime前后SHA exact。

## 合并裁决

`EXP405 SEALED-FAIL PRESERVED / EXP406 CONTRACT DESIGN ACCEPTED / IMPLEMENTATION BLOCKED`

当前允许实现exp406 CPU/static代码；不允许创建远端execution/output/assets或占用GPU。只有两次fresh static
byte-exact且三路新代码复审均`0B/0H`，才可进入远端MMPOSE-ABU static与fresh资产准备。

## 固定commit afe9d490的新代码盲审

三路最终计数分别为`1B/2H`、`2B/3H`、`1B/1H`，共同裁决`GPU NO-START`。共同问题是static覆盖不足和Hall
失败attempts丢失；复现审查另外抓到MMPOSE-ABU与exp406源码/冻结pose未硬绑定、CLIP字节检查晚于seal/CUDA、
formal缺少started provenance；代码审查抓到static未绑定自身字节。production formal科学投影与原matching门未发现
降低。

上述项已由static v2最小修复；第一次`15/17 FAIL`与随后两次byte-exact `17/17 PASS`均保留。当前等待对新固定
commit的一次最终三路盲审；旧v1审查不能授权v2。

## 固定commit c2eeefb的v2最终盲审

复现/once-only=`0B/0H`；代码=`1B/1H`；合并=`1B/0H`。共同剩余BLOCKER是formal控制流与caliper-delete
mutant不充分；代码路另将Hall witness schema列为HIGH，并要求非同质preference-order oracle。v3已逐项闭合并取得
两次byte-exact `19/19 PASS`。下一次审查只复核这些差异，不新增非阻塞要求。
