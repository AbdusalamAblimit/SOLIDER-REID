# exp406 donor-reserve合同监控

> 当前：`DESIGN/PROTOCOL FROZEN / EXP405 SEALED-FAIL PRESERVED / LOCAL STATIC V1 PASS /
> NEW-CODE BLIND REVIEW PENDING / GPU NO-START / FORMAL NO-START / STUDENT NO-START`

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
