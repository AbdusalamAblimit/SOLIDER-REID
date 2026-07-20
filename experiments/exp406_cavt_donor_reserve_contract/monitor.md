# exp406 donor-reserve合同监控

> 当前：`DESIGN/PROTOCOL FROZEN CANDIDATE / EXP405 SEALED-FAIL PRESERVED / STATIC NOT IMPLEMENTED /
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
