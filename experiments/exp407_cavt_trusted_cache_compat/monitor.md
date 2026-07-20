# exp407 CAVT cache兼容监控

> 当前：`DESIGN FROZEN / TARGETED MMPOSE ROUNDTRIP PASS BYTE-EXACT / BLIND REVIEW 0B/0H / REMOTE FRESH GATES PENDING / GPU NO-START`

## 2026-07-21：建立fresh编号

exp406已按`SEALED-FAIL / TRUSTED CACHE SELF-CHECK RUNTIME FAILURE / SCIENCE NOT EVALUATED`永久封板。MMPOSE-ABU
已确认Torch 1.13.1、CUDA可用，4090空闲。exp407保持core/teacher字节一致，只迁移fresh namespace并把同进程
受信任临时cache的回读自检改为`weights_only=False`。不读取exp406任何运行产物；只做一个针对性roundtrip和一次
盲审，通过后立即fresh运行。

## 2026-07-21：固定环境targeted roundtrip通过

代码语法检查通过，core SHA=`29ddd00ce03ed73b6d1c7ab722de88490e2490638bc83b192e215c6ab4bb0f8b`、teacher
SHA=`af255cbbb6eafca2024f7882023deda50445f9a01c1df0b28422a24e23cc35a0`，均与exp406字节一致。固定
MMPOSE-ABU/Torch 1.13.1连续两次fresh mixed tensor/metadata roundtrip全部PASS，byte-exact SHA=
`f24577f8f8d31f7824cab15fc3c3ccddf27ca6c5c11c3de37346250fd105e326`。验证覆盖`weights_only=False`源码绑定、
内容回读、临时文件原子消失和重复字节一致；未初始化或占用CUDA。按冻结节奏不追加static，等待一次聚焦盲审。

## 2026-07-21：盲审BLOCKER修复

首轮盲审发现runner虽只改cache loader，但donor模块把冻结排序salt由`exp406-donor`误迁移成`exp407-donor`；这会
重排camera内候选前缀，构成科学对象漂移。已恢复历史冻结salt并在design/protocol中明确其不读取exp406产物。
等待同一盲审者闭环；在0B/0H前GPU保持NO-START。

## 2026-07-21：盲审闭环通过

同一盲审者复核全部exp406→exp407源码差异，最终`0 BLOCKER / 0 HIGH`。core/teacher byte-exact；donor算法与
排序不漂移；runner仅fresh identity/path/schema/module name及trusted cache loader变化；没有读取exp406运行产物。
授权进入fresh远端隔离仓库/asset/output与GPU独占核对，随后启动唯一preflight。
