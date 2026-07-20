# exp407 CAVT cache兼容监控

> 当前：`UNIQUE PREFLIGHT COMPLETE PASS / FORMAL BLIND REVIEW 0B/0H / UNIQUE FORMAL RUNNING / STUDENT NO-START`

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

## 2026-07-21：唯一preflight启动

远端fresh隔离仓库=`/home/afr/SOLIDER-REID-exp407-p0b-preflight-37f9175`，HEAD=
`938b786d19a25b0e76b78797970c2609a108e716`且clean。fresh asset=
`/home/afr/reid-clean/assets/exp407-p0b-preflight-v1`，CLIP SHA=
`9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e`，manifest SHA=
`d78d02790b8e268d169ac41f26af5e52a7213e8dbe76526bc0bda7f1b16fd4ee`。隔离仓库内MMPOSE roundtrip再次
PASS；runner/donor/core/teacher/protocol SHA分别为`0037c855f53014d579c6f60b4da2aa80b449c6128cebbe4a328b8092e901d764`、
`b1ef0148ef5c39b06d8d8e41cb81e63bfd5eb81f497d4e26054bef1f4b069f4f`、
`29ddd00ce03ed73b6d1c7ab722de88490e2490638bc83b192e215c6ab4bb0f8b`、
`af255cbbb6eafca2024f7882023deda50445f9a01c1df0b28422a24e23cc35a0`、
`6b24ba65be78488d945aa53f06f9cd938021994a50eff4e94273a2f5d2b21636`。

fresh once-only路径、数据/pose/asset SHA和GPU独占全部通过，随后启动唯一`exp407-p0b-preflight-v1`，主PID=
`460864`。started SHA=`936d8777952564fdf1da8699f1612973a922634822bc58ada691c4d9fa0cbc7d`，started seal SHA=
`c0370e6edf113018a0d82b1e9f2c1f046b5603ea8351cbab3055a30e6109125d`。首次观测到original batch 200，GPU=
`2362 MiB / 96%`，无failure/result/complete。当前`CONTINUE`，自然完成前不改运行中源码、协议或参数。

后续观测到original batch 1500，主进程仍运行，GPU=`2362 MiB / 96%`，无failure/result/complete。判断继续不变。

heartbeat观测到original batch 2700，主进程仍运行，GPU=`2362 MiB / 95%`，无failure/result/complete。吞吐持续、
显存稳定，判断=`CONTINUE`；不修改运行中源码、协议或参数。

## 2026-07-21：唯一preflight自然完成

主进程自然退出，完成全15,618图original与20对diagnostic并发布result/COMPLETE；控制台最终为
`PREFLIGHT_PASS / PREFLIGHT_ONLY_PASS`。八项validity全部PASS：official train exact、coverage exact once、finite、
每槽geometry、每个pose-valid槽有patch、deletion count exact、wrong-mask matching strict、donor reserve strict。
`scientific_evaluated=false`，没有teacher科学结论或ReID mAP/R1。

SHA：result=`9ec2f9042a891ed6e0414bdad50760c543ffd2b625cbc63d0161faae478c39c0`，COMPLETE=
`68ef5f8f8429c6f98c21509b98146ae40c4a075d1c858e6812dad1d3b998cffb`，cache=
`f3ef8c1db0475baadfca2ece9945e58929edf6cef0e6cf0c758e1258b185b634`。COMPLETE绑定started、seal、五个源码
SHA与runtime SHA，`formal_measurement_authorized=true`、`transport_oracle_authorized=false`。运行耗时
612.7001秒；GPU已回到`2 MiB / 0% / 0 compute PID`。按协议立即冻结fresh formal manifest并做一次聚焦盲审。

## 2026-07-21：formal manifest与盲审通过

按远端真实MMPOSE runtime生成formal manifest，schema=`exp407-p0b-formal-manifest-v1`，SHA=
`3932125980989a634df87cb71904e8d2a4772e9bae98ea1dcfb8def35ca70571`。runner自身
`load_and_validate_formal_manifest`只读验证PASS。独立聚焦盲审为`0 BLOCKER / 0 HIGH`，确认formal会fresh重编码
全train并重算2,000对diagnostic/尺度/配对，不复用preflight cache/scale/pair；科学controls和门未漂移。授权立即
启动唯一fresh formal。

## 2026-07-21：唯一formal启动

最终fresh路径、manifest SHA、tracked source clean和GPU独占门通过后启动唯一
`exp407-p0b-iso-teacher-v1`，主PID=`462056`。started SHA=
`488f162df4d7e94c70c010bbdc0493a2756e19f5752e607994cd67310f6a3659`，started seal SHA=
`1ae687d2b0bf6c66cd3a4b221e5a50362506787c44a390c87dd0f52f81c2e776`。首轮观测original batch 100，GPU=
`2362 MiB / 95%`，无failure/result/complete。当前=`CONTINUE`：自然完成前不修改源码、manifest、协议或参数，
不按中间方向早停；student保持NO-START。
