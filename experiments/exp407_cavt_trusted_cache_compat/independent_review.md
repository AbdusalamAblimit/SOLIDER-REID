# exp407 独立聚焦盲审

## 范围

盲审者只读比较exp406与exp407的core、real teacher、donor reserve、runner、design/protocol及targeted roundtrip；
只报告BLOCKER/HIGH，不修改代码、不启动GPU。

## 首轮：1 BLOCKER / 0 HIGH

`phase0_donor_reserve.py`曾把冻结排序salt从`exp406-donor`迁移为`exp407-donor`。这会重排各camera内donor、
分阶段prefix和wrong-mask配对，不是纯namespace变化，与“唯一逻辑修改为cache loader”冲突。启动前必须恢复历史
冻结salt；schema、execution和错误文本仍可迁移exp407。

## 修复

恢复历史冻结salt `exp406-donor`，并在design/protocol中说明该常量只决定official输入的确定性顺序，不读取或授权
exp406运行产物。

## 闭环：0 BLOCKER / 0 HIGH

复核确认：core与real teacher仍byte-exact；donor除schema、错误文本和注释外算法与排序不漂移；runner仅fresh
execution/schema/path/module identity与trusted `weights_only=False`回读变化；未读取exp406运行产物；fresh preflight、
formal、output和asset绑定闭合。固定MMPOSE-ABU roundtrip已证明Torch 1.13 mixed payload两次byte-exact PASS。

**裁决**：授权启动唯一fresh `exp407-p0b-preflight-v1`。
