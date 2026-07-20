# 实验 exp407：CAVT受信任cache回读兼容修复

## 动机

exp406在MMPOSE-ABU中已完成15,618张图和20对diagnostic，但在正式发布result/COMPLETE前，Torch 1.13无法用
`weights_only=True`回读本进程刚写出的混合tensor/metadata cache。exp406因此永久封板为测量器runtime失败，科学
问题未评估。exp407只恢复该问题的可测性，不重跑或改判exp406。

## 核心假设

该临时文件由当前进程创建、fsync并在发布前立即回读，属于受信任输入；将自检改为
`torch.load(..., weights_only=False)`可兼容Torch 1.13，同时保留原子发布、schema校验、SHA和once-only语义。

## 唯一核心变量

- fresh execution：`exp407-p0b-preflight-v1`
- fresh output：`/home/afr/reid-clean/audits/exp407-p0b-preflight-v1`
- fresh asset：`/home/afr/reid-clean/assets/exp407-p0b-preflight-v1`
- formal：`exp407-p0b-iso-teacher-v1`
- 唯一逻辑修改：受信任临时cache自检的`weights_only=True -> False`

core与real teacher必须和exp406字节一致；teacher目标、donor合同（包括历史冻结salt `exp406-donor`）、20对controls、阈值、batch和formal门不变。
所有schema、execution和源码路径迁移到exp407。不得读取exp406 output、cache、pair、MAD或receipt作为运行输入。

## 对照与强反事实

科学对照保持原CAVT协议：correct region、wrong mask、25/50/75% support deletion、generic/NULL/random及formal中的
clean语义门。当前兼容合同只需在固定MMPOSE-ABU中证明包含tensor与metadata的代表性cache可两次fresh byte-exact
roundtrip，且发布文件的schema和内容不变。

## 授权门

简洁源码检查、一次MMPOSE-ABU roundtrip和一次独立聚焦盲审。盲审只处理BLOCKER/HIGH；0B/0H后立即启动唯一
fresh preflight，不追加大规模static。preflight机械PASS后立即冻结formal manifest并进入科学测量。

## 风险与失败解释

1. roundtrip仍失败：exp407测量器兼容FAIL，封板并用新编号修正，不解释CAVT科学性。
2. preflight validity失败：说明数据/pose/teacher/control合同失败，科学未评估，不调阈值救臂。
3. formal科学NO-GO：封板CAVT对象，立即切换下一种pose+CLIP训练机制。
4. formal科学GO但student不涨点：teacher可辨识不等于ReID收益，方法不能宣称有效，转下一机制。
