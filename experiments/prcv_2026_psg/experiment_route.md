# 当前实验路线（PRCV / PSG 线）

## 当前目标

目标不是再开一条全新大路线，而是把 `PSG` 这条线补成**可投 PRCV 的完整证据链**。

当前需要回答的核心问题只有三个：
1. `PSG` 本体是否稳定有效？
2. 为什么最终实现采用 `2-stage PSG`？
3. `GCN` 在最终系统中到底扮演什么角色？

## 当前路线原则

1. 不再把旧的 multi-stage 结果直接当最终消融闭环
2. 所有关键结论都回到干净对照上重跑
3. 优先补能支撑论文主叙事的实验，不再继续开 recipe 小改或 test-time 小变体

## 必做实验

### A. 基础 PSG stage 消融

目标：回答 `PSG` 本体与 stage 数选择

实验矩阵：
- no PSG
- 1-stage PSG
- 2-stage PSG
- 3-stage PSG

希望回答：
- `PSG` 是否稳定优于 no-PSG
- `2-stage` 是否值得作为最终实现
- `3-stage` 是否没有必要

### B. 结构分支依赖性消融

目标：回答 `2-stage PSG` 与高容量 `GCN` 分支的关系

实验矩阵：
- GCN256 + 1-stage PSG
- GCN256 + 2-stage PSG
- GCN512 + 1-stage PSG
- GCN512 + 2-stage PSG

希望回答：
- `exp255 vs exp255b` 看到的现象是否稳定成立
- `2-stage PSG` 是否主要在高容量结构分支上体现价值

### C. 可选 semantic 分支依赖性消融

目标：回答 `2-stage PSG` 的收益是否也依赖 semantic branch

实验矩阵：
- LGPA-only + 1-stage PSG
- LGPA-only + 2-stage PSG
- LGPA+GCN + 1-stage PSG
- LGPA+GCN + 2-stage PSG

说明：
- 这一组不是最优先
- 只有在 A/B 跑完后还有时间再补

## 当前不再优先投入的方向

- visibility 小改动
- retrieval-side scorer / gate / context 微调
- feature-level residual completion 小变体
- 单纯 test-time 涨点路线
- 临时强切到全新问题定义

## 可选补充 benchmark：Occ-PTrack

`Occ-PTrack` 可以测，但当前定位是**补充 benchmark**，不是主线 benchmark。

### 为什么值得补一组

1. `KPR` 已经在 `Occ-PTrack` 上重跑了大量常见方法
2. 可以直接引用其公开表格作为对标背景
3. 若我们的 prompt-free 方法能在这个 benchmark 上接近或超过 `KPR w/o prompt`，说服力很强

### 为什么不能抢主线

1. `Occ-PTrack` 本来是为 promptable / MPA 设定设计的
2. 它天然更偏向 `KPR` 这类方法
3. 对当前 PRCV 稿件来说，`Occluded-Duke` 仍然是主战场

### 当前推荐比较门槛

优先顺序：
1. `KPRSOL w/o prompt`
2. `SOLIDER`
3. `BPBreID`

不把 `KPRSOL with prompt` 设为最低门槛。

### 当前推荐执行方式

1. 只跑一组当前最强 `PSG` 主配置
2. 不开新的 `Occ-PTrack` 消融矩阵
3. 先看能否达到或超过 `KPRSOL w/o prompt`
4. 若结果不具竞争力，立刻止损，不影响主线消融

## 当前实验完成后的写作落点

实验补齐后，论文里要能稳定回答：

1. 为什么主创新是 `PSG`
2. 为什么最终实现是 `2-stage PSG`
3. 为什么 `GCN` 要被保留在最终方法里
4. 哪些东西只是系统资产，哪些才是主贡献

## 当前最推荐的落地顺序

1. 先跑 A：基础 stage 消融
2. 再跑 B：结构分支依赖性
3. 最后视时间决定是否补 C
4. 若主线完成且资源允许，再补一组 `Occ-PTrack`

## 当前结果引用口径

- 主表默认测试协议包含 `flip-test`
- `MaxSim` 单独作为附加匹配行，不并入默认主结果
- 若历史实验尚未记录 `equal_concat + flip-test` 数字，则先保持原始记录，待补测后再进论文主表
- `exp255` 仍是当前训练端主 scaffold
- `exp255 vs exp255b` 用来回答 two-stage 选择依据
- `Occ-PTrack` 若加入论文，默认放在 supplementary / secondary benchmark 区域
