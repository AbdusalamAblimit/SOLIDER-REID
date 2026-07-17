# exp375 PRSM 监控记录

## 2026-07-15 — 设计与实现启动

exp374 已正式 `COMPLETE / NO_GO`，因此本实验独立于 PSG gate 小变体。三路并行查新、仓库
接入检查与审稿人红队已完成：直接空间仍存在，但简单 `pose -> Delta/B/C`、解剖扫描顺序
和 skeleton state fusion 都已有强先例。主线收敛为 pose-routed part-state write/retain。

当前状态：`IMPLEMENTATION_IN_PROGRESS`。下一步直接完成模块、默认关闭配置、CPU
forward/backward 与反事实单测；随后做多路 Codex 代码/科学审查。审查通过后立即在 GPU
做单 batch AMP smoke，并启动 B0/M0/P0，不再增加形式性门禁。

## 2026-07-15 — 实现审查与本地机制测试 PASS

PRSM、默认关闭配置、B0/M0/P0 三份隔离 config、模块单测与真实 Swin 集成 smoke 已完成
编码。三路静态审查在修复 YAML、canonical cache、pre-write read 和 target-only 集成覆盖
后全部 PASS。

仓库 uv 隔离环境执行：

```text
.venv-exp374/bin/python tests/test_pose_routed_selective_memory.py
PRSM mechanism checks: PASS
```

该测试覆盖 shape/finite、关键参数与输入梯度、zero-pose exact identity、correct/shuffle
敏感性、双向纵翻转等变性、uniform 对 pose bitwise 不敏感及 CPU bfloat16 autocast。当前
状态升级为 `PASS_FOR_REAL_MODEL_GPU_SMOKE`；下一步只做 4090 production
B0/M0/P0 forward/backward/reload smoke，通过后直接启动训练。

首次远端 smoke 在进入模型构造前即以 `ModuleNotFoundError: config` 退出：直接执行
`tests/...py` 时 Python 只把 `tests/` 放入 import path。该失败未运行 forward、未产生指标，
属于测试入口缺失。测试已显式把仓库 root 放入 `sys.path`；模块与实验机制未改动。

## 2026-07-15 — 真实 Swin smoke PASS，Gate A 启动

修复测试入口后，4090 与 3090 均用各自正式训练解释器完成真实
`PoseBackboneModel` 构建、B0/M0/P0 配置隔离、目标实例 heatmap 接线、生产 forward、
PRSM backward 及严格 checkpoint reload smoke，结果均为 PASS。训练执行代码固定在
`f8d5570b6d5fd6f80292db7cbae3cfcc5ba925cf`；方法实现仍对应 `f566c0b`，后一个提交只修复
测试入口。

正式启动两条互不重复的探索臂：

- 4090：`P0`，当前实例姿态路由，seed 1234；
- 3090：`M0`，固定 canonical 路由，seed 1234。

3090 的 PyTorch/CUDA 运行时与 4090 不同，因此其 M0 只用于提前判断趋势；任何接近门槛的
差值必须用 4090 同机同运行时 M0/B0 复核。两臂均只有一个 main 进程及 8 个 DataLoader
worker，无重复 controller。

## 2026-07-15 — P0 epoch 10 首次评测

4090 P0 epoch 10 完整评测：

| arm | epoch | mAP | R1 | R5 | R10 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| P0 correct | 10 | 35.9 | 44.0 | 60.6 | 67.2 | 健康继续 |

早期参考只使用既有 4090 clean B0 的同 epoch 轨迹：epoch 10 mAP 33.4，因此 P0 暂为
`+2.5 mAP`。该比较尚未满足“同一 execution、同一当前运行时的 B0/M0”最终证据要求，
只说明 PRSM 有继续跑到首个可保存 checkpoint 并做反事实评测的燃料，不升级为 GO 结论。
当前未见 NaN/Inf/Traceback，训练继续；3090 M0 尚未到 epoch 10。

3090 M0 随后完成 epoch 10：35.1 mAP / 44.7 R1 / 60.3 R5 / 66.6 R10。
P0−M0 暂为 `+0.8 mAP / -0.7 R1`；由于 3090 与 4090 运行时不同，且仍处于前半程，
该差值只记录轨迹，不作 Gate A 裁决。

## 2026-07-15 — P0 epoch 20 评测与首个 checkpoint

| arm | epoch | mAP | R1 | R5 | R10 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| P0 correct | 20 | 40.5 | 50.1 | 64.8 | 71.2 | 健康继续；checkpoint 已保存 |

`transformer_20.pth` 为 `113822117` bytes，SHA256 =
`7470a8c930f40369a61e167d1f323004730e8f0b7d97834274023a0058d0923b`。历史同机 clean
B0 epoch 20 为 43.1 mAP，因此 P0 当前为 `-2.6 mAP`，epoch 10 的早期正差未保持。
该信号不满足 Gate A，但仍不足以单独终止。按用户新增裁决纪律，任何 `<60 epoch` 结果
只用于轨迹记录、反事实管线 smoke 与异常诊断，不得据此判负或终止。训练保持唯一进程
继续；该 checkpoint 可用于验证 correct / matched-shuffle / foreground-uniform / zero /
canonical 同权重评测入口，但正式机制裁决最早只能使用 epoch 60 或之后的 checkpoint。

## 2026-07-15 — P0 epoch 30 轨迹

| arm | epoch | mAP | R1 | R5 | R10 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| P0 correct | 30 | 49.6 | 59.3 | 73.4 | 78.8 | 健康继续 |

历史同机 clean B0 epoch 30 为 49.2 mAP，P0 当前暂为 `+0.4 mAP`。P0 轨迹为
e10 `35.9/44.0` → e20 `40.5/50.1` → e30 `49.6/59.3`（mAP/R1）。唯一 main 与 8 个
DataLoader workers 正常，未见 NaN/Inf/Traceback/OOM；按 `<60 epoch` 纪律只记录，不裁决。

## 2026-07-15 — P0 epoch 40 轨迹

| arm | epoch | mAP | R1 | R5 | R10 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| P0 correct | 40 | 50.0 | 60.0 | 73.5 | 79.3 | 健康继续 |

P0 完整轨迹更新为 e10 `35.9/44.0` → e20 `40.5/50.1` → e30 `49.6/59.3` →
e40 `50.0/60.0`。历史同机 clean B0 epoch 40 为 52.8 mAP；这里只记录差值，不作早期
裁决。唯一 main、8 workers 与 GPU 均健康，无 NaN/Inf/Traceback/OOM。

checkpoint 内部状态确认 PRSM 并非停在近零初始化：`residual_scale` 从初始化 `0.001` 增长为
epoch 20 的 `0.0152574`、epoch 40 的 `0.0280665`；retention mean 从初始化 `0.95` 变为
`0.9497741 / 0.9493675`。因此当前轨迹可以视为模块确实参与优化，不属于 alpha/梯度完全
死亡；其科学价值仍须由 epoch 60+ 反事实与 B0/M0 门禁决定。

## 2026-07-15 — 冻结反事实评测入口与真实 donor preflight PASS

新增同一模型实例、同一严格加载 checkpoint 的六臂离线入口：correct-start、
matched-shuffle、canonical、foreground-uniform、zero-bypass、correct-end。科学复审拒绝了
最初的 batch-local shuffle 草案，正式实现改为预冻结的 query/gallery split-local donor
map；映射必须为无 fixed point、异 PID 的双射，且与 batch size、worker 数和遍历顺序无关。
运行时另审计 target-only PRSM write mass、support、纵向中心/跨度和六部位质量；所有 arm
核对 RGB/path/PID/camera 顺序，zero 逐 forward 断言 exact identity，correct-start/end 核对
descriptor SHA 与指标精确复现。

本地隔离测试：`13 passed`，PRSM 机制测试 PASS，包含 foreground-uniform 逐位置总写入量
与 correct 严格相等。4090 真实 Occluded-Duke preflight 使用 exp374 已冻结且独立于 exp375
指标生成的唯一 mapping：query SHA =
`c8beb33b7d7ead13d0f47e891c44751de49fe5c55b1fc50c0d79e333c4e60452`，gallery SHA =
`904dc793c6940d744a194aa0e5c919c29f4b004d6edeecddc1acb7be259113e1`；全 `2210 + 17661`
样本通过 split、双射、无 fixed point、异 PID 与稳定路径检查。尚未读取任何新 arm 指标；
正式反事实最早在 epoch 60 checkpoint 执行。

## 2026-07-15 — P0 epoch 50/60/70 轨迹

| arm | epoch | mAP | R1 | R5 | R10 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| P0 correct | 50 | 51.0 | 62.2 | 76.2 | 81.1 | 健康继续 |
| P0 correct | 60 | 53.2 | 62.1 | 75.8 | 81.2 | 健康继续；正式反事实 checkpoint 已具备 |
| P0 correct | 70 | 56.3 | 66.4 | 80.3 | 85.8 | 健康继续 |

历史同机 B0 epoch 60 为 54.5 mAP，P0 epoch 60 暂低 `1.3 mAP`；该历史差值不是当前
execution 的最终同配方 B0 门禁，且 P0 在 epoch 60 后仍由 53.2 上升至 56.3 mAP，故不作
终止依据。唯一 main、8 个 DataLoader workers 与 GPU 正常，训练已进入 epoch 77，未见
NaN/Inf/Traceback/OOM。正式 target-only matched donor map 正在独立构建；通过预注册 nuisance
门槛后直接对 epoch 60+ checkpoint 跑同权重六臂，不等待训练结束。

3090 M0 已完成 epoch 30：49.3 mAP / 59.4 R1 / 73.2 R5 / 79.0 R10，并继续进入
epoch 35 之后；该机器运行时与 4090 不同，只记录容量对照趋势，不代替后续 4090 同机 M0。

## 2026-07-15 — P0 epoch 80/100 与 e80 五臂反事实

| arm | epoch | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|---:|
| P0 correct | 80 | 56.8 | 65.7 | 80.5 | 85.4 |
| P0 correct | 100 | 57.1 | 66.8 | 80.9 | 85.5 |

e80 checkpoint SHA256 =
`de8e02b22a868d4ceaa9d48df8d736065d8c862c67e5fd954fcc87e502d771f2`。同一模型实例、
同一 checkpoint 的非 shuffle 五臂结果如下（原始 mAP 为 `[0,1]` 比例）：

| arm | mAP | R1 | correct−arm mAP 百分点 |
|---|---:|---:|---:|
| correct-start | 0.5678210972 | 0.6574660540 | 0 |
| canonical | 0.5678340283 | 0.6574660540 | -0.001293 |
| foreground-uniform | 0.5678156348 | 0.6574660540 | +0.000546 |
| zero-bypass | 0.5678296803 | 0.6574660540 | -0.000858 |
| correct-end | 0.5678210972 | 0.6574660540 | 0 |

各 arm descriptor SHA 不同，证明 source/routing 切换实际生效；correct-start/end 的 descriptor
SHA 与全部指标精确复现。zero-bypass 共 622 次 forward，PRSM 输入输出逐次 bitwise identity。
当前结论边界：e80 的 memory 写回及 pose route 尚未改变检索排序到可测量量级，三项对应硬门禁
均未达到；训练仍跑满 120 epoch，并继续补 target-matched shuffle 与当前 execution 同机 B0/M0。
原 top-256 sparse assignment 在 query 阶段运行约 20 分钟仍无产物，已精确终止；top-32
快速确认无 full matching，现只把候选图扩为 top-64 重试，nuisance、随机基线与预注册阈值
均未改变。

3090 M0 轨迹继续更新：epoch 40 为 49.9 mAP / 59.3 R1 / 73.4 R5 / 79.8 R10，
epoch 50 为 52.8 mAP / 62.6 R1 / 77.0 R5 / 82.2 R10；训练健康继续。该跨运行时趋势仍不作
最终 P0−M0 裁决。

## 2026-07-15 — target-only mapping PASS、e80 matched-shuffle 与 P0 完成

原精确 sparse min-weight solver 的复杂度不适合 17,661 gallery；最终构建器保留每行 top-64
真实 nuisance 近邻，并增加一条固定 seed 的 constrained-random 完整双射边作为连通性保险，
先求完整双射再作确定性 2-opt 降成本。该工程改动不更改 target-only 35D cost、20 个随机基线
或预注册接受线；专门的 identical-cluster 单测证明等成本样本不会再集中抢同一批 donor。

冻结 map 的 builder 与真实 evaluator 双重门禁均 PASS：

| split | cost / random median | max dimension median abs-z | zero concordance |
|---|---:|---:|---:|
| query | 0.3203208 | 0.3889845 | 1.0 |
| gallery | 0.3735022 | 0.4683620 | 1.0 |
| combined | 0.3646880 | 0.4598496 | 1.0 |

三项预注册上限分别为 `0.75 / 0.65 / 1.0`，且 query/gallery 各自满足顺序绑定、双射、
无 fixed point、异 PID 与 split-local。MANIFEST SHA256 =
`f6aafd9935b301d264ab62d3372151c5acb199925eda5fce493f2afb3b8dc6ba`。

e80 同 checkpoint matched-shuffle 为 0.5678187445 mAP / 0.6574660540 R1；相对
correct-start 的 `correct−matched = +0.000235` mAP 百分点，远低于 `+0.5` 门槛。descriptor
SHA 与 correct 不同，且 evaluator 实时复算的 query/gallery/combined nuisance gate 全部 PASS，
故不是 arm 未切换或普通随机 pose 混淆。至此 e80 的 correct−matched、correct−foreground、
correct−zero 三条因果门禁均近似为零。

P0 已完整结束 120 epoch，最终为 57.1 mAP / 66.3 R1 / 80.3 R5 / 85.3 R10，无异常。
4090 训练位随后立即启动当前 execution 同机 B0，main PID `66995`，配置
`exp375_b0.yml`，输出 `log/occluded_duke/exp375_b0_s1234`；启动后主进程、workers 与 GPU
均正常。P0 最终 checkpoint 的六臂将在不阻塞 B0 训练的前提下补齐。

## 2026-07-15 — e120 最终六臂：PRSM 机制 Gate NO-GO

最终 checkpoint SHA256 =
`875829299b1a89125e8ad3d0ef9dc89f3e8e3579968c7abe16f1a0cdf089d04c`；六臂结果：

| arm | mAP | R1 | correct−arm mAP 百分点 |
|---|---:|---:|---:|
| correct-start | 0.5708388253 | 0.6628959179 | 0 |
| matched-shuffle | 0.5708422085 | 0.6628959179 | -0.000338 |
| canonical | 0.5708560170 | 0.6628959179 | -0.001719 |
| foreground-uniform | 0.5708457300 | 0.6628959179 | -0.000690 |
| zero-bypass | 0.5708208791 | 0.6628959179 | +0.001795 |
| correct-end | 0.5708388253 | 0.6628959179 | 0 |

六臂 R1/R5/R10 完全相同；各反事实 descriptor SHA 均不同，correct-start/end 的 descriptor
SHA 与指标精确复现；zero 共 622 次 forward 全部 exact identity；matched evaluator 再次实时
验证 query/gallery/combined target-write nuisance 门禁 PASS。因此三条硬门禁
`correct−matched >= +0.5`、`correct−foreground >= +0.3`、`correct−zero >= +0.5` 均以远超
数值噪声的幅度未通过，且 design 预注册的 `correct−matched < +0.2` NO-GO 条件已触发。

正式机制裁决：**当前 PRSM 的实例姿态路由、部位状态归属与推理时 memory 写回均未对检索排序
产生可报告贡献，PRSM 作为论文主创新 NO-GO。** 这不等同于提前终止证据收集：4090 同机 B0
继续运行，随后补同机 M0，以判断 P0 最终训练结果是否只来自通用容量/训练正则；该结果不能
挽回姿态路由机制 claim，但会决定本实验的完整失败解释。

P0 最终 checkpoint 内 `residual_scale=0.0324680`，retention logits mean `2.9194412`
（sigmoid 约 0.9488），进一步证明模块参数已离开初始化，并非实现未接入或参数完全死亡；但其
写回幅度仍未改变检索排序。

4090 同机 B0 前两次评测：epoch 10 为 34.5 mAP / 43.3 R1 / 59.5 R5 / 65.5 R10，
epoch 20 为 42.8 mAP / 53.3 R1 / 67.8 R5 / 73.8 R10。对应 P0 为 e10 35.9/44.0、
e20 40.5/50.1（mAP/R1）：早期优势未保持。B0 主进程与 8 workers 健康继续。

3090 跨运行时 M0 轨迹：epoch 70 为 56.5 mAP / 65.4 R1 / 80.0 R5 / 84.8 R10，
epoch 80 为 57.6 mAP / 67.6 R1 / 81.3 R5 / 86.0 R10；已超过 P0 e120 的 57.1 mAP，
进一步支持通用 canonical memory 不弱于实例 pose，但因运行时不同仍等待 4090 同机 M0
作最终训练对照。

后续控制轨迹：4090 B0 epoch 30 为 50.6 mAP / 60.5 R1 / 74.7 R5 / 80.0 R10，
相对同机 P0 epoch 30 的 49.6/59.3 已领先 `+1.0 mAP / +1.2 R1`。3090 M0 epoch 90
为 58.4 mAP / 68.3 R1 / 81.7 R5 / 86.2 R10，比 P0 e120 高 1.3 mAP（仍仅作跨运行时
趋势）。两路训练均健康继续，无 NaN/Inf/Traceback/OOM。

4090 B0 继续为 e40 `52.2/62.0`、e50 `53.6/63.7`、e60 `55.2/65.0`
（mAP/R1）；同机 P0 对应为 `50.0/60.0`、`51.0/62.2`、`53.2/62.1`，B0 分别领先
`+2.2/+2.0`、`+2.6/+1.5`、`+2.0/+2.9`。因此截至中点，PRSM 不仅没有实例 pose 因果
收益，整体训练结果也稳定低于 image-only B0。3090 M0 epoch 100 达到
58.7 mAP / 68.7 R1 / 81.8 R5 / 86.6 R10；两路进程均健康继续。

B0 后续为 e70 `56.8/66.6`、e80 `58.5/68.3`、e90 `57.7/67.2`、
e100 `58.7/68.3`（mAP/R1）。除 e90 正常单点评测波动外，B0 始终不弱于同阶段 P0，
且 e80 已比 P0 最终 e120 高 `+1.4 mAP / +2.0 R1`。

3090 跨运行时 M0 已完整正常结束：e110 为 58.8 mAP / 68.6 R1 / 81.8 R5 / 86.5 R10，
e120 最终为 58.9 mAP / 68.6 R1 / 82.0 R5 / 86.7 R10。主进程与 workers 已自然消失，
GPU 释放，日志结束于完整四项，全文无异常。相对 P0 最终，M0 跨运行时领先
`+1.8 mAP / +2.3 R1`；虽不替代 4090 同机 M0，但方向与最终同 checkpoint canonical
反事实完全一致。

4090 B0 已完整正常结束：e110 为 58.3 mAP / 67.4 R1 / 81.1 R5 / 85.7 R10，
e120 最终为 58.4 mAP / 67.1 R1 / 81.2 R5 / 85.6 R10；进程/workers 自然退出，GPU
释放，日志无异常。同机最终 `P0−B0=-1.3 mAP / -0.8 R1`，未达到预注册 `+0.8 mAP`
门槛，且方向明确为负。

B0 释放 4090 后立即启动同运行时 M0，main PID `103534`，配置
`exp375_m0_canonical.yml`，输出 `log/occluded_duke/exp375_m0_canonical_s1234`。启动日志明确
`source=canonical`、PRSM 参数 299649，主进程、workers 与 GPU 正常；这是 exp375 最后一项
训练对照。

4090 同机 M0 epoch 10 完整评测为 33.7 mAP / 42.3 R1 / 58.4 R5 / 65.1 R10；
主进程与 8 workers 健康进入后续 epoch，无 NaN/Inf/Traceback/OOM。对应 B0/P0 e10 为
34.5/43.3 与 35.9/44.0（mAP/R1），仅记录早期轨迹，不作最终裁决。

4090 同机 M0 epoch 20 为 42.5 mAP / 52.0 R1 / 68.5 R5 / 74.6 R10，健康继续。
对应 B0/P0 为 42.8/53.3 与 40.5/50.1（mAP/R1）；M0 当前接近 B0、优于 P0，但仍属
`<60 epoch` 轨迹，不作最终裁决。

4090 同机 M0 epoch 30 为 47.1 mAP / 56.5 R1 / 71.7 R5 / 77.9 R10，主进程与
8 个 DataLoader workers 健康继续，日志无 NaN/Inf/Traceback/RuntimeError/OOM。对应
B0/P0 为 50.6/60.5 与 49.6/59.3（mAP/R1）；该点仍处于 `<60 epoch`，只记录轨迹，
不据此作最终裁决。

4090 同机 M0 epoch 40 完整评测为 51.5 mAP / 61.1 R1 / 76.8 R5 / 81.6 R10；
对应 B0/P0 为 52.2/62.0 与 50.0/60.0（mAP/R1）。M0 当前位于两者之间，训练健康进入
后续 epoch；该点仍只作轨迹记录，不改变 e120 最终裁决规则。

4090 同机 M0 epoch 50 为 51.6 mAP / 61.2 R1 / 75.8 R5 / 81.2 R10；对应 B0/P0
为 53.6/63.7 与 51.0/62.2（mAP/R1）。M0 的 mAP 略高于 P0、低于 B0，R1 低于二者；
训练保持健康，继续到完整 e120。

4090 同机 M0 epoch 60 为 55.1 mAP / 65.1 R1 / 78.4 R5 / 83.2 R10；对应 B0/P0
为 55.2/65.0 与 53.2/62.1（mAP/R1）。到预注册中点，M0 与 B0 基本持平，并比 P0 高
`+1.9 mAP / +3.0 R1`；这进一步支持 canonical memory 不弱于实例 pose route，但最终
训练裁决仍等待三臂完整 e120。主进程、8 个 workers 与 GPU 健康进入 e61，异常扫描为零。

4090 同机 M0 epoch 70 为 57.0 mAP / 66.5 R1 / 81.2 R5 / 85.4 R10；对应 B0/P0
为 56.8/66.6 与 56.3/66.4（mAP/R1）。三臂此处接近，M0 的 mAP 略高；只记录完整轨迹，
继续按预注册规则等待 e120。

4090 同机 M0 epoch 80 为 58.4 mAP / 68.1 R1 / 82.1 R5 / 86.4 R10；对应 B0/P0
为 58.5/68.3 与 56.8/65.7（mAP/R1）。M0 与 B0 基本重合，并比 P0 高
`+1.6 mAP / +2.4 R1`；训练健康继续。

本地最终机制与反事实测试复跑：`15 passed in 0.99s`，覆盖 PRSM update/identity、
counterfactual source、mapping invariants、target-write gate 与 evaluator 关键不变量。

4090 同机 M0 epoch 90 为 58.4 mAP / 68.5 R1 / 81.7 R5 / 86.1 R10；对应 B0/P0
为 57.7/67.2 与 56.3/65.6（mAP/R1）。M0 此点评测高于二者，训练健康进入最后 30 epoch；
仍以 e120 作为最终同机 P0−M0 裁决。

4090 同机 M0 epoch 100 为 58.7 mAP / 67.7 R1 / 81.4 R5 / 86.0 R10；对应 B0/P0
为 58.7/68.3 与 57.1/66.8（mAP/R1）。M0 与 B0 的 mAP 相同，并比 P0 高
`+1.6 mAP / +0.9 R1`；训练健康进入最后 20 epoch。

4090 同机 M0 epoch 110 为 58.7 mAP / 67.3 R1 / 81.6 R5 / 86.1 R10；对应 B0/P0
为 58.3/67.4 与 57.0/66.4（mAP/R1）。M0 的 mAP 仍高于二者，主进程与 workers 健康
进入最后 10 epoch。

## 2026-07-15 — 同运行时 M0 完成与 exp375 最终裁决

4090 同机 M0 epoch 120 为 58.8 mAP / 67.5 R1 / 81.5 R5 / 86.2 R10。训练主进程与
8 个 workers 自然退出，GPU compute process 清空；train log 与 runner stdout 均完整结束于
e120 四项，NaN/Inf/Traceback/RuntimeError/OOM 扫描为零。

三条同运行时训练的每次完整 eval 如下，单元格顺序均为 `mAP / R1 / R5 / R10`：

| epoch | B0 image-only | M0 canonical PRSM | P0 instance-pose PRSM |
|---:|---:|---:|---:|
| 10 | 34.5 / 43.3 / 59.5 / 65.5 | 33.7 / 42.3 / 58.4 / 65.1 | 35.9 / 44.0 / 60.6 / 67.2 |
| 20 | 42.8 / 53.3 / 67.8 / 73.8 | 42.5 / 52.0 / 68.5 / 74.6 | 40.5 / 50.1 / 64.8 / 71.2 |
| 30 | 50.6 / 60.5 / 74.7 / 80.0 | 47.1 / 56.5 / 71.7 / 77.9 | 49.6 / 59.3 / 73.4 / 78.8 |
| 40 | 52.2 / 62.0 / 77.1 / 82.9 | 51.5 / 61.1 / 76.8 / 81.6 | 50.0 / 60.0 / 73.5 / 79.3 |
| 50 | 53.6 / 63.7 / 77.1 / 82.4 | 51.6 / 61.2 / 75.8 / 81.2 | 51.0 / 62.2 / 76.2 / 81.1 |
| 60 | 55.2 / 65.0 / 77.6 / 83.1 | 55.1 / 65.1 / 78.4 / 83.2 | 53.2 / 62.1 / 75.8 / 81.2 |
| 70 | 56.8 / 66.6 / 79.8 / 84.8 | 57.0 / 66.5 / 81.2 / 85.4 | 56.3 / 66.4 / 80.3 / 85.8 |
| 80 | 58.5 / 68.3 / 81.9 / 86.6 | 58.4 / 68.1 / 82.1 / 86.4 | 56.8 / 65.7 / 80.5 / 85.4 |
| 90 | 57.7 / 67.2 / 81.3 / 85.7 | 58.4 / 68.5 / 81.7 / 86.1 | 56.3 / 65.6 / 79.9 / 85.7 |
| 100 | 58.7 / 68.3 / 81.6 / 86.5 | 58.7 / 67.7 / 81.4 / 86.0 | 57.1 / 66.8 / 80.9 / 85.5 |
| 110 | 58.3 / 67.4 / 81.1 / 85.7 | 58.7 / 67.3 / 81.6 / 86.1 | 57.0 / 66.4 / 80.4 / 85.2 |
| 120 | **58.4 / 67.1 / 81.2 / 85.6** | **58.8 / 67.5 / 81.5 / 86.2** | **57.1 / 66.3 / 80.3 / 85.3** |

最终同机差值：`P0−B0=-1.3/-0.8/-0.9/-0.3`，`P0−M0=-1.7/-1.2/-1.2/-0.9`，
`M0−B0=+0.4/+0.4/+0.3/+0.6`（依次 mAP/R1/R5/R10）。因此 P0 不仅未达到
`P0−B0 >= +0.8 mAP` 与 `P0−M0 >= +0.4 mAP`，而且明确低于两个控制。

结合 e120 同 checkpoint 六臂：correct 与 matched/canonical/foreground/zero 的 mAP 差
分别仅 `-0.000338/-0.001719/-0.000690/+0.001795` 百分点，R1/R5/R10 全同；matched
target-write gate PASS、各反事实 descriptor 不同、correct-start/end 精确复现、zero 622 次
forward exact identity。故不是 arm 未切换、模块未训练或早期误判，而是当前 PRSM 的实例姿态
路由、部位槽归属和推理时 memory 写回均没有可测身份排序贡献。

**最终裁决：PRSM Gate A 正式 NO-GO。** 不进入 graph、动态 scan order、更多状态槽、额外
loss 或小参数变体。该结论只否定当前 pose-routed selective memory 作为自有论文创新，
不推翻历史 PSG/LGPA 的性能资产，也不外推为“所有 pose-controlled state-space 都无效”。

本地可复核小资产位于 `remote_artifacts/exp375/4090_same_runtime/`、
`remote_artifacts/exp375/3090_cross_runtime/` 与 `remote_artifacts/exp375/exp375_cf_e120_all.json`；
未回传 checkpoint。
