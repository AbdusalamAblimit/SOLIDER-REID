# exp409 PCHM 执行协议

## 冻结边界

- official RGB 只读：`/mnt1/afrdata`；pose 只读：`/mnt1/afrderived`；
- 远端只写 `/home/afr`，本机只写当前仓库；
- Python/CUDA 固定：`/usr/local/anaconda3/envs/mmpose-abu/bin/python`；
- Swin-Tiny、batch64、seed1234、120 epoch、D0 optimizer/schedule/augmentation/loss weight 全部不变；
- 新 config 开关默认关闭；eval 路径不得读取 PCHM cache、CLIP 或外部 pose；
- fresh repo、fresh cache asset、fresh output、fresh runner，不读取 exp408 cache/output，不续训；
- 任一时刻只允许一个 4090 compute task。

## 阶段 A：实现前冻结

1. 文献/代码审计确认没有发现 PCHM 整体同构实现，并收紧到 C 类候选；
2. 固定五槽 visibility、ordinal Borda、tie-break、correct/controls 和 D0 raw 双门；
3. 不允许在看到训练结果后改变 pair score、rank fusion 或候选集合。

## 阶段 B：必要检查

只执行一次紧凑检查链：

1. 语法/import；
2. synthetic 正反 shape/index contract：same-PID/non-self positive、different-PID negative、无 NaN/Inf；
3. default-off 与原 `TripletLoss` loss/distances/gradient exact；
4. fresh cache 小样本 roundtrip、SHA、路径唯一性及 wrong/generic/zero control active；
5. 真实 PK batch64 的 selected-pair snapshot：correct 与 D0、pose-shuffle、CLIP-only 均非等价；
6. MMPOSE-ABU batch64 CUDA/AMP 一次真实 backward/optimizer step，梯度 finite/nonzero；
7. 一名未参与实现的智能体盲审 design/config/diff/tests，修复全部 BLOCKER/HIGH 至 `0B/0H`。

除修复 BLOCKER/HIGH 外不展开无穷 static/CPU 合同；`0B/0H` 后立即进入 fresh cache/full run。

## 阶段 C：fresh cache

- 单独 exp409 asset 路径；
- official train 15,618 图完整、唯一、无遗漏覆盖；
- 每项为五槽 finite、非零、L2-normalized frozen CLIP region-isolated visual descriptor及validity；
- 写入 schema、relative path、逐图 image SHA、feature、CLIP checkpoint/pose manifest/preprocess/source HEAD/
  builder/teacher source provenance；loader必须逐项验证；
- 发布前验证文件 SHA，config 硬绑定该 SHA；失败目录冻结，新编号/新路径修复。

## 阶段 D：唯一 student

- 启动前 `nvidia-smi` 确认无 compute PID；
- fresh seed1234/e120，自然完成，不按中间性能早停；
- e10/20/.../120 记录 PCHM 与 sealed clean D0 同 epoch mAP/R1/差值；
- 每个记录点同时报告 selected positive/negative 的 pose/CLIP统计、相对 D0 index change rate、异常计数；
- 运行中不得修改源码/config/cache/参数。

## 最终裁决

- performance GO：e120 raw mAP `>57.5587756578` 且 R1 `>67.6923076923`；
- validity 还需 source/config/cache/checkpoint SHA、唯一运行、异常0、GPU释放、eval零cache/CLIP/pose访问；
- GO 后才按相同训练预算串行补 pose-shuffle 与 CLIP-only；
- FAIL 即封板 exp409，更新 monitor/results/decisions/innovation/story，转下一训练/结构对象，不调旧臂。
