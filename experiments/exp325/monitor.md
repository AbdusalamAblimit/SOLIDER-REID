# exp325 监控 / 结果：冻结 DINOv2-large + 轻量头 + 姿态部位匹配

## 配置

- 机器：lab-3090-d（3090，与他人 exp324d LoRA 等共享卡，~8G 已占；exp325 frozen 前向 ~6–8G）
- backbone：`facebook/dinov2-large`（hidden **1024**, patch14, 304M params，hf-mirror 下成功）
- 头/损失/采样/eval/超参：与 exp324b **完全一致**（仅 backbone 变）。embed 512，Adam lr3.5e-4 wd5e-4，cosine，id/tri/part=1/1/0.5，soft margin，seed 1234，60 epoch，PK16×4=BS64，eval_period 10。
- 独立缓存：`experiments/exp325/_cache`（large 部位特征维度 1024，不复用 base 缓存）
- 双审查：Claude broad + Codex `--search` 均 approve（仅 Low 非阻断）
- 启动：2026-06-16，`/tmp/exp325.log`

## baseline（exp324b，DINOv2-base，e60）

| 指标 | part-MaxSim ALL | part-MaxSim HEAVY | cos ALL | cos HEAVY |
|------|-----------------|-------------------|---------|-----------|
| exp324b base | 14.61 | 8.65 | 13.51 | 7.32 |

**成功口径**：part-MaxSim 重遮挡 > 8.65、全部 > 14.61？

## 进度

### 特征抽取阶段（large dense token 过一遍）
- train 15618：~13 img/s，进行中（3232/15618 @ 250s）；预计 train ~20min + query ~3min + gallery ~23min ≈ 46min 一次性缓存。
- GPU 8090 MiB / 24576，util 100%，无 OOM/Traceback。

### 训练 / eval 结果
【待填：每 10 epoch part/cos × ALL/HEAVY】

## 结论
【待填：large 天花板能否抬过 base 8.65 / 14.61？正向 → backbone 容量是有效杠杆（继续 v3/LoRA）；持平/略低 → 瓶颈在冻结+轻量头范式本身，DINO 线天花板低。】
