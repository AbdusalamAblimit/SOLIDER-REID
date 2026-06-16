# exp324d_large 监控日志 (DINOv2-large + LoRA rank16)

机器：hyy-5060ti-double GPU0 (RTX 5060Ti 16G)
脚本：`scripts/exp324d_lora.py --dino_model facebook/dinov2-large --lora_rank 16 --lora_alpha 16 --micro_bs 16 --epochs 30 --eval_period 5`
输出：`/hy-tmp/reid-clean/SOLIDER-REID/log/occluded_duke/exp324d_large`
日志：`/tmp/exp324d_large.log`

## 基线（对照）
- 冻结 DINOv2-base (exp324b)：part-MaxSim 重遮挡 mAP **8.65** / 全部 **14.61**（e20 触顶）
- Swin baseline：72.57
- base-rank16：lab-3090-d 在跑
- 平行 exp324d_r32 (base rank32)：hyy GPU1

## 环境准备 (07:50-08:00)
- rsync exp324d_lora.py / exp324b_train_head.py / exp324_dino.py from lab-3090-d → hyy
- peft 0.19.1 安装成功（pip，系统 py3.11）；transformers 5.12.0
- pool 缓存从 lab-3090-d rsync（train_n15618 / query_n2210 / gallery_n17661，363MB，geometry-only base/large 通用）
- 图像 symlink：repo data/occluded_duke/{bounding_box_train,query,bounding_box_test} → /hy-tmp/reid-clean/data/Occluded_Duke
- train pose 缺失但有 train pool 缓存 → prepare_split 走缓存不调 find_pose，OK

## 脚本改动（最小，单变量）
- build_lora_dino 加 model_name 参数，返回 (model, hidden)，从 base.config.hidden_size 读 hidden
- main 加 --dino_model；PartHead(in_dim=hidden) 用真实 hidden（large=1024）
- ROOT 改读 EXP324_ROOT 环境变量

## Dry-run 验证 (08:00)
- hidden=1024 正确检测；LoRA params=1,572,864（仅 LoRA 可训，DINO 冻结），head=1.24M
- 3 步 loss 下降（tri 7.66→5.83），梯度到 LoRA 确认
- peak GPU=5.34G (micro_bs=16)，远低于 16G，eval 有余量
- ~7.5s/step → 244 batch/epoch × 7.5s ≈ 30min/epoch × 30 ≈ 15h

## 启动 (08:05)
- nohup 后台启动 GPU0
- 加载 query/gallery pool 缓存 OK，heavy-occ queries 989/2210
- 进入 epoch 1：GPU0 5821 MiB / 100% util，进程 ALIVE

## 训练进度

### Epoch[1] (1843s, ~31min as预期)
`loss=10.20 id=5.47 tri=1.64 part=6.19 d_ap=16.62 d_an=16.69 acc=0.373 lr=1.0e-4`
- **acc 0.373**：与 r32 e1(0.412) 同档轨迹，loss 18→10.20，tri 8→1.64
- LoRA 在让 large backbone 自适应；large 每 epoch ~31min，e5 eval 约 2.5h 后
- 等 e5 part-MaxSim eval（决定性 heavy/all mAP vs 冻结 8.65/14.61）

### Epoch[2-4] 训练端收敛（与 r32 同轨迹）
- e2: loss=4.67 acc=0.811 d_ap=26.3 d_an=32.8
- e3: loss=1.62 acc=0.944 d_ap=34.7 d_an=45.3
- e4: loss=0.74 acc=0.973 d_ap=35.8 d_an=48.0

### ⚠️ e4→e5 静默死亡（疑似 eval 阶段 GPU OOM）
- e4 后进程消失，log 无 Traceback/OOM 行（OOM-killer 不留 python traceback），GPU0 归零。
- 系统 RAM 466G 空 → 不是 system-RAM OOM。判断：**eval 阶段 dinov2-large 前向全量 gallery(17661) 显存峰值超 16G**
  （训练峰值仅 5.34G，但 eval encode_split fwd_bs=32 无 checkpointing，large hidden=1024 前向激活更大）。
- 符合 CLAUDE.md 记录的 "5060 Ti 16G eval 阶段 OOM-killed" 模式。
- **修复**：relaunch large，`--eval_fwd_bs 8`（eval 前向 batch 32→8，降 eval 显存峰值）。
  不改 train batch（BS=64 硬约束）、不改 micro_bs。从头重训（无 resume），损失 ~4 epoch(~2h)。
- relaunch 已确认干净启动：GPU0 5821 MiB/100%，进入 e1。等 e5 验证 eval 不再 OOM。

### relaunch 训练进度（eval_fwd_bs 8）
- relaunch 复现原轨迹（同 seed 确定性）：e1 acc 0.373 / e2 0.811 / e3 0.944 / e4 0.973 / e5 0.983
- **✅ e5 eval 通过（OOM_FIX_OK）**：eval_fwd_bs 8 修复了 eval 阶段 OOM，进程存活继续训练。

### ★ Epoch[5] EVAL — capacity 对照（vs r32 base）★
```
cos  ALL  : mAP=46.75 R1=57.38   part ALL  : mAP=47.21 R1=59.19
cos  HEAVY: mAP=37.13 R1=45.40   part HEAVY: mAP=38.50 R1=49.34
```
| 片 | large e5 | r32(base) e5 | Δ(large−base) | 冻结 |
|----|----------|--------------|---------------|------|
| part HEAVY mAP | **38.50** | 36.72 | +1.78 | 8.65 |
| part ALL mAP | **47.21** | 44.54 | +2.67 | 14.61 |

- **关键结论：capacity 不是瓶颈。** large(hidden 1024) 仅比 base 高 +1.8 heavy / +2.7 all，
  落在**同一 ~40 heavy / ~48 all 带**，不是不同 regime。
- 叠加 r32(LoRA rank32 vs rank16) 同样 plateau ~40 → **backbone 容量、adaptation 容量都不是瓶颈**，
  瓶颈在**机制/问题结构**（pose-part-MaxSim 表征上限）。坐实"路线 2 刷绝对值无望，需机制重组/问题 reframe"。
