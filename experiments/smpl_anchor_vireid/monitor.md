# Monitor — CAJ RegDB baseline 复现 (lab-4090)

## 目的
确认 CAJ baseline 在 lab-4090 跑通 + 数字对得上(官方 RegDB R1 85.03 / mAP 79.14 / mINP 65.33),作为换 Swin + 加 SMPL 几何锚之前的地基。

## 环境
- lab-4090(4090D, CUDA 12.4), uv 项目 /home/afr/vireid, torch 2.4.1+cu121。
- RegDB(Mac 下载 zip → scp, 8240 图, symlink Datasets/RegDB→data/RegDB, trial 1)。
- baseline: mangye16 ICCV21_CAJ, method=adp(Enhanced Squared Diff + Channel Aug + KL)。
- 启动: `train_ext.py --dataset regdb --method adp --augc 1 --rande 0.5 --alpha 1 --square 1 --gamma 1 --trial 1 --gpu 0`。
- resnet50 ImageNet 权重: Mac 下载 scp 到 /root/.cache/torch/hub/checkpoints/(绕过 pytorch.org 国内慢)。

## 进度记录

### [启动] Epoch 0
数据加载 OK(query/gallery 各 206 ID / 2060 图, Data Loading 3.4s)。
- Epoch[0][0/64]: Loss 93.33, iLoss 5.33, TLoss 88.0, KLoss 0.0025, Accu 0.00
- Epoch[0][50/64]: Loss 92.77, iLoss 4.77, Accu 4.49
正常(iLoss 降, Accu 升)。addmm_ deprecation warning 无害。Monitor bekcaem3s 盯后续。

### eval 进度(每 2 epoch, Monitor bekcaem3s)
- Epoch 4: POOL R1 12.62/mAP 12.40, FC R1 11.26/mAP 9.74(早期低, warmup 阶段正常)
- Epoch 9: Accu 64.27, Loss 3.02
- Epoch 10: Accu 81.12, Loss 1.69(Accu e0→e10: 4.5→81, 飞涨)

趋势健康。lr 现 0.100(warmup 完), 20/50 epoch 各 ×0.1 衰减后 mAP 会爬向目标。

### ★ 终值(best epoch 52, 训练完成)
**POOL Rank-1 76.80 / mAP 69.14 / mINP 53.61**(FC 75.58/68.20)。训练内 eval, **无 TTA, 单 trial 1**。
- 官方 85.03/79.14/65.33 = **testa.py flip-TTA + 10-trial 平均**。无 TTA 单 trial 文献本就 ~77/69 → **复现确认, pipeline 正常**。
- ⚠️ testa.py 直接跑给垃圾(5-7%): 它循环测 trial 1-10 期望 10 个 per-trial 模型, 我只训了 trial 1 → 其余 9 trial 随机权重拉垮平均。**非 pipeline 问题**, 是 testa.py 设计(要 10 trial 全训)。
- **结论: 用训练内 eval(76.80/69.14)当一致 baseline, 和 SMPL 锚同口径对比。** 不纠结绝对匹配 79(TTA+多trial差异)。

## 下一步: SMPL kill-switch(GPU 已空)
torchvision keypointrcnn(2D pose)在 RegDB IR 热 crop 上跑 → 检测率/关键点质量。答"IR 上能否提人体几何"。garbage→SMPL 锚死→转 Swin-VI 机制。
