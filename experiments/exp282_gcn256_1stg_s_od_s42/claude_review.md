# Claude Review — exp282_gcn256_1stg_s_od_s42

**Review round**: v1 (Phase 3-B 启动前广范围审查)
**Reviewer**: Opus 子代理 (Broad Review 制度)
**Date**: 2026-04-20

## 审查范围

覆盖 `design.md`、`configs/occluded_duke/prcv_best_small.yml`、`config/defaults.py`、`model/pose_backbone_model.py`、`model/modules/skeleton_gcn.py`,与 Phase 1 exp262 (Small Full Scaffold GCN512 + 2-stage FINAL 73.8/83.1) 的消融变量对照。重点检查 lab4090 (RTX 4090 24G) 上跑 Swin-Small + Full Scaffold 的容量裕度、CLI 双 override 语法、与 Tiny exp278 的跨 backbone 一致性。

## 变量隔离与 baseline

本 exp 是 Phase 3-B 矩阵中 Small 侧的"最精简 full-scaffold"角 (对应 Tiny 的 exp278),**同时**改两个变量:
- `POSE_GCN_HIDDEN`: 512 → 256
- `POSE_PSG_STAGES`: `[-2,-1]` → `[-1]`

与 exp278 一样,双变量改是刻意设计: 本 cell 与 exp283 (GCN256+2stg)、exp284 (GCN512+1stg)、exp262 (GCN512+2stg) 一起构成 2×2 Small 交互表。单独 vs exp262 的 Δ 反映"两变量同时最精简"的综合影响,但矩阵内三组 pair-wise 对比才是论文核心。

跨 backbone 价值:
- exp282 Δ vs exp278 Δ (都相对各自 FINAL) → "容量×stage 交互是否跨 backbone 稳定"
- Small 上 Δ 显著缩小 → "Small backbone 容量已够,pose scaffold 边际收益小"
- Small 上 Δ 放大 → "Small 更依赖完整 pose 增强"

## CLI override 语法

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_small.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp282_gcn256_1stg_s_od_s42 \
  MODEL.POSE_GCN_HIDDEN 256 \
  MODEL.POSE_PSG_STAGES "[-1]"
```

- `prcv_best_small.yml` (83 行) 默认 `POSE_GCN_HIDDEN: 512` + `POSE_PSG_STAGES: [-2, -1]` + 所有 full-scaffold 开关打开,CLI 只 override 两项 → scaffold 定义清晰
- yacs int 字段 `MODEL.POSE_GCN_HIDDEN 256` 直接,list 字段 `MODEL.POSE_PSG_STAGES "[-1]"` 带 quote,与 Tiny exp278 同语法
- OUTPUT_DIR 路径 `/hy-tmp/log/occluded_duke/...` 是 lab4090 通用约定 (已在 exp274-277 验证可写),在 lab4090 上需确认是否同 `/hy-tmp/log` 或 `/home/afr/SOLIDER-REID/log`。**需要交叉验证**: design.md 中写 `/hy-tmp/log/` (与 srvB 约定一致),但 decisions.md L3662 记录 lab4090 代码仓库在 `/home/afr/SOLIDER-REID`,log 输出路径需与 lab4090 的实际软链一致 (queue_on_ckpt.sh 会把 OUTPUT_DIR 透传)。

**审查发现 (Low)**: design.md 的 OUTPUT_DIR 假设 `/hy-tmp/log/` 存在于 lab4090。若 lab4090 未做 `/hy-tmp/log` 软链,训练会在绝对路径上创建,可能撞到磁盘分区差异。**建议**: daemon 启动前验证 `ssh lab4090 'ls /hy-tmp/log/occluded_duke/'` 存在,否则改为 `./log/occluded_duke/exp282_gcn256_1stg_s_od_s42`(与 exp277 monitor.md L5 相同相对路径风格)。

## OOM 风险

lab4090 RTX 4090 24GB 显存 + Swin-Small + Full Scaffold (LGPA + GCN + OA-SD + ParAug) + WITH_CP=True。exp262 在原 srvA (5060 Ti 16GB) 也跑通,4090 裕度大得多。GCN hidden 降半 + PSG stage 减半 → 显存进一步减少。OOM 概率 ≈ 0。

## 与 Phase 1 共享

Small 版本与 exp262 完全同除两项 CLI override。数据、augmentation、sampler、optimizer (SGD), lr 8e-4, 120 epoch, seed 42, flip-test, equal_concat, GLOBAL_LOSS_SCALE=0.5, LGPA/OA-SD/PLBOA/ParAug 全开。

## 边界检查

- `POSE_GCN_HIDDEN=256` 下 SkeletonGCNHead 构造同 Tiny 版,Linear(768, 256) → LayerNorm → Linear(256, 768) → LayerNorm,hidden 不需是 768 因子,安全
- `POSE_PSG_STAGES=[-1]` 解析为 Stage 3 (Swin-Small num_stages=4,Stage 3 共 2 个 Swin block) → 2 个 PSG 模块
- lab4090 用 `/usr/local/anaconda3/envs/mmpose-abu/bin/python` (非系统 python3),queue_on_ckpt.sh 已在 decisions.md L3769 fix 通过 `PYTHON` 环境变量传入,需验证 daemon 启动环境正确
- pose_data 在 lab4090 已由 extract_visibility.py + compute_target_assignment.py 重建完成 (decisions.md L3744-3753),visibility ULP 差 5e-5 可忽略,训练等价

## 机器分配与 auto-chain

lab4090 第 1 个 Phase 3-B slot,跟在 exp277 (Small 3-stage PSG Phase 3-A) FINAL 之后。queue_on_ckpt daemon 链: exp277 → **exp282** → exp283 → exp284。Small 单 run ~1h42min × 3 runs = ~5h,快于 Tiny (srvB 10h)。与 srvB 上的 Tiny Phase 3-B 链不冲突。

## 结论

**审查通过 (含 Low 级别提醒)**。本 exp 设计意图明确 (Phase 3-B Small 2×2 矩阵的"双变量最弱角"),CLI 语法正确,scaffold 继承干净。Low-severity 提醒:

1. **OUTPUT_DIR 路径**: 确认 lab4090 上 `/hy-tmp/log/occluded_duke/` 可写 (或已软链),否则改用相对路径 `./log/occluded_duke/`。queue_on_ckpt.sh 传 OUTPUT_DIR 时应以 lab4090 本地路径为准。
2. **daemon PYTHON 路径**: 确认 `PYTHON=/usr/local/anaconda3/envs/mmpose-abu/bin/python` 已在 daemon env 中 (decisions.md L3775 已处理)。

可 auto-chain 启动。若 OUTPUT_DIR 路径实际上与 exp277 一致(/home/afr/SOLIDER-REID/log/...),在 daemon 起命令里覆盖即可,无需修改 design.md。
