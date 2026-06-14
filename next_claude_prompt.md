继续 exp/pose_heatmap 分支的 Pose-Guided Person ReID 实验。

## 第一步：恢复上下文（必须按顺序完成，不得跳过）

1. 精读 `CLAUDE.md` 和 `.claude/rules/` 下所有规则文件
2. 精读 `experiments/results.md`（很长，重点看 exp244-249 部分）
3. 精读 `experiments/decisions.md`（很长，重点看末尾最新决策）
4. 精读 `experiments/innovation_brainstorm.md` 和 `experiments/innovation_brainstorm_lgpa.md`
5. 精读 `experiments/paper_materials/story.md`
6. 读最新实验的 design.md 和 monitor.md：
   - `experiments/exp244/` — Tiny LGPA-D baseline (65.3/75.7)
   - `experiments/exp245/` — Small LGPA-D (含 exp245g 和 exp245h_v2 两个重要子实验)
   - `experiments/exp246/` — Tiny LGPA-D+GCN 双分支 (65.5/77.2)
   - `experiments/exp249/` — Small LGPA-D+GCN 双分支（正在远程跑）
7. 读代码关键文件：
   - `model/pose_backbone_model.py` — 核心模型（PSG、LGPA、GCN 路径）
   - `model/modules/clip_part_head.py` — LGPA 模块
   - `model/modules/skeleton_gcn.py` — GCN 模块
   - `processor/processor.py` — 训练 loss 逻辑
   - `config/defaults.py` — 所有配置项

以上全部读完后，才能开始工作。

## 当前状态

1. **本地 3090**: 被同学用着，暂时不能用。等用户通知恢复后启动 exp249 本地版
2. **远程 5060Ti**: exp249 (Small LGPA-D+GCN) 正在跑
   - SSH: `sshpass -p 'aZKBF3qdSS59Wf4uveVQgEwWAtHAwbeg' ssh -p 29162 -o StrictHostKeyChecking=no root@i-2.gpushare.com`
   - Log: `/tmp/exp249_remote.log`
   - 训练 log: `./log/occluded_duke/exp249_small_lgpa_gcn/train_log.txt`
   - 刚启动 (~ep2), ETA 约 14h
   - **重要**: SSH 默认在 /root，必须用脚本方式执行远程命令：
     ```bash
     sshpass ... ssh ... "echo '#!/bin/bash
     cd /root/work/SOLIDER-REID
     命令' > /tmp/run.sh && chmod +x /tmp/run.sh && /tmp/run.sh"
     ```
     或者用变量方式：
     ```bash
     CMD='...' && sshpass ... ssh ... "cd /root/work/SOLIDER-REID && $CMD"
     ```
     **千万不要忘记 cd！不然 python 找不到 test.py**

## 最强结果汇总

| 方法 | Backbone | mAP | R1 | MaxSim mAP | MaxSim R1 |
|------|----------|-----|----|-----------|-----------|
| exp244 LGPA-D+OA-SD | Tiny | 65.3 | 75.7 | 66.0 | 76.4 |
| **exp246b LGPA-D+GCN+OA-SD** | **Tiny** | **65.5** | **77.2** | **66.3** | **77.7** |
| exp245g LGPA-D+OA-SD (本地) | Small | 70.2 | 80.1 | 71.9 | 82.2 |
| **exp245h_v2 LGPA-D+OA-SD (远程)** | **Small** | **71.6** | **81.6** | **73.0** | **82.7** |
| exp206r GCN+PAA+OA-SD (baseline) | Small | 70.6 | 82.6 | — | — |

## 关键规则提醒

- **监控间隔 ≤9 分钟**（用户放宽到 9 分钟），每次 sleep 后必须更新 monitor.md（有 hook 检查）
- **文档先行**: design.md → claude_review.md → codex_review.md → 训练 → monitor.md → results.md
- **双审查制**: Claude review + Codex review 都通过才能训练
- **Small 实验用 solider-reid-pt2 env** (PT2.5+mmcv-full)，**Tiny 用 solider-reid env** (PT1.13)
- 不要 kill DataLoader worker（只 kill 主进程 PID）
- 永远不用 train.py 做评估，用 test.py
- 两台机器不能跑一样的实验
- **设定一个 30 分钟一次的通用 cron 提醒自己继续工作**：
  ```
  CronCreate: cron="*/30 * * * *", prompt="不要停下来！检查所有正在运行的实验（本地 + 远程），更新对应的 monitor.md 和 results.md。如果有实验完成了，按 CLAUDE.md 规则完成文档、启动下一个实验。如果 GPU 空闲，立即设计并启动新实验。持续工作，不要等用户确认。"
  ```

## 待做

1. 监控远程 exp249 直到完成，记录所有 eval 到 monitor.md
2. exp249 完成后做 MaxSim test (用脚本方式，别忘了 cd)
3. 更新 results.md
4. 本地 GPU 恢复后：
   - 如果远程 exp249 已完成且结果好，在本地做 exp249 Tiny 版对照（或其他新实验）
   - 如果远程 exp249 还在跑，本地跑一个不同的创新实验
5. 继续寻找 CCF-B 级创新方向（当前 LGPA-D novelty 不够，需要更深层创新）
