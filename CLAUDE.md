# Pose-Guided Person ReID — CLAUDE.md

## Compact / 新会话接手协议（最高优先级）

> **本节覆盖下文所有旧内容。接手第一步不是开实验，而是恢复上下文。**

必须按顺序读完以下文档，然后才可以启动任何实验或代码改动：
1. `experiments/prcv_2026_psg/README.md` → `story.md` → `experiment_route.md` → `decisions.md` → `kpr_comparison.md`
2. `experiments/paper_notes/2026-04-15_prcv_reset.md`
3. `experiments/results.md`
4. `experiments/decisions.md`（完整历史）
5. 最新 `experiments/exp{NNN}/design.md` + `monitor.md`

## 当前阶段（范式转向 deep work，2026-06-26 起 — 最高优先级，覆盖下方旧阶段）

- **用户授权重大转向（2026-06-26 深夜）**：cheap 小修小补训练端已 **100% 实测穷尽**（非 ReID 没创新，是 cheap+不换量级约束下到底）→ 用户令"全面换量级(自己训新预训练范式/上新大规模监督), 不要 cheap, 可花时间, 任何范式级创新, 有必要就改 claude.md 和 rules"。
- **模式切换**：从"cheap kill-switch + 心跳收敛" → **"paradigm deep work"**。不再 cheap-优先、不再连负止损收敛；允许多日训练 / from-ckpt continued-pretrain / 造数据 / 失败重来。GPU 空=排下个训练阶段，不是"诚实停"。
- **★分工铁律（用户明确指令）**：**调研（novelty 查 / gap analysis / 文献）+ 审查（代码 review / diff / train-test 对称 / AMP）全部交给 codex**（GPT 联网 token 无限，适配）；**Claude 专注 build——写代码 / 跑训练 / 造数据 / debug / 迭代 / 决策 / 写文档**，不自己埋头读论文做调研。
- **当前方向**：`exp360 Intruder Identity Suppression`（遮挡 ReID 根问题重定义=donor 行人身份污染 target embedding，训练端对抗 source separation，测试单图）。design `experiments/exp360_intruder/design.md`，3 路 gap+终局对比 `experiments/paradigm_shift/`。codex 选它(7.0) 否 T-SCD(5.0 撞项目 fgeu tracklet 16.3% + MVI²P 先例)。
- **算力现实**：4 单卡 slot（4090/3090/5060Ti×2），无 from-scratch foundation 算力。够得着：生成数据引擎 / 新自监督 pretext / 新监督信号。
- **仍守**：动手前 codex 查 novelty 避先例；full fine-tune/backbone/pretrain 训练前 codex 审 diff；文档先行；诚实报告；死区避（SMPL 几何无独特信号 / FM-import MLLM-DINO-SD / test-time trick 当主创新）。

## 当前阶段（PRCV 2026 已投 → 探索新创新点）

- **状态（2026-06-14）**：PRCV 2026 **已投出**。论文（PSG 主线）快照冻结在 `experiments/prcv_2026_psg/`，**不再赶 deadline、不再改投稿稿**。
- **下一阶段目标**：**探索新的创新点**（post-PRCV）。候选方向见 `.claude/rules/innovation_and_decisions.md`「值得推进的方向」与 `experiments/research_directions.md`；走那里的「默认路线」做 gap analysis → 新机制设计，**不在已投的 PSG branch 上堆小模块**。
- **以下为已投论文的固化内容**（作为新探索的 baseline / 起点，不是当前主攻）：
- **主创新**：`PSG`（Pose Spatial Gate），在 backbone 中间 stage 做 pose-guided spatial gating
- **最终实现**：`2-stage PSG`（作为 instantiation，不单独抬成主术语）
- **结构补充**：`GCN` = structural pose branch（不与 PSG 并列主创新）
- **系统资产**：`LGPA-D` / `OA-SD` / `PLBOA` / `MaxSim` / `POT` / `flip-test`（不抢主位）
- **主 benchmark**：Occluded-Duke；**补充 benchmark**：Occluded-PoseTrack-ReID（vs KPR w/o prompt）
- **跨域评测**：Market 训练 → Occluded-ReID 测试

### PRCV 实验矩阵（已完成，随投稿归档）

PSG stage 消融 / 结构分支依赖性（GCN×stage）/ 语义分支依赖性（LGPA）/ Occ-PTrack 强配置均已跑完，结果在 `experiments/results.md`（exp261-290 段）与 `experiments/prcv_2026_psg/`。**新阶段不再以此矩阵为目标。**

### 当前最强训练端配置

`exp255`：Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA = **73.2 / 83.3**（equal_concat），+ MaxSim + flip = 75.2 / 85.6。3-seed mean 72.97 / 82.80。

## 禁止回退的方向

1. visibility 小改动 / 小融合 / 小 head
2. retrieval-side scorer / gate / context 微变体
3. feature-level completion 小残差 / 小 bank / 小 gate
4. skeleton attention bias / SASA
5. symmetry aggregation / SCFA
6. 单纯 test-time 小涨点
7. 临时切全新问题定义（留作 discussion / future work）

## 持续执行铁律（强化版）

只要用户没明确说停，**必须持续工作**。默认循环：实验启动 → 用 Monitor 流事件监控 → 完成后立即补文档 → 更新 results/decisions → 下一个实验。

**禁止停止理由**："差不多了"、"等用户确认"、"转论文整理"、"汇报计划"、"实验太多先喘口气"。

GPU 空闲时**必须立刻**开下一个实验或补文档 / 审查，不允许 sleep 等待。

当你想"停下来"时，问自己：
1. 是否有服务器空闲？有就立刻排下一个实验
2. 是否有文档未补？补
3. 是否有 paper 素材可做（消融表、可视化、figure）？做

## 新工具使用铁律（2026-04 起）

这些工具替代了过去手动 sleep 轮询的低效方式：

- **`Monitor`**：流事件监控训练 log / 远程日志。`grep --line-buffered` 过滤关键事件（epoch marker、Traceback、OOM、mAP），每行一条通知。**替代 `sleep + Read log`**。
- **`Bash(run_in_background=true)`**：启动长任务（训练、大 rsync）后立刻返回，完成时收到通知。
- **`until` 循环**：`until <check>; do sleep 60; done`，放 Monitor 中轮询条件。
- **`CronCreate`**：每 N 分钟触发 prompt，保持上下文缓存 warm（已预设 30min 心跳 cron）。
- **`ScheduleWakeup`**：`/loop` 动态模式自排期。5 分钟以下保持 cache 热，不要选 300s。
- **`TaskCreate` + `TaskUpdate`**：tracking 实验队列、数据传输、审查进度。状态流转 pending→in_progress→completed。

**不要 sleep 轮询 log 文件**。用 `Monitor` 或后台 Bash + 通知。

## 创新门槛（维持）

新方向必须满足以下至少两条：
1. 问题层面有新意（重新定义）
2. 机制层面有新意（过去工作没写出）
3. 证据层面讲得清（可消融）

## 硬性约束

- **Backbone**：主用 Swin-Small（exp255 主线）；Base 可选；Tiny 用于快速消融
- **Batch Size**：不允许修改（BS=64）
- **不用计划模式**，大改直接改
- **不做组合实验逃避创新**
- **Commit**：实验开始前一次，结束后一次；重要 milestone 单独一次
- **语言**：md 文档中文，代码注释英文
- **评测**：永远用 `test.py`，永远不用 `train.py` 评估（会覆盖 train log）

## 资源基础设施（2026-06 最新）

### 3 台机器 / 4 个 GPU slot

本地 Mac 只做编排开发，无训练 GPU。训练全在远程，细节见 `.claude/rules/remote_server.md`。

| 别名 | GPU | slot | 状态 | 用途 |
|------|-----|------|------|------|
| `lab-3090-d` | RTX 3090 24G | 1 | ⚠️ 探测时不可达（ProxyJump `lab-3090`） | 原"本地 3090"，现为 lab 3090 docker 容器 |
| `lab-4090` | RTX 4090 24G | 1 | ⚠️ 探测时不可达（ProxyJump `relay4090`） | 历史跑过 exp285b |
| `hyy`(=`hyy-5060ti-double`) | 5060 Ti 16G ×2 | 2 | ✅ 在线，py3.11.12 + torch 2.9.1+cu128 | 双卡，同时跑 2 实验 |

三台全用 `ssh <别名>`（密钥已在 `~/.ssh/config`），**不要用 sshpass**。`lab-3090-d`/`lab-4090` 的磁盘/路径首次连通后补到 rules。

### 远程磁盘策略

- hyy（gpushare）：`/`(overlay) 仅 30G，数据/ckpt 放 `/hy-tmp`。仓库 `/hy-tmp/reid-clean/SOLIDER-REID`，数据 `/hy-tmp/reid-clean/data/`，输出 `<repo>/log/`。
- lab-3090-d / lab-4090（实验室 docker）：路径待确认，连上后 `find / -maxdepth 3 -name train.py -path '*SOLIDER*'` 定位。
- **训练输出统一到 `<repo>/log/`**，数据放各机数据目录。

### 数据同步

- **OSS 暂不可用**（`oss login` 账号未注册 / ent cert 过期，仅 hyy 这类 gpushare 机器相关）。需要时求用户提供可用凭证。
- 用 `rsync -az --partial <srv>:/path <srv>:/path` over SSH。带宽约 2 MB/s，大传输必须后台 + Monitor 跟踪。
- 拉日志回本地：`rsync -az --partial <srv>:<repo>/log/ ./log_remote_<srv>_backup/`（gitignore 外）。

### 4 个数据集

| 数据集 | 本地路径 | 用途 |
|--------|---------|------|
| Occluded-Duke | `data/occluded_duke` | 主 benchmark |
| Market-1501 | `data/market1501` | 训练后→Occluded-ReID 跨域 |
| Occluded-ReID | `data/occluded_reid` | 跨域评测 target |
| Occluded-PoseTrack-ReID | `data/occluded_posetrack_reid` | 补充 benchmark (vs KPR) |

## 文档先行（铁律）

每个实验开始前**必须**先写 `experiments/exp{NNN}/design.md`。
每次查看 log **必须**更新 `monitor.md`（hook 强制）。
实验结束后按序：
1. `monitor.md` 追加最终结果
2. `results.md` 更新总表（从 log 精确复制，不凭记忆）
3. `decisions.md` 记录决策
4. `innovation_brainstorm.md` 如有新想法

## 详细规则

见 `.claude/rules/`：
- `experiment_protocol.md` — 实验命名 / design.md 格式 / 双审查流程
- `monitoring.md` — Monitor 用法 + 旧 sleep 规则（仅作 fallback）
- `documentation.md` — 文档结构与数据一致性
- `innovation_and_decisions.md` — 决策 + 红蓝队辩论
- `paper_materials.md` — story 更新、table 管理
- `remote_server.md` — 3 台远程详细配置 + rsync + hy-tmp
