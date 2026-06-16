# 实验 exp324f: DINO 姿态部位对应 ⊕ exp255 Swin 融合（无训练，eval-only 分析）

> **来源**：过夜创新探索（overnight_innovation_log.md）。exp324b 已证冻结 DINO + 姿态部位 part-MaxSim 在重遮挡上有机制信号（part 重遮挡 8.65 / 全部 14.61），但**绝对分远低于 exp255 Swin SOTA（75 mAP MaxSim）**。本实验换一个**重量级角度**：不再单独评 DINO，而是问"**给 75-mAP 的 SOTA Swin 模型加上 DINO 的遮挡部位对应，重遮挡 query 能否比 Swin 单独更好？**"——建在 75 上而非 14 上。
> **性质**：**eval-only / 无训练**。两个 distmat 都是已训好模型（exp255 Swin ckpt + exp324b head_60）出的，融合是纯 numpy 后处理。不开新训练，但有新脚本 → 走 Claude review（Codex 可选，因无训练不触发 hook 阻断）。
> **机器**：lab-3090-d（exp255 ckpt、exp324b head_60、query/gallery 部位缓存全在本机；3090 idle）。原计划 lab-4090，但 4090 venv 缺 cv2/mmengine/transformers，落到 lab-3090-d。
> **环境约束（关键）**：lab-3090-d 上**没有一个 python 同时具备 mmengine（跑 SOLIDER swin backbone）和 transformers（DINO）**。conda `solider-reid`（torch1.13）有 mmengine 但无 transformers；系统 python3（torch2.7）有 transformers 但无 mmengine。故 exp324f **拆两阶段**，用 npz 桥接：
> - **Stage 1** `scripts/exp324f_swin_distmat.py`（`solider-reid` env）：跑 exp255 Swin，dump `d_swin` + q/g 文件名/pid/camid 到 `experiments/exp324f/swin_distmat.npz`。
> - **Stage 2** `scripts/exp324f_fuse.py`（系统 python3）：load npz + exp324b 缓存部位特征 + head_60 出 `d_dino`，按文件名对齐、归一化、扫 w 融合、eval。DINO 侧只用缓存（无 DINO forward），纯 torch。

## 动机

- exp255 Swin 是项目训练端最强（75.2 mAP MaxSim，Occ-Duke）。它已用 PSG/LGPA/GCN 处理遮挡，但**重遮挡 query 仍是最难的子集**。
- DINO 姿态部位对应（exp324b）提供一种**正交的、自监督的、姿态显式锚定的**遮挡鲁棒匹配信号：跨图只比双方可见部位（mutually-visible part-MaxSim），天然对遮挡部位免疫。
- 若把两者的距离矩阵融合后，**重遮挡子集**比 Swin 单独更好，则证明"DINO 对应给 SOTA 模型补遮挡鲁棒性"是有用的创新点（互补信号，而非冗余）。

## 核心假设

DINO 姿态部位对应的距离矩阵与 Swin MaxSim 距离矩阵**互补**：在重遮挡 query 子集上，加权融合 `d = (1-w)·d_swin + w·d_dino`（w>0）的 mAP/R1 高于 w=0（Swin 单独）。

## 技术方案

### 数据流（两个 distmat → 归一化 → 融合 → eval）

1. **Swin MaxSim distmat**（`d_swin`，Q×G）：
   - 用 `scripts/eval_fliptest_maxsim.py` 的特征提取 + `compute_maxsim_distmat`（global_weight=1.0，与主线 MaxSim 一致），exp255 ckpt = `log/occluded_duke/exp255_small_gcn512_2stage/transformer_120.pth`，config = `configs/occluded_duke/pose_psg_lgpa_gcn512_2stage_small.yml`。
   - flip-test ON（与 75.2 主线口径一致）。新脚本 `scripts/exp324f_fuse.py` 复用其函数，**额外 dump `d_swin` npy + q/g pids/camids + query 可见关键点数**（heavy mask 用）。
2. **DINO part-MaxSim distmat**（`d_dino`，Q×G）：
   - 复用 exp324b 缓存的 query/gallery 部位特征 npz（`experiments/exp324b/_cache_train/{query,gallery}_parts_224x448_n*.npz`）+ `head_60.pth`，用 exp324b 的 `PartHead.encode_parts` + `exp324_dino.part_maxsim_distmat` 出 `d_dino`。**不重抽 DINO 特征**（缓存已在）。
   - **对齐**：Swin 与 DINO 两条流的 query/gallery 顺序必须按文件名对齐。两边都 `sorted(os.listdir)` 取 jpg → 同序。脚本显式按文件名做 index 对齐（用 pid+camid+name 校验），不可靠时报错退出。
3. **归一化**：两个 distmat 各自做 **z-score**（按全矩阵 mean/std）和 **min-max**（按全矩阵 min/max）两种，分别 sweep，取更优的报告。z-score 默认主报。
4. **融合**：`d = (1-w)·d_swin_norm + w·d_dino_norm`，sweep `w ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5}`。
5. **eval**：`utils.metrics.eval_func`（同 cam 排除），mAP/R1/R5/R10。**全量 query** + **重遮挡子集**（query 关键点可见数 ≤ HEAVY_OCC_THR=8，与 exp324b 同口径，用 DINO 侧 pose 的 visibility_binary.sum()）。

### 关键正确性点（审查重点）

- **query/gallery 对齐**：Swin 流（make_dataloader，可能 shuffle=False 但顺序由 dataset 决定）vs DINO 流（sorted listdir）。必须用文件名做显式 join；pid/camid 不一致即报错。
- **heavy mask 一致性**：用同一套 query pose visibility（DINO 侧 find_pose），保证 8.65/14.61 baseline 口径可比。
- **w=0 必须等于纯 Swin**：sanity check，w=0 时 mAP 应 ≈ eval_fliptest_maxsim 报的 Swin MaxSim 数字（小数点级一致），否则对齐/归一化有 bug。
- **归一化不能跨 query 行泄漏**：z-score/min-max 按全矩阵统计（两个 distmat 各自独立统计），不按行——按行会破坏 MaxSim 的相对排序语义。这里只为把两个不同尺度的 distmat 拉到可加。

## 预期结果

- 假设成立：重遮挡子集存在 w*>0 使融合 mAP > Swin 单独（w=0）。涨幅哪怕 +0.5~+2 mAP（重遮挡）即算正向信号 = DINO 对应补 SOTA 遮挡鲁棒性有用。
- 全量 query 上预期持平或微涨/微跌（DINO 分低，全量上 Swin 主导）。
- 失败最可能：DINO distmat 噪声太大（14 mAP 量级），融合在任何 w>0 都拖垮 Swin → 说明冻结 DINO 信号不足以补 SOTA，这条融合线止损。

## 对照组

- **baseline**：w=0（exp255 Swin 单独 MaxSim+flip），全量 + 重遮挡。
- **消融变量**：融合权重 w（唯一扫的变量）；归一化方式（z-score vs min-max）作稳健性对照。
