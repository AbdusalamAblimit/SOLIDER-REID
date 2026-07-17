# exp366 Active Evidence Acquisition ReID（范式级方向 #1，2026-06-28）

## 动机

用户指令"放下 LM-ReID，找新 ReID 范式级创新"。codex 范式级调研 #1（7/10，最值得）：传统 ReID 给一张 query 就必须排序；**范式重定义=系统可花 1-3 次预算主动获取下一条视觉证据**（请求另一帧/另一 camera 视角/操作员二值 VQ）。先例 LLaVA-ReID/ChatReID/Inter-ReID 是**文本对话补全**，主动获取**视觉证据**（camera-view evidence acquisition）是空白；旧 human-in-loop 偏人工标注反馈，不是主动传感/证据预算。避所有探死方向（occluded/AG/DG/gallery/open-set/Wildlife/VI-ReID/lattice/SMPL/FM/test-time）。

## 核心假设

ReID 真实场景=多相机网络：query 在 camera A，系统可主动调 camera B 获取同人证据（预算受限）。**难 query 值得花预算获取第二证据，简单 query 不值；policy（预算分配给哪些 query）是真问题**。

## cheap kill-switch（零训练，cvpb_active_evidence_probe.py，frozen SOLIDER）

- baseline：single query mAP
- oracle-all：每 query + 同 ID 不同 camera 第二证据（multi-query mean）→ upper-bound
- **★policy**：只对 hard query（top1-top2 margin 小=不确定）花预算 20% 获取第二证据
- random：随机 20%（同 has_second 池公平对照）

**判定 GO**：policy gain / oracle-all gain ≥ 0.5 且 policy−random > 0.3 → 主动获取证据 policy 有真价值。
**DEAD**：policy ≈ random → trivial multi-query 无 policy 价值。

★**诚实设计**：避 codex 的 trivial oracle（multi-query 必涨 = upper-bound 不是创新），真验 policy（预算分配 vs random）。控 margin（top1-top2 = #false-in-topk 的代理）。自查抓到 2 个 bug（margins 长度 != len(qf) 退化 policy；policy hard 应只在 has_second 池选）已 fix。

## 预期

- GO → 设计轻量训练端 active-acquisition policy（学"选哪个 query/候选获取证据"），范式级第二 contribution。
- DEAD → 主动获取证据无 policy 价值（等 trivial multi-query），转 Generative Index（codex #2，6.5/10 真空白）。

## 状态

probe 跑中（3090，b984dv1y8，frozen SOLIDER exp260b 抽 Market query/gallery 特征 + camera split + oracle/policy/random）。
