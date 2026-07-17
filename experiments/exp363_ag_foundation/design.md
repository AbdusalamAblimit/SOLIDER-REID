# exp363 — Aerial-Ground / RGB-IR Video Foundation Adaptation（跳出盒子第一个 build）

## 选定经过（2026-06-27，用户点醒"别自我设限 occluded ReID+SOLIDER"后）

- 换量级 deep work 在 occluded ReID 内部多 build 全证伪接近墙（[[paradigm-shift-occluded-reid-wall]]）。用户点醒：没限定我在 occluded ReID+SOLIDER。
- codex 全 ReID 范式级 gap analysis（`paradigm_shift/codex_full_reid_gap.md`）：判"别再押 occluded ReID/SOLIDER/pose-part"，选 **Aerial-Ground / RGB-IR Video Foundation Adaptation**（换问题+换 backbone+换监督）。
- codex AG 核查（`paradigm_shift/codex_ag_verify.md`）：避 CARGO 死区确认，数据可及，cheap kill-switch 清晰，**6.5-7/10**，值 1-2 天 kill-switch 不直接开大工程。

## 真正跳出盒子（vs 之前自我设限）
- **换问题**：occluded ReID → aerial-ground / video / RGB-IR（AG-VPReID.VIR）
- **换 backbone**：SOLIDER → DINOv3 ViT-B/16（foundation，非 SOLIDER）
- **换监督**：identity CE/triplet → 跨视角/尺度/模态/时间 multi-axis consistency + foundation-teacher anchoring

## 核心 idea（novelty 窄缝，codex）
> 极端视角/模态/时间证据缺失下，低秩适配（LoRA）只学**跨帧/跨模态可验证的 residual identity evidence**，同时用**冻结 foundation teacher 的关系结构锚定 identity prior**，防 direct fine-tune 把 foundation prior 覆盖成数据集偏置。

**避死区**（codex + 项目历史）：不做 uncertainty containment（CARGO σ 前提错已死）/ 不做 avg-vs-MaxSim late-interaction（已死）/ 不做 view-aware semantic experts（撞 ViSA/SD-ReID/GSAlign）/ 不做几何 alignment / 不生成 view-specific feature / 不靠 SD / 不把 CLIP prompt 当主贡献。

## 数据（codex 核查 + 下载报告 codex_download_cmds.md）
**AG-VPReID.VIR**：GitHub `agvpreid25/AG-VPReID.VIR`（**只 assets+README，无代码** → dataloader 自己写）+ Google Drive folder `1Iy814PqWjwIZcv6CZpieFju-Dop9Y2G7`（gdown --folder）。**目录结构 README 未公开**（下载后 find 看真实结构）。
- **★规模修正（重要）**：train **326 ID / 978 tracklets / 24793 frames**（偏小！之前误写 1837 ID/124855 frames 是全集含 test）。小数据 + frozen foundation + LoRA 适合，但 kill-switch 要注意样本量（hard bucket 可能样本少，统计噪声）。
- test 协议：G→G / A→A / G→A / A→G，各 V2I+I2V；源 = UAV RGB + UAV IR/NIR + CCTV RGB + CCTV IR/NIR + wearable RGB。
（备：AG-VPReID 全集 9.6M frames；CARGO 已踩过；LAGPeR 申请制；G2APS 不可直接释放。）

## 第一步 cheap kill-switch（1-2 天，codex）
1. 下载 AG-VPReID.VIR。foundation = **DINOv2-reg-B 先用**（DINOv3 ViT-B/16 是 gated 需 HF 登录同意 dinov3-license；**DINOv2-reg-B `vit_base_patch14_reg4_dinov2.lvd142m` Apache-2.0 无 gate，timm 1.0.25 可加载，torch 1.13.1 兼容**）。DINOv3 等 HF token 到位再换对比。CLIP-L 只做 baseline（X-TFCLIP 贴脸）。
   - 环境：4090/3090 均 torch 1.13.1 + timm 1.0.25（缺 peft/gdown/transformers，已 pip install）。
2. frozen frame encoder，每 tracklet 采样 N=1/4/8 帧，frame embedding L2 norm。
3. temporal pooling 诊断：mean / quality top-k mean / score max（**不上 attention**）。
4. **4 baseline**：single center/random frame / frozen mean pooling / direct LoRA fine-tune CE+Triplet / direct fine-tune without anchor。
5. **method**：LoRA rank 8/16（qkv/proj/MLP adapter）+ frozen-teacher relational anchoring + tracklet/modality consistency。

**硬判定线（任一不过立即杀，不补 LoRA rank/attention pooling/view gate 小变体）**：
- frozen temporal mean vs single frame，hard bucket ≥ **+5 mAP/R1**（视频证据积累成立）
- oracle/top-k vs mean ≥ **+3 mAP**（选择/校准有空间）
- anchored-LoRA vs direct-LoRA/direct fine-tune ≥ **+2 overall 或 +3 hard bucket**（只赢 frozen 不算）
- 增益必须集中高海拔 A↔G / RGB↔IR / 短低清 tracklet（全桶平均涨=普通 adapter，杀）

## 风险
- 数据存储/IO（视频帧大）；DINOv3 权重下载/许可；又撞 AG-VPReID-Net/TCC-VPReID/X-TFCLIP（leading method 已打 temporal/CLIP）—— 靠 anchored residual-evidence 窄缝区别。
- backbone 训练（LoRA fine-tune）= 启动前 codex 三审 diff[[pre-experiment-review-discipline]]。

关联：`paradigm_shift/codex_full_reid_gap.md` + `codex_ag_verify.md`，memory [[paradigm-shift-occluded-reid-wall]] [[aerial-ground-containment-bet]]（CARGO 死区，本方向避开）。
