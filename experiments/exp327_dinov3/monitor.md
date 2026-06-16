# exp327 监控记录 — DINOv3 / DINOv2-with-registers pose-part-MaxSim

脚本：`scripts/exp327_dinov3.py`（training-free，frozen DINO 系，纯推理）
机器：hyy-5060ti-double GPU1（5060 Ti 16G），py3.11 + torch 2.9.1+cu128 + transformers 5.12.0
特征源（--model）：`dinov2reg-b`=facebook/dinov2-with-registers-base（默认，ungated）；`dinov3-b`=facebook/dinov3-vitb16-pretrain-lvd1689m（gated，需 token）；`dinov2-b`=facebook/dinov2-base（复现 exp324 sanity）

## 关键实现参数
- 几何自适应：输入宽=(224//patch)*patch，高=宽×2。patch14（dinov2/reg）→ 224×448 grid 32×16；patch16（dinov3）→ 224×448 grid 28×14。
- token 切片 `out[:, 1+nreg : 1+nreg+n_tok]`（布局 [CLS, registers, patches]），并 hard-assert `out.shape[1]==1+nreg+n_tok`（codex 收紧后）。nreg 从 config 读（dinov2-base=0，registers=4）。
- pose 锚定 5-part + MaxSim + 重遮挡口径与 exp324 **逐字节一致**。pose_data 同 exp326（slim npz，已在 hyy）。
- 特征不写盘（--cache 默认关），in-RAM float16。

## 双审查
- Claude Broad Review + Codex：通过（与 exp326 共享 `claude_review.md` / `codex_review.md`）。

## [smoke] dinov2-b 50 query × 2000 gallery（复现 exp324 验证 hyy 上 slim-pose pipeline）
| 方法 | ALL mAP/R1/R5/R10 | HEAVY mAP/R1/R5/R10 |
|------|-------------------|---------------------|
| (a) holistic CLS | 1.05/0.00/0.00/2.00 | 0.72/0.00/0.00/0.00 |
| (a) holistic mean-pool | 1.75/0.00/2.00/10.00 | 0.97/0.00/0.00/0.00 |
| **(b) pose part-MaxSim** | **8.19/12.00/22.00/28.00** | **2.55/0.00/8.70/13.04** |
| (c) grid part-MaxSim | 2.20/0.00/6.00/16.00 | 1.30/0.00/0.00/4.35 |

**复现核验**：ALL holistic CLS 1.05、mean 1.75、pose 8.19、grid 2.20 与 exp324 lab-3090-d smoke **数字完全一致**（exp324 monitor.md smoke 表同值）→ hyy 上 slim-pose pipeline 与 lab-3090-d 一致，**slim npz 剥 heatmap 无损**。no-pose 0/50 + 0/2000。

> 注：dinov2-b FULL sanity（复现 exp324 全量 1.86）启动后在 gallery rep-building 阶段与 DIFT 严重抢 CPU（>10min 仍未出），**因 smoke 已逐位复现 exp324 数字 + heavy-occ 989/2210 完全一致，full sanity 冗余，已 kill 以把 CPU 让给决定性的 DIFT 全量**。pipeline 一致性已由 smoke + heavy-occ count 双重确认。

## [FULL] dinov2reg-b 2210 query × 17661 gallery（2026-06-16，hyy GPU1）

heavy-occ 989/2210（与 exp324 完全一致）。no-pose 0。nreg=4 正确读取，grid 32×16。耗时 796s（feature 381s + rep building ~415s，与 DIFT 抢 CPU 偏慢）。

| 方法 | ALL mAP/R1/R5/R10 | HEAVY mAP/R1/R5/R10 |
|------|-------------------|---------------------|
| (a) holistic CLS | 0.74/1.00/3.44/5.20 | 0.58/0.61/2.33/3.64 |
| (a) holistic mean-pool | 0.88/1.09/3.98/5.79 | 0.69/0.71/2.43/3.13 |
| **(b) pose part-MaxSim** | **3.85/8.60/15.88/20.18** | **2.15/3.84/8.49/11.63** |
| (c) grid part-MaxSim | 1.04/1.67/4.57/6.47 | 0.72/0.71/2.73/3.54 |

**vs exp324 DINOv2-base（heavy pose-part 1.86/3.54）**：dinov2reg-b heavy **2.15/3.84（+0.29 mAP / +0.30 R1）**。ALL：3.85/8.60 vs exp324 3.21/7.87（+0.64/+0.73）。
**机制保持**：pose-part vs holistic CLS heavy **+1.57 mAP / +3.24 R1**；grid vs holistic 仅 +0.13 mAP（几乎无效）→ 涨点仍几乎全来自 pose 锚定（pose vs grid +1.44 mAP / +3.13 R1）。

## 结论（exp327）

- **registers（去 high-norm artifact token，更干净 dense 特征）给小幅正向**：训练-free heavy 1.86→2.15（+0.29 mAP），方向对但**幅度小，没破天花板**。
- 印证 exp324 假说：**训练-free 天花板瓶颈在"frozen"本身，不在 SSL 模型新旧/registers 干净度**。换更新 DINO 系只能蹭出 +0.3 量级，不足以独立可用。
- dinov3-b（更强）gated 下不了（hf-mirror 需 token），无法验证更激进升级；但按 registers 的小幅增益外推，预期也不会破天花板。
- **下一步**：registers 这点小增益**不值得单独上头**（vs exp324b 头已到 14）；若要上头优先用 DIFT（若 full 超 1.86 更多）。exp327 线**判定：更强冻结 DINO 源非天花板瓶颈解，止损**。
