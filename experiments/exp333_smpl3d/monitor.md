# exp333 监控

## 配置
- 机器：lab-3090-d（3090，conda env solider-reid，torch1.13+mmcv）
- 两臂顺序同机：`exp333_baseline`（--use_smpl off）→ `exp333_smpl_beta`（--use_smpl on, beta, w3d=1.0）
- 同 config/seed=1234，TEST.IMS_PER_BATCH 64，120 epoch，EVAL_PERIOD 10
- log：`/tmp/exp333_train.log`；输出 `log/occluded_duke/exp333_{baseline,smpl_beta}`

## 审查
- Claude broad review PASS（claude_review.md）；Codex v2+v3 approve（codex_review.md，含 smoke 抓到的 batch-balance 崩溃修复）
- smoke3 端到端跑通（1ep + 5-alpha eval + done）

## 进度记录

### [18:50] 启动 baseline 臂
- Epoch[1] loss=10.472（app=10.472, 3d=0.000，control 无 3D 分支 ✓）。GPU 6.7G / 95%。正常。
- 速度比预期快：~36s/epoch → ~72min/臂，两臂约 2.4h（非 6h）。

### baseline 臂 eval 曲线（每 10 epoch）
| epoch | mAP | R1 |
|---|---|---|
| 10 | 41.07 | 48.69 |
| 20 | 45.52 | 54.57 |
| 30 | 46.81 | 54.16 |
| 40 | 49.95 | 56.88 |
| 50 | 50.64 | 58.14 |
| 60 | 50.75 | 57.92 |
| 70 | 51.54 | 59.28 |
| 80 | 52.48 | 59.41 |
| 90 | 52.83 | 60.32 |
| 100 | 52.88 | 60.59 |
| 110 | 53.02 | 60.32 |
| **120** | **53.09** | **60.45** |
**baseline 最终 = 53.09 / 60.45**（与历史弱 baseline ~53.5 一致 ✓）。这是 A/B 对照基准。

### [20:05] baseline 完成 → SMPL 臂启动
- control 基准锁定：**mAP 53.09 / R1 60.45**。
- 接下来 SMPL 臂（--use_smpl, beta, w3d=1.0）训练，每 10 epoch 出 5 个 alpha 的融合 mAP。
- **判据**：SMPL 臂 best-alpha mAP vs 53.09。涨 = SMPL 外部信息有效。

### SMPL 臂 eval 曲线（mAP；每 10 epoch，5 alpha）
| epoch | a=0 | a=0.5 | a=1.0 | a=1.5 | a=2.0 | best-a |
|---|---|---|---|---|---|---|
| 10 | 41.07 | 36.40 | 32.87 | 32.76 | 32.73 | a=0 |
| 20 | 45.52 | 42.36 | 36.99 | 36.07 | 35.98 | a=0 |
| 30 | 46.81 | 44.38 | 38.88 | 37.05 | 36.73 | a=0 |
| 40 | 49.95 | 47.75 | 42.17 | 39.69 | 39.13 | a=0 |
| 50 | 50.64 | 48.63 | 43.50 | 40.59 | 39.83 | a=0 |
| 60 | 50.75 | 48.74 | 43.77 | 40.78 | 39.92 | a=0 |
（a0.5−a0 差距 e50/60 ≈ −2.0，渐近坐实。3D loss 卡 7.15 ≈ ln702 随机。）

### [21:15] smpl 臂 e60 后 kill（趋势已定：pooled 融合 null，best=a0），转排查三因
- **e55→e60 慢了 30min**：是我跑的 CPU 数据诊断与训练 8 个 DataLoader worker 抢 CPU；诊断随连接断而死、训练自行恢复。教训：训练时别在同机跑重 CPU 任务。
- jump host（Tailscale 100.95.201.91）期间抖动多次，nohup 训练不受影响。
- **kill smpl 臂**（e120 可预测=null），用 baseline e120 ckpt（53.09，348MB，完好）跑**决定性三因测试**：
  - Cause3 `exp333_data_diag.py`：β-only ReID mAP + 类内一致性（数据对不对）
  - Cause1 `exp333_rawbeta_test.py`：raw-β（valid-gated）融合冻结 baseline（绕开可疑监督头，方法对不对）
  - Cause2（代码）：本地审完，alpha=0 逐位=baseline 证外观路径正确；3D 路径无梯度/shape bug；~随机 3D-CE 与"10-d β 无法 702 类线性可分"自洽。
- 等 /tmp/exp333_decisive.log 出 DECISIVE_DONE。

### ⭐ 决定性结果（2026-06-18，raw-β 测试，绕开可疑环节）
`exp333_rawbeta_test.py`（冻结 baseline e120 + raw z-normed β + valid-gated 融合）：
- **β-only ReID mAP = 0.18% / R1 0.12%（1652 valid-q）→ ≈ 随机**。缓存 β 在 full-ReID 尺度无身份信号。
- **[ALL q]** app-only 53.09 | best α=0 **delta +0.0000**（α 越大越降：a0.25=49.84→a1=14.5）。
- **[valid-SMPL q]** app-only 55.90 | best α=0 **delta +0.0000**（a0.25 就 −4.2，β **主动伤**）。
- 数据诊断 `exp333_data_diag.py` 两次被 OOM-Killed（容器内存 + R1_mAP_eval 大矩阵），但 raw-β 的 beta-only 行已给出等价答案。

**三因定论**：
1. 代码错=否（appearance 逐位=baseline 53.09）。
2. 方法/监督头错=否（**raw β 绕开头、valid-gated、在有检测 q 上 delta 仍 +0.0000**）。
3. **SMPL 数据/提取=症结**：单图低分辨率遮挡 crop 上 ROMP 的 10-d 体型 β 被相机/姿态噪声主导、与细粒度身份无关 + 主动稀释外观。

**结论**：不是 bug，是 10-d 体型 β 这个**具体特征太弱/太脏**（区分不了 519 人）。强 ViT 外观已捕获有用信息。
**唯一公平的下一步**（用户直觉的延伸）：换更强/更干净的 3D 特征——3D 关节/肢长比、PARE/4D-Humans 强估计器、higher-res、canonical-pose 去纠缠。诚实预期：β-only=随机让"换特征即救"存疑。
（**a0.5−a0 差距 e10→40 = −4.67,−3.16,−2.43,−2.20，缩小量 +1.51,+0.73,+0.23 急剧减速 → 渐近 ~−2.0 不到 0**。3D 分支近饱和。整体 pooled 融合判：β 不涨反略伤。悬念=valid-query 子集。）
（a=0 与 baseline 同 epoch 逐位相同 ✓。**a0.5−a0 差距：e10 −4.67 → e20 −3.16 → e30 −2.43，缩小量减半（+1.51,+0.73），几何外推渐近 ~−1.3 → 整体大概率触不到 0**。pooled 融合疑似冗余/稀释 + 25% 缺失 query 拖累。valid-query 子集留 e120 分析。）

- e10：alpha=0 = baseline e10（外观同步 ✓）；alpha>0 单调变差。**预期早期现象**（3D β-MLP 从零起、e10 还是噪声）。**黄旗**：若 e120 仍 best=alpha0，则 β 在外观尚存时只稀释不互补（design 预判的失败模式）。看 e40/e80/e120 趋势。

## 判据
- headline = smpl 臂最佳 alpha 的 mAP vs baseline mAP（同 seed 同机）。
- 期望（用户强先验）：+1~3 mAP，重遮挡子集更高。
- 诚实：alpha 为 test-time 融合超参；模型贡献 = "加 3D 分支" 整体。
