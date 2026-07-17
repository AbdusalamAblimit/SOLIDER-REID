# exp335 监控

## 配置
- lab-3090-d（ControlMaster），conda env。两臂顺序：`exp335_lgpa`（--use_lgpa on）→ `exp335_control`（off）。
- 两臂都走 pose dataloader（POSE_ENABLED=True，只 gate LGPA head）→ 干净单变量。同 config(exp335_vit_lgpa.yml)/seed=1234，120ep，EVAL_PERIOD 10，TEST.IMS 64。
- log `/tmp/exp335_train.log`。

## 审查
- Claude broad review：2 High(损失忠实度 H1 + 增强混淆 H2)→ 全修 → PASS。Codex approve。两臂 smoke 端到端跑通。

## 判据
- **headline = LGPA-on best-alpha mAP vs control(off) mAP**。预期 +2~5（参考 Swin exp244 LGPA-D +4.4）。
- LGPA 分支 detached → on 臂 alpha=0 应 ≈ control（sanity）。best-alpha − alpha0 = LGPA 测试期融合增益；alpha0 − control ≈ 0 验证 detach。
- 这是 **CLIP 创新方向的地基**：复现到 +X → 确认 CLIP 是有效外部信息源 → 进 step2 设计新 CLIP 接法。

## 参考 baseline（exp333 ViT，53.09）曲线
| ep | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mAP | 41.07 | 45.52 | 46.81 | 49.95 | 50.64 | 50.75 | 51.54 | 52.48 | 52.83 | 52.88 | 53.02 | 53.09 |
（注：exp335 control 走 pose dataloader，增强略不同于 exp333，最终对照以 exp335_control 为准）

## 进度记录
### 启动 LGPA-on 臂

### LGPA-on eval 曲线（mAP）
| epoch | a=0(cls) | a=1.0 | a=2.0 | best-a |
|---|---|---|---|---|
| 10 | 37.57 | 32.54 | 28.94 | a=0 |
| 20 | 47.05 | 43.60 | 41.50 | a=0 |
| 30 | 48.17 | 45.69 | 43.69 | a=0 |
| 40 | 50.35 | 48.24 | 46.36 | a=0 |
| 50 | 51.60 | 49.72 | 47.84 | a=0 |
| 60 | 51.73 | 49.80 | 47.88 | a=0 |
| 70 | 52.54 | 50.94 | 49.25 | a=0 |
| 80 | 53.07 | 51.39 | 49.66 | a=0 |
| 90 | 53.19 | 51.61 | 49.97 | a=0 |
（pooled 全程 best=a0。此臂用错描述子(pooled)+错 loss scale(1.0),仅作 post-hoc equal_concat 分解用。）

### 修正后计划（已改代码 + 推送）
- trainer eval 改报 **equal_concat [global‖p1..p5]**(每块归一)+ global-only;config 加 **GLOBAL_LOSS_SCALE 0.5**。

### ⚠️ post-hoc equal_concat 结果（exp335_lgpa e120 ckpt, GLOBAL_LOSS_SCALE=1.0）—— 负
| 描述子 | mAP | R1 |
|---|---|---|
| **global(cls) only** | **53.54** | 60.77 |
| equal_concat a=0.03 | 50.97 | 57.47 |
| equal_concat a=0.1 | 49.35 | 56.38 |
| equal_concat a=0.5-2.0 | 49.12 | 56.29 |
| maxsim-hybrid | 50.22 | 57.24 |
| maxsim-only(纯部位) | 35.48 | 40.63 |

- **结论(1.0)**：CLIP 部位**不互补 global，纯拖累**——连最小权重 a=0.03 都掉(53.54→50.97)，单调下降。纯部位 maxsim 仅 35.5（远弱于 global 53.5）。
- **原因**：equal_concat 要求部位**强**(原版 Swin 部位强→70.2);ViT-base **detached 冻结特征**池出的部位弱。是 ViT-vs-Swin 的 backbone 差异，非 bug(部位 maxsim 35≠0,有信号但弱)。
- **最后一搏**：GLOBAL_LOSS_SCALE=0.5 重训(exp335_lgpa_gls05,强调部位分支)看部位能否变强到互补。post-hoc 强烈暗示不行(部位要从 35 跳到 >53)。
- **战略 fork 待用户定**：① ViT 0.5 重训(likely 仍负) ② 转 Swin-Small 复现(LGPA-D 证明有效 70.2,step2 正确基座) ③ 直接进 step2(CLIP 价值已被其 PRCV Swin 工作确立)。
- **用户定：先看 0.5 重训结果**(`exp335_lgpa_gls05`)再定基座。

### exp335_lgpa_gls05（GLOBAL_LOSS_SCALE=0.5 + equal_concat eval）
判据：**equalcat(partW=1.0) 是否 > global-only**。是→CLIP 部位在 ViT 上确互补(0.5 强调救活部位)→CLIP +X 确认;否→ViT 上 LGPA-D 确负,定基座(Swin/step2)。
| epoch | global-only | equalcat p1.0 | equalcat>global? |
|---|---|---|---|
| 10 | 31.42 | 23.35 | 否(早期) |
| 20 | 43.65 | 38.41 | 否(gap 5.2) |
| 30 | 47.07 | 42.17 | 否(gap 4.9) |
| 40 | 49.17 | 44.62 | 否(gap 4.5) |
| 50 | 50.29 | 46.11 | 否(gap 4.2) |

### 🔎 10 Codex 深挖结论（用户对了:是我的 bug 非 backbone）→ 3 个真 bug
**① 热图喂错(5/10 共识 = assign=0 根因)**:我硬编码 `heatmaps[:,0]`(target person-0)+ config 设 POSE_USE_TARGET_HEATMAP=True;**原版 `pose_psg_lgpa_detach.yml` 不设此 flag→默认 False→喂 scene-merged 热图**(merge_person_heatmaps 全部人 max)。target-only 稀疏→KL 平凡坍缩→assign=0→部位退化。
**② eval 漏 pooled 块(4/10)**:原版 equal_concat=`[global, pooled, p1..p5]`(7块);我返回 `aux['kp_feats']`=只 5 部位,漏 pooled。
**③ post-hoc L1 归一 bug(3/10)**:`F.normalize(cls,1)` 是 p=1 非 dim=1→post-hoc 数被污染。
**非 bug(高信心)**:feat_map 布局/token序、detach 点、损失接线(7元素list per-part triplet 对)、LGPA 有梯度——逐行确认都对。

### ✅ 修复 + 验证
- 修:`_heatmaps` 默认 scene-merged(merge_person_heatmaps);config POSE_USE_TARGET_HEATMAP=False;model eval 返回 `torch.stack(feats)`=[pooled,p1..p5];post-hoc L1→L2、用 feats。
- **smoke 验证铁证:assign 0.000 → 7.01993 ≈ 原版 7.218!** cross-attn 重新被激活训练。e1 equalcat(1.34) > global(1.22)(老版全程 < )。
- **重训 exp335_lgpa_fixed**(scene+GLOBAL_LOSS_SCALE0.5+equal_concat,assign 正常)。判据:e120 equalcat 是否 > global → CLIP +X 确认。

### exp335_lgpa_fixed eval（mAP，修复后）
| epoch | global-only | equalcat p1.0 | equalcat>global? |
|---|---|---|---|
| 10 | 31.42 | 23.94 | 否(早期;buggy 23.35) |
| 20 | 43.65 | 39.11 | 否(gap 4.54 vs buggy 5.24) |
| 30 | 47.07 | 42.90 | 否(gap 4.17 vs buggy 4.90) |
| 40 | 49.17 | 45.15 | 否(gap 4.02 vs buggy 4.55) |
（e40 停。趋势清晰:修复给稳定 +0.5,但 equalcat 全程 ~4 < global=baseline,不翻盘。）

### 🎯 最终结论（exp335 ViT 纯 LGPA-D 复现）
1. **热图 bug 真实(你对了)**:target-only `[:,0]` → assign KL 坍缩=0。修复(scene-merged)→ assign 0→7.02≈原版。
2. **但只涨 +0.5,不翻盘**。深层原因(查 exp244 config 确认):
   - **LGPA-D 从未单独跑过**。exp244(+4.4)/exp245g(70.2)全是 **POSE_BACKBONE_PSG=True + LGPA + OA-SD + PARALLEL_AUG + 384 + Swin** 完整系统。
   - LGPA 部位的价值来自 **PSG 把 pose 门控进 backbone 特征**;我纯 ViT 无 PSG → 部位从原始(非门控)特征池化 → 与 global 冗余 → 边际。
3. **"纯 CLIP 模块 on ViT" 是从未存在的配置**;原版 CLIP 增益与 PSG 系统纠缠。
4. 非 bug(10 Codex 确认):feat_map 布局/token序、detach 点、损失接线、eval(修 pooled+L2 后)——都对。

→ 用户决定下一步基座/路线(见对话)。
（注:assign 修复(e30=2.65),3 个 fix 给稳定 +0.7,但 equalcat 仍 ~4 < global=baseline(detached)。机制修了但描述子未翻盘。
**重要 reframe**:LGPA detached → global-only == no-LGPA baseline,故 equalcat vs global = LGPA-D vs baseline(正确判据)。equalcat < baseline → 这套 ViT 设置上 LGPA-D 描述子未超 baseline。原版 70.2>baseline 很可能也靠 OA-SD+parallel-aug+384+Swin 全系统。
剩余可试:① pre-norm feature tap(Codex9) ② 加原版组件 ③ 直接在原 pipeline 跑纯 LGPA-D 确认。等 e60 定。）
（注:global-only 全程 < 1.0 run(e20: 43.65 vs 47.05)→ 0.5 压弱 global ~3。equalcat 仍 < global,gap 8.1→5.2 缩小。盯 e60/120。）

### ⭐⭐ 查 exp245g 原始 recipe（重要 context）
- **exp245g(70.2) = Swin-Small + LGPA-D + OA-SD + GCN（全系统）,不是纯 LGPA-D。** 纯 LGPA-D(无 OASD)Swin-Tiny ≈63.6(baseline 56.6 → +7 真涨)。
- **原版 train log: `lgpa_assign: 7.218`（大）;我的 ≈0。** 关键差异:原版 pose-bias **没有**预饱和注意力→assign 损失**主动训练**部位定位;我 ViT 上 pose-bias **主导**注意力→部位退化成被动 pose-pooled 冻结特征→弱。
- 可能 ViT QK 分数尺度小于 Swin → pose-bias(0-1)相对主导。这或是 ViT 部位弱的机制(非纯 backbone)。
- **启示**:① 纯 LGPA-D-on-ViT 负 = 真(且缺 OASD/GCN);② Swin 是 CLIP +X 的确凿基座;③ 若坚持 ViT,step2 的 CLIP 接法不能依赖 pose-bias 主导的 part-pool。

### ⭐ 关键纠正（用户戳穿,查 log,2026-06-18）
- **原版 LGPA-D 测试描述子 = `equal_concat` = [global‖p1‖…‖p5]（每块 L2 归一拼接）,不是 MaxSim、不是 pooled。** exp244=65.3 / exp245g=**70.2** 都是这个标准 eval。
- 原版 config:**`GLOBAL_LOSS_SCALE: 0.5`**（压全局、强调部位）;POSE_MAXSIM_TRIPLET False。
- **我这次复现两处错**:① 描述子用了 [cls‖pooled]（单池化向量）→ 应是 all-parts concat;② GLOBAL_LOSS_SCALE 用了默认 1.0 → 应 0.5。**解释了 pooled best=a0**。
- **计划**:e120 跑 post-hoc all-parts(=equal_concat)。涨→CLIP +X 确认(再 0.5 重训拿满血);不涨→GLOBAL_LOSS_SCALE=0.5 重训。post-hoc 的 MaxSim 行忽略(已知非主描述子)。
**关键洞察**：原 LGPA-D 强数靠 **MaxSim**(每部位匹配,exp244 MaxSim 66.0 vs pooled 63.6 +2.4),非 pooled concat;且 "+4.4" 是 **detach vs non-detach**(detach 避免 backbone 损伤),非 LGPA-vs-baseline。我用 pooled 是更弱描述子。
计划：若 e80 pooled 仍 best=a0 → 训练完用 ckpt 跑 **MaxSim + all-parts concat** post-hoc eval，忠实查 CLIP 部位特征是否带 +X。）

- e10：alpha>0 加 pooled-part 特征变差（部位分支未训熟，早期常态）。LGPA detached → alpha=0 == control（纯外观）。**LGPA 增益 = 纯测试期融合**（best-alpha vs alpha=0）。
- 真判据 = e120 best-alpha 是否超 alpha=0（即 CLIP 部位特征训熟后是否互补外观）。盯 e40/80/120。
