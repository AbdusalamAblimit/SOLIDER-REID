# CVPB monitor

## kill-switch #1: 零训练 local-token MaxSim probe (maxsim_probe.py)

**机器**: lab-3090-d, baseline ckpt `log/cargo/afd_baseline/model_best.pth`
**日志**: lab-3090 `/tmp/cvpb_maxsim_probe.log`
**评测**: A->G (q_aerial=134, g_ground=32268) + G->A (q_ground=178, g_aerial=18444), market-style mAP, 同 afd_train eval。

### 流程校验
- `loaded checkpoint. missing=0 unexpected=0` —— ckpt 与 baseline 模型完全对齐。
- **global-only mean mAP = 32.48**（A->G 32.90 / G->A 32.05）== 已知 baseline 32.48 → **pipeline 正确复现**。

### 结果（mean A<->G mAP，combined-similarity 排序，token L2-norm）

| grid (K tokens) | global | β=0.1 | β=0.3 | β=0.5 | β=1.0 |
|-----------------|--------|-------|-------|-------|-------|
| 8x4 (32 tok)    | 32.48  | 32.74 | 32.62 | 32.64 | **33.34** |
| 4x2 (8 tok)     | 32.48  | 32.42 | 32.46 | 32.72 | 32.77 |

- 注: full(16x8=128 tok) grid 因 gallery 巨大(32268)需 ~34GB 内存装 token(31GB 机器)→ 不可行, 只扫 8x4/4x2 pooled grid。
- A->G 方向受益更明显(8x4 β=1.0: 32.90→34.18, +1.28); G->A 基本持平(~32)。

### 判据
- baseline 32.48 → **best hybrid = 8x4 β=1.0 = 33.34 (+0.86)** ≥ +0.5 → **PASS**。
- 结论: 局部 token MaxSim 在 trained baseline 上带跨视角增量证据(主要来自 aerial-as-query 方向)。Local Token MaxSim 模块值得进入训练(kill-switch #1 通过)。

### 下一步(按 design.md kill-switch 阶梯)
- kill-switch #2: OVP-Mem only 训练(afd_train.py --ovp), ep10/20 vs baseline 同 epoch ≥ +1.0 继续。
- kill-switch #3: OVP + Local 训练 final ≥ 35-36 mAP 进方法稿。

## --ovp 怎么训(已写好, 未开跑)
```
cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
PYTHONUNBUFFERED=1 python3 afd_train.py \
  --data_root /root/work/SOLIDER-REID/data \
  --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovp \
  --ovp --ovp_lambda 0.5 --ovp_tau 0.05 --ovp_momentum 0.2 \
  2>&1 | tee /tmp/cvpb_ovp.log
# baseline 复现: 去掉 --ovp（其余完全等同 afd_reid 基线）
```

## OVLI(★headline)实现状态(2026-06-22, 代码完成, 未训练)

**为什么转 OVLI**: OVP ep30 到 39.18(+大, 证方向有 BIG headroom), 但 OVP novelty 撞 CMPC/MBCE/PDPA。OVLI 保方向换机制——token-set late-interaction(MaxSim)的 sample-to-sample 跨视角检索 loss, 无 prototype/memory/EMA。

**实现位置**: `experiments/cargo_cvpb/afd_train.py`(加 `--ovli` 系列选项 + `OVLIHead` 类 + `ovli_rerank_eval`)。
- `OVLIHead`: hook model.layer4 → adaptive_pool 到 8×4 → **新增 Conv2d(2048→256) proj** + 逐 token L2-norm → 双向 sym MaxSim(B×B)+ opp-view logsumexp supcon loss(τ=0.05, α=0.5)。
- **proj 进 optimizer**: `opt_params = list(model.parameters()) + list(ovli.parameters())`, 且 assert 自检 proj 在 optimizer。
- λ warmup(`--ovli_warmup` 默认 10), 日志 `OVLI[lam_eff loss pos neg gap]`。
- AMP: OVLI loss 在 `autocast(enabled=False)` 真 fp32。`--ovli` off 精确复现 baseline; `--ovp`/`--ovli` 互斥。

**验证**: ast.parse + py_compile 过; 本地隔离 numeric smoke test(导入真实 OVLIHead)全过——token shape/L2-norm、MaxSim 对称、loss 有限>0、梯度回流 proj+global、空候选 batch loss=0 不崩、fp16→fp32 token、AdamW 推动 proj。

**OVLI 训练命令**(GPU 空出 + codex 审通过后):
```
cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
PYTHONUNBUFFERED=1 python3 afd_train.py \
  --data_root /root/work/SOLIDER-REID/data \
  --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovli \
  --ovli --ovli_lambda 0.5 --ovli_tau 0.05 --ovli_alpha 0.5 \
  --ovli_dim 256 --ovli_grid 8 4 --ovli_warmup 10 --ovli_rerank \
  2>&1 | tee /tmp/cvpb_ovli.log
# baseline 复现: 去掉 --ovli
```

**sync 状态**: lab-3090 跳板本会话反复 "Connection closed by UNKNOWN port 65535"(已记录), scp 同步待连通后补做。代码已在本地 + 验证通过。

## 训练结果汇总(2026-06-22 晚, OSS 打通三机四卡并行)

### 已完成
| 配置 | mean A↔G mAP | rerank | vs baseline | 备注 |
|------|--------------|--------|-------------|------|
| baseline (resnet50 BoT) | 32.48 | — | — | |
| **OVLI (α=0.5, ★headline)** | **45.19** | ~45.2 | **+12.7** | token-set late-interaction, 无 prototype/EMA, novelty 无 exact prior |
| OVP (强 ablation, 撞 CMPC) | 50.11 | — | +17.6 | 达 VDT SOTA~50 量级 |
| **OVP+OVLI (full)** | **52.14** | — | **+19.7** | 组合 > OVP(+2.0), late-interaction 与 prototype 互补不冗余 |
| dustbin topk8 (OVLI 变体) | 45.08 | 46.51 | +12.6 | --ovli_pool topk --ovli_topk 8; top-k 真丢非匹配 token 硬化 "partial 匹配" claim; ≈OVLI global, **rerank +1.34** |

> 超 VDT 42.76 / DTST 43.39 这代(在更弱的 resnet50 backbone 上)。多 seed 复核留用户。

**dustbin topk per-direction(vs OVLI mean,2026-06-22 提取)**:

| 方向 | OVLI mean | dustbin topk | Δ |
|------|-----------|--------------|---|
| A→G(强) | 49.21 | 47.68 | -1.53 |
| G→A(弱) | 41.16 | 42.48 | **+1.32** |
| mean | 45.19 | 45.08 | -0.11(持平) |

= top-k pooling **把强方向 A→G 的判别力匀给弱方向 G→A**(net mean 持平,rerank 46.51 略高于 OVLI)→ **design-choice 故事(非纯涨点);论文默认仍用 mean(mean 方向最优),topk 作为"鲁棒弱方向"消融**。

### 主结果 per-direction 全表(2026-06-22 从 lab-3090 log 精确提取, 论文主表素材)

| Method | A→G | G→A | mean | A→G 增益 | G→A 增益 |
|--------|-----|-----|------|---------|---------|
| baseline | 32.90 | 32.05 | 32.48 | — | — |
| OVLI(headline) | 49.21 | 41.16 | 45.19 | +16.31 | +9.11 |
| OVP(强 ablation) | 52.28 | 47.93 | 50.11 | +19.38 | +15.88 |
| **OVP+OVLI(full)** | **55.43** | **48.85** | **52.14** | **+22.53** | **+16.80** |
| dustbin topk | 47.68 | 42.48 | 45.08 | +14.78 | +10.43 |

**★ 关键 paper 故事(方向互补)**:
- **OVLI(late-interaction)偏 A→G**(A→G +16.31 >> G→A +9.11): 航拍 query→地面 gallery 时形变/尺度差最大, partial-token 集合匹配比全局增益最大。
- **OVP(prototype)更均衡**(A→G +19.38 / G→A +15.88 都强): 全局原型兜底两个方向。
- **组合在 OVP 上加 OVLI 主要再涨 A→G**(+3.15 vs G→A +0.92), 与 OVLI 的 A→G 偏好一致 → **两机制方向互补**(prototype 兜两向, late-interaction 专攻强形变 A→G), 这正是 52.14 > 50.11 的来源。
- dustbin topk 是 OVLI 内部的"方向 rebalance 旋钮"(把 A→G 判别力匀给 G→A)。

### 四卡并行消融(2026-06-22 晚, 训练中, ~60ep)
| 机器 | 配置 | flag | 隔离的变量 | log |
|------|------|------|-----------|-----|
| lab-3090 | allview | `--ovli_allview` | oppview-only vs all-view 正样本(跨视角约束是否关键) | /tmp/cvpb_ovli_allview.log |
| hyy GPU0 | α=1.0 | `--ovli_alpha 1.0` | 无 late-interaction 对照(score 只 global cos) | /tmp/cvpb_ovli_alpha1.log |
| hyy GPU1 | α=0.0 | `--ovli_alpha 0.0` | 纯 late-interaction(score 只 MaxSim) | /tmp/cvpb_ovli_alpha0.log |
| lab-4090 | τ=0.1 | `--ovli_tau 0.1` | τ 敏感性(默认 0.05) | /tmp/cvpb_ovli_tau01.log |

预期: allview ≈ OVLI 则跨视角约束非关键(弱化 headline); α=1.0 < OVLI 则 late-interaction 有价值(强化 headline); α=0.0 看纯 late-interaction 上限。

### 消融结果(陆续出)
- **allview DONE(2026-06-23)**: mean **44.67**(A→G 47.62 / G→A 41.71)vs OVLI(oppview)45.19(A→G 49.21 / G→A 41.16)。**oppview-only 约束: A→G +1.59 / mean +0.52 / G→A -0.55** → 跨视角约束**适度有用**(把 late-interaction 聚焦到更难的跨视角 A→G,不是无关也不是巨大)。论文框架: "restricting positives to opposite-view pairs improves cross-view A→G mAP by +1.59"。
- **★ τ=0.1 DONE(2026-06-23)**: mean **47.06**(A→G 50.26 / G→A 43.85)vs OVLI(τ=0.05)45.19(A→G 49.21 / G→A 41.16)。**τ=0.1 全面更好: mean +1.87 / A→G +1.05 / G→A +2.69**!→ OVLI 默认 τ=0.05 偏硬,**软 τ=0.1 更优**,尤其修弱方向 G→A(+2.69)。论文 τ sweep 报 {0.05,0.1} 用 τ=0.1 当最优(单 run,多 seed 待用户)。⚠️ **暗示组合 OVP+OVLI 用 τ=0.1 可能 > 52.14**——值得复跑组合。
- **🔴 match=avg DONE(2026-06-23,需战略分析)**: mean **51.00**(A→G 53.33 / G→A 48.67)**>> OVLI MaxSim 45.19**(A→G 49.21 / G→A 41.16),**+5.81(G→A +7.51)**!= proj-token 均值的全局软匹配远好于 MaxSim late-interaction。**match=avg(51.00)≈ OVP(50.11)都是全局**。**对 headline 的威胁**: late-interaction(MaxSim)单独不是最优 pooling。**辩护假说(待 OVP+avg 确认)**: avg 与 OVP 都全局→冗余; MaxSim 与 OVP distinct→互补(组合 52.14 中 OVP+MaxSim 的 +2.0 是 MaxSim 的独特贡献,avg 大概率给不出)。**★必做确认: OVP+avg 组合**(若 ≈OVP 50.11 不涨 → 证 avg 冗余、MaxSim 互补独特 → headline 保住;若 >52.14 → 改 headline)。codex 战略分析中(pid 89132)。
- **🔴🔴 match=avg FINAL = 52.37 @ ep60(2026-06-23)——超组合 52.14!** avg-pooled OVLI **单独**已是最强单机制(>OVP 50.11、>OVP+OVLI 组合 52.14)。这是 codex 的 ">52.14 场景":**MaxSim late-interaction 当 headline 严重受威胁**。codex 建议: reframe headline → "**opposite-view projected multi-token contrastive(avg-pool)**",late-interaction(MaxSim)降为可选 pooling/弱 ablation。✅ 好消息: match=avg 52.37 是**干净的单机制 SOTA**(不靠 OVP+OVLI 双机制组合),反而更好讲。OVP+avg 测试(b4vl20qv4 ep1)看 OVP 是否还有增量。**待定 headline: OVLI=opposite-view multi-token contrastive,avg-pool 最优,MaxSim/ordered 作为 pooling 消融。**
- **align=ordered FINAL = 45.51(2026-06-23,ep40 中间值 37.96 是误导,final LR 衰减后 climb 上来)≈ OVLI free 45.19** → **free 与 ordered 接近**(都 ~45),MaxSim 的"自由 vs 限行对齐"差异不大。真正的大差异是 **MaxSim(~45)vs avg(52.37,+7)**。即:不是"free 远好于 ordered",而是"**avg 远好于 MaxSim(任何对齐)**"。pooling(max vs mean)比 alignment(free vs ordered)重要得多 → 进一步印证 headline 应走 avg。
- **α sweep(ep40,hyy)独立印证**: α=1.0(score 纯 global gfeat,无 MaxSim)= **45.14** ≈ α=0.5(OVLI mix)45.19 >> α=0.0(纯 MaxSim 无 global)**26.53**。→ **global gfeat 是 workhorse,MaxSim 加成≈0(α=1.0≈α=0.5),MaxSim 单独很弱(26.53)**。与 match=avg 第二个独立证据: **late-interaction MaxSim 不是机制**。关键: avg(proj-mean)**52.37 >> α=1.0(raw global gfeat)45.14** → **学到的多-token 投影 + mean-pool 才是 +7 增量源**(不是 raw global,更不是 MaxSim)。★ **最终 headline 机制 = opposite-view 学习投影多-token 均值对齐;novelty 在"投影多-token + 对侧视角约束",pooling/alignment 是消融维度。**
- **codex 完整分析(2026-06-23,pid 89132)**: OVP+avg 判读: `<52.14/≈50`→avg 与 OVP 冗余、MaxSim 互补→headline 保住; `≈52.14`→MaxSim 互补不独特、降级; `>52.14`→改 global/projected headline。**codex 必做清单 + 执行**: ① OVP+avg(运行中, 优先级1) ② **OVP+MaxSim τ=0.1**(给 MaxSim 最优组合, 因单独 τ=0.1 47.06>45.19, 当前 52.14 可能用了未调好的 MaxSim——★待跑) ③ α=1.0/α=0.0(已收 45.14/26.53 → avg 52.37 来自 **proj-token mean** 非纯 global gfeat)。**最稳路线: full(MaxSim 52.14 或 avg 52.37)主结果 + OVP baseline + OVP+avg 定 MaxSim 是否非冗余互补。B类可投, 走收窄版 headline。**
- **τ×pooling 交互(2026-06-23)**: avg-τ0.1 final **48.31 < avg-τ0.05 52.37**(-4)。即 **τ 最优依赖 pooling**: MaxSim 喜软 τ=0.1(47.06>45.19), avg 喜硬 τ=0.05(52.37>48.31)。论文可作为一个有意思的 τ 敏感性观察(不同 pooling 对温度的偏好相反)。**确认 avg 最优 config = τ=0.05, mean-pool, α=0.5,= 52.37。**
- **★ OVP+avg FINAL = 52.28(2026-06-23, A→G 55.65 / G→A 48.92)≈ avg-alone 52.37(-0.09)→ OVP 对 avg 完全冗余**(组合不涨)。OVP+avg(52.28)≈ OVP+MaxSim(52.14)≈ avg-alone(52.37),三者都 ~52 → **OVP(prototype)既撞 PDPA/CMPC 又不给增量 → 彻底丢**。**headline 确定 = avg 单机制(DCVP 方向),无 OVP = 无撞车。** codex 角度8 的 feature-only 单机制路线被经验确认。
- **★★ 新方向执行中(2026-06-23)**: DCVP = opposite-view identity evidence distribution prediction(见 codex_fleet_synthesis.md)。**OVC-SetVLAD(netvlad)kill-switch 跑中**(lab-3090, --ovli_setpool netvlad, 双审过): feature-only >52.37 则分布故事成立。setpool 5 变体(mean/netvlad/attn/gated/secondorder)已实现双审。
- **★ curveball: OVP+MaxSim τ=0.1 FINAL = 52.76(2026-06-23)——目前最高**: > avg-alone 52.37(+0.39)、> OVP+MaxSim-τ0.05 52.14、> OVP+avg 52.28。给 MaxSim 最优 τ + OVP 后, MaxSim 在组合里**反超 avg**(standalone avg52.37>>MaxSim45.19, 但 with OVP: MaxSim-τ0.1 **52.76** > avg 52.28)→ **MaxSim+OVP 互补性 REVIVES**(codex 优先级2 测对了)。但 +0.39 小(单 run 噪声内)+ OVP 撞 PDPA。**三选一(等 netvlad)**: (a) avg-alone 52.37 干净单机制无撞车; (b) OVP+MaxSim-τ0.1 52.76 最高但 OVP 撞车+2机制(可配 ACVP 改装救); (c) **netvlad(OVC-SetVLAD)若 >52.37 则干净+novel+高**——最优解, 等 kill-switch。
- 全数字表(单run): baseline 32.48 / MaxSim-τ05 45.19 / MaxSim-τ1 47.06 / **avg 52.37** / OVP 50.11 / OVP+MaxSim-τ05 52.14 / OVP+avg 52.28 / **OVP+MaxSim-τ1 52.76**。
- **★★ setpool 消融最终结论(2026-06-23): 可学习池化全失败, mean-pool 最优。** standalone: netvlad 14.66(ep20)/ attn 37.66(final)。residual(mean+零初始化残差, 从 52.37 字节级起步): attn ep30 21.30→ep50 33.49(final~37, ≈standalone); clean netvlad ep10 16.00。**诊断: 即使无损起步, gate 一开残差就拖垮 eval(训练 loss 降但 mAP 降=残差过拟合训练对/梯度经 layer4 扰动 global)。优化器为降训练 loss 开 gate, 但 eval 掉。** → **OVC-SetVLAD(codex 角度2 涨数字赌注)证伪: 4 种 fancy 池化 × standalone/residual 两版全 < mean-pool 52.37。简单 mean(centroid)是最优跨视角 token 聚合。** 这是干净有教育意义的 ablation(试了 fancy 反而验证 simple 最优)。
- **★ headline 回落(2026-06-23)**: OVC-SetVLAD 死(涨数字失败)→ headline = **OVLI(cross-view 对比 + mean-pool 多token)52.37 单机制**, setpool 消融当支撑证据(反直觉: fancy 池化全输 simple mean)。备选 high-number: OVP+MaxSim-τ1 52.76(需角度6 ACVP 改装救 OVP 撞车)。等 clean netvlad final 确认 netvlad 也死。
- **★★ allview 消融 FINAL = 44.65(2026-06-23)<< avg(opposite-view)52.37,差 -7.72**: 把 OVLI 的 opposite-view 正样本换成 all-view 正样本(同身份所有视角都当正样本),掉 7.72 mAP → **跨视角约束(只跨视角拉正)是 OVLI 核心 novelty 的硬证据**(不是普通 contrastive,opposite-view 强制学跨视角不变性才是关键)。强支撑消融, 进 paper 消融表。
- **★ 支撑消融组合(2026-06-23)**: OVLI headline 52.37 + ① setpool(fancy 池化全输 mean,反直觉)② allview(跨视角约束 +7.72)③ α sweep(α=0.0 纯proj-token **32.26** / α=0.5 blend **52.37** / α=1.0 纯global 45.14 → **blend 最优**, proj-token 单独弱但与 global 互补 +7.23 超纯 global; α=0.7 跑中)④ ACVP(歧义负样本软化,ep20=28.53 爬升中)。paper 消融硬证据链成形。
- **★ setpool 表全 final(2026-06-23)**: mean 52.37 / netvlad(residual)29.60 / attn(residual)38.32 / attn(standalone)37.66 / netvlad(standalone)14.66 → 可学习池化全 <<mean, 干净证伪。
- **★ α sweep 完整(2026-06-23)**: 纯proj-token 32.26 < 纯global 45.14 < blend(α=0.5)52.37。结论: proj-token mean 单独弱(32.26), 但提供 global 没有的跨视角对齐信息 → 融合 +7.23。OVLI 双成分(global cosine + proj-token mean gram)设计被 α sweep 验证。
- **★★ ACVP 证伪(2026-06-23)**: ACVP(avg + 歧义负样本软化)一直低于 avg(OVLI)轨迹 ~8-10 mAP(ep10 19.36 vs avg 29.52 / ep20 28.53 vs 34.37 / ep30 35.94 vs 43.89)。负样本软化确实缩小 A→G/G→A gap(ACVP 4.31 < avg 6.73)但代价是**整体判别性下降**(软化负样本=模型少学难区分样本=整体掉)。codex 角度6 证伪。
- **★★★ 战略结论(2026-06-23)**: **两个涨点机制(OVC-SetVLAD 角度2 + ACVP 角度6)全失败 → resnet50 + 现有 OVLI loss 框架内"再加一个涨点模块"这条路走完。** headline 定为 **OVLI 52.37 单机制 + 4 个干净消融**(setpool/allview/α sweep + OVP 对照),5-codex 评 B 类可行。**下一步换思路: Swin/SOLIDER 强 backbone 冲 SOTA**(弱 resnet50 都 52.37 超 VDT 42.76,强 backbone 应上台阶)→ AG-ReID.v2 跨数据集 → paper 骨架。
- ⚠️ **教训(2026-06-23)**: 同步代码到 hyy/lab-4090 时漏了 `maxsim_probe.py`(rerank eval `from maxsim_probe import eval_from_distmat` L626)→ 3 个实验 ep10 首次 rerank 崩 `ModuleNotFoundError`。已补同步 + 重启。**另一坑**: `ssh host "cd X && a & b &"` 的 cd 只作用第一个 `&` 前 → b 在 home 跑找不到脚本; pkill+launch 同 shell 会自杀。多卡分开启动每条带 cd。

### subagent 在写下两个消融(未训练)
`--ovli_match {maxsim,avg}`(max vs mean 匹配)+ `--ovli_align {free,ordered}`(自由 vs AlignedReID 式有序对齐, novelty-defense)。完成后双审 → 排队跑。

### ★★★ Swin eval mAP=0.03 根因 = epoch-8 训练塌缩(非 eval-path bug)(2026-06-23)

**现象**: `cvpb_swin_ovli`(Swin-Small + OVLI, lr 3.5e-4 AdamW 均一)训练 **ep1-7 健康**(Acc 0.003→**0.472**, CE 7.8→3.5, OVLI gap→+0.32), 但 **ep8 Iter50 一步塌缩**(LR 升到 2.46e-4): Loss 4.16→10.36, CE→7.8(回到 random-logit 天花板 ln(2500)≈7.8), Acc→0.01, OVLI pos≈neg≈0.49 gap→0。ep9+ 不恢复 → eval mAP=0.03(≈随机)。`model_best.pth` 存在 ep10(唯一 eval, 已塌)→ 装出来就是塌缩权重。

**诊断**(`diag_swin_eval.py` fresh model + `diag_swin_ckpt.py` 加载塌缩 ckpt, lab-3090):
- **fresh model**(刚加载 teacher, 未训练): eval 特征**正常** —— 8 张真实 CARGO 图 final off-diag cos **+0.24**(可区分), 全 finite, BNNeck unit-norm。**→ eval forward / `.cuda()` semantic-weight / LayerNorm / 取 tensor 路径全部正确, 不是 train/eval 不对称 bug。**
- **塌缩 ckpt**: `outs[-1]`(Swin 末 stage map)off-diag cos **+0.992**, batch-chan-std **0.038**(健康 fresh model 是 2.67)→ **backbone 对所有输入输出近乎常数**。global_feat off-diag **+0.9995**(全塌)。final +0.72(BNNeck 因 running_var≈3.6e-4 极小, 把残差放大了一点点, 但仍 +0.72=基本不可区分)→ mAP 0.03。**所有权重 finite, 无 NaN/Inf** → 是表征塌缩, 不是数值溢出。

**根因**: **resnet50 调出来的峰值 LR(3.5e-4 AdamW 均一施加到 ~50M 参数 SOLIDER Swin transformer)对 Swin 过大 → warmup 升过 ~2.5e-4 时几步大更新把 backbone 推进"常数输出"退化吸引子**。resnet50 不塌(对它 3.5e-4 安全), 仓库主 SOLIDER config 训 Swin 用 SGD BASE_LR=8e-4 + 20-epoch warmup(对 transformer 等效步长温和得多)。这是 LR 问题, 不是 AMP/forward 问题(AMP 仅可能放大, 但 LR 是触发器; 全程无 inf/scale 警告)。

**修复**(只动 Swin 路径, 不碰 resnet50/eval):
1. `cargo_cvpb/afd_train.py`: backbone='swin_small' 时给 **Swin backbone 单独 param-group, LR×`--swin_lr_factor`(默认 0.1=3.5e-5)**, heads/BNNeck/OVLI proj 保持 full LR 3.5e-4(随机初始化要快学)。resnet50 无 `backbone_swin` → 单 group, 字节级不变。`WarmupCosineLR` 天然支持 per-group base_lr。
2. `model/backbones/swin_transformer.py` L1400: `w.cuda()` → `w.to(x.device)`(跟随输入设备, 鲁棒性修复; dtype 留 fp32 保持原行为)。非本 bug 根因, 但是真实潜在隐患。

**验证 ✓ 修复确认成功**(`cvpb_swin_fix256`, 打补丁 trainer, `--img_size 256 128 --ovli_match avg`, swin_lr_factor=0.1):
- **平稳过 ep8 不塌**: ep6 Acc 0.467 → ep7 0.619 → **ep8 0.701**(旧 run 此处崩到 0.008)→ ep9 0.754 → ep10 **0.779**, OVLI gap 全程 +0.32~+0.36 稳升。
- **ep10 eval mean mAP = 45.38**(A→G 46.03 / G→A 44.72, rerank 47.20)—— vs 旧塌缩 run 同 epoch **0.03**, ×1500, 且仍在爬。**根因(LR 过大致塌)+ 修复(Swin backbone 0.1× LR)双向证实, Swin eval 现可正常提判别特征。**
- 优化器分组确认: `backbone LR=3.5e-5 / heads=3.5e-4`。(我自启的 `diag_swin_fix` ep2 被 GPU 争用外部 kill, 健康但中断; `cvpb_swin_fix256` 是更完整确认 run。)

### 基建突破: OSS 打通数据传输(三机四卡并行的前提)
SSH/scp 给 hyy(入站 5KB/s≈40h)/lab-4090(0KB/s)废。**恒源云 OSS 官方客户端(curl gpucloud-static-public-prod.gpushare.com/installation/oss/oss_{darwin,linux}_x86_64)任意机器带账号可登录** → Mac 上传 18MB/s, hyy 下载 77MB/s(gpushare 内部), lab-4090 8MB/s。详见 memory `oss-data-transfer-to-gpushare`。
- hyy python = `/hy-tmp/solider2/bin/python`(torch 2.7+cu128, 5060Ti sm_120 实测能跑); data_root `/hy-tmp/reid-clean/data`(CARGO/{train,query,gallery})。
- lab-4090 python = `/home/afr/vireid/.venv/bin/python`(torch 2.4.1+cu121); data_root `/home/afr/SOLIDER-REID/data`。**lab-4090 缺 afd_reid/afd_train.py 会循环 import**(cargo_cvpb 从它 import CE/Triplet/LR), 已 OSS 同步修复。

---

## kill-switch (AIRL): 零训练 aerial-scale 分桶 A->G mAP 诊断 (airl_scale_diag.py)

**日期**: 2026-06-23
**机器**: lab-3090-d
**脚本**: `experiments/cargo_cvpb/airl_scale_diag.py`
**日志**: lab-3090 `/tmp/airl_scale_diag_swin.log` / `/tmp/airl_swin_height.log` / `/tmp/airl_r50_area.log`
**假设**: CARGO A->G 主误差来自 aerial crop 低像素预算(小 bbox=分辨率低=身份物理不可辨识), 非单纯视角对齐。
**方法**: 用 baseline ckpt(不训练), 按 aerial query 的**原生 bbox 像素**(resize 前 PIL.size)分 4 个等量分位桶, 各桶 query 对**同一全量 ground gallery** 算 market-style mAP(复用 afd_train.eval_market)。gallery 跨桶不变 → 桶间 mAP 差异只归因 aerial 尺度。

### 流程校验
- Swin ckpt(`cvpb_swin_fix256/model_best.pth`, OVLI-Swin, ep30): `missing=0 unexpected=0`; FULL A->G mAP=**54.14** == 训练 log ep30 的 54.14 → pipeline 精确复现。OVLI head 不在 eval 路径, eval=global BNNeck L2-norm。
- aerial query 原生尺度跨度极大: area 1170~19758 px(**17x**), height 26~125。median area 9545 ≈ ground(29640)的 1/3。

### 结果(最低尺度桶 = b0)

| 配置 | b0(最小)mAP | 顶桶 mAP | gap(高-低) | max spread | reliab AUROC |
|------|------------|---------|-----------|-----------|-------------|
| **Swin × area** | **40.78** | 54.19 | **+13.41** | 24.11 | 0.715 |
| **Swin × height** | **46.12** | 65.12 | **+19.01** | 19.01 | 0.715 |
| **ResNet50 × area** | **18.85** | 34.71 | **+15.86** | 26.83 | 0.805 |

- 三跑全部: **最低尺度桶塌陷**, gap 远超 +3~5 阈值。ResNet50 b0 灾难性(R1=12.5, R1acc=8.8%, R5=37.5 vs 其余 55-65)。
- 非单调: 顶桶(b3)有时略低于 b1/b2(对齐/裁剪噪声), 但 kill-switch 决定性信号 = **小 bbox 桶 vs 其余的塌陷**, 稳健成立。
- **强 Swin 上 b0 仍塌**(40.78 vs 54-65) = 与 OVLI 死因相反: 不是弱 backbone artifact, 是强 backbone 也解决不了的物理像素预算问题。正好过 OVLI 缺的"机制内在价值"关。
- reliability AUROC 0.715(Swin)/0.805(ResNet50): 模型 top-1 cos 置信度已部分知道哪些 aerial query 不可辨识 → 支持 AIRL "calibrated reliability" 子主张。

### 判据 → **PASS**
gap 13.4~19.0 mAP >> +3~5 阈值, 三配置一致, 强 backbone 上仍塌。**aerial 低像素预算是 A->G 主误差源, AIRL 角度值得推进。**

## kill-switch #4 (AIRL 决定性训练): 单模型双分支 (cvpb_airl_dualbranch)

**日期**: 2026-06-23
**机器**: lab-3090-d (GPU0, 启动前 11MiB/0% 空闲)
**日志**: lab-3090 `/tmp/cvpb_airl_dualbranch.log`
**out_dir**: `/root/work/SOLIDER-REID/log/cargo/cvpb_airl_dualbranch`
**启动命令**:
```
cd /root/work/SOLIDER-REID/experiments/cargo_cvpb && CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 setsid python3 afd_train.py \
  --data_root /root/work/SOLIDER-REID/data \
  --out_dir /root/work/SOLIDER-REID/log/cargo/cvpb_airl_dualbranch \
  --airl_dualbranch --airl_fuse_w 0.25 \
  --backbone swin_small --swin_pretrain /root/work/SOLIDER-REID/pretrained/swin_small.pth \
  --img_size 256 128 > /tmp/cvpb_airl_dualbranch.log 2>&1 < /dev/null &
```
(数据 `/root/work/SOLIDER-REID/data` = 默认; swin = `pretrained/swin_small.pth` 1.15GB; 与 OVLI-Swin `cvpb_swin_fix256` 同路径同 img_size 256x128。)

**机制**: 1 forward 2 features —— clean f_full(原 BNNeck head) + f_rec(新 2nd BNNeck head: 自带 ID-CE + AIRL resolution-degradation consistency), eval 软融合 `cos = w*cos_rec + (1-w)*cos_full`, w=0.25 固定。f_rec head 1920768 params(2 tensors)已进 optimizer(启动日志自检 True)。

**kill-switch #4 判据**: baseline-Swin fuse mean(A<->G) = **60.84**。
- **fuse mean ≥ 61.84 (+1.0) → 机制成立 (B 类候选)**
- **fuse mean < 61.84 → 杀**
- oracle 软融合上界 +1.46 → 期望 fuse mean ~62.3。
- 旁证: full 档应 ≈ baseline 60.84(clean head 未变); rec 档应像纯 AIRL 偏 A->G。

### 启动验证 (ep1 早期)
- 配置 banner: `airl_dualbranch=True (fuse_w=0.25 ...)`; swin pretrain `All keys matched successfully`; CARGO train 51451/2500pid, query 312/149pid。
- `[AIRL-DUAL] f_rec head params in optimizer: True (1920768 params, 2 trainable tensors)` —— 第二头进优化器。
- ep1 loss 分量全在且正常下降: Loss 47.6→36.0(iter50→300), CE≈CE_rec≈7.82(两头同源 OK), Tri 32→20, **AIRL_rec≈0.0002**(consistency 小, 不爆), Acc~0(warmup LR 3.5e-07)。GPU 91% util / 12.9GB。
- 训练健康, 等 ep10/ep20 eval(full/rec/fuse × A->G/G->A)。

### ep1-10 收敛(每 epoch heartbeat)
| ep | Loss | Acc | lam_eff | ce_rec | consistency | deg_scale |
|----|------|-----|---------|--------|-------------|-----------|
| 1 | 30.26 | 0.002 | 0.100 | 7.818 | 0.0002 | 0.622 |
| 5 | 9.91 | 0.313 | 0.500 | 4.581 | 0.1158 | 0.625 |
| 8 | 5.37 | 0.750 | 0.500 | 2.448 | 0.2778 | 0.626 |
| 10 | 4.75 | 0.818 | 0.500 | 2.150 | 0.3081 | 0.623 |
- consistency warmup 5 ep 完成(lam_eff 0.1→0.5), consistency 项 0.0002→0.31 平滑爬升后趋稳, 不爆。**无 ep8 Swin 塌陷**(split LR 生效)。AIRL_rec(per-iter)≈0.30 稳定, f_rec 头(CE_rec)与 full 头(CE)同步收敛。

### ep10 eval(f_full-only global, = dual block 的 full 档)
- `[A->G] mAP=41.46 R1=37.23` / `[G->A] mAP=42.53 R1=51.52` / **`[mean] mAP=41.99 R1=44.37`**
- dual-branch FUSE 块(full/rec/FUSE 第二遍, 50k gallery×2 方向)计算中, 待打印。

**⚠️ baseline 对照口径澄清**: 任务给的 baseline-Swin **60.84** 是**最终收敛(ep60)**数, 门槛 61.84 / oracle +1.46 都是**对终值**判据。ep10/20 是早期趋势点, 不能直接拿 41.99 比 60.84。
- 同机 OVLI-Swin(`cvpb_swin_fix256`)轨迹: ep10=45.38 → ep20=48.35 → ep60=67.33。
- 本 dual-branch(clean baseline + AIRL, **无 OVLI**)ep10 full=41.99 比 OVLI-Swin ep10(45.38)低 ~3.4 — 口径不同(无 OVLI)+ AIRL consistency(lam=0.5)早期正则压低 ID 收敛, 后期回升, 故 kill-switch 看 ep20+/终值而非 ep10。
- **决定性看两点**: (a) 每 epoch FUSE 是否 > full(rec 分支是否加跨视角证据); (b) 终值 FUSE mean ≥ 61.84。

### ep10 dual-branch FUSE 块(★决定性) — ⚠️ 弱信号
```
[A->G] full=41.46  rec=41.54  FUSE=41.54
[G->A] full=42.53  rec=42.54  FUSE=42.55
[mean] full=41.99  rec=42.04  FUSE=42.05  <- model-selection uses FUSE
```
- full=41.99 与 global eval 41.99 **bit-for-bit 一致** → dual eval 接线正确。
- **rec≈full≈FUSE(全 ~42.0)**: rec−full=**+0.05**, FUSE−full=**+0.06**。f_rec 头几乎产出与 f_full 相同特征, 软融合几乎不加东西。
- ⚠️ **核心 kill-switch 隐忧**: AIRL 双分支论点 = f_rec(resolution-degradation consistency 训)学到**互补**的 resolvability-aware 表示, 融合后 ≥+1.0。ep10 两头**坍缩成同一表示**, 无互补证据可融。oracle 预言 +1.46, 实测 +0.06。
- ep10 中段(两头仍共适应)不是终判, 但**早期轨迹弱**: rec 不从 full 分化 = 正是会杀掉机制的失败模式。fuse delta 需在 ep20/ep60 大幅长到 ~+1.0。
- 等 ep20: 若 FUSE−full 仍 <+0.3 且 FUSE mean 远不在冲 61.84 的轨道 → 早报杀。

### ep20 eval（global f_full）
- `[A->G] mAP=49.66 R1=54.26` / `[G->A] mAP=45.12 R1=59.09` / **`[mean] mAP=47.39 R1=56.67`**（ep10 41.99 → ep20 47.39, 正常收敛）。

### ep20 dual-branch FUSE 块(★决定性) — ❌ kill-switch #4 FAIL
```
[A->G] full=49.66  rec=49.92  FUSE=49.69
[G->A] full=45.12  rec=45.47  FUSE=45.14
[mean] full=47.39  rec=47.70  FUSE=47.41  <- model-selection uses FUSE
```
- **FUSE − full = +0.02**（ep10 +0.06 → ep20 +0.02, **不增反减**, 与"应随训练放大到 +1.0"的需求轨迹相反）。
- rec − full = +0.31（rec 单独略好, 但…）, **融合摧毁 rec 增益**: rec 单独 47.70, 但 w=0.25 软融合后 FUSE=47.41 < rec, 仅比 full 高 +0.02。full 与 rec 高度相关 → 混合回归到均值, 把 +0.31 稀释成 +0.02。
- oracle 预言软融合 +1.46; 实测 +0.02（差 50x+）。

### ⛔ 判决: kill-switch #4 FAIL → 杀（不跑到 ep60）
- **判据**: 终值门槛 FUSE mean ≥ 61.84（baseline 60.84 +1.0）。决定性的是 **FUSE−full delta** —— 与 epoch 无关、应随训练放大, 但 ep10→ep20 = +0.06→+0.02 平到缩, 无任何走向 +1.0 的轨迹。
- **机制层面已定**: f_rec(resolution-degradation consistency 训)≈ f_full 的冗余拷贝, consistency loss 没切出独立的 resolvability-aware 表示, 只是轻微正则了同一表示。两头坍缩 → 软融合无互补证据可融, 且 w=0.25 blend 主动稀释 rec 那点微弱增益。
- **与 oracle 矛盾的根因**: airl_gate_oracle 的 +1.46 上界是用**两个独立模型**(baseline + AIRL)的 score fusion 测的; 单模型双分支共享 backbone, 两头无法分化到独立模型那种互补度 → oracle headroom 不可在单模型内实现（与 exp109 identity-conditioned headroom 不可实现同类陷阱）。
- **决定**: 立即 kill 释放 GPU slot（跑到 ep60 ~3.2hr 换一个机制层面已定的负结果, 不值）。AIRL 冲 B 类(单模型双分支)**证负**。

### 执行 + 交接（2026-06-23 09:03）
- `cvpb_airl_dualbranch` 已 kill（ep21 iter50 处终止, log 冻结 09:01:28, 无 afd_train 残留进程）。**naive 单模型双分支证负 = 共享 trunk 两头坍缩。**
- **发现并行 rescue 实验**: 用户读到 ep10/20 弱信号后已启动 `cvpb_airl_iso`（`--airl_dualbranch_iso --airl_iso_stage 3`）—— **梯度隔离救援**: f_rec = 独立 late Swin stage(14185392 params, 28 tensors) 从 **detach 的 trunk** 分叉, consistency 梯度永不回到共享 trunk, clean trunk+f_full 保持干净, f_rec 专学 recover 极。正是针对"两头坍缩"失败模式的修复。out_dir `cvpb_airl_iso`, log `/tmp/cvpb_airl_iso.log`。
- **iso 健康运行中**（kill dualbranch 后独占 GPU0, 87% util, ep1 loss 正常下降, AIRL-ISO 头进 optimizer 自检 True）。iso 是用户的后续实验, 非本次任务范围, 未代为监控（用户要求时再接）。
- **iso 判据沿用**: 同 baseline-Swin 60.84 / 门槛 61.84; 关键仍看 ep10/20 的 **FUSE−full delta** 是否因梯度隔离而真正放大（naive 版 +0.02 → iso 若成立应明显 >0 走向 +1.0）。