Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/rr_logs
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019efab5-2505-7e31-af5e-1e21b25b7a52
--------
user
一个 ReID 团队的 **evidence-sufficiency** 方向零训练 kill-switch LIVE+强: ①诊断真——weak positive-support(query 对自己跨相机正样本的最弱相似度 lowtail-pos / 正样本散度 dispersion)驱动失败, partial 控 1x-margin+#false 后仍 +0.44~0.52(两域), 决定性区别于被 trivial 吃掉的旧诊断(Hubness 控 #false 后塌到 0); ②oracle 证据 union——给失败 query 加第 2 张同 ID 图, Occluded-Duke 从 17.2→52.5(+35, 恢复 76.8%), 加随机异 ID 图反而砸毁(证明增益是身份证据非降噪), 且是 k-reciprocal 增益的 3.3x(现成 re-rank 盖不住)。

要做 CCF-B **方法稿**。设计:
① **method 是什么**? 候选: multi-query/tracklet 证据融合(evidence-aware, 用 lowtail-pos 决定何时/如何融) / retrieval reliability calibration(预测 fragile-query 给可靠性分, 不做恢复) / evidence-conditioned retrieval / 别的。关键约束: 利用'证据 union 有真 headroom'但—— (a)T3 是 oracle(需第 2 张同 ID 图), 真实部署'从哪来第 2 份证据'? multi-query/tracklet 设定可拿(注意: 这**不是** exp109 的 gallery-side completion identity-conditioned 墙, 是 query-side 多帧), 但 multi-query/set-based ReID 是已知设定, novelty 必须在'evidence-aware'机制不在'用多帧'本身; (b)不撞 occlusion/completion 死区; (c)T2 显示'静态失败预测'被 #false 吃掉, 别把 method 建在静态失败预测上, 要建在 tax-residual/fragility 或 evidence-gain 上。
② **新问题/协议**怎么定义让 method 自然且 novel。
③ **撞车核查**(联网 2024-26): multi-query ReID / set-based ReID / temporal/tracklet aggregation / uncertainty-aware ReID / retrieval reliability calibration / evidence aggregation / TEMP 类。确切 novelty 边界 + 怎么切开。
④ 诚实: 能撑 CCF-B 方法稿吗(信心 1-10), 还是 oracle headroom 本质不可部署化(=exp109 墙的 query-side 变体)? 若能, 最该做哪个 method。
务实中文, 给 method 草案 + 协议 + 撞车 + verdict + 信心。
codex
我会先核一下仓库里当前实验文档是否在这个子目录可读，同时联网查 2024-2026 的多查询/tracklet/不确定性/ReID 可靠性相关工作，再给出可落地的方法边界和投稿判断。
exec
/bin/zsh -lc "find . -maxdepth 3 -type f | sed -n '1,120p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/rr_logs
 succeeded in 0ms:
./cvpb_rr_od_smoke.log
./rr_od_stage0.json
./cvpb_rr_market_s1.log
./cvpb_rr_market_s2.log
./cvpb_evidence_market.log
./cvpb_rr_od_s1.log
./cvpb_rr_od_s0.log
./cvpb_rr_od_s2.log
./rr_market_stage1.json
./rr_od_stage1.json
./cvpb_evidence_oduke.log
./rr_market_stage2.json
./rr_od_stage2.json

exec
/bin/zsh -lc "pwd && rg --files -g 'results.md' -g 'decisions.md' -g 'innovation_brainstorm.md' -g 'story.md' -g 'design.md' -g 'monitor.md' .. 2>/dev/null | sed -n '1,80p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/rr_logs
 succeeded in 0ms:
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/rr_logs
../design.md
../monitor.md

codex
这个沙箱只暴露了 `experiments/cargo_cvpb` 这一小段，顶层 `experiments/results.md` 等不在可读根里；我会基于这里的设计/日志和联网文献核查继续。
exec
/bin/zsh -lc "sed -n '1,260p' ../monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/rr_logs
 succeeded in 0ms:
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

exec
/bin/zsh -lc "sed -n '1,220p' ../design.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/rr_logs
 succeeded in 0ms:
# 实验 CVPB: CARGO 航拍-地面 跨视角原型桥接 + 局部证据匹配

> 战略重思后选定(5 死角后, 3 codex strat 强收敛)。★★方法论转向: **放弃"验证 confound 是真依赖"(杀光一切), 改 empirical 涨点导向**——在 CARGO baseline 上试 +module 能否 beat = 真 kill-switch。这是 167 篇库真实配方(命名 gap + 2 模块 + 多数据集 SOTA 表 + framing)。

## 动机
- CARGO 航拍-地面 baseline(resnet50 BoT)mean A↔G mAP **32.48%**, VDT SOTA ~50% → **大 headroom**。AGPReID 活赛道(VDT CVPR24 / SeCap CVPR25 / GSAlign NeurIPS25 / ViSA CVPR26), 范式都是"命名视角/几何/语义 gap + 2 模块 + CARGO/AG-ReID 表", **不证 confound 依赖**。
- ★framing(避开那堵墙): **不是"模型被 view confound 骗"**(那已被 trained 模型 handle)——而是 **aerial-ground 单张图的身份证据在两视角下不一一对应**, 这是 ID loss/triplet 的全局监督约束不到的真实 gap(局部对应/跨视角证据补全)。

## 核心假设(empirical, 非 confound)
全局 cosine 之外加 (1) 跨视角原型对齐 + (2) 局部 token 集合匹配, 能补全单图缺失的对侧视角证据 → mAP 涨。**判据 = 涨点, 不是 confound 真伪**。

## 技术方案
- **模块1 OVP-Mem(Opposite-View Prototype Memory)**: 每 train pid 维护 aerial / ground 两个 EMA 原型。样本除 CE+triplet 外, aerial image 拉近同 pid 的 **ground** 原型(InfoNCE: sim(z, P[y, opp_view])/τ 分类到 y)。≠死掉的 CV-triplet(那是 batch 内跨视角正对稀缺假设, 已证 88% batch 有; 这里是**全局原型级跨视角监督, 覆盖 hard tail**)。
- **模块2 Local Token MaxSim**: feature map 切 K 个局部 evidence token, 测试 `global cosine + β·双向 MaxSim(tokens)`, 训练 token-level ID/SupCon + diversity 防坍缩。**不强制同部位对齐**(non-correspondence set matching)。≠GSAlign(无 TPS warp)/ ViSA(无 expert graph)/ SeCap(无 prompt)。

## ★ kill-switch(empirical, 分阶段, 现成 baseline)
1. **零训练 token-MaxSim probe**(最便宜, 先跑): CARGO baseline checkpoint 抽 feature map, 测 `global cosine` vs `global + 局部 MaxSim hybrid` 的 A↔G mAP。**+0.5~1.0 mAP** = 局部匹配有料 → 继续。
2. **OVP-Mem only 训练**: ep10/20 vs baseline 同 epoch, **≥+1.0** 继续。
3. **OVP+Local 训练**: final **≥35-36 mAP**(baseline 32.48)→ 进方法稿; 接近 38+ → 扩 Swin/VDT 级主表。
- **不涨 = 换 module**(empirical, 不纠结 confound 真伪)。这是新 approach 的核心: 直接试涨点。

## 对照 / 消融
baseline 32.48 → +OVP-Mem → +Local-MaxSim → +both; K(token 数) / β / τ 敏感性; EMA momentum。

## novelty 切开(查重)
- VDT(view decoupling + orthogonal): 我们不做 view-related/unrelated 减法解耦。
- SeCap(adaptive prompt + local refine): 不做 prompt recalibration。
- GSAlign(LTPS 几何 warp + visibility mask): 不做 TPS / 不叫 visibility mask。
- ViSA(view-aware expert graph): 不做 expert/graph。
- SD-ReID(diffusion 生成 all-view): 不用 diffusion。
- 我们的: **cross-view local evidence densification via opposite-view prototype + non-correspondence set matching**。

## 数据 / venue
- CARGO(lab-3090 现成)。补 AG-ReID.v2 多数据集表。
- venue: AGPReID 活, ICME/ACCV/BMVC 稳, 强则冲 CVPR-tier 子方向。

## ★ kill-switch 进度(2026-06-22)
- **#1 零训练 MaxSim probe: 弱 PASS +0.86**(8×4 grid β=1.0 → 33.34 vs baseline 32.48; pipeline 验证 global cosine 精确复现 32.48; 增益集中 aerial-query A→G 32.90→34.18 **+1.28**, G→A 平~32)。局部 token 集合匹配有料但弱。
- **双审**: Claude review 无 Critical, **H1 训练动力学已修**(λ warmup `ovp_warmup=10` + inited 列数日志, 防冷启动梯度尖峰), 审查通过。⚠️**H2/M5 novelty 撞车**: OVP-Mem(对侧视角原型+InfoNCE)≈ **PDPA(2025 同 CARGO)/ CMPC(CVPR22)/ MBCE(AAAI23 VI-ReID)** → **OVP 当组件不当 headline, 用 Local-MaxSim 集合匹配(non-correspondence)差异化**。Codex review 进行中。
- **#2 OVP-Mem 训练完成: 🎯 final 50.11 mAP(R1 54.93)@ep60, baseline 32.48 → +17.6, 达 VDT SOTA~50!** 轨迹 28.35→36.32→39.18→43.83→48.76→50.11。在 resnet50 BoT(弱于 VDT 的 ViT)上做到 SOTA 量级。★**方向铁证金矿**(代码双审无泄漏: 原型训练期/eval 测试 pid 不相交; +17.6 达 SOTA 待用户多 seed 复核)。但 OVP 撞 CMPC → **headline 走 OVLI**(见下), OVP 降为强 ablation。

## 备选(此线不涨则转)
- strat_3: Camera Residual TTA(source-free open-camera test-time adaptation, Market→MSMT/CARGO OOD, 贴 DART3/TEMP)——OOD 方向, 也绕开那堵墙。
- strat_1: CARGO View-Conditioned Token Evidence Matching(和本方案 module2 同源)。

---

# OVLI: Opposite-View Late-Interaction Evidence Alignment(★headline, 5-codex 收敛)

## 为什么从 OVP 转 OVLI
- **OVP-Mem 的硬伤是 novelty 撞车**(claude/codex 双审都标 High): per-view EMA prototype + opposite-view InfoNCE ≈ **CMPC(CVPR22)/MBCE(AAAI23)/PDPA(2025 同 CARGO)** 近乎逐字同构, 当 headline 过不了 novelty review。
- **但 OVP 训练验证了方向有料**: baseline 32.48 → OVP ep30 **39.18(+大)**, 证明"跨视角身份证据对齐"有 **BIG headroom**(不是 confound 噪声, 是真信号)。
- **OVLI 保留方向、换掉撞车的机制**: 把 per-pid prototype contrast 改成 **token-set late-interaction(ColBERT/MaxSim 风格)的 sample-to-sample 跨视角检索 loss**——无 prototype/memory/EMA, 直接局部 token 集合**部分匹配**。

## framing(差异化 headline)
**跨视角身份证据是"部分 token-set 匹配"问题, 不是"全局原型对齐"问题。** 航拍-地面单图在两视角下**无 1-1 部位对应**(俯视看头肩/平视看全身)。全局原型(OVP/CMPC)对"缺失区域"一律惩罚, 把不可对应的部分也算进相似度; **partial MaxSim 让"能对上的 token"承担相似度, 对不上的不拖累**——这正是 late-interaction 的 retrieval 语义, 契合跨视角证据稀疏/不对齐的本质。

## 核心假设
全局 cosine 之外, 加 **opposite-view sample-to-sample 的 (global cos + 双向 MaxSim) 混合检索 loss**(supervised-contrastive), 让 encoder 学到"对侧视角下能局部匹配上的身份证据" → mAP 涨。判据 = 涨点。

## 技术方案(已实现, `afd_train.py --ovli`)
1. **token 抽取(复用 maxsim_probe 配方)**: hook `model.layer4`(GeM 前 spatial map, 16×8)→ `adaptive_avg_pool2d` 到 grid(默认 8×4=32 token)→ flatten → **新增 1×1 conv proj 到 256d + 逐 token L2-norm**。
   - ★**proj 是新可学参数**, `OVLIHead.proj`(Conv2d 2048→256), **已加进 optimizer**(`list(model.parameters())+list(ovli.parameters())`, 且有 assert 自检 proj 在 optimizer.param_groups 里)。这是与 OVP(无新参数)的关键结构差异。
   - hook **不 detach**, 梯度回流 layer4→proj。
2. **opposite-view retrieval loss**(`OVLIHead.loss`): batch 内, anchor(view v)的 positive = 同 pid 的 **opposite-view(1-v)** 样本, negative = 其它 pid 的 opposite-view 样本; **同视角样本完全排除出候选**(纯跨视角目标)。
   - `score(i,j) = α·cos(g_i,g_j) + (1-α)·sym_MaxSim(tok_i,tok_j)`, sym_MaxSim = 双向 mean-max `0.5*(mean_u max_s + mean_s max_u)`(对称)。α=0.5。
   - 多正样本: **logsumexp supervised-contrastive**(`L_i = -logsumexp(score(i,pos)/τ) + logsumexp(score(i,cand)/τ)`, τ=0.05), 对"有 ≥1 opp 正 且 ≥1 opp 负"的 anchor 求均值。**无 memory/EMA/prototype。**
3. **★H1 教训(从 OVP 继承)**: `--ovli_warmup`(默认 10)对 λ 线性 warmup, 防随机 proj 早期梯度尖峰。epoch 日志 `OVLI[lam_eff loss pos neg gap]` 监控塌缩/过强。
4. **train/test 对称 + AMP 安全**:
   - OVLI 是训练期 loss; eval 默认 **global-only 不变**(精确 == baseline)。`--ovli_rerank` 可选额外报 `global + MaxSim` rerank(global 与 rerank 两个数都打印), 用与训练**同一套** proj token + 双向 MaxSim。
   - OVLI loss 在 `autocast(enabled=False)` 内走**真 fp32**(cos/MaxSim/logsumexp 在 τ=0.05 下要 fp32; proj 也在 fp32 跑), 修了 OVP 审查里"注释说 fp32 但其实在 autocast 内"的 M1。
   - `--ovli` off 精确复现 baseline(`ovli=None`, OVLIHead 不构造, optimizer 只含 model params)。
   - `--ovp` 与 `--ovli` 互斥(两个不同跨视角机制, 不混跑混淆消融)。

## kill-switch / 判据(empirical)
- **#2′ OVLI only 训练**(GPU 空出后): ep10/20/30 vs baseline 同 epoch。OVP ep30 到 39.18, OVLI 至少要进同量级才说明 late-interaction 机制不输 prototype。
- final **≥35-36 mAP**(baseline 32.48)→ 进方法稿; 接近 OVP 的 39+ 且 novelty 干净 → 扩 Swin/VDT 级主表 + AG-ReID.v2 多数据集。
- 消融: α(global vs MaxSim 权重)/ τ / grid(token 数 K)/ λ / proj_dim; OVLI vs OVP 同设置对比(证 late-interaction ≥ prototype 且 novelty 更干净)。
- **不涨 = 换 module**。

## novelty 切开(查重)
- vs OVP/CMPC/MBCE/PDPA: **无 prototype/memory/EMA**, sample-to-sample late-interaction, 这正是绕开 OVP 撞车的点。
- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
- vs GSAlign(TPS warp + visibility): **不强制部位对应**(non-correspondence partial set matching), 无几何 warp、无 visibility mask。
- 我们的: **cross-view identity evidence as opposite-view late-interaction (partial token-set) retrieval, learned as a training objective**。

## ★★ OVLI 训练结果(2026-06-22): 45.19 mAP, headline 成立
| 方法 | mean A↔G mAP | vs baseline | novelty |
|---|---|---|---|
| baseline(resnet50 BoT) | 32.48 | — | — |
| **OVLI(headline, late-interaction)** | **45.19**(A→G 49.21 / G→A 41.16, R1 51.06) | **+12.7** | ★无 exact prior |
| OVP(ablation, prototype) | 50.11 | +17.6 | 撞 CMPC |
| **OVP+OVLI(full model)** | **52.14**(rerank 52.71, R1 57.74) | **+19.7** | OVLI 互补 prototype **+2.0 over OVP** |
- **★组合 52.14 > OVP 50.11(+2.0)= OVLI(late-interaction)与 OVP(prototype)互补不冗余**。干净消融故事: 我们提出 OVLI(novel), 它给已知 prototype 对齐加东西, full model 达 52.14。组合轨迹 29.73→29.69(plateau)→41.51→49.15→50.92→52.14(LR 衰减后大涨)。
- 轨迹: 14.91(ep10)→24.24(ep20)→...→45.19(ep60)。rerank(global+MaxSim)≈global(45.17), 收敛后 global 已够, MaxSim rerank 早期 +3 后期收敛。
- **OVLI(novel)+12.7 显著 beat baseline = headline 成立**。< OVP(50.11)4.9 但 OVP 撞 CMPC。OVLI A→G(49.21)≈OVP, G→A(41.16)弱 → α 偏 aerial-query(与 kill-switch #1 一致, MaxSim 信号集中 A→G)。
- **方法稿骨架**: headline=OVLI(opposite-view late-interaction, novel, 45.19); 强 ablation=OVP(prototype, 50.11 但撞 CMPC); baseline 32.48; + Swin/SOLIDER port 冲 SOTA + AG-ReID.v2 多数据集主表 + 消融。
- **下一步**: ① **OVLI+OVP 组合**(改互斥→both, 看是否 >50, 组合当 headline+OVLI novel 成分); ② OVLI α/τ/grid 调(close G→A gap); ③ Swin port; ④ 多数据集。

## ★ OVLI novelty 评估(5 codex 深查, 2026-06-22)
**总评: B 类方法稿 headline 可行, novelty 中偏强(~3.5-4/5), 无 exact prior(ovlinov_1 确认), 非致命撞车, 但非"发明 late-interaction"式突破。**
- **立得住**: 完整组合无 exact prior(person ReID/aerial-ground + opposite-view-only sample-to-sample + token-set MaxSim + supervised contrastive + 无 prototype/memory + 测试 global-only)。真空白 = 训练期 opposite-view-only 晚交互 loss + 测试 global-only。
- **撞车风险(必切开)**: ★AlignedReID/Learning-by-Aligning(最大, ReID 早有"训练期局部对齐、测试 global"→ 强调 OVLI 无部位对应); ColBERT/FILIP/ColPali(写"inspired by", 绝不"we propose late interaction"); CM-EMD/G2DA/CVFT(OT, OVLI 无 transport plan); DTST(token selection vs pairwise 跨视角证据)。
- **★技术硬伤(必修)**: "dustbin/对不上不拖累"夸大——现 sym_MaxSim 对全 32 token 取 mean, 未匹配 token 仍拉低分。→ (a)改表述"减弱非对应惩罚"不说"discarded"; 或 **(b)真做 dustbin(null token/top-k/thresholded MaxSim)= AG-ReID 特有设计, 修硬伤+冲更高 venue+可能涨 G→A**。
- **framing 收窄**: 重心 "training-time MaxSim 新" → "**AGPReID 的 opposite-view partial evidence supervision 新**"。headline = "Opposite-View Partial Evidence Learning for Aerial-Ground Person ReID via training-only late interaction"。
- **SOTA 别乱说**: OVLI 45.19 非 CARGO SOTA(GSAlign 61.55/ViSA +10.06), 但超 VDT 42.76/DTST 43.39 这代。写"resnet50/global-only 设定下大幅超 baseline, 达/超 VDT/DTST 量级"。
- **必做消融**: global-oppview-SupCon vs +OVLI / oppview-only vs all-view / test global vs +MaxSim-rerank / MaxSim vs OT/top-k/avg / vs AlignedReID / α-τ-grid sweep / token-match 可视化(航拍头肩→地面上身/背包)/ AG-ReID.v2 跨数据集。
- **下一步**: ① dustbin/top-k MaxSim 变体(修硬伤+AG-ReID 特有设计); ② 必做消融; ③ 组合结果; ④ Swin port + 多数据集。

## 代码审查 / 验证(2026-06-22)
- 已写 `OVLIHead`(token proj + 双向 MaxSim + opp-view supcon loss)+ `ovli_rerank_eval`(eval 期 global vs global+MaxSim 双报)。
- **本地隔离 numeric smoke test(导入仓库真实 OVLIHead, 非副本)全过**: token shape (64,32,256) 逐 token L2-norm; sym_MaxSim 对称、self 对角≈1; 正常 batch loss 有限>0 且梯度回流 **proj.weight + global feat**; all-same-view batch loss=0 不崩; 某 pid 无 opp 正样本时该 anchor 被排除、loss 仍有限、grad 有限; fp16 cached map→fp32 token; AdamW 实际推动 proj 权重。
- ast.parse + py_compile 通过。待 codex 双审 → GPU 空出后训练(当前 GPU 被 OVP 占)。

## ★★★ Swin backbone eval mAP=0.03 诊断+修复(2026-06-23)

**问题**: `cvpb_swin_ovli`(Swin-Small + OVLI, 复用 resnet50 配方 AdamW lr=3.5e-4 均一)→ eval 跨视角 mAP=0.03(≈随机), 而 resnet50 同配方 eval 正常(52.37)。

**诊断结论(不是 eval-path bug, 是训练塌缩)**:
- 失败 log 显示 **ep1-7 训练健康**(Acc 0.003→0.472, CE 7.8→3.5, OVLI gap +0.32), **ep8 Iter50 一步塌缩**(LR 升过 2.46e-4): Loss 4.16→10.36, Acc→0.01, OVLI pos≈neg≈0.49。`model_best.pth` 是 ep10(唯一 eval, 已塌)。
- `diag_swin_eval.py`(fresh model): eval 特征**正常**(8 真实图 final off-diag cos +0.24, finite, unit-norm)→ forward / `.cuda()` semantic-weight / LayerNorm / 取 tensor 路径全对。
- `diag_swin_ckpt.py`(塌缩 ckpt): `outs[-1]` off-diag cos **+0.992**, batch-chan-std 0.038(健康 2.67)= backbone 对所有输入近常数; global_feat +0.9995; 权重全 finite 无 NaN = 表征塌缩非数值溢出。

**根因**: resnet50 调出的峰值 LR 3.5e-4 AdamW 均一施加到 ~50M 参 SOLIDER Swin transformer 过大, warmup 升过 ~2.5e-4 时几步大更新把 backbone 推进常数输出退化吸引子。仓库主 SOLIDER config 训 Swin 用 SGD 8e-4 + 20ep warmup(对 transformer 温和得多)。

**修复(只动 Swin 路径)**:
- `afd_train.py`: backbone=swin_small 时 Swin backbone 单独 param-group LR×`--swin_lr_factor`(默认 0.1), heads/BNNeck/OVLI proj 保持 full LR。resnet50 字节级不变(走 else)。
- `swin_transformer.py` L1400: `w.cuda()`→`w.to(x.device)`(鲁棒性, 非根因)。
- Claude broad review: APPROVE(参数不漏不重, 冻结 BNNeck.bias 正确排除, WarmupCosineLR per-group base_lr 正确, resnet50 路径不变)。
- 验证: `diag_swin_fix`(swin_lr_factor=0.1, 14ep)看是否平稳过 ep8。

exec
/bin/zsh -lc "for f in rr_od_stage*.json rr_market_stage*.json; do printf '%s\\n' \""'$f"; sed -n '"'1,220p' \""'$f"; done' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/rr_logs
 succeeded in 0ms:
rr_od_stage0.json
{
  "dataset": "occluded_duke",
  "cheap_stage": 0,
  "ri_k": 20,
  "cheap_mAP": 0.26967981387290885,
  "full_mAP": 73.05459974540081,
  "stage_mAP": {
    "0": 0.26967981387290885,
    "1": 0.2341854815159438,
    "2": 47.6957874045896,
    "3": 72.7479409798522
  },
  "cheap_compute": 0.08333333333333333,
  "cum_frac": [
    0.08333333333333333,
    0.16666666666666666,
    0.9166666666666666,
    1.0
  ],
  "testA": {
    "overlap": {
      "rho": 0.12433339032036753,
      "p": 0.000999000999000999
    },
    "rbo": {
      "rho": 0.10024263936104356,
      "p": 0.000999000999000999
    },
    "tau": {
      "rho": 0.11447935701593517,
      "p": 0.000999000999000999
    }
  },
  "testB_proxy_rhos": {
    "cheap_margin(neg)": -0.0058923010980024134,
    "cheap_top1_top3_gap(neg)": 0.004167585927812569,
    "cheap_entropy": 0.012057446788221323,
    "cheap_density(neg)": 0.029608770340416878,
    "cheap_rawnorm(neg)": 0.0583328292735614,
    "full_feat_norm(neg)": 0.032939067382407365,
    "full_margin(neg)": 0.4046931714584715,
    "full_entropy": 0.31222223521417775
  },
  "testB_best_proxy": "full_margin(neg)",
  "testB_best_cheap_proxy": "cheap_rawnorm(neg)",
  "testB_RI_marginal": {
    "rbo": 0.10024263936104356,
    "overlap": 0.12433339032036753,
    "tau": 0.11447935701593517
  },
  "testB_partial_all": {
    "overlap": 0.1382627607930267,
    "rbo": 0.0957069007214258,
    "tau": 0.12755987991969062
  },
  "testC": {
    "overlap": {
      "multi_rho": 0.3225260565040244,
      "multi_R2": 0.1261644782448409
    },
    "rbo": {
      "multi_rho": 0.40292307381575443,
      "multi_R2": 0.2384543839747144
    },
    "tau": {
      "multi_rho": 0.36643310228074605,
      "multi_R2": 0.17708163358276996
    }
  },
  "testD_table": {
    "0.5": {
      "ri_oracle": 34.53527394840068,
      "ri_deploy": 33.580805003554715,
      "apgap_deploy": 34.28575171791251,
      "cheapstat": 34.40982596567249,
      "static_oracle": 38.33044775770232,
      "random": 33.393464573603474
    },
    "0.6": {
      "ri_oracle": 42.14783209554095,
      "ri_deploy": 41.196752846117846,
      "apgap_deploy": 42.41270273881461,
      "cheapstat": 42.51967034150518,
      "static_oracle": 47.04933890232007,
      "random": 41.32807333151514
    },
    "0.7": {
      "ri_oracle": 50.41750765847296,
      "ri_deploy": 49.28856474962177,
      "apgap_deploy": 50.38454777503664,
      "cheapstat": 50.38374342072741,
      "static_oracle": 54.89146417612618,
      "random": 49.32134720956014
    }
  },
  "testD_compute99": {
    "ri_oracle": 1.0,
    "ri_deploy": 1.0,
    "apgap_deploy": 1.0,
    "static_oracle": 1.0,
    "random": 1.0
  },
  "base_full_map": 73.05459974540081,
  "verdict": {
    "B_alive": false,
    "C_alive": true,
    "D_alive": false,
    "OVERALL_alive": false
  }
}rr_od_stage1.json
{
  "dataset": "occluded_duke",
  "cheap_stage": 1,
  "ri_k": 20,
  "cheap_mAP": 0.2341854815159438,
  "full_mAP": 73.05459974540081,
  "stage_mAP": {
    "0": 0.26967981387290885,
    "1": 0.2341854815159438,
    "2": 47.6957874045896,
    "3": 72.7479409798522
  },
  "cheap_compute": 0.16666666666666666,
  "cum_frac": [
    0.08333333333333333,
    0.16666666666666666,
    0.9166666666666666,
    1.0
  ],
  "testA": {
    "overlap": {
      "rho": 0.1386506849048745,
      "p": 0.000999000999000999
    },
    "rbo": {
      "rho": 0.12320146778573893,
      "p": 0.000999000999000999
    },
    "tau": {
      "rho": 0.13893393263485523,
      "p": 0.000999000999000999
    }
  },
  "testB_proxy_rhos": {
    "cheap_margin(neg)": -0.01090546618129326,
    "cheap_top1_top3_gap(neg)": 0.006783573458255331,
    "cheap_entropy": 0.0539364526928372,
    "cheap_density(neg)": 0.013204137720612148,
    "cheap_rawnorm(neg)": 0.16134679060406718,
    "full_feat_norm(neg)": 0.03390794498248202,
    "full_margin(neg)": 0.4026614728392236,
    "full_entropy": 0.31353953093544223
  },
  "testB_best_proxy": "full_margin(neg)",
  "testB_best_cheap_proxy": "cheap_rawnorm(neg)",
  "testB_RI_marginal": {
    "rbo": 0.12320146778573893,
    "overlap": 0.1386506849048745,
    "tau": 0.13893393263485523
  },
  "testB_partial_all": {
    "overlap": 0.1595679673579215,
    "rbo": 0.14002585648125482,
    "tau": 0.16660340637659396
  },
  "testC": {
    "overlap": {
      "multi_rho": 0.3681364997734976,
      "multi_R2": 0.14414017881894214
    },
    "rbo": {
      "multi_rho": 0.45928857718547195,
      "multi_R2": 0.2590725579520119
    },
    "tau": {
      "multi_rho": 0.40497413789123693,
      "multi_R2": 0.18475400753204652
    }
  },
  "testD_table": {
    "0.5": {
      "ri_oracle": 31.305002326535092,
      "ri_deploy": 29.152504984706155,
      "apgap_deploy": 31.47272955383324,
      "cheapstat": 31.73184519116595,
      "static_oracle": 33.776315736792775,
      "random": 29.362607764606544
    },
    "0.6": {
      "ri_oracle": 39.421156613669645,
      "ri_deploy": 37.92603154079424,
      "apgap_deploy": 40.091184108350966,
      "cheapstat": 40.53401516381509,
      "static_oracle": 43.58077496244324,
      "random": 38.1275473518323
    },
    "0.7": {
      "ri_oracle": 48.51307518909174,
      "ri_deploy": 47.09828497537966,
      "apgap_deploy": 48.774124931289236,
      "cheapstat": 48.990276875186225,
      "static_oracle": 52.51920965757579,
      "random": 46.91819032335606
    }
  },
  "testD_compute99": {
    "ri_oracle": 1.0,
    "ri_deploy": 1.0,
    "apgap_deploy": 1.0,
    "static_oracle": 1.0,
    "random": 1.0
  },
  "base_full_map": 73.05459974540081,
  "verdict": {
    "B_alive": false,
    "C_alive": true,
    "D_alive": false,
    "OVERALL_alive": false
  }
}rr_od_stage2.json
{
  "dataset": "occluded_duke",
  "cheap_stage": 2,
  "ri_k": 20,
  "cheap_mAP": 47.6957874045896,
  "full_mAP": 73.05459974540081,
  "stage_mAP": {
    "0": 0.26967981387290885,
    "1": 0.2341854815159438,
    "2": 47.6957874045896,
    "3": 72.7479409798522
  },
  "cheap_compute": 0.9166666666666666,
  "cum_frac": [
    0.08333333333333333,
    0.16666666666666666,
    0.9166666666666666,
    1.0
  ],
  "testA": {
    "overlap": {
      "rho": 0.5057154346930269,
      "p": 0.000999000999000999
    },
    "rbo": {
      "rho": 0.3932575988424618,
      "p": 0.000999000999000999
    },
    "tau": {
      "rho": 0.483299961903615,
      "p": 0.000999000999000999
    }
  },
  "testB_proxy_rhos": {
    "cheap_margin(neg)": -0.2517850576772276,
    "cheap_top1_top3_gap(neg)": -0.2783577466089863,
    "cheap_entropy": 0.41992852808110526,
    "cheap_density(neg)": 0.34850266092836296,
    "cheap_rawnorm(neg)": -0.18137214541047372,
    "full_feat_norm(neg)": -0.1556041963005892,
    "full_margin(neg)": -0.2690359348308651,
    "full_entropy": -0.18867796305495302
  },
  "testB_best_proxy": "cheap_entropy",
  "testB_best_cheap_proxy": "cheap_entropy",
  "testB_RI_marginal": {
    "rbo": 0.3932575988424618,
    "overlap": 0.5057154346930269,
    "tau": 0.483299961903615
  },
  "testB_partial_all": {
    "overlap": 0.21525195392175894,
    "rbo": 0.04457446658183941,
    "tau": 0.15518558421902184
  },
  "testC": {
    "overlap": {
      "multi_rho": 0.7709992287197598,
      "multi_R2": 0.5781559129194805
    },
    "rbo": {
      "multi_rho": 0.781109617795917,
      "multi_R2": 0.5848779317721972
    },
    "tau": {
      "multi_rho": 0.8266727252599161,
      "multi_R2": 0.6690775962342264
    }
  },
  "testD_table": {
    "0.937": {
      "ri_oracle": 57.710806746180694,
      "ri_deploy": 58.19571376594348,
      "apgap_deploy": 58.819666480276695,
      "cheapstat": 56.643272648389235,
      "static_oracle": 56.643272648389235,
      "random": 53.87616721959038
    },
    "0.958": {
      "ri_oracle": 65.60281040532149,
      "ri_deploy": 66.29267658274178,
      "apgap_deploy": 66.58013225337231,
      "cheapstat": 65.21482425166779,
      "static_oracle": 65.21482425166779,
      "random": 60.26385754578142
    },
    "0.98": {
      "ri_oracle": 70.71620546082278,
      "ri_deploy": 71.30742022692844,
      "apgap_deploy": 71.50284880480037,
      "cheapstat": 71.23825811633127,
      "static_oracle": 71.23825811633127,
      "random": 67.01631134921264
    }
  },
  "testD_compute99": {
    "ri_oracle": 0.9958521870286576,
    "ri_deploy": 0.9875188536953243,
    "apgap_deploy": 0.9875188536953243,
    "static_oracle": 0.9916666666666667,
    "random": 1.0
  },
  "base_full_map": 73.05459974540081,
  "verdict": {
    "B_alive": false,
    "C_alive": true,
    "D_alive": false,
    "OVERALL_alive": false
  }
}rr_market_stage1.json
{
  "dataset": "market1501",
  "cheap_stage": 1,
  "ri_k": 20,
  "cheap_mAP": 1.211649980534184,
  "full_mAP": 94.60864731131805,
  "stage_mAP": {
    "0": 1.1905089273591405,
    "1": 1.211649980534184,
    "2": 91.01223453057254,
    "3": 94.34284639568683
  },
  "cheap_compute": 0.16666666666666666,
  "cum_frac": [
    0.08333333333333333,
    0.16666666666666666,
    0.9166666666666666,
    1.0
  ],
  "testA": {
    "overlap": {
      "rho": 0.2031915329538129,
      "p": 0.000999000999000999
    },
    "rbo": {
      "rho": 0.1020144944736142,
      "p": 0.000999000999000999
    },
    "tau": {
      "rho": 0.18544849958733284,
      "p": 0.000999000999000999
    }
  },
  "testB_proxy_rhos": {
    "cheap_margin(neg)": -0.01568734551836736,
    "cheap_top1_top3_gap(neg)": 0.008458994789933085,
    "cheap_entropy": 0.09305736396812006,
    "cheap_density(neg)": -0.09646221035589943,
    "cheap_rawnorm(neg)": -0.13295999868169342,
    "full_feat_norm(neg)": -0.04194886790073165,
    "full_margin(neg)": 0.018594152618641654,
    "full_entropy": -0.33493538256399863
  },
  "testB_best_proxy": "full_entropy",
  "testB_best_cheap_proxy": "cheap_rawnorm(neg)",
  "testB_RI_marginal": {
    "rbo": 0.1020144944736142,
    "overlap": 0.2031915329538129,
    "tau": 0.18544849958733284
  },
  "testB_partial_all": {
    "overlap": 0.18786451780286167,
    "rbo": 0.12303791484033463,
    "tau": 0.19074565812635583
  },
  "testC": {
    "overlap": {
      "multi_rho": 0.4398853147898693,
      "multi_R2": 0.19290192930666616
    },
    "rbo": {
      "multi_rho": 0.605979600744641,
      "multi_R2": 0.3829976960649015
    },
    "tau": {
      "multi_rho": 0.5167025130792864,
      "multi_R2": 0.2700822456284381
    }
  },
  "testD_table": {
    "0.5": {
      "ri_oracle": 38.428622620830296,
      "ri_deploy": 38.07239087953966,
      "apgap_deploy": 38.45265581314161,
      "cheapstat": 38.41220614990269,
      "static_oracle": 38.99325575165197,
      "random": 38.59528479837198
    },
    "0.6": {
      "ri_oracle": 49.687735470611145,
      "ri_deploy": 49.34691500207099,
      "apgap_deploy": 49.766370540469374,
      "cheapstat": 49.68192977414362,
      "static_oracle": 50.52411714822923,
      "random": 49.803721577309304
    },
    "0.7": {
      "ri_oracle": 61.031511180038386,
      "ri_deploy": 60.704382742395495,
      "apgap_deploy": 60.86407371805945,
      "cheapstat": 60.868898828856615,
      "static_oracle": 61.839026360571246,
      "random": 61.000663951022645
    }
  },
  "testD_compute99": {
    "ri_oracle": 1.0,
    "ri_deploy": 1.0,
    "apgap_deploy": 1.0,
    "static_oracle": 1.0,
    "random": 1.0
  },
  "base_full_map": 94.60864731131805,
  "verdict": {
    "B_alive": false,
    "C_alive": true,
    "D_alive": false,
    "OVERALL_alive": false
  }
}rr_market_stage2.json
{
  "dataset": "market1501",
  "cheap_stage": 2,
  "ri_k": 20,
  "cheap_mAP": 91.01223453057254,
  "full_mAP": 94.60864731131805,
  "stage_mAP": {
    "0": 1.1905089273591405,
    "1": 1.211649980534184,
    "2": 91.01223453057254,
    "3": 94.34284639568683
  },
  "cheap_compute": 0.9166666666666666,
  "cum_frac": [
    0.08333333333333333,
    0.16666666666666666,
    0.9166666666666666,
    1.0
  ],
  "testA": {
    "overlap": {
      "rho": 0.12828894638202504,
      "p": 0.000999000999000999
    },
    "rbo": {
      "rho": 0.11368931812983313,
      "p": 0.000999000999000999
    },
    "tau": {
      "rho": 0.18099193909132083,
      "p": 0.000999000999000999
    }
  },
  "testB_proxy_rhos": {
    "cheap_margin(neg)": -0.10703024346492915,
    "cheap_top1_top3_gap(neg)": -0.14449216064121845,
    "cheap_entropy": 0.4248492600664739,
    "cheap_density(neg)": 0.16983114261992732,
    "cheap_rawnorm(neg)": 0.02551145219451821,
    "full_feat_norm(neg)": -0.06367865862073564,
    "full_margin(neg)": -0.15877133710473743,
    "full_entropy": 0.12986941321293444
  },
  "testB_best_proxy": "cheap_entropy",
  "testB_best_cheap_proxy": "cheap_entropy",
  "testB_RI_marginal": {
    "rbo": 0.11368931812983313,
    "overlap": 0.12828894638202504,
    "tau": 0.18099193909132083
  },
  "testB_partial_all": {
    "overlap": -0.03644189702560067,
    "rbo": -0.057295299447207806,
    "tau": -0.044444186186075776
  },
  "testC": {
    "overlap": {
      "multi_rho": 0.5182769941464733,
      "multi_R2": 0.265340057927107
    },
    "rbo": {
      "multi_rho": 0.37293030534856064,
      "multi_R2": 0.14283535532052583
    },
    "tau": {
      "multi_rho": 0.5422563204336238,
      "multi_R2": 0.2923935541627226
    }
  },
  "testD_table": {
    "0.937": {
      "ri_oracle": 92.45407041581852,
      "ri_deploy": 92.96792880237712,
      "apgap_deploy": 93.15690826267817,
      "cheapstat": 93.02052208915512,
      "static_oracle": 93.02052208915512,
      "random": 91.90353088580764
    },
    "0.958": {
      "ri_oracle": 93.44302090895819,
      "ri_deploy": 94.05742365630101,
      "apgap_deploy": 94.08764855035156,
      "cheapstat": 94.0498936820823,
      "static_oracle": 94.0498936820823,
      "random": 92.78898932681166
    },
    "0.98": {
      "ri_oracle": 94.11621574429209,
      "ri_deploy": 94.49025292868441,
      "apgap_deploy": 94.50450140244374,
      "cheapstat": 94.47800739217128,
      "static_oracle": 94.47800739217128,
      "random": 93.7424661224685
    }
  },
  "testD_compute99": {
    "ri_oracle": 0.9666716152019001,
    "ri_deploy": 0.9499950514647664,
    "apgap_deploy": 0.9499950514647664,
    "static_oracle": 0.9541765637371338,
    "random": 0.9791666666666667
  },
  "base_full_map": 94.60864731131805,
  "verdict": {
    "B_alive": false,
    "C_alive": true,
    "D_alive": false,
    "OVERALL_alive": false
  }
}
codex
本地日志能确认两条重要背景：旧 retrieval-reliability 路线在 deploy 策略上被静态/随机基线吃掉，但 evidence/oracle 方向确实留下了很大的可恢复空间。接下来我会把联网撞车核查集中在“多证据 query 端如何可部署化”上。
web search: 
web search: 2024 2025 person re-identification multi-query set-based tracklet aggregation uncertainty reliability calibration
web search: 
web search: site:openaccess.thecvf.com 2024 person re-identification uncertainty reid
web search: 
web search: "set-based" "person re-identification" 2025
web search: 
web search: https://arxiv.org/abs/2405.01101
web search: 
web search: https://arxiv.org/abs/1911.09143
web search: 
web search: "Uncertainty-aware multi-shot knowledge distillation" ReID
web search: 
web search: https://arxiv.org/abs/2311.03828
web search: 
web search: "multi-query" "person re-identification" "Market-1501" feature fusion
web search: 
web search: site:openaccess.thecvf.com "video-based person re-identification" "CVPR 2024"
web search: 
web search: 2024 "video-based person re-identification" transformer "arXiv" "MARS"
web search: 
web search: "retrieval reliability" "person re-identification"
web search: 
web search: https://arxiv.org/abs/2311.10605
web search: 
web search: k-reciprocal re-ranking person re-identification 2017 CVPR
codex
**结论先行**  
能做 CCF-B，但必须把题目定义成 **query-side evidence-sufficient ReID / budgeted multi-evidence query retrieval**，而不是“单图失败预测”或“普通 multi-query ReID”。我给这条线 **7/10**。如果第 2 份证据只能靠测试标签 oracle 取同 ID 图，而没有 tracklet / 多帧 query 来源，信心直接降到 **3/10**，那就是 exp109 的 query-side oracle 变体。

**最推荐的方法稿方向**  
方法名可以先叫：**Fragility-Guided Evidence Union, FGEU**。

核心设定：

给定一个 query packet `Q={q1,...,qm}`，这些图来自同一 camera tracklet / 同一次人工查询 / 同一目标短时跟踪结果。gallery 仍是标准跨相机 gallery。方法不是“凭空补全单图”，而是回答：

> 当前 query 证据是否足够？如果不够，怎样从同一 query tracklet 中选择、校准、融合额外证据，使检索恢复？

具体机制：

1. **lowtail-pos / dispersion 只做问题定义和训练标签，不直接当部署变量**  
   因为它们依赖跨相机真 positives。论文里应写成：它们定义了 fragile query / evidence insufficiency；部署时预测的是 **marginal evidence gain residual**。

2. **训练/校准目标不要是“失败概率”**  
   目标应是：
   ```text
   residual_gain(q, e)
   = ΔAP(q ∪ e) - h(margin, #false, hubness, static confidence)
   ```
   也就是“加这份 query-side 证据后，扣掉 trivial 排名因素后，还能带来多少恢复”。这正好接住你已有的 partial control 结果。

3. **融合用 evidence union，不用简单平均**  
   一个稳的公式：
   ```text
   S(Q, g) =
     (1 - α(Q)) * mean_i s(q_i, g)
     + α(Q) * logsumexp_i( w_i * calibrated_s(q_i, g) / τ )
   ```
   `α(Q)` 随 fragility / dispersion / predicted evidence gain 增大。  
   `w_i` 由 query packet 内一致性、tracklet 质量、rank-list agreement、hubness penalty 决定。  
   随机异 ID 图必须被 gate 掉或降权，否则方法不成立。

4. **方法输出允许是 action policy**  
   单图 fragile 时，不要假装能恢复；输出“证据不足，请求同 tracklet 额外帧”。拿到额外帧后再 union。这个比 reliability calibration 强，因为它真的恢复 ranking。

**新问题 / 协议**  
建议定义：

**Evidence-Sufficient Multi-Query ReID** 或 **Budgeted Query-Evidence ReID**。

协议要避免 oracle 漏洞：

1. query 初始为 1 张图。
2. 额外证据只能来自同 camera、同 tracklet、同人工查询包，不能从跨相机 gallery 按 ID 抽。
3. 预算 `B=1,2,4,8`。
4. 主指标：
   - standard mAP / R1
   - fragile subset mAP
   - `Recovery@B = (AP_B - AP_1) / (AP_oracle - AP_1)`
   - contamination robustness：插入随机异 ID 帧后 mAP 不应崩
   - evidence-cost AUC：恢复多少性能花了多少 query 帧
5. 必做 baseline：
   - single query
   - mean fusion / max fusion / logsumexp fusion
   - best-frame oracle
   - k-reciprocal / CA-Jaccard re-rank
   - UFFM/AMC 类 test-time multi-view fusion
   - set-based quality attention
   - tracklet average / temporal aggregation

**撞车核查**  
主要撞车点如下：

- **Set-based ReID 已经存在**。ID-aware Quality for Set-based ReID 明确就是 set matching，并用 feature fusion attention 忽略低质量图像、聚合判别信息。你不能把“用多张图”当 novelty。边界必须是 evidence insufficiency 诊断 + marginal evidence gain + budgeted recovery protocol。来源：arXiv 1911.09143。  
  https://arxiv.org/abs/1911.09143

- **UFFM/AMC 2024 很危险**。它也是多视角 feature fusion，而且是 inference-time、无需 fine-tuning，还在 Occluded-Duke 等数据上报大增益。区别应写清楚：UFFM 从 embedding 近邻里伪造 multi-view evidence；你的方法只接受 query-side verified evidence，并显式做随机异 ID harm / contamination gate。来源：arXiv 2405.01101。  
  https://arxiv.org/abs/2405.01101

- **MVI²P 也很近**。它利用同 ID 多图整合 occluded 信息，再蒸馏回单图；但它承认测试阶段没有多张同 ID 图，所以走 single-image distillation。你要反过来：真实部署 query packet 可得时，不蒸馏，不 hallucinate，只做 evidence-conditioned retrieval。来源：arXiv 2311.03828。  
  https://arxiv.org/abs/2311.03828

- **UMTS / multi-shot KD 更早已做**。multi-shot teacher-student + uncertainty 不是新点。不要写“first to use multiple shots”。  
  https://arxiv.org/abs/2001.05197

- **TEMP 2024 是 uncertainty/reliability 类邻居**。它用 re-id entropy 做 test-time similarity modification，处理 temporal distribution shift。你的 reliability 分数若只预测“这个 query 靠不靠谱”，会被这类工作压住；必须强调 action: request/select/fuse evidence。  
  https://arxiv.org/abs/2403.14114

- **k-reciprocal / CA-Jaccard 是 re-ranking 上限对照**。它们只改 gallery-neighborhood context，不能创造新的 query identity evidence。你已有 oracle union 是 k-reciprocal 增益 3.3x，这点很强。  
  https://arxiv.org/abs/1701.08398  
  https://arxiv.org/abs/2311.10605

**Verdict**  
最该做：**evidence-aware multi-query / tracklet query fusion**，但 headline 不能是 multi-query，而是：

> Single-image ReID fails when query evidence is insufficient; query-side verified evidence can recover it, and the key is to predict marginal evidence gain and union evidence without admitting false evidence.

不建议做纯 retrieval reliability calibration。它最多是辅助模块，撑不起方法稿。

CCF-B 可行条件：

1. 在 video/tracklet 或严格 query-packet 协议上复现 oracle headroom。
2. FGEU 明显超过 mean/max/logsumexp fusion、UFFM/AMC、set-quality attention、k-reciprocal。
3. random-ID contamination 不崩，最好能自动拒绝。
4. fragile subset 恢复显著，全量 mAP 也不掉。
5. 明确承认 image-only Occluded-Duke 的第 2 同 ID 图是 oracle diagnostic，不把它包装成部署方法。
tokens used
92,000
**结论先行**  
能做 CCF-B，但必须把题目定义成 **query-side evidence-sufficient ReID / budgeted multi-evidence query retrieval**，而不是“单图失败预测”或“普通 multi-query ReID”。我给这条线 **7/10**。如果第 2 份证据只能靠测试标签 oracle 取同 ID 图，而没有 tracklet / 多帧 query 来源，信心直接降到 **3/10**，那就是 exp109 的 query-side oracle 变体。

**最推荐的方法稿方向**  
方法名可以先叫：**Fragility-Guided Evidence Union, FGEU**。

核心设定：

给定一个 query packet `Q={q1,...,qm}`，这些图来自同一 camera tracklet / 同一次人工查询 / 同一目标短时跟踪结果。gallery 仍是标准跨相机 gallery。方法不是“凭空补全单图”，而是回答：

> 当前 query 证据是否足够？如果不够，怎样从同一 query tracklet 中选择、校准、融合额外证据，使检索恢复？

具体机制：

1. **lowtail-pos / dispersion 只做问题定义和训练标签，不直接当部署变量**  
   因为它们依赖跨相机真 positives。论文里应写成：它们定义了 fragile query / evidence insufficiency；部署时预测的是 **marginal evidence gain residual**。

2. **训练/校准目标不要是“失败概率”**  
   目标应是：
   ```text
   residual_gain(q, e)
   = ΔAP(q ∪ e) - h(margin, #false, hubness, static confidence)
   ```
   也就是“加这份 query-side 证据后，扣掉 trivial 排名因素后，还能带来多少恢复”。这正好接住你已有的 partial control 结果。

3. **融合用 evidence union，不用简单平均**  
   一个稳的公式：
   ```text
   S(Q, g) =
     (1 - α(Q)) * mean_i s(q_i, g)
     + α(Q) * logsumexp_i( w_i * calibrated_s(q_i, g) / τ )
   ```
   `α(Q)` 随 fragility / dispersion / predicted evidence gain 增大。  
   `w_i` 由 query packet 内一致性、tracklet 质量、rank-list agreement、hubness penalty 决定。  
   随机异 ID 图必须被 gate 掉或降权，否则方法不成立。

4. **方法输出允许是 action policy**  
   单图 fragile 时，不要假装能恢复；输出“证据不足，请求同 tracklet 额外帧”。拿到额外帧后再 union。这个比 reliability calibration 强，因为它真的恢复 ranking。

**新问题 / 协议**  
建议定义：

**Evidence-Sufficient Multi-Query ReID** 或 **Budgeted Query-Evidence ReID**。

协议要避免 oracle 漏洞：

1. query 初始为 1 张图。
2. 额外证据只能来自同 camera、同 tracklet、同人工查询包，不能从跨相机 gallery 按 ID 抽。
3. 预算 `B=1,2,4,8`。
4. 主指标：
   - standard mAP / R1
   - fragile subset mAP
   - `Recovery@B = (AP_B - AP_1) / (AP_oracle - AP_1)`
   - contamination robustness：插入随机异 ID 帧后 mAP 不应崩
   - evidence-cost AUC：恢复多少性能花了多少 query 帧
5. 必做 baseline：
   - single query
   - mean fusion / max fusion / logsumexp fusion
   - best-frame oracle
   - k-reciprocal / CA-Jaccard re-rank
   - UFFM/AMC 类 test-time multi-view fusion
   - set-based quality attention
   - tracklet average / temporal aggregation

**撞车核查**  
主要撞车点如下：

- **Set-based ReID 已经存在**。ID-aware Quality for Set-based ReID 明确就是 set matching，并用 feature fusion attention 忽略低质量图像、聚合判别信息。你不能把“用多张图”当 novelty。边界必须是 evidence insufficiency 诊断 + marginal evidence gain + budgeted recovery protocol。来源：arXiv 1911.09143。  
  https://arxiv.org/abs/1911.09143

- **UFFM/AMC 2024 很危险**。它也是多视角 feature fusion，而且是 inference-time、无需 fine-tuning，还在 Occluded-Duke 等数据上报大增益。区别应写清楚：UFFM 从 embedding 近邻里伪造 multi-view evidence；你的方法只接受 query-side verified evidence，并显式做随机异 ID harm / contamination gate。来源：arXiv 2405.01101。  
  https://arxiv.org/abs/2405.01101

- **MVI²P 也很近**。它利用同 ID 多图整合 occluded 信息，再蒸馏回单图；但它承认测试阶段没有多张同 ID 图，所以走 single-image distillation。你要反过来：真实部署 query packet 可得时，不蒸馏，不 hallucinate，只做 evidence-conditioned retrieval。来源：arXiv 2311.03828。  
  https://arxiv.org/abs/2311.03828

- **UMTS / multi-shot KD 更早已做**。multi-shot teacher-student + uncertainty 不是新点。不要写“first to use multiple shots”。  
  https://arxiv.org/abs/2001.05197

- **TEMP 2024 是 uncertainty/reliability 类邻居**。它用 re-id entropy 做 test-time similarity modification，处理 temporal distribution shift。你的 reliability 分数若只预测“这个 query 靠不靠谱”，会被这类工作压住；必须强调 action: request/select/fuse evidence。  
  https://arxiv.org/abs/2403.14114

- **k-reciprocal / CA-Jaccard 是 re-ranking 上限对照**。它们只改 gallery-neighborhood context，不能创造新的 query identity evidence。你已有 oracle union 是 k-reciprocal 增益 3.3x，这点很强。  
  https://arxiv.org/abs/1701.08398  
  https://arxiv.org/abs/2311.10605

**Verdict**  
最该做：**evidence-aware multi-query / tracklet query fusion**，但 headline 不能是 multi-query，而是：

> Single-image ReID fails when query evidence is insufficient; query-side verified evidence can recover it, and the key is to predict marginal evidence gain and union evidence without admitting false evidence.

不建议做纯 retrieval reliability calibration。它最多是辅助模块，撑不起方法稿。

CCF-B 可行条件：

1. 在 video/tracklet 或严格 query-packet 协议上复现 oracle headroom。
2. FGEU 明显超过 mean/max/logsumexp fusion、UFFM/AMC、set-quality attention、k-reciprocal。
3. random-ID contamination 不崩，最好能自动拒绝。
4. fragile subset 恢复显著，全量 mAP 也不掉。
5. 明确承认 image-only Occluded-Duke 的第 2 同 ID 图是 oracle diagnostic，不把它包装成部署方法。
