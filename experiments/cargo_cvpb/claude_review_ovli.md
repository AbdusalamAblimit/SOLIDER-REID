# Claude Broad Review — CVPB OVLI 启动前广范围审查

**日期**: 2026-06-22
**审查对象**: `experiments/cargo_cvpb/{design.md(OVLI 段 L46-90), afd_train.py(OVLIHead + ovli_rerank_eval + 训练循环 OVLI 分支 + optimizer 构造 + argparse), maxsim_probe.py}` + 对照 `experiments/afd_reid/{afd_train.py, afd_model.py, cargo_dataset.py}` + 同文件 `OVPMemory`(diff 出 OVLI 结构差异)
**审查方法**: 主 agent 逐行 + 5 处 torch/numpy 本地数值验证(loss 边界 / 梯度回流 fp16-map→fp32-proj→encoder / 共享 bn 混精单 backward / floor NaN 安全 / import 解析 / eval 缩放一致性) + 静态切片确认 10 项关键不变量
**结论**: **审查通过**(可启动 OVLI 训练)。无 Critical，无 High。OVLI 的载荷设计(proj 进 optimizer、梯度回流 encoder、双向对称 MaxSim、opp-view 多正 logsumexp、AMP fp32 隔离、--ovli off 精确复现 baseline、互斥、eval 默认 global-only)全部逐行 + 数值核实正确。仅余若干 Medium/Low（不阻断，记录备查 + 进稿前/扩展前处理）。

---

## 总评

OVLI 实现相对 OVP **结构正确性更难**(因为引入了新可学参数 proj + hook 捕获 fp16 map 跨 autocast 边界做 fp32 backward)，但实现把这些难点都处理到位了：

1. **新参数 proj 真进 optimizer 且有 assert 自检**(L559-572)——这是 design 反复强调的"与 OVP 的关键结构差异"，核实无误：`--ovli` 时 `opt_params = list(model.parameters()) + list(ovli.parameters())`，且 `assert proj_in`。`--ovli` off 时 `ovli=None`、OVLIHead 根本不构造、optimizer 只含 model params。
2. **梯度回流路径核实(本地 torch 微测)**：hook 存 `out` 不 detach → `tokens_from_cached_map()` 把 fp16 cached map `.float()` 后过 proj(fp32)→ loss.backward() 后 **proj.weight.grad、layer4.weight.grad、input.grad 全部非零**。即 OVLI 既学 proj 又把跨视角监督回灌进 backbone，这正是"训练期 loss 让 encoder 内化局部可匹配证据"的机制要求。
3. **共享 bn 的混精单 backward(本地 torch 微测)**：`bn` 同时进 autocast 内的 CE/triplet 路径与 autocast-disabled 的 OVLI fp32 路径，单次 `scaler.scale(loss).backward()` **无 double-backward / retain_graph 报错**，proj 与 backbone 梯度都到，loss 有限。
4. **边界全过(numpy 复算 loss)**：all-same-view→loss=0 不崩；单 pid(无 opp 负)→loss=0；某 pid 只在一个视角(无 opp 正)→该 anchor 被 `valid` 排除、其余正常；sym_MaxSim 自对角≈1 且对称。
5. **eval 默认 == baseline**：default eval 走 `run_cross_view_eval`(global-only，与 baseline byte-identical 同一份 import)；`--ovli_rerank` 才额外报 global+MaxSim 双数，opt-in。train/test 对称。

下面分级列出（所有级别按协议全列）。

---

## Critical

无。(无崩溃、无静默错误结果、无破坏 baseline 可复现性的问题。)

---

## High

无。(OVP 审查里的 H1「冷启动零梯度尖峰」在 OVLI 不复现——OVLI 无 prototype/EMA/冷启动稀疏，loss 从第 1 个 batch 起就在真实 token 上算；λ warmup 仍保留作随机 proj 早期平滑。novelty 撞车在 OVLI 段已被 framing 切开，见 M3，且属方法稿层面非代码阻断。)

---

## Medium

### M1. λ warmup 起点不是 0，epoch 1 已是 0.05·λ（与 OVP 同；非 bug，记录）
- **位置**: `afd_train.py` L589 `ovli_lambda_eff = ovli_lambda * min(1, epoch/ovli_warmup)`；epoch 循环 L581 `range(1, epochs+1)`。
- **事实**: epoch 从 1 起，故 epoch1 的 `ovli_lambda_eff = 0.5*min(1,1/10) = 0.05`（不是 0）。本地复算确认。即随机初始化的 proj 在 epoch1 已注入 10% 权重的 OVLI 梯度，不是"前若干 epoch 完全冻结只填东西"。
- **风险**: 低。proj 用 kaiming，token L2-norm 后 score∈[-1,1]、logits∈[-20,20]，0.05·λ 的早期梯度可控（且 backbone 主梯度来自 CE+triplet）。OVP 同样从 epoch1=0.05 起且 ep30 跑到 39.18，无早期爆炸先例。
- **建议**: 不阻断。若想要真正的"前 N epoch λ=0"，把 warmup 改成 `max(0, (epoch-1)/ovli_warmup)` 或加 `--ovli_freeze_epochs`。当前可接受，记录以免日后对 λ 标定困惑。

### M2. 训练 token 用 fp16 cached map，rerank-eval token 用 fp32 map → 训/评 MaxSim 空间有微小精度差
- **位置**: 训练 L602-631（forward 在 autocast 内→cached map fp16→`tokens_from_cached_map` `.float()`）vs `ovli_rerank_eval` extract L371-372（eval 无 autocast wrapper→cached map 原生 fp32→同 `.float()` 无损）。
- **事实**: 两边都最终在 fp32 跑 proj+MaxSim，但训练侧的输入 map 是 fp16 量化过的、eval 侧是 fp32。token 数值有 ~1e-3 级差异。
- **风险**: 极低且方向无害——eval 更精确；且 `--ovli_rerank` 只是 opt-in 诊断报数，主 eval 是 global-only。训练 loss 用 fp16 map 是有意（复用 autocast forward 已算好的 map，省一次 fp32 forward）。
- **建议**: 不阻断。若要训/评完全同分布，可在训练侧也对 layer4 做一次 enabled=False 的 fp32 重算（成本高，不值）。记录即可。

### M3. design OVLI novelty 切开比 OVP 干净，但仍需进稿前坐实「late-interaction-as-training-loss」的先例查重
- **位置**: `design.md` L81-85。
- **事实**: OVLI 相对 OVP 的撞车面**显著缩小**——去掉 prototype/memory/EMA 后，CMPC/MBCE/PDPA 的「per-id per-modality 原型 + momentum」结构不再适用。design 的三条切开(vs OVP/CMPC「无 prototype」、vs ColBERT/MaxSim-rerank「是训练 loss 非 test rerank」、vs GSAlign「无 TPS warp/visibility」)**框架上站得住**。
- **残余风险**: 「把 MaxSim/late-interaction 当**训练监督**而非 test rerank」这一点是 OVLI 的真 novelty 支点，但"token-set/部分匹配 contrastive 训练 loss"在 fine-grained retrieval / 局部对齐 ReID(如 part-based supcon、token-level alignment) 里有相邻工作。本地无法联网，**这条必须由 Codex `--search` 联网坐实**(是否有人已把对称 MaxSim 当跨视角训练 loss)。
- **建议**: 不阻断 empirical 训练(design 判据=涨点)。但进方法稿/扩 Swin·VDT 主表前，Codex 联网查重 + 在 design 显式记录"late-interaction-as-cross-view-training-objective"的先例边界。符合 CLAUDE.md 创新门槛。

### M4. import 解析依赖「以脚本名 afd_train.py 从 cargo_cvpb/ 运行」(继承 OVP M4)
- **位置**: 顶部 L107-114 `sys.path.insert('../afd_reid')` + `from afd_train import (...)`；`ovli_rerank_eval` 内 L363 `from afd_train import build_eval_loader`、L407 `from afd_train import eval_market`、L428 `from maxsim_probe import eval_from_distmat`、L357 `from cargo_dataset import filter_by_view`。
- **本地实测**(`importlib.util.find_spec`，模拟脚本运行 + sys.path 插入)：
  - `afd_train` → `afd_reid/afd_train.py`(✓ 含 build_eval_loader/eval_market/WarmupCosineLR/run_cross_view_eval/print_eval/set_seed)。因 cvpb 的 afd_train.py 以 `__main__` 运行、不占用 `afd_train` 模块名，故 import 命中 afd_reid 版本，**正确**。
  - `maxsim_probe` → `cargo_cvpb/maxsim_probe.py`(✓ 含 eval_from_distmat)。
  - `cargo_dataset` → `afd_reid/cargo_dataset.py`(✓)。
- **风险**: 当前文档命令(`cd cargo_cvpb && python3 afd_train.py`)**安全**。但若有人 `python -m`、或当模块 import 本文件，`from afd_train import` 有撞回自身(循环 import / AttributeError: build_eval_loader 不存在于 cvpb afd_train)的风险——尤其 `ovli_rerank_eval` 里这些 import 是**运行到 eval 时才触发**(epoch 10 才第一次跑)，万一解析错会在训练 10 epoch 后才炸。
- **建议**: 不阻断(文档命令正确)。低成本加固：把 eval 里的延迟 import 提到文件顶部(和现有顶部 import 一致解析)，或在 run 命令旁强调"必须 cd cargo_cvpb"。

### M5. CARGO PK 采样不保证每 batch 同 pid 跨双视角，OVLI 有效 anchor 比例可能偏低（数据动力学，非 bug）
- **位置**: `cargo_dataset.py` `RandomIdentitySampler`(按 pid 采 K=4 实例，**不按 view 分层**)；OVLI `valid` 要求 anchor 有「≥1 opp 正 且 ≥1 opp 负」。
- **事实**: 若某 pid 的 4 个采样实例恰好同视角(或对侧视角 pid 在 batch 内不足)，该 pid 的 anchor 进不了 `valid`。CARGO 训练集 aerial/ground 分布不均时，早期 epoch 有效 anchor 数可能波动。loss 仍正确(对 valid 求均值，无 valid 则 0)，但**有效监督密度**取决于数据。
- **监控**: 已有 `OVLI[lam_eff loss pos neg gap]` 日志，但**未打印每 epoch 平均有效 anchor 数 / 有效 pair 数**——若 loss 长期≈0 会分不清"已学好"还是"几乎没有效 anchor"。
- **建议(非阻断, 建议)**: 在 epoch summary 增打"本 epoch 平均 valid-anchor 比例"或"有 opp-pos 的 pid 数"，对上 CLAUDE.md「日志够重，能观察模块塌缩/过弱」铁律。当前 pos/neg/gap 三数已能间接判过强/塌缩，故仅 Medium-建议而非 High-强制(与 OVP H1 不同：OVP 有结构性退化零梯度的明确隐患，OVLI 没有，只是数据相关的密度波动)。

---

## Low

### L1.（已确认正确）proj 进 optimizer + assert 自检
`opt_params = list(model.parameters()) + list(ovli.parameters())`(L559-561)，assert `proj_in`(L570) 用 `id(p)` 集合核对 proj 两个 tensor(weight+bias)都在 param_groups。静态切片确认。OVP 路径不加 ovli.parameters()(ovp 无可学参数)。

### L2.（已确认正确）梯度回流 fp16-map→fp32-proj→encoder（本地 torch 微测）
hook 存 `out` 不 detach(L240)；`tokens_from_cached_map` 对 fp16 cached map `.float()`(L261) 后过 proj，在 `autocast(enabled=False)`(L627) 内。backward 后 proj.weight.grad / layer4.weight.grad / input.grad 全非零。fp16→fp32 cast 保留 autograd 图，无断图。

### L3.（已确认正确）共享 bn 混精单 backward 无 double-backward（本地 torch 微测）
`bn` 进 CE/triplet(autocast 内)与 OVLI(autocast disabled，`bn.float()`)两路，单次 `scaler.scale(loss).backward()` 不报 retain_graph，两路梯度都到。

### L4.（已确认正确）sym_MaxSim 对称 + 多正 logsumexp 是标准 SupCon-out 形（numpy 复算）
`sym_maxsim_matrix` 用 `flat@flat.t()` reshape(B,K,B,K)，`i2j=max(dim=3).mean(dim=1)`、`j2i=max(dim=1).mean(dim=2)`，`0.5*(i2j+j2i)` 构造上对称(实测 `allclose(M,M.t())`)，自对角≈1。loss 分子=logsumexp(pos)、分母=logsumexp(cand=pos∪neg)，即 Khosla `L=-log[Σpos exp / Σall exp]` 的 SupOut 形，正确。

### L5.（已确认正确）floor=-1e4 NaN 安全（本地 torch 微测）
score∈[-1,1]→logits=score/0.05∈[-20,20]，floor=-1e4。混合 real+floor 行 logsumexp=最大 real(floor 项 exp 下溢≈0)；全 floor 行(仅非 valid anchor，已被 mask 排除前)也给有限值(-9998)。`-inf-(-inf)=nan` 路径被 floor + valid-mask 双重排除。

### L6.（已确认正确）--ovli off 精确复现 baseline
`ovli=None`(L550)、OVLIHead 不构造、hook 不挂、optimizer 只 model params、OVLI loss 块 `if args.ovli:` 整体跳过、eval 走 global-only `run_cross_view_eval`。CE/Triplet/Warmup/eval/set_seed 从 `afd_reid/afd_train.py` import 同一份。baseline 可复现不破坏。

### L7.（已确认正确）--ovp/--ovli 互斥
L505-507 `if args.ovp and args.ovli: raise SystemExit`。两机制不混跑，消融不混淆。

### L8.（已确认正确）eval 缩放一致，无 double-*100（本地复算）
`ovli_rerank_eval` 内 global 用 `eval_market`(返回 mAP∈[0,1] + cmc 数组)→存时 `gmap*100, gcmc[0]*100`(L431)；rerank 用 `eval_from_distmat`(已返回 *100 的 mAP/R1)→直接存(L432)。两路缩放各自正确，无重复 *100。

### L9.（已确认正确）token 抽取复用 maxsim_probe 配方一致
训练/rerank 的 token 抽取(hook layer4 → `adaptive_avg_pool2d` 到 grid → flatten → proj → 逐 token L2-norm)与 maxsim_probe 的 zero-train probe 配方同源(差异仅 OVLI 多了可学 proj，probe 用 raw 2048-d token)。grid 默认 8×4=32 token，与 probe PASS 的 8×4 一致。

### L10.（已确认正确）hook 生命周期
hook 在 OVLIHead 构造时挂(L236)，训练全程保留(eval forward 也需它填 map)，仅在训练结束 L716 `ovli.remove_hook()`。无悬挂/无提前移除。eval 期 model.eval() 但 proj 是纯 Conv2d(无 BN/dropout)，train/eval 模式对其无影响。

### L11.（潜在，仅记录）rerank-eval 把全 gallery token 载入 CPU RAM
`extract()` 把 gallery token `.cpu()` 累积：A→G gallery=32268 → 32268×32×256×4B ≈ **1.06 GB** CPU(31GB 机器可容)；`maxsim_block` budget=80M floats、gblk 自适应(A→G gblk≈583，块≈320MB GPU)。本地复算确认不 OOM。但比 OVP(无 token)多吃 ~1GB RAM + eval 变慢(每 10 epoch 一次)。可接受，记录。

### L12.（潜在，仅记录）diagnostics ps/ns 用 `.any()` 守卫
L339-340 `score[pos].mean() if pos.any() else 0`——当 `valid.sum()>0` 时 pos/neg 必非空，守卫是冗余防御，无害。零 loss 路径 L318-320 返回三个 `new_zeros(())`，meters 累加 `float(0)*bs=0`，不崩(本地确认)。

### L13.（未来 DDP 风险，仅记录，继承 OVP L7）
单 `device='cuda'`、无 DataParallel/DDP。OVLI 有可学 proj——若未来上 DDP 需确保 proj 参数被 DDP 包裹/grad-sync(OVLIHead 是独立 nn.Module，DDP 需同时 wrap model+ovli 或合并)。当前单卡正确，仅未来告警。

---

## 启动前清单（按严重度）

1. **无 Critical / 无 High 阻断项** → **可直接启动 OVLI 训练**(empirical kill-switch #2′，design 判据=ep10/20/30 趋势 + final ≥35-36 进稿)。
2. **[M3]** 进方法稿/扩主表前：Codex `--search` 联网坐实「late-interaction-as-cross-view-training-loss」novelty 边界，design 显式记录。— **进稿前**。
3. **[M5 建议]** 建议加"每 epoch 平均 valid-anchor 比例"日志(分清 loss≈0 是学好还是无有效 anchor)；当前 pos/neg/gap 已能间接监控，故非强制。
4. **[M1][M2][M4][L11] 记录备查**：warmup 起点 0.05 非 0；train(fp16-map)/rerank(fp32-map) MaxSim 微差；import 解析依赖脚本运行(文档命令安全，建议把 eval 内延迟 import 提顶部);rerank 多吃 ~1GB RAM。

**审查结论：审查通过。** 无 Critical、无 High。OVLI 载荷设计(proj 进 optimizer 且自检、fp16-map→fp32-proj→encoder 梯度回流、共享 bn 混精单 backward、双向对称 MaxSim + opp-view 多正 logsumexp SupCon、AMP fp32 隔离、floor NaN 安全、--ovli off 精确复现 baseline、ovp/ovli 互斥、eval 默认 global-only + rerank opt-in、边界不崩)逐行 + 5 处本地数值核实**全部正确**。M3(novelty 联网坐实)进稿前处理，不阻断 empirical 训练。可启动 kill-switch #2′ → 交 Codex 第二轮独立审查。

（≥30 行，满足 hook 阻断条件。）
