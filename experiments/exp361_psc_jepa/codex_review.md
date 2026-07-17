# Codex Review — exp361 PSC-JEPA（纯 codex 三审，2026-06-26）

审查对象：`psc_jepa_pretrain.py`（PSC-JEPA continued-pretrain stage-A，SOLIDER swin_tiny backbone 训练）。
纯 codex 三审制（用户 2026-06-26 指令，省 claude token，替代 Opus Agent broad review）。
前置：Opus 三审已留档 `claude_review.md`（抓 4 bug：C1 坍缩 / H1 ckpt 断链 / H2 semantic / var 标定），codex 三审独立交叉验证。原始三轮输出见 `codex_review_r{1,2,3}.md`。

## Round 1 (23:39) — Verdict: approve
无 Critical/High。Findings：
- Medium：body group bbox 重叠污染 `L_visible_anchor`（torso/arm/leg 共享肩髋）。Stage-A 可接受，Stage-B 改互斥 mask 或 anchor 只池化未遮区。
- Medium：Stage-A novelty 不够（接近 I-JEPA + HAP/PersonMAE），定位骨架/对照，novelty 靠 Stage-B support bank。
- Low：smoke 尾批 B=1 触发 predictor BatchNorm1d 报错（主训练 drop_last B=64 不受影响）。
逐项 a-f 全通过（坍缩 / 坐标系 / swin featmap / part_pool / EMA / ckpt 往返）。

## Round 2 (23:47) — Verdict: approve
无 Critical/High。R1 Low（smoke B<2）修复成立。Findings：
- Medium：bbox 重叠（持续，Stage-B 处理）。
- Low：`B<2 continue` 正确；极端 `--smoke 1`/空数据 0 update 仍存 ckpt → 建议 update guard。
逐项复核全通过 + codex 查 `pose_dataset.py` 确认 `kp[:,0]=x`/visibility 0-1 score + `swin_transformer.py` 确认 forward `(global, outs)`。

## Round 3 (23:52) — Verdict: approve（最终轮）
无 Critical/High/Medium 阻断。两个 Low（smoke B<2 + update guard `and last`）修复正确，无全量训练副作用。
逐项结论全通过：坍缩（predictor + EMA stop-grad + var-reg×√C 成立，tokStd 坍缩警戒，teacher EMA 不构成新增坍缩）/ 坐标（ow,oh→[0,1]→12×4 grid，x/y/clip 正确）/ swin（out[1][-1]=[B,768,12,4]）/ 数值维度（part_pool/einsum/denom.clamp/EMA/predictor reshape 安全）/ ckpt（`backbone.` 前缀往返匹配下游 init_weights）。
仅保留运行前置提醒（swin_tiny.pth / pose_train.npz / pose 为原 crop 像素 (x,y) + visibility 0-1 score，已确认 4090 就位）。

## 三轮逐项核对汇总（a-f，三轮一致通过）

- **(a) 表征坍缩**：predictor（Linear-BN-GELU-Linear，student-only）+ EMA stop-grad teacher + var-reg（Svis std×√C，gvis 掩码）= BYOL/SimSiam/VICReg 组合，三轮一致判足够 Stage-A 防坍缩；teacher EMA 跟随 student 不引入新坍缩；tokStd（√C 标定 healthy≈1，<0.5 警戒）作坍缩 kill-switch，建议补 effective-rank。
- **(b) 坐标系**：keypoints 原 crop 像素 (x,y) normalize by ow,oh → [0,1] → 12×4 grid，x→GW/y→GH 映射 + clip 边界正确（codex R2 查 pose_dataset.py:376/468 确认 kp[:,0]=x、visibility 0-1 score）。
- **(c) Swin featmaps**：SOLIDER SwinTransformer.forward 返回 (global, outs)，outs[-1]=[B,768,12,4]，fwd_tokens 的 out[1]→[-1] 解析正确（codex R2/R3 查 swin_transformer.py:1396 确认）。
- **(d) part_pool/EMA/predictor/数值**：einsum('bchw,bghw->bgc')、denom.clamp_min(1)、零 group gvis 排除、EMA 参数更新、predictor 两次 reshape、BatchNorm1d（主训练 B=64 稳定）三轮通过。
- **(e) ckpt 往返**：保存 `{'state_dict': {'backbone.'+k: v}}`，与下游 swin init_weights 剥 backbone. 前缀 + make_model self.base.init_weights 路径匹配（H1 修复闭合）。
- **(f) novelty**：Stage-A = same-image latent JEPA 骨架/对照，不单独主张 novelty；B 类靠 Stage-B support bank（对照 PersonMAE/HAP/I-JEPA/PersonViT）。

Opus 前置三审已抓并修：C1 坍缩（加 predictor+var-reg）/ H1 ckpt 断链（backbone. 前缀）/ H2 semantic_weight（0.2）/ var 监控标定（√C+gvis 掩码）。codex 三审独立交叉验证全部修复正确。

> 注：smoke 实跑（codex/Opus 审逻辑未覆盖运行时）另抓到 SOLIDER SwinTransformer `.train()/.eval()` 不返回 self → 链式调用得 None，已修（分开调用，非机制改动）。

## 结论
**codex 三审全 approve（R1 / R2 / R3）**。Stage-A 通过，可启动 continued-pretrain。
Novelty 边界：Stage-A = same-image EMA latent JEPA 骨架/稳定性强对照，**不单独主张 novelty**；B 类 novelty 靠 Stage-B 的 pseudo same-ID support bank + pose-defined latent support completion，并用 PersonMAE/HAP/I-JEPA 类路线作对照。
**codex 审查通过**（verdict approve ×3）。

## Stage-A 训练 + Stage-B 路线（codex/Opus 一致）

- **Stage-A 跑法**：4090 主跑 SOLIDER swin_tiny continued-pretrain，pose-defined partial-view latent JEPA，50-100 ep。训练监控 `tokStd`（健康 >0.5，<0.5 坍缩警戒）+ `cosDrop`（不应快速→1，否则坍缩）+ `var`。
- **Stage-A 对照（novelty 生命线，必跑）**：3090 same-image full teacher（无 partial-drop）/ 5060Ti-1 random-mask latent / 5060Ti-2 PersonMAE-lite。主跑必须赢过这三个对照才证 partial-view + pose-defined 的价值。
- **Stage-B（真 B 类 novelty）**：加 pseudo same-ID support bank `T_bank` + `L_solider_anchor` 防遗忘 + pose-defined latent support completion（不补像素补身份证据）。
- **kill-switch（codex）**：continued-pretrain → fine-tune Occluded-Duke **≥+0.7 mAP**，且 plain/random/same-image 对照不同涨，pseudo bank top-k 精度过线。
- **参考先例**：PersonMAE(2311.04496) / HAP(2310.20695) / PersonViT(2408.05398) / I-JEPA(2301.08243) / SimSiam(2011.10566) / VICReg(2105.04906) / BYOL(2006.07733)。

原始三轮 codex 完整输出：`codex_raw_r{1,2,3}.md`。审查日期 2026-06-26 23:39–23:52，纯 codex 三审制（用户指令，省 claude token，替代 Opus Agent）。

## Stage-B 防遗忘 codex 三审（2026-06-27，frozen SOLIDER anchor + global GAP distillation）

针对 Stage-A forgetting（fine-tune 41 << plain ~55），加防遗忘：frozen SOLIDER anchor + L_solider_anchor（part token 局部 + global GAP 全局判别几何蒸馏）。

- **R1 (needs-attention)**：防遗忘只锚 5 part token 覆盖窄，Stage-A 坏的是全局 backbone 判别几何。建议补 global GAP / dense distillation。
- **R2 (approve)**：已补 global GAP（student full-view global 蒸 frozen SOLIDER global，梯度边界正确，anchor no-grad / student full-forward 有 grad）。修到 R1 核心。Medium：+1 forward 显存（smoke 验 OOM）；拆 sol_p/sol_g 日志。
- **R3 (approve, 最终)**：global GAP 梯度边界 + fwd_tokens unpack（out[0]=GAP global / out[1][-1]=part featmap）+ L_part+L_glob 覆盖局部+全局 + sol_p/sol_g 监控全通过。Medium：+1 student full forward 显存（smoke 必验，OOM 则 AMP/checkpoint 不改 batch）。

**结论（codex 守诚实）**：Stage-B 防遗忘 = **修复尝试成立**，**非"forgetting 已解决"**（重训 + fine-tune 验证 PSC-JEPA 能否从 41 追到 plain ~55 后才知）。codex 三审 R2/R3 approve，可 smoke → 重训。原始 `codex_raw_stageb_r{1,2,3}.md`。
