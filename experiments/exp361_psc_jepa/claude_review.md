# Claude Broad Review — exp361 PSC-JEPA Stage-A (`psc_jepa_pretrain.py`)

**Reviewer**: Claude (Opus, 独立 broad review)
**Date**: 2026-06-26
**Round**: v1
**Scope**: `psc_jepa_pretrain.py` 全文逐行 + `design.md` + 对照 `model/backbones/swin_transformer.py`、`model/make_model.py`、`config/defaults.py`、`datasets/pose_dataset.py`
**结论先行**: **需修复后重审**。坐标系 / swin shape 解析 / masked-pool / EMA 逻辑都正确，但存在 1 个 Critical（表征坍缩机制缺失）+ 3 个 High（checkpoint 存盘格式与本仓 loader 不兼容、semantic_weight 与下游不一致、pose npz 前置缺失），任何一个都可能让这次多天 continued-pretrain 白跑或下游加载即崩。

---

## 逐项检查（对应审查要点）

### 1. JEPA 表征坍缩（最关键）— 不通过
- 当前 loss 全是 `1 - cos(student_token, EMA_teacher_token)`，三项（L_jepa / L_anchor / L_union）都把 student token 直接拉向 teacher token。
- **没有 predictor / projector 头**，没有 variance-covariance 正则（VICReg），没有 centering+sharpening（DINO），没有 target normalization（data2vec），没有负样本。
- 这是 **BYOL / SimSiam 去掉 predictor** 的退化配置。SimSiam(CVPR21) 与 BYOL 系列消融均证实：predictor 是防坍缩的核心，去掉 predictor 必坍缩（L2-normalized 特征 std→0、loss→最小）。I-JEPA / data2vec（与本设计最像：EMA full teacher + masked student 预测 latent）都带 predictor/regression head 且对 target 做归一化——本脚本两者皆无。
- 全局常数解（backbone 输出常向量 → 每个 part token normalize 成同一单位向量 → 所有 cos=1 → loss=0）是可达最小值，EMA+stop-grad 单独**挡不住**。
- **监控反向误导**：`cosDrop→1` 被当作"预测得好"打印，但坍缩时 cosDrop 同样→1、loss→0。smoke test 会"通过"而实际坍缩 → 假阳性 → 误导 Stage-B/C。`cos_drop ≈ 1 - L_jepa`，与 jepa 冗余，**无法区分"学会预测"与"坍缩"**。
- 缺的是 spread 监控：跨 batch 的 per-dim embedding std / 有效秩（rank）。kill-switch 必须用 std/rank，不能用 cosDrop。
- **修复建议（至少其一，推荐前两条同上）**：(a) student 侧加非对称 predictor MLP；(b) 加 VICReg variance hinge `relu(1-std(S))` 强制每维 std≥1；(c) teacher token 做 batch centering（DINO）或 instance-norm（data2vec）；(d) 日志加 `S` 的 per-dim std 与 effective rank，kill-switch 切到 std。

### 2. 坐标系正确性 — 通过（含 2 个待确认）
- `kpn[:,0]/=ow; kpn[:,1]/=oh` 归一化到 [0,1] 再 `*GW / *GH`：因为 resize 是纯缩放、归一化坐标对 resize 不变，所以**无 resize 错位**。x→宽(GW=4)、y→高(GH=12) 映射一致；`gmask[gi, y0:y1+1, x0:x1+1]` 行=y/高、列=x/宽，索引一致。
- `clip(x*GW,0,GW-1e-3)` 后 `int()`：max=3.999→3（GW=4 末位）、11.999→11（GH=12 末位），**无 off-by-one 溢出**。
- 与项目既有 pose 流程一致：`datasets/pose_dataset.py` 也以 orig_h/orig_w 载入再 `_joint_resize`，确认 npz 关键点是原 crop 像素坐标。
- 本脚本对图像**不做**任何几何增广（无 flip/crop/erase），drop-mask 由同坐标系关键点算出，**自洽不会错位**。
- 待确认（见 M3/M4）：`visibility` 语义、关键点 (x,y) vs (y,x) 顺序。

### 3. swin featmaps shape 解析 — 通过
- `SwinTransformer.forward` 返回 `(x_global[B,C], outs_list)`；`outs[i]` 为 4D `[B,C_i,H_i,W_i]`。
- `fwd_tokens`: `out[1]`=outs_list → `[-1]`=stage3 featmap。384×128 输入下 stage3 输出空间 = **(12,4) = (GH,GW)**，`num_features[-1]=768`。`part_pool` 里 `(Hf,Wf)!=(GH,GW)` 判定为 False，不触发 interpolate。解析稳健、shape 假设正确。

### 4. part_pool / drop-mask / EMA / 数值安全 — 基本通过
- `einsum('bchw,bghw->bgc')` masked-sum 正确；`denom=gmask.sum.clamp_min(1)` 防除零；空 group → token=0，`F.normalize(0)`→0，且这些 group 被 `gvis` 乘掉、不进 loss，安全。
- drop-mask：`(gmask*drop).sum(1).clamp(0,1)` → nearest 上采样到图像 → `x*(1-dmask)` 正确。post-norm 置零 = 填 dataset mean（≈灰图），是标准 mask-fill，@REVIEW 担忧无害（Low）。
- EMA：`pt.mul_(ema).add_(ps,alpha=1-ema)` 公式正确。只遍历 `parameters()` 不含 buffers——swin 全 LayerNorm、无 BN/running-stats，唯一 buffer `relative_position_index` 是常量，**此处不构成 bug**。
- `teacher.load_state_dict(student.state_dict())` **必要且正确**：`build_backbone()` 调两次，未在 ckpt 中的参数（如 frozen `semantic_embed_w/b`）会各自随机初始化不同，此步把 teacher 同步成 student，消除系统性差异。

### 5. backbone init — 通过（一个环境注意点）
- `convert_weights=False` 与本仓一致：`make_model.py:190` 仅 imagenet 用 True，SOLIDER 格式 swin_tiny.pth 用 False；`init_weights` 按 `backbone.` 前缀过滤加载，正是 SOLIDER ckpt 格式。
- 注意：本地 `pretrained/` 只有 `clip_part_text_features.pt`，**无 swin_tiny.pth**；远程（4090 `/home/afr/SOLIDER-REID`）应存在，跑前确认。

### 6. 设计合理性 — 是真 continued-pretrain，但 Stage-A 稳定性无保障
- 不是换名小调参：确是 JEPA 式 latent part-token 自蒸馏 continued-pretrain loop。novelty 在 Stage-B support bank（design 已申明 Stage-A 是"是否=OA-SD/PCVT 换名"的对照），Stage-A 作骨架合理。
- 但如要点 1：**Stage-A 当前是 BYOL-without-predictor，坍缩风险实打实，且 smoke 监控测不出坍缩**。"跑通且不崩"在加上 predictor/var-reg + std 监控之前**不能认为成立**。

---

## 问题清单（按 severity）

### Critical
- **C1 表征坍缩机制缺失**：无 predictor / 无 variance-covariance / 无 centering / 无负样本，等价 BYOL-去-predictor，全局常数解可达；`cosDrop→1` 监控把坍缩读成成功（假阳性 smoke）。需加防坍缩组件 + per-dim std/rank 监控，kill-switch 改用 std。

### High
- **H1 checkpoint 存盘格式与本仓 loader 不兼容**：`torch.save(student.state_dict())` 存的是裸 SwinTransformer 键（`patch_embed... / stages...`，**无 `backbone.` 前缀**）。下游 `init_weights` 只保留 `backbone.` 前缀键 → 过滤后 state_dict 为空 → `list(state_dict.keys())[0]` **IndexError**（或啥都没加载）。**整条 pretrain→fine-tune 链路因此断裂**，违背 design 的"fine-tune 兼容"首要目标。修复：存 `{'state_dict': {f'backbone.{k}': v for k,v in student.state_dict().items()}}`（init_weights 会自动 strip `backbone.`）。
- **H2 semantic_weight 不一致**：`build_backbone` 硬编码 `semantic_weight=1.0`，但**所有**下游 swin_tiny 配置用 `SEMANTIC_WEIGHT: 0.2`。SOLIDER 前向 `x*softplus(sw)+sb` 随 semantic_weight 变，1.0 下预训练的 conv/attn 权重在 0.2 下 fine-tune/测试时激活分布偏移，可能直接抹掉 ≥+0.7 mAP 的 kill-switch 余量。应改为 0.2（或设成 CLI 参数并默认对齐下游）。
- **H3 pose_train.npz 前置缺失**：`data/occluded_duke/pose_train.npz`（键 `filenames/visibility/keypoints[17,2]`）本仓无生成脚本、`data/` 下也无任何 npz。需先离线生成，且：文件名须与 `bounding_box_train` basename 对齐、取主 person（person-0）、坐标为原 crop 像素。否则脚本无法运行。

### Medium
- **M1 重遮挡样本被排除出 JEPA loss**：`if len(vis_g) > drop_groups` 才 drop；可见 group ≤ drop_groups（=2）的**重遮挡样本一个都不 drop → drop_m=0 → 不进 L_jepa**。恰恰把方法目标人群（occluded）排除在主信号外，训练偏向易样本。建议 `drop = min(drop_groups, len(vis_g)-1)`。
- **M2 body-group bbox 区域重叠污染 anchor**：torso(肩+髋) 与 larm/rarm(肩)、legs(髋) 区域重叠。drop A 时其 bbox 也会把 kept B 的图像区域置零 → student 在被污染输入上算 L_anchor，"可见 group 不漂移"的干净语义被削弱，part-token target 也不互斥。
- **M3 visibility 语义未定**：`vis_thr=0.3` 作用于 `visibility`。若它是 COCO `{0,1,2}` flag，则 0.3 会把"标注但被遮挡"(=1) 也算可见 → "可见 group" 含实际被遮部位，破坏机制语义；若是 [0,1] 置信度（`config` 里 `POSE_THRESHOLD=0.3` 暗示是 score）则合理。需确认。
- **M4 关键点坐标顺序假设**：代码假设 `kp[:,0]=x, kp[:,1]=y`。若 npz 存成 (y,x)，掩码会转置错位。COCO 惯例是 (x,y)，但需对 npz 生成端核实。

### Low
- **L1** `cos_drop ≈ 1 - L_jepa`，冗余；真正缺的是 embedding std / effective-rank 监控（且 kill-switch 应用它，见 C1）。
- **L2** 恒定 LR 2e-4、无 warmup/cosine、EMA 恒 0.996 不 ramp→1.0、WD=0.05 施加到 norm/bias；非错误但次优。
- **L3** `DEV='cuda'` 硬编码，无 CPU/设备回退；全不可见样本 → `L_union=1` 常数项（罕见）。
- **L4** 无 AMP（纯 fp32）：数值安全、AMP-safety 不适用，仅较慢。
- **L5** post-norm 置零=mean-fill，可接受（@REVIEW 担忧无害）。

---

## 结论
坐标系、swin shape 解析、masked-avg-pool、EMA、backbone init 这些"易错点"实现都正确，作者对数据流把握扎实。但 **C1 坍缩风险 + 监控测不出坍缩** 是头号科学风险，**H1 存盘格式断链** 会让整条 pretrain→fine-tune 失效，**H2 semantic_weight 失配** 可能吞掉 kill-switch 余量，**H3 pose npz** 是运行前置。这些必须先修：
1. 加 predictor/VICReg/centering 之一 + per-dim std/rank 日志，kill-switch 改用 std（C1）；
2. 存 `backbone.` 前缀 checkpoint（H1）；
3. semantic_weight 对齐下游 0.2（H2）；
4. 确认/生成 pose_train.npz 并核实 visibility/坐标语义（H3/M3/M4）；
5. 重遮挡样本至少 drop 1（M1）、留意 group 区域重叠（M2）。

修完后需做**同范围全量复审**（非仅复审修复点）。

需修复后重审

---

## v2 重审（修复后）

**Reviewer**: Claude (Opus, 独立 v2)　**Date**: 2026-06-26　**Round**: v2　**Scope**: 与 v1 同范围全量（`psc_jepa_pretrain.py` 逐行 + design + 实测对照 `swin_transformer.py:init_weights/forward/num_features`、`make_model.py`、`datasets/pose_dataset.py`、`config/defaults.py` 与各 swin_tiny config）。

**结论先行**：v1 的 4 个修复点里 **H1 / H2 / M1 全部正确修复，C1 只修对了一半**。predictor（防坍缩主力）实现正确，但新加的 **VICReg variance hinge + tok_std 监控对 L2-normalized token 的标定整体错位**，这是本轮新发现的 1 个 **High**——它让 C1 fix 的第二个目标（可靠监控坍缩 / kill-switch）依然不成立，且把 var 正则做成了永远不满足的饱和项。另有 M2（bbox 重叠，未处理）、smoke<bs 静默空跑（新 Medium）。**verdict 仍为需修复后重审**。

### A. 逐一核实 v1 修复

- **C1-predictor — 正确**。`predictor=Linear(C,C)-BN1d-GELU-Linear(C,C)`，`Sp=normalize(predictor(S.reshape(-1,C)).reshape(B,G,C))` 维度正确（S [B,G,768]→[320,768]→predictor→reshape 回 [B,G,768]）；student 侧带 predictor、teacher 侧 stop-grad（`with torch.no_grad`）+ EMA，**= 标准 BYOL/SimSiam 非对称**，这是真正挡坍缩的机制。predictor 已进 optimizer（`list(student.parameters())+list(predictor.parameters())`）✅；predictor **未** EMA（teacher 无 predictor）—— **正确**，online-only predictor 本就不该进 target 网络。
- **BatchNorm1d 的 batch 统计 — 安全**。两次调用输入分别是 part 路 `[B*G=320,768]`、union 路 `[B=64,768]`，drop_last=True 保证 B 恒=64，N≥64 足够估计 BN 统计；predictor **从不存盘、从不 eval**（只在 train loop 里前向），故 BN 的 running-stats / eval 行为对下游无影响，无需担心 eval-time BN。
- **H1 ckpt 前缀 — 正确（已对 loader 实测）**。`init_weights`（swin_transformer.py:1341）取 `ckpt['state_dict']`，1352-1354 只保留 `backbone.` 前缀键并 `k[9:]` 剥前缀，1393 `load_state_dict(state_dict, False)`。exp361 存 `{'state_dict':{f'backbone.{k}':v}}` 正好命中；剥完前缀=裸 swin 键，与 `self.base` 完全匹配；1357 的 `keys()[0]` 不会 IndexError（所有键都过滤通过、非空）。与原 SOLIDER ckpt 同格式，往返一致。**断链已修复**。
- **H2 semantic_weight — 正确（含 1 个提醒）**。cli 默认 0.2，与**所有** swin_tiny config（`SEMANTIC_WEIGHT: 0.2`）一致，student/teacher 同走 0.2，forward(1396-1405) 用 `self.semantic_weight`。⚠️ 提醒：`config/defaults.py:79` 仍是 1.0，下游 fine-tune **必须用带 0.2 override 的 config**（现有都带），否则又会失配——非本脚本 bug，但要在启动 fine-tune 时确认。
- **M1 自适应 drop — 正确**。`nd=min(drop_groups,max(0,len(vis_g)-1))`：vis=0/1→nd=0（不 drop，保≥1 可见上下文）；vis=2→drop 1 留 1（**重遮挡目标人群已纳入 L_jepa**）；vis=5→drop 2 留 3。仅"恰好 1 可见 group"样本不进 L_jepa（无法构成 predict-dropped-from-visible 任务，不可避免），但仍进 L_anchor/L_union。逻辑正确。

### B. 新发现 High — var hinge / tok_std 对 normalized token 标定错位

- `S=fwd_tokens(...)` 末端经 `part_pool→F.normalize(tok,dim=2)`，**S 是单位向量**，不是修复说明里写的"raw S"。对 C=768 的单位向量，逐维 std 的均值 **恒 ≤ 1/√C ≈ 0.036**（因 Σ_i Var≤E‖x‖²=1）。
- 后果①：`L_var=relu(1.0-std_s).mean()` 的目标 std≥1.0 **数学上不可达**，L_var 恒≈0.96（健康）↔1.0（坍缩），几乎不动 → 作为"VICReg variance floor"是**永远饱和、永不满足**的项（梯度方向上仍轻微推 spread，但绝非设计意图的方差地板）。
- 后果②：`tok_std≈std_s.mean()` 健康态≈0.036，注释写的 **`<0.3=坍缩警戒`阈值偏高约 10×**，照此判定会**永远报坍缩**（cry-wolf），等于把 C1 想修的"可靠坍缩 kill-switch"再次做废。健康 0.036 vs 坍缩→0，真实判别区间是 [0,0.036]，阈值应 ~0.015–0.02。
- 后果③：`std_s` 在 `S.reshape(-1,C)` 上算，**未按 gvis 掩码**，把不可见 group 的零向量 token（`normalize(0)=0`）混进统计（其它 loss 都按 gvis/drop_m/vis_m 掩码了），遮挡多时进一步污染该指标。
- 修复建议：var 正则与坍缩监控改到 **pre-norm（part_pool 归一化前）token** 上算、并按 gvis 掩码只取可见 token；或干脆去掉 var、改用 **effective-rank / 协方差离对角** 作监控，阈值按 1/√C 量级标定。**坍缩"预防"靠 predictor 大概率没问题，但坍缩"监控/正则"这条线仍坏**——多天 backbone 预训练不该带坏掉的 kill-switch 上线。

### C. v1 遗留 Medium 复核

- **M2 bbox 重叠 — 未处理（维持 Medium）**。torso=[5,6,11,12] 与 larm/rarm 共享肩(5/6)、与 legs 共享髋(11/12)，各 group 的轴对齐 bbox 大面积重叠。drop torso 时其 bbox 会把 kept larm/rarm/legs 的图像区域一并置零 → kept group 在被污染输入上算 L_anchor、part-token target 不互斥，"可见锚不漂移"语义被削弱。非致命（Stage-A 可容忍），但应记，后续可用互斥分割或扣掉重叠区。
- **M3 visibility 语义 — 已解决**。实测 `pose_dataset.py:380-384` 证实 visibility 是 0-1 score（缺失时 `clip(scores)`），vis_thr=0.3 对 score 合理。注意项目自身 `visibility_binary` 用 0.5（:389），0.3 更宽松（刻意选择，OK，与 `POSE_THRESHOLD=0.3` 对齐）。
- **M4 (x,y) 顺序 — 确认正确**。`pose_dataset._joint_resize`：`kp[:,0]*=target_w/orig_w`（col0=x/宽）、`kp[:,1]*=target_h/orig_h`（col1=y/高），与 exp361 `kpn[:,0]/=ow; kpn[:,1]/=oh` **完全一致**；`gmask[gi,y0:y1+1,x0:x1+1]`（dim1=GH=y、dim2=GW=x）索引自洽。

### D. 新引入问题 / 数值与梯度流

- 梯度流正确：L_var 走 S→student backbone；L_jepa/anchor/union 走 Sp=predictor(S)→predictor+student。两路都回传。
- 数值安全：`drop_m/vis_m/denom.clamp_min(1)`、`max(ow,1)/max(oh,1)`、normalize 对零向量返回零，均安全；纯 fp32 无 AMP。
- **新 Medium（M-v2-1）smoke 静默空跑**：DataLoader `drop_last=True`+bs=64，`--smoke N` 若 N<64 → `len(dl)=0` → 内循环不执行 → `last={}`、不训练，却照常 `[done]` 并存一个**未训练的 ckpt**。Stage-A 交付物正是"smoke 跑通"，`--smoke 50` 会被静默坑。建议 smoke≥bs，或 smoke 时 `drop_last=False`/调小 bs。
- **H3（v1 High，外部数据前置，未在代码内解决）**：combined `pose_train.npz`（键 `filenames`/`visibility[N,17]`/`keypoints[N,17,2]`、取 person-0）仓库内无生成脚本，且 pose_dataset 用的是 per-image npz（格式不同）。`load_pose` 读法对（假设键存在即正确），但**文件不存在脚本就跑不起来**——启动前必须确认/生成。非代码 bug，是硬前置。

### E. Low

- L1 `DEV='cuda'` 硬编码，无 CPU/非默认卡回退。
- L2 全不可见样本 → `L_union=1.0` 常数项（罕见）。
- L3 恒定 LR、无 warmup/cosine、EMA 恒 0.996 不 ramp、WD 0.05 施于 norm/bias——次优非错。
- L4 predictor 同一模块每 step 被 part([320,C]) 与 union([64,C], 量级≠单位向量) 两次前向，BN 见双峰分布；因不存盘故无害。
- L5 无 AMP（fp32），安全仅慢。

### 结论（v2）
H1/H2/M1 修对，C1 的 predictor 修对（坍缩**预防**大概率成立），M3/M4 实测确认无误。但 **C1 的 var-reg + tok_std 监控对 L2-normalized token 标定整体错位（目标 1.0 不可达、阈值 0.3 偏高 10×、未掩码）= 新 High**，使"可靠监控坍缩"的修复目标仍未达成；外加 M2（未处理 Medium）、smoke<bs 静默空跑（新 Medium）、H3 数据前置。建议：① var/坍缩监控改到 pre-norm + gvis 掩码的 token、阈值按 1/√C 标定（或换 effective-rank）；② M2 记账或扣重叠；③ smoke 路径防空跑；④ 启动前确认 pose_train.npz。修完做同范围全量三审。

需修复后重审

---

## v3 重审（二次修复后）

**Reviewer**: Claude (Opus, 独立 v3)　**Date**: 2026-06-26　**Round**: v3　**Scope**: 与 v1/v2 同范围全量（`psc_jepa_pretrain.py` 逐行 + design + 实测对照 `swin_transformer.py:init_weights(1337-1394)/forward(1396-1429)/num_features`、`pose_dataset.py`、`make_model.py`、`config/defaults.py`）。

**结论先行**：v2 唯一 High（var-reg + tok_std 对 L2-normalized token 标定错位）与两个 Medium（smoke<bs 空跑、M1）针对的修复**逐项核实全部正确**——var 标定数学成立、监控阈值落入合理区、掩码与边界安全；smoke 不再空跑。M2（bbox 重叠）仍未处理但属 **Stage-A 可接受、非阻断**的设计权衡。本轮**无 Critical、无 High**；新发现仅 1 个窄 Low（smoke 末批 size-1 触发 predictor BN1d 训练态报错）。**verdict = 审查通过**。

### 1. var-reg / tok_std 标定（核实 v2-High 修复）— 正确
- **数学核验**：S 经 part_pool 末端 `F.normalize(tok,dim=2)` 为单位向量；对 C=768，`Σ_i Var(x_i)=E‖x‖²−‖μ‖²=1−‖μ‖²≤1` ⇒ `Σ std_i²≤1` ⇒（Cauchy-Schwarz）`mean(std_i)≤1/√C≈0.036`，等号仅各向同性取得。故 `std_s=std(S)·√C` 的**健康上限恰为 1.0**（各向同性）、坍缩→0，**target 1.0 可达**（v2"永不可达/饱和近 max"已解决）。
- **L_var 行为**：`relu(1−std_s).mean()` 各向同性→0；真实各向异性→>0 且只惩罚 std_s<1 的低方差维（每维差异化梯度，非 v2"全维恒定 0.964 永不达 0"），是有效 VICReg 方差地板，提供持续温和白化/抗坍缩压力。注意 √C 重标定使 L_var 对 student 的梯度比 v2 约强 √C≈27.7×（intended：predictor 仍是主防线，var 为后备；w_var=1.0 量级与 L_jepa 相当）→ 见 L1。
- **监控阈值**：`tok_std=std_s.mean()` 健康 ~0.5–1.0、坍缩→0；注释 `<0.5=坍缩警戒` 换算回未标定 std≈0.018，正落在 v2 建议的 [0.015,0.02] → **不再偏高 10×，cry-wolf 消除**。
- **掩码**：`vmask=gvis.reshape(-1).bool()` 正确剔除不可见 group 的零向量 token（`normalize(0)=0`），与其它 loss 的 gvis/drop_m/vis_m 一致。`std_s` 同时喂 L_var 与 tok_std，二者一致。Svis 含 dropped-group token（drop⊆visible，pooled 自置零区，spread 偏低）属保守纳入、非错。
- 梯度流：L_var→Svis（boolean-index 可导）→S→student backbone；fallback `ones(C)` 为常量无 grad（不可估方差时不施惩罚，合理）。

### 2. 边界 `Svis.shape[0]>1` — 安全
- `torch.std` 默认 unbiased（correction=1），N=1 → 除 0 → nan；`>1`（即 ≥2）是正确护栏。
- vmask 全 False → `Svis=[0,C]` → shape[0]=0 不 >1 → 落 fallback `torch.ones(C,device=DEV)`，**绝不在空张量上调 .std**。dtype float32、device 对齐。
- fallback 让 L_var=relu(0)=0、tok_std=1.0（看似健康，良性 false-negative）。触发=整 batch B·G=320 token 中可见 ≤1，对真实 pose（人均≥1 可见 group）几乎不可能。

### 3. smoke 修复 — 正确
- `bs=min(cli.bs,len(ds)) if smoke` + `drop_last=(not smoke)`：smoke 时 bs≤数据量且 drop_last=False → `len(dl)≥1`，不再空跑/存未训练 ckpt（v2-Med 已解）。
- 非 smoke：bs=64、drop_last=True 不变 → B 恒 64，predictor BN1d 统计稳定，与 v2 分析一致。

### 4. M2（body-group bbox 重叠）— Medium，Stage-A 可接受，**非阻断**（明确判断）
- 机制：torso[5,6,11,12] 与 larm/rarm 共肩、与 legs 共髋，drop torso 时其轴对齐 bbox 把 kept 组的肩/髋带一并置零 → kept 组 student token 在部分污染输入上算 L_anchor，"可见锚不漂移"语义被削弱、part target 不互斥。
- **判为非阻断**，理由：① 重叠是局部带状非整体（drop≤2 且 `nd=min(2,len(vis)−1)` 恒留≥1 可见 → 绝不清空全身，head/小腿/外侧臂存活，partial-view 任务不退化）；② 不崩、不 NaN、不威胁 Stage-A 稳定性目标（design 明定 Stage-A 目标=smoke 跑通+不崩）；③ 主信号 L_jepa（dropped 组）不受此污染，受影响仅 L_anchor 且为软污染（可视作温和 JEPA）；④ design 明确分阶段、novelty 裁决推迟到 Stage-C 带对照。
- 处置：作为已知设计权衡接受，**列 Stage-B TODO**（互斥分割 / 扣重叠区 / L_anchor 池化改用 `gmask*(1−dmask_grid)` 仅取干净像素），训练日志观察 anchor 是否异常。

### 5. 全范围逐行 + 新问题
- 既有正确点复核无回退：kpn 按原 crop 归一化、resize 纯缩放不变、grid clip 无 off-by-one、einsum masked-pool、`denom.clamp_min(1)`、`normalize(0)=0`、EMA over `parameters()`（swin 仅常量 buffer `relative_position_index`）、ckpt `backbone.` 前缀（对 `init_weights:1352-1354` 剥前缀往返一致、`keys()[0]` 不空）、semantic_weight 0.2 对齐下游——v1/v2 已逐项确认。
- swin forward 实测（:1413-1429）返回 `(x_global, outs)`，`x=avgpool(outs[-1])`，`outs[-1]=[B,768,12,4]=(GH,GW)`，`C=num_features[-1]=768` 与 predictor C 一致；`out[1][-1]` 解析正确。
- **新 Low（L-v3-1）smoke 末批 size-1 → BN1d 训练态崩**：smoke 时 drop_last=False，若 capped `len(ds)>bs` 且 `%bs==1`（如 `--smoke 65/129`），末批 B=1 → predictor `BatchNorm1d` 训练态报 "Expected more than 1 value per channel"。**崩得响（非静默）**，主训练 drop_last=True B=64 不受影响。建议 smoke 取 ≤bs 或末批 `B<2` 时 `continue`。
- L1 w_var √C-标定后梯度 ~27× 于 v2（intended，留意是否过度白化，可调 w_var）；L2 union 路 0 可见 → L_union=1 常数（罕见）；L3 `DEV='cuda'` 硬编码；L4 predictor 同模块每 step 被 part[320,C]/union[64,C] 两次前向（各用自身 batch 统计、不存盘 → 无害）；L5 无 AMP/warmup/cosine、EMA 不 ramp——次优非错。

### 前置（非代码缺陷，启动前必须满足，不影响代码 verdict）
- **H3**：combined `pose_train.npz`（键 `filenames`/`visibility[N,17]`/`keypoints[N,17,2]`，取 person-0、坐标为原 crop 像素、visibility 为 0-1 score）仓库内无生成脚本；`filenames` 须与 `bounding_box_train` basename 对齐。缺失/格式不符会在 `np.load` 或 `DataLoader(batch_size=0)` 处**响亮报错（非静默）**，仍属硬运行前置。M3/M4 已于 v2 实测确认（visibility=0-1 score、(x,y) 顺序）。
- semantic_weight：fine-tune 必须用带 `SEMANTIC_WEIGHT 0.2` override 的 config（`defaults.py:79` 仍 1.0）——非本脚本 bug，启动 fine-tune 时确认。

### 结论（v3）
预防坍缩（predictor 非对称 + EMA stop-grad，v2 已确认）与**监控/正则坍缩**（√C-标定 std-floor + 0.5 阈值 + gvis 掩码，本轮新确认）两条线**均成立**——v2 唯一 High 已正确闭合。smoke 修复正确。M2 维持 Medium 但属 Stage-A 可接受、非阻断的设计权衡，列 Stage-B TODO。本轮无 Critical、无 High；仅 1 个窄 Low（smoke 末批 BN）+ 既有 Low + H3/semantic_weight 数据/配置前置。Medium（M2）带条件接受。

审查通过
