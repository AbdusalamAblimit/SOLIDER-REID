# Claude Broad Review — exp356 PC-MSC (Pose-Conditioned Masked Semantic Completion)

**审查模型**: Opus 4.8 (broad pre-training review)
**日期**: 2026-06-21
**变更范围**: `git show HEAD` (647224d) — config/defaults.py (+5), configs/occluded_duke/exp356_pcmsc.yml (新 85 行), model/modules/clip_id_prompt.py (新 `CLIPVisualEncoder` 47 行), model/pose_backbone_model.py (+64: `_pcmsc_loss` + `__init__` 块 + forward 调用)

## 结论: 审查通过 (PASS) — 无 Critical/High 阻断项

逐行核对了 `CLIPVisualEncoder`、`_pcmsc_loss`、`__init__` PC-MSC 块、forward 调用、config diff、optimizer、processor 消费、test 端跳过。代码正确性、单变量隔离、AMP dtype 处理、RNG 保护、训练-only 守卫均通过。仅有 3 个 Low（潜在 footgun，对本 config 不触发）。

---

## 逐项核对（审查重点 a-h）

### a. CLIPVisualEncoder 正确性 — PASS
- **真冻结**: `clip_id_prompt.py:205-207` 所有 `self.visual.parameters()` `requires_grad_(False)` + `.eval()`。`part_targets` 整体 `@torch.no_grad()`（line 216）。optimizer (`solver/make_optimizer.py:8`) 跳过 `requires_grad=False` 参数 → CLIP visual 永不进优化器、无梯度、无 saved activation（焦点 h 同时解决：no_grad → 无激活内存爆炸）。
- **SOLIDER→CLIP 重归一化**: line 221-223 `img.float()*0.5+0.5`（un-norm mean=std=0.5 → [0,1]）→ `(x-mean)/std`（CLIP mean/std buffer，line 209-210）→ `interpolate` 224。正确。
- **hook 捕获 patch tokens**: hook 在 `resblocks[-1]`（line 212）。与已跑通的 exp339 (`scripts/exp339_clip_grounded_pool.py:87` hook `transformer` 整体) 等价——open_clip `Transformer.forward` 顺序跑 resblocks 后无额外处理直接返回最后一个 resblock 输出，故 hook `transformer` 与 hook `resblocks[-1]` 得到同一张量（均为 pre-`ln_post`）。
- **(seq,B,dim) vs (B,seq,dim)** — ★ 运行时实测确认: 子代理在 **lab-3090-d `solider-reid` env (open_clip 2.32.0)** 实跑 hook，`resblocks[-1]` 输出 = **`(2,257,1024)` = NLD batch-first**（2.32.0 `Transformer.__init__` 默认 `batch_first=True`，forward 仅在 `not batch_first` 时转置）。故 `t.shape[0]==B`，line 227 `if t.shape[0] != x.shape[0]` **为 False → 不 permute**，`t` 保持 (B,257,1024)，`t[:,1:]` 正确丢 CLS → (B,256,1024)。**代码因按实际 shape 判断（非硬编码 layout）而对 2.32.0 的 NLD 仍正确**。exp339:95-96 同构（同样按 shape 判断，故也不 permute），二者一致。
  - **注**: line 227 注释 `# (seq,B,dim) -> (B,seq,dim)` 在 2.32.0 下略误导（实际已是 NLD，permute 分支不触发），但逻辑稳健——若换回老版 open_clip 输出 LND，则 `t.shape[0]=seq≠B` 触发 permute，仍正确。跨版本健壮。
  - **ln_post 顺序**: 2.32.0 `_pool` 在 `final_ln_after_pool=False`(ViT-L/openai 默认) 下对**含 CLS 的全序列**做 `ln_post`；新代码对 `t[:,1:]` 做 `ln_post`。LayerNorm 沿 channel 逐 token 归一、不跨序列维 → `ln_post(t[:,1:]) == ln_post(t)[:,1:]`（patch token 结果一致）。正确。
- **ln_post + proj + 丢 CLS**: line 229 `ln_post(t[:,1:])` 丢 CLS（CLS 在 index 0，bundled model.py:222 `cat([class_embedding, x])` 确认）→ line 230-231 `@ self.visual.proj`。与 exp339:97-100 一致。
- **16×16 reshape**: ViT-L-14 @224 → 224/14=16 → 256 patch（实测 seq=257=1 CLS+256）。line 233 `reshape(B,16,16,-1)`。正确。
- **clip_dim 实测**: 子代理实测 `text_projection.shape=(768,768)` → `clip_dim=768`；`visual.proj.shape=(1024,768)` → patch `@proj` 映射 1024→768。故 `target`=(B,3,768)，`pcmsc_proj=Linear(768→768)`，维度自洽。`attn_pool` 属性存在但为 **None**（ViT-L/openai 无 attentional pooler），不影响 hook 路径。
- **per-region 池化**: line 234-236 rows 0:5/5:11/11:16（head 5/torso 6/legs 5）。与 design.md 一致。

### b. _pcmsc_loss 正确性 — PASS
- **region_of_token by row** (line 659-662): `h1=round(5/16·H)`, `h2=round(11/16·H)`，clamp `h1∈[1,H-2]`、`h2∈[h1+1,H-1]`。Swin-Tiny@384×128 → featmap H=12,W=4 → h1=4,h2=8 → region 0/1/2 各 4 行非空。
- **空区 NaN 防护** (line 666 `vis_map[:, region_of_token==r].mean(1)`): h1/h2 clamp 保证三区各 ≥1 行 → `region_of_token==r` 永不空 → `mean(1)` 不产生空切片 NaN。**焦点 b 的 NaN 担忧已被 clamp 闭合。**
- **可见性加权 multinomial** (line 668-671): pose `reg_vis.clamp(min=1e-6).softmax(dim=1)` → `multinomial` 永远有效（softmax 必和为 1，无 NaN）；control `randint(0,3)`。正确隔离。
- **masking** (line 673-676): `torch.where(mask.unsqueeze(-1), mt, tokens)`，`mt=mask_token.view(1,1,C).to(tokens.dtype)`。两操作数同 dtype（见 c）。
- **decoder** (line 678-679): `query=pcmsc_query[sel].unsqueeze(1)` (B,1,C)，`key=value=tok_masked` (B,HW,C)，batch_first MHA → (B,1,C) → squeeze (B,C)。维度一致。
- **cos loss** (line 680-683): `R=normalize(proj(R).float())` (B,clip_dim) fp32；`tgt=target[arange(B),sel]` (B,clip_dim)；`cos=(R·tgt).sum(-1)`；`loss=(1-cos).mean()`。正确。
- **梯度流塑造 backbone**: masked 区被 `mask_token`（param，无 backbone 梯度）替换，但**可见 token 携带 backbone 梯度**（来自 `featmap`）；decoder 从含可见 token 的 `tok_masked` 重建 → 梯度经可见 token 回流 backbone。机制正确（从可见证据重建被删区 CLIP 语义）。

### c. AMP/dtype — PASS（关键项，重点核对，无 PGPD 式 index_put 崩溃风险）
forward 整体在 `processor/processor.py:573 with amp.autocast(enabled=True)` 下。逐边界：
- `target = part_targets(img)`: `@no_grad`；内部 `img.float()` → fp32，`x.to(wdtype)` 转 CLIP 权重 dtype；hook 输出 fp16（autocast）→ `ln_post`（autocast 走 fp32）→ `@proj` → `.float()`(line 233) → `normalize` → **fp32 detached**。
- `tokens` ← `featmap`（autocast fp16）。`mt.to(tokens.dtype)` → fp16；`torch.where(mask, mt, tokens)` **两侧均 fp16** → 无 dtype mismatch（PGPD 崩溃正是 where/scatter 混 fp16/fp32，此处显式对齐避开）。
- `q.to(tokens.dtype)` → fp16；MHA 输入全 fp16 → 输出 fp16。
- `proj(R)` autocast fp16 → `.float()` → fp32 → `normalize` fp32；`tgt` fp32；`cos`/`loss` fp32。
- `multinomial` 在 fp32 (`reg_vis` 由 `pose.float()` 派生) 上 → 安全。
**autocast fp16 下不会 dtype 报错。**

### d. RNG 保护 — PASS
`__init__` PC-MSC 块 (line 263-275): `_rng=get_rng_state()`(265) 包住 `CLIPVisualEncoder(...)`(open_clip create_model 消耗 RNG) + `pcmsc_query=randn`(消耗) + `pcmsc_decoder=MHA`(消耗) + `pcmsc_proj=Linear`(消耗)，`set_rng_state(_rng)`(275) 在**全部 4 个消耗 RNG 的创建之后**复位。→ line 275 之后 RNG 状态 == line 265 之前 == exp341 同点。`bottleneck`/`classifier`/backbone 在 `super().__init__()`(line 37) 即创建，远早于本块，本就不受影响。复位位置正确（同 line 71/77、231/234 已复现的姊妹模式）。CPU-only `get/set_rng_state` 与姊妹模块一致（init 在 CPU）。

### e. 训练-only + 数据流 — PASS
- forward 守卫 (line 781): `use_pcmsc and scene_heatmaps is not None and self.training`，且整块在 `if self.training:`(line 730) 内 → 双重守卫。
- TEST（eval 模式）: `self.training=False` → PC-MSC 完全跳过（无 CLIP ViT forward、无 decoder）。
- 描述子: eval 分支 (line 1055+) `test_feat=global_feat`（NECK_FEAT='before'）；exp356 无 LGPA/VCSR/PPA/GCN/structural → `gcn_feats=None` → line 1184 不进 → **描述子 = 未 mask 的 global**，与 exp341 同。
- loss 传递: 默认返回路径 (line 1052-1053) `{'clip_id_loss': clip_id_loss}` → processor:1297-1302 加权 `POSE_CLIP_ID_WEIGHT`(1.0) 加入 loss，`.item()` 记录。pcmsc 内部已乘 `pcmsc_w`(1.0) → 净权重 1.0。

### f. 单变量 vs exp341 — PASS
config diff（忽略注释）仅: `+POSE_PCMSC True`, `+POSE_PCMSC_W 1.0`, `+POSE_PCMSC_RANDOM_MASK False`, OUTPUT_DIR。`POSE_PCMSC` 默认 False (defaults.py:245) → False == exp341。CLIP-visual 仅 `use_pcmsc` 时加载（line 263）。干净单变量。

### g. x 未被 backbone 改写 — PASS
`_run_backbone_with_psg(x,...)` 首行 `x, hw_shape = self.base.patch_embed(x)` 仅**重绑函数内局部 x**，不改写 caller 的 `x`。扫描 forward (699-783) 无对 `x` 的重绑/in-place。→ line 782 传给 `_pcmsc_loss` 的 `x` 是原始未改输入图（SOLIDER 0.5/0.5 归一），正是 `part_targets` 所需。

### h. frozen CLIP ViT-L 每 iter forward — PASS
`part_targets` `@torch.no_grad()` → 无 saved activation、无内存爆炸；非优化器参数。每 iter 多一次 ViT-L frozen forward（设计已诚实标注计算开销）。

---

## Findings

### Low (对 exp356 当前 config 均不触发，仅潜在 footgun / 可移植性)
- **L1 (启动机/可移植性)**: `configs/occluded_duke/exp356_pcmsc.yml:23` `POSE_CLIP_ID_PRETRAINED='/home/afr/SOLIDER-REID/clip_l14_openclip.safetensors'` 是 **lab-4090 专属路径**（1.7GB safetensors）。`CLIPVisualEncoder.__init__` 传给 `open_clip.create_model_and_transforms(pretrained=...)`（open_clip 2.32.0 支持本地 safetensors，已查证）。**启动机选择（子代理实测，2026-06-21）**：
  - `hyy` **当前不可达**（8 次重试全 `Connection closed by 198.18.0.89`，gpushare 代理拒连，非首连 banner 超时）→ 暂不能在 hyy 训。
  - `lab-4090` 可达，open_clip 2.32.0 在 `/usr/local/anaconda3/envs/mmpose-abu/bin/python`；config 里的 safetensors 路径正是此机 → **推荐在 lab-4090 用该 env 启动 exp356**。
  - `lab-3090-d` 可达，open_clip 2.32.0 在 `/root/miniconda3/envs/solider-reid/bin/python`（另有 3.3.0 在 `solider-reid-pt2`，**勿用于本实验**——3.3.0 layout/属性名与 2.32.0 不同）；若在 3090 训需先同步 safetensors 并改路径。
  - 训练用 **open_clip 2.32.0**（与 exp341 同），不要误用 3.3.0 env。
- **L2 (潜在 footgun)**: PC-MSC 的 `mask_token`/`decoder`/`proj` 按 `self.in_planes`（line 271-274 读取时值）创建，而 `_pcmsc_loss` 对 `featmap`（通道 = `base.num_features[-1]`）做 `torch.where(mask, mt, tokens)`。本 config `REDUCE_FEAT_DIM=False`（默认）→ `in_planes==num_features[-1]==768` 一致，安全。**若将来开 `REDUCE_FEAT_DIM`，`in_planes` 变 FEAT_DIM 但 featmap 仍 768 → shape mismatch**。当前无影响。
- **L3 (可移植性)**: `pcmsc_decoder=MultiheadAttention(in_planes, num_heads=8)`（line 273）要求 `in_planes%8==0`。Swin-Tiny/Small/Base 末层 768/768/1024 均可整除，安全。换非整除 backbone 会在构造期抛错。当前无影响。

### 一致性观察（非阻断）
- `_pcmsc_loss` 顶部 `import torch.nn.functional as F`（line 654）与模块顶部 import（line 11）重复，局部 shadow 无害（同姊妹方法 line 593/654 既有风格）。

---

## 与已跑通先例的交叉验证（强化置信）
- dense-token hook + `permute(1,0,2)` + `t[:,1:]` + `ln_post` + `@proj` 路径与 `scripts/exp339_clip_grounded_pool.py:91-101`（已实跑产出真实 mAP）**逐行同构**。
- CLS-prepend / LND permute / ln_post / proj 经 bundled `experiments/clip_reid_compare/CLIP-ReID/model/clip/model.py:201-228` 源码确认。
- kill-switch 结果（design.md:51-53，GLOBAL +0.022 / head +0.011 / torso +0.009 / legs +0.013）证明 per-region dense 特征逻辑已被实跑且带（弱）ID 信号。

审查通过。可进入 Codex 第二轮审查。
