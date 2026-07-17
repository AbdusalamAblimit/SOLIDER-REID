# 实验 exp324i: 解相关感知的 DINO-LoRA adaptation — 打破"判别性-互补性张力"

> **来源**：今夜发现的 headline 洞察（判别性-互补性张力：naive ID-adaptation 让 FM 趋同 SOTA、失互补）。这是直接攻击该张力的**真 method shot**——也是 FM-import 方向最后一个有原创性的方法尝试。
> **性质**：训练实验，开训前过 Claude+Codex 双审查。机器 lab-3090-d（已空）。

## 动机
- exp324h 证：让 DINO 判别化(8.65→37) 的同时它和 Swin 越来越像(top-10 Jaccard 0.062→0.253) → adaptation 把 FM 推向 SOTA-like 方向，判别升、互补降，融合只 +0.37(NFC 级)。
- **若能显式强迫 adapted-DINO 与 Swin 解相关（同时保 ID 判别），它会进入一个互补的判别子空间** → 融合(decorr-DINO ⊕ Swin) 可能真正 > Swin 单独(72.57 heavy / 75 all) = beat-SOTA 方法。

## 核心假设
DINO-LoRA + 一个"与缓存的 frozen-Swin 全局特征**线性解相关**"的损失项（同时保 ID CE + triplet + part CE），训出的特征判别但**与 Swin 线性无关**，融合后在重遮挡 query 上 **> Swin 单独**。

## 解相关损失的正确形式（设计修正 2026-06-16）
> 原稿写 `L_decorr = cos(d,s)^2` 有**维度错误**：DINO 全局特征 d（投影后 512 维）与 Swin 全局特征 s（exp255 输出维，≠512）不在同一空间，逐图 cos 无定义。改为**跨协方差解相关**（Barlow-Twins 冗余消除的跨网络版，维度无关、不会塌缩因为 s 是冻结缓存的）：
> - batch 内对 d、s **逐维度中心化 + 标准化**（z-score，per-dim across batch）得 d̂(B×Dd)、ŝ(B×Ds)。
> - 跨相关矩阵 `C = (1/B)·d̂ᵀ ŝ`（Dd×Ds），`L_decorr = ||C||_F² = Σ_{jk} C_{jk}²`。
> - 物理含义：逼每个 DINO 维度与每个 Swin 维度在 batch 上**线性不相关** → d 只能用 Swin 没用的方向编码 ID → 融合时 d 提供 s 没有的线性信息。比逐图 cos 既正确又更可能产生真互补。

## 技术方案
1. **预缓存 Swin 全局特征**（新脚本 `scripts/exp324i_swin_cache.py`，跑在 `solider-reid` env，torch1.13+mmengine）：复用 `exp324f_swin_distmat.py` 的 `make_model`+`load_param` 加载 exp255 ckpt（`log/occluded_duke/exp255_small_gcn512_2stage/transformer_120.pth`），对全部 train 图用 **val transform（无随机增广、no-flip、确定性）** 前向，存每图 global 特征 → `experiments/exp324i/_swin_cache/train_swin_feat.npz`（key=图名）。一次性 ~10min。
2. **新脚本 `scripts/exp324i_lora_decorr.py`**（复制 `exp324d_lora.py` 改，不原地改 d）：加载缓存，按 batch 图名查 `s`；DINO-LoRA 全局特征 `d`（head 的 bn 前 global，未 L2）；加 `λ·L_decorr`（上面跨协方差形式）。total = id + triplet + part_weight·part_id + **λ·L_decorr**。λ CLI 可配。
3. DINO 仍 LoRA 解冻（rank16），训 30 epoch，其余同 exp324d。
4. **eval**：decorr-DINO part-MaxSim（heavy/all） + 复用 `exp324h_lora_oracle.py` 的 **adapted-DINO ⊕ Swin oracle + fusion sweep**，看 fusion 重遮挡/全部能否 **> Swin 72.57/75**。

## 预期结果
- **假设成立（真 method！）**：λ>0 使 top-10 Jaccard 明显下降、oracle 上界 gain >+1、fusion 重遮挡/全部真超 Swin。
- 失败最可能：(1) 线性解相关伤 DINO 判别力（mAP 掉）而不换来有用互补——因为 Swin 已占据最判别方向，正交补里的 ID 信号更弱；(2) 解相关是全局线性的、不针对 Swin 的遮挡盲点；(3) 95.8% 训练全可见墙——d 没机会专门学遮挡；(4) decorr 与 ID 冲突训不稳。
- **无论成败都有价值**：成 → 真 method；败 → 把"判别性-互补性张力"从观察升级为"显式施压也打不破"的强结论（诊断论文的关键对照实验）。
- **已知 scope 限制（Claude review Medium#1）**：缓存的 `s` 只是 Swin 的 **holistic global**（前 768 维），decorr 只把 DINO-global 推离 Swin-global，**没动 Swin 的 part 子空间**；而 tension 指标 / fusion eval 是**完整 MaxSim（global+parts）**。所以 null 结果可能是"张力打不破"或"只解相关了 global 不够"二义。v1 先做 global-vs-global（最干净的单变量介入）；若 λ>0 在全 MaxSim Jaccard 上有移动→值得再做 part-level decorr v2。设计里诚实标注，不夸大。

## 对照组与机器
- **λ=0**（=普通 LoRA，趋同 Swin；与 exp324d r16 同种子复现，同脚本走 decorr=0 分支数值应等价）vs **λ>0**——干净隔离解相关因果。
- 机器：λ=1 主跑 **lab-3090-d**（已空，3090）；λ=0 control 待 hyy r32 跑完后上 **hyy GPU1**；可加 λ=2 看单调性。
- 主指标：fusion(decorr-DINO ⊕ Swin) vs Swin 单独，重遮挡 + 全部。诊断：随 λ 增大 Jaccard / oracle-gain 怎么变（直接量化张力是否被打破）。

## 协议待办
1. [ ] `exp324i_swin_cache.py`（solider-reid env，缓存 train Swin 全局特征）+ dry-verify
2. [ ] `exp324i_lora_decorr.py`（复制 exp324d 加跨协方差 decorr loss）+ `--dry_run` 验证 loss/peak-mem，并验证 λ=0 数值≈exp324d
3. [ ] Claude broad review + Codex review（hook 阻断，训练前必须）
4. [ ] 双审查 approve → 训练 λ=1（lab-3090-d）；λ=0 control（hyy GPU1 待 r32 完）
5. [ ] eval + fusion-vs-Swin（复用 exp324h oracle）→ 判 method 成/败 → 更新 results/decisions + study
