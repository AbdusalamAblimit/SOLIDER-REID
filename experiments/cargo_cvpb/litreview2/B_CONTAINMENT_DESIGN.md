# 候选 B [★DEAD 2026-06-24 kill-switch FAIL]: 航拍-地面 ReID 的"物理定向不确定性包含"

> **⛔ 此方向已死**（零训练 kill-switch 三假设全证伪，cosine A→G 67.41≈训练67.33 sanity过/codex审脚本approve）:
> - ① **σ_aerial < σ_ground**(航拍q156.96/g167.47 < 地面q171.64/g172.81 双侧; 合成退化σ反降）——"航拍更欠定=宽分布"前提**根本错了**, 航拍低清=更平滑=低TTA方差。
> - ② 包含"+1.2"(KL68.62 vs cosine67.41)是假象: equal-var Maha(σ-free)也67.94>cosine, 对称分布距离全远低(56/55/44); 非对称是"均值除query方差"检索artifact(G→A崩17.37)。
> - ③ image-level σ 无用(C3 view-mean 69.07不降反升/C4 67.47/C5 66.63 ≈correct不掉)。分桶收益散非低像素。
>
> 全数据: `cvpb_containment_killswitch_design.md` / `cvpb_containment_full.log`。**下文为原设计, 留作止损记录。** 下步: 3-codex 战略 panel(救援/转向/残酷否决)。

（2026-06-24, 4-codex 红蓝裁判 panel 完整读后定稿; 全自主决策 —— 后被零训练 kill-switch 推翻）

## 一、4 codex 收敛结论（全部完整 Read, 0 截断）

| codex | 角色 | 核心 verdict |
|-------|------|-------------|
| v_1 | 蓝队杀手 | B **存活但裸想法被打惨**: 无直接撞车, 但机制三件套(高斯/方差不确定性/KL-偏序包含)全有强先例(Word2Gauss/Order-Emb/HIB/**PFE**/PCME/**Pr-VIPE**/LPP-ReID)。只能当"任务重定义"卖, 不能当"概率/KL 机制"卖。 |
| v_2 | 红队辩护 | **信心 7/10**(过双 kill-switch→8)。B > C/D(C 退化成"别做局部匹配", D 难证 view-confounder)。headline 必须是"揭示物理定向信息不对称", 不是"我们提出 Gaussian/KL"。**训练版必须 beat avg 52.37**(不只 MaxSim)。 |
| v_3 | 独立裁判 | 给了更硬的 re-frame: **观测受限身份可恢复性**(avg>MaxSim 真意=低清航拍局部 token 是欠定噪声, MaxSim 捞假局部匹配)。C **撞 AGPReID 近作 ViSA**。B 排第一但方向要修。 |
| v_4 | kill-switch 批判 | 裸 kill-switch 不够, 给了 **8 个破坏性对照**。建议**先做加固版 B**。 |

**三方独立确认的硬修正: 包含方向写反了。** 物理上"航拍证据 ⊂ 地面外观集合"; 概率候选空间上 = **地面窄分布落入航拍宽分布(G ⊂ A)**, 打分 `-KL(N_G ‖ N_A)`。原 SYNTHESIS 写的"航拍⊆地面"是反的(=惩罚不确定性, 和叙事冲突)。

## 二、定稿 re-frame（融合 v_2 命名 + v_3 物理内核）

> **Cross-view (aerial-ground) ReID 不是对称对齐问题, 而是物理定向的不确定性包含。**
> 航拍是受像素预算限制的欠定投影: 其身份兼容的地面外观构成一个**更宽的候选分布**; 正确的地面证据应**落在这个航拍不确定性包络内**。对称 cosine / late-interaction(MaxSim) 用了错误的匹配假设——把欠定的航拍局部 token 当成可独立匹配的身份证据, 于是从 gallery 捞偶然高相似的**假局部匹配**(这就是 avg 52.37 > MaxSim 45.19 的真因, 不是"该用 avg")。

**隐藏变量(可测、可证伪)**: 视角成像导致的**信息欠定度不对称** σ_aerial ≫ σ_ground, 且 σ 由物理量(bbox 面积/SMPL 投影身体像素/俯视比/分辨率)决定, 不是难度代理。

## 三、novelty 切开点（避开所有先例 + 红海, v_1/v_2 核查）

- **不写"概率/高斯/KL 机制"**(Word2Gauss/PFE/PCME/Pr-VIPE/LPP-ReID 全占)。写"AGPReID 物理定向信息不对称的任务重定义"。
- vs **PDA**(文本分布⊇图像, 方差=语义范围): 我们方差=**航拍成像欠定度**, 方向由相机高度/分辨率/人体投影面积定, 不是语言粒度。
- vs **OT-ReID/CM-EMD**: 不求对齐/搬运, 而是**接受航拍欠定, 用非对称覆盖打分**。
- vs **AGPReID 红海**(VDT 解耦/GSAlign TPS+可见性/SeCap prompt/DTST token选择/**ViSA 视角特有线索**): 避开几何/可见性/局部选择; **SMPL 只当诊断物理欠定度, 不当主模块**。
- vs **cross-resolution ReID**(PS-HRNet/RFD 恢复/不变特征): 不幻想补不可见细节, 而是**显式表达候选身份范围**。
- vs **C(对齐伤判别性)**: 撞 ViSA, 降级。vs **D(因果)**: view-confounder 难证, 缓。

## 四、★零训练 kill-switch 协议（加固版, v_4 主导 + v_2/v_3 补）

冻结同一 Swin(swin_fix256, 67.33), 不训练, CARGO A→G。三条核心假设, 任一不成立 B 降级:

**假设1 — 航拍确实更欠定**: σ_A ≫ σ_G, 且 σ 随物理量变化。
- σ 来源: TTA / token / augmentation variance（只来自图像, 不用 ID label）。
- 诊断: trace(σ_A) 显著 > trace(σ_G); σ 与 bbox面积↓/SMPL投影像素↓/分辨率↓ 相关。
- **合成退化正控**: 地面图 downsample/blur/遮挡 → σ 必须单调上升(否则 σ≠信息欠定)。

**假设2 — 正确方向有效**: `-KL(N_G‖N_A)` 明显优于 cosine, 且优于最佳对称分布距离。
- 主比较: `cosine(μ_A,μ_G)` / `sym KL·JS·Bhattacharyya` / `-KL(G‖A)`正向 / `-KL(A‖G)`反向 / `equal-var Mahalanobis`。

**假设3 — 收益来自图像级非对称包含, 不是混杂**: 8 个破坏性对照全部必须掉分:
1. 方向破坏: G⊂A 必须 >> A⊂G(反向接近→死)。
2. 对称化破坏: sym-KL/JS 一样好→收益是 distribution metric 非包含。
3. view-median 方差: 航拍全换航拍均值方差→不掉=只是 view prior。
4. 同视角方差置换: 同视角内随机换 σ→不掉=非图像级不确定性。
5. hardness-matched 置换: 按 cosine难度/norm/分辨率分桶换 σ→不掉=σ只是难度代理。
6. 维度打乱: 图内 shuffle σ 维度→不掉=无维度级"证据范围"语义。
7. variance-only/norm-only baseline: 只用 σ/norm 打分→接近主方法=混杂严重。
8. 收益集中度: 按 SMPL投影像素/bbox面积/分辨率分 4 桶, 最小航拍桶包含收益最大, 高清/同视角弱→全桶平均涨=普通 metric trick。

**覆盖校准图(v_2)**: 每个航拍 query 高斯 envelope, 同 ID 地面 positive 是否落在 50/80/95% 覆盖区间, 对比 hard negative。

**通过标准**: -KL(G‖A) > cosine 且 > 最佳对称; 正向 >> 反向; 对照3 的 1/3/4/5/6/7 全明显掉; σ_A≫σ_G 且合成退化单调升 σ; true-pair 包含距离 << impostor 且差距在破坏对照里消失。
→ 全过 = 隐藏变量证实, B 是我们的 B 类 re-frame; 任一关键条不过 = B 降级, 转 v_3 的"可恢复性"变体或换方向。

## 五、训练版门槛（kill-switch 过后才做, v_2）
方法朴素即可: mean+variance head + directional containment loss + 物理欠定正则 + 包含检索分。
**判据: 单 seed ≥ +1.0 mAP over avg 52.37, 低清 A→G 分桶 +2~3 mAP。** 体量来自诊断证据+破坏对照+两数据集(CARGO+AG-ReID.v2), 不是模块复杂度。

## 六、headline
> We reveal a **physically directed information asymmetry** in aerial-ground person ReID, and show that symmetric alignment / late-interaction is the wrong matching assumption: the low-altitude aerial observation is an under-determined projection whose identity evidence must be matched by **directional containment**, not symmetric similarity.

## 七、下一步
写零训练 kill-switch 脚本(复用 error_analysis_geom.py 基建: frozen Swin 提 μ + σ, CARGO A↔G, 8 破坏对照 + SMPL 分桶)。GPU: lab-3090(CARGO + swin_fix256 ckpt 在)。先双审? —— 这是零训练诊断脚本不是训练, 但仍走 codex 审一遍脚本正确性再跑。
