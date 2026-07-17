# GOPL 零训练 kill-switch 设计（2026-06-24, B containment 死后转向; novelty 7/10 存活）

## re-frame（定稿，绕开所有红海）
> 大家以为跨视角/遮挡 ReID 难在"特征对齐不够"，其实是 **正样本关系粒度错了**：ReID 训练把同一 ID 的所有 pair 当**等价**正样本强拉近，但两张图**共同可观测的人体表面**可能几乎不重叠（一张遮腿一张遮躯干→共同只剩头），此时"同一身份"是**过强监督**，会把不可对齐的远程正边硬拉，污染 identity manifold。
> **GOPL**：SMPL 共同可见表面 overlap 只当 **same-ID positive edge 的可靠性度量**——高 overlap 早期强拉、低 overlap 延迟/弱约束/桥接。SMPL **不进身份表征、不做推理匹配、不做 part alignment、不做增广**（绕开 exp333 SMPL-β≈random + VPM/QPM/visibility 红海 + GSAlign/ViSA aerial-ground 红海）。

## novelty caveat（codex 7/10, 必须遵守）
- 最危险撞车 **VPM/PVPM/QPM**（pair-wise 共同可见区域匹配/对齐）。切口=训练监督关系重定义，**不是**匹配/对齐。
- 论文**不能写**: 首次发现正样本不等价 / 首次用 visibility / 首次用 SMPL / 新 pose alignment。
- 论文**能写**: "revisit supervised ReID from positive-relation granularity"; "same-ID labels correct but not uniformly reliable under disjoint visible surfaces"; "SMPL only as geometry reliability meter, not identity rep / alignment"。
- **实验硬门槛**: 必须 beat ① random/loss-self-paced ② feature-distance hard/moderate positive mining ③ **2D-part-visibility / VPM-QPM-style overlap**。③最关键——若 2D 关节可见性解释得一样好, GOPL=老visibility换名, 死。

## 零训练 kill-switch（先验隐藏变量存在 + 证 SMPL 几何独特）

数据 occluded_duke（SMPL 缓存现成）。冻结一个**强 occluded_duke ReID ckpt**（agent 找最强的, 如 exp255/PSG 系或 exp341base; 报 mAP sanity）。SMPL: `cache/smpl_geom/{train,query,gallery}.npz` = 71 关节 pj2d(2D)+j3d(3D)+conf+valid。

**共同可见度量（多版本, 互为对照）:**
- `vis_i` = 关节可见性向量（conf > 阈值, 或 conf 连续）。
- `cov2d(i,j)` = 2D 关节共同可见 IoU = |vis_i ∩ vis_j| / |vis_i ∪ vis_j|（**这就是 VPM/QPM 式 2D-visibility 对照**）。
- `cov3d(i,j)` = 用 j3d + 相机朝向估每关节"表面是否朝向相机可见"（自遮挡), 再算共同可见 IoU（**这才是 GOPL 主张的 SMPL-3D**, 若 verts 需要 agent 按需重算 ROMP mesh; 关节级先做近似）。

**核心测试（query=遮挡, A→G occluded_duke 标准协议）:**
1. **同 ID pair 距离 vs overlap**: 所有 same-ID pair, frozen cosine 距离 vs `1-cov`。Spearman, 控遮挡程度(可见关节数)/bbox。预期: cov3d 的负相关显著, 且强于 cov2d。
2. **overlap 分桶**: same-ID pair 按 cov 四分位, 看 cosine 距离均值是否低 overlap 桶明显更大。
3. **query AP 分桶**: 每 query 按其 true-match 的 max cov 分桶, 看底桶 mAP/AP 是否显著低于顶桶。
4. **hard-positive tail**: cosine 距离最大的 top-k% same-ID pair(最难正样本)里, 低 cov 占比是否显著高(证现 loss 在强拉不该早拉的边)。

**破坏性/对照（每个都要报, 决定 novelty 生死）:**
- D1 置换 cov（shuffle overlap 值）→ 相关性必须消失。
- D2 **bbox/相机对** 当 overlap 代理 → SMPL cov 必须解释更多(更高 Spearman / 更大桶 gap)。
- D3 **feature-distance 本身**当难度 → SMPL cov 必须在 partial correlation 里**额外**加信号(控住 feature-distance 后 cov 仍解释)。
- D4 **cov2d vs cov3d** → cov3d 必须明显强于 cov2d, 否则 GOPL=VPM/QPM 换名(致命)。
- D5 随机关节子集当"可见" → 必须掉。

**通过标准:** (核心1-4成立) + (D1置换破) + (D2 SMPL>bbox/cam) + (D3 SMPL beyond feature-dist) + **(D4 cov3d>cov2d 明显)**。D4 不过 = GOPL 没独特性, 降级。全过 = GOPL 隐藏变量真实且 SMPL 几何独特, 写训练版(geometry-ordered positive graph + 课程权重, beat random/self-paced/feature-hard/2D-vis 四对照)。

## 不过的退路
- D4 不过(cov3d≈cov2d): GOPL 降级。转 p_3 遮挡 source-separation(donor-ID probe) 或 p_2 备选 SCEI(surface ACE)。
- 核心1-4 不成立(距离与 cov 无关): 整个"正样本关系粒度"假设错, 换方向。

## 资产
SMPL 缓存 lab-3090 `/root/work/SOLIDER-REID/cache/smpl_geom/`。occluded_duke `/root/work/SOLIDER-REID/data/occluded_duke`。强 ckpt agent 在 `log/occluded_duke/` 找。复用 error_analysis_geom.py / cvpb_containment_killswitch.py 的 extract/per_query_ap 基建。
