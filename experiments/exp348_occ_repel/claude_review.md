# Claude Broad Review — exp348 (occluder repulsion)

**审查范围**: exp347 之上的增量（git diff HEAD）。exp347 = param-free de-occluded pooling 对齐纯-ID CLIP 原型；exp348 ADD occluder repulsion（把遮挡区/低可见特征推离 ID 原型）。
**审查方式**: 逐行读 `config/defaults.py`、`model/modules/clip_id_prompt.py`、`model/pose_backbone_model.py` diff + 上下文 + 两条 return 路径 + config 对照。
**日期**: 2026-06-20

---

## 1. 逐项核对（用户提出的 5 点 + 风险）

### (1) invert 符号正确性 — PASS
`clip_id_prompt.py:189`：`w = ((-vis if invert else vis) * self.pose_temp).softmax(dim=1)`。
- `vis = pose.amax(dim=1)` 是 person visibility（人体姿态热图在每个 token 的最大响应）。遮挡区 = 人体可见度低 = `vis` 小。
- `invert=True` → 用 `-vis`：低 `vis`（遮挡）→ `-vis` 大 → softmax 后权重高；高 `vis`（人体）→ `-vis` 小 → 权重低。
- 结论：符号正确，`invert=True` 确实把权重压在**遮挡/低可见**区域。仍然 param-free（无任何 learnable）。

### (2) 遮挡排斥 loss 项 — PASS（逻辑全部正确）
`pose_backbone_model.py:632-637`：
```python
if getattr(self, 'use_clip_id_occ_repel', False) and scene_heatmaps is not None:
    occ_feat = self.pose_weighted_pool(featmaps[-1], scene_heatmaps, invert=True)
    occ_proj = torch.nn.functional.normalize(self.clip_id_proj(occ_feat), dim=1)
    tp = torch.nn.functional.normalize(txt_proto, dim=1)
    repel = (occ_proj * tp).sum(1).clamp(min=0).mean()
    clip_id_loss = clip_id_loss + self.clip_id_occ_repel_w * repel
```
- (a) `repel = (occ_proj * tp).sum(1)` = 遮挡特征 proj 与 ID 原型的 cosine（两者都已 L2-normalize）。加进 loss 最小化它 → 把遮挡特征推离 ID。**方向正确**。
- (b) `clamp(min=0)`：只惩罚正相似度。sim≤0 时 loss=0、梯度=0 → 遮挡特征被推到「中性」(sim≤0) 即停，**不会被强推到 -1（相反方向）**。符合「neutral not opposite」意图。正确。
- (c) 梯度回流 backbone：`occ_feat` 来自 param-free pool 对 `featmaps[-1]` 的加权和（einsum），无 detach，`clip_id_proj` 是 learnable Linear；梯度经 `occ_proj → occ_feat → featmaps[-1]` 进 backbone，同时进 `clip_id_proj`。`txt_proto` 这条也有梯度（CoOp 的 `cls_ctx` 是 learnable，但 CLIP text encoder frozen）。梯度链完整。
- (d) `occ_repel_w=0.5`：主 supcon（628-629 两个 supcon_i2t 相加，量级 ~ O(1)~O(8) 交叉熵风格）远大于 `0.5 × repel`（repel∈[0,1]，故该项 ≤0.5）。不会压过主对齐。合理。
- (e) `torch.nn.functional`：文件顶部 `import torch`（pose_backbone_model.py 一定 import torch），`torch.nn.functional.normalize` 可直接用。PASS。验证：`clip_id_proj` 输出**未**预先 normalize（其内部 supcon_i2t 才 normalize），所以这里显式 normalize 是必要且正确的——与主对齐的几何一致（都在单位球面上算 cosine）。

### (3) __init__ gating + 单变量 — PASS
`pose_backbone_model.py:242-245`：`use_clip_id_occ_repel` / `clip_id_occ_repel_w` 设在 `if self.use_clip_id_noparam_pool:` 块**内部**（line 238 起）→ occ_repel 依赖 noparam pool 开启，且复用同一个 `self.pose_weighted_pool` 实例。
config 对照（exp347 vs exp348 diff）：仅多 `POSE_CLIP_ID_OCC_REPEL: True` + `POSE_CLIP_ID_OCC_REPEL_W: 0.5`（+注释/OUTPUT_DIR）。`POSE_CLIP_ID_NOPARAM_POOL: True`、`POSE_TEMP: 4.0`、`WEIGHT: 1.0` 均与 exp347 相同。**严格单变量**。PASS。
`config/defaults.py:236-237` 新增两个默认 `False` / `0.5`，对旧实验安全（默认关闭，不破坏 baseline）。

### (4) 姿态不准 → 遮挡区含真实人体 → 排斥真人特征 风险 — 已被 clamp 限幅，可接受
若 pose 不准，低 `vis` 区可能混入真实身体 token → `invert=True` 把这些真人特征也拉进 `occ_feat`，repel 会把它们推离 ID（有害）。但：
- `clamp(min=0)` 只「中性化」(推到 sim≤0)，不反转，伤害有上界。
- `w=0.5` 进一步缩小该项影响。
- 这是 **soft** 的（softmax 权重而非硬 mask），最坏情况是 occ_feat 含部分人体 → repel 略微稀释主对齐，而非灾难性反向。
风险等级：Medium（实验性 bet 的固有不确定，非 bug）。建议在 monitor.md 跟踪 `repel` 数值：若长期接近 0 → 遮挡区与 ID 本就正交（机制无效，NO-GO 信号）；若一直较大且 mAP 掉 → 姿态噪声主导（排斥伤了真人）。

### (5) exp348 仍做 exp347 de-occluded 对齐 — PASS
config `POSE_CLIP_ID_NOPARAM_POOL: True` 保留；forward line 621-622 走 `feat_for_clip = self.pose_weighted_pool(featmaps[-1], scene_heatmaps)`（invert 默认 False = de-occluded），628-629 主对齐照常。occ_repel（632-637）是在主 supcon **之后 ADDITIVE** 叠加。de-occluded 对齐未被改动。PASS。

---

## 2. Findings by severity

### Critical
- 无。

### High
- 无。

### Medium
- **M1（姿态噪声风险，见 (4)）**：pose 不准时遮挡区混入真人 → repel 稀释/轻伤主对齐。已被 `clamp(min=0)`+`w=0.5` 限幅，属实验固有风险，非 bug。建议 monitor.md 跟踪 `repel` 值与 mAP 曲线。
- **M2（pose dropout 与 invert 的交互）**：`pose_backbone_model.py:580-583` Stochastic Pose Dropout 训练时可能把某些 sample 的 `scene_heatmaps` 整张置零。此时 `vis` 全 0 → `-vis` 全 0 → `softmax(全0)=均匀权重` → `occ_feat` 退化为**整图 GAP-style 均值**，repel 会把这个「整图平均特征」推离 ID 原型。这与遮挡语义不符（把含大量真人的全局均值当遮挡推开），且同一 batch 里 de-occluded 分支对这些 dropout sample 也退化成均匀池化（与 raw global 接近）。影响有界（被 dropout 的样本占比 = `pose_dropout_p`，且 `w=0.5`、clamp 限幅），但确实是一处语义噪声。
  - 需确认 exp348 config 的 `pose_dropout_p`：若为 0（exp347/CLIP-ID 主线常见），M2 不触发，可忽略。建议复核 config 中 `POSE_DROPOUT`/对应键；若 >0，记入 design.md 已知交互，或考虑「dropout 置零的 sample 跳过 repel」（但为保持单变量，本次不建议改代码，仅记录观察）。

### Low
- **L1**：`repel` 在 `clip_id_loss is None`（无 noparam pool 或 `scene_heatmaps is None`）时不会执行——但 occ_repel 只在 noparam 块内启用，且与主对齐共享 `scene_heatmaps is not None` 守卫，逻辑自洽。若某 batch `scene_heatmaps is None`（无 pose_dict），整个 CLIP-ID 块连主对齐一起跳过，repel 自然不触发，行为一致。无需改。
- **L2**：`occ_feat` 与 `feat_for_clip`（de-occluded）共用同一 pool 实例、同一 `featmaps[-1]`、同一 `clip_id_proj`，仅 `invert` 不同——无状态污染（pool 无 buffer/param），两次调用独立。PASS。

---

## 3. train/test 对称性
- occ_repel 全部包在 `if self.training:` 分支下游的 loss 计算里（line 595 起为 training 分支；clip_id_loss 经 kp_data 在 732-733 / 906-907 携带，仅训练用）。**测试时不计算 repel、不改 descriptor**（descriptor 仍是 raw GAP global，与 exp347 一致）。train/test 对称无问题：推理路径零改动。

## 4. AMP 安全
- `pose.float()`（line 186）已显式 fp32 做 interpolate；softmax/einsum/normalize 在 AMP 下数值稳定。`clamp(min=0)` 无溢出风险。`(occ_proj*tp).sum(1)` 在单位球面上 ∈[-1,1]，无 inf/nan 风险（除非 occ_feat 全 0 → normalize 产生 nan？`F.normalize` 默认 `eps=1e-12` 防零除，安全）。PASS。

## 5. 日志充分性
- `__init__` 打印 `[CLIP-ID-Prompt] OCC-REPEL (exp348): ... w=0.50` 确认开关生效。
- 建议（非阻塞）：训练时把 `repel` 标量也打进 log（当前只并入 `clip_id_loss` 总量，无法单独观察遮挡排斥是否在工作/塌缩）。这对判断「机制是否真的在分离 visible/occluder」很关键。可在 processor 或 forward 里 detach 后塞进 kp_data 打印。**不阻塞训练**，但强烈建议加，否则 NO-GO 时无法归因。

---

## 结论

代码逻辑全部正确：invert 符号、clamp「中性不反向」、梯度回流 backbone、param-free、单变量、train/test 对称、AMP 安全均 PASS。无 Critical/High。两个 Medium（姿态噪声风险 M1 = 实验固有；pose dropout×invert 交互 M2 = 需复核 config 的 dropout_p，若 0 则不触发）属可接受/可观察范畴，非 bug。建议补「repel 标量单独打 log」以便归因（非阻塞）。

**审查通过**（建议训练时单独记录 repel 标量值；复核 config 的 pose_dropout_p 以确认 M2 是否触发）。
