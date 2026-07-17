# Claude Broad Review — exp347 (param-free de-occluded 对齐)

**审查范围**: git diff HEAD（config/defaults.py, model/modules/clip_id_prompt.py PoseWeightedPool, model/pose_backbone_model.py forward + __init__, configs/occluded_duke/exp347_noparam_deocc.yml）
**对照**: exp341（raw global 对齐, 59.8）/ exp343（Option A 有参数池化, 57.6）
**日期**: 2026-06-20

---

## 1. PoseWeightedPool 真无参数（核心断言）— 通过

逐行核对 `model/modules/clip_id_prompt.py:171-189`：
- `__init__` 只存 `self.pose_temp = float(pose_temp)`（Python float，非 buffer 非 Parameter）。
- forward 全是无参数算子：`flatten/transpose`、`F.interpolate`、`amax`、`softmax`、`einsum`。
- **AST 程序化校验**：类体内 `nn.Parameter / nn.Linear / nn.Conv / register_parameter / nn.Embedding / requires_grad` 命中数 = **0**。
- 形状：`featmap (B,C,H,W)` + `pose_heatmap (B,17,Hh,Ww)` → `tokens (B,N,C)`，`pose` 插值到 `(H,W)`，`vis (B,N)`，`w (B,N)` softmax over **dim=1（spatial N）**，`einsum('bn,bnc->bc')` → `(B,C)`。全部正确。
- 因此 `make_optimizer`（`solver/make_optimizer.py:7` 遍历 `named_parameters()` 跳过 `requires_grad=False`）对该模块**不新增任何参数组**。对齐梯度在池化内无可学习落点 → 只能沿 `w·tokens` 流入 `featmaps[-1]`。**这正是 Option A 吸收陷阱的针对性修复**。

## 2. 梯度路径流进 backbone（修复有效性）— 通过

- `_run_backbone_with_psg`（`pose_backbone_model.py:415-504`）返回 `outs`，**全程无 `.detach()`** 落在 featmap 上（唯一 `.detach()` 在 447 行，作用于 PosePrompt 的 `hm_with_bg` argmax，与本实验无关且 PosePrompt 未开）。`featmaps[-1] = outs[-1]` 是活的 backbone 张量。
- forward `pose_backbone_model.py:617-618`：`use_clip_id_noparam_pool and scene_heatmaps is not None` → `feat_for_clip = self.pose_weighted_pool(featmaps[-1], scene_heatmaps)`（**非 detach 输入**）。
- 完整链路：`clip_id_loss → img_proj(623) → clip_id_proj(Linear, 215) → pose_weighted_pool(无参数) → featmaps[-1] → backbone`。BACKBONE 拿到梯度，确认修复成立。
- 对照 Option A：`PoseGuidedPool`（120-121 行 `nn.Parameter query` + `nn.Linear k_proj`）会先吸收对齐 → backbone/global 拿不到 → 57.6。exp347 砍掉这两个参数，链路一致但无吸收点。

## 3. 对齐目标 = 纯 ID 原型（无 pose 污染）— 通过

- `pose_backbone_model.py:602-604`：`pose_vec` 仅当 `clip_id_prompt.pose_cond` 为真才非 None；exp347 **未设 `POSE_CLIP_ID_POSE_PROMPT`**（默认 False，defaults.py:234）→ `pose_cond=False` → `pose_vec=None` → `txt_proto = self.clip_id_prompt(label, None)` = **纯 ID 原型，不编码姿态**（与 exp341 字面相同）。
- 这正确隔离了 Option B 的失败模式（B 让 global 编码姿态 → 57.6）：exp347 只 de-occlude **图像端对齐特征**，目标端原型不动。

## 4. 描述子 = raw GAP global，池化 train-only — 通过

- exp347 路径返回（`pose_backbone_model.py:894-895`）：`cls_score, global_feat, featmaps, None, {'clip_id_loss': ...}`。携带的特征是 `global_feat`（raw GAP, 501-502 行），**不是** `feat_for_clip`。de-occluded 特征**仅**用于 CLIP 对齐 loss，绝不做描述子或 triplet anchor。
- test 路径（`897-901`，`else` 分支）：`POSE_TEST_FEAT='global'` → `test_feat = global_feat`。`pose_weighted_pool` 调用只在 `if self.training:` 块内（618 行），eval 完全不触达 → **train-only**，无测试期开销。

## 5. scene_heatmaps None fallback — 通过

`pose_backbone_model.py:617` 条件含 `and scene_heatmaps is not None`；为 None 时落到 `elif`（Option A，未开）→ `else: feat_for_clip = global_feat`（622 行）= 退化为 exp341 raw global 对齐。安全。

## 6. 单变量 vs exp341 — 通过（含一处需澄清的配置差异，已核实无功能影响）

`diff exp341 vs exp347` 共 4 处：(a) 注释/OUTPUT_DIR（无关）；(b) `+POSE_CLIP_ID_NOPARAM_POOL True`、`+POSE_CLIP_ID_POSE_TEMP 4.0`（即本实验变量）；(c) **`POSE_TEST_FEAT: equal_concat`(exp341) → `global`(exp347)**。

第 (c) 项乍看像第二个变量，但**逐行核实描述子等价**：
- exp341 关了 LGPA/GCN/PPA/VCSR（`POSE_LGPA:False`, `POSE_PSG_STAGES:[]`，无 skeleton/ppa/vcsr 开关）→ eval 中 `gcn_feats` 初始化为 None（904 行）后**从不被赋值** → `equal_concat` 的拼接块（`1026 if gcn_feats is not None`）**整段跳过** → `test_feat = global_feat`（901 行）。
- 即 exp341 的 `equal_concat` 实际**坍缩为 raw global**，与 exp347 的 `'global'` 描述子**逐元素相同**。
- 此外 exp343（Option A 对照）本身就用 `POSE_TEST_FEAT:'global'`，exp347 与其 test 设定一致。
- 结论：(c) 是把隐式坍缩**显式写明**，无功能差异。真实唯一功能变量 = `POSE_CLIP_ID_NOPARAM_POOL`（决定对齐用 de-occluded vs raw global）。**单变量成立**。

## 7. AMP / dtype 安全 — 通过（有先例）

- 训练全程 autocast（`processor/processor.py:573 amp.autocast(enabled=True)`，1304 `scaler.scale(loss).backward()`）。
- `pose` 显式 `.float()` → float32；`w`（softmax）float32；`tokens=featmaps[-1]` 在 autocast 下通常 float16 → `einsum('bn,bnc->bc', w_fp32, tokens_fp16)` 为混合 dtype。
- **先例证据**：Option A `PoseGuidedPool` 与 Option C `PoseGuidedPartPool` 用**完全相同**的 `torch.einsum('bn,bnc->bc', attn_fp32, tokens_fp16)`（135 / 167 行），且 exp343/Option C 均在同一 AMP 配置下**跑通**（exp343=57.6 已落库）。exp347 的池化是其严格子集（去掉 k_proj Linear），末尾 einsum 字面一致 → 继承同一已验证 AMP 行为。无新风险。

## 8. 创新性质疑（审查制度要求）

本实验改动 ~32 行 + 1 config，表面是"小改"，但**不是逃避创新的调参**：它是对"3 种 pose+CLIP 整合全失败"根因（参数通路吸收 / global 编码姿态）的**机制级反制**——用零参数池化强制对齐梯度进 backbone。属诊断驱动的受控验证，符合"机制层面有新意 + 证据可消融"。失败模式（de-occluded≈GAP 时增益被 wash）已在 design.md 预登记，结果可证伪。门槛达标。

---

## Findings by severity

- **Critical**: 无
- **High**: 无
- **Medium**: 无
- **Low**:
  - (L1) `POSE_TEST_FEAT` 由 exp341 的 `equal_concat` 改为 `global`。已逐行确认在 LGPA/GCN/PPA/VCSR 全关时两者描述子**逐元素相同**（gcn_feats 恒 None，拼接块跳过），故**无功能影响**；仅为澄清，非缺陷。design.md 与本审查均已记录该等价性，无需改动。
  - (L2) `PoseWeightedPool` docstring 形参注释写 `(B,17,Hh,Ww)`，依赖 COCO-17 关键点数；当前数据管线 scene_heatmaps 即 17 通道，一致。若未来换关键点集需同步，但当前无问题。

## 结论

**审查通过**。PoseWeightedPool 经 AST 校验确认真无参数；梯度路径 `clip_id_loss → clip_id_proj → 无参数池化 → featmaps[-1] → backbone` 完整无 detach，构成 Option A 吸收陷阱的有效修复；对齐目标为纯 ID 原型（pose_cond=False，无姿态污染，隔离 Option B 失败模式）；描述子为 raw GAP global、池化严格 train-only；AMP 混合 dtype einsum 有 exp343/Option A 同模式跑通先例；与 exp341 的唯一功能变量为 `POSE_CLIP_ID_NOPARAM_POOL`（`POSE_TEST_FEAT` 差异已证描述子等价，非第二变量）。无 Critical/High/Medium 问题，2 项 Low 均为澄清性、无需修复。可进入 Codex 审查。

## 变体
exp351 = un-detach + de-occluded config 组合, 复用 exp342b+exp347 已审代码. 审查通过.
