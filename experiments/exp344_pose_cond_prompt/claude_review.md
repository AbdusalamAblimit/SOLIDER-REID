# Claude Broad Review — exp344 (Option B: pose-conditioned prompt)

**Date**: 2026-06-20
**Reviewer**: Claude (Opus broad review)
**Scope**: 全范围审查（design.md / clip_id_prompt.py / pose_backbone_model.py / config / processor / 对照隔离 / test-time 泄漏 / AMP）
**Review round**: v1

## 审查对象
exp344 在 exp341（已证 +2.2 的 CLIP-ReID 可学习 ID prompt）基础上，让 per-image pose 调制 prompt context，
使 ID 文本原型 pose-aware。代码已在 commit `6c62cd1`，本次审查针对该 commit 的 diff（无未提交改动）。

变更面：
- `config/defaults.py` +1 行：`POSE_CLIP_ID_POSE_PROMPT = False`
- `model/modules/clip_id_prompt.py`：`CLIPIDPromptLearner.__init__` 加 `pose_cond/pose_dim`，建零初始化 `pose_encoder`；`forward(label, pose=None)` 加 pose_delta。
- `model/pose_backbone_model.py`：构造时传 `pose_cond`；forward 算 `pose_vec=scene_heatmaps.mean(dim=(2,3))` 传入。
- `configs/occluded_duke/exp344_pose_cond_prompt.yml`：= exp341 + `POSE_CLIP_ID_POSE_PROMPT: True`。

---

## 逐点核验（对应审查 prompt）

### 1. clip_id_prompt.py 改动

**(a) 零初始化 → step 0 == exp341 ✅**
`nn.init.zeros_(self.pose_encoder[-1].weight)` + `nn.init.zeros_(self.pose_encoder[-1].bias)`（line 74-75）使最后一层输出恒为 0。
forward 中 `pose_delta = pose_encoder(pose)`，ReLU 之后过零权重/零偏置的 Linear → 严格为 0 张量 →
`cls_ctx = cls_ctx + 0`，与 exp341 逐元素一致。**起点 = 已证 +2.2 的 baseline，只能从那里往上学，初始化不会回退。** 结论成立。

**(b) view shape 匹配 ✅**
encoder 末层 `nn.Linear(ctx_dim, _N_CLS_CTX * ctx_dim)` 输出 `(B, _N_CLS_CTX*ctx_dim)`，
`.view(b, _N_CLS_CTX, self.ctx_dim)`（line 83）严格对上（_N_CLS_CTX=4）。
注意：exp344 实际用 **ViT-L-14**（config line 28 指向 clip_l14 权重，design 也注明），故 ctx_dim=768，
encoder = Linear(17,768)→ReLU→Linear(768, 4*768=3072)。审查 prompt 里的 "ctx_dim" 是符号，实际 768，无碍。
`self.ctx_dim` 在 `__init__` 被正确存为 `clip_model.token_embedding.weight.shape[1]`（line 26），view 用的就是它，自洽。

**(c) dtype：无失配 ✅（关键，已展开 AMP 分析）**
- `pose.float()` → fp32 进 encoder。训练在 `amp.autocast(enabled=True)` 下（processor line 573），
  autocast 会把 Linear 自动转 fp16 算，输出 fp16。
- `.type(self._dtype)`（line 83）显式把 pose_delta 转成 CLIP token 的 `_dtype`（= `cls_ctx` 创建时的同一 dtype，line 62/27）。
- 因此 `cls_ctx + pose_delta`（line 84）**两侧 dtype 恒等**，与 `_dtype` 究竟是 fp16 还是 fp32 无关——
  因为是用同一个 `self._dtype` 对齐的。
- 加完的 `cls_ctx` 进入 `torch.cat([prefix, cls_ctx, suffix])` → 冻结 transformer。prefix/suffix 也是 `_dtype` buffer，
  positional_embedding 在 line 89 同样 `.type(self._dtype)`。**这条 "CLIP-`_dtype` 张量进冻结 transformer（autocast 下）" 的路径，
  exp341 已经训到 e120 成功**，exp344 只是在进 transformer 之前多加一个已对齐 dtype 的 delta，未引入任何新的 dtype 边界。无失配风险。

**(d) pose_encoder 可训练 + 进优化器 ✅**
`pose_encoder` 是普通 `nn.Sequential`，参数默认 `requires_grad=True`；零初始化只改值不改 grad flag。
`make_optimizer`（solver/make_optimizer.py line 7-23）遍历 `model.named_parameters()`，凡 `requires_grad` 即纳入，
按 SGD 同 BASE_LR/WEIGHT_DECAY 加入 param group。pose_encoder 会被训练、会收到梯度。
（细节：零初始化末层在 step 0 输出 0，但梯度对其权重非零——supcon loss 经 cls_ctx 回传给 pose_delta，再回传给末层权重——
所以它能"离开"零点开始学，不会卡死在 0。这是零初始化残差/适配器的标准行为，正确。）

### 2. pose_backbone_model.py 改动

**(a) scene_heatmaps (B,17,H,W) → mean (B,17) ✅**
`_prepare_pose` 文档与实现明确 `scene_heatmaps: (B, 17, H, W)`（merge_person_heatmaps element-wise max over persons，
pose_utils.py line 24-32）。`scene_heatmaps.float().mean(dim=(2,3))`（model line 591）→ `(B,17)`，对上 pose_dim=17。✅

**(b) scene_heatmaps None → pose_vec None → 退回 exp341 ✅**
line 591-592 的三元条件：`(pose_cond and scene_heatmaps is not None)` 才算 pose_vec，否则 None。
`forward(label, None)` 时 `self.pose_cond and pose is not None` 为假 → 不加 delta → 行为 == exp341。优雅回退成立。
（本 config POSE_ENABLED=True 且无 canonical fallback 问题，scene_heatmaps 正常非 None；但 None 分支仍是安全网。）

**(c) ★ per-image（非 per-ID）原型 vs SupCon 假设 —— 关键风险分析**
这是本实验最值得想清楚的点。结论：**不破坏 supcon_i2t 的正确性，且在语义上是合理的（甚至是本方法的卖点），但有一个需要承认的张力，建议训练时盯日志。**

机制回顾（clip_id_prompt.py line 98-108）：
`supcon_i2t(image_feat, text_feat, labels, t)`：logits = image_feat @ text_feat.T / t（B×B），
mask = same-label，loss = 对每个 i，把所有 same-label 的 text_feat[j] 当正样本做 log_softmax。
i2t + t2i 对称两次（model line 595-596）。

exp341：text_feat[j] = ID 原型（仅由 label[j] 决定），同一 ID 的所有样本 → **完全相同**的 text_feat 行。
exp344：text_feat[j] = ID 原型 + 该图 pose_delta[j]，同一 ID 不同图 → **不同**的 text_feat 行。

为什么不破坏正确性：
1. **supcon_i2t 从不假设 text_feat 行按 ID 唯一**。它只用 `labels.eq(labels.t())` 定义正负，
   对每行 text_feat 独立 normalize、独立算相似度。text_feat[j] 仍严格对应样本 j（同序），
   img_proj[i] 仍对应样本 i。"text_feat[i] 是 labels[i] 的原型" 这句话在 exp344 仍成立——
   只是现在它是 "labels[i] 在 pose[i] 下的原型"，依然是 i 的合法正确目标。i2t/t2i 配对没有错位。
2. **i2t 方向**（image→text）：图 i 要把自己拉近 batch 内所有 same-ID 的 pose-conditioned 原型。
   这些原型因 pose 不同而散开，等于让 image_feat[i] 对齐 "同 ID 多种姿态原型的集合中心"，
   是更宽的对齐目标，不会要求 image_feat 去匹配一个唯一点——这反而**鼓励 global 特征对姿态更不变**
   （要同时靠近同 ID 的多个 pose-原型 → 学到姿态无关的 ID 判别方向）。这正是想要的（global 更判别/更鲁棒）。
3. **t2i 方向**（text→image）：pose-conditioned 原型 j 要拉近所有 same-ID 图像。同样合法。

需要承认的张力（不是 bug，是设计取舍）：
- **退化可能**：若 pose_encoder 学到把 pose_delta 做成 ID 内"实例指纹"（把每张图的原型精确对齐到那张图的 image_feat），
  i2t/t2i 会变得过于容易（正样本几乎一一对应），contrastive 信号塌缩，对 global 的正则变弱 → 退化为"不涨也不跌"。
  这与零初始化的好处一致：最坏 == exp341，**不会跌破已证 baseline**。所以风险是上行空间被吃掉，不是回退。
- **缓解已天然存在**：(i) pose_delta 是低秩来源（仅 17 维 pose 经一个瓶颈 MLP），表达力有限，难以编码"逐图指纹"；
  (ii) supcon 在 batch 内、跨多 ID 竞争，原型不能只顾自对齐还要与其他 ID 拉开；
  (iii) i2t 这一支强制 image_feat 靠近同 ID 的"多个"pose-原型，本身就抵抗指纹化。
- **判据可观测**：监控 `clip_id` loss（processor line 1301 已打印）。若 exp344 的 clip_id loss 比 exp341（8.7→2.83）
  **明显更低甚至趋近 0**，就是原型在指纹化/塌缩的信号，此时即便 loss 漂亮 global 也未必涨——
  应以 test.py global mAP 为准（design 判据正确，只看 global）。

综合判断 2(c)：**理论上 help 的概率 > hurt**——pose-aware 原型给 i2t 提供"同 ID 跨姿态"的更丰富对齐目标，
利于 global 学姿态不变的 ID 判别；零初始化保证下界 == exp341。唯一现实风险是"涨不动"（原型指纹化吃掉增益），
而不是"训崩/回退"。**可以放行训练，用 global mAP + clip_id loss 曲线判定。**

### 3. config 单变量 ✅
exp344.yml = exp341 配置 + 仅 `POSE_CLIP_ID_POSE_PROMPT: True`（line 27）。
其余（ViT-L-14 权重、GLOBAL_LOSS_SCALE 1.0、POSE_TEST_FEAT global、关 PSG/LGPA/PLBOA/OA-SD/多视图）与 exp341 一致。
单变量隔离成立。
（备注：yml 顶部注释是从 exp244 模板继承的旧文案，与 exp344 实际意图不符，纯注释不影响运行，建议清理但非阻断。见 Low-1。）

### 4. test-time train-only，无泄漏 ✅
- prompt 分支与 pose_encoder 仅在 `if self.training:` 块内被调用（model line 572 起，clip_id_loss 计算在 582-596）。
- eval 分支（line 868 起）：`POSE_TEST_FEAT='global'` → `test_feat = global_feat`（line 872，neck_feat='before'）；
  use_lgpa/use_vcsr/gcn 均关或 pose_test_feat==global 短路 → `gcn_feats` 保持 None → 跳过 996-1018 拼接 →
  `return test_feat（=global_feat）`（line 1020）。**eval 描述子 = 纯 global，完全不触碰 prompt/pose_encoder/pose_vec。** 无泄漏。
- 测试期 model 输出走 processor line 67-69 的 `feat, _ = model(...)` 二元解包，与训练期 5-元组解包互不干扰。

### 5. Option A / B 独立 ✅
- A = `POSE_CLIP_ID_POSE_GUIDED`（model line 219 读，控制 pose_guided_pool），
  B = `POSE_CLIP_ID_POSE_PROMPT`（model line 214 读，控制 prompt 的 pose_cond）。两个 getattr 独立。
- exp344.yml 只设 B=True；A 未出现在 yml → defaults.py False → **A 关**。
- 两条路径在 forward 互不依赖：A 改 `feat_for_clip`（line 585-588），B 改 `txt_proto`（line 591-593），
  即使两者同开也只是"pose-guided 图特征 对齐 pose-conditioned 原型"，无冲突；本实验仅 B，干净。

---

## Findings by severity

### Critical
无。

### High
无。

### Medium
- **M-1（设计取舍，非代码缺陷）**：per-image pose-conditioned 原型存在"指纹化塌缩 → 增益被吃掉"的上行风险（见 2(c)）。
  非阻断（零初始化保证下界 == exp341），但训练中**必须**以 test.py global mAP 为判据，并监控 clip_id loss 是否异常趋零。
  若 exp344 global ≯ exp341（59.8）且 clip_id loss 远低于 exp341，记为"pose 调制原型对 global 无净增益（原型自对齐塌缩）"，
  按红蓝队/decisions 止损，不在此路线堆变体。

### Low
- **L-1**：exp344.yml 顶部 4 行注释（"问题：CLIP 模块(LGPA-D)…基于 pose_psg_lgpa_detach.yml(exp244 原版)…
  判据 equal_concat(LGPA) vs global"）是从旧模板继承、与 exp344 实际意图（pose 调制 prompt）不符的残留文案。
  纯注释不影响运行，但易误导后续接手，建议改为 exp344 的真实判据（exp341 vs exp344，单变量 POSE_CLIP_ID_POSE_PROMPT，看 global）。
- **L-2**：`pose_vec` 用的是 **scene-merged**（target+distractor，element-wise max）热图的均值，故 pose_vec
  含 occluder/旁人姿态，不是纯 target-only。这与 design "每图 pose" 表述一致（scene 级），且对"pose-aware 含遮挡上下文"
  反而合理；仅提示：若未来想做 target-only pose 调制，需改用 target_heatmaps（已有，_prepare_pose 产出）。非问题，仅记录语义。
- **L-3**：`pose_vec` 未做归一化（直接 mean activation）。不同图的热图峰值尺度可能不同，但首层 Linear + 后续 ReLU/Linear
  可吸收尺度，且零初始化使其从无影响开始学，无需提前归一化。非阻断，若日后 pose_delta 训练不稳可考虑 LayerNorm(pose) 前处理。

---

## 数据流复核（一句话版）
`pose_dict.heatmaps → merge(max over persons) → scene_heatmaps (B,17,H,W) → mean(2,3) → pose_vec (B,17)
→ [autocast] pose_encoder → (B,3072) → view (B,4,768) → .type(_dtype) → + cls_ctx[label] (B,4,768)
→ cat(prefix,cls_ctx,suffix) → 冻结 CLIP text transformer → txt_proto (B,768)
→ supcon_i2t(img_proj, txt_proto) + supcon_i2t(txt_proto, img_proj) → clip_id_loss
→ processor: loss += POSE_CLIP_ID_WEIGHT * clip_id_loss → scaler.scale(loss).backward()`。
forward/backward 每一步 shape、dtype、optimizer 纳管均已核对，无断点。

## 与对照（exp341）隔离性
唯一差异 = pose_delta 注入路径（pose_encoder + line 82-84）+ pose_vec 计算（model line 591-592）+ 1 个 config flag。
exp341 的 5-元组返回、processor 消费、eval 路径完全复用未改。零初始化使二者在 step 0 数值恒等。单变量成立。

---

## Verdict

**审查通过**（放行训练）。

理由：代码层面无 Critical/High——零初始化正确（下界 == 已证 +2.2 的 exp341，不会回退）、shape/dtype 自洽且复用 exp341
已验证的 autocast+CLIP-`_dtype` 路径、pose_encoder 正确进优化器并可训练、test-time 严格 train-only 无泄漏、
Option A/B 独立且本实验仅 B、单变量隔离干净。

关键风险 2(c)（per-image 原型 vs SupCon）已分析清楚：i2t/t2i 配对不错位、supcon 不要求原型按 ID 唯一，
pose-aware 原型理论上更利于 global 学姿态不变判别；唯一现实风险是"原型指纹化塌缩 → 增益被吃掉（涨不动）"，
而非训崩或回退——这与零初始化的下界保证一致。

放行条件（已在 M-1 写明，须执行）：训练以 **test.py global mAP** 为唯一判据，监控 **clip_id loss** 是否异常趋零；
若 global ≯ 59.8 即按止损流程记录，不堆变体。L-1 注释残留建议顺手清理（非阻断）。
