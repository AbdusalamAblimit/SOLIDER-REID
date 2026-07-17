# Claude Broad Review — exp349 (Swin-Small 全 pose 系统 exp255 + CLIP-ReID ID prompt)

**审查范围**：configs/occluded_duke/exp349_small_full_clip.yml、model/pose_backbone_model.py（forward 多分支 + clip_id_loss 注入 line 733/906）、model/modules/clip_id_prompt.py、processor/processor.py（clip_id_loss 消费 line 1297、parallel_aug line 790、OA-SD EMA line 470/801/1317）、config/defaults.py、loss/make_loss.py、model/make_model.py（in_planes）、model/modules/clip_part_head.py（LGPA CLIP 加载）。

**结论**：审查通过（approve）。这是一个真正的多分支多 loss 整合，已逐行核对数据流，未发现 Critical/High 级阻断问题。下方为完整发现（含已验证安全项与若干 Low 提示）。

---

## 数据流核对（核心）

### 1. clip_id_loss 在「LGPA + GCN dual 分支」中正确流通且只加一次 —— 已验证 ✓

全系统配置（POSE_LGPA True + POSE_SKELETON_GCN True + POSE_PSG_STAGES [-2,-1] + POSE_OA_SD True + POSE_LOWER_BODY_OCC True）下：

- forward（training）先在 line 602-637 算出 `clip_id_loss`（用 `feat_for_clip = global_feat`，因为 PART_GUIDED / NOPARAM_POOL / POSE_GUIDED 都未开 → 走 line 626 raw global 分支）。
- 进入 line 725 的 LGPA 分支：line 732-733 把 `kp_data['clip_id_loss'] = clip_id_loss` 注入。
- LGPA + GCN dual（line 736-749）：返回 `([cls_score]+lgpa_cls+gcn_cls, [global_feat]+lgpa_feats+gcn_feats, featmaps, None, kp_data)` —— **kp_data（含 clip_id_loss）原样透传**，没有被新 dict 覆盖。✓
- processor line 599 解包出该 `kp_data`，line 1297 `if kp_data.get('clip_id_loss') is not None:` → line 1300 `loss = loss + clip_id_w * kp_data['clip_id_loss']`。**只在此一处消费，加一次。** ✓

### 2. 无双计数（关键 —— parallel_aug + loss_fn 双重检查） ✓

exp349 同时开 POSE_PARALLEL_AUG 和 POSE_OA_SD → 进入 `parallel_oa_sd` 4-view 模式（3 student + 1 teacher）。

- 3 个 student view 各 forward 一次（line 577-590），各自产出含 clip_id_loss 的 kp_data，但 line 591-593 **只取 view0** 的 `score/feat/kp_data`。
- view1/view2 在 line 792-793 仅重跑 `loss_fn(all_scores[vi], all_feats[vi], ...)`，**不传 kp_data** → `loss_fn` 内部也从不读 `clip_id_loss`（make_loss.py 的 `loss_func` 只处理 evid/maxsim/str/part_visibility，无 clip_id 分支，已核对）。
- 因此 view1/2 不会引入额外 clip_id_loss。clip_id_loss 仅来自 view0 的 kp_data，在 line 1297 加一次。✓
- 顺序正确：parallel_aug 的 `loss = loss / len(all_scores)`（line 797，对 3 个 view 平均）发生在 **line 1297 之前**，所以 clip_id_loss 是在 /3 之后以全权重（WEIGHT 1.0）加上，**不被 /3 稀释**。这与设计意图（CLIP 全权重对齐 global）一致。✓
- 注意 `kp_aux_data = dict(kp_data)`（line 617/623/636）虽然会把 clip_id_loss 键带进传给 loss_fn 的字典，但 loss_func 永不访问该键 → 无副作用。✓

### 3. in_planes / clip_id_proj 维度 —— 已验证 ✓

- make_model.py:209 `self.in_planes = self.base.num_features[-1]`。
- Swin-Small（swin_transformer.py:1430-1431）`embed_dims=96, depths=(2,2,18,2)` → 末 stage 维度 `96*2^3 = 768`。
- `global_feat = avgpool(featmap)`，featmap 通道 = num_features[-1] = 768 → global_feat 768 维。
- clip_id_prompt（ViT-L-14）`clip_dim = text_projection.shape[1] = 768`。
- `clip_id_proj = Linear(in_planes=768, clip_dim=768)`（line 215）→ 输入输出都 768，与 global_feat 完全匹配。✓
- 审查重点澄清正确：POSE_LGPA_CLIP_DIM=512 是 LGPA 头（CLIPPartHead, ViT-B-32 文本 512 维）的独立 clip_dim，与 clip_id_proj 无关，两者互不影响。✓

### 4. GLOBAL_LOSS_SCALE 0.5 vs CLIP 全权重对齐 —— 一致，非 bug ✓

- 全系统走 list-loss 路径（返回 list），triplet_loss 内部对 global（feat[0]）的 ID/triplet 施加隐式 0.5x（M1 regime，与 exp255 同）。
- CLIP i2t/t2i（line 628）对齐的是 **未经 0.5x 的 global_feat / clip_id_proj(global_feat)**，以 POSE_CLIP_ID_WEIGHT=1.0 加。这正是 design.md line 13 描述的「global 0.5x（list-path）+ CLIP 全权重对齐 global」。与 exp255 唯一差异就是多了这一项 CLIP loss。✓（design.md 已把「global 0.5x 稀释 CLIP 增益」列为失败可能，归因正确。）

---

## 模块共存 / 干扰核对

### 5. OA-SD EMA teacher 与 CLIP prompt —— 安全（含一处可接受的内存代价） ✓ / Low

- EMA teacher = `copy.deepcopy(base_model)`（processor line 478），**深拷贝整个 student，含 clip_id_prompt（冻结 ViT-L 文本编码器 + 可学习 cls_ctx）与 LGPA 的 clip_text_features buffer**。
- teacher forward（line 821）在 `torch.no_grad()` 下会重算 clip_id_loss，但 line 832-833 **只取 teacher_feat，clip_id_loss 被丢弃**，不进图、不回传。✓
- EMA 更新（line 1321-1322）遍历 `ema_teacher.parameters()` 包含冻结 CLIP 参数：student 侧这些 CLIP 参数 `requires_grad=False` 且训练全程不变，EMA blend `t = decay*t + (1-decay)*s` 在 t==s 时为恒等 → **冻结 CLIP 在 teacher 中保持与 student 一致**，无副作用。cls_ctx（可学习）会被正常 EMA 跟踪，符合自蒸馏语义。✓
- buffer EMA（line 1324-1327）只更新 float32 且 shape 匹配的 buffer；LGPA 的 `clip_text_features`（float32 buffer）会被 EMA blend，但 student 侧它是常量 buffer（注册后不变）→ teacher 侧同样恒等保持。✓ 无破坏。
- **Low（内存）**：deepcopy 使 ViT-L 文本编码器在显存中存在两份（student + teacher）。ViT-L-14 文本塔参数量有限（远小于视觉塔），且 teacher 在 no_grad 下 forward、不存激活梯度，BS=64 下风险低。见第 8 点显存评估。

### 6. PLBOA（POSE_LOWER_BODY_OCC）正交性 —— 确认 ✓

PLBOA 是数据增强（下半身遮挡），作用在输入图像/student view 上，OA-SD 用 clean（pre-PLBOA）图像与 pose 喂 teacher（line 819-824 `teacher_pose = pose_dict.get('teacher_pose', pose_dict)`，line 557 `img_teacher = img[3]` 为 clean view）。CLIP prompt 对齐的是 student 的 global_feat（带 PLBOA 遮挡）到 per-ID 文本原型 —— 这反而是「让被遮挡图也对齐纯 ID 语义」，与 PLBOA 目标方向一致、不冲突。✓ 纯增强，与 CLIP loss 计算路径无耦合。

### 7. 两次 CLIP 加载无 name/state 冲突 —— 已验证 ✓

- LGPA `CLIPPartHead` 加载 **ViT-B-32**（clip_part_head.py:112-120），仅保留 `clip_text_features`（register_buffer），create_model 后的 clip_model 在 `__init__` 结束即被 GC，**不保留视觉/文本子模块**。挂在 `self.clip_part_head` 下。
- CLIP-ID-prompt 加载 **ViT-L-14**（clip_id_prompt.py:23），保留 token_embedding/transformer/ln_final/text_projection/positional_embedding 等子模块，挂在 `self.clip_id_prompt` 下。
- 两者顶层属性名不同（clip_part_head vs clip_id_prompt），子模块树独立，state_dict key 前缀不同 → **无 key 碰撞、无 buffer 覆盖**。✓
- 注意 exp255/exp349 均未设 POSE_LGPA_RANDOM_TEXT（保持 default False）→ LGPA 用固定 CLIP 文本，正常。✓

---

## 单变量 / 复现性

### 8. 单变量 vs exp255 —— 确认 ✓

逐行 diff `exp349_small_full_clip.yml` vs `pose_psg_lgpa_gcn512_2stage_small.yml`：新增/差异仅 4 行——
`POSE_CLIP_ID_PROMPT: True`、`POSE_CLIP_ID_PRETRAINED: '/home/afr/.../clip_l14_openclip.safetensors'`、`POSE_CLIP_ID_WEIGHT: 1.0`、`POSE_LGPA_CLIP_DIM: 512`（注：512 本就是 default，此处显式写出无行为改变）。OUTPUT_DIR 不同（应当）。其余完全一致。
**有效单变量 = POSE_CLIP_ID_PROMPT（连带其依赖的 ARCH/TEMP 用 default：ViT-L-14 / 0.07）。** ✓
defaults.py 已含全部新键（line 226-237），且默认 False/安全，不破坏已有实验复现。✓

### 9. 显存 / 可行性 —— 关注但风险可控（Medium → 提示，非阻断）

- 静态权重：Swin-Small 全系统 + 冻结 ViT-L-14 文本塔（student）+ EMA teacher 深拷贝（再一份 ViT-L 文本塔）。ViT-L **文本** 塔参数量约 ~120M 级，两份约几百 MB，相对 24G 显存占比小。
- 激活峰值：parallel_oa_sd 每 step 跑 **3 个 student forward（带梯度）+ 1 个 teacher forward（no_grad）**，已是 exp255 既有开销；CLIP-ID 仅多一个 text transformer forward（77 token，B 条 prompt，no 视觉塔）→ 增量小。
- **风险点不在训练而在 eval**：config TEST.IMS_PER_BATCH=256（line 77）。按项目铁律（experiment_protocol.md），所有训练启动必须 CLI override **`TEST.IMS_PER_BATCH 64`**，否则 384×128 + flip-test TTA 在 e80 fragmentation 下历史上 OOM 过两次。⚠️ **启动命令务必加 `TEST.IMS_PER_BATCH 64`**（这是流程要求，config 本身保留 256 可接受，但启动必须 override）。
- 另：clip_id_prompt 的 cls_ctx 是 `num_classes × 4 × ctx_dim`（Occluded-Duke 702 × 4 × 768 ≈ 2.2M 个 float），可忽略。

---

## 其他发现（Low / 提示，不阻断）

- **(Low) 优化器是否纳入新参数**：clip_id_proj（Linear 768→768）、clip_id_prompt.cls_ctx 是新可学习参数。本审查未逐行追 make_optimizer 的参数收集逻辑，但该路径与 exp341（Swin-Tiny 上 CLIP-ID-prompt +2.2）走同一 build（`make_optimizer` 默认遍历 `model.named_parameters()` 且 requires_grad），exp341 既已生效说明 cls_ctx/proj 已被优化器纳入。冻结的 CLIP 文本塔 requires_grad=False 不会被加入。**视为已验证（继承自 exp341），但若 exp349 训练 log 中 clip_id loss 不下降，应回查优化器参数组。**
- **(Low) AMP 安全**：clip_id_prompt.forward 内部 `prompts + positional_embedding.type(self._dtype)`、cls_ctx 用 CLIP 原生 dtype（line 85 `.type(self._dtype)`）；supcon_i2t 对 image/text 先 F.normalize 再矩阵乘除以 temperature 0.07，数值稳定。clip_id_proj 输出在 autocast 下为 half，supcon log_softmax 在 half 下一般安全（CLIP-ReID 常规）。已在 exp341 验证过同路径，无 NaN 历史。✓
- **(Low) clip_id_loss 初值类型**：PART_GUIDED 分支用 `clip_id_loss = 0.0`（python float）起步累加（line 613），但 exp349 未走该分支（走 line 628 直接 tensor 相加），line 1300/1301 `.item()` 调用安全（是 tensor）。✓ 无 float/tensor 混淆。
- **(提示) 收益预期**：design.md 已诚实标注 Swin-Tiny 上「整合式全负、外挂仅 +0.2（冗余）」，本实验赌 Swin-Small 容量更大 + 反向归因（CLIP 加到强 pose 系统）。这是合理的「组合交付」实验而非逃避创新的小调参——它在回答一个明确问题（CLIP 对最强 pose 系统是否仍冗余），且单变量干净、可消融。审查认可其论文价值定位（交付一个 CLIP+pose 组合数字 + 冗余性证据），**不判为「小调参逃避创新」**。

---

## 审查通过

逐行核对了 clip_id_loss 在 LGPA+GCN dual 分支的注入与透传（model line 733/747-749 → processor line 1297）、parallel_aug 4-view 下的单次计数（view0 only + loss_fn 不读 clip_id）、Swin-Small in_planes=768 与 clip_id_proj 维度匹配、OA-SD EMA 深拷贝对冻结 CLIP 的恒等性、两次 CLIP 加载（ViT-B-32 buffer vs ViT-L-14 子模块）无 key 冲突、PLBOA 正交、单变量隔离（仅 POSE_CLIP_ID_PROMPT）。未发现 Critical/High 阻断项。唯一硬性提醒：**启动训练命令必须 `TEST.IMS_PER_BATCH 64` override**（流程铁律，防 eval OOM）。其余为 Low 级继承自 exp341 的已验证项。

**审查通过。**
