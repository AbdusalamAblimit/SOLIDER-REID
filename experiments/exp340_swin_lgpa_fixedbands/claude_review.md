# Claude Broad Review — exp340 (Swin + LGPA fixed canonical-pose bands)

**审查范围**：design.md / config/defaults.py / model/pose_backbone_model.py（`_canonical_heatmap` / `_lgpa_heatmap` + 两处 forward 改动）/ configs/occluded_duke/exp340_swin_lgpa_fixedbands.yml / model/modules/clip_part_head.py（bias / assign KL / visibility 三路消费）/ processor.py（assign_loss 接线）。
**审查方式**：逐行读取所有改动，对照 commit `be92c5d~1 → be92c5d` 的 diff，验证数据流、shape、NaN 守卫、device/dtype、train/test 对称、单变量隔离。

---

## 核心结论先行

代码层面（canonical 热图生成 + 三路消费 + train/test 对称）**正确、无 shape/NaN/device 问题**，单变量隔离也成立。`_canonical_heatmap` 产出的 `(1,17,96,32) → expand(B,17,-1,-1).contiguous()` 完全满足 clip_part_head 对 `target_heatmaps (B,17,H,W)` 的契约，bias / assign-KL / visibility 三路均能正确流过，canonical 不触发 KL 的 `0*inf` NaN。

但有 **2 个 Critical 与运行/安全相关的问题必须先修**（与算法无关，但会污染历史实验、泄露密钥）：
1. exp340 的 `OUTPUT_DIR` 指向 **exp336 的日志目录**，一跑就覆盖 exp336 的 train log / checkpoint。
2. commit `be92c5d` 把一个真实 GitHub PAT（`ghp_…`，40 字符）以 `.git_token` 提交进仓库。

下面按严重度列出。

---

## Critical

### C1. `OUTPUT_DIR` 写错 → 覆盖 exp336 的日志和 checkpoint
- **文件**：`configs/occluded_duke/exp340_swin_lgpa_fixedbands.yml:80`
- **现状**：`OUTPUT_DIR: './log/occluded_duke/exp336_swin_lgpa_nopsg'`
- **问题**：这是从 exp336 复制 yml 时漏改的。train.py 会把 exp340 的 train log、`transformer_*.pth` 写进 **exp336 的目录**，直接覆盖对照实验（exp336 是本实验的 baseline 锚点 global=59.0，覆盖了就失去对照）。属于 `.claude/rules/experiment_protocol.md` 明令的"每个实验独立 OUTPUT_DIR"硬性违规，且 `documentation.md` 的"永远不用 train.py 评估会覆盖 train log"同源风险。
- **修复**：
  ```yaml
  OUTPUT_DIR: './log/occluded_duke/exp340_swin_lgpa_fixedbands'
  ```

### C2. 仓库里提交了真实 GitHub PAT
- **文件**：`/.git_token`（commit `be92c5d` 新增，`git ls-files` 已跟踪）
- **证据**：内容 40 字符、前缀 `ghp_`，是经典 GitHub classic PAT 格式。`.gitignore` 只忽略 `/viz_*.git_token`，根目录 `.git_token` 未被忽略。
- **问题**：密钥进 git 历史，任何能 clone 的人可读；`remote_server.md` 已记过 lab-3090 的 `.git/config` 明文 PAT 风险，这次直接把 token 文件提交进工作树，更严重。
- **修复**：
  1. 立刻在 GitHub 吊销/轮换这个 token；
  2. `git rm --cached .git_token` 并把 `.git_token`（或根级 `*.git_token`）加进 `.gitignore`；
  3. 该 token 已进 commit `be92c5d`，仅删文件不够，**历史里仍在**——吊销是第一要务，必要时 filter-repo 清历史。
- **备注**：这条与 exp340 算法无关，但出现在同一个待审 commit 里，按"宁可误报"必须拦下。

---

## High

（无）

代码、配置、数据流层面**没有发现 High 级问题**。下列为澄清性核查的结论（全部通过）：

- **Shape 契约**：`_canonical_heatmap` 返回 `(B,17,96,32)`。clip_part_head 三路全部先 `F.interpolate(..., size=(fH,fW))` 再用，对输入 H/W 不敏感，96×32 与真实 scene 热图同尺寸（yml `POSE_HEATMAP_SIZE:[96,32]`），无 mismatch。
- **`.contiguous()`**：`expand` 产生非连续视图，已显式 `.contiguous()`；下游 `interpolate` / `flatten(2)` / 高级索引 `hm[:, kp_indices]` 均安全。即便不加，interpolate 也会复制，但加了更稳，OK。
- **assign-KL 的 0*inf NaN 守卫**：canonical 的每个关键点都是 `exp(-...)`，峰值=1.0（中心处指数=0）。GT 每个 part 通道 = 对应 KP 组的 max → 至少在该 KP 中心格点处 =1，**没有全零通道**；background = `1 - body_max` 后 `clamp(min=0)`，body_max 峰值=1 → bg 最小=0（非负，clamp 安全）。GT 再 `/ sum(...).clamp(min=1e-6)` 归一化，每行 sum 远大于 0，不会除零。`log_attn_weights` 已 `clamp(min=-30)`，KL 前还有 `torch.isfinite(gt_labels).all()` 与 `torch.isfinite(raw_loss)` 双守卫。canonical **不会**触发 NaN。
- **body_max ≥ 1 ?**：canonical 峰值恰为 1.0（中心格点），不会 >1；`(1.0 - body_max).clamp(min=0)` 即使浮点误差到 1.0+1e-7 也被 clamp 兜住。pose_bias 路径同理。安全。
- **train/test 对称**：两处 `lgpa_hm = self._lgpa_heatmap(scene_heatmaps, x.shape[0], x.device)` 完全一致（train forward line 638、eval forward line 835），喂进 clip_part_head 的 `target_heatmaps` 是同一个 canonical。eval 的 LGPA 分支门控 `pose_test_feat != 'global'`，equal_concat 满足。**对称成立**。
- **flip-test 交互**：eval flip 路径（processor.py:74）对 `img_f` 仍调用同一 forward，canonical 与图像无关（image-independent），翻转图配同一固定 canonical，行为良定义、可复现，无对称破绽。
- **assign_loss 接线**：processor.py:1027 `lgpa_enabled and not ppa_enabled and 'assign_loss' in kp_data` → canonical 训练时 `self.training and target_heatmaps is not None` 成立，assign_loss 正常计算并以 `POSE_LGPA_ASSIGN_WEIGHT=0.5` 加权。canonical 是固定 GT，等于给所有图同一套"竖直布局软标签"——与 design 假设一致。
- **device/缓存**：`_canon_hm_cache` 在 CPU 建一次（`torch.zeros(...)` 默认 CPU），之后每次 `.to(device).expand(...)`。`.to(device)` 对已在该 device 的 tensor 是 no-op（不重复拷贝到同设备），缓存只建一次，正确。注意它是普通属性而非 `register_buffer`，所以 `model.to(device)` / `.half()` 不会动它——但因为每次调用都显式 `.to(device)`，device 始终正确；dtype 恒为 float32，clip_part_head 内 `_compute_pose_bias` / `_compute_gt_assignment` / visibility 都先 `.float()`，与 AMP 下 float16 featmap 的 bias 相加发生在 `_cross_attention_with_pose`：`attn_scores`(来自 featmap，可能 fp16) + `pose_bias.unsqueeze(1)`(fp32) → 触发上采 fp32，再 `clamp(-50,50)` → softmax，无 dtype 崩溃。安全（与 scene 热图路径完全同构，scene 热图本身也是 fp32）。
- **`_lgpa_fixed_bands` 与 `_lgpa_no_pose` 冲突**：`_lgpa_heatmap` 中 no_pose 先判，命中即 `return None`，不会与 fixed_bands 同时生效。exp340 yml 只设 `POSE_LGPA_FIXED_BANDS:True`、`POSE_LGPA_NO_POSE` 用默认 False，无冲突。两 flag 互斥逻辑虽未显式 `raise`，但短路顺序保证不会"两者都生效"。
- **单变量隔离**：`diff` 去注释/空行后，exp340 vs exp336 **仅多一行** `POSE_LGPA_FIXED_BANDS: True`（除 OUTPUT_DIR 的 C1 错误外）。detach / equal_concat / 0.5 global / 384×128 / SEED 等全同。单变量成立。

---

## Medium

### M1. yml 顶部注释与正文注释均为 exp336 残留，误导性强
- **文件**：`configs/occluded_duke/exp340_swin_lgpa_fixedbands.yml:1-4`（整段标题写的是"exp336 纯 LGPA-D 隔离实验"）、`:26` 行内注释 `# ★ 纯 LGPA-D`。
- **问题**：exp340 的实质是"用固定 canonical pose 替代 per-image pose"，注释却说"纯 LGPA-D / scene 热图忠实原版"，与 `POSE_LGPA_FIXED_BANDS:True` 的真实语义矛盾。`# POSE_USE_TARGET_HEATMAP 不设 → scene-merged 热图(忠实原版)` 这句在 fixed-bands 下尤其误导——scene 热图根本没被用到（被 canonical 顶替）。会让后人误判这是 scene-pose 实验。
- **影响**：纯文档/可读性，不影响运行，但违反 `documentation.md`"数据/描述必须与实际一致"。
- **修复**：把标题与 `:26`、`:37` 注释改写为 exp340 的真实意图（固定 CLIP 文本 + 固定 canonical 解剖先验，per-image pose 不参与）。

### M2. design.md 的对照锚点数字与前序实验需对齐确认
- **文件**：`experiments/exp340_swin_lgpa_fixedbands/design.md:21-23`
- **问题**：design 用 global=59.0 / no-pose part=58.8 / pose part=59.8 作为锚点，但 frozen 表（同文档:32-37）baseline 写 58.20。两套基线数字（59.0 vs 58.20）来源不同（一个像 equal_concat 训练端、一个像 frozen 池化），文档未注明各自口径，容易自相矛盾。
- **修复**：在 design 里标清 59.0 与 58.20 各自的 eval 口径（equal_concat 训练后 vs frozen 池化），避免成功判据 ">59.0" 与 frozen 58.20 混用。属文档严谨性，不阻断训练。

---

## Low

### L1. `.DS_Store` 被提交进 commit be92c5d
- commit 里含 `.DS_Store`（10244 bytes）。建议 `git rm --cached .DS_Store` 并确认 `.gitignore` 忽略 `.DS_Store`（macOS 噪音文件）。不影响实验。

### L2. canonical 关键点坐标为手工硬编码，缺少可视化自检
- **文件**：`model/pose_backbone_model.py:489-491`
- 17 个 KP 的归一化坐标是手填的（鼻 0.06、踝 0.95 等），左右对称性靠肉眼（如肩 0.36/0.64、踝 0.42/0.58）。数值上无 bug，但建议训练前 dump 一张 canonical 热图 PNG 自检布局（也正好做 paper figure 素材）。不阻断。

### L3. fixed-bands 与"诚实对照"在 design 已点出但未排期
- design.md:29 自己点出 reviewer 必问的"固定文本 vs random 文本 prototype（同 canonical 先验）"归因对照。这是本实验**成功后**证明增益来自"固定文本语义"而非"任何固定 query + 固定先验"的关键。目前未建对照实验号。建议 exp340 一旦 part>global，立刻补一个 random-text-prototype 对照（同 canonical），否则结论不可归因到"固定 CLIP 语义"。属后续排期，非本次代码问题。

---

## design.md 合理性判断（是否只是变相 pose trick / 小调参？）

- **不是变相 per-image pose trick**：canonical 热图对所有图**完全相同**（image-independent，`_canonical_heatmap` 不接任何 pose_dict 输入），确实把"固定文本 + 固定先验 = 全固定"这一命题做成了可证伪实验。判据清晰：part_only / equal_concat 超 global 即固定语义涨点，落 ≈ no-pose 即失败。**是一个干净的科学假设检验，不是小调参**。
- **创新门槛**：本实验本身是诊断性消融（验证"固定语义能否 standalone 涨点"），不是新机制；但它服务于一个明确的开放问题（fixed CLIP text 一直 standalone 不涨，根因是定位），用固定解剖先验替 pose 来隔离"定位"变量，方向正当，符合"证据层面讲得清/可消融"。
- **诚实风险（design 已自陈）**：固定先验对遮挡/非全身 crop 会误对齐，assign loss 可能把 attention 锁死在错位置 → 退化成 no-pose。这是预期中的失败模式，已在 design"失败最可能原因"写明，可接受。

---

## 验证清单（逐项结论）

| 检查项 | 结论 |
|---|---|
| `_canonical_heatmap` 产出 `(B,17,H,W)` | ✅ `(1,17,96,32)→expand→contiguous` |
| `.contiguous()` 是否需要 | ✅ 已加，下游安全 |
| train/test 用同一 canonical | ✅ line 638 / 835 完全一致 |
| assign KL 0*inf NaN | ✅ canonical 无全零通道，双 isfinite 守卫，不触发 |
| bg = 1-body_max 负值 | ✅ body_max≤1，clamp(min=0) 兜底 |
| device：CPU build → .to(device) 缓存 | ✅ 建一次，每次 .to() 同设备 no-op |
| dtype / AMP | ✅ canonical fp32，三路先 .float()，与 scene 路径同构 |
| 单变量 vs exp336 | ✅ 仅多 `POSE_LGPA_FIXED_BANDS:True`（除 C1 OUTPUT_DIR） |
| fixed_bands × no_pose 冲突 | ✅ 短路顺序保证不冲突 |
| flip-test 交互 | ✅ canonical 与图像无关，可复现 |
| OUTPUT_DIR 独立 | ❌ **C1：指向 exp336，会覆盖** |
| 无密钥泄露 | ❌ **C2：.git_token 提交了真实 ghp_ PAT** |

---

## 需修复项汇总（阻断训练）

- **C1**：`exp340_swin_lgpa_fixedbands.yml:80` 的 `OUTPUT_DIR` 改成 `./log/occluded_duke/exp340_swin_lgpa_fixedbands`。
- **C2**：吊销/轮换 `.git_token` 里的 GitHub PAT；`git rm --cached .git_token` + 加 `.gitignore`；提醒历史已含该 token。

建议同时修 M1（yml 注释误导）、L1（`.DS_Store`），但这两项不阻断。

---

## Verdict

存在 **2 个 Critical**（OUTPUT_DIR 覆盖 exp336 + 提交真实 GitHub PAT），均须在启动训练前修复。算法/数据流/NaN/对称/单变量全部通过，无 High 级问题。

**需修复**

## 修复确认（第二轮）
- **C1 OUTPUT_DIR**：已改为 `./log/occluded_duke/exp340_swin_lgpa_fixedbands`，不再覆盖 exp336 ✓
- **C2 .git_token**：已 `git rm --cached` + 加入 `.gitignore`（`/.git_token`）；`.DS_Store` 同样 untrack ✓。**⚠️ token 仍在 git 历史中，需用户去 GitHub 吊销/轮换该 PAT。**
- **M1**：exp340 yml 标题/注释已改写为固定 band 实质 ✓
- 算法层（canonical 热图生成 + bias/assign/visibility 三路 + train/test 对称 + 单变量）首轮已逐行核查通过，本次仅改 config/安全，算法未动。

## Verdict：审查通过
两个 Critical 已修复，算法本身首轮即通过。可进入 Codex 审查 + 训练。
