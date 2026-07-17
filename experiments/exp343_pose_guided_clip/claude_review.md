# Claude Broad Review — exp343 (Option A: 姿态引导 CLIP 机制对齐的图像特征)

**审查范围**: `git diff HEAD`（config/defaults.py + model/modules/clip_id_prompt.py 的 `PoseGuidedPool` + model/pose_backbone_model.py forward/init）+ configs/occluded_duke/exp343_pose_guided_clip.yml；并完整核对对照实验 exp341、processor 消费路径、make_optimizer、_prepare_pose、测试端 forward。
**审查轮次**: 第 1 轮（全范围）
**结论**: **审查通过**（无 Critical / 无 High；2 个 Low + 1 个文档级提醒，均不阻断训练）

---

## 1. 单变量隔离性（对照 exp341）—— 通过

- config diff 实测：exp343 = exp341 **逐字节相同**，仅多 `POSE_CLIP_ID_POSE_GUIDED: True` + OUTPUT_DIR + 标题注释。`GLOBAL_LOSS_SCALE` 两边都 1.0、`POSE_CLIP_ID_WEIGHT` 都 1.0、`POSE_CLIP_ID_ARCH/PRETRAINED` 一致、backbone（swin_tiny）/输入（384×128）/SOLVER 全一致。**真单变量**。
- 代码路径：exp341（`use_clip_id_pose_guided=False`）走 `feat_for_clip = global_feat`（line 587），exp343 走 `feat_for_clip = pose_guided_pool(featmaps[-1], scene_heatmaps)`（line 585）。两者随后**共用同一行** `img_proj = self.clip_id_proj(feat_for_clip)`（line 588）和同一 supcon i2t/t2i（591-592）。差异仅 `feat_for_clip` 一处。
- 两实验都从 forward 同一出口返回（line 861-862，`use_*` 部位分支全 False → 走 no-part 路径）。返回 5-tuple `cls_score, global_feat, featmaps, None, {'clip_id_loss': ...}`，processor line 598-599 正确解包。

## 2. PoseGuidedPool 正确性 —— 通过

逐项核对（clip_id_prompt.py 96-119）：
- (a) **shape**：`featmap (B,C,H,W)` → `flatten(2).transpose(1,2)` = `tokens (B,N,C)`，N=H·W；`pooled = einsum('bn,bnc->bc')` = `(B,C)`。维度链正确。
- (b) **pose 对齐 featmap 网格**：`F.interpolate(pose_heatmap, size=(H,W))`，H/W 取自 featmap.shape，对齐到 token 网格。正确。
- (c) **person visibility**：`pose.amax(dim=1)` 在 K=17 keypoint 维取 max → 每空间位置「是否落在任一关键点上」= 人体可见区域。`_prepare_pose` 返回 `scene_heatmaps (B,17,H,W)`（merge_person_heatmaps 后），与 `(B,K,Hh,Ww)` 形参吻合，amax over K 语义正确。
- (d) **softmax 维度**：`F.softmax(attn, dim=1)`，attn 形状 `(B,N)`，dim=1 = 空间 N 维。正确（不是 batch 维）。
- (e) **einsum 池化**：`'bn,bnc->bc'` 用注意力权重对 token 加权求和，标准 attention pooling。正确。
- (f) **可训练**：`self.query = nn.Parameter(...)`（requires_grad 默认 True）、`self.k_proj = nn.Linear`（默认 True）。梯度通路：`clip_id_loss → img_proj → clip_id_proj → feat_for_clip=pooled → einsum(attn,tokens) → attn=k@query → k_proj/query`。两参数都拿得到 i2t/t2i 梯度。
- **额外正确点**：pose 作为 **additive bias**（加在 logit 上、softmax 前），是 LGPA-D 同款 pose-bias 注意力范式（与 memory 中 LGPA-D 先例一致），不是硬 mask，可学 query 仍能在 pose 高响应区内挑判别 token。`/(C**0.5)` 缩放合理，`query*0.02` 小初始化使早期 attn 近均匀、pose_temp=1.0 让 pose 主导早期定位，收敛行为健康。

## 3. 优化器接入 —— 通过

make_optimizer.py line 7-9：遍历 `model.named_parameters()`，凡 `requires_grad` 即加入 param group。`pose_guided_pool` 是 model 子模块，其 `query`/`k_proj.weight`/`k_proj.bias` 均 requires_grad → **进优化器**。`k_proj.bias` 命中 line 12 "bias" 分支拿 `BIAS_LR_FACTOR=2` 的 lr（与全网 bias 同规则，无副作用）。无需额外 param_group。

## 4. None 回退 / 崩溃面 —— 通过

- forward line 584：`use_clip_id_pose_guided AND scene_heatmaps is not None` 才走 pose-guided，否则 `feat_for_clip = global_feat`。`scene_heatmaps` 为 None（无 pose_dict）时安全回退，**不崩**。
- 训练实际：processor line 540-542，`use_pose=True` 时 pose_dict 必传、_prepare_pose 产出非 None scene_heatmaps，且 exp343 `POSE_PARALLEL_AUG/OA_SD/USE_TARGET_HEATMAP` 全关、`pose_dropout_p` 未配（默认 0，不会把整张图清零）→ 训练期 pose-guided 路径**稳定激活**。

## 5. 维度自洽（关键，已排雷）—— 通过

潜在风险点：`featmaps[-1]` 是**原始 backbone 通道**（num_features[-1]），若 `REDUCE_FEAT_DIM` 开则 `global_feat` 被 fcneck 降维、`in_planes` 被改写，会与 `pose_guided_pool` 取的原始通道冲突。
- 实测：`REDUCE_FEAT_DIM` 默认 False（defaults.py:55），exp343/exp341 **均未设** → `reduce_feat_dim=False`，make_model.py line 230-233 fcneck 块跳过，`self.in_planes` 保持 = `base.num_features[-1]`。
- 故 `featmaps[-1]` 通道 == `in_planes` == `PoseGuidedPool(in_planes)` 期望 == `clip_id_proj` 输入维。**全链一致，无 shape mismatch**。`pooled (B,in_planes)` 与 `global_feat (B,in_planes)` 同维，喂同一 `clip_id_proj`，对照完全对称。

## 6. 测试端 train/test 对称（关键，已确认无泄漏）—— 通过

- eval 分支（forward line 864+）：exp343 的 `use_vcsr/use_lgpa/use_ppa/use_structural_routing/use_skeleton_gcn` **全 False**，line 875-1000 各 `elif` 全部跳过 → `gcn_feats=None`，test 描述子 = `test_feat = global_feat`（line 868，NECK_FEAT='before'）= **GAP 全局**。
- `pose_guided_pool` 只在 `self.training` 块内（line 571-862）被调用，**eval 永不触发**。无 train/test 泄漏，pose-guided 与 prompt 均训练专用。
- **对照公平性**：exp341 与 exp343 测试端都 eval 同一个 GAP global。exp343 的 pose-guided 仅通过 i2t/t2i 对齐梯度**塑造 backbone**，最终落到 global 上比较。`exp343 global vs exp341 global` 是同口径苹果对苹果。判据成立。

## 7. AMP / dtype 安全 —— 通过（含 1 Low）

- autocast 下 `featmap/tokens` 可能 float16，`k_proj` 在 autocast 内跑（float16），`query` 是 float32 Parameter（matmul 内部按 autocast 规则处理）；`pose.float()` 强制 float32 → `attn + pose_temp*pose_region` 触发 float16+float32 **自动提升为 float32**（PyTorch type promotion），不报错；`einsum(attn_f32, tokens_f16)` 同样提升 float32；`pooled(f32)` 喂 `clip_id_proj`（autocast 内）正常。**无 dtype 崩溃**。
- 见下 Low-1：与既有 PAPE/pose_prompt 的 `.to(x.dtype)` 显式约定略不一致，但此处 pooled 是终端输出（非加回 token 流），提升为 f32 无害。

---

## Findings

### Critical
（无）

### High
（无）

### Medium
（无）

### Low
- **Low-1 (AMP 风格一致性，非 bug)**：`PoseGuidedPool.forward` 中 `pose.float()` 引入 float32，与 float16 token 流混算，依赖 PyTorch 隐式 type promotion。功能正确、不崩，但与 backbone 既有约定（PAPE line 414 / pose_prompt line 438 显式 `.to(x.dtype)`）不一致。**不阻断**；若想风格统一，可在 einsum 前把 `attn` cast 回 `tokens.dtype`，但当前实现更安全（全程 f32 池化精度更高）。建议保留现状。
- **Low-2 (config 死设置，无害但误导)**：exp343/exp341 都设了 `POSE_TEST_FEAT: 'equal_concat'`，但 clip-only 路径**没有任何部位分支**会去读它——`self.pose_test_feat` 属性在 `__init__` 仅于部位分支 if 块内赋值（line 168/187/235/279/333），clip-only 下从不赋值；eval 端用 `getattr(self,'pose_test_feat','global')` 回退到 `'global'`，且所有 `use_*` 为 False 使每个 elif 跳过。故该设置**完全不被读取**，最终 eval 必为 GAP global。无功能影响，但字面误导（像在用 LGPA 的 equal_concat）。建议把它改成 `'global'` 或删除以免日后误读。

### 文档级提醒（不阻断、不计 severity）
- exp343 config 顶部注释（line 2-4）是从 exp244/LGPA-D-standalone 模板**复制残留**：「CLIP 模块(LGPA-D)本身能否 standalone 涨点…基于 pose_psg_lgpa_detach.yml…test.py equal_concat(LGPA) vs global」与本实验（pose-guided CLIP prompt 对齐、无 LGPA 分支、eval 恒 global）**完全不符**。design.md 写得正确，仅 config 注释陈旧。建议更新注释，避免与 design 冲突误导后续接手。

---

## 创新性质疑（审查制度要求）

本实验**不是**小调参 / 逃避创新：它直接回应用户澄清的核心诉求——**把姿态融进 CLIP 对齐机制本身**（pose 引导「CLIP 对齐什么图像特征」），而非 exp342 被否的「CLIP + 旁挂独立 LGPA 分支」。机制上有真实改动（对齐目标从 raw GAP → pose-bias attention-pooled feature），且建立在 exp341 已验证的 +2.2 真涨 CLIP 机制之上，判据清晰可消融（单变量 on/off，同口径 global）。属机制层面新意，符合创新门槛第 2 条。即便结果平/负，也能干净地回答「pose-guided 对齐 vs raw-global 对齐」这一问题，有论文/诊断价值。**不质疑为调参**。

---

## 结论

代码逐行已读：PoseGuidedPool 的 shape/pose-bias/softmax/可训四性正确；优化器正确接入新参数；None 回退安全；维度全链自洽（REDUCE_FEAT_DIM=False 排除降维冲突）；train/test 严格对称、eval 恒 GAP global、与 exp341 同口径公平对照；AMP 无崩溃；单变量隔离干净。仅 2 个 Low（AMP 风格一致性、config 死设置 POSE_TEST_FEAT）+ 1 个 config 注释陈旧，**均不阻断训练**。

**审查通过**。建议训练前顺手把 Low-2 的 `POSE_TEST_FEAT` 改 `'global'`、更新陈旧 config 注释（可选，非阻断）。可进入 Codex 第二轮审查。
