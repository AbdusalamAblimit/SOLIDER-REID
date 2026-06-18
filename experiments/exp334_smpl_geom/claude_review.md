# Claude Broad Review — exp334 SMPL 几何空间先验

**审查范围**: design.md + scripts/exp334_train_smpl_geom.py(主) + scripts/smpl_cache_geom.py
+ 对照机制 vit_pytorch.forward_features / make_loss / make_optimizer / scheduler / metrics /
sampler / occluded_duke / triplet_loss / defaults.py。逐行阅读 + fp16/all-zero 数值模拟。

**结论**: NEEDS-FIX（1 个 High 文档/对照诚实性问题 + 数个 Medium/Low；代码运行无 Critical）。

---

## 1. CORRECTNESS / RUNTIME

### [PASS] `_tokens()` 正确复刻 forward_features 的 else 分支
- backbone `forward_features`（vit_pytorch.py:305-335）在 SIE off（camera=0,view=0，line 318-319）时是
  `x = patch_embed(x)` → `cat(cls, x)` → `x + pos_embed` → `pos_drop` → 全部 blocks → `norm`。
- `_tokens()`（exp334:162-171）完全一致：`patch_embed` → `cat([cls, x]) + pos_embed`（无 sie_embed，
  正确对应 else 分支）→ `pos_drop` → blocks → `norm`，返回 `x[:,0]`(cls) 与 `x[:,1:]`(patches)。
- 形状：256×128 / stride16 → num_x=(128-16)/16+1=8，num_y=(256-16)/16+1=16 → num_patches=128。
  patches = (B,128,768) ✓ 与 GRID_H*GRID_W=16*8=128 一致。
- baseline 路径的 cls 与 exp333 的 `self.base(img)` 在 SIE off 下等价（exp333 走 backbone 的
  `forward`→`forward_features` else 分支，同样 cls=x[:,0]）。**weak baseline 的 cls 表征与 exp333 一致** ✓。

### [PASS] body-weighted pool 数值安全（无 NaN）
已用 fp16 + all-zero 行实测（见审查记录）：
- `w = softmax(hm.masked_fill(hm==0,-1e4))`：valid=1 行得 body 加权分布；valid=0 行 hm 全 0 →
  全部 masked → softmax 退化为 uniform(1/128)，finite。`-1e4` 在 fp16 精确可表示（−10000.0）。
- `einsum('bp,bpd->bd', w, patches)`：(B,128)·(B,128,768)→(B,768) ✓。
- gate `valid*f + (1-valid)*missing`：valid=0 行的 uniform-pool f 被完全覆盖为 missing token，
  uniform 池化结果不进入图（被 0 乘）。dtype 混合（fp16 f × fp16 valid + fp16 × fp32 missing param）
  自动提升 fp32，finite ✓。**无 NaN/Inf。**

### [PASS] forward 返回元数 train=4 / eval=2，循环解包正确
- train（181-187）：use_geom 返回 `(score_a, cls, score_b, fb)` 4 值；训练循环
  `score_a, feat_a, score_b, feat_b = model(...)`（284）4 值解包 ✓。
- use_geom=off train 返回 `(score_a, cls, None, None)` 4 值，`if args.use_geom` 分支跳过 body loss ✓。
- eval（189-192）：返回 `(feat_a, fb_bn)` 或 `(feat_a, None)` 2 值；evaluate 循环 `a, b = model(...)`（201）
  2 值解包 ✓。

### [PASS] loss masking 与 exp333 一致
`_valid_balanced_mask`（68-74）= exp333 `_balanced_valid_mask`（77-90）逐行等价：只保留 batch 内
**全部实例 valid** 的 identity（满足 hard_example_mining 要求 P×K 均衡——triplet_loss.py:78-84
`view(N,-1)` 要求每 anchor 同样多正样本），<2 id 存活则返回 None 跳过 body loss。✓

### [PASS] AMP / scheduler / R1_mAP_eval / optimizer 覆盖新参数
- AMP：`amp.autocast(enabled=True)` 包 forward+loss，`scaler.scale(loss).backward/step/update`（283-293）
  = 仓库 processor.py:573 验证过的范式 ✓。
- scheduler：`create_scheduler(cfg, optimizer)` → `step(epoch)`（295），`_get_lr(epoch)`（297）均存在
  （cosine_lr.py:67,98）✓。
- R1_mAP_eval：`R1_mAP_eval(num_query, feat_norm=True)`（209,215），cfg=None → compute() 里
  `self.cfg` 全部有 None 守卫（metrics.py:192,259...），NFC/RR/power 默认关 ✓。
- optimizer：`make_optimizer` 遍历 `model.named_parameters()` 且 `requires_grad`（make_optimizer.py:7-9），
  新参数 `missing`(requires_grad=True)、`bn_body.weight`(True，bias 被 `requires_grad_(False)` 故跳过，
  与 bottleneck 对称)、`classifier_body.weight`(True) **全部被纳入**。注意 bn_body.bias /
  bottleneck.bias 因 requires_grad=False 被 `continue` 跳过，正确（标准 BNNeck，不训练 BN bias）✓。

### [Low] center_criterion 始终被构造但 IF_WITH_CENTER='no'
`make_loss` 无条件建 `CenterLoss(feat_dim=2048)`（make_loss.py:100），`make_optimizer` 无条件建
`optimizer_center`（make_optimizer.py:33，引用 cfg.SOLVER.CENTER_LR=0.5 默认存在）。exp334 丢弃
`optimizer_center`（`optimizer, _ = make_optimizer(...)`，272）且 loss 不含 center 项 → 无副作用，
仅多占一点显存。与 exp333 完全一致，**非回归**。可忽略。

---

## 2. HEATMAP CORRECTNESS

### [Medium→设计已知] crop 增强下热图错位（acceptable soft-prior noise，但应量化）
- `joints_to_heatmap`（87-97）用 **PIL 原图 (W,H)**（GeomDataset.__getitem__:109 在 transform 前取
  `pil.size`）归一化 pj2d，splat 到 16×8 网格。
- 训练 transform：Resize([256,128]) → Pad(10) → RandomCrop([256,128])（253-257）。Resize 把整图均匀
  映射到 256×128，**保持归一化坐标**（关节相对位置不变）→ 归一化正确 ✓（审查点 2 末问：是的，
  Resize 均匀映射，按原 W,H 归一无误）。
- 但 Pad(10)+RandomCrop 把图像内容平移 ±10px（patch 尺度 ≤0.6 patch），热图却用未裁剪坐标 →
  **训练期热图与实际 token 网格有 ≤~0.6 patch 的随机错位**。test 无 crop（val_tf 只 Resize，258-259）→
  完美对齐。
- **判定：非致命 bug，是软先验噪声**。理由：(a) 错位是零均值随机平移（RandomCrop 对称），不引入系统
  偏置；(b) 软门控（softmax 加权池化，非 hard mask）对 ≤0.6 patch 抖动鲁棒；(c) 实际等效于对身体
  热图做轻微 spatial jitter 正则。**不会系统性破坏训练**。
- **但**：train/test 存在分布差（train 错位 / test 对齐），且这正是 design 风险①（location≠visibility）
  之外的第二个 train/test 不对称源。建议：要么在 design.md 明示该错位幅度（≤0.6 patch），要么把
  heatmap 也走同一 RandomCrop 几何变换（成本高，第一版不必）。**当前可接受，但必须在 monitor.md/
  design 里记录，避免日后误判增益来源。**

### [Low] sigma=0.8 在 8 宽网格上偏窄
sigma=0.8 patch 的高斯，单关节有效覆盖 ~1.5 patch。71 个关节（ROMP 24 SMPL + 47 extra）大量重叠，
整体身体覆盖足够。但极端：若某图所有关节挤在 1-2 列（侧身/近景），softmax 会高度集中 → body-pool
≈ 单 patch，方差大。非 bug，注意日志里观察 body loss 是否异常。

### [PASS] 越界关节正确丢弃
`if 0 <= x < GRID_W and 0 <= y < GRID_H`（95）丢弃出界投影点。ROMP 对完整身体的幻觉关节可能落在
crop 框外（负坐标或 >W），此处正确忽略，不会把热图能量泄漏到边界 patch ✓。

---

## 3. SINGLE-VARIABLE / 对照诚实性

### [HIGH] use_geom=on 的 alpha=0 ≠ exp333_baseline 53.09 —— 报告口径必须改正（design 已部分意识，但不够显眼）
这是本审查最重要的一条。

- **use_geom=off**（control）：forward 走 `score_a=classifier(bottleneck(cls))`，**完全不构造 geom 分支**
  （183-184 提前 return），无 missing/bn_body/classifier_body 参数，`set_seed` 在 model init 后重置
  （270）→ 与 exp333 `--use_smpl off` **同 backbone 同 seed 同 config**，应复现 ~53.09。✓（作为
  纯外观 baseline 是忠实的——但注意它是 exp334 自己的 off 臂，理论上应重训一次确认 ≈53.09，
  不能盲信跨脚本数值。见下方建议。）

- **use_geom=on**（treatment）：`_body_feat` 消费 `patches`（backbone 输出），`score_b/fb` 经
  `classifier_body/bn_body` → body loss `w_body * l_b` 反传**进共享 backbone**（284-292）。
  **因此 on 臂在 alpha=0 时的 cls 特征 ≠ baseline 的 cls**——backbone 已被 body 监督联合改写。

- 对比 exp333：exp333 的 SMPL-MLP 只吃**外部缓存向量** `smpl[:,:-1]`（exp333:199），与 backbone
  计算图**完全解耦**，3D loss 只更新 MLP/classifier3d，**不碰 backbone**。所以 exp333 实测
  "alpha=0 逐位=baseline 53.09"（monitor.md:64,76）。**exp334 没有这个性质。**

- **后果 / 必改口径**：
  1. exp334 的 alpha=0 **不是** "纯外观 baseline"，而是 "被 body-loss 联合训练后的 backbone 的
     cls-only 读出"。它可能 >53.09（body 监督当辅助正则帮了 backbone）或 <53.09（干扰）。
  2. **诚实的 A/B = on-臂-best-alpha vs 53.09**（design.md:30 行"alpha=0 truly appearance-only"
     的潜台词若指"等于 baseline"是**错的**；用户在 review 点 3/4 已正确质疑这一点）。design.md 正文
     "valid=0 时模块自动退化→只在有 SMPL 的图起作用" 描述的是**单图 missing 回退**，不等于
     "alpha=0 整模型回退到 baseline"——两者不可混淆。
  3. **建议新增对照**：on 臂额外记录 "alpha=0 mAP"（已在 eval_alphas 默认含 0.0，✓ 自动会打），
     用于诊断 body-loss 对 backbone 的净效应（alpha=0 vs 53.09 = backbone 正则效应；
     best-alpha vs alpha=0 = body 特征融合增量）。**这两个量都要进 results.md，分开报，不能合并
     成"+X mAP"一个数。**

- **修复动作**（必做，文档层面）：
  - design.md「对照组」「预期/判据」改写：明确 baseline=53.09 是 off 臂；on 臂 alpha=0 因 body-loss
    入 backbone **不等于** baseline；headline = on-best-alpha vs 53.09；附诊断 alpha=0 vs 53.09。
  - 强烈建议**也重跑一次 exp334 自己的 off 臂**确认 ≈53.09（排除跨脚本环境/库版本漂移），再用作分母。
    若用户接受跨脚本复用 53.09，至少在 results.md 注明分母来自 exp333 同机同 seed。

### [PASS] off 臂训练 RNG 与 baseline 对齐
`set_seed(SEED)`（238）→ build model → `set_seed(SEED)`（270 二次重置）= exp333:276/333 同款，
消除 geom 分支额外 param init 对 RNG 流（data order / dropout / drop-path）的扰动，保证 on/off 两臂
数据流一致 → geom 分支是唯一变量 ✓（这是 on vs off 的单变量保证；与上面"on 的 alpha=0 不等于
baseline 数值"不矛盾——单变量保证的是两臂*除 geom 外*一切相同，不保证 alpha=0 数值回到 baseline，
因为 geom-loss 本身就是那个变量且它入 backbone）。

---

## 4. EVAL FUSION

### [PASS] 融合与归一化正确
- `model.eval()`（196）已设 → BN 用 running stats，drop_path/dropout 关 ✓。无 test 期 BN 泄漏。
- cls：neck_feat='before'（config:54）→ `feat_a = cls`（before-BN，189）；body：`bn_body(_body_feat)`
  （after-BN，192）。两者各自 `F.normalize`（212），`comb = cat([fan, al*fbn])`（214），R1_mAP_eval
  feat_norm=True 再整体 renorm ✓。与 exp333 eval（245-247）同构。
- alpha=0：`comb=cat([fan, 0])` → 距离只由 fan 决定（body 维全 0，对 L2 距离无贡献，renorm 后
  fan 维等比放大不改排序）→ **alpha=0 = cls-only 检索**。但如 §3 所述，这个 cls 来自被 body-loss
  联合训练的 backbone，**不是 baseline 模型**。"alpha=0 truly appearance-only from the jointly-trained
  model" —— 是的，是 jointly-trained 模型的 appearance-only，**不是 baseline**。口径见 §3。

### [Low] body 特征用 after-BN、cls 用 before-BN 的不对称
cls before-BN（neck_feat=before）但 body after-BN（bn_body）。两者都 L2-norm 后融合，尺度由 norm
吸收，不影响正确性。但语义上 cls/body 处于不同 BN 阶段，属设计选择（body 新头需 BN 稳定），非 bug。

---

## 5. 与 exp333 的差异 / 机制风险

### [PASS] 与 exp333 β-concat 本质不同，不是偷偷重复
- exp333：SMPL-**β 当全局身份特征**，外部向量 → MLP → 独立分类/triplet，**不碰 backbone 计算图**，
  与 cls concat。证伪（β=随机 → 0.18% mAP）。
- exp334：SMPL **2D 关节当空间先验**，pj2d → patch 网格热图 → **对 backbone patch tokens 软加权池化**
  → body 特征**回传 backbone**。输入信号（2D 关节位置 vs 10-d β）、用法（空间池化权重 vs MLP 特征）、
  梯度路径（入 backbone vs 不入）三者全不同。**是真正不同的实验，非换皮**。✓
- design.md:4 的洞察（"几何空间先验，不是身份描述子"）成立：热图权重不依赖 β 数值，只依赖关节
  *位置*，绕开了"β 是随机"的诅咒。

### [Medium→设计已知未缓解] location≠visibility：遮挡处 body-pool 池到遮挡物特征
design.md:13 自陈风险①。代码层面**未缓解、也未加重**，如实反映该风险：
- SMPL 幻觉出完整身体关节（含被遮挡部位）→ 热图在遮挡 patch 上有高权重 → softmax 池化把**遮挡物的
  token**也加权进 body 特征。这正是 design 担心的。
- "软门控缓解" 的说法**部分成立但有限**：softmax 加权（非 hard mask）确实让模型可学习压低某些 patch，
  但热图是**固定先验权重**（不可学、masked_fill 死死压住 hm==0 的 patch），模型只能在 hm>0 的
  patch 内部重分配，**无法把高权重从"遮挡但 SMPL 说有身体"的 patch 移开**。所以遮挡物确实会被池化。
- **代码既没加重也没真正缓解**——它忠实实现了 design 的"第一版软门控"。真正的解（design.md:15,27）
  是 exp334b 的 ViTPose-visibility 融合。**这是已知的、记录在案的方法风险，不是代码 bug**。
- 唯一的被动缓解：valid-balanced mask 让 body loss 只在 *全 valid* 的 id 上训，但这与"单图内遮挡
  patch 被池化"无关——后者在所有 valid 图上都发生。
- **判定**：可接受进入第一版实验（design 已诚实预判，且 exp334a 的目的正是*实证*该风险是否抵消增益）。
  但 monitor/results 必须把"重遮挡子集"单列，因为理论上该子集最可能因 location≠visibility 而**不涨甚至
  降**——这恰是判据。

### [Low] cache 目标人选择 = 最居中检测，缓解但不解决遮挡物误选
smpl_cache_geom.py:74-78 用"2D 关节质心最接近图像中心"选目标，比 argmax-confidence 更稳（遮挡物
检测通常偏离中心）。但 Occluded-Duke 的 bbox 已裁剪到目标人，遮挡物若也居中（前景行人骑车/立柱）
仍可能误选。非本实验可控，记录即可。

---

## 必修项（must-fix，全部为文档/口径，无代码 Critical）

1. **[HIGH] 改正对照口径（§3）**：design.md 明确 (a) baseline 53.09 是 off 臂；(b) on 臂 alpha=0
   因 body-loss 入 backbone **不等于** baseline 53.09；(c) headline A/B = on-best-alpha vs 53.09；
   (d) 增列诊断 alpha=0 vs 53.09（已自动产出，需在 results.md 单独报，**禁止把 alpha=0≈baseline 当
   前提**）。强烈建议重跑 exp334 自身 off 臂确认 ≈53.09 再用作分母。
2. **[Medium] 记录 crop 错位（§2）**：design.md/monitor.md 注明训练期热图用未裁剪坐标 → 与 token 网格
   有 ≤~0.6 patch 随机错位（零均值软噪声，test 对齐），声明这是 acceptable soft-prior noise，避免日后
   误判增益来源。
3. **[Medium] 重遮挡子集单列（§5）**：results 必须把重遮挡子集 mAP 单独报，因为 location≠visibility
   风险在该子集最可能显形（判据）。

## 建议项（非阻断）
- on 臂日志加打 body loss 分量趋势（已有 `body={rb/nb}`，✓）与 alpha=0 vs best-alpha 差值。
- 若日后想消除 train/test 热图不对称，把 heatmap 随 RandomCrop 同步几何变换（第一版不必）。

## 代码层面结论
**运行正确性 PASS**：无 Critical/无 runtime error/无 NaN/单变量隔离（on vs off）成立/AMP·sched·eval·
optimizer 全部正确。**唯一硬伤是报告口径（alpha=0 不等于 baseline），属文档必修，不阻塞训练本身。**
建议先修文档口径（含可选的 off 臂复跑）再启动，确保产出能被诚实解读。

审查通过（代码可训练）；条件：上述 3 个文档必修项需在启动前补入 design.md，结果解读按修正口径。
