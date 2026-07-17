# 实验 CBCL: 文本-图像 person ReID 的绑定歧义 (Binding Ambiguity, Counterfactual Binding Contrastive Learning)

> 5 codex vet 收敛(vet_0/vet_4 推 B 先冲)。AFD/cross-view 死后选定角度。★confound **文献已证实为真**(不是 plausible 猜测)——直击 4 死角教训。

## 动机
- T2I person ReID(文本描述→检索行人图): CLIP backbone(IRRA/CFine 等)继承 CLIP 的 **compositional binding 弱点**。
- ★confound(**文献已证实**): CLIP 是 **bag-of-words**, 编码"词出现"多于"绑定正确"。证据: ARO(Yuksekgonul ICLR23 Oral, 5万测试)、Winoground(CVPR22, 同词异序≈随机)、SugarCrepe(NeurIPS23, SWAP negatives 难)、"Does CLIP Bind Concepts"(EACL24)。
- person ReID caption 多是颜色-部位绑定(`red shirt + black pants`)。若模型当 bag(red/black/shirt/pants), 则把 `red shirt + black pants` 与 `black shirt + red pants` 混淆 → 检索错但标准 mAP 不暴露。
- ★这是前 4 死角(frequency/cross-view 等 plausible-but-handled)缺的: **confound 本身文献证实, 只需验"对 fine-tuned T2I-ReID 是否仍 matter"**。

## 核心假设
T2I-ReID 模型(含 fine-tuned IRRA)对同词异绑定 caption 不敏感(BSR 低)→ 用同词异绑定 hard negative 做 binding-aware 对比学习, 强制 attribute-to-bodypart 绑定 → 涨点 + 抗 binding-swap。

## ★ kill-switch(便宜, 今晚): BSR (Binding Swap Rate)
1. 从 ICFG-PEDES / RSTPReid caption 抽含 ≥2 个颜色-部位绑定的句子(top/bottom/shoes/bag/hat)。
2. 自动生成 **same-bag-of-words swapped** hard negative: `red shirt + black pants` → `black shirt + red pants`(只换绑定, 不换词)。
3. 算 `sim(image, original)` vs `sim(image, swapped)`; BSR = original 胜率; margin 均值。
4. **随机负 caption 对照**(排除普通 text noise): 随机负应易, binding-swap 应难。
- 测序: ① frozen CLIP ViT-B/16(证通用 confound) → ② **IRRA official/复现 checkpoint(关键: fine-tune 是否已修)**。
- **PASS(GO)**: IRRA BSR 仍明显低(<0.75)/swapped 仍排原图高, 且随机负胜率 ≥85% → confound 对 fine-tuned 仍 matter → 建方法。
- **FAIL(判死)**: IRRA BSR 已 0.85-0.90 → 模型已 handle → 转 A(accessory)或改 benchmark 小稿。

## 技术方案(kill-switch 过后)
- **CBCL**: 每 caption 自动生成同词异绑定 hard negative + binding margin/ranking loss(image 必须更近 original 而非 swapped)。
- 可加 body-part phrase parser(color/attr → upper/lower/head/shoes/bag slot)作实现手段, **不作主创新**(避撞 CFine/CADA/DiCo 细粒度对齐)。
- scaffold: IRRA(CLIP ViT-B/16 + SDM + MLM + ID, 单 3090 可训)。

## 对照 / 消融
IRRA baseline → +CBCL。消融: random negative / attribute-drop / color-only swap / part-only swap / 无 CBCL。主表 CUHK/ICFG/RSTP + **Binding-Swap stress test**(新 benchmark)。

## novelty 切口(避撞车)
1. **问题定义**: Binding Ambiguity in T2I-ReID(同属性词都在, 绑定错, 标准指标不暴露)。
2. **诊断指标**: Binding Swap Rate/Accuracy(新)。
3. **机制**: Counterfactual Binding Contrastive Learning(同词异绑定 hard negative)。
- 近邻切开: RaSa(IJCAI23, 改**词集合**=replaced words; 我们保词集合换绑定)、DualFocus/DAPL(属性**有无**)、CFine/CADA(细粒度对齐, 无 swapped-caption 训练)、DiCo(2026, concept disentangle)、⚠️**InterPartAbility(2026, phrase-region binding + counterfactual region masking, 最近, 必读切开**)。

## 数据 / venue
- ICFG-PEDES(图=**MSMT17, lab-3090 已有**; 标注 captions 需取)、RSTPReid(Drive/Baidu, 20505 img/4101 ID)。CUHK-PEDES(需学校邮箱申请, 今晚跳过)。
- venue: T2I-ReID 常投 ACM MM/AAAI/ICME/TMM/Neurocomputing。

## 备选(B 死则转)
- A: accessory 因果去混淆(detachable belongings; 先 Market-UPAR 无检测 probe, 强则 GroundingDINO/SAM remove/swap)。vet_1/vet_2 说 novelty 不死但 confound 不如 B 硬(ISP 把 belongings 当正向线索)。

## ★ kill-switch 实跑结果 — frozen CLIP ViT-B/16 (2026-06-22, lab-3090)

脚本 `bsr_killswitch.py`，RSTPReid，同词异绑定(只对调两个 garment 的颜色 token，词集合不变)。

| 设置 | 可swap样本 | BSR(原文胜 swap) | margin(orig−swap) | 随机负胜率 | gap(rand−BSR) |
|------|-----------|------------------|-------------------|-----------|---------------|
| test split | 1145 | **0.5729** | +0.00114 | 0.7983 | **+0.225** |
| 全量(cap4000, seed7) | 4000 | **0.5840** | +0.00133 | 0.7785 | +0.195 |

**判读: confound 在 frozen CLIP 上成立(✅)。**
- BSR≈0.57-0.58 ≈ 接近随机(0.5)，原文 vs 同词异绑定几乎不可分(margin~0.001，比随机负的 ~0.028 小一个量级)。
- 随机负胜率 0.78-0.80（普通 text noise 容易区分），BSR 明显 << 随机负胜率 → 难点确实来自**绑定**而非普通噪声，不是检索信号整体崩。
- frozen CLIP 确实把 person caption 当 bag-of-words，颜色-部位绑定基本不编码。

**下一步(决定 GO/FAIL 的关键)**: 同脚本 `--encoder irra` 跑 fine-tuned IRRA checkpoint(正 scp 到 `irra_rstp/`)。
- GO: IRRA BSR 仍 <0.75 且随机负胜率≥85% → fine-tune 没修好绑定 → 建 CBCL。
- FAIL: IRRA BSR 已 0.85-0.90 → 已 handle → 转 A 或改 benchmark 小稿。

**注意点(已知，不影响结论)**:
- 随机负胜率仅 ~0.78（非 ≥0.85），说明 frozen CLIP 在 RSTPReid 上整体检索信号本就偏弱(zero-shot)，BSR 绝对值的解释要谨慎；真正判据是 **BSR vs 随机负的 gap**（稳定 +0.19~0.23），而非 BSR 绝对值。fine-tuned IRRA 上随机负胜率应显著更高，对照更干净。
- 多词颜色边角：'navy blue' 这类会被拆成内层 'blue'(navy 留原位、blue↔другой 对调)，仍是合法同词 swap、绑定仍翻转、词集合仍守恒，不破坏指标，仅个别例子读起来略怪。

## ★★ kill-switch 实跑结果 — fine-tuned IRRA (2026-06-22, lab-3090) — 决定性

同脚本 `--encoder irra`，IRRA(CVPR23, anosorae/IRRA) 在 RSTPReid fine-tune 的 checkpoint(`irra_rstp/best.pth`)。
实现: `irra_encoder.py` 复现 IRRA 检索特征——图像 ViT CLS token + 文本 EOT token，
用 IRRA 自己的 SimpleTokenizer，base_model.* 权重(301 keys)**完全覆盖** openai seed
(loaded=301 / skipped=0 / not-overridden=0，Claude 子代理在机上验证 301==301 对称差 0/0，
无 openai 残留)。CLS/EOT/cos 排序与 IRRA 官方 evaluator 一致。**与 frozen CLIP 同一批 1145 swappable 样本。**

| 设置 | 可swap样本 | BSR(原文胜 swap) | margin(orig−swap) | 随机负胜率 | gap(rand−BSR) |
|------|-----------|------------------|-------------------|-----------|---------------|
| **frozen CLIP** test | 1145 | 0.5729 | +0.00114 | 0.7983 | +0.2253 |
| **IRRA** test | 1145 | **0.9459** | **+0.12495** | **0.9817** | **+0.0358** |
| **IRRA** train(cap4000) | 4000 | 0.9515 | +0.12726 | 0.9927 | +0.0412 |

**判定: FAIL（confound 不再 matter，转备选 A）。**
- IRRA BSR **0.946** ≥ 0.90，落在 design.md 预设的 FAIL 区间(0.85–0.90 及以上)。fine-tune 把绑定基本修好了。
- swapped margin 从 frozen 的 **+0.001（≈随机不可分）跳到 +0.125（强偏好原文）**：fine-tuned IRRA 在 1145 例里 94.6% 把正确绑定的 caption 排在同词异绑定 swap 之上。
- gap(rand−BSR) 从 frozen **+0.225 坍到 +0.036**：swap 负样本对 IRRA 几乎和随机负一样易区分了——绑定不再是检索难点。
- 随机负胜率 **0.982**（远 ≥0.85，对照干净），印证了 frozen 段(line 63)的预判：fine-tuned 检索信号强，对照可信。design.md 主判据(BSR vs 随机负 gap)在 IRRA 上一致指向 FAIL。
- 同例对照(idx 373 / 402)：frozen CLIP margin ±0.008（无所谓 swap），IRRA 同例 +0.243 / +0.195（强烈偏好正确绑定）。
- 两 split 一致(test BSR 0.946 / train 0.952)，非小样本噪声。

**结论**: Binding Ambiguity 对 fine-tuned T2I-ReID 已被 in-domain 对比训练(SDM+ID+MLM)隐式 handle，
CBCL(同词异绑定 hard negative)在已修好的 baseline 上无可改空间——撞上"frozen kill-switch 误导，
trained baseline 已 handle"这一已知反复出现的陷阱(同 burstiness/compositional-occluder 教训)。
**不建 CBCL 方法。转备选 A(accessory 因果去混淆)或换其它角度。**
