# 实验 CVCL: CARGO 跨视角正样本稀缺 confound (cross-view positive scarcity)

> 6 codex 重评估收敛点(re2_0/2/5 列 #1)。AFD 频率角度死后的下一角度。confound **今晚就能验证为真/假**(无训练), 数据现成(lab-3090 CARGO), 复用 baseline。

## 动机
- CARGO 航拍-地面 baseline(resnet50 BoT)mean A↔G mAP **32.48%**, 但 **train Acc 0.999 vs cross-view eval 32% = 严重视角过拟合**。
- ★confound: **cross-view positive scarcity** —— 标准 RandomIdentitySampler(P×K)随机采 pid 的图, 不保证一个 batch 里同 ID 同时有 aerial+ground。batch-hard triplet 的 positive 多是 same-view(easy), 跨视角正对(hard 但关键)很少进 batch → 模型学不到跨视角不变性, 靠 view-specific 捷径拟合 train。
- novelty: aerial-ground ReID 里(VDT/SeCap/GSAlign 都做 view module / semantic alignment)**没人把"训练 batch 缺 view-complete positives"当主问题**(re2_2 查重)。⚠️风险: 薄(sampler+triplet)——confound 验真 + 增益大小决定能否撑方法稿。

## 核心假设
跨视角正对在标准 batch 里稀缺 + baseline 同 ID 跨视角特征距离接近 hard-neg → 显式保证跨视角正对(view-balanced sampler)+ 跨视角 triplet 能缩 view gap 涨点。

## ★ kill-switch(无训练, 今晚, 复用 baseline)
用 `model_best.pth`:
1. **batch 采样统计**: CARGO train 每 pid 的 aerial/ground 图数; 双视角 pid 占比; 模拟 RandomIdentitySampler(P=16×K=4)1000 batch, anchor 有 opposite-view positive 的比例。
2. **baseline 特征距离**: same-id same-view / same-id cross-view / diff-id cross-view(hard neg)。
- **PASS**: 跨视角正对在 batch 稀缺(<70%)AND same-id cross-view 距离明显 > same-id same-view 且接近 hard-neg → confound 成立 → 建方法。
- **FAIL**: 跨视角正对充足 OR cross-view 距离≈same-view → 模型已学跨视角不变 → 判死, 转 accessory(re2_1)/T2I-binding(re2_4)。

## 技术方案(kill-switch 过后)
- **VC-PK sampler**: 每 pid K=4 优先 2 aerial + 2 ground(不改 batch size 64)。
- **CV-triplet**: positive 只取 opposite-view same-id, negative opposite-view diff-id; loss = CE + 标准 triplet + λ·CV-triplet(λ=0.5)。
- 消融: baseline → +VC-sampler → +CV-triplet → +both; λ 敏感性。

## 对照组
baseline(afd_train.py use_afd=False, 32.48%)。单变量: 只加 sampler / 只加 CV-triplet / both。ep20 mean mAP ≥ baseline ep20(11.01)+1.0 才继续; final ≥ 34.5 进方法线。

## 数据 / venue
- CARGO(lab-3090 现成)。补 AG-ReID.v2 做多数据集表。
- venue: ICME/ICPR/Neurocomputing → 强则冲 ACCV/BMVC。

## 备选(此角度薄/死则转)
- re2_1 accessory 因果去混淆(counterfactual accessory, 需 GroundingDINO/SAM)——更 novel。
- re2_4 T2I binding CBCL(同词异绑定 caption, 需 ICFG/RSTP)——不挤子领域。
- re2_3 camera-homophilous(Market/MSMT 现成, 但 camera bias 拥挤)。

---

## ★ kill-switch 结果(2026-06-22, lab-3090, cv_diagnostic.py)→ **FAIL,confound 证伪**

`/tmp/cv_diagnostic.log`。CARGO train: 51451 img / 2500 pid / aerial 22338 + ground 29113。

### (a) batch 采样统计
- aerial imgs/pid mean=8.94(min2 max31); ground imgs/pid mean=11.65(min4 max35)。
- **DUAL-view pid = 2500/2500 = 100.0%**(aerial-only 0, ground-only 0)。每个 ID 都同时有航拍+地面 → "稀缺"前提在数据集层面就不成立。
- 模拟**真实 RandomIdentitySampler**(P16×K4)1000 batch: **anchor 有 opposite-view 正样本占 88.0%**(≥1 同 ID 正样本 100%); 每 anchor opp-view 正样本 mean=1.51, 仅 12% anchor 为 0。→ 远高于 <70% 稀缺阈值, **跨视角正对在标准 batch 里并不稀缺**。

### (b) baseline 特征距离(BN feat, cosine; eucl 括号内)
| 类别 | cos 距离 |
|------|---------|
| same-id same-view (easy pos) | 0.5803 |
| same-id cross-view (hard pos) | 0.6647 |
| diff-id cross-view 最近 (hard neg) | 0.7418 |
| diff-id cross-view 均值 | 0.9279 |

- 跨视角正样本惩罚 scv−ssv = **+0.0844**(很小, 仅高 14.5%, 未达 1.15× 阈值)。
- 正-硬负 margin dcv_hard−scv = **+0.0771 > 0**(正样本**没有**淹没在负样本里, 还留着健康间隔)。
- hardest cross-view 正样本比最近 cross-view 负样本远的 anchor 仅 42%(<50%, 多数正样本仍排在负样本前)。

### (c) 判定 FAIL(C1/C2/C3 全 no)
- C1 batch 内 opp-view 正样本稀缺(<70%)→ no(88%)。
- C2 same-id cross-view 明显 > same-view → no(0.6647 vs 1.15×ssv=0.6674, 差一点点)。
- C3 cross-view 正样本贴近硬负 → no(margin 还有 +0.077)。
- **结论**: CARGO 100% pid 双视角 + 标准 sampler 已让 88% anchor 拿到跨视角正对; baseline 特征上同 ID 跨视角距离只比同视角略大、且明显小于硬负。模型**已经学到足够的跨视角不变性**, "cross-view positive scarcity" 不是 32% mAP 低的主因。VC-PK sampler / CV-triplet **预期无增益, 不开训**。
- 32% cross-view mAP 低的真正瓶颈在别处(视角 gap 在 hard tail / 难样本, 非 batch 缺正对)。→ 此角度判死, 转 re2_1 accessory 因果去混淆 / re2_4 T2I-binding。
