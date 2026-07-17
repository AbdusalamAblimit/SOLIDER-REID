# 实验 AFD-ReID: 航拍-地面 频率解耦 person ReID (v0)

> 全自动重评估选出的 #1 角度(8 codex + 合成裁决)。死亡清单 0 风险、novelty 空白、kill-switch 最廉价。

## 动机
- 航拍-地面(UAV-ground / aerial-ground)行人 ReID: query/gallery 跨极端视角(俯视 vs 平视)+ 尺度差(UAV 把人压成低分辨率小目标)。子领域不挤(VDT/CARGO CVPR24、AG-ReID 系列, 篇数少)。
- ★confound: **视角-频率纠缠**——UAV 俯视下高频纹理(logo/鞋包/细节)被压糊、不稳; 地面视角高频强。模型错把"由高度决定的不稳高频"当身份线索 → 跨视角检索掉点。
- 空白: 没人把 frequency × aerial-ground person ReID 当 confound(VI-ReID 频域挤死、event 有 SFE-Net, aerial-ground 频域空白)。不沾遮挡死亡清单、不 VI 硬磕。

## 核心假设
航拍-地面跨视角检索中高频带显著比低/中频更不可靠; 按视角/高度自适应路由频带可靠性 + 跨视角频率反事实约束, 缩小视角 gap → 涨点; 增益来自频率可靠性建模(非通用正则, 由频带消融证)。

## ★ kill-switch(无训练, 先跑, 几小时)
CARGO(+AG-ReID.v2)上, pretrained Swin/CLIP 抽特征, 对 原图 / low-pass(FFT保低频) / high-pass(FFT保高频)三套各算 Aerial↔Ground 的 mAP/R1。
- **PASS**: low/mid band 明显比 high band 稳, 或不同 altitude best-band 可分; 最好 full < gated-oracle(每query选最优频带) +2 mAP。
- **FAIL(判死)**: 各频带同涨同跌、altitude 对频带可靠性无差异 → confound 不成立 → 停, 不补频带变体。
- 数据没到前: 先用本地 market1501 把 FFT 分频 + 跨组检索脚本 debug 跑通。

## ★★ kill-switch 结果 (2026-06-22): INCONCLUSIVE
CARGO 29M 子集(Mac 抽 3000 aerial+3000 ground, A∩G=1045 pid)。pretrained resnet50 V1(ImageNet)抽特征, A↔G mAP%:
| band | A→G | G→A |
|---|---|---|
| orig | 0.36 | 0.58 |
| low | 0.44 | 0.46 |
| mid | 0.27 | 0.28 |
| high | 0.36 | 0.28 |
- **全部近随机**(orig 都 ~0.4% = ImageNet 特征压根做不了 CARGO 航拍↔地面检索, 本就难, VDT 要训练才 ~50%)。各带接近, 只 G→A 弱 hint(high 0.28<low 0.46)但在噪声内。
- **判定: 无训练 kill-switch 是错工具**(off-the-shelf 特征太弱→无信号比频带)。公平验 confound 需训练 baseline。
- ★★META 阻碍: **lab-4090 连接病态慢(~0.015MB/s, 29M 都 ~30min)**, 全 CARGO 590M 没法传→AFD 真训练被卡。子集策略已绕过大半但仍慢。
- → 决策: 测 lab-3090 速度(快→迁 AFD 训 baseline 验); 也慢→数据传输全局阻碍→转**数据现成角度**(lab-4090 已有 occluded_duke/occluded_reid/market/MSMT17/RegDB, 避开传输)。

## ★★★ confound 验证 (2026-06-22): 证伪, AFD 死
迁 lab-3090(快 30x)。训 baseline 32.48% mean A↔G mAP(健康, 8.5→11→14.8→26.8→32.5)。band_analysis(在训好 baseline 上 ablate 频带):
| band | A→G mAP | G→A mAP |
|---|---|---|
| orig | 32.90 | 32.05 |
| 去high(no_high) | 30.10(掉2.8) | 30.68(掉1.4) |
| 去low(no_low) | **0.04** | 0.06 |
| 只low | 11.54 | 10.38 |
| 只high | 0.03 | 0.15 |
- **模型几乎全靠 low+mid 频**(去 low 崩到 0.04, 只 high 也 ~0)。high 频次要且**对航拍 query 还略有用**(去 high 掉 2.8>地面 1.4)。
- **confound("高频不可靠、模型误用它致跨视角失败")正好被证伪**: 模型早避开高频靠低频, AFD(降权高频)前提错→不该帮。第 3 个 kill-switch 判死的角度。
- band_analysis kill-switch 兑现价值: 在 +AFD 大投入前证伪 confound。
- 跑 +AFD 最终经验判定: **epoch-10 mAP 8.91 ≈ baseline 8.51(无提升)→ kill, AFD 确认死**(符合 confound 证伪)。CVFC loss 仅 0.0003 近无效。
- ★教训(3死: 撞车/技术不可行/confound证伪): **confound 必须可验证为真, 不只物理 plausible**。

## 技术方案(kill-switch 过后)
- 模块1 **Frequency Reliability Router**: DWT/FFT 分 low/mid/high band, 按 camera/altitude/view 学轻量 gate → 频带可靠性权重。
- 模块2 **Cross-View Frequency Counterfactual**: 训练时 high-band dropout / amplitude swap / low-high consistency, 约束同 ID 在 A↔G 下身份不变 + view-adversarial/orthogonal 防视角泄漏。
- backbone: SOLIDER-Swin(团队资产)。

## 对照组 / 消融
baseline(裸 pretrained A↔G)→ +Router → +Counterfactual → +both; 频带切分消融; 有无 view-adversarial。kill-switch 的三路特征(原图/low/high)对比即第一张图。

## 数据 / venue
- CARGO(Drive 单文件 id 1yDjyH0VtW7efxP3vgQjIqTx2oafCB67t, Mac-scp; camID_time_personID_index.jpg 命名带视角/时间/ID)。AG-ReID.v2 补充。
- venue: ICME/ICPR/Neurocomputing → 四数据集稳正可冲 ACCV/BMVC。

## 备选并行(数据风险对冲)
Event-LUPI(RGB/CLIP 特权 teacher→event student, 测试单 event): EvReID(Baidu/Dropbox, 国内网络风险)。死亡清单 0、novelty 干净, 但数据通道独立、工程量更大。AFD 数据顺利就优先 AFD。
