# ReID 方法稿"真实可发"角度综合 (2026-06-22) — 8 路 codex + 79 引用

## 1. 真实过审配方(不是理想化的)
B 类 ReID 方法稿真实门槛 = **"一个清楚命名的 confound + 2 个能画图的模块 + 3-5 个标准数据集 + 完整消融/SOTA 表"**。大多数近期论文本质=旧机制换新子题 + task/framing 包装, **扛不住打乱语义对照, 照样发**。

**可复用 checklist**: ①先命名一个 confound(比模块新更重要); ②只需 2 模块(结构模块 + 训练正则/增强); ③实验像"系统验证"(SOTA+消融+敏感性+可视化+复杂度+跨数据集), 单卡抓手=多数据集非大模型; ④小正则写成"旧目标忽略 confound X→引入 X-aware regularization"; ⑤生成/增强很好发(离线缓存); ⑥特权信息(LUPI: 训练用pose/SMPL/CLIP/RGB老师, 推理丢掉)很吃香。
8 套过审套路: 命名confound、任务重定义、新benchmark、堆2-3模块刷SOTA、生成训练分布、CLIP文本锚、频率域、特权蒸馏。
配方样本: KPR(ECCV24)、ProFD(MM24)、PGDS、IDKL(CVPR24)、AG-ReID.v2(TIFS24)、DPL-ReID(2605.19527)、MTRL(2511.02685)...

## 2. ★ Top-6 可发角度(按真实可发性排, 不上严苛红队)

### #1 ScribbleBridge(多模态 VI-ReID, sketch 桥)— 首选, 风险最低
- 卖点: 用 sketch/contour 当"光谱无关中间模态", 把 RGB/NIR/TIR 投到保留人体轮廓、去颜色/热强度的视图对齐。
- 组合: 共享Swin/ViT+modality token / 离线伪sketch(Canny/HED/PiDiNet) / sketch-centered transition contrastive(+可选CLIP部件锚)。
- 为何不挤: VI主流做shared-specific/frequency/middle-image(MTRL灰图)/CLIP prompt; **没人把sketch系统包装成跨光谱桥**。
- 最小验证: SYSU-MM01/RegDB/LLCM三表; 消融 baseline→+gray→+edge→+human-masked sketch→+transition contrastive。成功线: 两集稳+1.5, 三集平均正。
- Venue: ICME/ICPR/Neurocomputing/PR Letters → 补齐冲ACCV/BMVC。
- **可发性最高**: 数据今天就能下、贴Swin/ViT/CLIP资产、不训diffusion、单卡两周出第一版。

### #2 Event-only 特权蒸馏(事件相机)— 子领域最不挤, 次发抢位
- 卖点: 现有RGB-event推理依赖RGB破坏隐私优势; 训练用RGB/CLIP/attribute特权老师, **测试只用event**。
- 组合: event-only student(ViT/Swin-T) / frozen RGB teacher(CLIP-ReID)蒸global+logits+关系矩阵 / attribute→CLIP文本锚。
- 验证: 复现EvReID/TriPro(2507.13659,AAAI26,代码开源单3090可训)→event-only baseline→加蒸馏。成功线: +2~4 mAP逼近融合但推理只用event。
- **风险: EvReID数据/代码release先确认能拿**; 拿不到退ICCV23 Event-ReID(2308.04402)。

### #3 Time-Conditioned 跨年ReID(anytime)— 任务重定义故事最完整
- 卖点: long-term当一个桶错了, **不同time-gap下cue可靠性不同**(短期衣服有用, 跨年衣服发型都变→转身体结构)。
- 组合: DeepChange timestamp切5 split(新protocol=资产) / stable-volatile parser(SCHP) / time-conditioned cue router + CLIP stable anchors。
- 风险: DeepChange需签学术协议; AT-USTC可得性不确定。anytime刚被AT-ReID(2509.16635,IJCAI25)命名窗口刚开。

### #4 3D Canonical Surface(空地)— 贴SMPL资产
- SMPL不当身份特征、当跨视角"规范身体表面坐标系"。风险: SMPL俯视图valid rate低(团队exp333/334证β当身份不行→写成"几何坐标系非身份")。视频AG-VPReID别碰。

### #5 Frequency-View Decoupled(空地VIR视频)— 机制嫁接示范
- VI频率解耦搬进空地可见-红外视频, confound="modality-view-frequency"。风险: 视频数据重+AG-VPReID.VIR可得性不确定。

### #6 打包现有遮挡资产→PR/Neurocomputing — 最快保底(实验已做完)
- 团队已有75.2/85.6遮挡系统+186消融, 写成方法稿(LGVPA / SupCon+Pose-Aug)。**Ship概率最高但留红海、机制是旧的pose/CLIP/part**=清存货非新探索。

### ⚠️死亡清单碰撞(排除): 07的CIL-ReID=已kill 3次的donor-leakage; AGP/RWOA=遮挡红海pose/CLIP/part二次包装。

## 3. 二次包装playbook
模板: 取成熟子领域机制A→移植不挤子领域B→重命名confound→加1正交模块→跑B标准表。
例: ①频率解耦(VI)→空地VIR视频; ②RGB/CLIP特权蒸馏→event-only(隐私); ③diffusion生成(SD-ReID)→anytime(time-counterfactual); ④middle-image(MTRL灰图)→contour/sketch=ScribbleBridge。

## 4. 诚实结论
**首选 #1 ScribbleBridge**: 数据今天能拿(SYSU/RegDB/LLCM, 不像event/aerial卡release)、贴资产、单卡两周、失败风险最低、干净离开遮挡红海。
**强备选 #2 Event-only特权蒸馏**: 子领域最不挤、隐私+LUPI写法吃香, 但先确认EvReID数据可得。
**建议两条并行**: ScribbleBridge直接开跑(无数据阻塞)+ 半天确认event可得性, 谁先拿正信号主推谁。
最快保底=#6。坚决不碰=donor-leakage及遮挡红海pose/CLIP/part再包装。
