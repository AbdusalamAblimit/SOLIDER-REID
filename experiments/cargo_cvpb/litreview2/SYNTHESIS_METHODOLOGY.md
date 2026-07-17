# B 类 ReID 方法稿"创新构造方法论"综合(28 deep-codex 完整读, ~167 篇方法, 2026-06-24)

> 用户铁律: 不抄模块, 学**怎么把一个观察构造成能发的创新**。目标 CCF-B 方法稿。
> 本文 = 28 份 deep-codex(每份完整 Read, 0 截断)的横向综合 + 应用到我们自己资产。

## 一、通用配方(28 份全收敛, 无一例外)

1. **绝不先抛模块**, 先抓一个**具体失败 / 反直觉观察**(baseline 在哪崩、什么数字反常)。
2. **重定义**: 把失败改名成更尖锐的"隐藏变量 / 旧假设不成立"——"大家以为 X, 其实是 Y"。**这步最值钱、抄不来、是 novelty 的真正来源。**
3. **机制从重定义自然长出来**, 每个模块 ↔ 一个失败源(绑定越紧越好; 松了就是堆模块)。
4. **关键证据证"重定义对"**(不是证"机制涨点"): 一个**如果失败就推翻叙事**的诊断指标/消融/可视化。
5. **reviewer 买视角不买模块**。模块可以全是旧零件(GNN/attention/OT/CLIP), 只要被一个新问题串住。

## 二、20+ 个"重定义动作"catalog(我们的招式库)

| # | 动作 | 样本 |
|---|------|------|
| 1 | 隐藏变量 | DMDL 模态偏置"从数据传到标签传到特征"(因果图); DMPF 模态+姿态双因子 |
| 2 | **把问题数学化** | GAReID: 相似度=所有 part-pair 平均, 错配对 >> 对齐对, 一个公式解释 misalignment |
| 3 | 可测中间变量 | 梯度一致性; MDRR 模态分歧降低率; NCC(C→X) 因果强度 |
| 4 | 物理/常识约束 | GSTNET 地理可达性(5秒跨20km不可能=图边) |
| 5 | 偏差诊断 | EAIBC 颜色过度依赖; MSP 发型捷径 |
| 6 | **"太晚了"/用错位置** | HCCL 噪声在前向传播就污染了; Pose-Skeleton 遮挡信息扩散→中间层拦; training-free TI-ReID 图库结构当**测试期先验**("Prototype-in-Training 反而掉点") |
| 7 | **改信号的角色** | 衣服文本不识别人而是"告诉模型压制什么"; LVLM→身份语义token; prompt→可学习中间模态 |
| 8 | 数据中心反直觉 | 车辆从"噪声该丢"→"positive-incentive noise" |
| 9 | per-pair 最优条件 | 每对图像有自己的最优光照距离, 最优不一定是正常光照 |
| 10 | **非对称包含** | PDA: 文本分布⊇图像分布; 方差=语义范围(非噪声) |
| 11 | 回收"该丢的" | 形状不是噪声(红外形状估计错误才是); 噪声样本历史自校正 |
| 12 | **表示形态错了** | 3D Gait: 稀疏SMPL参数没法和稠密appearance融→蒸成稠密时空场; "换辅助模态"消融证 dense>skeleton>SMPL参 |
| 13 | 新协议贴部署 | severe modality imbalance; mix-modality; anytime |
| 14 | 隔离混杂变量(负结论也发) | Rethinking Joint Opt: 联合优化的收益其实来自尾部解析 |
| 15 | **修训练组织非模型** | curriculum CC-ReID: 先单衣后逐步加最难衣, 不改模型不加输入 |
| 16 | "顺序错了" | Two-stage KD: 先同模态收紧再跨模态 |
| 17 | **因果(最强)** | P(Y\|X)→P(Y\|do(X)) backdoor 切断衣服捷径 |
| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
| 19 | 旧法在新基座失效 | CLIP 细粒度: 朴素局部切分伤预训练空间("朴素 baseline 比 global 还差") |
| 20 | **对齐伤判别性** | CycleTrans neutral-yet-discriminative; BDLF base/detail; 多频身份线索 |
| 21 | 目标函数错配 | CAP: 直接优化跨模态 AP |
| 22 | 数字驱动 | CLNS: 跨相机正样本距 1.06 > 同相机 0.78 → 相机结构噪声 |

**证"重定义对"的硬手段**: 数学分解 / 失败检索样例 / 可测诊断指标 vs per-query AP 相关 / 互补性 Venn / 伪标签质量曲线 / 前后 attention map / "替换机制"破坏性对照 / 参数曲线(证不是越多越好)。

## 三、应用到我们自己(候选 re-framing, 每个带廉价 kill-switch)

我们独有资产: CARGO/AG-ReID.v2 极端跨视角(航拍↔地面 90°, 航拍低清俯视); **观察: avg-pool 52.37 > token-MaxSim 45.19(差7分)**; SMPL 几何基建; Swin 67.33(backbone)。

> ⚠️ 用户已打掉"MaxSim<avg→局部不可靠→用avg"(推到底=零贡献)。**新候选的硬门槛: 机制必须 beat avg, 不能退化成 avg。** aerial-ground 几何/可见性是红海(避开)。

### 候选 B(主推): 非对称包含 — 把对称匹配换成"航拍证据⊆地面证据"
- **观察**: 航拍低清俯视 = 信息**欠定**(看不清脸/纹理); 地面高清正面 = 信息**确定**。但所有 cross-view ReID 用**对称** cosine/MaxSim 匹配, 默认两视角信息对等。
- **重定义(move 10+22)**: "大家以为跨视角是对称对齐问题, 其实航拍↔地面是**非对称信息包含**: 航拍身份证据是一个**范围(宽分布)**, 应被地面的**窄分布包含**。" 方差=该视角的信息欠定度(非噪声)。
- **机制(自然长出)**: 每张图建成分布(均值+方差), 航拍方差大/地面方差小; 包含损失让航拍分布⊆地面分布(非对称, 不是拉近两个点)。
- **证重定义**: ① 航拍图特征方差是否系统性 > 地面(分布可视化); ② 非对称包含距离是否同时 beat 对称cosine 和 MaxSim; ③ 只在高视角差样本收益更大。
- **切开(避红海)**: vs PDA(文本-图像) = 我们是 cross-VIEW; vs OT-ReID(CM-EMD/G2DA) = 我们的非对称方向(航拍⊆地面)由**成像物理**(俯视低清欠定)定, 不是纯视觉 cost。
- **kill-switch(零训练)**: frozen Swin(swin_fix256, 67.33)提特征, CARGO A↔G: 对称cosine vs 非对称包含距离(航拍当宽高斯/地面窄高斯, 用马氏或KL包含)。包含明显赢 → re-framing 有腿; 打平 → 死, 回头。

### 候选 C: 对齐伤判别性(move 20, CycleTrans 思路迁到 cross-view)
- **观察**: avg>MaxSim 说明强行局部对齐(MaxSim)在极端跨视角**有害**。
- **重定义**: "航拍和地面不共享可对齐的**局部**, 只共享**全局身份**+**视角特有**判别线索; 强行对齐(MaxSim)塌掉视角特有线索, 均匀平均(avg)稀释它。" → 学"视角中性但保判别"特征。
- **风险**: CycleTrans/BDLF 已做 neutral-yet-discriminative(cross-modal), 切开点是 cross-view; 较弱, 当候选 B 的备胎。
- **kill-switch**: frozen Swin, 中性特征(去视角分量)+视角特有残差 分开重组 vs avg。

### 候选 D: 因果(move 17, 最强范式但需想清 confounder)
- **重定义**: "cross-view ReID 普通训练学 P(Y|view-entangled X), 视角是 confounder; 应学 P(Y|do(view)) 用 backdoor 对所有视角求和切断'视角-身份'伪相关。"
- **风险**: Causal CC-ReID 已用 backdoor(confounder=衣服); 我们 confounder=视角, 切开点要硬。需想清 confounder dictionary 怎么建(视角桶?)。
- **kill-switch**: NCC(view→X) 因果强度, do-intervention 前后 per-view ID 可分性。

### 候选 E(弱, 记录): 表示形态 + SMPL view-canonical
- "Beyond geometry"(deep_3#6)已做 UVTexture canonical-viewpoint for aerial-ground(AG-ReID.v2)→ **直接撞车**, 降优先级。

## 四、下一步(用户授权: 拿不定和 codex 讨论)
启 4-5 个 codex(--search)对候选 B/C/D 做: ① novelty/撞车检索(PDA/OT-ReID/CycleTrans/causal-ReID 边界); ② 哪个最强 + 为什么; ③ kill-switch 设计是否真能证伪。**绝不动手前先查 novelty + 必须有廉价 kill-switch(铁律)。**
