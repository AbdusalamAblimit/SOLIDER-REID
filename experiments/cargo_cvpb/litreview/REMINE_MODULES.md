# Re-mine: 论文库可移植模块(2026-06-24,用户质疑"光强baseline不是论文")

## 背景
用户问:Swin port(67.33)是 backbone 不是方法,库里到底有没有可借鉴的模块升级 OVLI?
re-mine codex 重读 19 份 lit_X.md 逐篇反推,专捞**通用可移植机制**(非 aerial-ground 重定义)。

## 残酷事实
团队资产(aerial-ground/SMPL/OVLI)把候选拽回几何空间,多数仍撞红海(GSAlign/AG-VPReID/multi-platform 占)。

## Top 5(codex 输出,Top1-3 因 tail 截断只有取舍 gist)
- Top1: SMPL 可解性约束 OVLI(怎么比)— 偏 visibility,红海
- Top2: surface prototype memory(缺什么怎么补)— OVP-like
- Top3: base/detail 解耦(SMPL base/detail + late-interaction)— 借 BDLF,解耦赛道挤
- **★Top4: Geometry-conditioned OT / Containment Matching ← 真机制升级**
- Top5: SMPL 反事实增广(diffusion)— 生成红海,只能当配套
- codex 取舍: Top1+Top2 组合(可比+补全)

## ★ 我选定: Top4 OT/包含匹配(唯一把"借来机制"升成"自己机制"的)
- **OVLI 从 ColBERT 对称 MaxSim → 几何约束非对称包含**:航拍低清 query 只证身份**范围**(宽分布),地面清晰 gallery 窄分布;transport cost = Swin相似度 + SMPL表面距离 + 可见性惩罚
- **headline 升级**:"我们用了 ColBERT late-interaction" → "跨视角身份证据是非对称包含关系,由 3D 表面几何定 cost"
- **切开**:vs OT-ReID(CM-EMD/G2DA/CVFT)= transport cost 非纯视觉,是 3D body-surface cost + **非对称包含语义**(query范围⊆gallery)
- **诚实坎**:OT 有先例,新意全压切开点;没切好就是普通 OT-ReID

## kill-switch(零训练,首验值不值得做)
frozen Swin(67.33 那个 swin_fix256)提特征,CARGO A↔G:`cosine MaxSim`(=现 OVLI)vs `几何 OT/包含距离`。
- OT/包含明显赢 MaxSim → 机制有腿,这是"自己的方法"
- 打平/更差 → 老实承认普通 OT,留 OVLI
GPU 两卡现忙(CARGO+AGv2 baseline),脚本先写好,等 GPU 空跑。
