# exp260: Swin-Base GCN512 + 2-stage PSG 全套

## 动机
exp255 (Small) MaxSim+FlipTest 已达 75.2/85.6，但这是 Swin-Small。
Base backbone 参数量 ~2x Small，预计训练端 +2-3% mAP。
论文需要多 backbone 尺度数据来支撑跨尺度一致性论述。

## 核心假设
Base backbone 更强表达能力 → 所有模块(PSG/GCN/LGPA-D/OA-SD)均应保持或放大增益。

## 技术方案
- Backbone: Swin-Base (WITH_CP=True 控制显存)
- 完全继承 exp255 架构: GCN512 + 2-stage PSG + LGPA-D + OA-SD + PLBOA(OccDuke) / 无PLBOA(Market)
- LR=4e-4 (Base backbone 标准, 比 Small 低一半)
- TEST.IMS_PER_BATCH=128 (Base 模型更大，eval 需要降 batch)

## 变体
- exp260: Base GCN512+2stage on **Occluded-Duke** (本地 3090)
- exp260-mkt: Base GCN512+2stage on **Market-1501** (本地 3090, OccDuke 之后)
- 跨数据集: Market 权重 → Occluded-ReID 测试

## 对照组
- exp255 (Small GCN512+2stage): 73.2/83.3, MaxSim 74.1/84.6, MaxSim+flip 75.2/85.6
- exp249 (Small GCN256+1stage): 71.9/81.8

## 预期结果
- OccDuke: equal_concat 75+, MaxSim 76+, MaxSim+flip 77+
- Market: mAP 94+, R1 97+
