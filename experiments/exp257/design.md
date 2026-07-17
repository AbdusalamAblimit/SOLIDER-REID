# exp257: ArcFace + Label Smoothing on Small GCN512 + 2-stage PSG

## 动机
exp255 (Small GCN512 + 2-stage PSG) 当前最佳 73.2/83.3, MaxSim 73.5/83.8。
目标 75% mAP，差距 1.5%。
ArcFace (angular margin) 预估 +0.3-0.7%, Label Smoothing 预估 +0.2-0.5%。

## 核心假设
ArcFace 增加类间可分性 (702 classes)，Label Smoothing 正则化防过拟合。

## 技术方案
- MODEL.ID_LOSS_TYPE: softmax → arcface
- SOLVER.COSINE_MARGIN: 0.35 (conservative start)
- SOLVER.COSINE_SCALE: 30
- MODEL.IF_LABELSMOOTH: off → on
- 注意: ArcFace 仅作用于 global classifier，LGPA/GCN part classifiers 仍用 softmax（代码限制，合理设计）
- Label Smoothing 作用于 ArcFace 输出 logits 的 CE loss（双重正则化，intentional）

## 变体
- exp257: ArcFace + Label Smoothing (远程)
- exp257b: Label Smoothing only (本地, 消融)

## 对照组
- exp255 (softmax, no label smooth): 73.2/83.3, MaxSim 73.5/83.8

## 预期结果
- 成功: 74+ mAP eq, MaxSim 74.5+
- 中性: ≈ exp255
- 失败: < exp255 (ArcFace margin 过大导致不收敛)
