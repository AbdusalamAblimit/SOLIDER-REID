# exp011: PSG Stage 3 — 200 Epochs Extended Training

## 动机
exp007 (PSG Stage 3, 120 epochs) 的 mAP 在后段仍在上升（ep100: 58.2% → ep120: 58.3%），曲线没有完全收敛。exp009 (Multi-stage PSG) 也展示了"后发优势"模式（ep50 才追上 exp007）。这些都暗示 PSG 模型可能需要更长训练时间来充分学习 pose-to-gate mapping。

200 epochs 训练是一个零风险的实验：同样的模型架构、相同的超参数，只是训练更长时间。如果确实能带来额外提升，说明 PSG 有被低估的潜力；如果无提升，说明 120 epochs 已足够。

## 核心假设
PSG 的零初始化 gate 需要更多 epoch 来逐步学习有效的 spatial attention pattern。120 epochs 可能不够让 PSG 达到其性能上限。

## 技术方案
- 与 exp007 完全相同的架构：PSG Stage 3, 2 个 PSG gate, 102K extra params
- 唯一变化：MAX_EPOCHS 120 → 200, WARMUP_EPOCHS 20 → 30 (等比例), CHECKPOINT_PERIOD 200
- LR schedule: cosine warmup 30 epochs, peak LR 0.0008, cosine decay 到 ep200
- Config: `configs/occluded_duke/pose_psg_200ep.yml`
- Output: `./log/occluded_duke/exp011_psg_200ep`

## 预期结果
- 最优情况：mAP 59-60%，证明 PSG 有更高上限
- 中性情况：mAP 58.3-58.5%，说明 120 epochs 已足够
- 最差情况：mAP < 58%，说明过拟合（不太可能，因为 cosine decay 会收敛）

## 对照组
- Baseline: exp007 (PSG Stage 3, 120 epochs, mAP 58.3%, R1 67.9%)
- 消融变量: 训练总 epochs (120 → 200)
