# 实验 exp383：Market→Occluded-ReID TAPF 数据集与跨域验证

## 动机

完整 `anchor+PSG` 已在 Occluded-Duke 的 Swin-T、ResNet-50、ViT-B 三个骨干上取得同骨干
D0−B0 的正 mAP 差，但全部是同一数据集、seed 1234。下一笔算力应优先回答数据集/域迁移，而不是
继续堆 backbone 或重复已关闭的 Hierarchical/Video 路线。

Occluded-ReID 没有官方训练 split，不能把它写成“第二训练数据集”。本实验使用 Market-1501 作为
第二训练域，fresh 训练 matched B0/D0；同一个 e120 checkpoint 同时在 Market-1501 域内协议与
Occluded-ReID 遮挡跨域协议上评估。

## 核心假设

如果 `anchor+PSG` 学到的是可迁移的训练期结构先验，而不只是 Occluded-Duke recipe 的容量重标定，
则 Market-D0 相对 Market-B0 至少不应明显损伤域内检索，并应在完全 pose-free 的
Market→Occluded-ReID 上提供独立正向证据。

## 数据审计

2026-07-17 远端只读核验：

- Market-1501：`12,936` train JPG、`3,368` query JPG、`19,732` gallery JPG；
- Market `pose_data/train|query|gallery/index.json` 均存在；
- Occluded-ReID：200 ID，`1,000` 遮挡 `.tif` 与 `1,000` 全身 `.tif`，无 train split；
- 原始目录为 `occluded_body_images/` 与 `whole_body_images/`，评测仓库只允许在独立 repo 中建立
  `query/`、`gallery/` 只读软链接，不修改原始数据；
- 4090 当前 `2 MiB / 0%`，无训练进程。

## 单变量矩阵

固定 Swin-T、同一预训练权重、Market 标准增强、batch 64、seed 1234、SGD、120 epochs、同一
sampler/optimizer/scheduler：

| arm | 训练期外部 pose | 内部 anchor | PSG | 推理期外部 pose | 评测 |
|---|---|---|---|---|---|
| M-B0 | 否 | 否 | 否 | 否 | Market + Occluded-ReID |
| M-D0 | 是，仅 teacher target | 是 | 是，单 anchor 对应后继 PSG | 否 | Market + Occluded-ReID |

两臂都使用 Market recipe 的 `RE_PROB=0.5`。D0 相对 B0 只增加完整 `anchor+PSG` 原子方法及其训练期
pose loss；不加入 HT0、geometry residual、LGPA、GCN、test-time prompt、flip-test、re-ranking
或 MaxSim。

## 必须先修复的评测边界

当前 `test_on_occluded_reid.py` 只要 `POSE_ENABLED=True` 就强制要求 query/gallery pose 文件，
这与 TAPF 的 RGB-only inference 定义冲突。正式训练前必须：

1. 让 `POSE_TAPF=True` 的 global evaluator 使用普通 `ImageDataset`，不读取或伪造 external pose；
2. 用同一 checkpoint 验证 correct/shuffle/None/exploding external pose descriptor exact parity；
3. 在没有 Occluded-ReID pose_data 的情况下完成 dry-run 与 batch64 CUDA eval；
4. 保持其他历史 pose-enabled model 的 evaluator 行为不变。

这属于部署协议修复，必须同时作用于 D0 的 Market 与 Occluded-ReID eval，不能成为性能变量。

## 训练前门禁

1. 独立 fresh repo、exact commit/full-history bundle/预训练权重/config SHA 全记录；
2. tracked source clean，两个 output 均不存在，GPU 空闲，只允许串行 B0→D0；
3. config diff 证明除原子方法与 output 外均 matched；
4. unit、full-model invariants、真实 batch64 CUDA/AMP、真实 overflow、state/RNG/optimizer 门禁通过；
5. B0 两次 10-step parity；D0 e1/e11 route、loss、梯度归属与 handoff 通过；
6. Market 与 Occluded-ReID 的 pose-free exact parity 通过；
7. 对 B0/D0 报告总参数、trainable 参数、FLOPs、batch64 训练显存/吞吐与单图推理时延；
8. 任一门禁失败均不得启动正式训练。

## 运行与监控

- 两臂都自然跑满 e120，不以单个 epoch、Market 饱和或跨域早期值提前停止；
- 每 10 epoch 保存 checkpoint 并完整评估 Market；
- B0 终审通过、原 PID/workers 退出且 GPU 空闲后，才 fresh 启动 D0；
- D0 每次 eval 显式计算相对同 epoch B0 的 mAP/R1/R5/R10；
- e120 后用固定 checkpoint 分别跑 Market 与 Occluded-ReID，禁止挑 best 替代 final；
- 全程检查 NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow 与参数轨迹。

## 结果裁决

本实验是描述性单 seed 数据迁移证据，不设置早停阈值。final 后按以下边界解释：

- **支持跨域扩展**：Market mAP 不低于 B0 超过 `0.3` pp，且 Occluded-ReID mAP 正差至少 `+0.8` pp、R1 非负；
- **弱描述性支持**：Occluded-ReID mAP 为 `+0.3` 至 `+0.8` pp，且 Market 无明显退化；
- **数据集迁移 NO-GO**：Occluded-ReID mAP 非正，或 Market mAP 退化超过 `0.5` pp；
- 无论结果如何，都不把单 seed 写成统计显著性；多 seed 只在两域方向支持后补主骨干 B0/D0。

## 风险与失败解释

- Market 是相对饱和的域内 benchmark，D0 域内中性不自动否定遮挡跨域价值；但明显负差必须报告。
- Occluded-ReID query 为遮挡图、gallery 为全身图，正差可能来自跨域鲁棒性，不能直接等同于原生
  Occluded-ReID 训练收益。
- 若 evaluator 仍暗中读取 pose，本实验全部无效。
- 若只在 Occluded-ReID 上涨而 Market 明显下降，不能写成无代价迁移。
