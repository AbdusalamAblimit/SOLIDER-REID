# exp260: Swin-Base GCN512 + 2-stage PSG

- exp260: OccDuke Base (本地 3090)
- exp260-mkt: Market Base (OccDuke 完成后)
对照: exp255 (Small): 73.2/83.3, MaxSim+flip 75.2/85.6

## 检查点

### [03:13] 启动 — ep1, 11.4 GB 显存, healthy

Base backbone 成功加载。Loss=20.3 (正常初期)。
训练脚本: OccDuke → MaxSim+flip eval → Market → cross-dataset eval (全自动)。
