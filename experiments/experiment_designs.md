# VPReID 实验设计

---

## exp001: Swin-Tiny Baseline (无 pose, 120 epoch)

### 目的
确认纯 Swin-Tiny 在 Occluded-Duke 上的性能，作为所有改进的基准。

### 配置
使用 `configs/occluded_duke/swin_tiny.yml`，无修改。

### 运行命令
```bash
python train.py --config_file configs/occluded_duke/swin_tiny.yml \
  DATASETS.ROOT_DIR '/root/data' \
  OUTPUT_DIR './log/occluded_duke/exp001_baseline'
```

### 预期结果
- mAP: ~55%, R-1: ~65%

### 论文用途
- 主实验表格 "Baseline (Swin-Tiny)" 行

---

## exp002: VPReID v1 (完整 120 epoch)

### 目的
验证 VPReID 在 Occluded-Duke 上的完整训练性能。

### 配置
使用 `configs/occluded_duke/vpreid_tiny.yml`，无修改。

### 运行命令
```bash
python train.py --config_file configs/occluded_duke/vpreid_tiny.yml \
  DATASETS.ROOT_DIR '/root/data' \
  OUTPUT_DIR './log/occluded_duke/exp002_vpreid_v1'
```

### 预期结果
- global mAP: 55-58%
- parts mAP: 50-55%
- fused mAP: 57-62% (预期比 baseline 提升 2-5%)

### 关键观察点
- 训练是否稳定（无 loss 爆炸）
- 哪个评估模式（global/parts/fused）表现最好
- Part visibility 分布是否合理

---

## exp003: 消融 — 无 Part ID Loss

### 目的
验证 per-part ID loss 的贡献。

### 运行命令
```bash
python train.py --config_file configs/occluded_duke/vpreid_tiny.yml \
  DATASETS.ROOT_DIR '/root/data' \
  MODEL.VPREID.PART_ID_WEIGHT 0.0 \
  OUTPUT_DIR './log/occluded_duke/exp003_no_part_id'
```

---

## exp004: 消融 — 无 Push Loss

### 目的
验证 push diversity loss 的贡献。

### 运行命令
```bash
python train.py --config_file configs/occluded_duke/vpreid_tiny.yml \
  DATASETS.ROOT_DIR '/root/data' \
  MODEL.VPREID.PUSH_WEIGHT 0.0 \
  OUTPUT_DIR './log/occluded_duke/exp004_no_push'
```

---

## exp005: Part Temperature 敏感性

### 目的
分析 softmax temperature 对 part attention mask 质量的影响。

### 温度值
- 0.01 (接近 argmax)
- 0.05
- 0.1 (默认)
- 0.5
- 1.0 (接近 uniform)

### 运行命令
```bash
for temp in 0.01 0.05 0.1 0.5 1.0; do
  python train.py --config_file configs/occluded_duke/vpreid_tiny.yml \
    DATASETS.ROOT_DIR '/root/data' \
    MODEL.VPREID.PART_TEMP $temp \
    OUTPUT_DIR "./log/occluded_duke/exp005_temp_${temp}"
done
```

---

## exp006: N_PARTS 敏感性

### 目的
分析不同部件数量对性能的影响。

### 部件数
- 3 (head+torso+legs)
- 5 (默认)
- 注意: 需要修改 COCO_PART_GROUPS

### 运行命令
```bash
python train.py --config_file configs/occluded_duke/vpreid_tiny.yml \
  DATASETS.ROOT_DIR '/root/data' \
  MODEL.VPREID.N_PARTS 3 \
  OUTPUT_DIR './log/occluded_duke/exp006_3parts'
```

---

## 实验优先级

| 优先级 | 实验 | 预计时间 | 依赖 |
|--------|------|---------|------|
| P0 | exp001 (Baseline) | ~2h | 无 |
| P0 | exp002 (VPReID v1) | ~2h | 无 |
| P1 | exp003 (消融 Part ID) | ~2h | exp002 |
| P1 | exp004 (消融 Push) | ~2h | exp002 |
| P2 | exp005 (Temperature) | ~10h | exp002 |
| P2 | exp006 (N_PARTS) | ~4h | exp002 |

**建议**: exp001 和 exp002 可以在两张卡上并行跑。
