# SOTA 对比表 (Occluded-DukeMTMC-reID)

## 数据来源: KPR (ECCV 2024) Table 3 + 我们的实验

### 无后处理结果

| Method | Venue | Backbone | mAP | R1 |
|--------|-------|----------|-----|-----|
| PCB | ECCV'18 | ResNet50 | 40.8 | 51.2 |
| PGFA | ICCV'19 | ResNet50 | 37.3 | 51.4 |
| BoT | CVPRW'19 | ResNet50 | 44.7 | 51.4 |
| HOReID | CVPR'20 | ResNet50 | 43.8 | 55.1 |
| ISP | ECCV'20 | ResNet50 | 52.3 | 62.8 |
| MHSA | AAAI'21 | ResNet50 | 44.8 | 59.7 |
| PAT | CVPR'21 | DeiT-Small | 53.6 | 64.5 |
| PGFL | MM'22 | ResNet50 | 54.1 | 63.0 |
| HG | ECCV'22 | ResNet50 | 50.5 | 61.4 |
| LDS | PR'22 | ViT-B | 55.7 | 64.3 |
| FED | CVPR'22 | ViT-B | 56.4 | 68.1 |
| OAMN | MM'22 | ResNet50 | 46.1 | 62.6 |
| VGTri | PR'23 | ViT-B | 46.3 | 62.2 |
| SSGR | AAAI'23 | ViT-B | 57.2 | 69.0 |
| **Ours (PCFC+GiLt)** | **-** | **Swin-Tiny (28M)** | **58.0** | **68.0** |
| SOLIDER | CVPR'23 | Swin-Base (88M) | 61.9 | 71.2 |
| PFD | AAAI'22 | ViT-B (86M) | 61.8 | 69.5 |
| BPBreID | WACV'23 | HRNet-W48 | 62.5 | 75.1 |
| KPR w/o prompt | ECCV'24 | Swin-Base (88M) | 73.3 | 82.5 |
| KPR | ECCV'24 | Swin-Base (88M) | 75.1 | 84.3 |

### 分析

**我们 Swin-Tiny (28M params) 的竞争力**:
- 超越 FED (CVPR'22, ViT-B): +1.6% mAP
- 超越 SSGR (AAAI'23, ViT-B): +0.8% mAP
- 超越 LDS (PR'22, ViT-B): +2.3% mAP, +3.7% R1
- 超越 PAT (CVPR'21, DeiT-Small): +4.4% mAP, +3.5% R1
- 超越所有 ResNet50 方法

**与大 backbone 方法的差距**:
- vs SOLIDER (Swin-Base, 88M, 3.1x params): -3.9% mAP
- vs PFD (ViT-B, 86M, 3.1x params): -3.8% mAP
- vs BPBreID (HRNet-W48): -4.5% mAP
- PCFC 在 28M backbone 上缩小了部分差距 (baseline 56.6 → 58.0)

### 含后处理结果

| Method | Post-Processing | mAP | R1 |
|--------|----------------|-----|-----|
| Ours | 无 | 58.0 | 68.0 |
| Ours | + NFC (global+part) | 64.7 | 69.4 |
| Ours | + Re-ranking + Part Dist | 75.3 | 74.4 |

注意: NFC 和 Re-ranking 是通用后处理方法，非我们的训练端创新。
KPR 的 75.1/84.3 也包含了 test-time prompting (可视为后处理)。
