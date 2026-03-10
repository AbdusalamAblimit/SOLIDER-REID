# exp015: PSG with Spatial-Aware Gate (3×3 Depthwise Conv) 监控日志

## 训练配置
- Config: `configs/occluded_duke/pose_psg_spatial.yml`
- Output: `./log/occluded_duke/exp015_psg_spatial`
- Epochs: 120, LR: 0.0008, Warmup: 20ep cosine
- PSG: 2 modules (Stage 3) + 3×3 depthwise conv
- 额外参数 vs exp007: +576 per module = +1152 total (可忽略)
- 对照: exp007 PSG-only (mAP 58.3%, R1 67.9%)

