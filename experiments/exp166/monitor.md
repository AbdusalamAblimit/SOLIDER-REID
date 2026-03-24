# exp166 STD-PR Per-Token + PLBOA
- 6 tokens each independently classified
- test: 6 tokens L2-norm concatenated (global_768 + 6×768 = 5376-d)

## Bug 已修复: tri_part=inf → L2 normalize per-token features

## 早期结果: R1 大幅改善！
- Per-token+PLBOA ep40: +1.3 mAP / **+3.2 R1** vs V1 mean
- Per-token alone ep40: **+3.9 mAP / +6.5 R1** vs V1 mean
