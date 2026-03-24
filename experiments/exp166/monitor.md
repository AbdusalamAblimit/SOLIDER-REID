# exp166 STD-PR Per-Token + PLBOA
- 6 tokens each independently classified
- test: 6 tokens L2-norm concatenated (global_768 + 6×768 = 5376-d)

## Bug: tri_part=inf
- 6 individual token features 的 euclidean distance 太小
- Softplus(near-zero margin) → inf
- 需要 L2 normalize per-token features before triplet
- 或者改用 cosine distance for triplet
- 留待修复
