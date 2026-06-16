# Codex Review — exp326 (DIFT) + exp327 (DINOv3)

**Verdict**: approve（修复后）
**Date**: 2026-06-16 07:1x
**Review round**: 1（needs-attention）→ 修复 → 2

## Findings（codex --search exec 原文摘要）

第一轮 verdict = **needs-attention**，无 train/test 不对称、无强制 disk-write 阻断。3 条 finding：

- **Medium — exp327 token assert 不够严**：原 `assert patch_tok.shape[1] == n_tok` 只查切片长度，若 `nreg` 读错仍可能切出 n_tok 个 token 但混入 register、丢尾部 patch。HF DINOv3 文档要求先 assert 完整布局 `[CLS, registers, patches]` 再切。
  → **已修**：改为 `assert out.shape[1] == 1 + nreg + n_tok`（查完整序列长度）。
- **Medium — exp327 默认 dinov3-b 可能因 HF gated 运行失败**：`facebook/dinov3-*` 需接受 license / token，hyy 无 token 会在 eval 前 401。
  → **已修**：默认改为 `--model dinov2reg-b`（ungated，registers 干净，正是 apples-to-apples 升级）。dinov3-b 仅在确认可下载后显式传参跑。
- **Low — exp326 非 canonical DIFT 设置**：默认 t=100/up_block=1/ensemble=4，官方 DIFT 语义用 t=261/up_ft_index=1/ensemble=8，HPatches 用 t=0/up_ft_index=2。
  → **处置**：负结果须表述为"此廉价 SD-v1.5 DIFT 设置未超 exp324"，**不等于"DIFT 被证伪"**。已在 design.md 注明可 smoke 扫 t∈{50,100,200}、up_block。

## Checks Passed（codex 确认）
- exp326 DIFT 路径：VAE latent scaling、DDIM add_noise、timestep tensor、hook clear/assert、tensor/tuple hook guard、fp32 cosine 路径均正确。
- exp327：用 AutoModel，读 patch_size/hidden_size/num_register_tokens，切片匹配 HF 文档布局。
- 下游 comparability 对齐：同 5 part group、POOL_RADIUS=1、p0 find_pose、heavy-occ vis.sum()<=8、mutually-visible part-MaxSim，与 exp324 一致。
- Disk：feature cache 可选（--cache 默认关），无强制写盘。

## Web premise check（codex 联网）
DIFT "SD 特征在语义对应上超 DINO/OpenCLIP" 前提对 SPair/PCK 成立（project page 报 +19 vs DINO / +14 vs OpenCLIP，含遮挡/姿态变体）。但 SD-DINO / Tale-of-Two-Features 把 SD 与 DINO 视作**互补**，并非对每个下游 retrieval 指标都 SD 主导 → 与 design.md 预期一致，DIFT 未必必胜，故训练-free 首验有必要。

## 结论
3 条 finding 全部处置（2 Medium 改码、1 Low 表述约束）。codex 审查通过。
