# Claude Broad Review — exp324h_lora_oracle.py

**审查通过**（eval-only 诊断脚本，无训练、无 commit）。Opus 子代理逐行审查 + 在 lab-3090-d 上实跑 LoRA 加载验证。

## 范围
`scripts/exp324h_lora_oracle.py` + 5 个被复用的兄弟脚本（exp324g/f/d/b/_dino）。对照 live artifacts（peft 0.19.1、e10 checkpoint、swin npz、缓存 pooling 矩阵、运行中的 exp324d GPU 状态）。

## 实跑验证（非纯静态）
- 保存的 adapter = 48 keys（12 层 × {query,value} × {lora_A,lora_B}），config r=16 alpha=16 dropout=0.05 targets=[query,value] —— 与脚本 CLI 默认完全一致。
- `set_peft_model_state_dict`: unexpected_keys=0；缺失的全是 frozen base 权重（已被 from_pretrained 实例化），无 LoRA key 缺失。
- `lora_B` norm 0.0 → 0.745（sum|lora_B|=1469.6），证明训练好的 adapter 真正覆盖了随机/零初始化 → **结果不是 random-init LoRA，是真判别化的 adapted-DINO**。
- exp324d e10 log: part ALL=44.67, part HEAVY=36.78；exp324g summary: dino_heavy=8.65, oracle_gain=+0.12, p_only=0.20%, jaccard=0.0619 —— 脚本硬编码的 frozen_baseline 四个数全对。
- GPU: 4542/24576 MiB（exp324d 一个进程），~20G free，eval no_grad fwd_bs=16 无 OOM 风险。

## Findings
### Critical / High
无。

### Medium
- **M1（已修）**：`set_peft_model_state_dict` strict=False 会把 223 个 frozen base 权重报成 missing，原 WARN 会每次误报"223 missing keys"。已改为只对 `lora_` key 报警（真缺失才是 adapter 加载失败）。
- **M2（文档澄清，非 bug）**：`part-MaxSim ALONE (all)` 打印的是 ALL=44.67，heavy=36.78 在 oracle 块的 dino_heavy_map 单独复现，两者都对。

### Low
- L1 fwd_bs=16 vs exp324d 的 32：eval/no_grad + BNNeck running stats + per-image part-MaxSim，bs 不影响数值，只影响吞吐。
- L2 re-rank 主动跳过并诚实披露（repo re_ranking(only_local) 需要完整 (Q+G)² distmat，仅有 q-g block 不可重建）。fusion sweep 是可行的 beat-75 测试。
- L3 GPU OOM 风险低，~20G free，无需处理。

## 正确性确认
- 对齐：exp324g/exp324h 都用 list_imgs-sorted 枚举顺序建 DINO distmat，align_dino_to_swin 按 filename 排到 swin 顺序 + pid 相等 + camid 常偏移断言；heavy mask 在 swin 顺序上算并应用 —— 与 exp324g 一致。
- oracle 数学：topk_excluded / per_query_ap verbatim 复用 exp324g，同 exclusion、同 junk skip、p_dino_only=n_dino_only/n_valid、oracle_gain=oracle-swin。
- import 绑定：list_imgs←exp324b(1-arg)、compute_heavy_mask←exp324f(1-arg)，无与 exp324b 2-arg 版冲突。
- eval 对称：encode_split 返回 (bn,pp,vis)，正确用 pp(L2-norm parts) + vis>0 喂 part_maxsim_distmat，与 exp324d run_eval / exp324f get_dino_distmat 一致。

## VERDICT: PASS
脚本正确、安全可跑。M1 已修。无 Critical/High，无 train/test 不对称，无 OOM 风险。
