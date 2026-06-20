# Codex Review — exp349

**Verdict**: approve
**Date**: 2026-06-20

## 结论
codex 审查通过。clip_id_loss 经 LGPA+GCN dual 分支正确透传、processor 只加一次(parallel_aug+OA-SD 4-view 下仅 view0 kp_data 被用、loss_func 不读 clip_id_loss,无双计数);Swin-Small in_planes=768 = clip_id_proj 输入 = ViT-L clip_dim=768 维度匹配;OA-SD EMA teacher deepcopy 冻结 CLIP 安全;两次 CLIP 加载(LGPA ViT-B-32 buffer + clip_id ViT-L-14 submodule)无 state_dict key 冲突;GLOBAL_LOSS_SCALE 0.5 全系统 list-path 一致(M1);单变量 vs exp255。run349.sh 含 TEST.IMS_PER_BATCH 64 防 OOM。Verdict: approve。
