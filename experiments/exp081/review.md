# exp081 PQTD 代码审查记录

## 完整审查 — 通过 ✅

### 关键验证结果
- **接口兼容**: PQTD.forward() 返回 (list, list, dict)，与 GCN 路径完全兼容
- **Shape 链**: feat_map(B,768,12,4) → memory(B,48,256) → decoder(B,5,256) → output(B,768) ✅
- **Loss 路径**: list-loss 自动 50/50 global/part split ✅
- **Test feat**: equal_concat(global_768, pqtd_768) = 1536d ✅
- **kp_data={}**: 空 dict 在 processor 中安全跳过所有辅助 loss ✅

### 发现的问题
| ID | 严重程度 | 描述 |
|----|---------|------|
| M1 | Medium | _init_weights() 重写 MultiheadAttention 的 out_proj (xavier，可接受) |
| M2 | Medium | forward() 中动态 import merge_person_heatmaps (工作正常，只是风格问题) |
| L4 | Low | pose PE grid 每次 forward 创建 (12×4 很小，开销忽略) |

### 结论
零 Critical/High 问题。训练可以继续。
