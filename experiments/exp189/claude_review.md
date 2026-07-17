# Claude Broad Review: exp189 Visibility-Weighted SupCon (Opus 4.6)

## 审查通过（第二轮，修复 Critical 后）

### 第一轮 Critical Bug（已修复）
kp_data 没有传到 loss function — visibility weights 永远不会激活。
修复：在 processor.py 的 kp_aux_data 条件中加入 `supcon_vis_enabled`。

### 代码审查
1. **part_visibility 传递** (pose_backbone_model.py): 正确。part_w (B,6) 添加到 kp_data。
2. **Visibility weighting** (make_loss.py): pv.mean(dim=0) batch 平均 → (6,) 权重 → 归一化 → weighted sum。正确。
3. **梯度流**: part_w 来自 scene_heatmaps（非梯度 leaf tensor）。无梯度流过权重。安全。
4. **后向兼容**: VIS_WEIGHT=False → 跳过，uniform average。默认 False。
5. **Fallback**: part_visibility 不在 kp_data → uniform。安全。
6. **Config**: POSE_STR_SUPCON_VIS_WEIGHT = False 默认。

### kp_aux_data 修复验证
processor.py: `supcon_vis_enabled = getattr(cfg.MODEL, 'POSE_STR_SUPCON_VIS_WEIGHT', False)`
加入 kp_aux_data 条件：`maxsim_tri_enabled or evid_enabled or supcon_vis_enabled`。
现在 kp_data（含 part_visibility）会正确传到 loss function。

### 显存
无额外显存：只改 loss weighting，不加模块。可在远程 5060 Ti 上跑。

### 单变量
vs exp176: 仅增加 POSE_STR_SUPCON_VIS_WEIGHT=True。
所有其他设置（SupCon T=0.05, PLBOA, triple injection 等）不变。

### 预期效果
PLBOA 遮挡下半身 → 下半身 token visibility 低 → weight 小 → SupCon 更关注上半身
→ 上半身 metric space 更精确 → R1 可能提升

零 remaining issue。
