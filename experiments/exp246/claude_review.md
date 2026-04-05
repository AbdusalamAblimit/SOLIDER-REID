# exp246 Claude Review: LGPA-D + GCN 双分支

## 审查范围
Config-only: 同时开启 POSE_LGPA + POSE_SKELETON_GCN + POSE_LGPA_DETACH。
代码已支持此组合 (pose_backbone_model.py 的 LGPA+GCN dual branch path, line 506-515)。

## 验证
| 检查项 | 状态 |
|--------|------|
| LGPA+GCN dual path 存在 | PASS (line 506-515: if self.use_skeleton_gcn) |
| 两者都 detached | PASS (LGPA via _lgpa_detach, GCN via featmaps[-1].detach()) |
| 输出结构正确 | PASS ([global] + lgpa_cls + gcn_cls, [global] + lgpa_feats + gcn_feats) |
| Test path | PASS (line 692-699: LGPA feats + GCN feats) |
| Loss 结构 | PASS (list-loss, 自动 0.5x global) |
| 单变量 (仅加 GCN) | PASS (vs exp244 LGPA-D only) |

## 结论

审查通过。已有代码路径, config-only 开启。
