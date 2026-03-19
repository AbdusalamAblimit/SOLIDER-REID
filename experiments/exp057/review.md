# exp057 KDL 审查报告

## 审查范围
- `experiments/exp057/design.md`
- `configs/occluded_duke/pose_psg_gcn_kdl.yml`
- `config/defaults.py`
- `loss/make_loss.py`
- `processor/processor.py`
- `log/occluded_duke/exp057_kdl/train_log.txt`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | LOW | loss/make_loss.py | KDL 固定使用 `17x17` 上三角 mask，写法依赖 COCO 17 关键点设定，不够泛化 | 接受 |

## 审查通过项

- `POSE_KP_DISSIMILAR` 和 `POSE_KP_DISSIMILAR_WEIGHT` 已正确进入 loss 路径
- `processor.py` 已把 `kdl_enabled` 纳入 `kp_data` 传递条件，当前代码不会再出现早期的 `kp_data` 漏传
- KDL 只作用于 `kp_feats`，不改 backbone 默认行为
- 自相似对角线被正确排除，只惩罚 cross-keypoint similarity
- 总损失叠加方式正确，默认关闭时与 baseline 完全一致
- 日志显示训练完整跑通

## 结论

✅ **通过**

当前 KDL 实现是正确的，且之前 monitor 中提到的 `kp_data` 接线问题已经被修复。剩余问题仅是代码写法对 17 keypoints 的硬编码，不影响本仓库当前实验结论。
