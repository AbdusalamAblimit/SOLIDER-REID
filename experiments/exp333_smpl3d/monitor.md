# exp333 监控

## 配置
- 机器：lab-3090-d（3090，conda env solider-reid，torch1.13+mmcv）
- 两臂顺序同机：`exp333_baseline`（--use_smpl off）→ `exp333_smpl_beta`（--use_smpl on, beta, w3d=1.0）
- 同 config/seed=1234，TEST.IMS_PER_BATCH 64，120 epoch，EVAL_PERIOD 10
- log：`/tmp/exp333_train.log`；输出 `log/occluded_duke/exp333_{baseline,smpl_beta}`

## 审查
- Claude broad review PASS（claude_review.md）；Codex v2+v3 approve（codex_review.md，含 smoke 抓到的 batch-balance 崩溃修复）
- smoke3 端到端跑通（1ep + 5-alpha eval + done）

## 进度记录

### [18:50] 启动 baseline 臂
- Epoch[1] loss=10.472（app=10.472, 3d=0.000，control 无 3D 分支 ✓）。GPU 6.7G / 95%。正常。
- 预期：baseline ~3h（120ep），随后 smpl 臂 ~3h。每 10 epoch eval mAP。

## 判据
- headline = smpl 臂最佳 alpha 的 mAP vs baseline mAP（同 seed 同机）。
- 期望（用户强先验）：+1~3 mAP，重遮挡子集更高。
- 诚实：alpha 为 test-time 融合超参；模型贡献 = "加 3D 分支" 整体。
