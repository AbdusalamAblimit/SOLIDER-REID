# exp376 审查说明

用户明确禁止 Claude。本实验未调用 Claude；为兼容仓库既有“启动前审查文件必须存在”的协议，
本文件只记录替代关系，不伪称 Claude 已审查。实际的两路独立 Codex 审查、问题、修复和裁决见
`codex_review.md`。

当前裁决：需先通过真实 GPU batch64 AMP preflight，之后方可启动训练。
