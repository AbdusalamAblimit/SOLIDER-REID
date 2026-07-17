# exp383 Market→Occluded-ReID TAPF 监控

## 2026-07-17 14:54：设计与数据门禁

- 状态：`DESIGN ONLY / 未实现 / 未启动`
- GPU：4090 `2 MiB / 0%`，无训练进程。
- Market：12,936 train / 3,368 query / 19,732 gallery JPG；三 split pose index 存在。
- Occluded-ReID：200 ID，1,000 遮挡 query 候选 + 1,000 全身 gallery 候选，均为 `.tif`；无训练 split。
- 结论：数据门禁可进入实现预检，但不得把 Occluded-ReID 称为第二训练数据集。
- 当前阻塞：`test_on_occluded_reid.py` 对所有 `POSE_ENABLED` 模型强制索取测试 pose；TAPF 必须先获得真正无 pose 文件的 evaluator 路径与 exact parity。
- 动作：只完成 design/claim-gap 文档；未创建 config、repo、output，未启动训练。

## 2026-07-17 23:12：RGB-only evaluator 本地实现门禁

- 状态：`LOCAL IMPLEMENTATION PASS / REMOTE CUDA PREFLIGHT PENDING / 未启动`
- 专用 Occluded-ReID evaluator 新增 `_uses_external_pose_at_eval`：只有
  `POSE_ENABLED=True && POSE_TAPF=False` 的 legacy 模型才构造 `PoseImageDataset`、检查 pose index
  并解析七元组 batch；TAPF 改用普通 `ImageDataset` 与六元组 RGB batch。
- TAPF 特征抽取入口会把任何误传的 external pose 对象强制置为 `None`；同时将 TAPF 从
  `_plain_pose_part_model` 排除，避免无 PSG wrapper 的 ViT-TAPF 被误路由到 legacy part evaluator。
- legacy 行为保持：缺少 `pose_data/query|gallery/index.json` 时仍明确抛错，不静默伪造姿态。
- 新建 matched 配置：
  - `configs/market/exp383_b0.yml` SHA256=
    `b847ab3dbe4c90b34696250883784eda75a0df06711f78128dc152547f5e20a9`；
  - `configs/market/exp383_d0.yml` SHA256=
    `c456f9487ff85fbe21acd099a1b1d6ed9b801cf604ccedd3f898a51b9631683c`。
- 配置静态门禁：两臂 `INPUT/DATASETS/DATALOADER/SOLVER/TEST` 全节点相同，均为
  Market、`RE_PROB=0.5`、batch64、seed1234、120 epochs、每10 epoch checkpoint/eval、
  `FLIP_TEST=False`；B0为标准RGB Swin-T，D0只增加完整训练期`anchor+PSG`原子方法及独立output。
- 本地 `uv run pytest -q tests/test_exp383_occluded_reid_posefree.py`：`4 passed`。覆盖：
  TAPF/legacy policy、无pose文件RGB loader、legacy缺pose失败、六元组batch解析、误传
  exploding pose强制为None、两臂配置单变量合同。`py_compile`与`git diff --check`通过。
- 远端只读复核（15:11 UTC）：4090=`2 MiB / 0%`、无训练进程；Market与Occluded-ReID目录仍在。
- 当前边界：这只证明本地policy与配置合同，不等同于真实数据、full model、CUDA/AMP或checkpoint
  parity。尚未创建生产repo/output，尚未启动训练；下一步必须在原生torch1.13.1环境完成完整预检。
