# exp378 RG0 Codex 双重审查

日期：2026-07-16

## 审查范围

- `config/defaults.py`
- `model/modules/task_adaptive_pose_field.py`
- `model/pose_backbone_model.py`
- `processor/processor.py`
- `configs/occluded_duke/exp378_rg0_external_gaussian.yml`
- RG0 design、unit、full-model invariants、data audit 与 CUDA preflight

本轮只使用两个独立 Codex 子代理，未使用 Claude。

## 静态实现结论

两名审查者均未发现阻塞性 correctness/backward-compatibility bug：默认 `raw` 不实例化
renderer、不增加参数或持久 buffer、不消耗 RNG；RG0 使用 person-0 heatmap、score 与 mask；
shared renderer 与旧 TAPF 方程顺序一致；zero-mass 正置信度显式失败；PSG 保持唯一 sigmoid
边界；RG0 stats 不进入 loss/optimizer。

R0 与 RG0 config 除注释、独立 output、renderer mode 与固定 sigma 外一致。RG0 只能归因于完整
Gaussianization 数值域变换，不能拆成 positive clamp、confidence、sigma 或 Gaussian 的独立贡献，
也不能回答 internal/external source。

## 审查提出的启动阻塞门禁

1. 扫描 train/query/gallery 全部 target-person cache 与实际 dataset 输出，确认 shape、dtype、finite、
   score 范围、positive mass、zero-mass 正置信度、renderer 范围与 peak-confidence parity；
2. 在 4090 PyTorch1.13.1+cu117 使用真实 pretrained、生产 batch64、train-mode、标准 loss、AMP、
   backward 与 scaler step，确认 PSG gradient/delta、显存和单次 sigmoid；
3. candidate R0 与已报告 R0 exact commit 在同一固定生产 batch 比较 batch、初始 state、optimizer、
   descriptor、featmaps、loss、gradient 和一步 update；
4. 共享 TAPF renderer 重构必须补旧实现 CUDA parity、旧 checkpoint strict load，并重跑既有
   12 项 unit、full invariants、MR-F0/MR-P0 e11 CUDA preflight。

## 审查后已落实的改动

- 新增全量真实数据审计脚本，覆盖 cache、无增强 dataset 输出和固定 seed 的完整 train augmentation；
- 对train的`pad=10`全部`21×21=441`个crop位置做解析式穷举，严格复现keypoint OOB后
  confidence release和`384×128→96×32` bilinear采样质量；水平flip由镜像offset及左右关节置换
  等价覆盖，不以单次seed增强冒充120 epoch全覆盖；
- 新增生产 batch64 CUDA/AMP preflight 与跨 repo R0 snapshot；
- 跨repo snapshot现在强制expected HEAD、tracked-clean、repo cwd、config/pretrain/pose-index SHA，
  并由脚本硬比较batch、初始state、optimizer、descriptor、featmaps、loss、gradient、全部PSG边界
  及一步update SHA；两次独立PASS不再视为exact parity；
- 训练日志增加 raw negative、score out-of-range、mass、sigma min/max/上下界命中率与
  rendered peak-confidence error，只在 `LOG_PERIOD` 采样，避免每 batch 多次 `.item()` 同步；
- 增加 Stage-3-only、non-spatial PSG、无其它 pose 模块、external/TAPF sigma exact match 的
  fail-closed guard；
- 明确 RG0 禁止无 score 语义的 `scene_heatmaps_override`。

远端四项生产门禁通过前，RG0 不得正式启动。
