# exp395 AMP 首步梯度归属监控

## 状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / PHASE 0S STATIC-CPU SEALED-PASS /
CUDA ATTRIBUTION IMPLEMENTATION STATIC SEALED-PASS / CUDA EXECUTION NO-START /
FORMAL NO-START`

## 2026-07-19 接手与边界冻结

- exp394 保持 `CUDA_AMP_PREFLIGHT_SEALED_FAIL`；未修改、未重跑；
- 本地 HEAD=`be6844ee13d2da031c229a376c4c877861c8d4b8`，接手时 tracked clean；
- exp395 目录此前不存在，本轮作为独立实验新建；
- 已只读核对 exp394 loss seam、两个 `exec_losses`、原 preflight 的 parameter groups 与
  `scale -> backward -> unscale -> finite check -> step` 顺序；
- 当前没有证据将 non-finite 归到任何具体 loss/head/router；文档明确禁止此类推测；
- 当前只允许 design/protocol 与 static/CPU contract，CUDA、正式训练和 semantic multi-stage 均
  `NO-START`。

## 2026-07-19 Phase 0S static/CPU执行

- 使用工作目录uv环境，设置`CUDA_VISIBLE_DEVICES=''`；未读取official dataset、pose、CLIP checkpoint或
  codebook；
- 首遍与repeat均`PASS`，13/13 gates全通过；
- 11-loss×15-group synthetic ownership全exact；两个consumer `L_exec`独立，aggregate pose/total公式
  exact；
- scaled/unscaled finite range在`65536`固定scale下逐项exact；sentinel的absent/zero/finite/NaN/
  `+Inf`/`-Inf`分类exact；
- model state SHA before/after=
  `dae29e6787dd3220cfa964b4fe2a32b8760212dcf74a5cbc31b4748a84d9eb83`；
- RNG SHA before/after=
  `f4aa705a28568a8bf40be7a2d795162c855a0ed42592daa03b0e45231bb5b670`；
- CUDA initialized before/after=`false/false`，optimizer update=`0`，checkpoint=`0`；
- 两遍result/runner逐字节exact。

冻结SHA256：

- script：`d4c6d67b082e4e4f68ff215de3e7cf1f2a2ac1c4c59e17ceb265353b8810083a`；
- result/runner：`89afc893409957ee5ad356e0e2d5789683b36bcce449076d26a7dec3d3bed91c`；
- repeat result/runner：同上。

裁决：`PHASE0S_STATIC_CPU_SEALED_PASS`。只证明归因器数学与静态seam成立，不提供exp394 AMP根因，
不授权CUDA或正式训练。

## 2026-07-19 封板后环境终审

- 本地提交=`7b4541d`，提交后tracked clean；
- 远端sealed exp394 repo HEAD仍为
  `11d7a35788c4645c355d96d76a2a4ff20a9801ac`且tracked clean；
- 远端`model/tapf.py`与rich config SHA仍为
  `95c5d0ff80bf9e4529589a5f31819e7aad5db644b88e2a33d6af07c9ffc42886`/
  `e0413a497976ad6dbf4c74cf13b55c86c169d659bab6d967455e87c592e47f4e`；
- 4090=`2 MiB / 0%`，compute process=`0`；本轮未复制资产、未启动CUDA任务。

## 2026-07-19 CUDA attribution implementation静态封板

- 已实现`cuda_amp_attribution.py`，未执行；
- D0 baseline=`reid/heatmap/confidence/pose/total`五行；rich为冻结11行；
- 每行只执行fresh forward、scaled backward、scaled capture、unscale、unscaled capture和清理；源码中
  optimizer/scaler/scheduler step/update计数与调用均为0；
- 15组parameter名称覆盖在backward前检查，D0旧PSG只进入对应baseline expert bucket，rich-only组
  `not_applicable`；
- fresh exp395 regular CLIP/codebook实体名、原SHA、canonical runtime版本、唯一4090、结果路径不存在均为
  前置门；
- CPU-only AST/static contract连续两遍29/29 PASS，result/runner逐字节exact；
- 隐藏CUDA后的module import PASS，loss/group计数=`5/11/15`，CUDA仍未初始化；
- CUDA initialized before/after=`false/false`，本轮仍未复制资产或占用GPU。

冻结SHA256：

- CUDA implementation：`64840b710db587720aa8807571212b246af3eabb54306bd5aa1bbf692f5ea08b`；
- static contract：`345d26309043dd8d14119316a7ca186e1cf9faea2e666bd01d652ded50663c1b`；
- static result/runner与repeat：
  `30b7b7ae06ff2bd3153208fe4384e11e06a097608c6ce876d6c254c079f2e314`。

裁决：`CUDA_ATTRIBUTION_IMPLEMENTATION_STATIC_SEALED_PASS / CUDA EXECUTION NO-START`。没有actual
gradient归属结果，不得推测exp394根因；formal训练继续`NO-START`。

## 执行前边界

1. 没有新的明确CUDA授权前不复制远端资产、不占用4090；
2. 若获授权，先在新的exp395 execution路径复制regular CLIP/codebook并逐SHA复核；
3. actual诊断只能运行一次，必须保持zero optimizer/scaler update、checkpoint 0；
4. 进程退出后另做GPU空闲、result/runner/manifest SHA和异常终审；
5. 无论归因结果如何，exp394、formal e120与semantic multi-stage继续`NO-START`。

## 2026-07-19 implementation提交后终审

- 本地提交=`a215f72`且tracked clean；
- 远端sealed exp394 repo HEAD仍为`11d7a35788c4645c355d96d76a2a4ff20a9801ac`且tracked clean；
- 4090=`2 MiB / 0%`，compute process=`0`；
- 本轮只完成本地实现与CPU静态审计，未传输脚本、未复制fresh asset、未执行CUDA。
