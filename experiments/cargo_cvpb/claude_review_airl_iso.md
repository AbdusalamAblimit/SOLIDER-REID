# Claude Broad Review — AIRL 梯度隔离单模型双分支 (`--airl_dualbranch_iso`)

**日期**: 2026-06-23
**审查范围**: `experiments/afd_reid/afd_model.py`(SwinBackboneReID iso 分支 + AFDModel forward）、`experiments/cargo_cvpb/afd_train.py`(CLI/guard/优化器自检/训练环 consistency/eval 派发）、`experiments/cargo_cvpb/smoke_airl_iso.py`(14 项 smoke)。对照 `model/backbones/swin_transformer.py` 的 `SwinTransformer.forward` 验证 split 复刻忠实度。

## 机制定性

**不是调参 / 小 head**。这是真正的架构改动：f_rec 是在 **独立深拷贝的 Swin 末段** 上的 BNNeck，从 stage `iso_stage` 输入处 **detach() 的残差流** 分叉出来。detach 切断 f_rec 的 consistency/ID-CE 梯度回流共享 trunk，让 clean trunk + f_full 保持"干净极"、f_rec 学"recover 极"。这是对已失败的全共享 `--airl_dualbranch`(+0.06 坍缩，根因=consistency 梯度经共享 global_feat 污染 trunk)的针对性修复。

## 审查轮次

### 第 1 轮 — 发现 Critical（已修）

**C1（Critical）**: `airl_lambda_eff` 的 warmup 门控 `if (args.airl or args.airl_dualbranch) else 0.0` **漏了 `args.airl_dualbranch_iso`**。三个 AIRL flag 互斥，所以 iso 运行时该表达式恒走 `else` → `airl_lambda_eff = 0.0` 每个 epoch → consistency loss 乘 0 → **f_rec 的 recover 信号永远不训练**。整个 iso 退化成"两个纯 ID-CE 头 + eval 软融合"，degraded forward 纯属浪费算力，kill-switch 会对一个从未收到 consistency 的模型误判。smoke 的 loss-only 检查（直接 backward 原始 consistency）结构上抓不到这个 trainer 层乘子 bug；per-epoch KILL flag 也抓不到（0.0 是 finite）。

**修复**: warmup 门控改为 `(args.airl or args.airl_dualbranch or args.airl_dualbranch_iso)`。并新增 smoke I11：用 `inspect.getsource` 断言 trainer 源码 guard 含 iso flag，且复刻表达式在 epoch≥1 时 >0。

## 确认正确的部分（架构本身 sound）

- **梯度隔离真实**: rec 分支唯一入参是 `x.detach()` + `semantic_weight.detach()`；rec 内复用的 `self.swin.softplus`(无参) / `self.swin.num_features`(python list) 不漏梯度；rec_stages/rec_norm/rec_semantic_embed 均独立 deepcopy。smoke I4/I5/I9 验证 rec 有限非零梯度、trunk(早期 stage/patch_embed/共享末段) + f_full BNNeck/classifier 严格 None/0。
- **OFF 字节级一致**: iso off 不构造任何 rec 模块；forward 走原 `self.swin(x)` 单图路径。smoke I1/I1b：ON 模型加载 OFF 权重 + legacy eval == baseline f_full（max|df|=0）。
- **split f_full 图忠实**: `_forward_swin_split` 先跑完整 f_full stage 循环（patch_embed → 每 stage: stage(x,hw) → semantic_embed → norm/reshape），rec 拷贝在循环 **之后** 跑 → smoke I2(eval) max|d|=0、I13(train) DropPath RNG 也忠实。
- **iso_stage=2 形状记账正确**: 拷贝的 stage2 含自己的 PatchMerging，逐迭代更新 hw_shape，末段用 num_features[3]。smoke I12 验证隔离 + shape + eval。
- **优化器 LR 分组**: rec 末段在 `backbone_swin` 内 → 缩放 Swin-LR 组；rec BNNeck 在外 → full-LR 组。frozen rec_semantic_embed 不进优化器。smoke I6 + trainer 自检断言。
- **train/test 对称**: train(want_iso) 与 eval(return_dual) 同走 `_forward_swin_split` → 同 f_full/f_rec 特征。`airl_dualbranch_eval` 复用无改。
- **AMP/NaN 安全**: consistency 在 `autocast(enabled=False)` fp32 + `.float()` + `nan_to_num` floor。smoke I8/I10。
- **guard 完整**: iso ⊥ {airl, airl_dualbranch}、swin-only、iso_stage∈[1,3]、standalone 无 ovp/ovli、min_scale/tau/fuse_w 校验；模型层 assert 镜像。

## kill-switch（必须遵守）

训练后 fuse mean 必须比 best single head **≥ +0.7~1.0 mean mAP**，否则停。注意：该判据只在 C1 修复后才有意义（修复前 f_rec 从未收到 consistency）。

## 结论

C1 已修复并新增回归 smoke；全 14 项 smoke 通过；既有 `smoke_airl_dualbranch`(11/11) 与 `smoke_airl`(21/21) 回归通过。**审查通过**。
