# Claude Broad Review — exp333 SMPL-3D 辅助分支

**审查范围**：design.md + scripts/exp333_train_smpl3d.py（主）+ configs/occluded_duke/exp333_vit_base_smpl.yml + scripts/smpl_cache_occduke.py，并交叉核对 vit_pytorch.py / make_loss.py / make_optimizer.py / scheduler_factory.py / cosine_lr.py / metrics.py / sampler.py / occluded_duke.py / bases.py / defaults.py。

**结论：PASS（可训练）**。无 Critical、无阻断性 High。

## 逐项核对（全部 PASS）
1. **ViT 直建**：`vit_base_patch16_224_TransReID(img_size,stride_size,drop_path_rate,drop_rate,attn_drop_rate,camera=0,view=0)` 合法；**正确避开** pretrained/convert_weights/semantic_weight（那是 swin 路径，ViT 不接受 → 否则 TypeError）。`load_param(hw_ratio=1)` 签名正确，方形 14×14 pretrain → 16×8 由 resize_pos_embed 处理。
2. **backbone 返回单 tensor (B,768)**：`TransReID.forward→forward_features` 返回 `x[:,0]`，脚本按单 tensor 处理。证实绕开 `build_transformer`（它错误地解包成 tuple，ViT 路径本就坏）是对的。
3. **forward 解包**：train 返回 4 值、loop 解 4；eval 返回 2 值、解 2。✓
4. **loss_func(score,feat,target,target_cam)** 对外观头和 3D 头都用纯 tensor 调用（非 list → 走 scalar 路径）；NO_MARGIN→soft triplet，LABELSMOOTH off→F.cross_entropy。✓
5. **优化器含 3D 分支参数**：make_optimizer 遍历 named_parameters，smpl_mlp/missing/bn3d/classifier3d 全包含；bottleneck/bn3d 的 bias requires_grad=False 被正确跳过。✓
6. **scheduler.step(epoch) + _get_lr(epoch)** 与仓库 timm CosineLRScheduler 用法一致。✓
7. **AMP**：GradScaler + autocast + scale/step/update + zero_grad 顺序正确。✓
8. **R1_mAP_eval**：reset→单次 update（喂全量 concat 张量等价于多次 append）→compute；feat_norm=True 再归一。✓

## 单变量隔离（PASS）
--use_smpl off 时：smpl_dict 空、stats 恒等、baseline forward 路径不读 smpl、`if use_smpl` 块整体跳过 → 无任何 3D 参数 → 优化器/调度/seed 与 control 逐字相同，唯一差异 = 3D 分支。✓

## 无测试泄漏（PASS）
- 缓存与查表都用 basename 键；train 用 train.npz、val(query+gallery) 用各自 npz。
- z-norm 仅用**训练集 valid 样本**统计 → 无 test-stat 泄漏。
- 缺检测(valid=0)→zeros+learnable missing 向量，永不 NaN。
- train/test 的 SMPL 向量构造同一函数，无不对称。
- **BN 泄漏检查**：evaluate 顶部 model.eval()，bn3d/bottleneck 用 running stats，不会因 test-batch 组成泄漏身份。✓

## 吸收陷阱/对称（PASS）
SMPL 为预计算外部 ROMP 输出，无梯度回到像素；3D 分支只在固定输入上学 MLP → "外部非吸收输入"主张成立。

## "是否只是小 late-fusion / 伪装调参"
诚实评估：3D 分支架构上确是小 β-MLP late-fusion，机制朴素。**使其成为正当独立实验的是输入源（外部不可吸收信号），不是架构**——这正是前 9 个 NO-GO 机制都缺的性质。设计可干净给出双向可解释结论（独立 ID+triplet 监督 + alpha 扫含 alpha=0 + 重遮挡子集分解）。写作时须以"外部输入新颖性"领起，而非 MLP。

## 已修复
- **H1**（已修）：`os.environ["CUDA_VISIBLE_DEVICES"]` 移到 set_seed 前（否则 CUDA 已初始化、env 失效，多卡机上两臂可能都落 GPU0）。

## 非阻断建议（不改）
- M1：cudnn.benchmark=True 与 deterministic=True 并存（矛盾）。保持与仓库 train.py 一致；两臂同设置，A/B 公平，determinism 视为 best-effort。
- M2/M3/L1-L3：均非真实风险（AMP 三元组与仓库一致；全 batch 缺检测概率天文级小；末轮双判 or 短路单 eval；train valid 93%）。

**审查通过**：脚本忠实复用仓库精确机制，A/B 真单变量，无测试/ BN 泄漏，所有 tricky 契约（ViT 直建 kwargs、load_param、单 tensor 返回、4/2 值解包、loss 签名、优化器覆盖、scheduler、AMP、评测）经源码核对全部成立。must-fix = 无（H1 已修）。
