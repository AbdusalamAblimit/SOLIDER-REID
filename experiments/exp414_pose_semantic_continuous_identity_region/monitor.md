# exp414 PSCIR 监控记录

## 2026-07-23：切换与设计冻结前状态

- 用户明确要求停止exp413余下controls并立即进入下一创新点。
- exp413已永久封板为`PERFORMANCE GO / JOINT ATTRIBUTION FAILED`；q-only为
  `USER-DIRECTED STOP / E120 VOID / NO RESUME`，text-shuffle为`NO-START BY USER PRIORITY`。
- 远端核验无训练/CUDA compute，GPU=`2 MiB / 0%`；exp413 formal tracked worktree/index=`0/0`。
- 本地分支=`codex/solider-official-tapf-clean`，切换时HEAD=`34d241e8`；唯一未跟踪项为受保护用户文件
  `experiments/exp411_pose_complete_multi_positive_set_ranking/创新性判断.md`，保持不动。

## 历史去重结论

已核对exp148 PCVT、exp109--142 feature completion、exp356 PC-MSC、exp361 PSC-JEPA、exp371 CASD、
exp405 CAVT及exp408--413：

1. “pose定位缺失槽 + CLIP定义语义 + 同PID donor/token补一个目标点”已被多条旧路线覆盖；
2. PSC-JEPA明确暴露不可观测身份细节的point prediction会推动表征平滑并伤害ReID；
3. 因此exp414不得做feature transport、point KD、prefix排序、owner加权或gradient routing。

当前新对象为PSCIR：真实student support端点之间的连续identity region；pose×identity-free CLIP只决定合法
连边拓扑。状态=`DESIGN WRITTEN / C-CLASS CONDITIONAL / INDEPENDENT REVIEW NEXT / GPU IDLE`。

## 查新边界

- arXiv精确查询`person re-identification + set-valued + occlusion`无直接结果；
- GitHub窄检索`person re-identification + box/set embedding + occlusion`无直接仓库；
- 检出的P2LR为UDA概率伪标签精炼，不是pose×CLIP LOO连续identity region；
- 但概率embedding、image-set convex hull、MVI²P与set metric均为强近邻，不能据此声称首次。

下一步只做独立设计盲审；盲审仅拦截致命bug、变量混淆或旧机制同构。`0B/0H`后才实现，GPU继续空闲。

## 2026-07-23：独立子agent设计盲审PASS

Claude CLI未认证后，用户明确要求停止使用Claude并改由独立子agent完成盲审。审查者只读核对设计与历史路线，
未编辑文件、未连接远端或使用GPU；结论=`PASS / 0B / 0H`。

非阻塞NOTE已冻结进实现边界：合同中的未选第三边扰动必须固定MST索引，不能伪称可独立修改共享端点；`q_only`
只表述为无在线pose visibility轴；归一化descriptor上的线段解释为欧氏弦。状态更新为
`DESIGN REVIEW PASS / IMPLEMENTATION ACTIVE / GPU IDLE`。

## 2026-07-23：实现完成与独立实现盲审

已完成默认关闭的PSCIR配置、连续region loss、zero-owner等权组合、processor资产/状态/日志接线、formal config与唯一
PK64 runner。正式config关闭PSCIR并清空其四个配置字段后，与sealed exp411 zero-owner config dump exact；本地
`py_compile`、config exact与CPU几何oracle通过。本地轻量环境未安装`torchvision`，因此完整runner只允许在固定
MMPOSE-ABU远端解释器中执行，不创建第二合同。

实现盲审首轮发现`0B/3H`：近零距离被epsilon抬高、上端clamp漏测、combined梯度差不足以证明纯region路径。
现已分别改为精确vector norm并加入线上零距离反向、加入`t>1`上端oracle、对纯region loss独立backward并检查
Stage-3/norm3 finite nonzero。复审=`PASS / 0B / 0H`。

当前状态=`IMPLEMENTED / STATIC CHECK PASS / IMPLEMENTATION REVIEW PASS / UNIQUE PK64 CONTRACT NEXT /
GPU IDLE`。尚无exp414合同或性能数字。

## 2026-07-23：唯一runner首次启动在CUDA前micro退出

- formal首冻HEAD=`f0bc9fec5feeca0819040f26f2d838b73e6be764`；
- runner SHA256=`b1f0e807ba3691cf0d891952d7ab0bd0cf511e1a7051ff584a126fc933c0e34d`；
- runner在`micro_oracle()`覆盖检查处退出，错误为远端旧版PyTorch不支持`Tensor.any(dim=(2,3))`；
- 退出发生在dataloader、model与`torch.device("cuda",0)`执行前，GPU始终=`2 MiB / 0%`，因此没有真实PK64
  batch、CUDA forward/backward或GradScaler update，分类=`PRE-CUDA MICRO FAIL / REAL PK64 NOT CONSUMED`。

按合同规则只修致命兼容bug：等价改为连续两次`.any(dim=3).any(dim=2)`，不改公式、topology、control、loss或
合同门。修复后更新formal HEAD与runner SHA，再启动唯一真实PK64；不把本次前置micro失败计作CUDA合同结果。

## 2026-07-23：兼容修复后首次调用缺少必填config参数

- 兼容修复已在远端formal提交为`4739285d40c514ae41ab0f16d292af135d4dd51a`，固定解释器静态编译通过；
- 新日志`exp414-pscir-contract-v1-attempt2.runner.log`因调用命令遗漏runner必填的`--config`参数，在
  `argparse`阶段立即以`CONTRACT_EXIT=2`退出；
- GPU全程=`2 MiB / 0%`，没有执行micro oracle、dataloader、真实PK64 batch、CUDA forward/backward或
  GradScaler update；
- 分类=`PRE-ARGPARSE INVOCATION FAIL / REAL PK64 NOT CONSUMED`。保留该日志，不覆盖、不伪装成合同结果；
  下一次只补入预注册formal config路径，不改runner、机制、合同门或任何训练变量。

## 2026-07-23：唯一真实PK64合同PASS并自然退出

- 合同执行formal HEAD=`c6739b402a9e9a16f2427324536251e5ed059598`，仅在attempt2命令中补入预注册
  `configs/occluded_duke/swin_tiny_tapf_pscir_exp414.yml`，runner与机制均未修改；
- 真实batch=`64`、身份数=`16`、每身份实例数=`4`，状态=`PASS`，`CONTRACT_EXIT=0`；
- 四个control均真实改变topology与region distance：
  `pose_only=0.375/0.375`、`q_only=0.390625/0.390625`、
  `text_shuffle=0.359375/0.359375`、`all_edges=1.0/1.0`；
- strict MST两边覆盖三support、excluded-image mutation invariant、unused-candidate-record invariant、
  default-off forward/loss/gradient/RNG exact与zero-owner loss/distance exact均为`true`；
- isolated region loss=`1.9516464472`，纯region独立反向使Stage-3/norm3 `28/28`个可比梯度tensor
  finite nonzero；combined路径有`26`个Stage-3非零梯度tensor；
- 原生GradScaler前4次overflow退火，第5次真实更新
  `base.stages.3.blocks.0.ffn.layers.1.weight`，scale=`4096→4096`；
- runner/test/config/region-loss SHA256分别为
  `88fd7c858c7f4fc2a5d0ef4bc5afc5b2e54f969b8904c724ac74bbb5d0a4ba17`/
  `b1f0e807ba3691cf0d891952d7ab0bd0cf511e1a7051ff584a126fc933c0e34d`/
  `f887222e371642556009f99a5e1d165e5ae10fea9a97c773a2b4486e032c7b8b`/
  `05562df2e79e45d5a95cf6a2760ca4d86803173e748e5228e1599aa11106e85a`；
- runner严格异常=`0`，进程自然退出后GPU=`2 MiB / 0%`且CUDA compute=`0`，formal tracked
  worktree/index=`0/0`；运行生成的未跟踪pyc保持不动。

结论：唯一真实PK64合同已消费且`PASS`，不得重跑或追加preflight。冻结上述formal HEAD，下一步从fresh
OUTPUT_DIR启动correct正式e120；只有correct性能GO后才串行启动matched controls。

## 2026-07-23：correct正式训练fresh启动

- frozen formal=`/home/afr/SOLIDER-REID-exp414-pscir-formal-v1`，HEAD=
  `c6739b402a9e9a16f2427324536251e5ed059598`，tracked worktree/index=`0/0`；
- output=`/home/afr/reid-clean/logs/exp414-pscir-s1234-v1`，runner=
  `/home/afr/reid-clean/train-logs/exp414-pscir-s1234-v1.runner.log`，启动前二者均不存在；
- 固定解释器、Swin-Tiny、batch64、16×4、seed1234、correct模式、fresh启动且无checkpoint恢复；
- wrapper主PID=`108593`，训练主PID=`108598`；8个同命令子进程均为DataLoader worker，唯一CUDA compute为
  训练主PID，首检GPU约=`6992 MiB / 73%`；
- 首批zero-owner loss=`1.970790`；PSCIR
  `loss/zero/region/positive/negative/segment=2.313745/1.970790/2.656701/47.542694/57.520821/58.427376`，
  edge-weight=`[8.890625,8.796875,9.0]`；
- 首批control topology-change=`pose-only 0.390625 / q-only 0.375 / text-shuffle 0.328125 /
  all-edges 1.0`，证明正式路径读取correct状态且controls非同构；
- heartbeat复核最新=`e2 iter160/227`，loss=`6.979`；wrapper/训练主PID存活，唯一CUDA compute约
  `6994 MiB / 42%`，runner/train严格异常=`0`，formal tracked worktree/index=`0/0`。当前判断=
  `CONTINUE TO NATURAL E120`，不因中间点早停。

## 2026-07-23：e7完成 / e8启动健康检查

- e7自然完成，time=`169.978s`、speed=`79.4 samples/s`；最新为e8首批；
- e8首批PSCIR `loss/zero/region=0.074353/0.053594/0.095112`，
  `positive/negative/segment=23.412945/42.908546/42.031464`；
- wrapper/训练主PID存活，唯一CUDA compute约`6994 MiB / 43%`，runner/train严格异常=`0`，formal
  HEAD保持`c6739b402a9e9a16f2427324536251e5ed059598`且tracked worktree/index=`0/0`；
- 尚未到首个正式评测点，不作性能判断。当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：correct e10首个正式评测

correct自然完成e10=`28.6 mAP / 38.1 R1 / 53.7 R5 / 60.7 R10`；同epoch sealed zero-owner=
`28.4/38.1/53.9/60.8`，sealed clean D0=`33.4/42.7/59.8/65.2`。rounded四项差为：

| epoch | PSCIR correct | zero-owner | clean D0 | Δzero-owner | ΔD0 |
|---|---|---|---|---|---|
| 10 | 28.6/38.1/53.7/60.7 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | +0.2/+0.0/-0.2/-0.1 | -4.8/-4.6/-6.1/-4.5 |

e10仅mAP微高宿主、R1持平，R5/R10微低，且四项低于D0；这是早期正式点，不能据此作e120性能或机制裁决。
最新已进入e13，wrapper/训练主PID存活，唯一CUDA compute约`7064 MiB / 42%`，runner/train严格异常=`0`，
formal tracked worktree/index=`0/0`。当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：e18健康检查

- e17自然完成，最新=`e18 iter80/227`，loss=`2.470`、pose=`0.578`、acc=`0.654`；
- e18首批PSCIR `loss/zero/region=0.021884/0.020508/0.023261`，正负距离保持分离；
- wrapper/训练主PID存活，唯一CUDA compute约`7066 MiB / 43%`，runner/train严格异常=`0`，formal
  tracked worktree/index=`0/0`；
- 尚未到e20正式点，当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：correct e20正式评测

correct自然完成e20=`47.1 mAP / 56.7 R1 / 71.4 R5 / 76.9 R10`；同epoch sealed zero-owner=
`45.6/55.0/70.6/75.8`，sealed clean D0=`42.2/52.4/67.6/74.0`。完整rounded轨迹更新为：

| epoch | PSCIR correct | zero-owner | clean D0 | Δzero-owner | ΔD0 |
|---|---|---|---|---|---|
| 10 | 28.6/38.1/53.7/60.7 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | +0.2/+0.0/-0.2/-0.1 | -4.8/-4.6/-6.1/-4.5 |
| 20 | 47.1/56.7/71.4/76.9 | 45.6/55.0/70.6/75.8 | 42.2/52.4/67.6/74.0 | +1.5/+1.7/+0.8/+1.1 | +4.9/+4.3/+3.8/+2.9 |

e20相对两个关键对照均四项领先，扭转e10的混合/落后关系；该点是积极中间证据，但不能提前确认性能GO或机制
归因。读取时最新=`e23 iter120/227`，wrapper/训练主PID存活，唯一CUDA compute约`7074 MiB / 58%`，
runner/train严格异常=`0`，formal tracked worktree/index=`0/0`。当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：e28健康检查

- 最新=`e28 iter180/227`，loss=`0.587`、pose=`0.490`、acc=`0.962`；
- wrapper/训练主PID存活，唯一CUDA compute约`7074 MiB / 62%`，runner/train严格异常=`0`，formal
  tracked worktree/index=`0/0`；
- 尚未到e30正式点，不作新增性能判断。当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：correct e30正式评测

correct自然完成e30=`50.5 mAP / 62.0 R1 / 74.9 R5 / 79.9 R10`；同epoch sealed zero-owner=
`49.2/60.3/75.0/80.0`，sealed clean D0=`46.6/56.2/71.3/76.4`。完整rounded轨迹更新为：

| epoch | PSCIR correct | zero-owner | clean D0 | Δzero-owner | ΔD0 |
|---|---|---|---|---|---|
| 10 | 28.6/38.1/53.7/60.7 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | +0.2/+0.0/-0.2/-0.1 | -4.8/-4.6/-6.1/-4.5 |
| 20 | 47.1/56.7/71.4/76.9 | 45.6/55.0/70.6/75.8 | 42.2/52.4/67.6/74.0 | +1.5/+1.7/+0.8/+1.1 | +4.9/+4.3/+3.8/+2.9 |
| 30 | 50.5/62.0/74.9/79.9 | 49.2/60.3/75.0/80.0 | 46.6/56.2/71.3/76.4 | +1.3/+1.7/-0.1/-0.1 | +3.9/+5.8/+3.6/+3.5 |

e30相对zero-owner继续保持预注册核心mAP/R1优势，但R5/R10各低`0.1`；相对D0仍四项领先。完整保留混合
结果，不把核心门扩写成四项全胜。读取时最新=`e33 iter200/227`，wrapper/训练主PID存活，唯一CUDA compute约
`7078 MiB / 45%`，runner/train严格异常=`0`，formal tracked worktree/index=`0/0`。当前=
`CONTINUE TO NATURAL E120`。

## 2026-07-23：e39健康检查

- e38自然完成，最新=`e39 iter60/227`，loss=`0.302`、pose=`0.469`、acc=`0.986`；
- e39首批PSCIR `loss/zero/region=0.000002/0.000003/0.000001`，均为有限非负值，严格异常正则未命中；
- wrapper/训练主PID存活，唯一CUDA compute约`7078 MiB / 42%`，formal tracked worktree/index=`0/0`；
- 尚未到e40正式点，当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：correct e40正式评测

correct自然完成e40=`55.2 mAP / 67.1 R1 / 79.9 R5 / 84.5 R10`；同epoch sealed zero-owner=
`55.0/66.2/79.8/84.4`，sealed clean D0=`50.0/60.7/76.2/81.0`。完整rounded轨迹更新为：

| epoch | PSCIR correct | zero-owner | clean D0 | Δzero-owner | ΔD0 |
|---|---|---|---|---|---|
| 10 | 28.6/38.1/53.7/60.7 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | +0.2/+0.0/-0.2/-0.1 | -4.8/-4.6/-6.1/-4.5 |
| 20 | 47.1/56.7/71.4/76.9 | 45.6/55.0/70.6/75.8 | 42.2/52.4/67.6/74.0 | +1.5/+1.7/+0.8/+1.1 | +4.9/+4.3/+3.8/+2.9 |
| 30 | 50.5/62.0/74.9/79.9 | 49.2/60.3/75.0/80.0 | 46.6/56.2/71.3/76.4 | +1.3/+1.7/-0.1/-0.1 | +3.9/+5.8/+3.6/+3.5 |
| 40 | 55.2/67.1/79.9/84.5 | 55.0/66.2/79.8/84.4 | 50.0/60.7/76.2/81.0 | +0.2/+0.9/+0.1/+0.1 | +5.2/+6.4/+3.7/+3.5 |

e40相对zero-owner恢复四项微幅领先，相对D0保持清晰四项优势；宿主差值较小，不能夸大稳定性。读取时最新=
`e44 iter80/227`，wrapper/训练主PID存活，唯一CUDA compute约`7062 MiB / 66%`，runner/train严格异常=`0`，
formal tracked worktree/index=`0/0`。当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：e49健康检查

- 最新=`e49 iter120/227`，loss=`0.191`、pose=`0.466`、acc=`0.994`；
- wrapper/训练主PID存活，唯一CUDA compute约`7062 MiB / 73%`，runner/train严格异常=`0`，formal
  tracked worktree/index=`0/0`；
- 尚未到e50正式点，当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：correct e50正式评测

correct自然完成e50=`55.9 mAP / 67.9 R1 / 80.1 R5 / 84.8 R10`；同epoch sealed zero-owner=
`55.1/66.1/80.0/83.7`，sealed clean D0=`52.1/62.8/77.0/81.9`。完整rounded轨迹更新为：

| epoch | PSCIR correct | zero-owner | clean D0 | Δzero-owner | ΔD0 |
|---|---|---|---|---|---|
| 10 | 28.6/38.1/53.7/60.7 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | +0.2/+0.0/-0.2/-0.1 | -4.8/-4.6/-6.1/-4.5 |
| 20 | 47.1/56.7/71.4/76.9 | 45.6/55.0/70.6/75.8 | 42.2/52.4/67.6/74.0 | +1.5/+1.7/+0.8/+1.1 | +4.9/+4.3/+3.8/+2.9 |
| 30 | 50.5/62.0/74.9/79.9 | 49.2/60.3/75.0/80.0 | 46.6/56.2/71.3/76.4 | +1.3/+1.7/-0.1/-0.1 | +3.9/+5.8/+3.6/+3.5 |
| 40 | 55.2/67.1/79.9/84.5 | 55.0/66.2/79.8/84.4 | 50.0/60.7/76.2/81.0 | +0.2/+0.9/+0.1/+0.1 | +5.2/+6.4/+3.7/+3.5 |
| 50 | 55.9/67.9/80.1/84.8 | 55.1/66.1/80.0/83.7 | 52.1/62.8/77.0/81.9 | +0.8/+1.8/+0.1/+1.1 | +3.8/+5.1/+3.1/+2.9 |

e50继续对两个关键对照四项领先；相对宿主的R5仅`+0.1`，仍不能夸大为稳定宽裕优势。读取时最新=
`e54 iter160/227`，wrapper/训练主PID存活，唯一CUDA compute约`7080 MiB / 73%`，runner/train严格异常=`0`，
formal tracked worktree/index=`0/0`。当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：e60启动健康检查

- e59自然完成，最新为e60首批；PSCIR
  `loss/zero/region=0.000028/0.000039/0.000016`，均有限；
- wrapper/训练主PID存活，唯一CUDA compute约`7080 MiB / 42%`，runner/train严格异常=`0`，formal
  tracked worktree/index=`0/0`；
- e60评测尚未产生，当前=`CONTINUE / WAIT E60 EVAL / NATURAL E120`。

## 2026-07-23：correct e60正式评测

correct自然完成e60=`57.4 mAP / 69.6 R1 / 81.6 R5 / 85.7 R10`；同epoch sealed zero-owner=
`57.6/70.3/81.0/85.2`，sealed clean D0=`55.1/66.1/79.0/83.3`。完整rounded轨迹更新为：

| epoch | PSCIR correct | zero-owner | clean D0 | Δzero-owner | ΔD0 |
|---|---|---|---|---|---|
| 10 | 28.6/38.1/53.7/60.7 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | +0.2/+0.0/-0.2/-0.1 | -4.8/-4.6/-6.1/-4.5 |
| 20 | 47.1/56.7/71.4/76.9 | 45.6/55.0/70.6/75.8 | 42.2/52.4/67.6/74.0 | +1.5/+1.7/+0.8/+1.1 | +4.9/+4.3/+3.8/+2.9 |
| 30 | 50.5/62.0/74.9/79.9 | 49.2/60.3/75.0/80.0 | 46.6/56.2/71.3/76.4 | +1.3/+1.7/-0.1/-0.1 | +3.9/+5.8/+3.6/+3.5 |
| 40 | 55.2/67.1/79.9/84.5 | 55.0/66.2/79.8/84.4 | 50.0/60.7/76.2/81.0 | +0.2/+0.9/+0.1/+0.1 | +5.2/+6.4/+3.7/+3.5 |
| 50 | 55.9/67.9/80.1/84.8 | 55.1/66.1/80.0/83.7 | 52.1/62.8/77.0/81.9 | +0.8/+1.8/+0.1/+1.1 | +3.8/+5.1/+3.1/+2.9 |
| 60 | 57.4/69.6/81.6/85.7 | 57.6/70.3/81.0/85.2 | 55.1/66.1/79.0/83.3 | -0.2/-0.7/+0.6/+0.5 | +2.3/+3.5/+2.6/+2.4 |

e60相对zero-owner的预注册核心mAP/R1转负，R5/R10仍为正；相对D0四项继续领先。这打断了e20--e50的
核心正差轨迹，必须完整保留，不能用高阶CMC掩盖核心门风险。读取时最新=`e62 iter100/227`，wrapper/训练主PID
存活，唯一CUDA compute约`7062 MiB / 73%`，runner/train严格异常=`0`，formal tracked worktree/index=`0/0`。
当前=`CONTINUE TO NATURAL E120 / NO EARLY STOP`。

## 2026-07-23：e67健康检查

- 最新=`e67 iter160/227`，loss=`0.132`、pose=`0.462`、acc=`0.996`；
- wrapper/训练主PID存活，唯一CUDA compute约`7062 MiB / 41%`，runner/train严格异常=`0`，formal
  tracked worktree/index=`0/0`；
- 尚未到e70正式点，当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：correct e70正式评测

correct自然完成e70=`57.9 mAP / 70.0 R1 / 82.1 R5 / 86.0 R10`；同epoch sealed zero-owner=
`57.8/70.2/81.7/85.5`，sealed clean D0=`55.4/65.2/79.5/83.6`。完整rounded轨迹更新为：

| epoch | PSCIR correct | zero-owner | clean D0 | Δzero-owner | ΔD0 |
|---|---|---|---|---|---|
| 10 | 28.6/38.1/53.7/60.7 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | +0.2/+0.0/-0.2/-0.1 | -4.8/-4.6/-6.1/-4.5 |
| 20 | 47.1/56.7/71.4/76.9 | 45.6/55.0/70.6/75.8 | 42.2/52.4/67.6/74.0 | +1.5/+1.7/+0.8/+1.1 | +4.9/+4.3/+3.8/+2.9 |
| 30 | 50.5/62.0/74.9/79.9 | 49.2/60.3/75.0/80.0 | 46.6/56.2/71.3/76.4 | +1.3/+1.7/-0.1/-0.1 | +3.9/+5.8/+3.6/+3.5 |
| 40 | 55.2/67.1/79.9/84.5 | 55.0/66.2/79.8/84.4 | 50.0/60.7/76.2/81.0 | +0.2/+0.9/+0.1/+0.1 | +5.2/+6.4/+3.7/+3.5 |
| 50 | 55.9/67.9/80.1/84.8 | 55.1/66.1/80.0/83.7 | 52.1/62.8/77.0/81.9 | +0.8/+1.8/+0.1/+1.1 | +3.8/+5.1/+3.1/+2.9 |
| 60 | 57.4/69.6/81.6/85.7 | 57.6/70.3/81.0/85.2 | 55.1/66.1/79.0/83.3 | -0.2/-0.7/+0.6/+0.5 | +2.3/+3.5/+2.6/+2.4 |
| 70 | 57.9/70.0/82.1/86.0 | 57.8/70.2/81.7/85.5 | 55.4/65.2/79.5/83.6 | +0.1/-0.2/+0.4/+0.5 | +2.5/+4.8/+2.6/+2.4 |

e70相对zero-owner的mAP微正、R1仍负，R5/R10为正；相对D0四项持续领先。核心mAP/R1门仍未同时恢复，
不能用高阶CMC优势替代。读取时最新=`e72 iter180/227`，wrapper/训练主PID存活，唯一CUDA compute约
`7078 MiB / 73%`，runner/train严格异常=`0`，formal tracked worktree/index=`0/0`。当前=
`CONTINUE TO NATURAL E120`。

## 2026-07-23：e78健康检查

- e77自然完成，最新=`e78 iter40/227`，loss=`0.125`、pose=`0.462`、acc=`0.997`；
- e78首批zero/region均打印为`0.000000`，但epoch累计loss有限且严格异常正则未命中；
- wrapper/训练主PID存活，唯一CUDA compute约`7078 MiB / 42%`，formal tracked worktree/index=`0/0`；
- 尚未到e80正式点，当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：correct e80正式评测

首次SSH读取无输出并以255退出；立即重连后确认主PID、CUDA训练与日志时间连续，分类为瞬时SSH连接失败，
不是训练或基础设施中断。

correct自然完成e80=`58.9 mAP / 71.1 R1 / 82.5 R5 / 86.0 R10`；同epoch sealed zero-owner=
`58.6/71.6/82.4/86.3`，sealed clean D0=`56.1/66.3/79.5/84.0`。完整rounded轨迹更新为：

| epoch | PSCIR correct | zero-owner | clean D0 | Δzero-owner | ΔD0 |
|---|---|---|---|---|---|
| 10 | 28.6/38.1/53.7/60.7 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | +0.2/+0.0/-0.2/-0.1 | -4.8/-4.6/-6.1/-4.5 |
| 20 | 47.1/56.7/71.4/76.9 | 45.6/55.0/70.6/75.8 | 42.2/52.4/67.6/74.0 | +1.5/+1.7/+0.8/+1.1 | +4.9/+4.3/+3.8/+2.9 |
| 30 | 50.5/62.0/74.9/79.9 | 49.2/60.3/75.0/80.0 | 46.6/56.2/71.3/76.4 | +1.3/+1.7/-0.1/-0.1 | +3.9/+5.8/+3.6/+3.5 |
| 40 | 55.2/67.1/79.9/84.5 | 55.0/66.2/79.8/84.4 | 50.0/60.7/76.2/81.0 | +0.2/+0.9/+0.1/+0.1 | +5.2/+6.4/+3.7/+3.5 |
| 50 | 55.9/67.9/80.1/84.8 | 55.1/66.1/80.0/83.7 | 52.1/62.8/77.0/81.9 | +0.8/+1.8/+0.1/+1.1 | +3.8/+5.1/+3.1/+2.9 |
| 60 | 57.4/69.6/81.6/85.7 | 57.6/70.3/81.0/85.2 | 55.1/66.1/79.0/83.3 | -0.2/-0.7/+0.6/+0.5 | +2.3/+3.5/+2.6/+2.4 |
| 70 | 57.9/70.0/82.1/86.0 | 57.8/70.2/81.7/85.5 | 55.4/65.2/79.5/83.6 | +0.1/-0.2/+0.4/+0.5 | +2.5/+4.8/+2.6/+2.4 |
| 80 | 58.9/71.1/82.5/86.0 | 58.6/71.6/82.4/86.3 | 56.1/66.3/79.5/84.0 | +0.3/-0.5/+0.1/-0.3 | +2.8/+4.8/+3.0/+2.0 |

e80相对zero-owner仅mAP/R5微正，R1/R10为负；相对D0四项仍为正。预注册核心mAP/R1连续e70/e80未同时
成立，性能GO风险持续。读取时最新=`e83 iter100/227`，wrapper/训练主PID存活，唯一CUDA compute约
`7050 MiB / 42%`，runner/train严格异常=`0`，formal tracked worktree/index=`0/0`。当前=
`CONTINUE TO NATURAL E120 / NO EARLY STOP`。

## 2026-07-23：e88健康检查

- 最新=`e88 iter140/227`，loss=`0.114`、pose=`0.462`、acc=`0.997`；
- wrapper/训练主PID存活，唯一CUDA compute约`7050 MiB / 42%`，runner/train严格异常=`0`，formal
  tracked worktree/index=`0/0`；
- 尚未到e90正式点，当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：correct e90正式评测

correct自然完成e90=`59.4 mAP / 71.4 R1 / 83.0 R5 / 86.9 R10`；同epoch sealed zero-owner=
`59.1/71.2/82.6/86.8`，sealed clean D0=`57.5/67.9/81.2/85.3`。完整rounded轨迹更新为：

| epoch | PSCIR correct | zero-owner | clean D0 | Δzero-owner | ΔD0 |
|---|---|---|---|---|---|
| 10 | 28.6/38.1/53.7/60.7 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | +0.2/+0.0/-0.2/-0.1 | -4.8/-4.6/-6.1/-4.5 |
| 20 | 47.1/56.7/71.4/76.9 | 45.6/55.0/70.6/75.8 | 42.2/52.4/67.6/74.0 | +1.5/+1.7/+0.8/+1.1 | +4.9/+4.3/+3.8/+2.9 |
| 30 | 50.5/62.0/74.9/79.9 | 49.2/60.3/75.0/80.0 | 46.6/56.2/71.3/76.4 | +1.3/+1.7/-0.1/-0.1 | +3.9/+5.8/+3.6/+3.5 |
| 40 | 55.2/67.1/79.9/84.5 | 55.0/66.2/79.8/84.4 | 50.0/60.7/76.2/81.0 | +0.2/+0.9/+0.1/+0.1 | +5.2/+6.4/+3.7/+3.5 |
| 50 | 55.9/67.9/80.1/84.8 | 55.1/66.1/80.0/83.7 | 52.1/62.8/77.0/81.9 | +0.8/+1.8/+0.1/+1.1 | +3.8/+5.1/+3.1/+2.9 |
| 60 | 57.4/69.6/81.6/85.7 | 57.6/70.3/81.0/85.2 | 55.1/66.1/79.0/83.3 | -0.2/-0.7/+0.6/+0.5 | +2.3/+3.5/+2.6/+2.4 |
| 70 | 57.9/70.0/82.1/86.0 | 57.8/70.2/81.7/85.5 | 55.4/65.2/79.5/83.6 | +0.1/-0.2/+0.4/+0.5 | +2.5/+4.8/+2.6/+2.4 |
| 80 | 58.9/71.1/82.5/86.0 | 58.6/71.6/82.4/86.3 | 56.1/66.3/79.5/84.0 | +0.3/-0.5/+0.1/-0.3 | +2.8/+4.8/+3.0/+2.0 |
| 90 | 59.4/71.4/83.0/86.9 | 59.1/71.2/82.6/86.8 | 57.5/67.9/81.2/85.3 | +0.3/+0.2/+0.4/+0.1 | +1.9/+3.5/+1.8/+1.6 |

e90相对两个关键对照重新四项为正，核心mAP/R1门在当前点恢复，但相对宿主仅`+0.3/+0.2`，不能提前GO。
读取时最新=`e93 iter180/227`，wrapper/训练主PID存活，唯一CUDA compute约`7074 MiB / 72%`，runner/train
严格异常=`0`，formal tracked worktree/index=`0/0`。当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：e99健康检查

- e98自然完成，最新=`e99 iter80/227`，loss=`0.113`、pose=`0.459`、acc=`0.997`；
- wrapper/训练主PID存活，唯一CUDA compute约`7074 MiB / 42%`，runner/train严格异常=`0`，formal
  tracked worktree/index=`0/0`；
- 尚未到e100正式点，当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：correct e100正式评测

correct自然完成e100=`59.0 mAP / 70.4 R1 / 82.4 R5 / 86.3 R10`；同epoch sealed zero-owner=
`58.8/70.5/82.2/86.1`，sealed clean D0=`56.9/67.1/79.6/83.8`。完整rounded轨迹更新为：

| epoch | PSCIR correct | zero-owner | clean D0 | Δzero-owner | ΔD0 |
|---|---|---|---|---|---|
| 10 | 28.6/38.1/53.7/60.7 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | +0.2/+0.0/-0.2/-0.1 | -4.8/-4.6/-6.1/-4.5 |
| 20 | 47.1/56.7/71.4/76.9 | 45.6/55.0/70.6/75.8 | 42.2/52.4/67.6/74.0 | +1.5/+1.7/+0.8/+1.1 | +4.9/+4.3/+3.8/+2.9 |
| 30 | 50.5/62.0/74.9/79.9 | 49.2/60.3/75.0/80.0 | 46.6/56.2/71.3/76.4 | +1.3/+1.7/-0.1/-0.1 | +3.9/+5.8/+3.6/+3.5 |
| 40 | 55.2/67.1/79.9/84.5 | 55.0/66.2/79.8/84.4 | 50.0/60.7/76.2/81.0 | +0.2/+0.9/+0.1/+0.1 | +5.2/+6.4/+3.7/+3.5 |
| 50 | 55.9/67.9/80.1/84.8 | 55.1/66.1/80.0/83.7 | 52.1/62.8/77.0/81.9 | +0.8/+1.8/+0.1/+1.1 | +3.8/+5.1/+3.1/+2.9 |
| 60 | 57.4/69.6/81.6/85.7 | 57.6/70.3/81.0/85.2 | 55.1/66.1/79.0/83.3 | -0.2/-0.7/+0.6/+0.5 | +2.3/+3.5/+2.6/+2.4 |
| 70 | 57.9/70.0/82.1/86.0 | 57.8/70.2/81.7/85.5 | 55.4/65.2/79.5/83.6 | +0.1/-0.2/+0.4/+0.5 | +2.5/+4.8/+2.6/+2.4 |
| 80 | 58.9/71.1/82.5/86.0 | 58.6/71.6/82.4/86.3 | 56.1/66.3/79.5/84.0 | +0.3/-0.5/+0.1/-0.3 | +2.8/+4.8/+3.0/+2.0 |
| 90 | 59.4/71.4/83.0/86.9 | 59.1/71.2/82.6/86.8 | 57.5/67.9/81.2/85.3 | +0.3/+0.2/+0.4/+0.1 | +1.9/+3.5/+1.8/+1.6 |
| 100 | 59.0/70.4/82.4/86.3 | 58.8/70.5/82.2/86.1 | 56.9/67.1/79.6/83.8 | +0.2/-0.1/+0.2/+0.2 | +2.1/+3.3/+2.8/+2.5 |

e100相对zero-owner的mAP/R5/R10为正、R1微负`0.1`；相对D0四项继续为正。e90的核心双正未在e100
连续保持，最终性能门仍高度边缘。读取时最新=`e104 iter100/227`，wrapper/训练主PID存活，唯一CUDA compute约
`7072 MiB / 42%`，runner/train严格异常=`0`，formal tracked worktree/index=`0/0`。当前=
`CONTINUE TO NATURAL E120`。

## 2026-07-23：e109健康检查

- 最新=`e109 iter160/227`，loss=`0.105`、pose=`0.460`、acc=`0.998`；
- wrapper/训练主PID存活，唯一CUDA compute约`7072 MiB / 71%`，runner/train严格异常=`0`，formal
  tracked worktree/index=`0/0`；
- 尚未到e110正式点，当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：correct e110正式评测

correct自然完成e110=`59.1 mAP / 70.6 R1 / 82.6 R5 / 86.2 R10`；同epoch sealed zero-owner=
`58.8/70.4/81.8/86.1`，sealed clean D0=`57.4/67.4/80.5/84.6`。完整rounded轨迹更新为：

| epoch | PSCIR correct | zero-owner | clean D0 | Δzero-owner | ΔD0 |
|---|---|---|---|---|---|
| 10 | 28.6/38.1/53.7/60.7 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | +0.2/+0.0/-0.2/-0.1 | -4.8/-4.6/-6.1/-4.5 |
| 20 | 47.1/56.7/71.4/76.9 | 45.6/55.0/70.6/75.8 | 42.2/52.4/67.6/74.0 | +1.5/+1.7/+0.8/+1.1 | +4.9/+4.3/+3.8/+2.9 |
| 30 | 50.5/62.0/74.9/79.9 | 49.2/60.3/75.0/80.0 | 46.6/56.2/71.3/76.4 | +1.3/+1.7/-0.1/-0.1 | +3.9/+5.8/+3.6/+3.5 |
| 40 | 55.2/67.1/79.9/84.5 | 55.0/66.2/79.8/84.4 | 50.0/60.7/76.2/81.0 | +0.2/+0.9/+0.1/+0.1 | +5.2/+6.4/+3.7/+3.5 |
| 50 | 55.9/67.9/80.1/84.8 | 55.1/66.1/80.0/83.7 | 52.1/62.8/77.0/81.9 | +0.8/+1.8/+0.1/+1.1 | +3.8/+5.1/+3.1/+2.9 |
| 60 | 57.4/69.6/81.6/85.7 | 57.6/70.3/81.0/85.2 | 55.1/66.1/79.0/83.3 | -0.2/-0.7/+0.6/+0.5 | +2.3/+3.5/+2.6/+2.4 |
| 70 | 57.9/70.0/82.1/86.0 | 57.8/70.2/81.7/85.5 | 55.4/65.2/79.5/83.6 | +0.1/-0.2/+0.4/+0.5 | +2.5/+4.8/+2.6/+2.4 |
| 80 | 58.9/71.1/82.5/86.0 | 58.6/71.6/82.4/86.3 | 56.1/66.3/79.5/84.0 | +0.3/-0.5/+0.1/-0.3 | +2.8/+4.8/+3.0/+2.0 |
| 90 | 59.4/71.4/83.0/86.9 | 59.1/71.2/82.6/86.8 | 57.5/67.9/81.2/85.3 | +0.3/+0.2/+0.4/+0.1 | +1.9/+3.5/+1.8/+1.6 |
| 100 | 59.0/70.4/82.4/86.3 | 58.8/70.5/82.2/86.1 | 56.9/67.1/79.6/83.8 | +0.2/-0.1/+0.2/+0.2 | +2.1/+3.3/+2.8/+2.5 |
| 110 | 59.1/70.6/82.6/86.2 | 58.8/70.4/81.8/86.1 | 57.4/67.4/80.5/84.6 | +0.3/+0.2/+0.8/+0.1 | +1.7/+3.2/+2.1/+1.6 |

e110相对两个关键对照再次四项为正，但核心mAP/R1宿主裕量仍仅`+0.3/+0.2`，不能覆盖前序反转或提前GO。
读取时最新=`e114 iter180/227`，wrapper/训练主PID存活，唯一CUDA compute约`7066 MiB / 48%`，runner/train
严格异常=`0`，formal tracked worktree/index=`0/0`。当前=`CONTINUE TO NATURAL E120 / FINAL POINT NEXT`。

## 2026-07-23：correct自然e120完成并封存

correct自然完成e120=`59.2 mAP / 70.7 R1 / 82.7 R5 / 86.3 R10`；同epoch sealed zero-owner=
`58.9/70.3/81.9/86.2`，sealed clean D0=`57.6/67.7/80.8/84.6`。最终一行：

| epoch | PSCIR correct | zero-owner | clean D0 | Δzero-owner | ΔD0 |
|---|---|---|---|---|---|
| 120 | 59.2/70.7/82.7/86.3 | 58.9/70.3/81.9/86.2 | 57.6/67.7/80.8/84.6 | +0.3/+0.4/+0.8/+0.1 | +1.6/+3.0/+1.9/+1.7 |

预注册性能门要求mAP/R1同时严格胜zero-owner与D0；e120四个核心差值均严格为正，因此correct裁决=
`PERFORMANCE GO / ATTRIBUTION PENDING`。这只授权matched controls，不证明pose×CLIP联合归因。

封存核验：

- `120/120`完整epoch、`12/12`正式评测，`TRAIN_EXIT=0`，主进程自然退出；
- runner/train严格异常=`0`，GPU=`2 MiB / 0%`且CUDA compute=`0`；
- formal HEAD=`c6739b402a9e9a16f2427324536251e5ed059598`，tracked worktree/index=`0/0`；
- checkpoint/train-log/runner SHA256=
  `46e6290a0883fe6bf7e5d005da7ffea5f4e058f9cf36b95d7f62ac0a4a7e8513`/
  `328febda29a1061b30829b6d3b89478707773a5c3ddedef348bdd8152d1d1e1c`/
  `4a18ca28b8c565fafa2c3874f2098c53cd935c22fb07a9c5580000454524abfb`。

correct产物永久封存，禁止修改、覆盖、续训或重跑。下一步按预注册以同一formal/seed/recipe和fresh OUTPUT_DIR
串行启动`pose_only`；在其运行期间不启动其他CUDA任务。

## 2026-07-23：pose-only matched control fresh启动

- output=`/home/afr/reid-clean/logs/exp414-pscir-pose-only-s1234-v1`，runner=
  `/home/afr/reid-clean/train-logs/exp414-pscir-pose-only-s1234-v1.runner.log`，启动前均不存在；
- frozen formal、config、seed1234与recipe不变，只命令行切换
  `MODEL.TAPF.PSCIR_CONTROL_MODE=pose_only`并覆盖fresh OUTPUT_DIR，无checkpoint恢复；
- wrapper主PID=`144432`，训练主PID=`144437`；唯一CUDA compute约`6992 MiB / 42%`，8个同命令子进程为
  DataLoader workers；
- 首批PSCIR `loss/zero/region=2.269678/1.970790/2.568565`，
  `positive/negative/segment=47.240055/57.413406/58.243492`，edge-weight=
  `[6.546875,6.765625,6.6875]`；zero-owner首批与correct臂exact；
- 最新=`e1 iter20/227`，runner/train严格异常=`0`，formal tracked worktree/index=`0/0`。

当前=`POSE-ONLY RUNNING / NATURAL E120 REQUIRED`；不得并行启动q-only或修改任何运行内容。

## 2026-07-23：pose-only e4健康检查

- e3自然完成，最新=`e4 iter60/227`，loss=`6.659`、pose=`0.905`、acc=`0.130`；
- e4首批PSCIR `loss/zero/region=0.101798/0.146454/0.057143`，均有限；
- wrapper/训练主PID存活，唯一CUDA compute约`6994 MiB / 42%`，runner/train严格异常=`0`，formal
  tracked worktree/index=`0/0`；
- 尚未到e10正式点，当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：pose-only e9健康检查

- 最新=`e9 iter140/227`，loss=`5.512`、pose=`0.793`、acc=`0.134`；
- wrapper/训练主PID存活，唯一CUDA compute约`6994 MiB / 42%`，runner/train严格异常=`0`，formal
  tracked worktree/index=`0/0`；
- 尚未到e10正式点，当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：pose-only e10首个正式评测

pose-only自然完成e10=`28.2 mAP / 38.1 R1 / 52.5 R5 / 59.0 R10`；同epoch sealed correct=
`28.6/38.1/53.7/60.7`，sealed zero-owner=`28.4/38.1/53.9/60.8`，sealed clean D0=
`33.4/42.7/59.8/65.2`。rounded四项差为：

| epoch | pose-only | correct | zero-owner | clean D0 | Δcorrect | Δzero-owner | ΔD0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 28.2/38.1/52.5/59.0 | 28.6/38.1/53.7/60.7 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | -0.4/+0.0/-1.2/-1.7 | -0.2/+0.0/-1.4/-1.8 | -5.2/-4.6/-7.3/-6.2 |

e10时pose-only的mAP/R5/R10低于correct，R1持平；这只是支持correct的早期方向性证据，不能据此提前作联合
归因或停止matched control。读取时最新=`e11 iter40/227`，wrapper/训练主PID存活，唯一CUDA compute约
`7074 MiB`，runner/train严格异常=`0`，formal tracked worktree/index=`0/0`；当前=
`CONTINUE TO NATURAL E120`。

## 2026-07-23：pose-only e14健康检查

- 最新=`e14 iter180/227`，loss=`3.524`、pose=`0.652`、acc=`0.454`；
- wrapper/训练主PID存活，唯一CUDA compute约`7074 MiB / 58%`，runner/train严格异常=`0`，formal
  tracked worktree/index=`0/0`；
- 最近正式点仍为已登记e10，当前=`CONTINUE TO NATURAL E120`。

## 2026-07-23：pose-only e20正式评测

pose-only自然完成e20=`46.9 mAP / 57.0 R1 / 71.4 R5 / 77.2 R10`；同epoch sealed correct=
`47.1/56.7/71.4/76.9`，sealed zero-owner=`45.6/55.0/70.6/75.8`，sealed clean D0=
`42.2/52.4/67.6/74.0`。完整rounded轨迹更新为：

| epoch | pose-only | correct | zero-owner | clean D0 | Δcorrect | Δzero-owner | ΔD0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 28.2/38.1/52.5/59.0 | 28.6/38.1/53.7/60.7 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | -0.4/+0.0/-1.2/-1.7 | -0.2/+0.0/-1.4/-1.8 | -5.2/-4.6/-7.3/-6.2 |
| 20 | 46.9/57.0/71.4/77.2 | 47.1/56.7/71.4/76.9 | 45.6/55.0/70.6/75.8 | 42.2/52.4/67.6/74.0 | -0.2/+0.3/+0.0/+0.3 | +1.3/+2.0/+0.8/+1.4 | +4.7/+4.6/+3.8/+3.2 |

e20时pose-only相对correct为mAP微负、R1/R10微正、R5持平，形成混合关系；correct尚未在预注册核心mAP/R1
同时严格胜pose-only，不能提前支持联合归因。读取时最新=`e21 iter20/227`，wrapper/训练主PID存活，唯一
CUDA compute约`7054 MiB`，runner/train严格异常=`0`，formal tracked worktree/index=`0/0`；当前=
`CONTINUE TO NATURAL E120`。

## 2026-07-23：pose-only e25健康检查

- 最新=`e25 iter100/227`，loss=`0.931`、pose=`0.501`、acc=`0.920`；
- wrapper/训练主PID存活，唯一CUDA compute约`7054 MiB / 43%`，runner/train严格异常=`0`，formal
  tracked worktree/index=`0/0`；
- 最近正式点仍为已登记e20，当前=`CONTINUE TO NATURAL E120`。
