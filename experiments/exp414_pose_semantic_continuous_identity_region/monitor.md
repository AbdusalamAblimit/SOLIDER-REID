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
