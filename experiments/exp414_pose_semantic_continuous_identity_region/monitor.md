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
