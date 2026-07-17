# exp378 N0：固定关节置换 bootstrap 对照

## 动机

hard F0、D0与外部pose臂都在单seed上达到约`+1.0 mAP`，但这还不能证明收益需要正确的17关节
通道语义。随机初始化或随机噪声控制会同时改变学习难度、空间统计与优化轨迹，不能作为干净的
“无正确pose bootstrap”对照。N0因此只破坏teacher的关节标签对应关系，保留同一人的全部姿态
空间信息与置信度分布。

## 核心假设

如果F0的收益依赖正确解剖语义，那么把每个输出通道固定监督为错误关节后，N0应低于matched
hard F0。如果N0与F0持平，则当前收益更可能来自pose-like foreground/part场、短期辅助监督或
Stage-3 PSG容量，而不能归因于正确关节名称。

## 严格单变量定义

N0的唯一直接对照是corrected hard F0，不是P0、MR-F0或D0：

- 两臂均为`POSE_TAPF_MODE=f0`、`POSE_TAPF_ANCHOR_TRANSITION=hard`；epoch 1–10 bootstrap，
  epoch 11–120 anchor严格冻结、pose loss关闭、推理期不读取external pose；
- N0固定destination→source置换
  `[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,0]`。它是17-cycle，`17/17`通道改变且无
  fixed point，不在运行时采样，不消耗额外RNG；
- teacher heatmap与`joint score × target-person mask`按同一置换重排。输出通道`j`在N0中学习
  原teacher通道`perm[j]`；bootstrap handoff送入PSG的teacher field也使用同一重排，禁止只置换
  pose loss或只置换confidence；
- target person、RGB、cache、数据顺序、augmentation、Swin-Tiny、LiteHR-style anchor、Gaussian
  renderer、Stage-3 PSG、geometry adapter实例、参数量、初始化、optimizer groups、LR/AMP日程、
  batch 64、seed 1234与120 epochs全部matched；
- geometry adapter虽与F0一样实例化，但residual始终OFF且不得更新。N0不能借用P0/J0 mode，不能
  开启SGD relaxation，也不能使用旧`POSE_CHANNEL_SHUFFLE`的逐样本随机置换。

实现开关为`MODEL.POSE_TAPF_BOOTSTRAP_JOINT_PERMUTATION`。默认值为空列表，必须逐位保持全部
既有TAPF与baseline行为；N0 config显式写入上述17-cycle。置换只在TAPF内部、teacher shape/score
校验之后和bootstrap loss/field blend之前执行一次。模型state_dict不得新增持久buffer或参数，
运行日志必须记录置换active、fixed-point数与确定性校验。

## 对照与验收门禁

启动前必须全部通过：

1. 空置换下旧TAPF单元、full-model state/init/RNG/optimizer groups与forward逐位不变；
2. 配置拒绝长度非17、重复/越界以及含fixed point的N0置换；N0 17-cycle在CPU/CUDA上保持固定；
3. 内部置换N0与“关闭开关、在调用前显式以同一列表重排heatmap和score”逐位等价；带joint tag
   的输入证明heatmap和score恰好各置换一次，不能二次还原；
4. epoch 1的PSG field逐位等于置换teacher；epoch 10 handoff、pose loss、anchor gradient有限，
   feature/backbone与geometry adapter不接收pose objective gradient；
5. epoch 11后anchor hard-freeze、adapter`0/6 changed`、teacher不被读取，correct/shuffle/None
   external pose descriptor exact parity；
6. PyTorch1.13.1+cu117 batch64 CUDA/AMP、真实overflow 128→64、完整e11 preflight通过；N0和F0
   只有置换config/output不同，独立output不存在，4090无其它训练；
7. 生成新的exact execution commit、完整history bundle与config SHA后，才允许fresh串行启动。

## 结果解释边界

- `N0 < F0`：只支持“正确关节通道对应有贡献”，仍不证明几何residual、SGD relaxation或完整
  TAPF新颖性成立；
- `N0 ≈ F0`：说明当前F0收益不需要正确关节名称，但不能推出姿态空间支持无效，因为N0保留了
  同一人的正确pose-like空间结构；随后必须结合错误姿态、常量场、teacher agreement与通道占用
  审计区分generic part field和真实解剖状态；
- `N0 > F0`：优先检查标签置换是否形成更有利的PSG通道排列与单seed波动，不能解释为错误姿态
  更优；
- 本臂为单seed必要归因，不以一个阈值终止TAPF，也不触发H0或抢跑Hierarchical TAPF。

## 风险与失败解释

固定channel permutation仍保留每张图的正确空间姿态，属于“错误解剖标签”而不是“无姿态”对照；
这是为保持训练难度和统计量matched所作的有意限制。17-cycle也可能与PSG的随机通道权重产生偶然
配对，因此final小差值只能作探索证据。若门禁发现置换改变RNG、参数集合、target person或发生
double permutation，N0无效且不得启动训练。
