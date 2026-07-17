# exp379 Progressive Hierarchical TAPF 监控

## 当前状态

- 阶段：4090 fresh HT0已自然完成120 epoch并通过结束审计，禁止重启、续训或重复训练；
- 直接对照：fresh同机D0=`56.2/67.6/79.8/83.4`；
- 总基线：同机B0=`55.1/66.7/79.5/83.8`；
- HT0 final=`56.1/67.6/79.9/83.4`，相对D0=`-0.1/+0.0/+0.1/+0.0`，
  相对B0=`+1.0/+0.9/+0.4/-0.4`（固定顺序mAP/R1/R5/R10）；
- 目标arm：HT0，source stages=`[1,2]`，consumer stages=`[2,3]`；
- batch=`64`，seed=`1234`，120 epochs，OUTPUT_DIR=
  `log/occluded_duke/exp379_ht0_hierarchical_tapf_s1234`；
- exact execution commit=`2181e940c4b8b4d032b9e5fb0de2ce57c9e84720`，原4090 main
  PID=`652316`已自然退出。

## 论文与解释边界

- anchor+PSG作为一个完整模块，不要求分别报告独立增益；
- 主张是训练期姿态监督、推理期RGB-only，并基本替代原始测试期外部ViTPose+PSG；
- exp378语义审计只限制“精确关节通道在推理时因果生效”的强措辞，不否定完整模块相对B0的
  `+1.1 mAP`；
- exp379不是把同一外部热图复制到多个stage，必须证明field1/field2由对应视觉层逐级产生；
- 不因单一epoch提前裁决；Swin final中性也不自动取消一次预注册ResNet/ViT迁移。

## Gate A清单

- [x] config-off旧B0/D0 state/init/RNG/optimizer/forward exact parity；
- [x] source `[1,2]`、consumer `[2,3]`拓扑与field路由hook通过；
- [x] stage projections独立、LiteHR decoder严格共享一份；
- [x] Stage-2 posterior/confidence显式读取并修正detached Stage-1 state；
- [x] 两节点pose loss均有效且总loss为算术平均；
- [x] pose/ReID梯度归属通过；
- [x] eval external pose sentinel exact parity；
- [x] CPU单元测试通过；
- [x] PyTorch1.13.1 batch64 CUDA/AMP e1/e11通过；
- [x] 10-step legacy parity与真实overflow门禁通过；
- [x] exact execution commit、full-history bundle、config SHA与output不存在审计通过；
- [x] 4090唯一训练与GPU空闲审计通过。

### 2026-07-17 本地实现门禁

- 新增默认关闭的`POSE_TAPF_HIERARCHICAL`与source stages配置；config-off旧exp378
  `TAPF_MODEL_INVARIANTS_PASS / RG0_MODEL_INVARIANTS_PASS / N0_MODEL_INVARIANTS_PASS`，旧TAPF
  单元回归退出码为0；
- exp379独立CPU utility `7/7 PASS`：D0 handoff课程、单共享decoder、e1两个consumer teacher exact、
  两节点loss严格均值、持续pose梯度、ReID→pose隔离、真实prior refinement与stage order均通过；
- full-model `EXP379_MODEL_INVARIANTS_PASS`：B0/D0/HT0构造后RNG exact，共享backbone/classifier与
  Stage-3 PSG初始化exact，初始descriptor和四级featmap相对B0逐位相同，strict reload逐位相同；
- HT0 state keys=`283`，pose module参数=`71,074`；Stage-1/Stage-2内部场最大差=
  `0.101957679`，证明deeper field不是浅层field复用；Stage-2/Stage-3 hook路由、两节点聚合、
  external exploding sentinel不读取、pose/ReID objective梯度归属均PASS；
- 此处为本地门禁节点；其后4090原生CUDA/AMP与legacy门禁见下一节。

### 2026-07-17 4090原生CUDA/AMP与legacy门禁

- 门禁代码提交=`994e35d`，运行时=`torch 1.13.1+cu117 / CUDA 11.7`，真实Occluded-Duke
  batch=`64×3×384×128`；专用preflight完整日志SHA256=
  `a7014f352ccfc06184a00f6234df405ad4abbf8215ade495d143a5bd6ece8827`；
- `EXP379_CUDA_PREFLIGHT_PASS`：
  - e1 `identity/pose/total=18.70896912/2.89328432/21.60225296`，Stage-2与Stage-3 PSG均收到
    exact target-person teacher，两个raw field差=`0`；
  - e11同一fresh初态下loss有限，两个consumer均改用内部field，Stage-2/Stage-3 field max差=
    `0.164029941`；
  - source Stage-1/2 projection、唯一shared decoder、Stage-2/3 PSG与backbone均有非零有限梯度和
    optimizer delta；pose objective不进入backbone/PSG，ReID objective不进入pose module；
  - eval的external correct/shuffle/None/exploding descriptor逐位一致；真实GradScaler overflow严格
    `128→64`，全模型参数和optimizer state逐位不变；
- 直接把整套CUDA pose参数摘要跨进程要求逐位一致并不成立：同一新提交重复运行也会在第2步后出现
  数个FP32 ULP的pose-only差异；PyTorch1.13.1在强制deterministic时明确报出
  `upsample_bilinear2d_backward_out_cuda`无确定性实现。该现象不影响identity路径，但不能被伪写成
  exact；
- 最终10-step legacy gate依据已验证的detach边界做严格因子化：full CUDA batch64逐位比较
  identity trace、ReID/PSG shared model和optimizer；再把首步真实Stage-2 feature/teacher/score送入
  CPU TAPF，逐位比较10步pose loss、参数和optimizer。父提交`d4ccdca`与新生产代码得到完全相同的
  exact signature=`fb76a29a1afdb7b2e5db0dfe2534cc607102ab5cf30a128964bd5ac99dd1a72c`；父/新日志
  SHA256分别为`5bb258164acf260028e30b94c76d1e2465082e9fa481af1c152dfe99a8a145f0`和
  `64e3308891026118f7fee826507f6e36dd67bbdeed16849dc860df2974191021`；
- 三份门禁日志均无`NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow warning`（唯一overflow为
  主动注入并通过的128→64测试）；门禁结束后GPU=`2 MiB / 0%`且无compute process。最终execution
  commit、branch full-history bundle与全新remote repo在下一节点固化后才启动HT0。

### 2026-07-17 final execution固化与fresh启动

- exact execution commit=`2181e940c4b8b4d032b9e5fb0de2ce57c9e84720`；branch full-history bundle
  SHA256=`31f504e6dc170c2fbf3a089bd1ef773ffbb2bc79bdd2b3989dbae7db45265d9a`；config
  SHA256=`6af849ee589d3b4eb87c3de05518c3874be52586b2738a3655dac7540ca27e47`；
- 独立4090 repo=`/home/afr/SOLIDER-REID-exp379-2181e94`；启动前tracked diff clean、目标output
  不存在、GPU=`2 MiB / 0%`且无compute process；
- 最终提交重新执行四道门禁并全部通过：unit=`7/7 PASS`、
  `EXP379_MODEL_INVARIANTS_PASS`、`EXP379_CUDA_PREFLIGHT_PASS`、legacy exact signature=
  `fb76a29a1afdb7b2e5db0dfe2534cc607102ab5cf30a128964bd5ac99dd1a72c`；四份终检日志SHA256依次为
  `42c1d5ab3513209218750fd2e885c4fab7d98de0c61a4572ba0d9d61c177b1fc`、
  `55b17af5d4a67c9a1aee11e8eae1263e76ab9c1250082d3762cccd7abbd22b0f`、
  `88f359ac7c03ec9ef0d93bc12c0bb293d021e940568714e3336d754ad3465906`、
  `213d2ccabfcf59cf71e90b581df20a3db4d6033abd2c53e880101914ce4b067a`；
- fresh启动main PID=`652316`，output=
  `log/occluded_duke/exp379_ht0_hierarchical_tapf_s1234`；初始化后唯一main+8 DataLoader workers，
  GPU约`8122 MiB / 85%`；
- e1已自然完成（`32.893 s`），当前健康进入e2。e1全程`student_fraction=0`，
  `hierarchical_stage_count=2`，Stage-1 `refinement_active=0`、Stage-2 `refinement_active=1`，两个节点
  pose/shape/confidence loss均持续有限；Stage-2 confidence refinement约`0.047～0.048`。日志未出现
  `NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow warning`，AMP init scale=`1024`；继续至final，
  不以bootstrap或任何单epoch裁决。

## 训练记录

每次完整eval固定记录`mAP/R1/R5/R10`，并现场计算：

1. `HT0−同epoch D0`四项显式正负差值；
2. `HT0−同epoch B0`四项显式正负差值；
3. 两节点pose loss、posterior/confidence、Gaussian统计；
4. field1→Stage-2、field2→Stage-3路由持续有效；
5. shared decoder与两个projection参数有限更新，PSG参数有限更新；
6. external pose在eval不读取；
7. AMP与严格异常。

| 机器 | arm | epoch | mAP | R1 | R5 | R10 | vs D0 | vs B0 | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 4090 | HT0 | 10 | 37.6 | 47.0 | 63.1 | 69.5 | `+1.5/+2.0/+2.8/+2.8` | `+0.7/+0.0/+1.3/+1.2` | bootstrap端点，继续 |
| 4090 | HT0 | 20 | 46.8 | 57.8 | 71.5 | 76.6 | `+1.5/+2.2/+1.7/+1.3` | `+4.5/+4.1/+5.8/+5.8` | 早期正向，继续 |
| 4090 | HT0 | 30 | 51.6 | 62.8 | 75.1 | 80.2 | `-0.6/-1.2/-1.2/-0.3` | `+1.0/+0.9/-0.1/-0.2` | 轨迹回落，继续 |
| 4090 | HT0 | 40 | 53.1 | 64.4 | 76.7 | 82.0 | `-0.1/+0.5/-0.9/+0.1` | `+0.1/-0.7/-0.9/+0.1` | 混合，继续 |
| 4090 | HT0 | 50 | 53.5 | 65.1 | 77.6 | 82.1 | `+0.1/+0.6/-0.4/+0.2` | `+1.4/+1.6/+0.7/+1.1` | 混合，继续 |
| 4090 | HT0 | 60 | 54.8 | 66.4 | 79.0 | 82.9 | `-0.8/-0.7/-0.3/+0.2` | `+1.0/+1.2/+1.3/+0.9` | 不以单点裁决 |
| 4090 | HT0 | 70 | 55.6 | 67.2 | 80.0 | 83.4 | `+0.2/+1.0/+0.5/-0.1` | `+1.2/+0.8/+1.1/+0.3` | 混合，继续 |
| 4090 | HT0 | 80 | 55.6 | 66.2 | 79.3 | 83.6 | `+0.1/+0.1/+0.3/+0.9` | `+1.0/-0.4/+0.4/+0.5` | 继续 |
| 4090 | HT0 | 90 | 56.0 | 67.3 | 79.8 | 83.8 | `-0.3/-0.3/+0.0/+0.3` | `+1.1/+0.9/+0.4/+0.6` | 接近D0，继续 |
| 4090 | HT0 | 100 | 56.0 | 67.3 | 79.9 | 83.8 | `-0.1/-0.1/-0.1/+0.5` | `+1.2/+0.7/+0.6/+0.5` | 接近D0，继续 |
| 4090 | HT0 | 110 | 56.1 | 67.5 | 80.0 | 83.5 | `+0.0/+0.0/+0.3/+0.2` | `+1.0/+0.7/+0.5/-0.1` | 等final |
| 4090 | HT0 | 120 | **56.1** | **67.6** | **79.9** | **83.4** | `-0.1/+0.0/+0.1/+0.0` | `+1.0/+0.9/+0.4/-0.4` | final，中性于D0 |

## 结束审计

- [x] 原main PID=`652316`及workers自然退出，GPU=`2 MiB / 0%`且无训练进程；
- [x] `transformer_{10..120}.pth`共12个checkpoint齐全；
- [x] final checkpoint / runner / train SHA256依次为：
  `6d1370ac0287a0ceaa08a13918e6bd5a1c7c0cb32b351a3ce79d4de57543ca31`、
  `0c6adb6036c10b3c9ecf2f368b60ac1e10a8f59759c792ed294de9eb6508aa6e`、
  `2b5efad3be83fd43e740d0e517d71b5b4d8a32af8bc616e0788489fb3f2f2c01`；
- [x] 12个checkpoint均为283个相同state keys且全部有限；e10→e120的Stage-1 projection
  `3/3 changed(max=0.173674196,L2=2.254205172)`、Stage-2 projection
  `3/3(max=0.086073801,L2=2.068884943)`、共享decoder
  `26/26(max=0.243545890,L2=3.661112233)`、Stage-2 PSG
  `24/24(max=0.108398169,L2=0.377824853)`、Stage-3 PSG
  `8/8(max=0.273603261,L2=0.600098332)`；e20至e120每个节点均逐checkpoint通过全参数有限变化审计；
- [x] final checkpoint在真实Occluded-Duke batch上复核external correct/shuffle/None/
  exploding pose descriptor逐位相同，shape=`2×768`，推理期确实为RGB-only；
- [x] runner/train严格扫描未出现`NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow`；
  生产processor不输出实时AMP scale/skip，故不虚构累计skip，只保留最终preflight已通过的真实
  overflow `128→64`整步跳过证据；
- [x] final对D0与B0的四项显式差值已由日志现场计算并记录；
- [x] 已更新results/decisions/innovation/story；
- [x] Swin-T结论为HT0与D0基本中性，不以该单点取消完整`anchor+PSG`方法；按预注册转入
  ResNet-50同backbone内B0/D0/HT0三臂，每个anchor继续配置一个PSG，再决定ViT与Video ReID。

## Final裁决

HT0在Swin-T上没有形成相对单点D0的额外可分辨增益：final mAP低`0.1`，其余三项为
`+0.0/+0.1/+0.0`。因此不宣称“逐层优于单层”，也不在Swin上补救层数、loss权重或独立decoder
小变体。另一方面，逐层拓扑、每anchor对应PSG、共享decoder、连续pose监督和推理期RGB-only均已
被完整执行与审计，且HT0仍相对B0保留`+1.0 mAP/+0.9 R1`。这支持把`anchor+PSG`作为完整方法
对象继续做backbone迁移，但单seed不足以支持稳定性或显著性结论。
