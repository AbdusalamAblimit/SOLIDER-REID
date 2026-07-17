# exp378 N0 固定置换实现审查

## 审查范围

- 预注册：`design.md`、`n0_permutation_design.md`；
- candidate commit：`6535981`；
- 默认配置、N0 config、TAPF module、full-model接线、processor日志、CPU单元、full-model
  invariants与CUDA preflight门禁；
- 直接对照：corrected residual-OFF hard F0。

本轮未使用Claude，未连接4090训练，也未启动任何N0进程。

## 结论

**本地实现与4090 PyTorch1.13.1+cu117 candidate生产门禁PASS；尚需把本证据写入新的exact
execution commit/bundle并在最终repo复验，复验前不得启动训练。**

N0已收敛为F0的单变量关节语义对照：固定17-cycle只在TAPF内部对bootstrap teacher heatmap与
joint confidence各重排一次；mode仍为`f0`、transition仍为`hard`，geometry residual关闭。
N0 config相对F0除注释外只增加置换列表并更换独立output。没有新增参数、persistent buffer或
optimizer group，也不消耗运行时随机数。

## 静态与动态证据

1. `POSE_TAPF_BOOTSTRAP_JOINT_PERMUTATION`默认空列表；旧12项TAPF单元全部保留通过，说明默认
   forward/gradient/freeze/relaxation路径未回归；
2. 新增N0单元拒绝长度错误、重复/越界和含fixed point的列表；固定置换为
   `[1,2,...,16,0]`，destination→source定义与design一致；
3. 内部置换与“关闭开关后在调用前显式重排teacher+score”在epoch 1/10的field和pose loss逐位
   相等；带唯一score tag的直接检查证明heatmap/score都恰好重排一次且原输入未原地修改；
4. epoch 1 PSG field逐位等于一次置换后的teacher；pose loss只更新anchor，不进入feature或
   geometry adapter；
5. F0/N0 state_dict keys、初值、named parameters和SGD param groups逐位相等；full-model CPU
   invariants进一步验证构造CPU RNG、完整state、production optimizer groups相同，N0 PSG输入
   恰为一次置换teacher；
6. processor新增`teacher_permutation_active/fixed_points`，只扩展TAPF诊断日志，不改变loss；
7. CUDA preflight已增加fail-closed检查：N0只能使用预注册17-cycle、`f0`、hard transition；e1
   检查PSG raw field exact once-permuted，e11沿用`_ExplodingPoseDict`证明不读取teacher，并保留
   batch64、AMP、梯度归属、adapter零更新与真实overflow门禁。

本地结果：

```text
15 passed
TAPF_MODEL_INVARIANTS_PASS
RG0_MODEL_INVARIANTS_PASS
N0_MODEL_INVARIANTS_PASS
device=cpu b0_keys=211 tapf_keys=259 r0_keys=227 descriptor=(2, 768) featmaps=4
```

本地uv环境缺少生产repo遗留但当前forward未使用的`cv2`与checkpoint loader；invariants脚本只在
这些可选包不存在时注入fail-closed stub，且配置强制`PRETRAIN_CHOICE=none`。4090生产复验必须
使用真实依赖，不把本地stub结果冒充PyTorch1.13证据。

4090 candidate=`8aa473898921da100608dee501f6dad489fc59b5`、full-history bundle
SHA256=`3e91e7269c649555b574500ad47bbca29bef6a7b23ad1c2bb38bdd515a9f476b`、config
SHA256=`50b516f78458bc08c8d0d0192d561934facbeb938eddcd30d03f93b41b090814`。第一次candidate单元运行
发现PyTorch1.13 restricted loader不能读取测试自产生的optimizer-state payload；生产代码未进入
失败路径，未跑CUDA或训练。commit `8aa4738`将该可信内存测试包显式兼容回退后，独立新repo重跑：

```text
N0_UNIT_PASS count=15
TAPF_MODEL_INVARIANTS_PASS
RG0_MODEL_INVARIANTS_PASS
N0_MODEL_INVARIANTS_PASS
TAPF_CUDA_PREFLIGHT_PASS  # epoch 1, batch64
runtime_parity_steps=10 overflow_scale=128.0->64.0
TAPF_CUDA_PREFLIGHT_PASS  # epoch 11, batch64
runtime_parity_steps=10 overflow_scale=128.0->64.0
```

e1为`mode=f0`、anchor delta=`9.72232689e-4`、adapter delta=`0`，固定17-cycle/fixed-points=0；
e11 pose=`None`，anchor/adapter的独立gradient与optimizer delta均为`0`，hard transition生效，
teacher raw=`None`。两次preflight都通过eval correct/None/不可索引external pose exact parity与真实
GradScaler overflow。原始日志位于`remote_artifacts/exp378_n0_preflight_8aa4738/4090/`，SHA256：

- unit：`9918315890e27fba94ebf54eed055792e2a592eba4b790936c6b84df2c2de53d`；
- CUDA full-model：`859e1cd9871b45d21f6b9274770b6643302858e1a76b8a1b5fb1a2e1c962e62f`；
- e1：`c249b2601afd9b576e03925bdb57507dea487283fb6acaed53bc77059b8a40ca`；
- e11：`685a4bfe4a4015ddabc952b1793eb6c82baa32bb3aa162d360727ccb64632190`。

## 风险与剩余门禁

- 固定置换作为非persistent metadata有意保持F0/N0 state完全相同；因此checkpoint语义依赖exact
  config与execution bundle，后续SHA审计不可省略，也禁止跨F0/N0 config静默加载后作解释；
- N0保留同一人的正确空间pose支持，只破坏通道解剖名称。持平只能否定“正确关节标签必要”，
  不能否定generic pose-like part field；
- 单一17-cycle可能与PSG随机通道初始化偶然配对，单seed小差值不能写成显著性结论；
- candidate门禁结束时tracked clean、N0 output不存在、无训练主进程、GPU=`2 MiB/0%`。尚需把
  本证据提交为新的exact execution commit，生成完整history bundle和独立final repo，再复验
  HEAD/config、15项unit、full-model与e1/e11 CUDA关键门禁；只有最终repo全部通过且output仍不存在，
  才能fresh启动。
