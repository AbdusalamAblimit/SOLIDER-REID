# exp394 Production CUDA/AMP预检协议

## 当前状态

`PROTOCOL-FROZEN / CUDA AMP PREFLIGHT SEALED-FAIL / FORMAL NO-START`。

该协议定义的唯一actual-batch门已在step 1、`scaler.step`之前因non-finite gradient正式FAIL，不得
重跑、补步或修改门槛。预检不是正式实验，
不得据其mAP、loss趋势或descriptor gap调rho、loss、teacher、样本或阈值；不得产生可续训checkpoint。

## 冻结source与资产

- 本地实现commit：`11d7a35`；远端执行repo必须由该exact commit建立fresh独立目录，不得修改
  exp393/exp387 sealed repo；
- production source/config SHA沿用CPU result中的六项冻结值；Swin backbone与pose dataset必须继续
  保持`b389b7...8eef`/`d04e74...1bbc` exact；
- config：`configs/occluded_duke/swin_tiny_tapf_rich_budget_c0.yml`；正式字段不得由命令行覆盖；
- CLIP canonical实体：
  `/home/afr/reid-clean/weights/exp394_clip_l14_openclip_9ce2e8a8.safetensors`，必须regular file、非
  symlink，SHA256=`9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e`；
- full codebook SHA256=
  `fb87da370ea945d526f499bef78093a6b07203d87c6d84efe06b5eb6594f954a`；
- pose artifact/manifest、official train映射、pre-RE `teacher_rgb`和clean runtime必须与RZ-C0/D0 sealed
  口径一致；禁止旧runtime、旧pose/cache/path mapping；
- 4090启动前必须`2 MiB/0%`且无compute PID；全程只允许该单一preflight进程及其8个DataLoader worker。

## 冻结CUDA runtime

- 初始误用`/home/afr/reid-clean/.venv`在任何optimizer step前因缺`open_clip`退出；该FAIL只关闭该
  runtime入口，result/runner必须保留，不得向原环境补包；
- 唯一production runtime固定为
  `/home/afr/reid-clean/runtimes/exp394-openclip-reid-py310`：以clean ReID runtime的新实体副本为基础，
  只安装公开精确版本`open-clip-torch==3.3.0`，禁止`PYTHONPATH`拼接、旧runtime、symlink回链或现场再装包；
- 完整freeze为`cuda_runtime.freeze.txt`，SHA256=
  `3d38c99c7f06502d8b40467d2674c966723e5c913d2edf962c5a7088ec60cddb`；CPU-only import必须同时确认
  Torch/OpenCLIP/OpenCV/timm=`2.6.0+cu124/3.3.0/4.13.0/1.0.27`及production repo import PASS；
- 初始缺包入口成功更新数exact 0，因此当时允许canonical runtime从step 0执行一次；该唯一actual入口
  随后已在step 1 gradient finite门FAIL，不再构成任何重跑或补步授权。

## 冻结样本与步数

使用official Occluded-Duke train loader、seed1234、batch64、原sampler与增强。只从fresh iterator读取
24个实际batch，不另选“容易”的图、不读取query/gallery、不按中间loss替换batch。若epoch边界前
iterator耗尽才允许按原sampler自然重建；需记录24批path/PID canonical SHA。

24次AMP optimizer成功更新分为两个不混合阶段：

1. step 1–12：传`tapf_epoch=1`，rho必须Python float exact `0.0`；
2. step 13–24：传`tapf_epoch=6`，rho必须exact
   `0.01615108996629715=rho_star/5`。

不得用epoch10直接放大gap，不得改变teacher/student handoff、loss、microbatch或optimizer。每一步都必须
完成forward、scaled backward、unscale finite检查与真实optimizer update；任何overflow/skip都不计为
成功更新，并使正式门FAIL，而不是补跑第25步。

## 启动前静态门

1. exact HEAD、tracked clean、六个production source/config SHA与两个保护blob SHA；
2. config merge后batch64/seed1234/SGD/lr0.0008、teacher/handoff=`5/5`、rho与全部资产路径/SHA exact；
3. canonical CLIP为非symlink实体，codebook definition/shape/finite/orthogonality exact；
4. 只构建一个model、一个外置rich teacher、一个optimizer和一个GradScaler；teacher不在model children、
   optimizer param groups或model state；
5. 初始化state strict finite；rho不在parameter/buffer/state；两个consumer参数对象与storage独立；
6. GPU无其它任务，runner/result目录为空，不存在checkpoint。

任一启动前门失败时必须在CUDA model forward前退出并保存FAIL资产。

## 真实AMP执行门

### A. teacher阶段 exact identity（step 1–12）

- 每步teacher code=`[64,5,16]`、mask=`[64,5,96,32]`、valid=`[64,5]`，target全部detach/finite，invalid
  code exact zero，valid code norm finite；
- rho逐步exact zero；两个router均被真实调用，`unit_delta/proposal` finite，但production输出相对同一
  输入tokens逐tensor exact；首批完整descriptor与临时all-router-bypass逐tensor exact；
- 即使descriptor route identity，`L_exec`必须nonzero finite，并在12步内使两个consumer各自的
  T/C/E/Expert离开初始化；evidence head、anchor trunk/mask/presence heads、backbone与ID head也必须更新；
- teacher visual参数、slot mean/shared basis和teacher output不得有grad或状态变化。

### B. handoff非零执行（step 13–24）

- rho逐步exact `rho_star/5`且不接受grad；每步两个consumer的applied delta、descriptor与全部loss finite；
- 首个handoff batch full相对all-router-bypass descriptor max-abs与mean-L2都必须严格大于0；不得设性能型
  最小gap门；
- 24步结束时backbone、ID head、evidence head、anchor、两个consumer的T/C/E/Expert均相对初始化有
  nonzero finite轨迹；两个consumer都必须对同一final descriptor有独立非零路径；
- correct/wrong-RGB-style batch permutation/static-zero evidence在同tokens/mask/presence下产生的两个
  consumer pre-budget proposal必须两两非exact，且全部finite；该项只判接口可分，不判统计优劣；
- FP32 per-token channel RMS、normalized proposal与unit/applied delta全部finite；zero-mass与presence=0
  注入的对应slot normalized proposal/scatter exact zero，整批NULL mask/presence descriptor exact identity。

## actual-batch分loss梯度所有权门

在一个冻结actual batch上另做不更新参数的四次隔离backward；每次先清grad并恢复model/RNG，结束后不得
改变optimizer/scaler/state：

1. evidence cosine+relation：只更新evidence head，不更新anchor trunk、backbone、router或ID head；
2. mask+presence（pose heatmap/confidence单独报告）：更新anchor trunk与对应heads，不更新backbone、
   evidence head、router或ID head；
3. 两consumer mean `L_exec`：更新evidence head和两个router的T/C/E/Expert，不更新backbone、anchor
   trunk/mask/presence或ID head；
4. ReID loss：更新backbone、两个router与ID head，不更新anchor trunk/mask/presence/evidence head。

所有应更新组要求每个parameter都有finite nonzero grad；所有应隔离组要求grad为`None`或exact zero，并在
result中区分。isolated backward不计入24次optimizer更新。

## reload、RGB-only与隔离终审

1. 24步后的model state全部finite，保存到临时内存/临时文件后由fresh model `strict=True`零missing/
   unexpected reload；reload前后correct descriptor逐tensor exact；
2. state key不得包含teacher/CLIP/codebook/text/pose，必须包含evidence head、两个consumer的T/C/E/Expert；
3. eval固定rho_star，`pose_batch=None`、shuffle、None与exploding external pose四种输入逐descriptor exact，
   exploding对象访问次数必须0；query/gallery loader不得构建teacher或读pose/codebook；
4. correct start/end、hook restore、model state SHA与所有临时反事实结束后exact；
5. teacher object销毁后显存可回收；不保存optimizer、scaler或checkpoint；输出目录checkpoint数必须0；
6. 报告24步teacher时间、model forward/backward时间、吞吐与peak allocated/reserved。peak allocated必须
   `<22 GiB`且全程无OOM；不得通过改batch或microbatch绕过FAIL；
7. runner严格扫描`NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow/AMP warning`，命中0；进程自然
   退出后4090恢复空闲，fresh repo tracked clean，source/asset SHA前后exact。

## 裁决边界

全部门同时PASS才可记为`CUDA_AMP_PREFLIGHT_SEALED_PASS`。该PASS只授权在另一个明确步骤冻结fresh
e120启动清单，不自动授权训练；不得直接从preflight权重续训。任一FAIL先封存script/result/runner并
归因；只关闭失败的implementation/runtime接口，不改变rho、loss、样本、门槛或Phase 0E结论。

本次实际结果=`CUDA_AMP_PREFLIGHT_SEALED_FAIL`：teacher target shape/finite前置门已过，但step 1
unscale后的model gradient集合含non-finite，成功optimizer update exact `0/24`。当前result未保存具体
parameter归属，因此不得把FAIL过度归到某一个loss/head；它只关闭本production CUDA/AMP接口。

formal e120与semantic multi-stage保持`NO-START`。
