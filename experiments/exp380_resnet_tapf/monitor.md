# 实验 exp380：ResNet-50 TAPF 监控

## 当前状态

- 阶段：R50-B0已完整结束并审计通过；fresh R50-D0正在4090训练；HT0尚未启动；
- 训练顺序：R50-B0→R50-D0→R50-HT0，4090严格串行；
- batch=`64`，seed=`1234`，120 epochs；
- 解释顺序：先`D0−B0`，再`HT0−D0`，最后`HT0−B0`；
- 用户禁止Claude，采用unit/full-model/CUDA/AMP/legacy parity门禁；
- exp379 HT0与R50-B0均已结束并禁止重启/续训；4090只允许当前R50-D0单训练。

## Gate A清单

- [x] config默认行为与旧Swin路径不变；
- [x] ResNet B0现有路径exact parity；
- [x] B0/D0/HT0共享初始化与构造后RNG matched；
- [x] ResNet四级shape与source→consumer路由；
- [x] D0单anchor→layer4 PSG；
- [x] HT0每anchor一PSG bank与single shared decoder；
- [x] e1/e11 teacher/internal-field语义；
- [x] pose/ReID梯度归属；
- [x] eval external correct/shuffle/None/exploding exact parity；
- [x] CPU unit与full-model strict reload；
- [x] optimizer groups/membership exact matched；
- [x] PyTorch1.13.1 batch64 CUDA/AMP与真实overflow；
- [x] ImageNet权重、commit/bundle/config/output/repo/GPU固化；

## 2026-07-17 本地实现与CPU门禁

- 新增`ResNetPoseBackboneModel`，不改旧`Backbone`的参数、state keys或forward：
  - D0：layer3 anchor→layer4的3个Bottleneck PSG；
  - HT0：layer2 anchor→layer3的6个Bottleneck PSG，layer3 refined anchor→layer4的3个
    Bottleneck PSG；每个anchor严格对应一个后继PSG bank；
  - 两个stage projection独立，decoder/anchor严格共享一份；
  - eval不索引`pose_dict`。
- 修复生产processor原先只兼容Transformer输出形态的问题：非pose训练现在同时接受旧ResNet的
  `(score, feature)`与Transformer的三项输出；eval同时接受旧ResNet直接返回的tensor和pose/
  Transformer的tuple。未改模型数值路径。
- `py_compile`通过；`test_resnet_tapf.py`=`8/8 PASS`：
  - 官方权重加载同时兼容PyTorch 1.13（无`weights_only`参数）与2.9+；
  - B0旧ResNet类型、完整state与构造后RNG exact；
  - 三臂共享ResNet/BNNeck/classifier初始化exact；D0与HT0的layer4 PSG exact；
  - e1两个consumer输入与teacher逐位相等；e11两个内部field均脱离teacher且彼此不同；
  - pose loss仅更新anchor，ReID loss仅更新ResNet/PSG；
  - correct/shuffle/None/exploding external pose descriptor逐位一致；
  - strict reload及全部state finite。
- 本地`uv`缺少仅由顶层Swin导入链要求的`cv2/mmengine`，且PyPI下载超时；CPU单测进程仅对这两个
  未执行依赖注入最小import stub。模型/processor代码未使用stub，远端PyTorch1.13.1生产门禁必须
  在真实完整环境重跑，不能把本地stub视为CUDA门禁。
- 已新增独立`cuda_preflight.py`并通过`py_compile`；它不启动epoch runner、不写checkpoint。下述
  远端真实环境门禁已验证matched state/RNG/optimizer、B0生产接口、真实数据batch64 D0/HT0
  e1/e11、梯度归属、10-step legacy parity、eval pose-free exact parity及真实AMP overflow整步跳过。

## 2026-07-17 4090生产门禁与R50-B0启动

- Gate repo：`/home/afr/SOLIDER-REID-exp380-90ed55c`；原生环境
  `torch-1.13.1+cu116/CUDA 11.6`；`test_resnet_tapf.py=8/8 PASS`。
- `cuda_preflight.py`使用真实Occluded-Duke `batch=64`、输入`(64,3,384,128)`，全部通过：
  - matched state/RNG/optimizer，参数量B0/D0/HT0分别为
    `24,949,824 / 25,532,262 / 25,891,426`；
  - B0 batch64生产step，loss=`9.19384384`，scale=`1024→1024`；
  - B0 10-step legacy processor parity exact；
  - D0 e1/e11 identity=`9.34684849`、pose=`2.88509893`，均scale=`1024→1024`；
  - HT0 e1/e11 identity=`9.34684849`、pose=`2.93400764`，均scale=`1024→1024`；
  - D0/HT0每个anchor→PSG路由、objective ownership、eval external pose exact parity通过；
  - 真实HT0 AMP overflow整步跳过，scale=`128→64`，model/optimizer state exact不变。
- 固化信息：
  - exact execution commit=`90ed55cf4798f06d1b08e70f84d0e32ca212ff27`；
  - full-history bundle SHA256=`ccc203eba2e0325a6035b6de1dc1a4bfa465860d57f49355448be35aeeaf2b74`；
  - ImageNet ResNet-50权重102,502,400 bytes，SHA256=
    `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`；
  - unit/preflight log SHA256分别为
    `a5bd37a9dbcb93915c71f978152c3f4ad6b5442bb6b1f06f64fe39c39e02136f`、
    `13b850a2d9f6fb026b868a465fa376a23da4a8253b7950d51d5349befd3dc717`。
- 首次启动被tracked-clean门禁安全拦截：1.13单测刷新了仓库历史误提交的`__pycache__`；未创建
  output/训练进程。随后从同一exact commit建立全新生产repo，并固定
  `PYTHONDONTWRITEBYTECODE=1`，不忽略源码树门禁。
- fresh R50-B0：
  - repo=`/home/afr/SOLIDER-REID-exp380-b0-90ed55c`；
  - config SHA256=`66f19177c05e4406614e560832da19422fa43a1c9ec408c711ac3e37a8ad35ee`；
  - output=`log/occluded_duke/exp380_r50_b0_s1234`；main PID=`701633`；
  - 启动时tracked source diff为空、GPU=`2 MiB/0%`；启动后main+8 workers，约9.65 GiB；
  - e1→e120全部自然完成；轨迹非单调且e80高于final，但本实验固定使用e120 arm final；
    loss全程有限，严格NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow=`0`。

## 训练记录

每个完整eval记录`mAP/R1/R5/R10`，并由日志现场计算同epoch matched差值。不得因单一epoch提前
裁决，所有已启动arm必须自然跑满。

| arm | epoch | mAP | R1 | R5 | R10 | matched差值 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| R50-B0 | 10 | 32.0 | 40.6 | 59.1 | 66.2 | 基线锚点 | 继续；已进入e11 |
| R50-B0 | 20 | 29.8 | 38.0 | 56.7 | 64.7 | 基线锚点 | 继续 |
| R50-B0 | 30 | 30.6 | 39.4 | 56.9 | 64.4 | 基线锚点 | 继续 |
| R50-B0 | 40 | 30.2 | 41.1 | 57.1 | 63.3 | 基线锚点 | 继续 |
| R50-B0 | 50 | 30.8 | 40.3 | 56.2 | 62.3 | 基线锚点 | 继续 |
| R50-B0 | 60 | 34.4 | 45.3 | 61.1 | 67.2 | 基线锚点 | 继续 |
| R50-B0 | 70 | 35.3 | 44.3 | 60.2 | 66.7 | 基线锚点 | 继续 |
| R50-B0 | 80 | 36.2 | 47.1 | 63.0 | 69.9 | 基线锚点 | 继续；当前最佳单点 |
| R50-B0 | 90 | 33.7 | 44.0 | 60.9 | 67.2 | 基线锚点 | 继续；不以回落早停 |
| R50-B0 | 100 | 35.4 | 46.7 | 61.6 | 68.0 | 基线锚点 | 继续 |
| R50-B0 | 110 | 35.3 | 45.4 | 61.9 | 68.4 | 基线锚点 | 继续 |
| R50-B0 | 120 | 35.0 | 45.3 | 61.3 | 68.2 | final基线 | 完整结束并审计通过 |
| R50-D0 | 10 | 26.5 | 35.8 | 53.0 | 59.7 | 相对同epoch B0=`-5.5/-4.8/-6.1/-6.5` | 继续；早期点不裁决 |
| R50-D0 | 20 | 22.4 | 31.8 | 47.3 | 54.1 | 相对同epoch B0=`-7.4/-6.2/-9.4/-10.6` | 继续；早期点不裁决 |
| R50-D0 | 30 | 28.9 | 37.9 | 55.4 | 62.0 | 相对同epoch B0=`-1.7/-1.5/-1.5/-2.4` | 继续；轨迹已回升，仍不裁决 |
| R50-D0 | 40 | 29.1 | 38.5 | 55.8 | 62.4 | 相对同epoch B0=`-1.1/-2.6/-1.3/-0.9` | 继续；早期点不裁决 |
| R50-D0 | 50 | 30.6 | 40.2 | 57.6 | 64.5 | 相对同epoch B0=`-0.2/-0.1/+1.4/+2.2` | 继续；未到e60不裁决 |
| R50-D0 | 60 | 36.5 | 47.4 | 64.3 | 70.8 | 相对同epoch B0=`+2.1/+2.1/+3.2/+3.6` | 继续；首个中期全正点 |
| R50-D0 | 70 | 35.7 | 45.7 | 64.1 | 71.7 | 相对同epoch B0=`+0.4/+1.4/+3.9/+5.0` | 继续；不以单点裁决 |
| R50-D0 | 80 | 37.1 | 47.1 | 64.5 | 71.4 | 相对同epoch B0=`+0.9/+0.0/+1.5/+1.5` | 继续；等final |
| R50-D0 | 90 | 35.4 | 44.8 | 62.7 | 68.9 | 相对同epoch B0=`+1.7/+0.8/+1.8/+1.7` | 继续；等final |
| R50-D0 | 100 | 38.7 | 49.1 | 65.9 | 71.6 | 相对同epoch B0=`+3.3/+2.4/+4.3/+3.6` | 继续；等final |
| R50-D0 | 110 | 38.5 | 49.5 | 65.2 | 71.5 | 相对同epoch B0=`+3.2/+4.1/+3.3/+3.1` | 继续；等final |
| R50-D0 | 120 | 38.1 | 49.4 | 64.6 | 71.1 | 相对B0 final=`+3.1/+4.1/+3.3/+2.9` | 完整结束并审计通过 |
| R50-HT0 | 10 | 25.3 | 34.9 | 52.1 | 60.1 | 相对同epoch D0=`-1.2/-0.9/-0.9/+0.4` | 继续；bootstrap端点不裁决 |
| R50-HT0 | 20 | 19.4 | 29.1 | 44.3 | 51.7 | 相对同epoch D0=`-3.0/-2.7/-3.0/-2.4` | 继续；早期点不裁决 |
| R50-HT0 | 30 | 26.6 | 36.9 | 54.2 | 61.2 | 相对同epoch D0=`-2.3/-1.0/-1.2/-0.8` | 继续；差距收窄但不裁决 |
| R50-HT0 | 40 | 28.3 | 38.2 | 54.6 | 61.3 | 相对同epoch D0=`-0.8/-0.3/-1.2/-1.1` | 继续；早期点不裁决 |
| R50-HT0 | 50 | 32.7 | 41.9 | 60.5 | 67.6 | 相对同epoch D0=`+2.1/+1.7/+2.9/+3.1` | 继续；单点不裁决 |
| R50-HT0 | 60 | 35.7 | 44.8 | 62.8 | 69.3 | 相对同epoch D0=`-0.8/-2.6/-1.5/-1.5` | 继续；轨迹混合 |
| R50-HT0 | 70 | 36.5 | 46.0 | 63.6 | 70.2 | 相对同epoch D0=`+0.8/+0.3/-0.5/-1.5` | 继续；四项混合 |
| R50-HT0 | 80 | 33.8 | 43.2 | 60.8 | 67.4 | 相对同epoch D0=`-3.3/-3.9/-3.7/-4.0` | 继续；不以单点裁决 |
| R50-HT0 | 90 | 35.9 | 46.6 | 62.1 | 68.5 | 相对同epoch D0=`+0.5/+1.8/-0.6/-0.4` | 继续；四项混合 |
| R50-HT0 | 100 | 38.5 | 49.9 | 65.5 | 71.5 | 相对同epoch D0=`-0.2/+0.8/-0.4/-0.1` | 继续；近中性 |
| R50-HT0 | 110 | 39.2 | 51.2 | 66.2 | 72.1 | 相对同epoch D0=`+0.7/+1.7/+1.0/+0.6` | 继续；等final |
| R50-HT0 | 120 | 38.9 | 50.5 | 65.9 | 72.0 | 相对D0 final=`+0.8/+1.1/+1.3/+0.9` | 完整结束并审计通过 |

## 2026-07-17 R50-B0 final审计与R50-D0启动

- R50-B0 final=`35.0/45.3/61.3/68.2`；e80是更高的中途单点，但论文arm对照固定使用
  e120 final，不做best-checkpoint挑选。
- 原main PID=`701633`及8 workers自然退出，GPU=`2 MiB/0%`，12个10-epoch checkpoint齐全；
  tracked source diff为空，严格NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow=`0`。
- final checkpoint/runner/train SHA256依次为：
  - `115b498077397bf55a3e7b010cae77e0f2bd084f8c8c2d56d8cde78a28301f2e`；
  - `80f31290de409cfac5ca7661dadc39aa7d4a070871bea8c251e361124f8fe8f6`；
  - `8dbdb04a3a98baa8689857de4ec4523e55602c6e211d0af1f3b8de5458a670e0`。
- final checkpoint strict load通过：324个state tensor全部有限，固定synthetic descriptor shape=`(2,2048)`，
  descriptor SHA256=`74cc3083d5aeca08db13e3894d15bbfa162d0f9fa7be9ea1f3a5a93575c505bc`。
- fresh R50-D0已串行启动：
  - repo=`/home/afr/SOLIDER-REID-exp380-d0-90ed55c`；
  - exact execution commit/full bundle/weight与B0完全一致；
  - config SHA256=`35e9ccb3154bddeb109bae16cf06871b5ce7948566fe11b0cfb9bd76d75d95e0`；
  - output=`log/occluded_duke/exp380_r50_d0_s1234`；main PID=`735147`；
  - 启动前output不存在、tracked source clean、GPU空闲；启动后main+8 workers，约9.54 GiB；
  - e1自然结束并进入e2；teacher exact端点、pose loss/anchor confidence/sigma均有限，geometry
    shift/log-scale为0，joint permutation关闭，严格异常=`0`。继续到final。

## 2026-07-17 R50-D0 e10-e120监控

- e10/e20/e30/e40/e50日志原始四项依次为`26.5/35.8/53.0/59.7`、
  `22.4/31.8/47.3/54.1`、`28.9/37.9/55.4/62.0`、`29.1/38.5/55.8/62.4`、
  `30.6/40.2/57.6/64.5`；相对同epoch B0依次为
  `-5.5/-4.8/-6.1/-6.5`、`-7.4/-6.2/-9.4/-10.6`、
  `-1.7/-1.5/-1.5/-2.4`、`-1.1/-2.6/-1.3/-0.9`、`-0.2/-0.1/+1.4/+2.2`。
  e50的mAP/R1已接近同epoch B0且R5/R10转正，但仍早于e60，不作性能裁决，继续自然跑满
  120 epoch。
- e10→e20逐参数审计通过：anchor=`26/26 changed`（max=`0.416441143`，
  L2=`6.524588708`），后继layer4 PSG=`12/12 changed`（max=`0.175001919`，
  L2=`2.861893760`），ResNet state=`318/318 changed`，两端全部有限。
- e10→e30/e40/e50继续逐checkpoint审计通过：anchor均为`26/26 changed`，e50最大差值/L2=
  `0.921935439/12.865280297`；后继layer4 PSG均为`12/12 changed`，e50最大差值/L2=
  `0.379593313/5.305650246`；各checkpoint全部有限。
- e60/e70/e80原始四项依次为`36.5/47.4/64.3/70.8`、`35.7/45.7/64.1/71.7`、
  `37.1/47.1/64.5/71.4`；相对同epoch B0分别为`+2.1/+2.1/+3.2/+3.6`、
  `+0.4/+1.4/+3.9/+5.0`、`+0.9/+0.0/+1.5/+1.5`。e60-e80的mAP连续为正，说明完整
  anchor+PSG在ResNet-50上的中后期信号比Swin-T更强；仍不以三个checkpoint替代e120 final。
- e10→e60/e70/e80逐checkpoint审计继续通过：anchor均为`26/26 changed`，e80最大差值/L2=
  `1.014974952/14.131167317`；后继layer4 PSG均为`12/12 changed`，e80最大差值/L2=
  `0.337169051/5.284060472`；各checkpoint全部有限。
- e90/e100/e110原始四项依次为`35.4/44.8/62.7/68.9`、`38.7/49.1/65.9/71.6`、
  `38.5/49.5/65.2/71.5`；相对同epoch B0分别为`+1.7/+0.8/+1.8/+1.7`、
  `+3.3/+2.4/+4.3/+3.6`、`+3.2/+4.1/+3.3/+3.1`。e60-e110六个连续checkpoint的
  mAP均为正，继续等待固定e120 final，不挑选e100/e110。
- e10→e90/e100/e110逐checkpoint审计继续通过：anchor均为`26/26 changed`，e110最大差值/L2=
  `1.015400887/14.361219347`；后继layer4 PSG均为`12/12 changed`，e110最大差值/L2=
  `0.282567263/4.842732086`；各checkpoint全部有限。
- 检查时remote HEAD仍为`90ed55cf4798f06d1b08e70f84d0e32ca212ff27`，tracked source diff为空；
  D0自然完成e120后main PID=`735147`及8 workers均退出，GPU=`2 MiB/0%`。全程pose loss、
  anchor confidence、sigma有限，joint permutation关闭，shift/log-scale持续为0；严格NaN/Inf/
  Traceback/RuntimeError/OOM/nonfinite/overflow=`0`。生产processor不输出实时AMP scale/skip，
  不虚构累计值。

## 2026-07-17 R50-D0 final审计

- D0 final=`38.1/49.4/64.6/71.1`，相对B0 final=`+3.1/+4.1/+3.3/+2.9`。这是同骨干、
  同seed、同batch、同训练长度下完整anchor+PSG原子方法的明确正结果；不与Swin-T绝对指标横比。
- 原main PID=`735147`及workers自然退出，GPU=`2 MiB/0%`；12个10-epoch checkpoint齐全，
  tracked source diff为空，严格异常=`0`。
- final checkpoint/runner/train SHA256依次为：
  - `3f8f25facd6cb328d93a34f9b9146ff112612117d84017df69acae9f53cabd15`；
  - `fda373758e605cde22c5b273eff038f48bc727096d4808ee0ca3092678a5d60e`；
  - `27bf1d213ad5d8c4e5c9fdc25872a82c5984b0504aa0f619e990c008ed04ea23`。
- e10→e20/.../e120每个checkpoint均为anchor=`26/26 changed`、后继layer4 PSG=
  `12/12 changed`且全部有限；e10→e120 ResNet state=`318/318 changed`。
- final checkpoint strict load通过：368个state tensor全部有限、missing/unexpected=`0/0`；真实batch的
  correct/shuffle/None/exploding external pose descriptor逐位一致，shape=`(2,2048)`，descriptor
  SHA256=`144ca961c2d2c609060fb18a60ce7a769bb716298cb07c1eaf3b1efcd889b45b`。

## 2026-07-17 fresh R50-HT0启动

- D0终审完成且GPU空闲后，从完整history bundle建立独立生产repo：
  - repo=`/home/afr/SOLIDER-REID-exp380-ht0-90ed55c`；
  - exact execution commit=`90ed55cf4798f06d1b08e70f84d0e32ca212ff27`；
  - bundle SHA256=`ccc203eba2e0325a6035b6de1dc1a4bfa465860d57f49355448be35aeeaf2b74`，
    `git bundle verify`确认完整history；
  - config SHA256=`c67532184fa061b769f1d3c09945f95bc212d13481e311dfefc994db28fcc773`；
  - ImageNet权重SHA256=`19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`；
  - output=`log/occluded_duke/exp380_r50_ht0_s1234`，main PID=`770625`。
- 启动门禁确认tracked source diff为空、output不存在、无训练进程、GPU=`2 MiB/0%`。首个启动命令
  被进程检查表达式自匹配安全拦截，未创建output/训练；修正为按`comm=python`过滤后重新执行全部
  门禁并fresh启动，不属于重复训练或续训。
- 启动后main PID=`770625`已由PID 1接管，唯一main+8 workers，GPU约`9.55 GiB`；e1 iter150
  健康，两个hierarchical stage均启用，stage2 refinement active=`1`，两个stage的pose loss、anchor
  confidence与field统计均有限，严格NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow=`0`。
  固定`PYTHONDONTWRITEBYTECODE=1`，继续自然跑满120 epoch。

## 2026-07-17 R50-HT0 e10-e30与handoff监控

- e10=`25.3/34.9/52.1/60.1`；相对同epoch D0=`-1.2/-0.9/-0.9/+0.4`。这是bootstrap
  端点且早于e60，不作性能裁决。
- e11健康越过handoff：总student fraction与stage1/stage2 student fraction均为`1.0`；两个stage
  pose loss、anchor confidence、field统计持续有限，stage2 confidence refinement L1约`0.012`且
  refinement active=`1`。stage1是首级anchor，不执行二次refinement，符合设计。
- 检查时HEAD/config SHA保持固化值，tracked source diff为空；main PID=`770625`及8 workers唯一，
  GPU约`9.47 GiB`，严格异常=`0`。继续到e20并做首个shared anchor、stage projection、全部后继
  PSG bank与ResNet逐参数轨迹审计。
- e20/e30分别为`19.4/29.1/44.3/51.7`、`26.6/36.9/54.2/61.2`；相对同epoch D0分别为
  `-3.0/-2.7/-3.0/-2.4`、`-2.3/-1.0/-1.2/-0.8`。e30较e20明显回升，但仍早于e60，
  不作性能裁决。
- e10→e20/e30逐参数审计通过：shared anchor均为`26/26 changed`，stage projections均为
  `6/6 changed`，全部9个后继PSG bank共`36/36 changed`，ResNet state=`318/318 changed`；
  所有checkpoint全部有限。e10→e30各组最大差值/L2依次为anchor=
  `0.694823682/8.192951311`、projection=`0.346082449/5.381372564`、PSG=
  `0.300782770/5.382225769`。
- 最新健康完成e37并进入e38；两级student fraction持续为1，stage2 refinement active持续为1，
  两级pose/anchor/field/refinement统计有限，唯一main+8 workers、GPU约`9.47 GiB`、严格异常=`0`。
- e40/e50/e60分别为`28.3/38.2/54.6/61.3`、`32.7/41.9/60.5/67.6`、
  `35.7/44.8/62.8/69.3`；相对同epoch D0分别为`-0.8/-0.3/-1.2/-1.1`、
  `+2.1/+1.7/+2.9/+3.1`、`-0.8/-2.6/-1.5/-1.5`。e50四项全正但e60转负，轨迹混合，
  不以任何单点裁决。
- e10→e40/e50/e60逐参数审计继续通过：shared anchor=`26/26 changed`、stage projections=
  `6/6 changed`、全部PSG banks=`36/36 changed`、ResNet state=`318/318 changed`，各checkpoint
  全部有限。e10→e60各组最大差值/L2依次为anchor=`0.876454651/10.873823070`、
  projection=`0.496742606/7.215299945`、PSG=`0.370275140/6.474857052`。
- 最新运行e70，两个stage student fraction持续为1，stage2 refinement active持续为1，pose loss、
  anchor/field/refinement统计有限；唯一main+8 workers、GPU约`9.47 GiB`、严格异常=`0`。
- e70/e80分别为`36.5/46.0/63.6/70.2`、`33.8/43.2/60.8/67.4`；相对同epoch D0分别为
  `+0.8/+0.3/-0.5/-1.5`、`-3.3/-3.9/-3.7/-4.0`。e70四项混合、e80四项全负，
  与此前e50全正共同说明轨迹波动较大，继续等待e90-e120。
- e10→e70/e80逐参数审计继续通过：shared anchor=`26/26 changed`、stage projections=
  `6/6 changed`、全部PSG banks=`36/36 changed`、ResNet state=`318/318 changed`，全部有限。
  e10→e80各组最大差值/L2依次为anchor=`0.876465201/11.458959220`、projection=
  `0.516223192/7.553706388`、PSG=`0.357003808/6.582933823`。
- 最新健康完成e89；唯一main+8 workers、GPU约`9.47 GiB`，两级统计持续有限，严格异常=`0`。
- e90/e100/e110分别为`35.9/46.6/62.1/68.5`、`38.5/49.9/65.5/71.5`、
  `39.2/51.2/66.2/72.1`；相对同epoch D0分别为`+0.5/+1.8/-0.6/-0.4`、
  `-0.2/+0.8/-0.4/-0.1`、`+0.7/+1.7/+1.0/+0.6`。e110四项全正，但固定使用e120
  final，不挑选e110。
- e10→e90/e100/e110逐参数审计继续通过：shared anchor=`26/26 changed`、stage projections=
  `6/6 changed`、全部PSG banks=`36/36 changed`、ResNet state=`318/318 changed`，全部有限。
  e10→e110各组最大差值/L2依次为anchor=`0.876465201/11.712183604`、projection=
  `0.518754959/7.671758711`、PSG=`0.277350724/5.961777820`。
- 最新健康完成e112；唯一main+8 workers、GPU约`9.44 GiB`，两级student/anchor/field/refinement
  统计持续有限，严格异常=`0`。继续自然完成e120与final审计。

## 2026-07-17 R50-HT0 final审计

- HT0 final=`38.9/50.5/65.9/72.0`；相对D0 final=`+0.8/+1.1/+1.3/+0.9`，相对B0
  final=`+3.9/+5.2/+4.6/+3.8`。逐层增量在ResNet-50单seed上超过预注册`+0.3 mAP`
  描述线，但尚不能据此声称跨backbone稳定优于D0；Swin-T对应差值仍为`-0.1 mAP`。
- 原main PID=`770625`及8 workers自然退出，GPU=`2 MiB/0%`；12个checkpoint齐全，tracked
  source diff为空，严格NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow=`0`。
- final checkpoint/runner/train SHA256依次为：
  - `0de30767818a66c7dd75f962d7727fe55d28d764a770767764a9607938a23457`；
  - `a63d7dc6c979b88a874f2ccbbafc1bb536fe2bf5105e97a0b36ac9d9a20eb8a7`；
  - `d5d6454b16404ca3216b0fcf8a25a499f511c8316591f390d91b2df2805957df`。
- e10→e20/.../e120全部checkpoint均为shared anchor=`26/26 changed`、stage projections=
  `6/6 changed`、全部9个后继PSG bank=`36/36 changed`、ResNet state=`318/318 changed`，
  所有state全部有限。
- final checkpoint strict load通过：392个state tensor全部有限、missing/unexpected=`0/0`；真实batch
  correct/shuffle/None/exploding external pose descriptor逐位一致，shape=`(2,2048)`，descriptor
  SHA256=`665d37a5d8c17b06c782d3daea39b0cee1365ce47a96c84700de489d8ce150a9`。

## 结束审计

- [x] main/workers自然退出且GPU空闲；
- [x] 12 checkpoint齐全；
- [x] final/checkpoint/runner/train SHA；
- [x] anchor/PSG/ResNet参数轨迹与严格异常；
- [x] final external pose exact parity；
- [x] B0/D0/HT0同骨干结果表与显式差值；
- [x] 更新results/decisions/innovation/story。
