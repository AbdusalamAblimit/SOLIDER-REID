# exp381 ViT TAPF 监控记录

## 2026-07-17 设计启动

- exp380 已封板：R50-B0/D0/HT0 final=`35.0/45.3/61.3/68.2`、
  `38.1/49.4/64.6/71.1`、`38.9/50.5/65.9/72.0`；文档收尾 commit=`5511332`。
- 4090 当前无训练，GPU=`2 MiB/0%`；未启动 exp381 任何 arm。
- 4090 找到既有 ViT-B/16 ImageNet 权重：
  `/home/afr/reid-clean/weights/jx_vit_base_p16_224-80ecf9dd.pth`，大小=`346292833` bytes，
  SHA256=`80ecf9dd5e3a58895e959af554c5666c4e7b4da4410de4f1f2b0025e93435d8c`。
- 代码盘点确认：TransReID ViT 有 12 个等宽 block、128 个 patch token+1 CLS token；现有
  `PoseBackboneModel` 依赖 Swin `stages`，不能直接复用；当前通用 `build_transformer` 也使用
  Swin 风格构造/加载接口，exp381 必须走独立、默认关闭且有严格加载统计的 wrapper。
- 已固定分组：`[0–2]/[3–5]/[6–8]/[9–11]`。D0 是 block8 anchor→blocks9–11 PSG；
  HT0 是 block5 anchor→blocks6–8 PSG、block8 refined anchor→blocks9–11 PSG。PSG 仅调制
  patch token，CLS exact 旁路。
- 当前状态：**继续设计/实现门禁，不训练**。下一步实现专用 wrapper、三臂 config 与 CPU unit；
  任何 matched/CUDA/AMP/pose-free parity 门禁未通过都不启动。

## 2026-07-17 本地实现与 CPU 门禁

- 新增默认关闭的 `VIT_TAPF_EXPERIMENT` 路由和专用 `VitTapfModel`；旧 Swin/ResNet 路径不改。
- 三臂共同使用 12-block TransReID forward、CLS global descriptor、BNNeck/classifier；额外模块
  构造后恢复公共 RNG。D0/HT0 均先构造 `G3` PSG bank，保证共享 consumer init exact。
- 明确实现 CLS exact 旁路：每个 PSG 只接收 `H×W` patch tokens。HT0 在 block5 产生 field-1，
  blocks6–8消费；block8消费后再产生 refined field-2，blocks9–11消费。
- ViT 权重加载新增 PyTorch1.13/2.x 兼容入口，并要求成功加载数至少为 backbone state keys−2
  （仅允许分类 head 不加载）；生产环境仍需用真实权重验证具体计数。
- 三份独立 config 已创建：ViT-B0/D0/HT0 固定 TransReID 官方 `256×128` recipe、batch64、
  seed1234、120epoch、每10epoch checkpoint/eval。
- 本地 uv 环境 CPU unit=`9/9 PASS`：config 单变量、PyTorch1.13 load fallback、公共
  state/init/RNG exact、D0/HT0 `G3` PSG exact、make_model 路由、CLS 旁路、e1/e11 两级 field、
  梯度归属、eval correct/shuffle/None/exploding parity、strict reload/finite state均通过。
- 本地环境缺少仅由顶层 Swin import 链要求的 `cv2/mmengine`；测试只为导入注入最小 stub，
  exp381 模型没有调用 stub。该结果不替代4090完整环境的真实 ViT-B/CUDA/AMP 门禁。
- 当前状态：**继续，尚未训练**。下一步编写生产 preflight，并在独立 gate repo 用真实权重、
  PyTorch1.13.1、真实 Occluded-Duke batch64 验证加载计数、matched optimizer/state/RNG、
  e1/e11、legacy parity、真实 overflow 与 pose-free parity。
- 已新增只读生产 `cuda_preflight.py` 并通过本地 `py_compile`；脚本不调用 epoch runner、不写
  checkpoint，覆盖真实权重load-count、三臂state/RNG/optimizer、batch64 B0/D0/HT0 e1/e11、
  两次10-step B0 parity、梯度归属、pose-free parity和真实AMP overflow整步跳过。
- 首次4090 unit 门禁在模型构造前安全失败：本地依赖 stub 判断只检查 `sys.modules`，会把远端
  已安装但尚未 import 的真实 `cv2`覆盖为空模块，继而令 mmcv 导入报错。未启动训练、未创建
  output。修正为仅捕获真实 `ImportError` 时注入 stub；必须以新 commit/bundle 重跑全部门禁。

## 2026-07-17 4090生产门禁与fresh ViT-B0启动

- 修正后的 exact execution commit=`caf97468797968ea50959ad859a5ea45516d0655`；独立 gate repo=
  `/home/afr/SOLIDER-REID-exp381-gate-caf9746`。
- 由exact repo在4090生成完整history bundle：
  `/home/afr/exp381-vittapf-caf9746-full.bundle`，大小=`23247412` bytes，SHA256=
  `d9c27502ddecf290fc9a3ee6bbf78c5599a3a6655a87ef467fc79c6d413e1c08`；`git bundle verify`确认
  complete history。
- 4090原生环境=`torch-1.13.1/CUDA 11.7`；权重大小=`346292833` bytes，SHA256=
  `80ecf9dd5e3a58895e959af554c5666c4e7b4da4410de4f1f2b0025e93435d8c`。
- 原生CPU unit=`9/9 PASS`，log SHA256=
  `2c85749281e99f03c4d2a902d33fd526e2a0f46fa81bd773cab987645e920622`。
- 真实Occluded-Duke `batch=64` CUDA/AMP preflight全通过，log SHA256=
  `3f6e9ec93367ddf2efeb4bb13c6f34a9370f547ab7bca2822ef14bf7b4658139`：
  - 三臂权重均严格加载`150/152`（只跳过ImageNet分类head），公共state/RNG/optimizer exact；
  - 参数量B0/D0/HT0=`87,056,104 / 87,351,310 / 87,495,050`，D0/HT0 `G3` PSG exact；
  - B0 batch64 loss=`11.50876999`，AMP scale=`1024→1024`，两次10-step legacy parity exact；
  - D0 e1/e11 identity=`12.13442421`、pose=`3.00800037`，scale=`1024→1024`；
  - HT0 e1/e11 identity=`12.13442421`、pose=`2.99406195`，scale=`1024→1024`；
  - 两层路由、objective ownership、eval external pose exact parity均通过；
  - 真实HT0 overflow整步跳过，scale=`128→64`，model/optimizer state exact不变。
- 首个启动命令的进程表达式自匹配，被门禁在建repo/output前安全拦截；没有训练进程或重复output。
  改为按可执行名检查后重新执行全部启动门禁。
- fresh ViT-B0：
  - repo=`/home/afr/SOLIDER-REID-exp381-b0-caf9746`；
  - config SHA256=`6041f71e7aa95ece3b671fabe8d36e9e7e6d7dcb61b42c6f09743a5cd56843c3`；
  - output=`log/occluded_duke/exp381_vit_b0_s1234`；main PID=`806909`；
  - 启动前output不存在、tracked source clean、GPU=`2 MiB/0%`，固定
    `PYTHONDONTWRITEBYTECODE=1`；启动后唯一main+8 workers，GPU约`5.97 GiB`；
  - e1已自然完成：末次记录loss=`10.508`、acc=`0.006`，epoch耗时约`16.6s`，已进入e2；
    日志中严格NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow=`0`。
- 当前状态：**ViT-B0继续自然跑满120 epoch**。每次检查exact HEAD、tracked source clean、
  唯一main+8 workers、GPU、runner/train log、epoch、每10epoch的mAP/R1/R5/R10、checkpoint及
  严格异常；健康时只更新本文件。B0终审完成并确认GPU空闲前，不启动D0或HT0。

## 2026-07-17 ViT-B0 e10与e19健康检查

- 远端时间=`2026-07-17T11:38:17+00:00`；exact HEAD仍为
  `caf97468797968ea50959ad859a5ea45516d0655`，tracked source diff为空；`data`仅为生产数据入口，
  未改动运行中代码或config。
- main PID=`806909`持续运行，唯一main+8个DataLoader workers；GPU=`5982 MiB/92%`，计算进程显存
  `5974 MiB`。output中的`wrapper.pid`仍精确记录`806909`。
- e10完整评估=`40.8/48.5/67.7/74.3`（固定顺序mAP/R1/R5/R10），已生成
  `transformer_10.pth`。这是ViT同骨干B0轨迹锚点，不与Swin/ResNet绝对值横比，也不据此裁决。
- 检查时e19已自然完成：末次记录loss=`0.387`、acc=`0.976`，训练已继续进入下一epoch；
  日志边界正则下NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow=`0`。
- 当前判断：**继续到e120**。B0完整终审及GPU空闲前不启动D0/HT0，不早停、不续训、不挑best。

## 2026-07-17 ViT-B0 e20–e50与e60运行检查

- 远端时间=`2026-07-17T11:51:32+00:00`；exact HEAD仍为
  `caf97468797968ea50959ad859a5ea45516d0655`，tracked source diff为空。
- 新增完整评估（固定顺序mAP/R1/R5/R10）：
  - e20=`46.2/52.5/71.9/77.8`；
  - e30=`47.1/54.7/72.7/78.5`；
  - e40=`49.8/57.0/74.1/80.2`；
  - e50=`51.4/57.9/75.7/81.8`。
- `transformer_10/20/30/40/50.pth`共5个checkpoint齐全；main PID=`806909`持续运行，
  唯一main+8 workers，GPU=`5982 MiB/93%`、计算进程显存=`5974 MiB`。
- 检查时e59已完成、e60运行至iter 150/227；最新loss=`0.077`、acc=`0.995`，训练统计有限。
  runner/train日志边界正则下NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow=`0`。
- 当前判断：**继续自然跑满e120**。以上仅建立ViT-B0同骨干轨迹，不与其他backbone横比，
  不因e50或任何单点提前裁决。

## 2026-07-17 ViT-B0 e60–e100与e105运行检查

- 远端时间=`2026-07-17T12:06:32+00:00`；exact HEAD仍为
  `caf97468797968ea50959ad859a5ea45516d0655`，tracked source diff为空。
- 新增完整评估（固定顺序mAP/R1/R5/R10）：
  - e60=`50.9/56.8/76.3/81.3`；
  - e70=`51.9/58.4/75.8/81.5`；
  - e80=`52.1/59.0/76.5/81.9`；
  - e90=`52.8/59.1/76.9/82.5`；
  - e100=`52.8/59.3/77.1/82.1`。
- `transformer_10.pth`至`transformer_100.pth`共10个checkpoint齐全；main PID=`806909`
  持续运行，唯一main+8 workers，GPU=`5982 MiB/93%`、计算进程显存=`5974 MiB`。
- 检查时e104已完成、e105运行至iter 100/227；最新loss=`0.060`、acc=`0.997`，训练统计有限。
  runner/train日志边界正则下NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow=`0`。
- 当前判断：**继续自然跑满e120**。e90/e100均不能替代固定e120 final；进程退出和终审前不启动D0。

## 2026-07-17 ViT-B0 final审计

- e110=`52.8/59.5/76.9/81.9`，固定e120 final=`52.9/59.5/77.1/82.0`
  （顺序mAP/R1/R5/R10）。B0仅作为同ViT骨干锚点，不与Swin/ResNet绝对值横比，也不以中途
  checkpoint替代e120。
- 原main PID=`806909`及8个workers自然退出；GPU=`2 MiB/0%`且无计算进程。exact HEAD仍为
  `caf97468797968ea50959ad859a5ea45516d0655`，tracked source diff为空，12个10-epoch
  checkpoint齐全。
- final checkpoint/runner/train SHA256依次为：
  - `af530be09e435dba9da8d42ca8bb9ed940ac31e52b78b9951bd88db5a2c6cc95`；
  - `a781678bfb007714dc5e26759ccb50ff3fb7bd0ea8a99efd50bfe458511191c5`；
  - `5b5940da0c3cbf05c48488d102947c6f87eafcd787935540ff9728a8b96d77ef`。
- final checkpoint strict load通过：158个state tensor全部有限，missing/unexpected=`0/0`；
  固定synthetic descriptor shape=`(2,768)`，descriptor SHA256=
  `9a4ab2b08225a8329f5c3e0d620abdc5f972b889bbcaabf97302afadf7bb27ef`，ViT预训练加载仍为
  `150/152`。runner/train日志边界正则下NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/
  overflow=`0`。

## 2026-07-17 fresh ViT-D0启动

- 本地24090转发一度失效；仅终止失效的本地SSH隧道并经relay4090/tailscale恢复，新隧道exec
  session=`34738`，未重启或影响远端训练。
- B0终审完成且GPU空闲后，从完整history bundle串行建立独立生产repo：
  - repo=`/home/afr/SOLIDER-REID-exp381-d0-caf9746`；
  - exact execution commit=`caf97468797968ea50959ad859a5ea45516d0655`；
  - bundle SHA256=`d9c27502ddecf290fc9a3ee6bbf78c5599a3a6655a87ef467fc79c6d413e1c08`，
    `git bundle verify`确认完整history；
  - config SHA256=`1a4792ba3ebd81e9019506ff7b95c1eec6eac764fa222626024cdea96386e504`；
  - ViT-B权重SHA256=`80ecf9dd5e3a58895e959af554c5666c4e7b4da4410de4f1f2b0025e93435d8c`；
  - output=`log/occluded_duke/exp381_vit_d0_s1234`，main PID=`841096`。
- repo初始化首次因bundle已含`pretrained/`目录而在创建目录处安全停止；output尚不存在、未启动训练。
  首次启动命令又被进程表达式扫描当前shell中的启动文本而安全拦截；确认output仍不存在、无训练进程、
  GPU空闲后，改为按可执行名筛选真实Python进程并重跑全部门禁后fresh启动，不属于重复或续训。
- 启动后exact HEAD/config/tracked source clean持续通过；唯一main+8 workers，GPU约
  `6368 MiB/86%`。e1已自然结束并进入e2：e1末次loss=`13.479`、acc=`0.005`、pose loss=
  `2.882`、anchor confidence=`0.619`、sigma mean=`0.250`，field/teacher统计有限，joint
  permutation关闭，shift/log-scale=`0/0`；严格异常=`0`。
- 当前判断：**ViT-D0继续自然跑满e120**。每次完整eval现场计算相对同epochB0的四项显式差值；
  D0终审与GPU空闲前不启动HT0。

## 2026-07-17 ViT-D0 e10/e20与终端PSG边界审计

- 远端时间=`2026-07-17T12:39:49+00:00`；exact HEAD仍为
  `caf97468797968ea50959ad859a5ea45516d0655`，tracked source diff为空。main PID=`841096`
  持续运行，唯一main+8 workers，GPU=`6348 MiB/90%`、计算进程显存=`6340 MiB`。
- 完整评估及相对同epoch B0差值（固定顺序mAP/R1/R5/R10）：
  - e10=`42.8/51.3/70.5/76.2`，相对B0 e10=`+2.0/+2.8/+2.8/+1.9`；
  - e20=`47.0/54.8/73.3/79.4`，相对B0 e20=`+0.8/+2.3/+1.4/+1.6`。
  两个早期点均为正，但不据此裁决，继续固定e120。
- e10→e20全state有限；可训练anchor=`26/26 changed`（max=`0.270314127`，L2=
  `3.713222020`），residual-OFF geometry adapter=`0/6 changed`；ViT state=`150/152 changed`
  （max=`0.118410826`，L2=`13.814733241`）。
- PSG逐模块审计发现设计边界：`g3_b9/g3_b10`可训练参数=`8/8 changed`；但PSG是在每个block
  **之后**调制patch tokens，`g3_b11`位于最后一次CLS–patch交互之后。其最终零初始化投影
  `encoder.2.weight/bias`在e10→e20保持逐位不变，因而不能影响最终CLS descriptor；前两项非零
  encoder参数的变化只来自SGD weight decay。生产前门禁只检查了PSG集合梯度，未逐consumer捕获此点。
- 该边界不会制造D0正增益：当前D0实际是block8 anchor→两个有效G3 consumer（post-block9/10）；
  terminal post-block11 consumer是无效冗余。它在D0/HT0间完全共享，因此不混淆未来HT0−D0的G2
  层级增量；但最终论文/设计描述不得声称block11 PSG有效，ViT证据按实际两个有效G3 consumer解释。
- 检查时e25已完成、e26运行中；handoff后student fraction=`1.000`，pose/anchor/field/sigma持续有限，
  shift/log-scale=`0/0`，严格NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow=`0`。
- 当前判断：**继续D0到e120，不改运行中代码/config**。后续每个checkpoint继续分别审计active PSG与
  terminal dead PSG；D0终审前不启动HT0。

## 2026-07-17 ViT-D0 e30–e50与e57运行检查

- 远端时间=`2026-07-17T12:52:43+00:00`；exact HEAD/config SHA仍分别为
  `caf97468797968ea50959ad859a5ea45516d0655`、
  `1a4792ba3ebd81e9019506ff7b95c1eec6eac764fa222626024cdea96386e504`，tracked source diff为空。
- 新增完整评估及相对同epoch B0差值（固定顺序mAP/R1/R5/R10）：
  - e30=`47.5/55.6/72.8/79.9`，差值=`+0.4/+0.9/+0.1/+1.4`；
  - e40=`51.5/60.0/77.5/83.4`，差值=`+1.7/+3.0/+3.4/+3.2`；
  - e50=`52.0/59.4/76.1/82.2`，差值=`+0.6/+1.5/+0.4/+0.4`。
  e10–e50五个评估的mAP均高于同epoch B0，但幅度波动，继续到固定e120，不提前裁决。
- e10→e30/e40/e50逐checkpoint全state有限：anchor均=`26/26 changed`，e50 max/L2=
  `0.307048172/7.718091427`；active `g3_b9/g3_b10` PSG均=`8/8 changed`，e50 max/L2=
  `0.112053797/0.945388078`；ViT均=`150/152 changed`，e50 max/L2=
  `0.345098972/43.615712128`。
- residual-OFF geometry adapter在e20/e30/e40/e50均=`0/6 changed`；terminal `g3_b11`最终投影
  在所有checkpoint均=`0/2 changed`、max/L2=`0/0`，持续验证其无CLS下游路径的解释边界。
- main PID=`841096`持续运行，唯一main+8 workers，GPU=`6352 MiB/84%`、计算进程显存=
  `6344 MiB`。检查时e57运行中；student fraction=`1.000`，pose/anchor/field/sigma有限，
  shift/log-scale=`0/0`，严格异常=`0`。
- 当前判断：**继续D0到e120**。不改运行中代码/config，不以e40较大正差或e50回落作单点裁决。

## 2026-07-17 ViT-D0 e60–e90与e92运行检查

- 远端时间=`2026-07-17T13:07:38+00:00`；exact HEAD/config SHA与tracked source clean持续通过。
- 新增完整评估及相对同epoch B0差值（固定顺序mAP/R1/R5/R10）：
  - e60=`52.0/59.1/76.0/81.9`，差值=`+1.1/+2.3/-0.3/+0.6`；
  - e70=`53.7/61.0/78.2/83.3`，差值=`+1.8/+2.6/+2.4/+1.8`；
  - e80=`53.9/60.5/77.8/83.1`，差值=`+1.8/+1.5/+1.3/+1.2`；
  - e90=`54.8/61.8/79.0/84.4`，差值=`+2.0/+2.7/+2.1/+1.9`。
  除e60 R5外，中后期四点整体稳定为正；仍固定使用e120 final，不以e90代替。
- e10→e60/e70/e80/e90逐checkpoint全state有限：anchor均=`26/26 changed`，e90 max/L2=
  `0.324131995/9.104620436`；active `g3_b9/g3_b10` PSG均=`8/8 changed`，e90 max/L2=
  `0.086963944/1.051695377`；ViT均=`150/152 changed`，e90 max/L2=
  `0.496888399/61.525851999`。
- residual-OFF geometry adapter持续=`0/6 changed`；terminal `g3_b11`最终投影在e60–e90继续
  `0/2 changed`、max/L2=`0/0`，没有把weight decay误记为有效CLS consumer学习。
- main PID=`841096`持续运行，唯一main+8 workers，GPU=`6352 MiB/84%`、计算进程显存=
  `6344 MiB`。检查时e92运行中；student fraction=`1.000`，pose/anchor/field/sigma有限，
  shift/log-scale=`0/0`，严格异常=`0`。
- 当前判断：**继续自然跑满e120**。D0终审与GPU空闲前不启动HT0。

## 2026-07-17 ViT-D0 final审计

- e100/e110/final依次为`54.7/61.2/78.7/84.0`、`55.0/61.7/79.0/84.0`、
  `54.9/61.4/78.9/84.0`；相对B0 final=`+2.0/+1.9/+1.8/+2.0`。固定使用e120，
  不以e110较高单点替代final。
- 原main PID=`841096`及8个workers自然退出；GPU=`2 MiB/0%`且无计算进程。exact HEAD/config
  SHA与tracked source clean持续通过，12个10-epoch checkpoint齐全，严格异常=`0`。
- final checkpoint/runner/train SHA256依次为：
  - `d7219825ad7fa3dd319212871f5b1c29f7f7609853c27a28ac37a636bd3f7f90`；
  - `6d5538591dff5f990889dbf8a48f5eaaa2f0bdf73d356361ee88d196b6b2399b`；
  - `4d038152da19eeca18d6771d3e1f6c4e534a1401a59c81122abd1fd8cd1219bf`。
- e10→e100/e110/e120全state有限；e120 anchor=`26/26 changed`（max/L2=
  `0.325442433/9.216743176`），active `g3_b9/g3_b10` PSG=`8/8 changed`（max/L2=
  `0.089638643/1.063139880`），ViT=`150/152 changed`（max/L2=
  `0.512174368/63.393503123`）。geometry adapter持续`0/6`，terminal `g3_b11`最终投影持续
  `0/2`、max/L2=`0/0`。
- final checkpoint strict load通过：202个state tensor全部有限、missing/unexpected=`0/0`；真实
  batch64中的correct/shuffle/None/exploding external pose descriptor逐位一致，shape=`(2,768)`，
  descriptor SHA256=`fcfd926828b3673a6a4ebcb29980da859972b2c06acc2da4165b058132f9264c`，
  预训练加载仍为`150/152`。
- 结论边界：ViT上的完整训练期pose监督、测试期pose-free模块在单seed上相对同骨干B0四项均正；
  实际G3 consumer仅post-block9/10有效，不能把terminal post-block11写成贡献。逐层贡献仍必须由
  HT0−D0 final判定。

## 2026-07-17 fresh ViT-HT0启动

- D0终审完成且GPU空闲后，从同一完整history bundle建立独立生产repo：
  - repo=`/home/afr/SOLIDER-REID-exp381-ht0-caf9746`；
  - exact execution commit=`caf97468797968ea50959ad859a5ea45516d0655`；
  - bundle SHA256=`d9c27502ddecf290fc9a3ee6bbf78c5599a3a6655a87ef467fc79c6d413e1c08`；
  - config SHA256=`c54a414f9924701907d410e8d4819cb8d421c793d4f0b63ac7f0fa611cd4d2ff`；
  - ViT-B权重SHA256=`80ecf9dd5e3a58895e959af554c5666c4e7b4da4410de4f1f2b0025e93435d8c`；
  - output=`log/occluded_duke/exp381_vit_ht0_s1234`，main PID=`875359`。
- 首次repo门禁因`git bundle verify`需要现有repo上下文而安全停止，未创建repo/output或训练；改从
  已终审D0 repo验证完整history后重跑全部门禁并fresh启动，不属于重复或续训。
- 启动前output不存在、无训练进程、GPU空闲、tracked source clean；启动后唯一main+8 workers，
  GPU约`6616 MiB/84%`。e1已自然结束并进入e2：末次loss=`13.484`、acc=`0.006`、pose loss=
  `2.890`；两级field/anchor/teacher统计有限，stage2 refinement active=`1.000`，strict异常=`0`。
- 当前判断：**ViT-HT0继续自然跑满e120**。每次完整eval现场计算相对同epoch D0四项显式差值；
  G2 active PSG、G3 active PSG与terminal `g3_b11`边界分别审计，不改运行中代码/config。

## 2026-07-17 ViT-HT0 e10–e30与e33运行检查

- 远端时间=`2026-07-17T13:39:37+00:00`；exact HEAD/config SHA仍分别为
  `caf97468797968ea50959ad859a5ea45516d0655`、
  `c54a414f9924701907d410e8d4819cb8d421c793d4f0b63ac7f0fa611cd4d2ff`，tracked source diff为空。
- 完整评估及相对同epoch D0差值（固定顺序mAP/R1/R5/R10）：
  - e10=`43.0/50.9/70.2/76.4`，相对D0 e10=`+0.2/-0.4/-0.3/+0.2`；
  - e20=`47.6/55.6/73.0/80.0`，相对D0 e20=`+0.6/+0.8/-0.3/+0.6`；
  - e30=`49.5/56.7/75.5/81.1`，相对D0 e30=`+2.0/+1.1/+2.7/+1.2`。
  三个早期点由混合差值转为e30四项正差，但轨迹仍不足以裁决，固定继续到e120。
- e10→e20/e30 checkpoint的214个state tensor全部有限。e20/e30逐参数结果依次为：shared
  stage projections=`6/6、6/6 changed`（e30 max/L2=`0.107237473/3.196516296`）；shared
  anchor decoder=`26/26、26/26`（`0.200842038/3.543007503`）；G2 active PSG=
  `12/12、12/12`（`0.078331150/0.991082089`）；G3 active `g3_b9/g3_b10` PSG=
  `8/8、8/8`（`0.100984022/0.717457523`）；ViT=`150/152、150/152`
  （`0.185335636/25.556535804`）。terminal `g3_b11`最终投影持续=`0/2`、max/L2=`0/0`，
  与其对CLS descriptor无下游路径的边界一致。
- main PID=`875359`持续运行，唯一main+8 workers，GPU=`6616 MiB/84%`、计算进程显存=
  `6608 MiB`。检查时e33运行中；handoff后两级student fraction=`1.000`，两级anchor/field/teacher
  统计有限，stage2 refinement active=`1.000`，shift/log-scale=`0/0`，严格NaN/Inf/Traceback/
  RuntimeError/OOM/nonfinite/overflow=`0`。
- 当前判断：**继续ViT-HT0自然跑满e120**。不以e30单点裁决，不改运行中代码/config。

## 2026-07-17 ViT-HT0 e40–e60与e62运行检查

- 远端时间=`2026-07-17T13:52:38+00:00`；exact HEAD/config SHA与tracked source clean持续通过。
- 新增完整评估及相对同epoch D0差值（固定顺序mAP/R1/R5/R10）：
  - e40=`51.1/58.6/76.6/82.6`，差值=`-0.4/-1.4/-0.9/-0.8`；
  - e50=`52.2/59.8/77.0/82.0`，差值=`+0.2/+0.4/+0.9/-0.2`；
  - e60=`51.7/58.3/75.2/82.2`，差值=`-0.3/-0.8/-0.8/+0.3`。
  e30四项正差没有在e40–e60持续，当前轨迹为混合且多数点偏负；仍不作中途性能裁决，固定继续e120。
- e10→e40/e50/e60 checkpoint的214个state tensor全部有限。三点逐参数均为：stage projections=
  `6/6 changed`、shared anchor decoder=`26/26`、G2 active PSG=`12/12`、G3 active
  `g3_b9/g3_b10` PSG=`8/8`、ViT=`150/152`。e60各组max/L2依次为
  `0.167925268/4.628619873`、`0.307758301/5.336817390`、
  `0.102504440/1.248329378`、`0.086189315/0.922843660`、
  `0.403277636/50.329928179`；terminal `g3_b11`最终投影持续=`0/2`、max/L2=`0/0`。
- main PID=`875359`持续运行。首个快照恰逢epoch切换、worker池重建，只捕获4个子进程；随即在训练段
  复核为唯一main+8 workers，非worker丢失或重复训练。GPU=`6626 MiB/89%`、计算进程显存=
  `6618 MiB`。检查时e62完成；两级student fraction=`1.000`，anchor/field/teacher统计有限，stage2
  refinement active=`1.000`，shift/log-scale仍为`0/0`，严格NaN/Inf/Traceback/RuntimeError/OOM/
  nonfinite/overflow=`0`。
- 当前判断：**继续ViT-HT0自然跑满e120**。不因中期多数负差早停，不改运行中代码/config。

## 2026-07-17 ViT-HT0 e70–e90与e97运行检查

- 远端时间=`2026-07-17T14:07:38+00:00`；exact HEAD/config SHA与tracked source clean持续通过。
- 新增完整评估及相对同epoch D0差值（固定顺序mAP/R1/R5/R10）：
  - e70=`53.3/59.7/77.5/83.3`，差值=`-0.4/-1.3/-0.7/+0.0`；
  - e80=`53.5/59.5/77.5/82.7`，差值=`-0.4/-1.0/-0.3/-0.4`；
  - e90=`54.4/60.5/78.9/84.5`，差值=`-0.4/-1.3/-0.1/+0.1`。
  e70–e90的mAP与R1稳定低于D0，R5/R10接近持平；这是多点中后期证据，但仍等固定e120 final定性。
- e10→e70/e80/e90 checkpoint的214个state tensor全部有限。三点逐参数均为：stage projections=
  `6/6 changed`、shared anchor decoder=`26/26`、G2 active PSG=`12/12`、G3 active
  `g3_b9/g3_b10` PSG=`8/8`、ViT=`150/152`。e90各组max/L2依次为
  `0.184057698/5.014735627`、`0.321608603/5.916263710`、
  `0.104499377/1.322508551`、`0.072575092/0.985597276`、
  `0.497671604/61.513569492`；terminal `g3_b11`最终投影持续=`0/2`、max/L2=`0/0`。
- main PID=`875359`持续运行，唯一main+8 workers，GPU=`6614 MiB/84%`、计算进程显存=
  `6606 MiB`。检查时e96完成并进入e97；两级student fraction=`1.000`，anchor/field/teacher统计有限，
  stage2 refinement active=`1.000`，shift/log-scale=`0/0`，严格NaN/Inf/Traceback/RuntimeError/
  OOM/nonfinite/overflow=`0`。
- 当前判断：**继续ViT-HT0自然跑满e120**。e100/e110/e120均按计划记录，不以当前负差提前停止。

## 2026-07-17 ViT-HT0 final审计

- e100/e110/final依次为`54.3/60.2/78.1/83.8`、`54.6/60.6/78.3/84.0`、
  `54.6/60.6/78.4/84.1`；相对同epoch D0依次为`-0.4/-1.0/-0.6/-0.2`、
  `-0.4/-1.1/-0.7/+0.0`、`-0.3/-0.8/-0.5/+0.1`。固定使用e120，不以任何中间点替代final。
- 原main PID=`875359`及8个workers自然退出；GPU=`2 MiB/0%`且无计算进程。exact HEAD/config SHA
  与tracked source clean持续通过，12个10-epoch checkpoint齐全，严格异常=`0`。
- final checkpoint/runner/train SHA256依次为：
  - `ca986e62f3855265c2def708ce42f1f486fa3c8055c6a4e8ace6e2a0df5421bc`；
  - `6abcb124ba3837baf464949495593524f6bd7917afd3134e739000dd2310f7f4`；
  - `730e4831e33cc58263e5416aa25bb358b6916422a28c273639c0acfc7a617489`。
- e10→e20至e120的全部checkpoint均为214个有限state tensor。e120相对e10：stage projections=
  `6/6 changed`（max/L2=`0.185373053/5.066775772`），shared anchor decoder=`26/26`
  （`0.323903680/6.005096656`），G2 active PSG=`12/12`
  （`0.105418034/1.332784856`），G3 active `g3_b9/g3_b10` PSG=`8/8`
  （`0.070487350/0.996421206`），ViT=`150/152`
  （`0.512967348/63.382235401`）。terminal `g3_b11`最终投影全轨迹保持=`0/2`、max/L2=`0/0`。
- final checkpoint在原生PyTorch `1.13.1+cu117`下strict load通过：214个state tensor全部有限，
  missing/unexpected=`0/0`，ViT预训练加载=`150/152`。真实batch中的correct/shuffle/None/
  exploding external pose descriptor逐位一致，shape=`(2,768)`，descriptor SHA256=
  `8bf78bfcb61b407ffb4436ae4d468b2e8435ed896b204ba30203a3fc3e962ef2`。
- final结论：ViT-D0相对B0=`+2.0/+1.9/+1.8/+2.0`，支持完整训练期姿态监督、测试期
  RGB-only的`anchor+PSG`原子方法；ViT-HT0相对D0=`-0.3/-0.8/-0.5/+0.1`，不支持当前四等分
  ViT逐层扩展提供额外稳定收益。结合Swin HT0−D0 mAP=`-0.1`和ResNet=`+0.8`，逐层机制只能记为
  backbone-conditional探索，不能升为跨架构核心贡献。terminal post-block11 PSG无CLS下游路径的边界
  继续保留，但其在D0/HT0中共享，不混淆HT0−D0的G2增量。
