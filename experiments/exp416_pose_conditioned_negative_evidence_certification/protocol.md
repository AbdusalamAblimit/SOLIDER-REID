# exp416 PC-NEC fuel audit执行协议

## 1. 当前授权边界

当前只允许一次无训练fuel audit。禁止创建optimizer、正式训练`OUTPUT_DIR`、PK64合同、checkpoint或e120进程。
fuel任一主门失败即永久记：

`PC-NEC FUEL NO-GO / TRAINING NO-START / NO CANDIDATE`

禁止在结果后修改visibility、crop、CLIP层、top-K、聚合、lambda、control或统计门重跑。

## 2. 固定只读输入

- official data root：`/mnt1/afrdata`；
- official train：`Occluded_Duke/bounding_box_train`，固定15,618张；
- frozen pose：`/mnt1/afrderived/exp386_occluded_duke_vitpose_huge_train`；
- pose manifest SHA256：
  `cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8`；
- sealed OpenCLIP ViT-L/14：
  `/home/afr/reid-clean/weights/exp401_clip_l14_openclip_9ce2e8a8.safetensors`；
- CLIP SHA256：
  `9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e`；
- sealed D0 checkpoint：
  `/home/afr/SOLIDER-REID-exp387-d0-0d1822a/log/occluded_duke/exp387_clean_swin_tiny_d0_s1234/transformer_120.pth`；
- D0 SHA256：
  `59017755d61370754aa2e852a487d8e242fcee8814685f77f5388ba3a430e069`；
- D0 config：`configs/occluded_duke/swin_tiny_tapf_d0.yml`；
- D0 config SHA256：
  `510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b`；
- 唯一解释器：`/usr/local/anaconda3/envs/mmpose-abu/bin/python`。

`/mnt1/afrdata`与`/mnt1/afrderived`全程只读。所有新输出只能位于`/home/afr`。

## 3. 四个物理阶段

### 3.1 D0-only candidate bank

脚本：`build_candidate_manifest.py`。

fresh namespace：
`/home/afr/reid-clean/assets/exp416-pcnec-bank-v1`。

该进程手工枚举`bounding_box_train/*.jpg`，不得实例化会同时检查或读取query/gallery的`OccludedDuke`。在输出
bank前不得导入`PoseTargetStore`或`open_clip`。固定流程：

1. 按official train相对路径升序建立raw PID、连续train PID和zero-based camera；
2. sealed D0只读推理全部15,618张canonical `384×128` RGB；
3. query按固定salt path hash排序；
4. 仅保留至少一个跨相机同PID真的query；
5. 每query保留全部跨相机同PID真匹配和D0最近20个跨PID impostor；
6. query内按`D0 distance → candidate relative path`排序；
7. 原子写出路径、PID、camera、RGB SHA、D0 global、pair row、offset、label与distance。

任何pose/CLIP导入、路径逃逸、重复图、非15,618样本、D0 strict-load失败、非有限feature、候选重构不exact都使
stage INVALID。

### 3.2 pose/RGB-only geometry census

脚本：`geometry_census.py`。

fresh namespace：
`/home/afr/reid-clean/assets/exp416-pcnec-geometry-v1`。

该进程只读已封存bank、原始RGB尺寸/哈希与frozen pose；禁止导入OpenCLIP。机械定义在census前冻结：

- 五槽沿用`head / upper_torso_arms / lower_torso / upper_legs / lower_legs_feet`；
- joint必须坐标有效且ViTPose score `>=0.30`；
- 一个槽至少有2个这样的joint才`available`；
- instance center为该槽有效joint的归一化坐标算术均值；
- 每槽crop高宽为全train有效样本的归一化joint span
  `q90 + 上下/左右各5% canvas padding`；
- pixel高宽向上取整，至少`16×16`，最大不超过`384×128`；
- canonical center为全train有效instance center的逐维中位数；
- center只clamp到使完整矩形落在canvas内的最近位置，不padding、不变面积；
- correct和canonical逐图逐槽的height、width、availability、输出shape必须exact。

几何硬门在CLIP前固定为：

- fixed query中至少80%有至少一个candidate共享可见槽；
- 每槽至少100,000个fixed pair共同可见；
- 每槽至少覆盖300个不同query raw PID。

未过几何门时直接fuel NO-GO，不加载CLIP，不修改门。

### 3.3 frozen feature/energy cache

脚本：`build_fuel_cache.py`。

fresh namespace：
`/home/afr/reid-clean/assets/exp416-pcnec-fuel-v1`。

该stage在GPU空闲、formal tracked worktree/index均为0、前两stage SHA exact后执行。D0和CLIP严格串行加载：

1. D0阶段读取同一canonical RGB，验证global descriptor与bank exact，并从正式eval返回的
   `featmaps[-1]`按area-downsample fractional rectangle mask池化student-part；
2. 释放D0后才加载CLIP；
3. CLIP分别编码instance rectangle、canonical rectangle和whole-image letterbox；
4. raw-color为固定CIELAB `8×8×8`概率直方图，距离为total variation；
5. 对同一sealed pair row计算correct、raw-color、student-part、canonical CLIP、canonical raw、
   common-slot cyclic shuffle、wrong-RGB、global CLIP和D0九臂能量；
6. wrong-RGB donor必须五槽全available、camera与candidate相同、PID同时不同于query和candidate，并按
   `query path || candidate path || slot`固定SHA选择；空pool固定`E=0`并记录invalid；
7. 无共同槽pair统一`UNDECIDED/E=0`，不删pair、不换candidate。

### 3.4 OOF统计与唯一裁决

脚本：`fuel_audit.py`。

fresh namespace：
`/home/afr/reid-clean/assets/exp416-pcnec-audit-v1`。

该stage只读前三阶段封存资产，不使用CUDA。固定使用：

- query内D0与arm energy经验mid-rank；
- lambda grid=`[0,.25,.5,.75,1]`；
- raw PID hash五折OOF，训练折按mAP、R1、最小lambda依次tie-break；
- query内metric后先query PID内等权，再PID间等权；
- AUROC/AUPRC中impostor为positive；
- 每主指标分别按预注册control顺序确定最强control；
- PID cluster PCG64、base seed=`4161234`、10,000次、线性5%单侧下界；
- bootstrap前冻结OOF score、lambda、最强control与每query metric。

## 4. 唯一GO门

以下条件必须全部成立：

1. correct AUROC/AUPRC相对各自最强非D0 control均至少`+0.03`；
2. correct combined mAP/R1相对D0均至少`+1.0/+1.0`个百分点；
3. correct combined mAP/R1相对各自最强control均至少`+0.5/+0.5`个百分点；
4. 六个核心paired PID-bootstrap单侧95%下界全部`>0`；
5. geometry的80% query、每槽100,000 pair、每槽300 query PID门全部通过；
6. correct四项主指标分别严格胜七个非D0 control；
7. bank row、pair count、offset、SHA与各臂完整率exact。

fuel GO只允许另立future certificate设计与真实PK64复审，不自动授权训练。

## 5. 一次性与异常分类

- 每个namespace必须fresh，禁止覆盖；
- 进程消失先分为自然完成、程序异常、基础设施中断或用户终止；
- 科学NO-GO不重跑；
- 只有在Python未启动/未创建namespace/未读模型资产时发生的纯launcher基础设施错误，才能保留失败日志后重新发起；
- 所有不利row、UNDECIDED、donor-invalid、coverage与control必须完整保留；
- 任一阶段完成均记录source HEAD、脚本SHA、输入/输出SHA、严格异常计数与GPU终态。
