# exp392 监控记录

## 当前状态

- `PHASE 0A SEALED / PHASE 0B NAIVE TEACHER SEALED-NO-GO / B2-SI SEALED-PASS /
  PHASE 0C SEMANTIC C0 SEALED-NO-GO / PHASE 0D FROZEN ATTRIBUTION SEALED`；
- exp390与exp391均已封板，禁止重启、续训或把本实验记为exp391 Phase B/C；
- 当前GPU任务：无；Semantic C0已自然跑满e120并完成只读终审，GPU=`2 MiB / 0%`；
- final=`56.9/67.1/80.6/85.0`，相对clean D0=`−0.7/−0.6/−0.2/+0.4`，因此当前
  teacher/readout/router bundled组合未超过D0；
- 唯一checkpoint、strict finite、teacher隔离、RGB-only exact、两consumer可达与NULL identity均PASS；
  该结果不永久否定CLIP–TAPF，也不授权semantic multi-stage。

## 已确认边界

1. current PSG是field-only自由`17→32`空间门控，结构上不要求joint-channel语义可辨识；
2. exp391 H2-M相对D0为`−0.4 mAP`，但early-bypass仍有`+0.141 mAP`，说明route可达而纯结构
   topology不足；
3. RegionCLIP、π-VL、PAFormer、MUVA、ProFD、ALADIN等已覆盖CLIP局部语义、pose-aware part、
   multi-level guidance或inference-free KD的主要构件；
4. 可争对象必须是counterfactually identifiable executable anatomical mediator，而不是“首次CLIP+pose”；
5. exp391只封板无语义校准的纯结构链；semantic single-stage通过后允许重新验证semantic multi-stage。

## 2026-07-18 Phase 0 协议冻结

新增 `phase0_protocol.md`，当前仍为文档审查，不执行脚本、不占GPU。已冻结：

1. **Phase 0A clean D0 内部field seam**：clean runtime不是旧TAPF tuple；干预点必须是
   `model.base.tapf.anchor`输出dict中的`field`（或等价的`prepare()`返回dict），确保替换发生在
   student field形成后、两个PSG消费前。预注册channel-cycle、left/right channel swap、
   confidence permutation、matched-wrong、spatial-constant、zero与逐consumer/all bypass；外部
   correct/shuffle/None/exploding仍需exact。
2. **Phase 0B 双编码teacher门禁**：teacher RGB必须与student复用同一次resize/flip/pad/crop，主teacher
   读取RandomErasing前RGB，并用post-erasing作为clean-view KD混淆控制。square-stretch与
   aspect-letterbox只做预注册teacher-only比较，mask必须使用完全相同的RGB→CLIP grid变换。
3. **五类coarse ontology**：head/torso/arms/upper-legs/lower-legs；固定incidence、limb segments与
   prompt ensemble。主teacher用pose-region池化frozen CLIP patch tokens，再与全部frozen text
   prototypes形成sample-specific分布；不允许text-only常量冒充双编码teacher。
4. **强反事实与kill-switch**：wrong RGB/mask/text、channel shuffle、uniform/fixed bands、
   text-only、image-only cluster、flip equivariance、样本方差/JSD/effective-rank和遮挡分组均为
   必报项。任何关键输入不敏感、分布近常量或双编码不优于单编码控制时，当前teacher定义NO-GO。
5. **历史exp356边界**：已只读定位其ViT-L/14末层patch hook、强制eval、224方形拉伸和固定
   top-5/mid-6/bottom-5池化。历史pose-mask 57.1、random-mask 57.3只封板“固定水平条带completion
   + pose选mask”，没有实现pose-mask池化CLIP tokens或可执行语义中介，不能等价否定exp392。

协议同时记录未来Phase 0C的NULL identity、梯度所有权、config-off exact、semantic mismatch和
generic adapter强控制；Phase 0A/0B通过也只授权0C，不直接授权120-epoch训练。

## 2026-07-18 Phase 0A 开始执行

用户明确授权开始。当前只执行封板checkpoint的frozen audit，不修改exp387 repo/config/checkpoint，
不启动正式训练。

执行边界：

- exact repo：`/home/afr/SOLIDER-REID-exp387-d0-0d1822a`；
- execution HEAD：`0d1822a07dda8daac0210b68916035b1886d5d99`；
- config SHA256：`510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b`；
- checkpoint SHA256：`59017755d61370754aa2e852a487d8e242fcee8814685f77f5388ba3a430e069`；
- 审计脚本：`phase0a_internal_field_audit.py`，SHA256=
  `e3343e96b2b6b4202cabcdeedf5300da3704cff1bb60947babb17cddd9d21682`；
- 审计输出独立位于`/home/afr/reid-clean/audits/exp392_phase0a/`；
- 启动前GPU=`2 MiB / 0%`，无其他compute process。

真实首批256图route smoke=`EXP392_PHASE0A_ROUTE_SMOKE_PASS`，JSON SHA256=
`b54389b729e4e2c57ad34cc298a0ae54caa01d8a7e4329483444fdf4c0dfe8c7`。确认：

1. external correct/shuffle/None/exploding descriptor逐元素exact；
2. `zero_field == psg_bypass_all` descriptor逐元素exact；
3. descriptor最大绝对变化：channel-cycle `0.0423`、left/right swap `0.0619`、
   confidence permutation `0.0729`、matched-wrong `0.2884`、spatial-constant `0.3439`、
   PSG0 bypass `0.7897`、PSG1 bypass `3.0584`、all bypass/zero `3.9160`；
4. 所有预注册内部干预均非dead，hook seam与NULL identity可进入全验证集审计；
5. smoke只验证执行路径，不用于mAP裁决。

## 2026-07-18 Phase 0A 全验证集封板

唯一main PID=`1330438`自然退出，父shell同时退出；GPU恢复`2 MiB / 0%`，封板exp387 repo tracked
source保持clean。审计覆盖official Occ-Duke query+gallery共19,871图、query=2,210；correct复现
`57.558776/67.692308/80.769231/84.570136`。

| arm | mAP/R1/R5/R10 | 相对correct |
|---|---|---|
| channel-cycle | `57.582812/67.737557/80.723982/84.615385` | `+0.024036/+0.045249/−0.045249/+0.045249` |
| left/right channel swap | `57.541479/67.692308/80.723982/84.524887` | `−0.017296/+0.000000/−0.045249/−0.045249` |
| confidence permutation | `57.535337/67.692308/80.723982/84.434389` | `−0.023439/+0.000000/−0.045249/−0.135747` |
| matched-wrong field | `57.553827/67.828054/80.723982/84.570136` | `−0.004949/+0.135747/−0.045249/+0.000000` |
| spatial-constant | `57.905057/68.190045/81.266968/85.067873` | `+0.346281/+0.497738/+0.497738/+0.497738` |
| zero-field | `56.200175/66.018100/79.049774/83.303167` | `−1.358601/−1.674208/−1.719457/−1.266968` |
| PSG0 bypass | `56.883217/67.013575/79.909502/83.936652` | `−0.675558/−0.678733/−0.859729/−0.633484` |
| PSG1 bypass | `56.843903/66.696833/79.547511/83.891403` | `−0.714872/−0.995475/−1.221719/−0.678733` |
| all-PSG bypass | `56.200175/66.018100/79.049774/83.303167` | `−1.358601/−1.674208/−1.719457/−1.266968` |

paired bootstrap mAP 95% CI：channel-cycle=`[+0.0101,+0.0386]`、left/right=
`[−0.0306,−0.0049]`、matched-wrong=`[−0.0276,+0.0192]`、spatial-constant=
`[+0.2873,+0.4121]`、PSG0 bypass=`[−0.7620,−0.5943]`、PSG1 bypass=
`[−0.8097,−0.6254]`、all bypass=`[−1.5141,−1.2081]`。

终审：

1. correct start/end、external correct/shuffle/None/exploding逐元素exact；
2. zero-field与all-bypass descriptor逐元素exact；
3. 223-state SHA前后均为
   `c75e9d2e26f83255ae122a6c84b1717bc9474493453c7e04d95163da3cea96a3`；
4. 所有anchor/PSG hooks均移除，两个单bypass arm各调用78次，全bypass两个bank各调用78次；
5. donor map覆盖19,871图，same-camera=100%、different-PID=100%、无fixed point、不跨split；
6. runner严格NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow命中为0；
7. result/donor/runner SHA256分别为
   `e1a8ab0f3ab93939c9f7acbd05cf634cff47998c36f85719cea38545828b6511`、
   `17dfed2db93997f576120d680a671eafd2e73a88b8223affce13fc2b41d1b501`、
   `cd397826b020c3e7b59127dc014a6519c1b7957a63113e9204ae8f6235cd235a`。

裁决=`CONSUMER_EFFECTIVE_JOINT_SEMANTICS_NOT_IDENTIFIED`。all-bypass下降`1.359 mAP`证明PSG路径
确实有用；但channel-cycle与matched-wrong都远低于`0.3 mAP`语义门槛，且空间常量更高，说明当前
consumer主要利用低频/全局条件调制，而不是正确joint identity或精确geometry。这是exp392语义校准
问题成立的直接证据，不是对CLIP后新机制的否定。

## 下一步

进入Phase 0B coarse-region CLIP双编码teacher-only实现与门禁；保持同一时间一条4090任务，正式
训练继续`NO-START`。

## 2026-07-18 Phase 0B 实现与 smoke

Phase 0B 使用独立teacher-only脚本，不构建ReID模型、loss或optimizer。为避免误用远端旧且脏的
`/home/afr/SOLIDER-REID`，执行时只加载由本地当前树导出的三文件最小runtime，逐文件SHA固定为：

- `datasets/bases.py`：`03d231558f46264e4cff0c251b9b728ab4971232ed6c4bb7324ce1964f139c2c`；
- `datasets/occluded_duke.py`：`f0e7b25e75251643430b699d9c9969fae207c0a85c48855cd0404d61a4228f8e`；
- `datasets/pose_targets.py`：`42f6e35eff2ad572445143cb3ecc5b6a22d856facc4453b989411300dec22624`。

脚本另外把RGB donor与matched-mask donor分开：两者均为same-camera、different-PID、无fixed point；
mask donor使用五region的mask面积、纵向中心与pose confidence做最近邻匹配。15,618条matched map的
平均绝对差为area=`0.00757`、vertical center=`0.01440`、confidence=`0.01951`，避免把明显几何
分布差异误记成CLIP语义敏感性。

首次64图smoke在进入CLIP前被维度门禁拦截：384×128 RandomErasing mask尚未下采样到96×32
anatomical mask grid。失败runner与donor JSON原样保留，无GPU计算、无result JSON。修复只把
erasing mask用4×4 average pooling对齐到96×32；commit=`5277254bae1f9d18f1368ad4202aca7c9d223cc8`，
脚本SHA=`03b8f707bc6f189dd3de34505af82e63f7ee71bd23d70b6e9663aee318afcd70`。

第二次64图smoke=`CLIP_TEACHER_SMOKE_PASS`：

1. 两种geometry的RGB/mask synthetic alignment逐元素exact；
2. frozen ViT-L/14 hook contract均为raw `64×257×1024`、projected patch `64×256×768`；
3. 同一batched input重复forward逐元素exact，所有arms finite；
4. square-stretch / aspect-letterbox有效patch均非空，padding mask mass严格为0；
5. 进程与8 workers退出，GPU恢复`2 MiB / 0%`，严格异常扫描PASS；
6. result/donor/runner SHA分别为
   `24c2551db53a77f451f49692d58a2a02e33b957ba0de6913c995ed3c7c60d77d`、
   `27f31fa69ec223c4506218ce468b01a540882da70380ad85cd8449333c9d5a74`、
   `7d006d1254ee0b17984f4b8b2ff1b254900689d66ad3e4efdf99ddd84ce359b7`。

smoke中的correct macro top-1仅square=`4.38%`、letterbox=`3.44%`且margin为负，提示当前dense
CLIP patch teacher可能无法直接绑定五类ontology；但64图smoke预注册只裁结构契约，不以其挑版本或
下GO/NO-GO。下一步仍按协议跑完整15,618图teacher-only audit，以bootstrap、per-class confusion、
wrong RGB/mask/text、flip、confidence与erasing全门禁封板；无论结果如何都不直接授权正式训练。

## 2026-07-18 Phase 0B 全15,618图封板

唯一main PID=`1338776`携带8 workers自然退出；运行期GPU约`3496 MiB / 100%`，结束恢复
`2 MiB / 0%`。冻结ViT-L/14、text bank、pose artifact、isolated runtime与脚本SHA全程不变，
`open_clip=2.32.0 / torch=1.13.1`。两种geometry均通过RGB/mask synthetic exact、raw
`64×257×1024`→patch `64×256×768` token contract、repeat exact、finite、nonempty和padding
mass=`0`边界。

主语义门禁明确失败：

| geometry | correct macro top-1（bootstrap 95% CI） | expected margin（95% CI） | shuffle top-1 | wrong-text top-1 |
|---|---:|---:|---:|---:|
| square-stretch | `2.692% [2.583,2.801]` | `−0.11349 [−0.11383,−0.11312]` | `16.107%` | `29.996%` |
| aspect-letterbox | `4.637% [4.508,4.777]` | `−0.11099 [−0.11137,−0.11063]` | `15.511%` | `33.546%` |

20%随机水平下，correct不仅显著低于chance，而且两个geometry的correct-minus-counterfactual均为负：

- square：wrong RGB=`−2.741 top-1 / −0.02423 margin`、matched wrong mask=
  `−1.436 / −0.01651`、channel shuffle=`−13.415 / −0.03129`、wrong text=
  `−27.304 / −0.07901`个百分点/绝对margin；
- letterbox：wrong RGB=`−3.297 / −0.02241`、matched wrong mask=`−1.780 / −0.01473`、
  channel shuffle=`−10.875 / −0.02757`、wrong text=`−28.909 / −0.08247`。

失败不是常量或数值问题：五region centered effective rank约`2.58–3.64`，wrong-RGB q-JSD约
`0.0032–0.0033`，说明local feature确实随图像变化；但变化没有与预定义body-part text identity
对齐。更强的归因是image-only spherical K-means在同一local visual feature上达到square=`59.99%`、
letterbox=`52.77%` best-permutation region accuracy，远高于双编码correct top-1。这说明当前末层
dense patch feature含有明显解剖/位置结构，失败集中在**局部visual token与global text space的
cross-modal binding接口**，不是“CLIP图像特征完全没有人体结构”。

其余门禁同样不支持当前teacher：flip q-JSD虽低至`0.00026/0.00050`，top-1 consistency仅
`89.06%/84.43%`，低于95%；low-confidence entropy更高，但margin反而比high-confidence更不负，
confidence direction失败；synthetic erasing使entropy升高但expected margin反而改善，方向失败。
两种geometry均只通过`distribution_not_constant / wrong_rgb_sample_sensitive / flip_jsd / nonempty /
repeat_exact`，不通过top-1、positive margin、correct-vs-mask/shuffle/text、confidence、erasing和
flip top-1门禁。

裁决=`CURRENT_CLIP_TEACHER_NO_GO`，`phase0c_authorized=false`，禁止创建训练config或启动正式
semantic single-stage。该裁决只封板当前`naive last-block dense patch pooling + 5 coarse regions +
current prompt bank`，不永久否定CLIP语义校准或其后的balanced multi-stage。下一步先做只读代码与
机制归因，比较能恢复local-text alignment的冻结dense extraction（如MaskCLIP/SCLIP/ClearCLIP类
表示）与region-crop global embedding；另写单变量teacher-only协议后才可执行新审计。

全量result/donor/runner SHA256分别为：

- `af8e654565396f338a9a1b1f8ce5fe4d8178d551ec2767c500d230d477d7e6f8`；
- `27f31fa69ec223c4506218ce468b01a540882da70380ad85cd8449333c9d5a74`；
- `bcb588175a54ecb175d4c6a60efd71bfd0e8aa5a5bec032117abbed18cb28b02`。

sample manifest=`93a120d3c23cf547481a91de83dff58e7c38cdd37f97de4f3d8fefc75b98bfac`，
prompt SHA=`ae5db4cc4cd28a1aee2e88dcaf02702a26f412803081fbf3c212fd191e0fb07b`，
严格`NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow`扫描为0。

## 2026-07-18 Phase 0B 失败源只读诊断

用户要求先排除“简单任务结果差是实现写错”。当前完成四层parity：

1. **pose坐标无上下/xy错误**：抽查原图后，head→torso/arms→upper legs→lower legs的mean-y严格
   单调向下；同一mask生成的tight crop能恢复明显语义，进一步排除mask整体错位；
2. **OpenCLIP hook无偏差**：脚本hook末block的raw token，经官方`ln_post + visual.proj + L2Norm`
   后，与`VisionTransformer(output_tokens=True)`返回token走同一路径逐元素exact，最大绝对差`0`；
3. **预处理插值不是根因**：同一128图square路径，bilinear/bicubic correct top-1仅
   `5.469%→5.625%`，confusion结构基本不变；
4. **prompt/label顺序未循环写错**：保持同一prompt bank，把同一pose region改为tight crop并走CLIP
   真正受对比学习监督的global CLS，64/128图诊断macro top-1升至`44.688%`；head/torso/lower-leg
   分别`78.9%/54.7%/72.7%`。因此label bank与至少三类文本原型可正常对齐；arms仅`0.8%`、
   upper-leg`16.4%`说明ontology/prompt重叠仍是次要问题，但解释不了dense路径仅`3–5%`。

结论是**实现数学上复现了OpenCLIP官方patch tokens，但teacher选择了未被CLIP对比目标直接校准的
局部读出**。CLIP只监督final global CLS↔text；last-block patch token虽然包含位置/人体结构（image-only
cluster可达`52.8–60.0%`），其方向不等于body-part text轴，导致head/legs近似反向confusion。
这属于teacher接口设计错误，而不是tensor维度、hook、坐标或label index bug。

另做共享前23层、只在最后block让CLS按region mask重读patch的诊断：全1 mask与原CLIP CLS逐元素
exact；soft/hard region mask分别得到`20.0%/32.5%` top-1，证明沿受监督CLS readout可恢复部分
语义，但单个末block仍受既有global CLS residual支配，尚不足以过门禁。region tight-crop global
CLS当前最高`44.7%`，但每图五次完整CLIP forward不适合直接作为120-epoch online teacher。

因此下一步不是修补当前脚本后重跑同一定义，也不是直接训练。先把候选收敛到两类单变量teacher
接口：`pose-conditioned multi-block CLS readout`（共享早期trunk）与`region-crop global CLS`的可缓存/
成本边界；同时重构arms/upper-leg的互斥ontology。历史exp354的MaskCLIP value-only在另一归属任务上
已有小样本负证据，不能未经clean body-part门禁直接复活。

## 2026-07-18 Phase 0B2 问题重构开始

用户明确要求继续深度耦合CLIP与TAPF，不把Phase 0B当前teacher的NO-GO扩张为整个方向失败。已新增
`phase0b2_protocol.md`，当前保持`READ-ONLY / GPU NO-START`。

本轮纠偏不是换prompt救场，而是修正teacher职责：pose已经给出不可置换的anatomical slot identity，
CLIP不再重复猜五类body-part name；frozen image encoder只读取该slot当前RGB的视觉证据，frozen text
encoder把它投影到slot内visual-support/occlusion与appearance轴。未来这些量必须进入TAPF的
semantic expert gather-transform-scatter执行路径，禁止terminal auxiliary head或final descriptor KD。

新协议冻结两个readout：主候选为共享前20层、后4层pose-conditioned CLS的`PC-MBCLS`，理论block
计算量为单CLIP的`1.67x`；region-crop global CLS只作约`5x`成本的受监督路径参考。按
`ontology-only -> readout-only -> semantic-object-only`拆开，避免把接口变化、ontology变化和teacher
目标变化混为一个实验。

已启动三路独立只读审查：OpenCLIP后段attention实现、MaskCLIP/SCLIP/ClearCLIP/DenseCLIP近邻代码、
以及Phase 0B2反事实与kill-switch。审查完成前不实现脚本、不占4090；正式训练继续`NO-START`。

## 2026-07-18 Phase 0B2 三路独立审查闭合

三路审查均支持继续CLIP–TAPF深耦合，但要求先修teacher readout和审计边界。已据此修订
`phase0b2_protocol.md`：

1. 五个region不能在同一token sequence追加五个CLS；必须把block20输出沿batch复制为五个彼此独立、
   各只有一个官方CLS的sequence，再分别跑后4个block，否则region会互相attention且all-one parity
   不可能成立；
2. 原逐patch `eps=0.05`会使稀疏region的背景总先验质量接近前景，现改为总leak budget
   `rho=0.01`，并增加单patchquery/key方向测试；all-one本身不足以排除float mask方向错误；
3. 审计拆为`B2-O ontology-only -> B2-I readout-only -> B2-S semantic-object-only`；support首版从四级
   改为visible-vs-occluded二分类，只有三种合成遮挡均出现方向正确的单调响应才允许再拆等级；
4. appearance晚于support，并把dominant color与texture拆成两个分布；增加完整pose→增强→mask峰值、
   增强后donor matching、四个cycle、PID-cluster bootstrap、raw cosine margin、PID-disjoint cluster与
   cache geometry key门禁；
5. region-crop global CLS不是同信息条件下的理论上界，也不授权作为`5x` online teacher。

公开近邻复核显示MaskCLIP取最后attention的V、SCLIP使用QQ/KK self-self attention、ClearCLIP去除末层
residual/FFN干扰、DenseCLIP训练context decoder；它们支持“普通CLIP patch residual不是可靠局部文本
轴”，也说明dense CLIP readout本身已拥挤，不能作为创新主张。可争对象仍只能是pose固定slot
identity、CLIP提供slot内实例support/appearance、该state真实控制TAPF semantic expert，以及最终
descriptor上的反事实可辨识证据。

独立反事实审查曾报告现有`spherical_kmeans()`可能只做初始化；主agent逐行复核后确认其确实执行
20轮assignment和center update，因此不采纳该条错误报告。保留的真实问题是fit/eval使用同一全集，
B2将改为PID-disjoint fit/eval。正式训练继续`NO-START`，4090空闲。

## 2026-07-18 Phase 0B2 CPU/static 门禁

新增两个不读取ReID数据、不构建optimizer、不占GPU的契约脚本：

1. `phase0b2_static_audit.py`：本地uv环境CPU PASS，脚本SHA256=
   `afcfcc055d91b4d83bcc695357ffc42904383f293fd4d2b5d007257f9b247587`。五region partition非空
   像素sum误差`1.79e-7`，总背景leak=`0.009527<=0.01`，all-one prior严格0，zero严格invalid/NULL，
   attention mask只改CLS-query/patch-key，非CLS行exact，B×5 branch展开还原exact；完整synthetic
   resize/flip/pad/crop链中pose与RGB marker误差`0.5 px`，96×32 grid误差`0.0601`；
2. `phase0b2_openclip_contract.py`：在远端正式`open_clip=2.32.0/torch=1.13.1`、同一
   ViT-L/14 checkpoint上以`CUDA_VISIBLE_DEVICES=""`纯CPU PASS。官方24-block与手动20+4路径
   max error=`0`；五个独立all-one branch相对官方CLS max error=`1.34e-7`；sparse五region最小descriptor
   change=`0.05557`，repeat exact，zero mask严格invalid且输出全0，全部finite。脚本SHA256=
   `7206dc13bf69b5666b54169ae3333f838c48b16d0c963512e7c67d906354c2c7`，runner SHA256=
   `db1bc3a43953aef4363424ffc28899aa560805f26394180514ba001b9ead2f82`；耗时`4.84s`，CPU
   maxRSS=`3,654,036 KiB`。

CPU运行前后远端4090均为`2 MiB/0%`，无训练或审计计算进程。该结果只证明PC-MBCLS执行数学与
OpenCLIP官方路径兼容，不证明teacher语义有效；下一步仍先做B2-O互斥ontology/crop-reference小样本
CPU smoke，正式训练和GPU teacher-only全量均保持`NO-START`。

## 2026-07-18 Phase 0B2-O ontology-only CPU smoke 封板

B2-O保持Phase 0B原始五类part-name prompt、region-crop global CLS、RGB、geometry与CLIP不变，
只把原重叠region renderer替换为唯一joint owner、limb segment两端trim `15%`且保留幅度的
soft partition。审计脚本SHA256=
`8c482ea5ece56cea02122ad005481810b8caab4a82010fd06a5c1c4ce9d18f2f`。

首次8图CPU运行已完成40个crop和CLIP前向，只在`os.chdir(repo_root)`后解析相对
`__file__`时收尾失败；无result JSON、无GPU占用，失败runner保留且SHA256=
`12ae662de38543460788f8cb39c80c7f9fe4597539945e92de85d9907d893025`。修复仅在改变cwd前保存
absolute script path。

修正后8图v2完成：40 crops、coverage exact、finite、static contract全PASS，macro top-1=
`60.0%`；但overlap median/P95/max=`0.000658/0.270802/0.556638`，P95略超预注册`0.25`，
因此verdict=`B2_O_SMOKE_FAIL`。result/runner SHA256分别为
`96c4fdfe4ab1b7c185dc98e63aac2bd0bc8af7734e1eaa93cb2b2e852d66aba2`/
`bb31b19b3324bd8ea0ae4e95cf5d5805f858c65738eb57860c05b7dda6682ea9`。

按协议继续同一冻结定义128图CPU smoke，639 crops、wall=`2:30.60`、maxRSS=`3,702,624 KiB`；
coverage exact、finite、macro top-1=`52.93% [50.74,56.14]`，但overlap median/P95/max=
`0.000322/0.438040/0.997638`，明确不是8图随机波动。分类结果同时定位了独立的文本
混淆：head=`88.98%`、torso=`79.58%`、lower-leg=`75.94%`，而arms=`0.93%`、upper-leg=
`19.23%`；原`torso and upper body`与arms语义重叠不能与ontology修正绑定。result/runner SHA256
分别为`acd8e36821929623a96ed29f77c5d621f718513e54c6a17f86b1fb4fc1f6da2a`/
`7de7c5c52b1b2fc9086e68c6004eb9ac4fb17e33e5d3c29d42d3af026ea9340a`。远程4090始终为
`2 MiB/0%`，进程退出后无训练或审计进程。

裁决=`B2_O_SOFT_PARTITION_NO_GO`。不改overlap阈值、trim或sigma。下一步另开单变量`B2-O2`：
raw anatomy、证据幅度与其余teacher全不变，只以固定tie-break hard owner代替soft composition，使
teacher slot在像素级真正互斥。B2-O2通过后再独立做`B2-P prompt-only`；两者不同步修。
正式训练和GPU teacher-only全量继续`NO-START`。

## 2026-07-18 Phase 0B2-SC crop-support feasibility

B2-SC实现commit=`f32b7c9dc3d611cfc7b6ab41db1836ed6ac4d151`，脚本SHA256=
`692a3662d0de9613a6a1c573d2d86bfd7f40b3082f215d005aa4f8857869496a`，support prompt SHA256=
`a88a1405b629402b647b6075325b2821362bbcf89372466a27f9d4cfcee3af12`。hard-owner ontology、
crop-global CLS、RGB、geometry与CLIP固定，只把part-name目标换成每slot visible/occluded二分类；
原crop bbox固定，按row-major support pixels构造0/25/50/75三档嵌套CLIP-mean遮挡。

首次外层启动前HEAD检查发现隔离runtime不含`.git`后退出；脚本未启动、无runner/result/CLIP前向。
正式边界恢复为source commit记录加三个runtime文件逐SHA，与Phase 0B/B2-O一致。

8图CPU smoke完成40 valid-slot crops/160 variants，wall=`42.64s`、maxRSS=`3,703,436 KiB`；
coverage、finite、repeat exact、overlap level误差与严格递增全PASS。macro `q_visible`=
`0.50458/0.49077/0.48675/0.48448`，三档相邻差值全正；五slot的
`Spearman(overlap,-q_visible)`=`0.756/0.695/0.629/0.615/0.703`，0→75%下降=
`0.0263/0.0154/0.0175/0.0212/0.0201`，verdict=`B2_SC_SMOKE_PASS`。result/runner SHA256=
`783fa561ed0a37d14e14b9918ba8af44da063bbe34739291dab7990cc9fc2619`/
`ac3328430fe3ef0496e1bca1c9a9a3831e9f58956ca326c56b12b32c0068bd25`。

128图CPU复核完成639 valid-slot crops/2,556 variants，wall=`9:13.56`、maxRSS=
`3,706,396 KiB`；全部八项gate继续PASS。macro `q_visible`=
`0.49671/0.48281/0.48116/0.47970`，相邻下降=
`0.01390/0.00165/0.00146`。head/torso/arms/upper-leg/lower-leg的相关性=
`0.346/0.520/0.450/0.601/0.495`，0→75%下降PID-cluster mean及95% CI分别为：

- head `0.01433 [0.00814,0.02052]`；
- torso `0.01347 [0.01068,0.01602]`；
- arms `0.02376 [0.01911,0.02840]`；
- upper-leg `0.01642 [0.01462,0.01855]`；
- lower-leg `0.01865 [0.01680,0.02061]`。

每类CI均不跨0；arms是响应最强的slot，证明其B2-P失败来自part-name互斥任务，而不是CLIP不能读取
arms当前visual support。result/runner SHA256=
`d77b9b8ffd9b69401ccdcf341d39f317b094ca17b21991eb173dfebccb473734`/
`f2ddbf10bfa243e2f2a5da0fcbba17397cfee7f485d634184ffe1711e2a8b08c`。严格异常扫描0，进程退出，
4090全程`2 MiB/0%`。

裁决=`B2_SC_CROP_SUPPORT_SMOKE_PASS`。结果仍是CLIP-mean单一合成遮挡、128图/4 PID的零训练证据，
不能包装为完整teacher或训练增益。下一步B2-SI固定同一support任务，只把readout换成PC-MBCLS，并
要求target slot响应强于non-target slots与official global CLS；通过后才授权完整三材质反事实teacher
审计。正式训练继续`NO-START`。

## 2026-07-18 Phase 0B2-SI PC-MBCLS support readout

B2-SI实现commit=`b2df230`，脚本SHA256=
`c95dbed6c569a324c781d4ff382922179d81365f2fbe40b479027f4a8f345b29`；依赖的ontology/support/
PC-MBCLS contract脚本SHA256分别为
`b0d5ce6a53e94d09fa5d15c338392ea31437eee036299256c424ed30489028ca`/
`692a3662d0de9613a6a1c573d2d86bfd7f40b3082f215d005aa4f8857869496a`/
`7206dc13bf69b5666b54169ae3333f838c48b16d0c963512e7c67d906354c2c7`。support prompt SHA仍为
`a88a1405b629402b647b6075325b2821362bbcf89372466a27f9d4cfcee3af12`，没有改prompt、遮挡level、
CLIP或温度。full RGB使用aspect-letterbox，hard-owner mask nearest映射后再14x14 average pool；
target slot同时与同图non-target slots及official global CLS比较。

1图运行期contract完成20 full-image variants，NULL/repeat/finite/hard-mask/target-valid均PASS，五类
target下降均为正且macro target−global=`+0.04824`；仅lower-leg target−non-target=
`−0.00235`，故按预注册记`B2_SI_SMOKE_FAIL`，不改阈值且不按单图裁决。result/runner SHA256=
`77f43d03763bd9900d0085c4ba9dc4b40ab3ed8579f32c93f0e502df60fc596c`/
`15127af4fb8fa285e4a175346881dd2ba2107da0199d2204cb754765fa292327`。

同一冻结代码的8图CPU smoke完成40 valid targets/160 variants，wall=`1:16.24`、maxRSS=
`3,706,968 KiB`，全部12项gate PASS：

- macro target `q_visible`=`0.51943/0.49182/0.47904/0.47270`，三档相邻下降全正；
- head/torso/arms/upper-leg/lower-leg target 0→75下降=
  `0.08492/0.03081/0.04840/0.03799/0.03157`；
- 五类target−non-target下降=
  `+0.08474/+0.02363/+0.04630/+0.03736/+0.03094`；
- 五类target−global下降=
  `+0.07074/+0.02565/+0.04416/+0.03289/+0.02841`，macro=`+0.04037`；
- 五类Spearman=`0.808/0.521/0.666/0.536/0.621`；
- pixel product max=`0`、NULL/repeat/finite/target-valid exact。

verdict=`B2_SI_SMOKE_PASS`，result/runner SHA256=
`825248b8573ebacf7f3c46c7ef7325c3fd1c66329c1a2e9f01bc61204b734cf6`/
`6fd6c81ef8fcb3a6ef7a389eaef3ef70579dd8fc66a8cb3504d6b0c04cb3b21c`，严格异常0，4090=
`2 MiB/0%`。该结果首次同时支持slot内support语义与target-localized readout，但仍只有1 PID，
不作最终裁决。

同一冻结代码的128图CPU复核已自然完成：128图、639个valid target slots、2,556个full-image
variants，wall=`18:08.44`、maxRSS=`3,705,508 KiB`，全部12项预注册gate继续PASS。macro target
`q_visible`在0/25/50/75%遮挡下为
`0.51104/0.48925/0.47765/0.46990`，三档相邻下降=
`0.02178/0.01160/0.00775`，均严格为正。head/torso/arms/upper-leg/lower-leg分别得到：

- target Spearman=`0.685/0.482/0.641/0.665/0.678`；
- target 0→75%下降=`0.07253/0.02985/0.04758/0.03238/0.03000`，五类PID-cluster
  95% CI均不跨0；
- target−non-target下降=`+0.06919/+0.02273/+0.04528/+0.03202/+0.02879`，五类CI均严格为正；
- target−global下降=`+0.05849/+0.02485/+0.04043/+0.02768/+0.02705`，五类CI均严格为正，
  macro=`+0.03570`。

hard-owner pixel product max=`0`，NULL/repeat/finite/target-valid和level误差全部exact。result/runner
SHA256分别为
`63223862e87ba2d73919bca03fb019c90706b8d31ec1c7225decd71248bb9108`/
`22d154e82938acdf33cb4320759e17e7e0d0527fa2981684f8a1f981d07a76b0`；脚本SHA仍为
`c95dbed6c569a324c781d4ff382922179d81365f2fbe40b479027f4a8f345b29`。parent/child自然退出，
exit status=`0`，严格异常扫描0，4090=`2 MiB/0%`。

裁决=`B2_SI_PCMBCLS_SUPPORT_READOUT_PASS`。这证明PC-MBCLS不是只读取全图扰动，而能把特定
anatomical slot内的support变化局部化到对应readout；它只授权完整B2-S teacher-only反事实审计，
不授权Phase 0C、训练config或120-epoch正式训练。下一步先冻结三材质overlap/non-overlap、
wrong RGB/mask/text、flip、PID-cluster bootstrap及GPU效率协议，再通过static/CPU small smoke后才允许
唯一4090执行全train teacher审计。

## 2026-07-18 Phase 0B2-O2 hard-owner CPU smoke

B2-O2实现commit=`11a3303e7088acc9e62d259a741f058116732b92`，审计脚本SHA256=
`86d7349fc23c2fc3adb8c2727f42013582578c82156b63631efafed42fbda00e`。默认`soft`路径保留；
本臂显式`--partition-mode hard-owner`，唯一变化是
`M_c=min(sum raw,1)*1[c=argmax raw]`。远端OpenCLIP/PyTorch正式环境中的soft/hard双静态契约均
PASS；hard的pairwise product max严格`0`、sum support max=`1`、flip max error=
`8.34e-7`。

8图CPU smoke完成40 crops，overlap median/P95/max=`0/0/0`、macro top-1=`57.5%`、
coverage exact、finite，verdict=`B2_O2_SMOKE_PASS`；result/runner SHA256分别为
`7ee442dc603bff1696dfd9e2146ac3e023477254bcba1e26ede180bb2bcd77ea`/
`6c85be3a6cd98ef022cfde5fe640efa188c6c0a89677c964364452a950f246e8`。

同一冻结定义的128图CPU复核完成639 crops，wall=`2:24.77`、maxRSS=`3,700,532 KiB`；
overlap median/P95/max继续严格`0/0/0`，macro top-1=`51.56% [48.92,55.18]`，
coverage exact、finite，verdict=`B2_O2_SMOKE_PASS`。head/torso/lower-leg top-1=
`88.98/75.07/74.80%`，arms/upper-leg=`0.93/18.04%`；后二者没有随hard-owner改善，确认
prompt语义混淆与ontology是独立问题。result/runner SHA256分别为
`ce6d80f65f31749a227429ef6eda4efa9e0dcfcb6bfc4946d79b3262e3a4b8a9`/
`6eae2d981312e6fff2029b46500100c4ae038b817acf76e483ba37c50709ee98`。运行前后4090均为
`2 MiB/0%`，无残留进程。

裁决=`B2_O2_ONTOLOGY_SMOKE_PASS`，hard-owner ontology冻结。下一步只做一次`B2-P prompt-only`：
删除跨slot umbrella term并使用预注册互斥解剖词表；mask/crop/readout/CLIP均不动。正式训练与GPU
teacher-only全量仍为`NO-START`。

## 2026-07-18 Phase 0B2-P prompt-only 封板

B2-P实现commit=`ed2f9ce`，脚本SHA256=
`b0d5ce6a53e94d09fa5d15c338392ea31437eee036299256c424ed30489028ca`，prompt payload SHA256=
`a2db3121860210f8abc2184eabc8c501f87fd6a98d5b0759e32c92b795d7dbb2`。B2-O2 hard-owner、
crop、RGB、CLIP、四template与样本全不变，只把五个region phrase替换为预注册的互斥解剖词表。

8图CPU smoke完成40 crops，coverage/finite/exact-zero ontology均PASS，macro top-1=`57.5%`，
但arms=`0%`、upper-leg=`25%`且二者margin为负，verdict=`B2_P_SMOKE_FAIL`。未按1个PID提前裁决，
继续自然完成预注册128图。8图result/runner SHA256分别为
`e20fff30bad439cfb94c4935c6c3cdf341ce3425155bac4b898bea8aa616e73e`/
`55b8fc6fa0055752843d4c3f6f263178e74f510719c40333db9b38ba08978f2b`。

128图CPU结果：639 crops、wall=`2:30.14`、maxRSS=`3,708,888 KiB`，overlap继续
`0/0/0`、coverage exact、finite；macro top-1=`51.57% [47.30,55.83]`。分项为
head/torso/arms/upper-leg/lower-leg=`93.08/56.38/0.00/30.17/78.21%`，raw cosine margin=
`+0.02710/+0.00043/-0.02603/-0.00375/+0.01060`。upper-leg相对原prompt改善，证明prompt变量
真实生效；arms仍严格0且负margin，不能通过同义词消歧。verdict=`B2_P_SMOKE_FAIL`，
result/runner SHA256分别为
`8e18d141ffff182851e468b0b3aa16d9730dcd5bafd80d44d7c2e4367212d3fc`/
`60fc74c9ba96f9279b09e7028fb77065ec48df69d3bbe75cc10c504cc28ed319`。运行前后4090=
`2 MiB/0%`，无残留进程。

裁决=`B2_P_PART_NAME_DIAGNOSTIC_NO_GO`，禁止继续prompt搜索，也不把arms删除或合并。该裁决只
关闭“五类part-name作为readout gate”，不关闭原设计的slot-conditioned support teacher。下一步按
`B2-SC crop-support feasibility -> B2-SI support-readout`重新对齐真实语义对象：先保持crop-global
readout，只测三档嵌套局部遮挡是否让每slot的`q_visible`单调下降；通过后才把readout换成PC-MBCLS。
正式训练和GPU teacher-only全量继续`NO-START`。

## 2026-07-18 Phase 0B2-S connected-occluder static contract

B2-SI 128图PASS后，最初B2-S static草案虽通过数值契约，但独立反事实审查指出三个阻塞，故未运行
8图CLIP：

1. different-PID **同slot可见内容**不应被要求使`q_visible`下降，否则正确的support teacher也可能
   被假判负；它已移到appearance/binding，对support的第三材质改为different-PID wrong-slot occluder；
2. hash-randperm散点替换只证明local corruption sensitivity，不能证明occlusion；正式定义改为从
   target bbox固定一侧连续增长的25/50/75%连通矩形；
3. non-overlap必须在384×128 pre-image坐标排除target的24-pixel dilation，且8图target/donor不能在
   子集重算，必须先由全部15,618图pose-only map冻结。

据此重写`phase0b2_full_teacher_static_audit.py`。首轮独立终审又发现两处代码级门禁偏差与一处协议
歧义：overshoot误用了上一目标level而非立即前一strip、random texture按hard support而非协议冻结的
target bbox统计、前文仍残留未限定wrong-slot的CutMix措辞。三项均在运行8图前修正；同时预注册
矩形与五slot的几何泄漏报告，并要求逐样本逐level的target交集严格大于任一单个non-target交集。

修订后本地uv纯CPU static再次PASS，脚本SHA256=
`a9fc32a68a0dc13645e8e45a43fe84f0a5174bc7eb997c658a5b06c709cb1e1f`，result SHA256=
`43984cff1f428c9ba1959cf65635f2c56f871a1602222084dd3359b1e27ff767`。全部29项gate PASS，包括：

- connected overlap rectangles严格嵌套、repeat exact，realized=`0.26/0.50/0.76`，overshoot不超过
  新增最后一条strip；
- non-overlap control rectangles严格嵌套、repeat exact、与24px dilation交集严格0，normalized-y
  error=`0`；
- CLIP-mean/random-texture/wrong-slot CutMix三材质在全部level均target/control tensor exact、低level
  已写像素在高level保持exact、框外RGB exact、finite；
- random texture跨level共享同一场、repeat exact、换seed非exact，mean/std max error=`0/0`；
- same-slot与wrong-slot donor均different-PID/path、same-camera优先、无fixed point；固定visual feature
  后替换text bank的slot-cycle/state inversion exact。

独立复审对上述script/protocol SHA终审通过，明确只授权下一步full-train pose-only feasibility，
不授权8图CLIP、4090或正式训练。下一步必须对15,618图生成并封板target map、same-slot/wrong-slot/
wrong-mask donor map、连通矩形non-overlap可行率、24px dilation exact、y-error与target/non-target
几何泄漏分布；只有全门禁PASS后才从full map抽取覆盖五slot的8图CPU contract。

### Full pose-only feasibility 启动前审查

新增`phase0b2_full_pose_feasibility.py`，只复现B2-SI的path-hash geometry与hard-owner support，读取
official train list和exp386 strict pose artifact；不加载RGB tensor、CLIP、ReID、optimizer或CUDA。
它在全15,618图上一次生成balanced target、same-slot appearance/wrong-mask donor、wrong-slot
occluder donor及connected overlap/control map，并把逐record实际增强后wrong-mask IoU写入冻结map。

本地uv已通过py_compile、40样本balanced donor synthetic和20样本connected/control synthetic；
独立只读代码终审先后拦截并修正了output碰撞、FAIL map伪FROZEN、逐条IoU缺失三项阻塞。最终脚本
SHA256=`eff0293fc572169343e2e0ec0dd150c944465c6291745cd4e485e8817ffb558c`，终审裁决只授权首次
全15,618图CPU-only pose feasibility。8图CLIP、4090、Phase 0C和训练继续`NO-START`；full运行后还须
终审进程退出、result/map/runner SHA、15,618 records、PASS与map SHA绑定及执行repo tracked clean。

首次full CPU运行已启动。因当前本地历史包含与exp392无关的1.9GB wildlife模型blob，完整bundle经
SSH relay传输停滞后立即终止，255KiB残片移入远端quarantine，未作为执行源。正式执行改用Git
partial/sparse exact-HEAD snapshot：真实detached HEAD=
`a25a9d91cf93e9607a41faadf682ef04d78e444e`，只物化`datasets/`与
`experiments/exp392_clip_semantic_ciam/`所需tracked blob；archive SHA256=
`ac2a11691b669f529888d5b85802ccd724243184e3a2228bbb5aeb3b2b4f1c74`。远端fresh repo=
`/home/afr/SOLIDER-REID-exp392-b2s-pose-a25a9d9`，execution HEAD、四依赖script SHA、tracked clean、
exp386 manifest、三个输出路径不存在、无同类进程与GPU=`2 MiB/0%`均在启动前复核PASS。

唯一CPU任务的`time` PID=`1354894`、python PID=`1354895`；map/result/runner分别为
`/home/afr/reid-clean/audits/exp392_phase0b2/b2s_full_pose_map_a25a9d9.json`、
`b2s_full_pose_result_a25a9d9.json`、`b2s_full_pose_a25a9d9.runner.log`。启动后python约119% CPU，
CUDA_VISIBLE_DEVICES为空，GPU仍`2 MiB/0%`，result尚不存在且runner为0字节，符合计算阶段预期。

### Full pose-only feasibility 封板结果

唯一CPU任务自然结束，wall=`9:38.99`、maxRSS=`886,708 KiB`、exit=`2`；parent/python均退出，
GPU=`2 MiB/0%`，严格异常扫描0。result/map/runner SHA256分别为
`8ab726a15bfdc4ae36500e8208a6f56cf973579cbf16bf992ff829cf2911be09`/
`0c57660ddb6158721b9379094933e416612e2db94771d1113d271ea727ac3026`/
`1c9d2fc73790e32938481a2145f31b438418a5143f7193095c54c518e72729a4`。map含完整15,618 records，
status=`EXP392_PHASE0B2_FULL_POSE_MAP_FAILED_NOT_FROZEN`，没有把FAIL伪装为可复用冻结映射。

通过项说明实现与基础几何并未失效：balanced target counts=`3124/3124/3123/3124/3123`；五slot
valid counts=`15604/15618/15618/15607/15508`；hard-owner product严格0；donor完整、different
PID/path、无fixed point、same-camera priority violation=0；connected overshoot/nesting/realized和
control 24px dilation product均0失败。

裁决=`B2_S_FULL_POSE_CONSTRUCTION_FAIL`，失败来自三组定义冲突：

1. **24px non-overlap不可全split实现**：324/15,618图无可平移control，insufficient fraction=
   `0.02075`；其余15,294图normalized-y error mean/P50/P95=`0.24765/0.26110/0.41775`，远超
   `1/8`与`2/8`门槛。arms/torso/legs多数slot×direction组均失败，说明224×75人像内容只有约5个
   patch宽时，“离target一patch且纵向匹配”的control空间本身不足，而非单个方向偶发问题。324项
   全是control translation失败（torso=`234`、arms=`87`、其余三slot各`1`），connected overlap
   本身没有construction failure；旧gate名`connected_construction_complete`只是不精确的汇总命名。
2. **bbox矩形不是slot-local occluder**：共5,542个level-case、涉及2,759图，违反target intersection
   严格大于任一non-target intersection；其中arms占`5,065` cases。axis-aligned slot bbox会在arms等
   非凸/双侧解剖区域跨过torso，不能把该矩形解释成纯target support occlusion；不得继续拿它进入
   CLIP后归因。
3. **geometry-nearest与low-IoU wrong mask互相冲突**：实际增强后wrong-mask IoU overall
   P50/P95=`0.35387/0.73044`；head/torso/arms/upper-leg/lower-leg分别为
   `0.39195/0.80491`、`0.41352/0.71684`、`0.19785/0.55244`、`0.45706/0.74904`、
   `0.32957/0.68398`。最近的同slot area/y/conf donor自然也有相似空间mask，不能同时被预设成
   low-IoU反事实。

因此当前connected-bbox/non-overlap/nearest-low-IoU组合按预注册门禁封板，不改阈值、不换方向、
不重复运行，也不启动8图CLIP/GPU/训练。该FAIL只否定B2-S当前反事实构造，不否定已通过的PC-MBCLS
slot-support readout，更不否定CLIP语义校准TAPF。下一步先做独立机制归因并另写B2-Sv2预注册设计；
新设计必须用不读取CLIP结果的slot-local连通遮挡与可实现强对照，不能把本次FAIL map当正式冻结map。

独立只读归因复核同意上述边界：没有connected overlap实现错误证据；96×32 pixel IoU也不是
PC-MBCLS 16×16 token-grid（patch size=14px）的充分统计。其建议的新定义是
`B2-Sv2 slot-evidence deletion`：只在target
hard support内做方向性嵌套前缀，并用共享完全相同RGB的`base/corrupted × correct/cyclic-wrong mask`
difference-in-differences验证slot binding；same-slot donor只保留为appearance hard negative。该结论
支持继续CLIP–TAPF语义校准路线，不授权直接复活v1、8图CLIP、GPU或训练。

### B2-Sv2设计冻结与独立终审

针对“单次FAIL不能否定全部”的边界，B2-Sv2已把旧物理遮挡构造与teacher机制分开：v1只封板
`connected-bbox + 24px non-overlap + geometry-nearest low-IoU mask`，B2-SI已经通过的PC-MBCLS
slot-support证据保持成立。v2改测`slot-evidence deletion`，不再声称physical occlusion；wrong mask
使用同图四个固定nonidentity slot cycles，以共享RGB的2×2 DID直接检验anatomical binding。

独立只读终审对protocol SHA256=
`9255910a2dccfc202c35959a11f9cc46141e4d30a0afa5044d61f890c8e7db23`裁决PASS，确认：

1. 16×16 pooled coverage、nonzero patch count、matched top-K、coverage排序、patch坐标tie与binary mask
   定义完整可复现；
2. cycle固定为`w=(t+k) mod 5, k=1..4`，DID始终固定target support text，只替换spatial mask；
3. Spearman lower bound、correct-vs-nontarget/global paired差值、natural/top-K DID均使用PID-cluster
   95% CI，并逐`material × slot × cycle`裁决，不允许macro或另一材质救场；
4. random texture在384×128每个path/slot/seed只生成一次，固定`sigma=1.5`并跨level按同坐标复用；
5. Section九状态已更新为`B2-Sv1 SEALED-FAIL / B2-Sv2 DESIGN-FROZEN NO-START`，不会把单次构造
   FAIL扩张成CLIP–TAPF路线NO-GO。

当前只授权target/augmentation submap提取与official+exp386双源逐字节exact，以及pure synthetic
static实现和独立审查；full pose-only、8图CLIP、GPU、Phase 0C与训练继续`NO-START`。

## 2026-07-19 Phase 0C single-stage fast-track实现与独立复核

用户明确要求不能让B2-Sv1一次反事实构造FAIL无限期阻塞训练，因此更正旧边界：B2-Sv2继续作为
独立机制证据，但不再是首次single-stage训练的绝对前置。首个训练臂固定为bundled feasibility，
正确性与真实运行时门禁通过后即fresh启动e120；成功后必须补pose-only/static-q/generic-router拆因，
失败也只封板当前teacher/readout/router组合，不扩张为CLIP–TAPF永久NO-GO。

当前实现已经完成：

1. frozen PC-MBCLS在processor外部模型路径创建，同一次forward返回96×32 hard-owner `M`、五slot
   `q`与`valid`；三者同步detach/clone，teacher不注册进student、optimizer、checkpoint或eval；
2. teacher `M`只通过固定4×4 average pooling进入24×8 anchor尺度，不再用17-joint Gaussian/amax
   另画consumer几何；
3. student保留旧17-joint辅助头，同时直接预测五slot mask、q与presence；presence用straight-through
   hard gate，forward严格0/1，invalid teacher presence target=0；
4. 执行mask显式乘presence，router再乘q，因此`M=0`、`q=0`或`presence=0`均为逐元素identity；
5. loss冻结为旧`heatmap+confidence`加`mean(region-mask BCE,presence BCE,q BCE)`，不按中间性能调权；
6. e1-5/e6-10/e11+用同一fraction完整handoff `(M,q,presence)`，router前统一detach，ReID梯度只更新
   backbone/ID head/router，semantic梯度只更新anchor；
7. 训练日志分开记录mask/presence/q loss，并在首batch记录五slot q mean/std/entropy/constant-prior gap。

本地新增5项semantic unit contract全部PASS：teacher单源与4×pool、invalid/all-NULL exact、完整handoff、
anchor/router梯度所有权、M=0/q=0 identity。legacy D0相对当前HEAD的state与prepare关键tensor逐字节
exact，py_compile与`git diff --check` PASS。

独立子agent第二轮只读终审确认两个旧P0（q/M不同源、student无真实NULL）均已关闭，未发现阻止
进入正式CUDA/AMP preflight的新P0。其保留的P1风险是B2-SI q靠近0.5、q BCE只占semantic mean的
1/3且整体再乘0.1，可能只学到slot prior；本臂不偷调温度或权重，远端preflight必须原样报告q动态
范围、constant-prior gap、q-head梯度及稀疏mask的foreground/background预测质量。

下一步只做一次真实Torch1.13.1/OpenCLIP2.32 CUDA/AMP preflight：old/new PC-MBCLS parity、teacher/
checkpoint隔离、batch64/8-worker/micro4连续24步finite与GradScaler、两个consumer梯度、峰值<24GiB、
RGB-only eval和strict checkpoint reload。全部PASS后立即启动fresh e120，不再等待B2-Sv2全量审计。

首次preflight在构造完teacher/student、进入第一个训练step前因审计脚本自身属性名错误退出：隔离检查
读取了已在teacher初始化末尾删除的临时`model`外壳，而正式teacher按设计只保留`visual`与缓存text
tensor。traceback=`AttributeError: FrozenClipSlotTeacher has no attribute model`，GPU回到`2 MiB/0%`、
result/formal output均未创建、没有optimizer step或正式训练数据。该失败不涉及teacher前向、TAPF实现或
性能；隔离检查已改为遍历实际保留的`teacher.visual.parameters()`，须以新exact commit重新preflight。

第二次preflight已通过teacher/student构造、PC-MBCLS真实前向与batch64首个forward/backward，随后
审计脚本因默认GradScaler首步scale回退而主动退出；result/formal output仍未创建，GPU恢复空闲。
对照已封板exp387/exp391 CUDA门禁后确认，官方D0同样允许默认`65536`初始scale产生若干exact skip，
其正确门禁是overflow时model/optimizer严格不更新、scale下降，随后至少一次成功更新且连续8步finite，
而不是错误地要求第一个step永不skip。现已按既有clean D0协议修正：显式`unscale_`读取found-inf，
逐step核对q-head、两个consumer、backbone与head probe的exact skip/update，记录overflow count与scale
history，并保留24步内至少8个连续finite更新。该更正只修审计判据，不修改模型、loss、config或recipe。

第三次preflight自然完成24步：19次成功更新、前5步为默认GradScaler exact skip、随后连续19步finite；
q-head与两个consumer每个finite step均有非零梯度和参数更新，teacher隔离、checkpoint strict load、
RGB-only eval、state finite和batch64/8-worker全部PASS。峰值allocated/reserved=`7.55/8.02 GB`，吞吐=
`145.03 samples/s`。唯一FAIL是封板scalar renderer与新vector renderer的mask max-abs=
`2.6822e-6`超过脚本临时设定的`1e-6`，但PC-MBCLS q逐tensor exact、valid exact。

没有放宽阈值。根因定位为浮点运算顺序：vector版先在Python计算`31/127`与`95/383`再单次乘法，
封板renderer按tensor先乘整数再除分母。把vector坐标缩放改为完全相同的`tensor * integer / denominator`
两步后，远端正式runtime同4图mask逐tensor `torch.equal=True`、max-abs=`0`、valid exact，同时保留batch
vectorization。该修复不改变ontology、sigma、prompt、temperature、mask partition、loss或recipe；须以新
exact commit重跑一次完整preflight，不能复用FAIL JSON裁决正式启动。

新exact commit的完整preflight再次得到19/24成功更新、连续19步finite、约`8.02 GB`峰值、两个
consumer/q-head/隔离/checkpoint/eval全PASS；parity仍唯一FAIL，mask max-abs=`2.6226e-6`、q逐tensor
exact、valid exact。此时同一vector renderer在CPU已与封板scalar逐tensor exact，剩余差异确定来自
CPU与CUDA的`exp`舍入，而不是公式或运算顺序。

仍不放宽阈值：正式teacher改为在CPU以冻结vector renderer生成96×32 hard-owner mask/valid，再把
结果传到CLIP device；CLIP image/readout和student仍在GPU。真实batch64计时显示CPU vector render约
`63 ms/batch`，相对完整teacher+ReID preflight约增加12%而非不可接受的scalar loop开销。这样M与
B2-SI封板teacher在相同设备、相同算术、相同renderer下逐tensor同源；改动不涉及训练语义、loss或
阈值。须以新exact commit完成最终preflight后才启动正式e120。

## 2026-07-19 Phase 0C 最终预检 PASS 与正式训练启动

最终exact commit=`ed5783416528be4284adce11fa192fe119e344f4`的完整CUDA/AMP预检已PASS：

- PC-MBCLS parity：mask/q max-abs=`0/0`，valid逐tensor exact；
- 24步AMP共19次成功更新，前5步为默认GradScaler exact skip，随后连续19步finite；
- q-head、两个feature-dependent consumer、backbone与ID head均有非零梯度和参数更新；
- frozen teacher不进入student/optimizer/checkpoint/eval，RGB-only eval、strict checkpoint reload与state
  finite全部PASS；
- peak allocated/reserved=`7.56/8.00 GB`，吞吐=`130.45 samples/s`；
- result=`/home/afr/reid-clean/audits/exp392_phase0c/semantic_c0_ed57834_preflight.json`，SHA256=
  `dea9c8093be1559db8debbf2d59efcd2b72d4f046508ad65861473753b26cd7b`；runner SHA256=
  `73f7bd4251e5fdda82a9ffde5b810f2bc4443ff0ccf55e27746f41808925a64b`。

已知但不阻塞本臂的风险是teacher q动态范围较弱：五slot std约`0.011–0.024`，constant-prior gap很小。
这要求final后补pose-only、static-q与generic-router拆因，但不得在当前臂内偷调温度、loss权重或挑中间点。

全部预注册门禁通过后，已fresh启动single-stage Semantic TAPF bundled feasibility正式训练：

- repo=`/home/afr/SOLIDER-REID-exp392-semantic-c0-ed57834`，exact detached HEAD=
  `ed5783416528be4284adce11fa192fe119e344f4`；
- config=`configs/occluded_duke/swin_tiny_tapf_semantic_c0.yml`，SHA256=
  `ecf3403c3e3d61af575f49420f21247f93029785364d32a23711eab66458d39c`；
- output=`log/occluded_duke/exp392_clean_swin_tiny_semantic_c0_s1234`；runner=
  `/home/afr/train-logs/exp392_semantic_c0_s1234.runner.log`；main PID=`1375252`；
- recipe=`120 epoch / batch64 / seed1234 / SGD / lr0.0008 / eval10 / checkpoint120`，fresh且不续训、
  不挑best，必须自然跑满e120。

实时复核时已自然完成e1并进入e2：唯一main+8 workers，4090约`8184 MiB`；HEAD/config/tracked source
clean，无NaN/Inf/Traceback/RuntimeError/OOM/overflow异常，无checkpoint。e1末Loss从首个记录点`18.983`
下降，Semantic/RegionMask/Presence/Q约`0.693`，Student=`0`符合e1-5 teacher handoff，Reliability约
`0.51`，GateAbs从`5.965e-11`增长到`7.723e-09`且finite。当前状态为`FORMAL RUNNING`，此前
teacher反事实审计的单次FAIL只作为机制风险，不再是一票否决训练或整条CLIP–TAPF路线的理由。

### 15分钟接手：e3自然完成

远端实时复核确认e1-e3均自然完成，当前训练继续：exact HEAD与config SHA保持不变，tracked source
clean；唯一main PID=`1375252`加8 workers，4090约`8186 MiB/99%`且无第二项计算任务，e120前
checkpoint数=`0`。e3末记录为Loss=`7.961`、Pose=`1.597`、Semantic=`0.686`、RegionMask=
`0.684`、Presence=`0.682`、Q=`0.693`、Student=`0`、Reliability=`0.508`、GateAbs=
`7.761e-08`，全部finite。Mask/Presence已从初始化附近下降而Q仍接近随机BCE，这与预检中q动态范围弱的
已知风险一致，只记录观察，不据此早停、调权或裁决。runner/train log未见AMP warning、NaN/Inf、
Traceback、RuntimeError、OOM、nonfinite或overflow，状态=`继续自然e120`。

### 首次e10完整评测与e12接手

- e10完整query/gallery评测mAP/R1/R5/R10=`32.3/42.0/57.4/63.0`；
- 同epoch exp387 clean D0=`33.4/42.7/59.8/65.2`，Semantic C0−D0=
  `−1.1/−0.7/−2.4/−2.2`；同epoch exp389 HT0=`34.2/44.4/59.7/65.8`，Semantic C0−HT0=
  `−1.9/−2.4/−2.3/−2.8`；
- e6--e10 Student依次=`0.2/0.4/0.6/0.8/1.0`，e11+保持`1.0`，完整handoff符合冻结设计；
- e10末Pose=`1.373`、Semantic/RegionMask/Presence/Q=`0.589/0.549/0.526/0.693`、Reliability=
  `0.502`、GateAbs=`1.707e-05`，全部finite；接手时已自然完成e11并进入e12，e12 iter140的
  Pose=`1.284`、Semantic/RegionMask/Presence/Q=`0.548/0.492/0.459/0.693`、Reliability=`0.503`、
  GateAbs=`2.728e-05`。

首次中间评测低于两个参考，且Q分量仍在日志三位精度下保持`0.693`；前者不用于早停或final裁决，
后者继续作为“teacher q弱动态范围/可能只学slot prior”的预注册拆因风险。不得在当前运行中修改温度、
loss权重、代码或config。实时边界仍为exact HEAD/config/tracked clean、唯一main+8 workers、4090约
`8270 MiB`且唯一任务、e120前checkpoint=`0`，严格异常扫描命中`0`，状态=`继续自然e120`。

### e20完整评测与e21接手

- e20完整query/gallery评测mAP/R1/R5/R10=`41.8/53.9/69.6/75.0`；
- 同epoch exp387 clean D0=`42.2/52.4/67.6/74.0`，Semantic C0−D0=
  `−0.4/+1.5/+2.0/+1.0`；同epoch exp389 HT0=`42.8/53.1/68.9/74.4`，Semantic C0−HT0=
  `−1.0/+0.8/+0.7/+0.6`；
- e20末Pose=`0.933`、Semantic/RegionMask/Presence/Q=`0.383/0.269/0.188/0.692`、Student=`1.0`、
  Reliability=`0.509`、GateAbs=`2.928e-05`，全部finite；评测后自然进入e21，接手时e21 iter120
  Loss=`1.934`、Pose=`0.910`、Semantic/RegionMask/Presence/Q=`0.372/0.255/0.170/0.692`。

相对参考的mAP负差与Rank正差只构成中间混合轨迹，不作GO/NO-GO、早停或best选择。Q从三位精度
`0.693`轻微降至`0.692`，表明不是完全零更新，但其变化仍显著弱于mask/presence，继续保留为final后
拆因风险。exact HEAD/config/tracked clean、唯一main+8 workers、4090约`8404 MiB`且唯一任务、
e120前checkpoint=`0`，严格异常扫描命中`0`，状态=`继续自然e120`。

### 15分钟接手：e30训练中，尚无完整e30评测

e21--e29已自然完成，接手时位于e30 iter100/227，尚未产生e30完整query/gallery评测，因此本轮不写
半成品指标。现场Loss=`0.709`、Pose=`0.801`、Semantic/RegionMask/Presence/Q=
`0.320/0.188/0.080/0.692`、Student=`1.0`、Reliability=`0.512`、GateAbs=`1.527e-05`，全部
finite。exact HEAD/config/tracked clean、唯一main+8 workers、4090约`8406 MiB/99%`且唯一任务、
e120前checkpoint=`0`，严格异常扫描命中`0`。Q仍处弱动态区间但非新增故障，继续自然训练；下次接手
只从正式日志读取完整e30及之后的新增评测。

### e30完整评测与e39接手

- e30完整query/gallery评测mAP/R1/R5/R10=`45.5/55.8/71.5/77.6`；
- 同epoch exp387 clean D0=`46.6/56.2/71.3/76.4`，Semantic C0−D0=
  `−1.1/−0.4/+0.2/+1.2`；同epoch exp389 HT0=`47.7/58.3/72.0/77.1`，Semantic C0−HT0=
  `−2.2/−2.5/−0.5/+0.5`；
- e30末Loss=`0.639`、Pose=`0.803`、Semantic/RegionMask/Presence/Q=
  `0.320/0.188/0.079/0.692`、Student=`1.0`、Reliability=`0.512`、GateAbs=`1.637e-05`，
  全部finite；评测后e31--e38自然完成，接手时已进入e39。

e30相对D0/HT0仍为正负混合且mAP偏弱的中间轨迹，不用于早停、best选择或路线裁决。e39 iter40
现场Semantic/RegionMask/Presence/Q=`0.305/0.170/0.052/0.692`、GateAbs=`1.133e-05`，Q继续弱
动态，mask/presence继续下降。exact HEAD/config/tracked clean、唯一main+8 workers、4090约
`8392 MiB`且唯一任务、e120前checkpoint=`0`，严格异常扫描命中`0`，状态=`继续自然e120`。

### e40完整评测与e48接手

- e40完整query/gallery评测mAP/R1/R5/R10=`49.4/59.2/74.4/79.3`；
- 同epoch exp387 clean D0=`50.0/60.7/76.2/81.0`，Semantic C0−D0=
  `−0.6/−1.5/−1.8/−1.7`；同epoch exp389 HT0=`49.0/59.3/74.0/79.0`，Semantic C0−HT0=
  `+0.4/−0.1/+0.4/+0.3`；
- e40末Loss=`0.358`、Pose=`0.772`、Semantic/RegionMask/Presence/Q=
  `0.304/0.169/0.049/0.692`、Student=`1.0`、Reliability=`0.512`、GateAbs=`1.138e-05`，
  全部finite；评测后e41--e47自然完成，接手时已进入e48。

e40相对D0全面偏低、相对HT0近零混合，仍只是中间轨迹。e48 iter20的Q=`0.692`、Reliability=
`0.512`、GateAbs=`1.025e-05`，语义q与router幅度继续偏弱但finite，禁止据此修改运行或提前裁决。
exact HEAD/config/tracked clean、唯一main+8 workers、4090约`8288 MiB`且唯一任务、e120前
checkpoint=`0`，严格异常扫描命中`0`，状态=`继续自然e120`。

### e50完整评测与e56接手

- e50完整query/gallery评测mAP/R1/R5/R10=`52.5/63.4/77.9/82.5`；
- 同epoch exp387 clean D0=`52.1/62.8/77.0/81.9`，Semantic C0−D0=
  `+0.4/+0.6/+0.9/+0.6`；同epoch exp389 HT0=`52.7/62.1/76.4/81.7`，Semantic C0−HT0=
  `−0.2/+1.3/+1.5/+0.8`；
- e50末Loss=`0.259`、Pose=`0.763`、Semantic/RegionMask/Presence/Q=
  `0.298/0.163/0.039/0.692`、Student=`1.0`、Reliability=`0.512`、GateAbs=`9.160e-06`，
  全部finite；评测后e51--e55自然完成，接手时已进入e56。

e50相对D0四项为正，但只是单个中间epoch，不能取代final或反转此前混合轨迹。e56 iter140的
Semantic/RegionMask/Presence/Q=`0.296/0.161/0.035/0.692`、GateAbs=`8.745e-06`，进一步表明
当前可执行state主要由mask/presence获得明显监督，q仍接近弱动态先验；此观察留待final后的
pose-only/static-q/generic-router拆因。exact HEAD/config/tracked clean、唯一main+8 workers、4090约
`8386 MiB`且唯一任务、e120前checkpoint=`0`，严格异常扫描命中`0`，状态=`继续自然e120`。

### e60完整评测与e65接手

- e60完整query/gallery评测mAP/R1/R5/R10=`53.9/64.4/78.2/83.1`；
- 同epoch exp387 clean D0=`55.1/66.1/79.0/83.3`，Semantic C0−D0=
  `−1.2/−1.7/−0.8/−0.2`；同epoch exp389 HT0=`54.5/64.8/78.6/83.9`，Semantic C0−HT0=
  `−0.6/−0.4/−0.4/−0.8`；
- e60末Loss=`0.203`、Pose=`0.759`、Semantic/RegionMask/Presence/Q=
  `0.295/0.160/0.033/0.692`、Student=`1.0`、Reliability=`0.512`、GateAbs=`8.583e-06`，
  全部finite；评测后e61--e64自然完成，接手时已进入e65。

e50相对D0的四项正差未在e60保持，进一步确认不能选择中途节点。e65 iter100的
Semantic/RegionMask/Presence/Q=`0.294/0.159/0.030/0.692`、GateAbs=`7.605e-06`，q与router
幅度持续弱，但训练、consumer执行与数值均正常。exact HEAD/config/tracked clean、唯一main+8 workers、
4090约`8238 MiB`且唯一任务、e120前checkpoint=`0`，严格异常扫描命中`0`，状态=`继续自然e120`。

### e70完整评测与e74接手

- e70完整query/gallery评测mAP/R1/R5/R10=`55.4/66.0/79.4/84.6`；
- 同epoch exp387 clean D0=`55.4/65.2/79.5/83.6`，Semantic C0−D0=
  `+0.0/+0.8/−0.1/+1.0`；同epoch exp389 HT0=`55.1/64.6/78.8/83.1`，Semantic C0−HT0=
  `+0.3/+1.4/+0.6/+1.5`；
- e70末Loss=`0.176`、Pose=`0.755`、Semantic/RegionMask/Presence/Q=
  `0.294/0.159/0.029/0.692`、Student=`1.0`、Reliability=`0.512`、GateAbs=`7.980e-06`，
  全部finite；评测后e71--e73自然完成，接手时已进入e74。

e70相对D0再次呈近零mAP与Rank混合正差，和e60回落共同证明中途波动仍大，不作final替代或机制
claim。e74 iter40的Q=`0.692`、Reliability=`0.512`、GateAbs=`7.764e-06`，弱q/小router幅度模式
未改变。exact HEAD/config/tracked clean、唯一main+8 workers、4090约`8364 MiB`且唯一任务、
e120前checkpoint=`0`，严格异常扫描命中`0`，状态=`继续自然e120`。

### e80完整评测与e82接手

- e80完整query/gallery评测mAP/R1/R5/R10=`55.8/66.4/79.6/84.4`；
- 同epoch exp387 clean D0=`56.1/66.3/79.5/84.0`，Semantic C0−D0=
  `−0.3/+0.1/+0.1/+0.4`；同epoch exp389 HT0=`55.4/65.4/78.9/82.9`，Semantic C0−HT0=
  `+0.4/+1.0/+0.7/+1.5`；
- e80末Loss=`0.162`、Pose=`0.752`、Semantic/RegionMask/Presence/Q=
  `0.293/0.158/0.028/0.692`、Student=`1.0`、Reliability=`0.512`、GateAbs=`7.470e-06`，
  全部finite；评测后e81自然完成，接手时已进入e82。

e80继续呈相对D0的mAP略负、三项Rank略正，不能包装为全面提升。e82 iter200的Q=`0.692`、
Reliability=`0.512`、GateAbs=`7.236e-06`，弱q与小router幅度持续稳定而无数值退化。exact
HEAD/config/tracked clean、唯一main+8 workers、4090约`8262 MiB`且唯一任务、e120前
checkpoint=`0`，严格异常扫描命中`0`，状态=`继续自然e120`。

### e90完整评测与e91接手

- e90完整query/gallery评测mAP/R1/R5/R10=`56.6/67.1/80.3/85.1`；
- 同epoch exp387 clean D0=`57.5/67.9/81.2/85.3`，Semantic C0−D0=
  `−0.9/−0.8/−0.9/−0.2`；同epoch exp389 HT0=`56.5/66.1/79.8/84.4`，Semantic C0−HT0=
  `+0.1/+1.0/+0.5/+0.7`；
- e90末Loss=`0.153`、Pose=`0.753`、Semantic/RegionMask/Presence/Q=
  `0.292/0.158/0.027/0.692`、Student=`1.0`、Reliability=`0.512`、GateAbs=`7.041e-06`，
  全部finite；评测后自然进入e91。

e90轨迹支持“当前优于纯结构HT0、仍未追平clean D0”的暂时描述，但不能上升为final结论。e91
iter120的Q=`0.692`、Reliability=`0.512`、GateAbs=`6.156e-06`，弱q/小router模式保持稳定。
exact HEAD/config/tracked clean、唯一main+8 workers、4090约`8264 MiB`且唯一任务、e120前
checkpoint=`0`，严格异常扫描命中`0`，状态=`继续自然e120`。

### 15分钟接手：e100训练中，尚无完整e100评测

e91--e99已自然完成，接手时位于e100 iter60/227，尚未产生e100完整query/gallery评测，本轮不写
半成品指标。现场Loss=`0.156`、Pose=`0.751`、Semantic/RegionMask/Presence/Q=
`0.292/0.158/0.026/0.692`、Student=`1.0`、Reliability=`0.512`、GateAbs=`6.781e-06`，全部
finite。exact HEAD/config/tracked clean、唯一main+8 workers、4090约`8264 MiB/96%`且唯一任务、
e120前checkpoint=`0`，严格异常扫描命中`0`。继续自然训练，下次只从正式日志读取完整e100及后续
新增评测。

### e100完整评测与e109接手

- e100完整query/gallery评测mAP/R1/R5/R10=`56.5/66.7/80.3/84.3`；
- 同epoch exp387 clean D0=`56.9/67.1/79.6/83.8`，Semantic C0−D0=
  `−0.4/−0.4/+0.7/+0.5`；同epoch exp389 HT0=`56.4/65.9/79.2/84.3`，Semantic C0−HT0=
  `+0.1/+0.8/+1.1/+0.0`；
- e100末Loss=`0.148`、Pose=`0.752`、Semantic/RegionMask/Presence/Q=
  `0.292/0.158/0.026/0.692`、Student=`1.0`、Reliability=`0.512`、GateAbs=`6.604e-06`，
  全部finite；评测后e101--e108自然完成，接手时已进入e109。

e100继续呈mAP/R1略低于D0、R5/R10略高的混合模式，不能包装为全面提升。e109 iter40的
Q=`0.692`、Reliability=`0.512`、GateAbs=`6.650e-06`，弱q与小router模式保持。exact HEAD/config/
tracked clean、唯一main+8 workers、4090约`8394 MiB`且唯一任务、e120前checkpoint=`0`，严格异常
扫描命中`0`，状态=`继续自然e120`。

### e110完整评测与e118接手

- e110完整query/gallery评测mAP/R1/R5/R10=`56.7/66.6/80.3/84.7`；
- 同epoch exp387 clean D0=`57.4/67.4/80.5/84.6`，Semantic C0−D0=
  `−0.7/−0.8/−0.2/+0.1`；同epoch exp389 HT0=`56.6/65.9/79.5/83.9`，Semantic C0−HT0=
  `+0.1/+0.7/+0.8/+0.8`；
- e110末Loss=`0.141`、Pose=`0.751`、Semantic/RegionMask/Presence/Q=
  `0.292/0.158/0.025/0.692`、Student=`1.0`、Reliability=`0.512`、GateAbs=`6.779e-06`，
  全部finite；评测后e111--e117自然完成，接手时已进入e118。

e110仍是“优于HT0、未追平D0”的混合轨迹，不是final。e118 iter40的Q=`0.692`、Reliability=
`0.512`、GateAbs=`6.540e-06`，训练约剩3个自然epoch。exact HEAD/config/tracked clean、唯一main+8
workers、4090约`8382 MiB`且唯一任务、e120前checkpoint=`0`，严格异常扫描命中`0`。下一次接手
必须先确认自然e120完成，再做进程/GPU/唯一checkpoint/SHA/strict finite/teacher隔离/参数轨迹与
counterfactual终审；不得把e110当final。

## 2026-07-19 Phase 0C Semantic C0自然e120与终审封板

正式运行自然完成全部120个epoch及12次固定间隔评测，未早停、未续训、未选择中途best。final
mAP/R1/R5/R10=`56.9/67.1/80.6/85.0`：

- 相对exp387 clean D0=`57.6/67.7/80.8/84.6`为`−0.7/−0.6/−0.2/+0.4`；
- 相对official B0=`57.4/67.4/80.6/85.2`为`−0.5/−0.3/+0.0/−0.2`；
- 相对exp389 HT0=`56.9/65.9/80.0/84.1`为`+0.0/+1.2/+0.6/+0.9`。

e120末Pose=`0.752`，Semantic/RegionMask/Presence/Q=`0.292/0.158/0.026/0.692`，Student=`1.0`，
Reliability=`0.512`，GateAbs=`6.981e-06`，全部finite。mask与presence相对初始化获得明显监督，
但q仍接近弱动态先验，router执行幅度也很小；优于纯结构HT0不足以把该bundled差值归因给CLIP语义。

运行边界终审：main PID=`1375252`及8 workers自然退出；GPU恢复`2 MiB/0%`；output仅含
`train_log.txt`和唯一`transformer_120.pth`。checkpoint size=`113544187`，checkpoint/train-log/
runner SHA256分别为`8f8e4a8af1280f17f736053a3068dfae0384ac54915f9c68fb0c779350c3638e`、
`7d2640208f86c11a256b2ea27b1b1ec17cde2fe1fcbc0a72a49c6f12e21f46bd`、
`57e9eecae14bf741aa8ab2e29439458687f235f87bfc9bc8804e30552c1ba9ef`。exact HEAD/config/tracked
source保持不变，严格AMP warning/NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow扫描命中0。

独立只读checkpoint终审v2=`PASS`：231个state tensor strict load，全部floating state finite，且
state key中无CLIP/teacher/text prototype。相对同seed、同预训练初始化，anchor、q-support head、
consumer0/1的L2轨迹分别为`1.86247/0.020724/0.030388/0.033860`，两router expert也均离开零初始化。
RGB-only correct/shuffle/None/exploding descriptor逐tensor exact；两个consumer单独置零时final descriptor
L2分别改变`0.001131/0.002968`；两router的zero-mask与zero-q输出均为逐tensorexact identity。
8图state把五slot混合统计时support mean/std/min/max=`0.51172/0.01686/0.48159/0.53281`，两consumer
gate-delta abs-mean=`3.606e-06/1.040e-05`，说明路径可达但执行信号弱；该pooled std不能区分
between-slot prior与同slot跨图动态，后者由Phase 0D单独审计。

审计v1曾因把`context_projection`内的字符串`text`误报为text encoder而返回FAIL；这是审计器键名
规则错误，不是checkpoint失败，v1证据保留且未改训练。修正为明确组件名后v2全部门禁PASS。v2
script/result/runner SHA256分别为
`00131e54a603edf59fee81085ea7ba32ecd5e40520dd272aa338129d45e4ef93`、
`1712d2ecc774cb5c76792fbc526d9cab29dcdcc7dfc46d547cd8564898471f53`、
`f1adc39db8896fb11f5454ab8d4c85c8aa26f64bd1218fc7b99776ba6504f5e2`。

裁决：`PHASE 0C SEMANTIC C0 = SEALED / CURRENT BUNDLED COMBINATION NO-GO`。当前
PC-MBCLS teacher、弱动态q readout与双router组合没有超过clean D0，不能声称CLIP语义增量成立，
也不授权balanced semantic multi-stage。该结论只关闭当前组合，不否定CLIP–TAPF；下一步必须做
必要的单变量拆因/机制修复（优先验证static-q/pose-only与真正扩大sample-specific support动态范围），
不得重跑、换seed或用中途best救场。

## 2026-07-19 Phase 0D冻结拆因设计

已新增`phase0d_protocol.md`。本阶段不训练，只在Semantic C0唯一final checkpoint上依次评测correct、
static-slot-q、q-one、spatial-constant mask、slot-cycle、expert-mean、router0/1/all bypass。预注册
`0.1 mAP`只用于单checkpoint近零归因，不作为论文显著性阈值；correct start/end descriptor与state
SHA必须exact。

## 2026-07-19 Phase 0D全验证集冻结拆因封板

唯一只读进程自然完成全部10个pass，每个pass覆盖query+gallery共19,871图。correct-start/end均为
`56.920063/67.058825/80.588233/85.022622`，descriptor逐tensor exact，模型state SHA前后均为
`8ce9a1a0da07e33d4a8936b9726071de0653dca7656409ed14000c68468dd284`。全部arm相对correct：

| arm | ΔmAP | ΔR1 | ΔR5 | ΔR10 |
|---|---:|---:|---:|---:|
| static-slot-q | `+0.000056` | `0` | `0` | `0` |
| q-one | `−0.000060` | `0` | `0` | `0` |
| spatial-constant-mask | `+0.000654` | `0` | `0` | `0` |
| slot-cycle | `+0.000009` | `0` | `0` | `0` |
| expert-mean | `−0.000092` | `0` | `0` | `0` |
| router-0-bypass | `−0.000067` | `0` | `0` | `0` |
| router-1-bypass | `−0.000029` | `0` | `0` | `0` |
| all-router-bypass | `−0.000077` | `0` | `0` | `0` |

五slot q全验证集均值为`0.532707/0.481671/0.509869/0.513475/0.521318`，同slot跨图std仅
`0.000293/0.000163/0.000090/0.000121/0.000191`。因此此前8图pooled std=`0.01686`几乎全部来自
固定slot均值差，不是sample-specific动态。所有干预远低于预注册`0.1 mAP`近零线，且rank排序完全
不变。checkpoint审计中“单consumer置零会产生非零descriptor L2”只证明数值可达；本次完整检索审计
证明该差异小到不能改变任何报告rank，all-router-bypass也近乎exact retrieval parity。

script/result/runner SHA256分别为
`fa58f0f59a6d84c415aca9479f076df71308d628a82d6b1cca526d5c6f9fe2ab`、
`6c9eba824ca7d779d45c93e3761253fa17b049d82ab164766c539970323899c9`、
`521bf90ab495523bb95b562086929ff02e22dd2cb52ec48c70442cd9f22f2942`。全部descriptor finite、
correct start/end exact、state SHA exact、异常0，进程自然退出，GPU=`2 MiB/0%`。

裁决：`PHASE 0D = SEALED / CURRENT SEMANTIC ROUTE RETRIEVAL-INERT`。当前失败不只是q较弱，而是
整条semantic route在e120检索排序上近似失活；mask geometry、sample-specific q、slot binding与
slot-specific expert均无可辨识边际贡献。下一机制不能只调q温度或复制stage，必须让语义路由从训练
早期就获得有量级约束的可执行残差，并以all-router-bypass的final差值作为预注册门禁。
