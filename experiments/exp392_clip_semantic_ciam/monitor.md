# exp392 监控记录

## 当前状态

- `PHASE 0A SEALED / PHASE 0B SEALED-CURRENT-TEACHER-NO-GO / FORMAL TRAINING NO-START`；
- exp390与exp391均已封板，禁止重启、续训或把本实验记为exp391 Phase B/C；
- 当前GPU任务：无，Phase 0B结束后GPU=`2 MiB / 0%`；
- 已完成文献、公开代码、当前TAPF执行路径、机制审查与Phase 0A frozen audit；
- 未创建训练config/output/checkpoint，未启动任何正式训练。

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
