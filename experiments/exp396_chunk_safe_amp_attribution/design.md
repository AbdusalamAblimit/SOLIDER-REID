# 实验 exp396：超大梯度组chunk-safe exact AMP归因门

## 当前状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / PHASE 0Q STATIC-CPU SEALED-PASS /
CUDA ATTRIBUTION SEALED-PASS / SHARED_D0_OR_RUNTIME_NONFINITE /
FORMAL NO-START`。

exp396是独立测量器实验，不是exp395重跑。exp395保持
`CUDA_ATTRIBUTION_EXECUTION_SEALED_INVALID / REPORTER_RUNTIME_FAIL`：唯一actual在第一行D0
`reid`的scaled backward之后、unscale之前，由canonical `torch.quantile`对backbone级大输入抛出
`input tensor is too large`。没有完整matrix，也没有任何exp394根因证据。

## 动机

exp395证明小张量synthetic contract不足以保证真实梯度组可统计。下一版不能删掉P50/P95/P99、抽样、
缩小group或把backbone拆小来绕过失败，而应保持原证据单位，给出不受`torch.quantile`单张量上限约束的
exact统计器。测量器必须先回答自身的规模契约，之后才有资格再次读取actual AMP梯度。

## 核心假设

对每个parameter group做两遍只读chunk扫描，可以在不构造全量Torch拼接tensor的前提下精确得到：

1. absent/present、zero/nonzero tensor数；
2. finite/NaN/`+Inf`/`-Inf`元素数；
3. finite abs-max与稳定L2；
4. 把finite absolute values写入临时regular memmap并原地排序后，按
   `index=(N-1)q`线性插值得到与`torch.quantile`定义一致的P50/P95/P99。

临时memmap只属于reporter内部，必须在每格结束后删除；进程正常或异常退出后均不得留下scratch或把它
当第四类结果资产。

## 冻结诊断对象

除reporter实现外，全部继承exp395且不得改变：

- source commit=`11d7a35788c4645c355d96d76a2a4ff20a9801ac`；
- D0 loss顺序=`reid/heatmap/confidence/pose/total`；
- rich loss顺序=`reid/heatmap/confidence/mask/presence/evidence_cosine/evidence_relation/
  exec_consumer0/exec_consumer1/pose/total`；
- 15个互斥parameter group及D0旧PSG映射；
- official first batch64、seed1234、fresh default GradScaler scale=`65536`；
- 每行恢复同一initial state/RNG，执行
  `zero_grad -> fresh forward -> scaled backward -> scaled capture -> unscale -> unscaled capture -> discard`；
- optimizer/scaler/scheduler update=`0`，checkpoint=`0`；
- exp394、exp395与formal训练均保持sealed/NO-START。

## chunk-safe exact reporter

冻结chunk上限=`1,048,576` elements。每个present gradient以原参数顺序、flat index顺序扫描，不修改原
gradient：

### Pass 1

- 每chunk复制为CPU FP64；
- 累加四类finite/non-finite计数；
- `abs-max`取chunk max的全局max；
- L2先用FP64 vector norm得到chunk norm，再用`math.hypot`归约，避免简单平方和溢出；
- 得到finite元素总数N。

### Pass 2

- 创建shape=`[N]`、dtype=`float64`的regular memmap；
- 按相同顺序写入所有finite absolute values；
- flush后用NumPy原地exact sort；
- 对q=`0.50/0.95/0.99`，读取floor/ceil order statistic并做线性插值；
- 关闭并删除memmap。

不得使用近似quantile、reservoir sampling、histogram binning、随机子采样或仅保存top-k。空finite集合仍
写JSON `null`。scaled/unscaled非有限计数与range比例规则保持exp395定义。

## Phase 0Q static/CPU contract

任何CUDA前必须连续两遍通过：

1. exp395 actual result/runner/manifest SHA与失败位置exact；
2. exp396 source中reporter不含`torch.quantile`或全量`torch.cat`；
3. 小张量含absent/zero/finite/NaN/±Inf时，计数与exp395 reference exact；
4. 小张量P50/P95/P99与`torch.quantile(..., interpolation="linear")` exact；
5. L2在严格FP64容差内与reference一致；
6. 输入规模至少`16,777,217`元素时完整运行，超过exp395失败级别且结果与解析order statistic一致；
7. 多parameter、多chunk顺序与单一拼接reference一致；
8. 输入gradient逐tensor逐字节不变；
9. scratch在成功和注入异常两条路径均清零；
10. D0/rich loss、15组、default scaler与zero-update静态边界不变；
11. contract设置`CUDA_VISIBLE_DEVICES=''`且before/after均未初始化CUDA；
12. 两遍result/runner逐字节一致。

static PASS只授权独立exp396 CUDA actual；不修复exp394，不恢复exp395，也不授权训练。

## CUDA有效性门

若static封板，后续actual仍必须fresh：新的execution repo、exp396命名regular CLIP/codebook、新
result/runner/manifest路径与启动前空闲4090。完整矩阵、state/RNG/teacher exact、scratch清零、更新0、
checkpoint 0、进程退出和GPU空闲共同决定归因结果是否有效。gradient non-finite是预注册输出，不是
提前退出条件；未捕获RuntimeError/OOM/provenance/state失败则立即封板INVALID且不补跑。

## 风险与失败解释

exact memmap sort会增加CPU时间与临时磁盘I/O，但这是保留原百分位定义的代价，不允许用近似算法换取
表面通过。若超大static仍失败，只关闭当前reporter；若actual完整而未复现exp394，只记录fresh exp396
non-reproduction，不推翻exp394。无论actual结果如何，下一步AMP-stable机制仍需另立实验。

## Phase 0Q static/CPU封板

production reporter与独立contract已连续两遍逐字节PASS。33项gate全部通过：小张量含NaN/±Inf与
multi-parameter/multi-chunk统计对齐reference；P50/P95/P99与linear `torch.quantile` exact；
`16,777,217`元素case得到解析exact的`8,388,608 / 15,938,355.2 / 16,609,443.84`；输入SHA前后
exact；success与注入异常scratch均清零；production reporter不含`torch.quantile`或全量`torch.cat`；
CUDA initialized=`false/false`，update=`0`、checkpoint=`0`。

implementation/static/result SHA256=
`6e8b2b67efcda7cfaca8527fb0ae1dd4c6aedcebef3fec6ded2e5ba6ddab8164`/
`f3a2ee3ccafa4caa1606b92b93b86177cc0b5ef6cfe7ac2b6f0d31fa195c415b`/
`e5d68df7731042a98f440f43acc45c9cf11b70aa7df25e09397ff6375f355394`；runner与repeat result/runner均为
同一SHA。

裁决=`PHASE0Q_STATIC_CPU_SEALED_PASS / CUDA ATTRIBUTION FRESH-EXECUTION GO`。根据用户持续授权，提交
封板后可直接建立fresh exp396 execution与regular资产并执行唯一actual，无需再次确认。正式训练仍
`NO-START`。

## CUDA attribution actual封板

唯一fresh actual完整执行D0五行、rich十一行和十五组scaled/unscaled matrix，所有13项validity gate、
D0 7项arm gate与rich 7项arm gate均PASS。common initial state 211个tensor exact；teacher target
64×5=`320`个slot有效；model/optimizer/teacher/codebook/RNG前后exact；scratch=`0`、update=`0`、
checkpoint=`0`。

预注册outcome=`SHARED_D0_OR_RUNTIME_NONFINITE`。D0与rich的`reid` scalar均为
`20.846956253051758`，且两者`reid/total`只在`backbone`组出现完全相同的non-finite支持：每个stage
共`27,519,354`个gradient元素，其中finite=`27,511,050`、NaN=`368`、`+Inf=3,753`、`-Inf=4,183`；
scaled和unscaled计数一致。D0 heatmap/confidence/pose与rich heatmap/confidence/mask/presence/
evidence cosine/evidence relation/两个exec consumer/pose全部finite；ID head、anchor、router及expert
组也未出现non-finite。D0/rich `total`的backbone统计与各自`reid`完全相同，排除了rich auxiliary或
aggregate作为本次首步overflow的必要条件。

该证据把exp394宽泛FAIL收紧到canonical default-scale下的**shared D0/rich ReID backbone graph**，
但十五组报告不能进一步定位某个backbone parameter或算子。它也不推翻exp394按自身绝对首步finite门
封板FAIL；只是证明该门同样会拒绝matched clean D0，不能再把exp394现象称为rich-specific数值失败。

actual内部矩阵耗时=`7.281521141529083 s`，peak allocated CUDA memory=
`7,631,537,152 bytes`。script/result/runner/manifest SHA256=
`6e8b2b67efcda7cfaca8527fb0ae1dd4c6aedcebef3fec6ded2e5ba6ddab8164`/
`58ae4beb56c9dabbff7fd77202d87b53f3ccecc9edec725051f04ed3c60ed96c`/
`58ae4beb56c9dabbff7fd77202d87b53f3ccecc9edec725051f04ed3c60ed96c`/
`3a0ef5d98dd6387b330958bbfb1e9d893e60745e8857237bbbbe375778886c64`。进程退出后4090=`2 MiB/0%`，
无compute process。

最终裁决=`CUDA_ATTRIBUTION_SEALED_PASS / SHARED_D0_OR_RUNTIME_NONFINITE`。exp396不得重跑。下一步
另立exp397 matched native GradScaler dynamics gate：保持默认initial scale，不手调scale，让D0与rich
分别按原生`step/update`记录skip与scale轨迹，以baseline-relative而非绝对首步finite判断rich是否额外
不稳定。formal训练仍`NO-START`。
