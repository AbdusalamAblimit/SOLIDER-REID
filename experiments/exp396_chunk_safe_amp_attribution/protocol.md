# exp396 chunk-safe exact AMP归因协议

## 状态

`PROTOCOL-FROZEN / PHASE 0Q STATIC-CPU SEALED-PASS /
CUDA ATTRIBUTION SEALED-PASS / SHARED_D0_OR_RUNTIME_NONFINITE /
FORMAL NO-START`。

## 冻结上游

- exp394 actual script/result/runner SHA=
  `bae2210bc606048371b4750f85919595c0b8fdbd1e11681abac59fe9727ea4f0`/
  `3897d76fd6b6aeb0d9ed2a27e527053874f6cdf32b56cc80d5bc2f12e584b152`/
  `c76e9285a41f65f0e9333dda2ef10a75bd1a17bf85538019ac3871d000b0c879`；
- exp395 actual script/result/runner/manifest SHA=
  `64840b710db587720aa8807571212b246af3eabb54306bd5aa1bbf692f5ea08b`/
  `cdffff60b1b6e04e6bb0b13bb54e12518380421675c59c2f2c785f1b7a5adb75`/
  `cdffff60b1b6e04e6bb0b13bb54e12518380421675c59c2f2c785f1b7a5adb75`/
  `3a0ef5d98dd6387b330958bbfb1e9d893e60745e8857237bbbbe375778886c64`；
- source/runtime/CLIP/codebook SHA沿用exp395冻结值；
- exp395失败仅为reporter大输入RuntimeError，不带gradient根因先验。

## 唯一变量

唯一变量是`gradient_report`的finite range实现：从全量`torch.cat + torch.quantile`改为固定chunk双遍扫描
加temporary memmap exact sort。loss、group、batch、forward、AMP、scale、optimizer、teacher、资产、
source和裁决规则均不得改变。

## exact percentile定义

对finite absolute values排序为`x[0] <= ... <= x[N-1]`。对q：

```text
r = (N - 1) * q
lo = floor(r)
hi = ceil(r)
value = x[lo] + (x[hi] - x[lo]) * (r - lo)
```

q固定为`0.50/0.95/0.99`。N=0时为`null`；N=1时三个值均为`x[0]`。只允许finite值进入排序。

## scratch边界

- scratch root必须位于新的exp396 audit目录内，启动前不存在；
- 文件必须是regular file，不得symlink到cache、旧runtime或其他路径；
- 每格最多一个FP64 memmap，名称只含arm/loss/stage/group的净化标识；
- 写入完成必须核对offset exact N，sort后只读取六个order-statistic位置；
- success、loss non-finite、exception三条路径均关闭memmap并删除scratch；
- 最终允许保留的actual输出仍只有result/runner/manifest。

## static执行

使用工作目录uv环境并设置`CUDA_VISIBLE_DEVICES=''`。超大case固定N=`16,777,217`，输入为解析可复核的
非负FP32单调序列；禁止为了速度减小N。小张量reference可以调用`torch.quantile`，但production
reporter源码不得含该调用。连续两遍必须逐字节一致，任何FAIL只修exp396且保留失败资产。

## actual执行

static封板后，用户持续授权允许自主执行一次新的CUDA门，不再等待逐次确认。仍必须：

1. fresh execution与exp396 regular资产逐SHA；
2. GPU compute process=`0`；
3. D0五行后rich十一行，十五组双时点全部自然完成；
4. 每行state/buffer/optimizer/RNG恢复；
5. teacher/codebook版本与state exact；
6. update=`0`、checkpoint=`0`、scratch=`0`；
7. result/runner/manifest SHA封板，外部进程与GPU终审。

任一前置FAIL在backward前停止；任何未捕获RuntimeError/OOM/state漂移立即停止并封板INVALID，不调参、
不补跑。记录到矩阵内的NaN/±Inf不是诊断器异常，必须继续完成预注册行。

## 结论边界

exp396 PASS只表示实际归因矩阵可信；它不授权exp394修补或formal训练。矩阵结果只能按exp395冻结分层规则
解释，不能把单一loss支持组写成唯一算子根因。

## Phase 0Q封板记录

独立static contract连续两遍33/33 PASS且四份result/runner逐字节一致。超大case固定
`16,777,217`元素，没有缩小；三处分位数与解析order statistic exact。小张量reference、multi-chunk、
non-finite分类、L2容差、输入不变、success/exception scratch清零、source/exp395资产、loss/group/
scale/zero-update与CUDA未初始化全部通过。

SHA256：

- CUDA implementation：`6e8b2b67efcda7cfaca8527fb0ae1dd4c6aedcebef3fec6ded2e5ba6ddab8164`；
- static contract：`f3a2ee3ccafa4caa1606b92b93b86177cc0b5ef6cfe7ac2b6f0d31fa195c415b`；
- v1/repeat result与runner：
  `e5d68df7731042a98f440f43acc45c9cf11b70aa7df25e09397ff6375f355394`。

该PASS只授权一次fresh exp396 CUDA归因，不授权训练或修改exp394/exp395。

## CUDA actual结果与协议裁决

actual完整满足本协议全部有效性门，result=`PASS`。D0与rich `reid/total`的唯一non-finite组均为
`backbone`，NaN/±Inf计数、finite范围及scaled/unscaled支持逐项相同；所有pose/semantic/exec
auxiliary行均finite。故按冻结规则裁决为`SHARED_D0_OR_RUNTIME_NONFINITE`，不得归因到rich teacher、
evidence head、router或loss聚合。

状态/隔离终审全部exact：D0/rich model与optimizer、teacher、codebook、RNG、source/assets、scratch、
update和checkpoint门均PASS。result/runner逐字节相同，SHA=
`58ae4beb56c9dabbff7fd77202d87b53f3ccecc9edec725051f04ed3c60ed96c`；manifest SHA=
`3a0ef5d98dd6387b330958bbfb1e9d893e60745e8857237bbbbe375778886c64`。

exp396至此封板，不得再次执行。后续新实验只能把D0作为matched数值基线，保持default initial scale并
观察canonical GradScaler自身skip/update轨迹；不得手工降低initial scale来改写本结果。
