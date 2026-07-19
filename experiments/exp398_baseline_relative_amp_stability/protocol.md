# exp398 production-shaped baseline-relative AMP稳态协议

## 状态

`PROTOCOL-FROZEN / IMPLEMENTATION STATIC SEALED-PASS / STATIC-CPU SEALED-PASS /
CUDA FRESH-EXECUTION GO / FORMAL NO-START`

## 上游封板

- exp394=`CUDA_AMP_PREFLIGHT_SEALED_FAIL`；
- exp395=`CUDA_ATTRIBUTION_EXECUTION_SEALED_INVALID`；
- exp396=`CUDA_ATTRIBUTION_SEALED_PASS / SHARED_D0_OR_RUNTIME_NONFINITE`；
- exp397=`CUDA NATIVE-PARITY SEALED-FAIL / NATIVE_GRADSCALER_PARITY_FAIL`；
- exp397 result/runner/manifest SHA=
  `eef02328fb4026459fa28a7095d8d5c7b5703834e25ba950a78d9a3f1978faa2`/
  `eef02328fb4026459fa28a7095d8d5c7b5703834e25ba950a78d9a3f1978faa2`/
  `a9c2acd912f57a7020e129c59c4b24b615d1e4065f0bf482f71ad631eb7b3c51`。

exp398是新门，不修改、重跑或重新解释上述实验。

## 固定执行

```text
ATTEMPTS = 32
TAPF_EPOCHS = 1 × 16, 6 × 16
STAGE_TAIL = 8

for arm in (D0, rich):
    fresh model / optimizer / default GradScaler("cuda")
    for matched batch, matched RNG, frozen epoch:
        zero_grad
        autocast fresh forward
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        capture 15 groups
        scaler.step(optimizer)
        scaler.update()
```

禁止手工scale、retry、补step、clip、scheduler、checkpoint和loss重权重。D0/rich允许因不同辅助图在成功
update后参数分叉，不要求全轨迹exact。

## baseline-relative判定

- 两阶段最后8步必须各自连续success/finite；
- D0 success而rich skip的attempt数必须为0；
- rich成功update数必须`>=`D0；
- rich non-finite group必须是同attempt D0 non-finite group的子集；
- rich-specific 11组全程finite，且每组在e6成功步出现非零梯度并有state变化；
- 每行`had_nonfinite == optimizer_skipped`，skip只允许scale原生减半，success时scale保持；
- validity、teacher/codebook、scratch/checkpoint和退出审计全部通过。

## static/CPU前置

独立contract必须检查32步schedule、8步尾窗、default scaler无override、单一materialized loader、matched
RNG、15组、11个rich-specific组、每attempt一次step/update、zero checkpoint与sealed边界。synthetic
必须覆盖：

1. shared warm-up + shared e6 backoff + 双尾窗稳定 PASS；
2. rich在D0成功处extra skip FAIL；
3. 两臂共同在尾窗持续skip FAIL；
4. rich-only non-finite group FAIL；
5. rich-specific组无e6非零梯度或state不更新 FAIL；
6. rich比D0更早稳定且其余门满足 PASS。

static连续两遍逐字节一致后，按用户持续授权直接执行唯一fresh CUDA actual。

## 结论边界

PASS只授权另立final production preflight，不授权e120；FAIL只关闭当前baseline-relative稳态门。无论结果
如何，exp394–397保持sealed，禁止通过修改initial scale/loss/batch复活旧臂。

## static封板记录

连续两遍24/24 gate PASS，四份result/runner逐字节exact。implementation/static/result SHA=
`8bc599e94264eb3fb89b3cdc94810c483f4c4a037ebe86311c30a741011aeac9`/
`b137fadaf0463ae51eb2e552945cf87923a2c788582dbf0a4aaf00e296829414`/
`d7efa6894411f7b7433c8819422e14c2495110484544d86b9b21083d0bb24317`。该PASS只授权唯一fresh CUDA
baseline-relative执行，formal训练仍`NO-START`。
