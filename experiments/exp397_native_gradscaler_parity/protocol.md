# exp397 matched native GradScaler动态轨迹协议

## 状态

`PROTOCOL-FROZEN / STATIC-CPU SEALED-PASS /
CUDA NATIVE-PARITY SEALED-FAIL / NATIVE_GRADSCALER_PARITY_FAIL /
FORMAL NO-START`。

## 冻结上游结论

- exp394保持absolute-first-step `SEALED_FAIL`；
- exp395保持reporter runtime `SEALED_INVALID`；
- exp396保持`CUDA_ATTRIBUTION_SEALED_PASS / SHARED_D0_OR_RUNTIME_NONFINITE`；
- exp396 actual result/runner/manifest SHA=
  `58ae4beb56c9dabbff7fd77202d87b53f3ccecc9edec725051f04ed3c60ed96c`/
  `58ae4beb56c9dabbff7fd77202d87b53f3ccecc9edec725051f04ed3c60ed96c`/
  `3a0ef5d98dd6387b330958bbfb1e9d893e60745e8857237bbbbe375778886c64`。

## schedule与数据

`ATTEMPTS=12`，`TAPF_EPOCHS=(1,1,1,1,1,1,6,6,6,6,6,6)`。唯一rich dataloader materialize前12个
official batch64，并保存每batch relative path/RGB SHA/PID/cam/view/input/pose/teacher RGB SHA。
D0与rich使用同一CPU tensor clone，禁止各自重新抽样。

## 原生GradScaler边界

每arm只允许：

```text
scaler = torch.amp.GradScaler("cuda")
for attempt:
    restore matched step RNG
    zero_grad
    autocast fresh total forward
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    capture 15 groups
    scaler.step(optimizer)
    scaler.update()
```

禁止传`init_scale`等参数、直接写`_scale/_growth_tracker`、手工retry同batch、gradient clipping、loss重权重
或在skip后补step。默认growth interval远大于12，故finite success时scale应保持，shared overflow skip时
只允许native backoff。

## matched判定

`optimizer_succeeded`由parameter `_version`与optimizer state变化联合确认；`optimizer_skipped`是其反值。
任何gradient group含NaN/±Inf即`had_nonfinite=true`。每行必须满足二者一致。

D0/rich轨迹键固定为：

```text
(attempt, tapf_epoch, scale_before, scale_after, optimizer_skipped)
```

两arm逐键exact；成功数各≥10，首个成功≤3且相同；attempt 7–12全部success/finite；rich-specific组全程
finite。若D0也在同attempt skip而rich支持完全不额外扩大，属于matched shared adaptation，不判rich FAIL。

## 输出与终审

只允许新的result/runner/manifest；不写checkpoint。result必须含两arm完整12行、所有group报告、scale与
update计数、teacher/codebook、RNG、source/runtime/assets、scratch、peak memory。结束后外部核查PID、
GPU空闲、repo clean、SHA与异常词。

## 结论边界

PASS只说明canonical native loss scaling下rich数值轨迹不劣于D0，并授权另立production preflight；
FAIL只关闭当前rich production dynamic parity。任何结果都不修改exp394/395/396封板资产，也不直接
授权formal训练。

## static封板记录

连续两遍21/21 gate PASS，四份result/runner逐字节exact。implementation/static/result SHA=
`4ad2c40a8d679e8dd52619d9216016aaecdc0fd6530d7ca679e0bb16b7cfa9ba`/
`99ad9a0d34db4bcbc0816ecd05c62d361322f47d214bca21c9927f92738269dd`/
`82d52315d1472e996fc50f330d332853c2e025ecf1c333651aca6cd7385f06eb`。该PASS只授权唯一fresh CUDA
native parity执行。

## actual封板记录

唯一actual自然完成。D0/rich的12步scale/skip轨迹逐项exact，rich-specific 11组全程finite；但两臂均
只有`6/12`次成功update，首个成功均为attempt 6，且e6首步再次shared-backbone skip。因此冻结门中的
minimum success、early first success、post-first-success finite与handoff all-success四项FAIL，协议
裁决固定为`NATIVE_GRADSCALER_PARITY_FAIL`。result/runner/manifest SHA=
`eef02328fb4026459fa28a7095d8d5c7b5703834e25ba950a78d9a3f1978faa2`/
`eef02328fb4026459fa28a7095d8d5c7b5703834e25ba950a78d9a3f1978faa2`/
`a9c2acd912f57a7020e129c59c4b24b615d1e4065f0bf482f71ad631eb7b3c51`。禁止以matched轨迹为由改判
exp397，也禁止补跑；formal训练不获授权。
