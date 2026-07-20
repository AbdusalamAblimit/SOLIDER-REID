# exp404 SPK 冻结协议

## 0. 当前边界

standalone、production CPU/source、actual CUDA v3与formal prelaunch门均已通过。当前只允许通过冻结的
`formal_once_wrapper.sh`启动唯一fresh seed1234/e120；启动后自然跑满，不按中间性能早停、不续训、不修改
运行中代码/config。

v1已因5-slot region field误接17通道D0 gate封板`SEALED-INVALID`，禁止重跑。修复后production v3与v2 static
门通过；actual CUDA v2使用`cuda_amp_preflight_v2.py`和fresh v2 output完成并封板。

v2已因4次窗口内默认GradScaler持续backoff、无optimizer update封板`CUDA_AMP_PREFLIGHT_FAIL`，禁止重跑。
v3只能使用`cuda_amp_preflight_v3.py`：初始scale保持默认，最多8次自然backoff，仍须实际更新及全部原门通过。

v3已在第5次自然attempt通过全部26门。formal只能通过`formal_once_wrapper.sh`的fresh v1锁启动；不得手工绕过
output/runner/launch/lock、remote clean、source/config/runtime/preflight SHA与独占GPU检查。

## 1. 不变量

- backbone=`Swin-Tiny`，batch=`64`，seed=`1234`，epoch=`120`，workers=`8`；
- official data只读`/mnt1/afrdata`，冻结pose只读`/mnt1/afrderived`；
- teacher epochs/handoff、pose/evidence supervision与optimizer recipe沿用sealed clean链；
- SPK group=`16`，group width=`48`，不设temperature/scale/rho；
- fresh output/assets/execution，不resume、不best-pick、不按中间性能早停；
- exp394–403任何代码/config/result/checkpoint均不修改。

## 2. static正合同

连续两次byte-exact，至少验证：

1. CUDA未初始化、torch CPU-only执行；
2. `768 -> 16 x 48`固定映射exact；
3. SPK没有参数、projection、bias、concat或additive branch；
4. NULL evidence产生全1 factor，输出逐元素exact等于输入global feature；
5. correct/wrong/generic/random-key/random-cluster factor finite、非负、均值exact为1且干预active；
6. correct factor对构造的semantic-aligned positive utility优于所有null controls；
7. global feature与correct evidence均获得finite nonzero梯度；
8. evidence-ignored、auxiliary-only、additive-bypass三个mutant全部被抓；
9. deterministic donor为same-camera、different-PID、无fixed point；
10. deterministic random-key保持每sample evidence范数与绝对值多重集；
11. random-cluster count exact平衡，每簇PID覆盖`>=40`、camera覆盖exact 2；
12. source与result SHA写入monitor。

任何失败都保持`GPU NO-START`；once-only scientific gate失败不得换seed、降门或补跑。

## 3. 生产CPU/source门

static PASS后另行实现并冻结：

- config开关默认False，off-parity相对当前HEAD逐tensor exact；
- C0 expert与ELO operator在SPK图中均不存在；
- default D0 pose path不变，semantic head只向final SPK提供student evidence；
- train classification/triplet与eval descriptor均读取同一`D(e)`；
- teacher、generic/random control资产不进入model state/checkpoint；
- strict reload、optimizer覆盖、NULL/bypass exact与三种mutant在production shape再通过。

## 4. CUDA/AMP preflight

只有production CPU/source PASS后才创建fresh preflight编号资产：

- 先确认4090无其他任务；
- real batch64/default GradScaler，沿用exp399稳态比较纪律；
- SPK correct/NULL/random-key在rho无关条件下finite/active；
- evidence与16个feature group均有finite梯度，新增参数仅来自student evidence head；
- eval teacher/pose/codebook访问0，checkpoint=0，输出fresh；
- runtime必须由`runtime_requirements.txt`在fresh exp404 uv venv中构建，不得调用旧实验runtime；
- preflight自然退出并postflight全部PASS后才可授权唯一formal。

## 5. formal once-only与终审

唯一fresh e120自然跑满。最终八类arm必须逐臂全量覆盖query/gallery，state/RNG/patch/source/config/checkpoint
逐臂恢复exact；正式按design.md两级门裁决。runtime测量器错误只封板该执行记录，新编号修contract；scientific
FAIL不调temperature/loss/batch、不中途删control、不以新seed补跑。
