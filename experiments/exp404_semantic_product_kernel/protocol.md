# exp404 SPK 冻结协议

## 0. 当前边界

standalone、production CPU/source、actual CUDA v3、formal prelaunch及唯一fresh seed1234/e120均已完成。
训练封板后不得重跑、续训或修改训练代码/config。当前唯一活动是sealed e120 checkpoint的
`exp404-spk-counterfactual-v1`九臂终审。

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

唯一fresh e120已自然跑满。最终九臂必须逐臂全量覆盖query/gallery，state/RNG/patch/source/config/checkpoint
逐臂恢复exact；正式按design.md两级门裁决。runtime测量器错误只封板该执行记录，新编号修contract；scientific
FAIL不调temperature/loss/batch、不中途删control、不以新seed补跑。

### 5.1 v1冻结顺序

1. `correct`；
2. `wrong_rgb`：same split/same camera/different PID donor的evidence与presence；
3. `generic_mean`：train-split RGB-only frozen pooled-evidence mean；
4. `null_zero`；
5. `all_product_bypass`；
6. `random_key`：absolute-index hash确定的逐样本signed permutation；
7. `random_cluster`：8个generic signed-permutation原型、hash平衡分配；
8. `wrong_mask`：只循环SPK presence；
9. `slot_cycle`：只循环evidence。

NULL与all-product-bypass的最终descriptor和四项metric必须逐元素/逐值exact。所有九臂必须finite；wrong、
generic、NULL、bypass、random-key与random-cluster六个主control必须相对correct active。wrong-mask与slot-cycle
保留为补充归因并完整报告，在uniform hard presence下允许按SPK定义不变。random-key须保持逐样本每槽范数和
绝对值多重集；random-cluster须满足8簇、count最大差1、每簇PID覆盖`>=40`且camera覆盖等于验证集全集。

### 5.2 执行资产和启动门

- 本地资产：`counterfactual_core.py`、`actual_counterfactual_audit.py`、
  `counterfactual_static_contract.py`、`counterfactual_postflight.py`、
  `counterfactual_once_wrapper.sh`；
- 远端fresh审计根：`/home/afr/reid-clean/audits/exp404-spk-counterfactual-v1`；
- 只读checkpoint：formal output中的唯一`transformer_120.pth`；
- runtime固定为fresh exp404 runtime；
- 先连续两次CPU/static byte-exact，再做fresh小样本CUDA wiring preflight；
- 正式result/runner/manifest/lock任何一个预先存在都禁止启动；GPU必须无compute PID。

### 5.3 CUDA wiring preflight执行编号

- `preflight-v1`已封板`SEALED-INVALID_CONTRACT`：九臂、资产恢复和主control均正常，但reporter错误要求
  wrong-mask/slot-cycle active；result/runner不得覆盖或重跑；
- `preflight-v2`是fresh执行，只允许将active门限定回六个主control；九臂顺序、输入、数据、checkpoint、
  random资产、主mAP门与formal once-only资产全部不变；
- v2全部validity PASS后才授权formal full；v2 scientific/main-control失败不得继续。
