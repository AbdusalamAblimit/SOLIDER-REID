# exp411 PCMPSR 监控

## 2026-07-21：对象选择与设计冻结

exp410唯一fresh correct自然e120=`45.0/56.4/71.3/76.7`，相对clean D0全面大幅下降，已永久封板为
`SEALED NO-GO / PERFORMANCE FAIL`；GPU恢复`2 MiB/0%/0 compute PID`。下一对象明确禁止固定CLIP classifier、
projection/adapter救臂、局部relation重试或单hard-pair微调。

独立候选审计推荐PCMPSR：在每个PK batch内为每个身份构造等大小leave-one-position-out三图支持集，五槽pose
coverage与同PID CLIP槽共识只离散选择slot owner；final student descriptor对全部16个身份集合做listwise排序。
learned CE、D0 pose loss、student坐标与eval保持不变。该对象同时回应exp409的R1/AP分裂与exp410的外部轴错配。

近邻审计把SupCon、lifted/listwise metric、episodic set loss、pose-aware sampling和CLIP-ReID/ProFD列为已有原子；
未发现“等支持leave-one-position-out身份集合+五槽pose×CLIP owner multiplicity+final student全身份排序”的同构
实现。问题门PASS、证据门PASS、机制门CONDITIONAL PASS，定位C类候选。当前状态=
`DESIGN/PROTOCOL FROZEN / IMPLEMENTATION NEXT / GPU IDLE`。

## 2026-07-21：实现与盲审闭环

已完成default-off PCMPSR config、fresh cache builder/strict loader、等支持set/owner构造、FP32 listwise loss、
`make_loss`与processor接线；model/eval零修改。本地synthetic PK64合同PASS：support/owner shape=
`[64,16,3]/[64,16,5]`，owner unique mean=`2.421875`，wrong-RGB/generic/pose-only owner change=
`0.096875/0.059375/0.05625`，listwise loss与final feature梯度finite/nonzero。

独立智能体盲审首轮`0B/2H`，指出pose-invisible owner与真实default-off/isolated梯度合同缺口。两轮聚焦修复后最终
`0B/0H`：正常owner严格`visibility>0 & clip_valid`，显式pose-first fallback单独报告；唯一真实PK64脚本已冻结为
同时检查D0-vs-default-off四类RNG/state/forward/loss exact、isolated PCMPSR descriptor/Stage-3/backbone梯度及
combined native AMP update。当前状态=`IMPLEMENTATION REVIEW 0B/0H / FRESH CACHE NEXT / GPU IDLE`；真实CUDA
合同仍待cache后一次执行，尚无exp411性能结果。

## 2026-07-21：fresh cache已启动

relay恢复后，从exp410 formal clean基底建立fresh远端repo=
`/home/afr/SOLIDER-REID-exp411-pcmpsr-feb56c1-v1`，显式传输目标文件并提交；运行source HEAD=
`ebf60f2b4a5c943958f7077779d8500c2855874a`，关键loss/builder/real-batch/config SHA与本地byte-exact。启动前repo
tracked/untracked均clean、CLIP checkpoint SHA=`9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e`、
GPU=`2 MiB/0%/0 compute PID`，asset与runner路径均fresh。

唯一cache主PID=`574498`，asset=`/home/afr/reid-clean/assets/exp411-pcmpsr-cache-v1`，runner=
`/home/afr/reid-clean/train-logs/exp411-pcmpsr-cache-v1.runner.log`。首次观测已编码`8/15618`，GPU约
`2186 MiB/94%`且只有该compute PID，无异常。当前=`FRESH CACHE RUNNING / SOURCE+PARAMETERS FROZEN`；只监控
自然完成，不修改builder/source/参数，不复用exp408/409/410 cache。
