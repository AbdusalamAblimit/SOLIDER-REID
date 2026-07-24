# exp415 PACIT revision-3执行协议

## 固定资产

- interpreter：`/usr/local/anaconda3/envs/mmpose-abu/bin/python`；
- official data：`/mnt1/afrdata`只读；
- pose：`/mnt1/afrderived/exp386_occluded_duke_vitpose_huge_train`只读；
- pose manifest SHA：
  `cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8`；
- CLIP checkpoint：
  `/home/afr/reid-clean/weights/exp401_clip_l14_openclip_9ce2e8a8.safetensors`；
- CLIP SHA：
  `9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e`；
- D0 SHA：
  `59017755d61370754aa2e852a487d8e242fcee8814685f77f5388ba3a430e069`；
- oracle只写fresh
  `/home/afr/reid-clean/assets/exp415-pacit-oracle-v3`与独立runner；
- smoke只写
  `/home/afr/reid-clean/assets/exp415-pacit-smoke-v3`，不得创建/读取oracle namespace。

## A. 本地静态门

必须全部PASS：

1. py_compile；
2. 512 sample hash对input record顺序不敏感；
3. P+/P-各35 proposals，预定active anchor各7，index/area/aspect一一对应；
4. fixed generator函数签名与调用图不接受pose；
5. actual selector从标准whole-image letterbox/preprocess、OpenCLIP encode到centered-margin源码完整冻结；
   公开调用只接受original RGB与7张edited RGB，不接受pose/slot/mask/D0；
6. blind evaluator不import/接收CLIP；
7. invalid pose fallback不向Y输入arm-specific bit；
8. achromatic fill全部unclassified，mask外byte-exact；
9. 四条直接逐图caliper边与共同quartet-zero规则；
10. D0-hard/raw-color都受同一identity/severity caliper；
11. row accumulator硬断言512个有序相同row id，故障row仍输出Y=0；
12. 512条factorial/bootstrap，不允许4条toy数组代替正式统计合同；
13. full-asset四factorial+两strong control任一失败时六臂共同NOOP、P×K不drop；
14. 三路复审`0B/0H/0 old-isomorphism`。

## B. formal前只读机械门

1. fresh formal clone，记录HEAD与tracked worktree/index=`0/0`；
2. 在formal运行不读official的synthetic contract；
3. CPU geometry census覆盖15,618，只报告机械可达性，不计算Y；
4. fresh smoke namespace固定hash前8图，只验证真实接口、finite与I/O回读；
   - 唯一输出目录固定为`/home/afr/reid-clean/assets/exp415-pacit-smoke-v3`，脚本不接收、不读取、
     不创建oracle路径；
   - D0接口固定覆盖`clean + pose7 + canonical-anchor7 + ROA8=23`个变体；
   - CLIP/D0真实tensor只在内存检查shape/dtype/finite，落盘仅保存不可逆SHA，禁止保存score、winner、Y、
     agreement、rate或GO；
   - cache/result使用专属smoke schema并原子写入、exact回读；失败保留`failure.json`且
     `resume_allowed=false`，不得清理后同名重跑；
5. smoke自然退出后封存其result/runner SHA，oracle namespace仍不存在；
6. 核验唯一4090 CUDA任务=`0`；
7. 冻结source/prompt/scorer/prototype/caliper/schema/checkpoint全部SHA；
8. 才创建oracle once-only started seal。

本轮最终冻结源：

- runner：
  `b9083a6dd4923e0eec6c1b4f29e67813fc352b937d9a46363ff9b8583f7d836a`；
- core：
  `15ad21a7a79dc59819cee61a2971bf334f2683bb1ea77d71e0d6b155c3020311`；
- selector：
  `4b10a9899c203e51e67fed9dbe119d8f52150c60b8e41ffef9c68fc366bf78a9`；
- prompt：
  `1fb55c6ca451e132084293c9c583cbcab4ee3e45b993a6f7fbaf672dd99e60bd`。

正式命令必须同时显式传入最终formal HEAD与以上四个SHA，并设置
`PYTHONDONTWRITEBYTECODE=1`、`PYTHONHASHSEED=0`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`；
解释器必须使用固定路径并带`-B`。runner在创建once-only namespace之前和写入started seal之前各检查一次
GPU无compute process。

## C. 唯一512 oracle

- 一次执行完整512，不按中间rate早停；
- 不改代码、prompt、prototype、阈值、候选、样本或output；
- 每图写四factorial臂、两个强control、8 ROA或显式失败码；
- failure不reselect、不丢row；
- optimizer update=`0`、checkpoint=`0`；
- 严格异常：
  `Traceback|RuntimeError|CUDA error|out of memory|segmentation fault|killed|(^|[^[:alpha:]])(nan|inf(inity)?)([^[:alpha:]]|$)`；
- PID消失先分类自然退出、脚本异常、基础设施中断或用户终止，禁止续跑同一output。

## D. oracle终审

1. `ORACLE_EXIT=0`且GPU恢复空闲；
2. 512个相同有序row id与固定分母；
3. proposals、ROA、mask SHA与外部资产provenance完整；
4. matched quartet、五个factorial效应、bootstrap、agreement、top5与强control逐项裁决；
5. strict异常0、optimizer update0、checkpoint0；
6. result/summary/manifest/runner SHA；
7. formal tracked worktree/index=`0/0`。

任一科学门失败为`ASSET NO-GO`；执行/SHA/row不完整为`ORACLE INVALID`。两者均禁止重跑和e120。

## E. oracle GO后

1. 用同一frozen builder生成全15,618四factorial+两strong control共同manifest；
2. 任一arm失败时所有arm该图统一clean-NOOP，禁止drop；
3. full common-intervention-valid率`>=70%`才封存全量资产；
4. Random Erasing、padding与crop关闭；全量builder预验证canonical/hflip两方向的blind+D0+全部caliper，
   训练只在两个封存方向中选择；
5. 先double-view clean-pair e120，再P+C e120；
6. correct过`+.5 mAP/+.5 R1`双门后才串行controls；
7. controls与三seed顺序、终门完全按design，不临场修改。
