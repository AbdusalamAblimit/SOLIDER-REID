# exp415 PACIT子agent审查记录

## 审查边界

- 禁止使用Claude；
- 审查只拦截致命bug、变量混淆与exp401--414旧机制同构；
- 审查不连接GPU、不修改文件；
- 设计审查PASS不替代formal、geometry census、runtime smoke或512 oracle。

## 第一版：BLOCK

两路独立审查一致指出：

1. 同一CLIP选择属性/候选后又用自身margin验证，形成自证；
2. 所有control继承pose候选池，没有真正CLIP-only；
3. arm几何/D0难度、固定分母与batch64双视图合同未闭合；
4. oracle canonical view与production actual view定义冲突。

第一版未执行oracle、未训练、未占用CUDA。

## revision-2：BLOCK

revision-2切断了blind evaluator对CLIP的直接依赖，并建立instance-pose/fixed proposal形式2×2；但复审继续发现：

1. invalid P+ anchor bit单独进入Y，污染P因素；
2. actual CLIP encoder调用图未冻结；
3. interaction允许短数组，未硬断言512；
4. 难度仍为总体门而非逐图caliper；
5. strong controls可落ROA P90外或identity-unsafe；
6. canonical crop/Random Erasing后不再是同一语义事实；
7. 512到15,618失败图与共同NOOP未定义。

revision-2也未执行oracle、未训练、未占用CUDA。

## revision-3最终回归

最终三路只读回归一致：

`PASS / 0 BLOCKER / 0 HIGH / 0 OLD-ISOMORPHISM`

核验内容：

- P准确收窄为instance-pose center相对canonical anatomical anchor；
- 每图结果前固定active layer，P+/P-各7个一一对应shape；
- actual OpenCLIP selector从whole-image letterbox、image/text encode到centered margin完整冻结，公开调用只接受
  original RGB与7张edited RGB；
- blind evaluator不接收CLIP，并用CIELAB、4连通颜色、pose anatomy与D0 identity-safe裁决；
- 四条direct caliper edge齐全，任一arm/edge失败四factorial臂共同Y=0；
- 强control允许选择P+C同一mask，机制等价时不强迫次优；两条pair accumulator任一侧失败共同0；
- 固定512 row id、顺序、短数组拒绝与单臂故障注入均闭合；
- 四factorial+两strong control共同valid/NOOP，不drop sample；
- production关闭crop、padding与Random Erasing；全量builder预验证canonical/hflip两方向全部blind+D0+caliper；
- zero-owner只作全部训练臂共同强宿主，先double-view clean-pair再correct；
- structured erasing近邻已披露，联合贡献必须胜raw-color、D0-hard和factorial controls。

最终回归只授权下一阶段：

1. fresh formal；
2. synthetic contract；
3. 全15,618 geometry census；
4. 独立8图runtime smoke；
5. 全部SHA与GPU独占门。

上述机械门完成前，唯一512 oracle仍为`NO-START`。

## formal前runner实现复审

新增：

- `geometry_census.py`：全15,618图只读pose/RGB绑定与canonical/hflip几何普查；
- `runtime_smoke.py`：固定8图真实decode、P+/canonical-anchor各7编辑、OpenCLIP与sealed D0接口、
  cache/result原子回读。

首轮runtime复审发现并在执行前修复：

1. 删除smoke对formal oracle路径的参数与`exists`读取，唯一输出路径构造性锁死为
   `/home/afr/reid-clean/assets/exp415-pacit-smoke-v3`；
2. D0变体从`clean+pose7+ROA8`补全为`clean+pose7+canonical-anchor7+ROA8=23`；
3. 失败namespace不再删除，改为永久保留`failure.json`且禁止续跑；
4. `local_cfg`清空历史pretrain path后才构模，并以sealed D0 checkpoint `strict=True`加载；
5. device只允许逻辑`cuda:0`，在创建namespace前固定；
6. raw CLIP/D0 tensor不落盘，避免把8图smoke变成小样本科学oracle；只保存不可逆SHA，但真实内存tensor仍做
   shape/dtype/finite检查。

三路最终只读回归一致：

`PASS / 0 BLOCKER / 0 HIGH / 0 VARIABLE-CONFUSION / 0 OLD-ISOMORPHISM`

本地使用工作区`.venv`与`uv run`执行：

- 两个runner `py_compile=PASS`；
- `geometry_census.py --self-test=PASS`；
- `runtime_smoke.py --self-test=PASS`。

复审未连接远端或GPU。该结论只授权同步formal并依次执行CPU census与唯一fresh 8图smoke；不授权512 oracle。

## 完整512 oracle runner最终复审

最终审计字节：

`asset_oracle.py SHA256=b9083a6dd4923e0eec6c1b4f29e67813fc352b937d9a46363ff9b8583f7d836a`

三路独立只读复审全部完成：

1. 确定性与provenance路：`PASS / 0B / 0H`。固定环境设置发生在CUDA初始化前，状态回读、
   source/checkpoint/pose provenance、once-only seal与运行中二次SHA检查闭合；
2. 机制与变量路：`PASS / 0B / 0H / 0 variable-confusion / 0 old-isomorphism`。CLIP只从7个编辑候选
   选择颜色，blind evaluator只接RGB/pose/D0；raw-color、D0-hard保留同一P+C reference和完整
   caliper/identity门，未复活owner、prefix或MST同构；
3. 固定分母与统计/schema路：`PASS / 0B / 0H`。固定512有序row、四factorial与两strong control
   的共同complete/NOOP、P+/P-各35 proposal、active各7、ROA8、14个blind候选、五个factorial effect、
   10,000次paired bootstrap、agreement/top-5和全部原子输出/readback均闭合；失败row不丢弃，
   科学Y=0仍保留在固定分母。

三路均未修改文件、未连接远端、未触发GPU。该结论只授权最终formal显式同步、CPU自测、第二次GPU/namespace/
worktree preseal门和唯一512 oracle；不授权绕过oracle启动全量资产或e120。
