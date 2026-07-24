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
