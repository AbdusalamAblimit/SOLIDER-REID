# exp414 PSCIR 独立设计盲审

## 审查方式

- 原计划的Claude CLI在本机未认证，用户明确要求停止使用Claude并改由独立子agent审查；
- 子agent只读检查`design.md`及exp109--148、356、361、371、405、408--413必要旧记录；
- 审查范围严格限制为致命公式/张量/梯度bug、变量混淆与旧机制实质同构；
- 子agent未编辑文件、未连接远端、未运行训练或GPU任务。

## 结论

`PASS / 0 BLOCKER / 0 HIGH`

未发现公式、张量语义或梯度目标不可执行的问题；未发现anchor/support、正负身份或pose/CLIP轴混淆；与历史
prototype/teacher点补全、feature写回、跨图蒸馏、donor运输、pair mining、classifier proxy、support
multiplicity、梯度路由及离散prefix均值均不实质同构。PSCIR的训练对象确实变为anchor到同PID support
polyline region的全身份listwise距离。

## 非阻塞NOTE

1. 原合同“改变未包含第三边”不能解释为独立修改共享端点；实现合同改为固定MST索引，仅篡改未选候选边记录或其
   边权且保证仍未入选，再验证region distance不变。
2. `q_only`只是不读取在线visibility；q仍来自pose-defined region-isolated CLIP cache，最终只能称“无在线pose
   visibility轴”，不能称完全pose-free。
3. 若`TRP_L2=True`，线段是单位descriptor之间的欧氏弦而非球面测地线；这是预注册的几何假设，不构成执行阻塞。
4. 三节点完整图任取两条不同边必覆盖三节点，因此maximum-spanning-tree构造成立。

计数：0B / 0H

## 实现盲审

独立子agent随后只读检查default、loss、processor、formal config与唯一PK64 runner。首轮为
`0B / 3H`：

1. 线段距离用`sqrt(clamp_min(1e-12))`会把真实零距离抬为`1e-6`并截断近零梯度；
2. endpoint oracle只覆盖`t<0`，漏掉`t>1`上端；
3. combined-vs-host梯度变化不能单独证明纯region分支进入Stage-3。

已做最小修正：改用`torch.linalg.vector_norm`并增加线上精确零距离/finite backward；增加`[2,0]`上端clamp；
抽出未detach的纯`continuous_region_ranking_loss`并独立forward/backward验证Stage-3/norm3 finite nonzero。
复审结论=`PASS / 0B / 0H`，未引入新的公式、张量、梯度或合同错误。

计数：0B / 0H
