# exp375 PRSM 直接先例与声明边界

## 最近邻

- TM-Mamba（arXiv:2404.11375）已让外部文本条件共同生成 `B/C/Delta`，覆盖一般
  conditional selective SSM；
- Hamba（arXiv:2407.09646）已用预测关节定位采样视觉 token，并执行 graph-guided
  bidirectional Mamba scan；
- PoseMamba（arXiv:2408.03540）、PS-Mamba、SasMamba 与 MeshMamba 已覆盖骨架、部位、
  图结构或人体模板引导的扫描/序列化；
- SAMA（arXiv:2507.19852）已用关节运动控制 `Delta`，并按骨架邻接融合 hidden states；
- ReIDMamba（arXiv:2511.07948）及 MambaReID/MambaPro/ReMamba 已覆盖 Person ReID 与
  Mamba；
- Tac-Mamba（2026, DOI:10.3390/electronics15071535）已经使用 pose-guided cross-modal
  state-space 标题，但 pose 主要用于 skeleton teacher 蒸馏与可靠性融合，不直接路由 RGB
  patch 的 recurrent write/retain。

## 不能声称

不能写首个 Mamba ReID、首个 pose×Mamba、首个人体拓扑/解剖扫描、首个 graph Mamba、
首个条件 selective update、首个 pose-guided cross-modal Mamba，也不能把 `Delta=f(pose)`
本身当作新函数类。

## 可争边界

截至本轮检索，没有发现工作直接覆盖以下完整组合：单图 RGB 遮挡 ReID；当前图像的 2D
pose soft-part/visibility；由它直接路由 RGB token 对多个部位状态的 write/retain；读取端
保持 RGB-only；并用 matched-pose shuffle 证明收益依赖正确 image-pose correspondence。

因此 PRSM 的主张必须落在“pose-observable state corruption + instance-pose-routed visual
part memory + 因果对应性证据”，而不是泛化的 pose Mamba 或扫描顺序。
