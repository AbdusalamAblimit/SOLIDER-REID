# Codex Review — exp325 (+ exp324f)

**Verdict**: approve
**Date**: 2026-06-16 07:40
**Review round**: 1
**Tool**: `codex --search exec -s read-only`（联网查新颖性）

## Findings

- **Low** — `scripts/exp324f_swin_distmat.py:81` dump 的是 Swin **MaxSim hybrid** distmat（global cosine 与 part-MaxSim 1:1 混合，alpha=global_weight/(global_weight+1)=0.5），非纯 part-only。与项目惯例一致，但 writeup 应称"Swin MaxSim hybrid"避免歧义。
- **Low** — `scripts/exp324f_fuse.py:182` w=0 sanity 仅打印未断言匹配 npz 里存的 swin_alone_map/r1。因 z-score/min-max 是正仿射变换，w=0 必复现纯 Swin 排序，不阻断；加个 assert 可防未来回归。
- **Low** — `scripts/exp325_train_head.py:59` 注释措辞易误解（说 HIDDEN 需 import 后再 rebind），但实现正确（import exp324b 前 patch e324.HIDDEN，line 67 assert 验证 =1024）。代码对，注释可收紧。

## Checked（codex 确认正确）

- **exp325 monkeypatch 正确**：`e324.HIDDEN=1024` 与 `e324.load_model=_load_large` 在 import exp324b 前；exp324b 捕获 HIDDEN=1024，用 patched load_model()，`b.CACHE_TRAIN` 在 b.main() 读取前改绑。
- **DINOv2 几何正确**：HF config base hidden=768 / large hidden=1024，均 patch14；224×448 → 16×32=512 patch → (B,1+512,1024) reshape (B,32,16,1024) 成立。
- **单变量隔离 OK**：仅冻结 backbone / 原生特征宽度变；optimizer/sampler/loss/epochs/seed/eval/part pipeline 全继承 exp324b 默认。投影层 768→512 变 1024→512 是 backbone hidden 的机械后果。
- **exp324f 对齐稳健**：文件名 join + pid 全等 + camid 偏移恒定校验 + eval 用 Swin camid，杜绝静默错位。DINO 缓存/head 引用指向 exp324b 正确。
- **新颖性（web search）**：DINOv2/基础模型用于 ReID、score-level/late fusion、姿态引导可见部位匹配均有先例。未找到完全相同组合"frozen DINOv2 dense + 姿态锚定 5 部位轻量头 + mutually-visible part-MaxSim on Occluded-Duke"，但应**窄框为实验机制/诊断**，非广义新颖。来源：DINOv2 HF config、ECHO-BID、DinoGRL、Query Adaptive Late Fusion、PFD、PGFL-KD。

## 结论

codex 审查通过（verdict: approve）。3 个 Low 均非阻断；其中注释收紧（Low#3）已顺手修。exp324f 已作为 eval-only 跑完（负结果）；exp325 训练实验双审查通过，可开训。
