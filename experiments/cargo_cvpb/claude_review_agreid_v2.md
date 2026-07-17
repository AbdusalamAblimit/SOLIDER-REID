# Claude Broad Review — AG-ReID.v2 接线(--dataset agreid_v2)

**范围**:`afd_reid/agreid_v2_combined.py`(新)、`cargo_cvpb/afd_train.py` 3 处接线 hunk、
对照已验证的 `afd_reid/agreid_v2_dataset.py`。接线无新机制,审查聚焦"数据切换正确 + 不破坏
CARGO 路径"(任务明确要求)。

## 审查范围与结论(逐条)

### a. design.md 合理性 / 单变量
`design_agreid_v2.md` 清晰:动机=跨数据集第二支柱(codex 硬要求),假设=AIRL 机制非 CARGO 特例,
单变量=AIRL on/off(同机同配置)。**这不是小调参逃避创新**——是已定稿 CARGO 机制(+2.37)的
跨数据集外部验证,方法稿实证厚度必需。诚实预期(headroom 小可能涨幅缩)已写入。PASS。

### b. 新增代码逐行(agreid_v2_combined.py)
- `AGReIDV2Combined.__init__`:加载 exp1(A→G)+exp4(G→A)两个已验证 `AGReIDV2` 实例。
- train 取 exp1 的 `.train`(协议无关),并 assert exp1/exp4 train pid 数 + img 数一致(防 loader
  漂移)。PASS。
- 共享 test pid 映射:对 exp1.query/gallery + exp4.query/gallery 全部 folder-name 取并集 →
  `sorted` → folder-name→int。query 与其 gallery match(同 folder-name 字符串)→ 同 int pid,
  两方向一致。PASS。
- query = exp1.query(view=Aerial)+ exp4.query(view=Ground);gallery = exp1.gallery(Ground)
  + exp4.gallery(Aerial)。`_relabel(expect_view=...)` 对每个样本 assert view,守住布局契约
  (filter_by_view 依赖)。PASS。
- `filter_by_view(query,'Aerial')`=exp1 A query;`filter_by_view(gallery,'Ground')`=exp1 G
  gallery → A→G=官方 exp1。`filter_by_view(query,'Ground')`=exp4 G query;
  `filter_by_view(gallery,'Aerial')`=exp4 A gallery → G→A=官方 exp4。**exp4 的 Aerial gallery
  不会污染 A→G**(A→G 的 gallery 被 filter 到 Ground,排除了 exp4 的 Aerial)。PASS。

### c. 配置 / 接线 hunk(afd_train.py 三处)
- L153 import `AGReIDV2Combined`(afd_reid 已在 sys.path 上,L148-149)。
- L1241-1242 `--dataset` choices 增 `agreid_v2`,default 仍 `cargo`。
- L1749-1753 `elif args.dataset == 'agreid_v2'` 分支,cargo 分支(L1747)在前、未动。PASS。

### d. defaults / 破坏性
AST 检查:agreid_v2_combined.py 顶层无可执行语句(仅 import/class/def,`if __name__` 不在
import 时运行)→ import 无副作用。`--dataset cargo` 行为字节级不变(import 增加是唯一恒执行改动,
且无副作用)。PASS。

### e. processor / eval / loss
**零改动**。run_cross_view_eval / airl_dualbranch_eval / ovli_rerank_eval / eval_market /
extract_features 全部复用 CARGO 路径。关键:extract_features 做 `torch.cat(pids).numpy()` 需
int pid——combined 的 q/g pid 已是 int(共享映射),smoke 实测 collate 成 int tensor 通过。
eval_market 的 same-(pid,camid) junk removal 在 A→G/G→A 跨平台方向为 no-op(query cam0 vs
gallery cam3,永不同 cam)。PASS。

### f. 对照隔离
唯一变量 = AIRL on/off。两实验同机(尽量)、同 Swin pretrain、同 256×128/bs64/60ep。AIRL-iso 用
`--airl_iso_trunk_recce 0`(全 detach WORKING,= CARGO 定稿),非 iso2 的有害 `1`。PASS。

## 验证证据
- smoke_agreid_v2_wiring.py(CPU,lab-3090,不扰动 GPU 训练)**全过**:官方计数 2356/6347 &
  1811/14362、534/534 query id 可匹配、全 int pid、collate int tensor、eval_market 100/100/100。
- agreid_v2_combined.py `__main__` 直接跑(lab-3090)输出官方计数完全正确。
- py_compile 三文件通过。
- 接线时 lab-3090 正在跑 iso2(ep43+),覆盖 afd_train.py 后实测训练未受影响(进程已 import 入内存)。

## 分级与结论
- Critical / High / Medium / Low:**均无**。
- **审查通过**。接线正确、CARGO 路径不破坏、AIRL/iso/eval 逻辑零改动。可启动训练。
