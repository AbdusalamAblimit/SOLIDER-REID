# AG-ReID.v2 cross-dataset reproduction (AIRL 方法稿第二支柱)

## 动机
CARGO 主结果已定稿:全 detach 梯度隔离 iso(`--airl_dualbranch_iso --airl_iso_trunk_recce 0
--airl_fuse_w 0.25`)在 CARGO A↔G 上 FUSE 净超 baseline **+2.37**(固定 w),方向感知 **+3.76**
(`airl_iso_result.md` ep60 定稿)。codex 反复强调跨数据集主表第二列是硬要求——单数据集不足以
支撑方法稿。**AG-ReID.v2 是真实低清航拍-地面 ReID**(SOTA 81-88),用它复现验证 AIRL 是否仍净超
baseline,补足实证厚度。

## 核心假设
AIRL 的"clean/recover 双 head 梯度隔离 + 软融合"机制不是 CARGO 特例;在 AG-ReID.v2 上
AIRL-iso 的 FUSE mean 仍 ≥ baseline-Swin mean。**诚实预期**:AG-ReID.v2 是真实低清、headroom 比
CARGO(合成跨视角)小,涨幅可能缩;若不超,如实报。

## 技术方案(接线,无新机制)
**只改数据切换,AIRL/iso/eval 逻辑零改动。**

1. 新增 `afd_reid/agreid_v2_combined.py::AGReIDV2Combined`:包装已验证的
   `agreid_v2_dataset.AGReIDV2`,加载官方两协议:
   - A→G = exp1 aerial_to_cctv(query C0 UAV 2356 → gallery C3 CCTV 6347)
   - G→A = exp4 cctv_to_aerial(query C3 CCTV 1811 → gallery C0 UAV 14362)
   - 合并 query/gallery 使 `filter_by_view` 精确还原两个官方方向(`run_cross_view_eval` 的
     A→G=q_aerial vs g_ground 即 exp1,G→A=q_ground vs g_aerial 即 exp4),mean 起来 =
     AG-ReID.v2 跨平台 mean(对应 CARGO 的 A↔G mean)。
   - test pid 用 exp1+exp4 共享的 folder-name→int 映射(extract_features 需 int pid)。
2. `cargo_cvpb/afd_train.py` 3 处接线 hunk(import / `--dataset` choices 增 `agreid_v2` /
   selection elif),`--dataset cargo` 默认路径字节级不变。

数据流:train=AG-ReID.v2 train_all(51530 img / 807 pid,协议无关);eval=官方 exp1+exp4 子集。
其余(Swin backbone / GeM-avg / BNNeck / CE+triplet / PK sampler / warmup-cosine / AMP / AIRL-iso
软融合 eval)全部复用 CARGO 路径。

## 两个实验(均 AG-ReID.v2,均 Swin-Small + SOLIDER pretrain,256×128,bs64,60ep)
1. **baseline-Swin**(无 AIRL):
   `--dataset agreid_v2 --backbone swin_small --swin_pretrain .../swin_small.pth --img_size 256 128`
2. **AIRL iso**(全 detach 正确版,= CARGO 定稿配置):
   `--dataset agreid_v2 --airl_dualbranch_iso --airl_iso_stage 3 --airl_iso_trunk_recce 0
    --airl_fuse_w 0.25 --backbone swin_small --swin_pretrain .../swin_small.pth --img_size 256 128`

注意:`--airl_iso_trunk_recce 0`(全 detach,WORKING),不是 iso2 的 `1`(有害消融对照)。

## 预期结果
- baseline-Swin:AG-ReID.v2 mean 比 CARGO 高(真实数据,SOTA 81-88;但 cross-platform mean 含
  G→A 难方向,具体看跑)。
- AIRL-iso FUSE mean ≥ baseline,机制立则净超(对应 CARGO +2.37)。
- 失败最可能原因:AG-ReID.v2 headroom 小(真实低清,occlusion/视角降质本来就被训练吸收),
  AIRL 互补空间被压缩 → 涨幅缩或打平。**如实报,不粉饰。**

## 对照组
- 对照 baseline = 同机同配置 baseline-Swin on AG-ReID.v2(唯一变量 = AIRL on/off)。
- 与 CARGO 的 +2.37 对照(同机制、不同数据集,看是否复现)。

## 审查
- 接线无新机制,审查聚焦"数据切换正确 + 不破坏 CARGO 路径"。
- smoke(CPU):`smoke_agreid_v2_wiring.py` 全过(官方计数 2356/6347 & 1811/14362、int pid collate、
  selection branch、eval_market sanity)。
- claude_review_agreid_v2.md + codex_review_agreid_v2.md(codex approve / 0 findings)。
- py_compile 通过。
