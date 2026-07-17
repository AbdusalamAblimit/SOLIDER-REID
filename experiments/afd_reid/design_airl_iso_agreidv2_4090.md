# 实验 AIRL-iso @ AG-ReID.v2 (lab-4090) — directional evidence specialization 命门验证

> 本文档对应 lab-4090 RTX 4090 上的 AIRL gradient-isolated dual-branch 运行，与 lab-3090 的 baseline-Swin 并行。
> 验证 **paper point 4（directional specialization，codex 判 5/5 主贡献）** 是否在 AG-ReID.v2 复现 CARGO 上观察到的方向特化。

## 动机
- CARGO 上 AIRL dual-branch（clean f_full + recover f_rec，固定 prior 软融合）已验证 +2.37~3.76 mean。
- CARGO 上观察到**方向特化（directional evidence specialization）**：clean(full) head 强 A→G、degradation-robust(rec) head 强 G→A（机制级现象，非偶然）。
- 命门问题：这个方向分化是 CARGO 的偶然，还是 AIRL 机制的内在性质？→ 在第二个 aerial-ground 数据集 **AG-ReID.v2** 上复验。

## 核心假设
AG-ReID.v2 baseline（lab-3090）A→G/G→A 是**均衡**的（ep10: 73.39 / 73.97，mean 73.68）。
若 AIRL iso 引入后：
1. **出现方向分化**：f_full 与 f_rec 在 A→G vs G→A 上拉开差距（打破均衡），且
2. **净超 baseline**：FUSE mean ≥ baseline mean，
则证明 directional evidence specialization 是 AIRL 机制的内在性质，不是 CARGO 偶然。

## 技术方案
- 代码：`experiments/cargo_cvpb/afd_train.py`（150KB，与 lab-3090 md5 一致 `6cf8fef9...`）+ `agreid_dataset.py`（`889ed7f7...`）+ `agreid_v2_combined.py`（官方 exp1 A→G + exp4 G→A）。
- 机制（`--airl_dualbranch_iso`）：f_rec 是从共享 trunk 在 iso_stage 处 fork 出的**独立晚期 Swin stage** 上的 BNNeck。degradation-CONSISTENCY 梯度只更新 rec 晚期 stage + BNNeck_rec，**绝不回流共享 trunk**（degraded pass 从 DETACHED trunk fork）→ f_rec 保持 "recover expert"，避免 +0.06 collapse。
- 本次用 `--airl_iso_trunk_recce 0`（**原始全隔离消融**：clean ID-CE 也 detach，不回流 trunk）。注意这是消融臂，不是默认 fix（默认=1 会把 clean rec ID-CE 回流 trunk 强化 f_full）。
- eval：报每方向 `{full, rec, fuse}` mAP/R1，fuse `cos = 0.25*cos_rec + 0.75*cos_full`（固定 prior，非 test-tuned）。

## 启动命令
```
CUDA_VISIBLE_DEVICES=0 python3 afd_train.py \
  --data_root /home/afr/SOLIDER-REID/data \
  --dataset agreid_v2 \
  --out_dir /home/afr/SOLIDER-REID/log/cargo/airl_iso_agreidv2_4090 \
  --backbone swin_small \
  --swin_pretrain /home/afr/SOLIDER-REID/pretrained/swin_small.pth \
  --airl_dualbranch_iso --airl_iso_stage 3 --airl_iso_trunk_recce 0 --airl_fuse_w 0.25 \
  --img_size 256 128 --test_batch 64
```
- `TEST.IMS_PER_BATCH` 等价物 = `--test_batch 64`（此脚本无 yacs，用 argparse；CLAUDE.md 的 64 铁律对应此项）。
- GPU 0，log → /tmp/agreidv2_airl_4090.log。

## 预期结果
- 假设成立：ep20+ FUSE mean ≥ baseline 73.68，且 full/rec 在两方向出现分化（如 rec 强 A→G、full 强 G→A）。
- 失败最可能原因：
  - iso_trunk_recce=0（全隔离）历史上 ep20 f_full 偏弱（CARGO 上 45.56 < baseline 48.98），FUSE 可能不超 baseline → 但**方向分化现象本身**仍可观察（这是命门，不一定要净超）。
  - 方向不分化（full/rec 在两方向同强同弱）→ CARGO 的方向特化是偶然，point 4 不成立。

## 对照组
- baseline-Swin（lab-3090，`/tmp/agreidv2_baseline.log`）：ep10 mean 73.68（73.39 / 73.97 均衡）。
- 消融变量：仅加 AIRL iso（单变量），backbone/img_size/test_batch/数据完全一致。

## 数据/环境就绪确认（2026-06-24）
- lab-4090 在线，GPU 0 全空（2 MiB）。
- 代码 md5 与 lab-3090 一致（afd_train.py / agreid_dataset.py）。
- swin_small.pth 1.15GB 已在 pretrained/。
- data/AG-ReID.v2 经 OSS（oss://agreid_v2.zip 1002MB）重新解压：807/808/808 ids，51530/8499/40473 jpgs（旧的 1.4M 残缺版已删）。
- 注：task 给的 "md5 f916b8ef..." 实为 OSS 多段 ETag（`-34` 后缀），非内容 md5；实际下载内容 md5=23d4af73...，解压计数全对，数据完整。

## 起跑踩坑记录（2026-06-24，三次才起来）
1. **python 环境**：lab-4090 系统 `/usr/bin/python3` 无 torch/numpy。SOLIDER-REID 自己没 .venv/pyproject。解决：用 afr 现成 uv venv `/home/afr/reid-clean/.venv/bin/python`（torch 2.6.0+cu124，含 numpy/timm/yacs，cuda 可用）。**今后 lab-4090 跑这套必须用这个 python，不是 python3。**
2. **协议文件缺失**：OSS zip 只含图像，缺官方协议 `exp{1,2,4,5}_*.txt`。combined loader 需 exp1(A→G)+exp4(G→A)。从 lab-3090 tar+base64 经本地中转推过去（337KB，md5 5e744874... 三处一致），解压进 data/AG-ReID.v2/。
   - 注：OSS 客户端只收 .zip（拒 .tgz "Unsupported file type"），lab-3090 又无 zip 命令 → 改走 base64-over-ssh 本地中转。
3. **依赖文件分布**：afd_train.py（cargo_cvpb）`sys.path.insert ../afd_reid`，复用 cargo_dataset/agreid_v2_combined/afd_model（在 afd_reid/，与 lab-3090 md5 全一致）；agreid_dataset/agreid_v2_dataset 在各自目录。代码三处校验 md5 一致，无 drift。

## 起跑确认（ep1-2 健康，2026-06-24）
- 数据：A→G exp1 2356q/6347g、G→A exp4 1811q/14362g、train 51530/807pid ✓（与 task 2356/1811 一致）
- AIRL iso 构建：iso_stage=3 rec late stage 14.18M params(28 tensors)+rec BNNeck 620K(2 tensors) 入优化器；trunk_recce=0 = clean ID-CE+consistency 双 detach（原始全隔离）；eval cos=0.25*rec+0.75*full ✓
- loss 分量全在：CE / Tri / CE_rec / AIRL_rec，Acc ep1 0.116→ep2 0.371 正常爬升，AIRL warmup 渐开（lam_eff 0.1，AIRL_rec 0.0002→0.0046）
- GPU 14.25G/24G，94% util，无 OOM；113s/epoch，60ep≈2h
- log: /tmp/agreidv2_airl_4090.log；PID 见 launch

## ★数据完整性教训（2026-06-24）
- Monitor 推送的事件文本**可能损坏/错位**：一条 ep20 eval 事件曾推来 `full 70.92 / FUSE 72.45` 一组数，但 log 里 grep 无此值（exit 1），真值是 `full 73.13 / FUSE 74.82`（log 行 339-342）。
- 铁律强化：**所有要写进 monitor.md / results.md 的数字必须从 log 文件直接 grep 核对**，Monitor 事件只当"该看一眼了"的触发器，绝不直接当数据源。已据此更正 ep20 全段。

