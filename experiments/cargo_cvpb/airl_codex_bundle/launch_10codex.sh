#!/bin/bash
# 10-codex 全量审查 AIRL code + logs。每个独立审查员全量过一遍,各带一个重点 lens。
BUNDLE=/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
mkdir -p "$BUNDLE/reviews"
cd "$BUNDLE" || exit 1

CTX='背景: AIRL = aerial-ground person ReID 的 degradation-consistency 双分支方法。
- code/afd_model.py: 模型 AFDModel,含 _forward_swin_split / iso dual-branch (f_full 干净头 + f_rec recover 头)。
- code/afd_train.py: 训练 + eval。do_inference 双分支 eval 报 full/rec/FUSE 各方向。
- config: --airl_dualbranch_iso --airl_iso_stage 3 --airl_iso_trunk_recce 0 --airl_fuse_w 0.25;soft-fuse: cos = 0.25*cos(f_rec) + 0.75*cos(f_full)。
- logs/: agreidv2_airl_4090.log (AIRL final, lab-4090, torch2.6) / agreidv2_airl_iso.log (AIRL, lab-3090, torch2.7, 跑到 ep40+) / agreidv2_baseline_4090.log (baseline seed1, lab-4090) / agreidv2_baseline_4090_s2.log (baseline seed2, lab-4090) / agreidv2_baseline.log (baseline, lab-3090)。
- 每 epoch eval 打两行: [A->G] full mAP= rec mAP= FUSE mAP= 和 [G->A] 同。baseline 不带 full/rec(单特征)。

我(主控)反复改判过这个结果,可能有错,请独立核这 3 条待验结论:
(1) AIRL net gain (FUSE mean vs baseline final mean) ≈ 融合增益 FUSE−full ~+1.5,双数据集一致;
(2) AIRL 的 f_full 头 ≈ baseline ± seed 噪声(±1.2):CARGO 时 full 头走运比 baseline +0.6 → net +2.37;AG-ReID.v2 lab-4090 时 full 头背运 −1.2 → net ≈ 0(我一度据此判 AIRL 死,后改判是 seed 噪声);
(3) 方向特化(rec/full 哪个头帮 A->G/G->A 哪个检索方向)是 CARGO-specific,第二数据集方向翻号/消失。'

LENSES=(
'eval 对称性 + 泄漏: do_inference 提取 f_full/f_rec 与 train forward 是否一致?FUSE eval (cos=w*rec+(1-w)*full) 在 distmat 层实现对吗?有无 train/test 信息泄漏、有无用到 query/gallery label?'
'融合机制是否 trivial: soft-fuse w=0.25 固定合理吗?FUSE−full 的 +1.5 是真互补证据还是两个相关 head 的平庸 ensemble?换成两个独立随机 init 单 head 平均会不会也涨这么多(即增益是否来自 degradation 机制还是单纯双头平均)?'
'降质一致性实现: resolution degradation 怎么做的(bilinear down 再 up?降到什么 budget)?consistency loss 形式对吗?会不会有 degenerate/捷径解让 rec 头学到平凡映射?'
'梯度隔离正确性: iso_trunk_recce=0 全 detach 是否真隔离了 f_rec 对 shared trunk 的梯度?full_map.detach() 时机/位置对吗?降质 pass 真的没改 f_full 的权重/激活?'
'BN/Norm 污染: Swin trunk 是 LayerNorm(无 running stats)吗?f_full 的 BNNeck (self.bottleneck) 在降质 (rec_only) pass 是否真不 forward、不更新 running stats?主控声称"不更新",请逐行验真伪。eval 时 BN 用 running 还是 batch?'
'seed/RNG/数据加载: rec 分支 deep-copy + bottleneck_rec 的 kaiming init 是否偏移了全局 RNG,导致 AIRL 与 baseline 的数据顺序/增强不同?注释称 rec stages 在 f_full loop 后跑保证 f_full RNG 不变——真的吗?net 的 ±1.2 seed 方差(主控估计)与 logs 里 baseline seed1 vs seed2 的实际差是否吻合?'
'指标计算: mAP/Rank/eval_from_distmat 实现对吗?cos→dist 转换(2-2cos?)对吗?A->G / G->A 的 query/gallery 划分对吗?pid 解析有无错(folder name vs P-prefix 之类的坑)?camid 过滤对吗?'
'数值/AMP: 有无 AMP autocast?f_full/f_rec 是否都 L2-normalize?cos 数值稳定吗?有无潜在 NaN/Inf?fuse_w 的 dtype/device 对吗?'
'结果与 log 一致性: 逐档对照 logs/ 各 epoch 的 full/rec/FUSE 真实数,核主控报的 (CARGO+2.37 / AG-ReID.v2 lab-4090 net≈0 / 融合~+1.5 / 末段 baseline surge 后 net 收缩) 是否与 log 吻合,有无挑 epoch / 抄错 / 拿 baseline 谷底比的灌水?'
'整体可信度 + novelty: 有无任何致命 bug 让全部结果不可信?这套 AIRL(degradation-consistency 梯度隔离双头 + 方向感知融合,net~+1.5)作为 B 类 aerial-ground ReID 方法稿,卖点撑得住吗?与 ViSA(dual-branch local fusion)/ AdvProp(aux-BN)/ AG-VPReID 低清流 等先例的实质区别?用 web search 查。'
)

for i in "${!LENSES[@]}"; do
  N=$((i+1))
  PROMPT="你是 10 名独立审查员中的第 ${N} 名,对 AIRL 做全量代码 + log 审查(不是只看你的 lens,要全量,但本轮额外深挖下面这个重点)。
${CTX}

== 本轮重点 lens ==
${LENSES[$i]}

== 要求 ==
逐行读 code/afd_model.py 和 code/afd_train.py 的相关部分,并对照 logs/ 里的真实数字。
输出格式:
1. Verdict: approve(代码与结果可信)/ needs-attention(有问题)
2. Findings: 每条给 severity(Critical/High/Medium/Low)+ 文件:行号 + 具体问题 + 为什么
3. 对主控 3 条待验结论的独立判断: 逐条 同意 / 反对 / log数据不支持,给依据
4. 本轮 lens 的专项结论
用 web search 查相关 novelty/先例。中文输出,务实不客套。"
  nohup env HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 ALL_PROXY=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 all_proxy=http://127.0.0.1:7890 NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1 /opt/homebrew/bin/codex --search exec -s read-only --color never "$PROMPT" > "$BUNDLE/reviews/codex_${N}.md" 2>&1 &
  echo "launched codex ${N} (PID $!)"
  sleep 2
done

echo "=== 10 codex 已全部并行启动 ==="
