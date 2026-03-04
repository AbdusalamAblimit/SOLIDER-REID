#!/bin/bash
# =============================================================================
# 实验运行脚本 — 在 4090 (24GB) 上依次运行所有实验
#
# 用法:
#   bash run_experiments.sh          # 运行全部
#   bash run_experiments.sh exp001   # 只跑 exp001 baseline
#   bash run_experiments.sh exp002   # 只跑 exp002 NFC test
#   bash run_experiments.sh extract  # 只提取关键点
#
# 前提:
#   1. conda activate solider-reid  (或对应环境名)
#   2. pretrained/swin_tiny.pth 存在
#   3. pretrained/best_coco_AP_epoch_210.pth 存在 (用于关键点提取)
#   4. data/occluded_duke/ 数据集就位
# =============================================================================

set -e

CUDA_DEVICE=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# ------- 检查前提 -------
echo "========== 环境检查 =========="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
echo ""

if [ ! -f pretrained/swin_tiny.pth ]; then
    echo "[ERROR] pretrained/swin_tiny.pth 不存在!"
    exit 1
fi

if [ ! -d data/occluded_duke ]; then
    echo "[ERROR] data/occluded_duke/ 数据集不存在!"
    exit 1
fi

TARGET=${1:-all}

# =============================================================================
# EXP001: Baseline — Swin-Tiny + ID + Triplet (标准单分支)
# =============================================================================
run_exp001() {
    echo ""
    echo "========== EXP001: Baseline =========="
    mkdir -p experiments/exp001_baseline

    python train.py \
        --config_file configs/occluded_duke/exp001_baseline.yml \
        2>&1 | tee experiments/exp001_baseline/train.log

    echo ""
    echo "[EXP001] 训练完成。查看日志: experiments/exp001_baseline/train.log"
    echo "[EXP001] 最终评估结果:"
    grep -E "mAP:|Rank-" experiments/exp001_baseline/train.log | tail -8
}

# =============================================================================
# EXP002: NFC Test — 使用 exp001 的权重 + NFC 后处理
# =============================================================================
run_exp002() {
    echo ""
    echo "========== EXP002: NFC Test =========="

    # 需要 exp001 的权重
    WEIGHT_PATH="experiments/exp001_baseline/transformer_120.pth"
    if [ ! -f "$WEIGHT_PATH" ]; then
        echo "[EXP002] 未找到 $WEIGHT_PATH，跳过。请先运行 exp001。"
        return
    fi

    mkdir -p experiments/exp002_nfc_test

    # 用 baseline 权重做推理，开启 NFC
    python test.py \
        --config_file configs/occluded_duke/exp002_nfc_test.yml \
        TEST.WEIGHT "$WEIGHT_PATH" \
        2>&1 | tee experiments/exp002_nfc_test/test.log

    echo ""
    echo "[EXP002] NFC 测试完成。"
    grep -E "mAP:|Rank-" experiments/exp002_nfc_test/test.log | tail -8
}

# =============================================================================
# 关键点提取 — 用 ViTPose 提取所有图片的关键点+可见性
# =============================================================================
run_extract() {
    echo ""
    echo "========== 关键点提取 (ViTPose) =========="

    if [ ! -f pretrained/best_coco_AP_epoch_210.pth ]; then
        echo "[ERROR] pretrained/best_coco_AP_epoch_210.pth 不存在!"
        return
    fi

    python tools/extract_keypoints.py \
        --pose-config pose/config_vispredict.py \
        --pose-checkpoint pretrained/best_coco_AP_epoch_210.pth \
        --data-root data/occluded_duke \
        --output-dir data/occluded_duke/pose \
        --batch-size 64 \
        --device cuda:0 \
        2>&1 | tee experiments/extract_keypoints.log

    echo ""
    echo "[提取完成] 结果保存在 data/occluded_duke/pose/"
    ls -lh data/occluded_duke/pose/
}

# =============================================================================
# 执行入口
# =============================================================================
case $TARGET in
    exp001)
        run_exp001
        ;;
    exp002)
        run_exp002
        ;;
    extract)
        run_extract
        ;;
    all)
        run_exp001
        run_extract
        run_exp002
        ;;
    *)
        echo "用法: bash run_experiments.sh [exp001|exp002|extract|all]"
        exit 1
        ;;
esac

echo ""
echo "========== 所有实验完成 =========="
