#!/bin/bash
# =============================================================================
# VPReID 完整实验脚本
# 数据集: Occluded-Duke
# GPU: 单卡 (RTX 4070 Laptop 8GB / RTX 4090 24GB)
# =============================================================================

set -e

# ---- 配置 ----
PROJECT_ROOT="."
DATA_ROOT="data"

cd "$PROJECT_ROOT"

# ---- 检查前置条件 ----
check_prerequisites() {
    echo "=========================================="
    echo "检查前置条件"
    echo "=========================================="

    # 1. 检查数据集
    if [ ! -d "${DATA_ROOT}/occluded_duke/bounding_box_train" ]; then
        echo "[ERROR] 找不到 Occluded-Duke 数据集"
        echo "  期望路径: ${DATA_ROOT}/occluded_duke/"
        echo "  需要包含: bounding_box_train/ bounding_box_test/ query/"
        echo "            train.list gallery.list query.list"
        echo ""
        echo "  请先下载数据集并放置到上述路径。"
        echo "  下载方式: https://github.com/lightas/Occluded-DukeMTMC-Dataset"
        exit 1
    fi
    echo "[OK] Occluded-Duke 数据集: ${DATA_ROOT}/occluded_duke/"

    # 2. 检查预训练权重
    if [ ! -f "pretrained/swin_tiny.pth" ]; then
        echo "[ERROR] 找不到 SOLIDER Swin-Tiny 预训练权重"
        echo "  期望路径: pretrained/swin_tiny.pth"
        exit 1
    fi
    echo "[OK] Swin-Tiny 预训练权重: pretrained/swin_tiny.pth"

    # 3. 检查 ViTPose 权重 (VPReID 实验需要)
    if [ ! -f "pretrained/best_coco_AP_epoch_210.pth" ]; then
        echo "[WARN] 找不到 ViTPose 权重, VPReID 实验将无法运行"
        echo "  期望路径: pretrained/best_coco_AP_epoch_210.pth"
    else
        echo "[OK] ViTPose 权重: pretrained/best_coco_AP_epoch_210.pth"
    fi

    # 4. 检查 GPU
    python -c "import torch; print(f'[OK] PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

    # 5. 显存检查
    FREE_MEM=$(python -c "import torch; print(torch.cuda.mem_get_info()[0] // (1024**2))" 2>/dev/null || echo "0")
    echo "[INFO] GPU 空闲显存: ${FREE_MEM} MB"
    if [ "$FREE_MEM" -lt 6000 ]; then
        echo "[WARN] 显存不足 6GB, 可能需要降低 NUM_WORKERS 或开启更激进的显存优化"
    fi

    echo ""
}

# ---- 通用训练函数 ----
run_experiment() {
    local EXP_NAME=$1
    local CONFIG=$2
    shift 2
    local EXTRA_OPTS="$@"
    local OUTPUT_DIR="./log/occluded_duke/${EXP_NAME}"

    echo ""
    echo "=========================================="
    echo "实验: ${EXP_NAME}"
    echo "配置: ${CONFIG}"
    echo "输出: ${OUTPUT_DIR}"
    echo "额外参数: ${EXTRA_OPTS}"
    echo "=========================================="

    # 记录 git commit hash
    local GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    mkdir -p "${OUTPUT_DIR}"
    echo "git_hash: ${GIT_HASH}" > "${OUTPUT_DIR}/meta.txt"
    echo "config: ${CONFIG}" >> "${OUTPUT_DIR}/meta.txt"
    echo "extra_opts: ${EXTRA_OPTS}" >> "${OUTPUT_DIR}/meta.txt"
    echo "start_time: $(date '+%Y-%m-%d %H:%M:%S')" >> "${OUTPUT_DIR}/meta.txt"

    # 复制配置文件留档
    cp "${CONFIG}" "${OUTPUT_DIR}/config_snapshot.yml"

    # 运行训练
    python train.py \
        --config_file "${CONFIG}" \
        DATASETS.ROOT_DIR "${DATA_ROOT}" \
        OUTPUT_DIR "${OUTPUT_DIR}" \
        ${EXTRA_OPTS} \
        2>&1 | tee "${OUTPUT_DIR}/train.log"

    echo "end_time: $(date '+%Y-%m-%d %H:%M:%S')" >> "${OUTPUT_DIR}/meta.txt"
    echo "[DONE] ${EXP_NAME} 完成"
}

# =============================================================================
# 实验定义
# =============================================================================

# ---- exp001: Swin-Tiny Baseline (无 pose) ----
run_exp001() {
    run_experiment "exp001_baseline" \
        "configs/occluded_duke/swin_tiny.yml" \
        MODEL.PRETRAIN_PATH "'pretrained/swin_tiny.pth'"
}

# ---- exp002: VPReID v1 完整训练 ----
run_exp002() {
    run_experiment "exp002_vpreid_v1" \
        "configs/occluded_duke/vpreid_tiny.yml" \
        MODEL.PRETRAIN_PATH "'pretrained/swin_tiny.pth'"
}

# ---- exp003: 消融 — 无 Part ID Loss ----
run_exp003() {
    run_experiment "exp003_no_part_id" \
        "configs/occluded_duke/vpreid_tiny.yml" \
        MODEL.PRETRAIN_PATH "'pretrained/swin_tiny.pth'" \
        MODEL.VPREID.PART_ID_WEIGHT 0.0
}

# ---- exp004: 消融 — 无 Push Loss ----
run_exp004() {
    run_experiment "exp004_no_push" \
        "configs/occluded_duke/vpreid_tiny.yml" \
        MODEL.PRETRAIN_PATH "'pretrained/swin_tiny.pth'" \
        MODEL.VPREID.PUSH_WEIGHT 0.0
}

# ---- exp005: Temperature 敏感性 ----
run_exp005() {
    for temp in 0.01 0.05 0.1 0.5 1.0; do
        run_experiment "exp005_temp_${temp}" \
            "configs/occluded_duke/vpreid_tiny.yml" \
            MODEL.PRETRAIN_PATH "'pretrained/swin_tiny.pth'" \
            MODEL.VPREID.PART_TEMP "${temp}"
    done
}

# ---- exp006: N_PARTS 敏感性 ----
run_exp006() {
    for nparts in 3 5 7; do
        run_experiment "exp006_nparts_${nparts}" \
            "configs/occluded_duke/vpreid_tiny.yml" \
            MODEL.PRETRAIN_PATH "'pretrained/swin_tiny.pth'" \
            MODEL.VPREID.N_PARTS "${nparts}"
    done
}

# =============================================================================
# 8GB 显存版本 (RTX 4070 Laptop / RTX 3070 等)
# 区别: NUM_WORKERS=4, IF_LABELSMOOTH=off, WITH_CP=True
# =============================================================================

run_exp001_8g() {
    run_experiment "exp001_baseline_8g" \
        "configs/occluded_duke/swin_tiny_8g.yml"
}

run_exp002_8g() {
    run_experiment "exp002_vpreid_v1_8g" \
        "configs/occluded_duke/vpreid_tiny_8g.yml"
}

run_exp003_8g() {
    run_experiment "exp003_no_part_id_8g" \
        "configs/occluded_duke/vpreid_tiny_8g.yml" \
        MODEL.VPREID.PART_ID_WEIGHT 0.0
}

run_exp004_8g() {
    run_experiment "exp004_no_push_8g" \
        "configs/occluded_duke/vpreid_tiny_8g.yml" \
        MODEL.VPREID.PUSH_WEIGHT 0.0
}

# =============================================================================
# 主入口
# =============================================================================

usage() {
    echo "用法: bash scripts/run_experiments.sh <命令>"
    echo ""
    echo "命令:"
    echo "  check       检查前置条件(数据集/权重/GPU)"
    echo "  exp001      Swin-Tiny baseline (无 pose, 120 epoch)"
    echo "  exp002      VPReID v1 完整训练 (120 epoch)"
    echo "  exp003      消融: 无 Part ID Loss"
    echo "  exp004      消融: 无 Push Loss"
    echo "  exp005      Temperature 敏感性 (5 组)"
    echo "  exp006      N_PARTS 敏感性 (3 组)"
    echo "  baseline    运行 exp001 + exp002"
    echo "  ablation    运行 exp003 + exp004"
    echo "  all         运行全部实验"
    echo ""
    echo "8GB 显存版本 (WITH_CP=True, NUM_WORKERS=4):"
    echo "  exp001_8g   Baseline (8GB 显存优化)"
    echo "  exp002_8g   VPReID v1 (8GB 显存优化)"
    echo "  exp003_8g   消融: 无 Part ID Loss (8GB)"
    echo "  exp004_8g   消融: 无 Push Loss (8GB)"
    echo "  baseline_8g 运行 exp001_8g + exp002_8g"
    echo ""
    echo "示例:"
    echo "  bash scripts/run_experiments.sh check"
    echo "  bash scripts/run_experiments.sh exp001"
    echo "  bash scripts/run_experiments.sh baseline"
    echo "  nohup bash scripts/run_experiments.sh all > experiments_all.log 2>&1 &"
}

case "${1:-}" in
    check)
        check_prerequisites
        ;;
    exp001)
        check_prerequisites
        run_exp001
        ;;
    exp002)
        check_prerequisites
        run_exp002
        ;;
    exp003)
        check_prerequisites
        run_exp003
        ;;
    exp004)
        check_prerequisites
        run_exp004
        ;;
    exp005)
        check_prerequisites
        run_exp005
        ;;
    exp006)
        check_prerequisites
        run_exp006
        ;;
    baseline)
        check_prerequisites
        run_exp001
        run_exp002
        ;;
    ablation)
        check_prerequisites
        run_exp003
        run_exp004
        ;;
    all)
        check_prerequisites
        run_exp001
        run_exp002
        run_exp003
        run_exp004
        run_exp005
        run_exp006
        ;;
    exp001_8g)
        check_prerequisites
        run_exp001_8g
        ;;
    exp002_8g)
        check_prerequisites
        run_exp002_8g
        ;;
    exp003_8g)
        check_prerequisites
        run_exp003_8g
        ;;
    exp004_8g)
        check_prerequisites
        run_exp004_8g
        ;;
    baseline_8g)
        check_prerequisites
        run_exp001_8g
        run_exp002_8g
        ;;
    *)
        usage
        ;;
esac
