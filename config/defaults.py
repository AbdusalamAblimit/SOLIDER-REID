from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.PRETRAIN_HW_RATIO = 1

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

_C.MODEL.DROPOUT_RATE = 0.0
# Reduce feature dim
_C.MODEL.REDUCE_FEAT_DIM = False
_C.MODEL.FEAT_DIM = 512
# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.GEM_POOLING = False
_C.MODEL.STEM_CONV = False

# JPM Parameter
_C.MODEL.JPM = False
_C.MODEL.SHIFT_NUM = 5
_C.MODEL.SHUFFLE_GROUP = 2
_C.MODEL.DEVIDE_LENGTH = 4
_C.MODEL.RE_ARRANGE = True

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# Semantic Weight
_C.MODEL.SEMANTIC_WEIGHT = 1.0

# -----------------------------------------------------------------------------
# OBJGATE (TARDIS 身份条件目标性门控)；默认 ENABLED=False，全部退化为基线
# -----------------------------------------------------------------------------
_C.OBJGATE = CN()
_C.OBJGATE.ENABLED = False
_C.OBJGATE.HIDDEN = 192
_C.OBJGATE.TAU = 1.0
_C.OBJGATE.LAMBDA_TARGET = 1.0
_C.OBJGATE.LAMBDA_WARMUP_EPOCHS = 10
_C.OBJGATE.FEATMAP_INDEX = -1
_C.OBJGATE.SPLIT_W = 1.0
_C.OBJGATE.ANTI_W = 0.1
_C.OBJGATE.ENTROPY_MIN = 0.0
_C.OBJGATE.ENTROPY_MAX = 1.0e9
# DETACH_SCORE=True 时打分头吃 detach 后的主干特征，使 L_split 只训练打分头、不污染主干
# （诊断发现 L_split 梯度流进主干会把特征往"分辨目标侧"任务拽，掉 mAP）。默认 False 保留原行为。
_C.OBJGATE.DETACH_SCORE = False
# MODE='softmax' 原尖注意力替换池化；'suppress' 宽池化软抑制（门控只产抑制图、权重≥SUPPRESS_MIN，
# 保留全局池化主体，避免尖门控窄化主干）。SUPPRESS_MIN=1 时退化为全局平均池化。
_C.OBJGATE.MODE = 'softmax'
_C.OBJGATE.SUPPRESS_MIN = 0.5

# -----------------------------------------------------------------------------
# MULTIHYP (exp003 全局锚定的多假设集合匹配)；默认 ENABLED=False，全部退化为基线
# -----------------------------------------------------------------------------
_C.MULTIHYP = CN()
_C.MULTIHYP.ENABLED = False
_C.MULTIHYP.K = 3                # 每图假设槽数
_C.MULTIHYP.DETACH = True        # 假设分支对主干 detach（第一版只训头，不污染主干）
_C.MULTIHYP.LOSS_W = 1.0         # 集合损失权重
_C.MULTIHYP.POS_MARGIN = 0.3     # 同身份：soft-min 槽距离应 < 此
_C.MULTIHYP.NEG_MARGIN = 0.7     # 不同身份：soft-min 槽距离应 >= 此
_C.MULTIHYP.DIV_W = 0.5          # 同图槽多样性权重
_C.MULTIHYP.SET_TEMP = 0.1       # soft-min 温度
# 检索时保守距离 d = d_global + ALPHA*gate*unique*clamp(d_set-d_global,-CAP,0)
_C.MULTIHYP.ALPHA = 0.1          # 修正强度；=0 时检索距离矩阵与基线逐数值相等
_C.MULTIHYP.BONUS_CAP = 0.15     # 残差修正上限（只减距离、有上限）
_C.MULTIHYP.GATE_TAU = 1.2       # gate：d_global<此(已判可能相似)才启用修正
_C.MULTIHYP.GATE_SIGMA = 0.3
_C.MULTIHYP.UNIQUE_MARGIN = 0.1  # 最优与次优槽匹配的间隔门槛
# C2 DSHS（判别充分性对齐损失）：DSHS_W=0 时逐数值等于现有集合损失（干净退回）
_C.MULTIHYP.DSHS_W = 0.0         # DSHS 损失权重（0=关闭）
_C.MULTIHYP.DSHS_HARD = 'global' # 硬负来源：global=按全局相似度(正式)/random=随机负(对照)/set=按集合距离(消融)
_C.MULTIHYP.DSHS_MARGIN = 0.3    # set-distance triplet 间隔
_C.MULTIHYP.DSHS_NHARD = 10      # 每 anchor 取的硬负个数

# -----------------------------------------------------------------------------
# OSS (Occluder-Shortcut Suppression)；默认 ENABLED=False，训练和评测都退化为基线
# -----------------------------------------------------------------------------
_C.OSS = CN()
_C.OSS.ENABLED = False
_C.OSS.AUG_PROB = 0.3
_C.OSS.POOL_SIZE = 256
_C.OSS.W = 0.5
_C.OSS.GRL_ALPHA = 1.0
_C.OSS.RANDOM_LABEL = False

# -----------------------------------------------------------------------------
# DONOR_DECOUPLE（双出口反事实解耦）；默认 ENABLED=False，完全不构造新模块
# -----------------------------------------------------------------------------
_C.DONOR_DECOUPLE = CN()
_C.DONOR_DECOUPLE.ENABLED = False
_C.DONOR_DECOUPLE.PASTE_PROB = 0.5
_C.DONOR_DECOUPLE.DONOR_REPEAT = 4
_C.DONOR_DECOUPLE.AUX_DETACH = True
_C.DONOR_DECOUPLE.SYN_ID_W = 0.25
_C.DONOR_DECOUPLE.CF_W = 0.50
_C.DONOR_DECOUPLE.SAMEB_NEG_W = 0.50
_C.DONOR_DECOUPLE.DONOR_CLS_W = 0.20
_C.DONOR_DECOUPLE.ORTH_W = 0.03
_C.DONOR_DECOUPLE.NEG_MARGIN = 0.02

# -----------------------------------------------------------------------------
# PARTIAL_EVIDENCE（部分证据训练）；默认 ENABLED=False，训练和测试都退化为基线
# -----------------------------------------------------------------------------
_C.PARTIAL_EVIDENCE = CN()
_C.PARTIAL_EVIDENCE.ENABLED = False
# True 为部分证据校准方法；False 为严格同路径 aug-only 对照，合成图直接走原始 loss_fn。
_C.PARTIAL_EVIDENCE.CALIBRATE = True
_C.PARTIAL_EVIDENCE.PASTE_PROB = 0.5
_C.PARTIAL_EVIDENCE.MIN_KEEP = 0.2
_C.PARTIAL_EVIDENCE.LS_MAX = 0.2
_C.PARTIAL_EVIDENCE.MARGIN_SCALE = True
_C.PARTIAL_EVIDENCE.NO_HARDNEG_BELOW = 0.4

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10
# TARDIS 去偏合成混合；MIX_PROB=0 时 dataloader 与基线完全一致
_C.INPUT.MIX_PROB = 0.0
_C.INPUT.MIX_RATIO_RANGE = [0.3, 0.7]
_C.INPUT.MIX_TYPE = 'both'  # 'cross_id' / 'self_mix' / 'both'

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')
_C.DATASETS.ROOT_TRAIN_DIR = ('../data')
_C.DATASETS.ROOT_VAL_DIR = ('../data')


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16
# remove tail data
_C.DATALOADER.REMOVE_TAIL = 0

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "cosine"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64
_C.SOLVER.TRP_L2 = False

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
