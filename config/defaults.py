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

# Checkpoint (gradient checkpointing to save memory)
_C.MODEL.WITH_CP = False

# Pose-guided part pooling
_C.MODEL.POSE_ENABLED = False
_C.MODEL.POSE_DATA_DIR = ''         # directory containing pose_data/{split}/
_C.MODEL.POSE_THRESHOLD = 0.3       # minimum keypoint score for part validity
_C.MODEL.POSE_PART_WEIGHT = 1.0     # weight for part losses
_C.MODEL.POSE_PART_TRI_WEIGHT = 1.0 # weight for part triplet loss
_C.MODEL.POSE_HEATMAP_SIZE = [96, 32]  # (H, W) heatmap size from dataloader
_C.MODEL.POSE_HEATMAP_NORM = 'spatial_softmax'  # 'sigmoid' or 'spatial_softmax'
_C.MODEL.POSE_TEMPERATURE = 1.0        # temperature for spatial softmax
_C.MODEL.POSE_TEST_FEAT = 'concat_scaled'  # 'concat_scaled', 'part_only', 'equal_concat', 'cvk_only', 'cvk_hybrid'
_C.MODEL.POSE_PART_STAGE = -1              # which backbone stage for part pooling (-1=last, -2=second-to-last)
# Pose Feature Modulation (PFM)
_C.MODEL.POSE_PFM_ENABLED = False
_C.MODEL.POSE_PFM_HIDDEN = 64              # hidden dim in PFM encoder

# Pose Backbone Injection (PSG)
_C.MODEL.POSE_BACKBONE_PSG = False          # use PoseBackboneModel instead of PoseReIDModel
_C.MODEL.POSE_PSG_PART = False              # PSG + Part Pooling combination
_C.MODEL.POSE_PSG_STAGES = [-1]             # which stages to inject PSG (e.g. [-1] = last only, [2,3] = stage 2+3)
_C.MODEL.POSE_ATTN_BIAS = False             # use Pose Attention Bias (PAB) instead of PSG
_C.MODEL.POSE_ATTN_MASK = False             # use Pose-Guided Attention Masking (PGAM) with PSG
_C.MODEL.POSE_ATTN_MASK_THRESHOLD = 0.3     # heatmap threshold for PGAM body/non-body
_C.MODEL.POSE_ATTN_MASK_STAGES = [-1]       # which stages to apply PGAM (default: last stage only)
_C.MODEL.POSE_PSG_PAB_COMBO = False         # use both PSG + PAB simultaneously
_C.MODEL.POSE_PSG_SPATIAL = False           # use 3x3 depthwise conv in PSG gate
_C.MODEL.POSE_CROSS_ATTN = False           # use Pose Cross-Attention (PXA) instead of PSG
_C.MODEL.POSE_GUIDED_ERASING = False        # use pose-guided erasing instead of random erasing
_C.MODEL.POSE_ROA = False                   # Realistic Occlusion Augmentation (paste VOC objects)
_C.MODEL.POSE_ROA_PATH = 'data/VOCdevkit/VOC2012'  # path to VOC2012 root
_C.MODEL.POSE_ROA_PROB = 0.5               # probability of applying ROA per image
_C.MODEL.POSE_ROA_POSE_AWARE = False       # use pose-aware placement instead of random
_C.MODEL.POSE_CHANNEL_GATE = False          # use Pose-Conditioned Channel Gate (PCG) after GAP
_C.MODEL.POSE_PCG_HIDDEN = 64              # hidden dim in PCG MLP
_C.MODEL.POSE_PSG_CONTENT_ADAPTIVE = False # use Content-Adaptive PSG (CAPSG) gate
_C.MODEL.POSE_RECON_HEAD = False            # use Pose Reconstruction Head (PRA) auxiliary task
_C.MODEL.POSE_RECON_WEIGHT = 0.1            # weight for PRA MSE loss
_C.MODEL.POSE_DUAL_STREAM = False           # use Pose Dual Stream (PDS) model
_C.MODEL.POSE_PART_STOP_GRAD = False       # stop gradient from Part branch to shared stages
_C.MODEL.POSE_STOP_GRAD_EPOCHS = 0        # delayed stop_grad: block Part gradients for first N epochs, then release (0=use POSE_PART_STOP_GRAD as static flag)
_C.MODEL.POSE_GLOBAL_PSG = True            # use PSG in Global branch (set False for ablation)
_C.MODEL.POSE_SPLIT_STAGE = -1            # PDS split point: -1=last stage only (default), 2=split at stage 2+3, etc.
_C.MODEL.POSE_DROPOUT_P = 0.0            # Stochastic Pose Dropout: probability of zeroing heatmaps during training (0=disabled)
_C.MODEL.POSE_PCRA_ALPHA = 0.0           # Pose-Contrastive Representation Alignment: pose similarity weight for triplet distance (0=disabled)
_C.MODEL.POSE_PART_LR_FACTOR = 1.0      # LR multiplier for Part branch params in PDS (1.0=same as global)
_C.MODEL.POSE_WEIGHTED_POOL = False     # Replace GAP with pose-weighted pooling
_C.MODEL.GLOBAL_LOSS_SCALE = 1.0       # Scale factor for global loss (0.5 simulates PDS list-loss effect)
_C.MODEL.POSE_SKELETON_GCN = False     # Use Skeleton GCN in Part branch (replaces Part Pooling)
_C.MODEL.POSE_KEYPOINT_POOL_ONLY = False  # Use keypoint sampling + confidence pooling only (no graph propagation)
_C.MODEL.POSE_GCN_LAYERS = 2          # Number of GCN layers
_C.MODEL.POSE_GCN_HIDDEN = 256        # GCN hidden dimension
_C.MODEL.POSE_KP_WEIGHT_MODE = 'score'  # Keypoint pooling weight: 'score', 'visibility', 'score_visibility', 'binary_visibility'
_C.MODEL.POSE_KP_TRIPLET = False         # Per-keypoint triplet loss for GCN branch
_C.MODEL.POSE_KP_TRIPLET_WEIGHT = 1.0    # Weight for per-keypoint triplet loss
_C.MODEL.POSE_KP_DISSIMILAR = False      # Keypoint Dissimilar Loss (prevent feature collapse)
_C.MODEL.POSE_KP_DISSIMILAR_WEIGHT = 0.1 # Weight for KDL
_C.MODEL.POSE_KP_UNCERTAINTY = False     # Learned Keypoint Uncertainty head
_C.MODEL.POSE_KP_UNCERTAINTY_REG = 0.1  # Regularization weight to prevent uncertainty collapse
_C.MODEL.POSE_PKE = False               # Probabilistic Keypoint Embeddings (Gaussian mu+sigma)
_C.MODEL.POSE_ADDITIVE_ADAPTER = False  # Pose Additive Adapter (PAA) alongside PSG
_C.MODEL.POSE_PAA_ROUTED = False        # Reliability-routed PAA: only add to low-confidence regions
_C.MODEL.POSE_PAA_BOTTLENECK = 32      # PAA bottleneck dimension
_C.MODEL.POSE_TOKEN_DECODER = False      # Pose-Token Distillation (PTD): learned part tokens
_C.MODEL.POSE_TOKEN_NUM_PARTS = 5       # Number of part tokens
_C.MODEL.POSE_TOKEN_DIM = 256           # Cross-attention dimension
_C.MODEL.POSE_TOKEN_HEADS = 8           # Number of attention heads
_C.MODEL.POSE_TOKEN_LAYERS = 2          # Number of decoder layers
_C.MODEL.POSE_TOKEN_HM_WEIGHT = 1.0     # Heatmap distillation loss weight
_C.MODEL.POSE_KP_LEARNABLE_ATTN = False  # Learnable Keypoint Attention for GCN pooling
_C.MODEL.POSE_CSGT = False               # Common-Support-Guided Triplet mining on global branch
_C.MODEL.POSE_CSGT_WEIGHT = 1.0          # Extra loss weight for CSGT
_C.MODEL.POSE_CSGT_MIN_OVERLAP = 0.3     # Minimum common-support overlap for support-aware mining
_C.MODEL.POSE_CSGT_MINE_MODE = 'both'    # 'both', 'pos', or 'neg'
_C.MODEL.POSE_SGMKC = False              # Skeleton-Guided Masked Keypoint Completion
_C.MODEL.POSE_SGMKC_RATIO = 0.3          # Fraction of keypoints to mask during training
_C.MODEL.POSE_SGMKC_WEIGHT = 1.0         # Reconstruction loss weight
# PAMC (Pose-Aware Masking Consistency) — self-supervised consistency via pose-guided body masking
_C.MODEL.POSE_PAMC = False               # Enable PAMC training
_C.MODEL.POSE_PAMC_WEIGHT = 0.5          # Consistency loss weight
_C.MODEL.POSE_PAMC_WARMUP = 10           # Warmup epochs before enabling PAMC
_C.MODEL.POSE_PAMC_PROJ_DIM = 2048       # Projector MLP hidden dimension
_C.MODEL.POSE_PAML = False               # Enable PAML (Pose-Aware Metric Learning) for part triplet
_C.MODEL.POSE_KP_RPE = False             # Enable KP-RPE (Keypoint Relative Position Encoding) in Swin attention
_C.MODEL.POSE_KP_RPE_HIDDEN = 32        # Hidden dim for KP-RPE MLP
# XCAD (Cross-Attention Decoder) — replaces GCN with cross-attention for keypoint features
_C.MODEL.POSE_XCAD = False               # Use cross-attention decoder instead of GCN
_C.MODEL.POSE_XCAD_DIM = 256             # Internal attention dimension
_C.MODEL.POSE_XCAD_HEADS = 8             # Number of attention heads

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
_C.SOLVER.FREEZE_BACKBONE_EPOCHS = 0  # freeze backbone for first N epochs (0 = no freeze)

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
_C.TEST.CVK_GLOBAL_WEIGHT = 1.0
_C.TEST.CVK_KP_WEIGHT = 1.0
# NFC (Neighbor Feature Centralization) test-time augmentation
_C.TEST.NFC = False
_C.TEST.NFC_K1 = 2
_C.TEST.NFC_K2 = 2
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
