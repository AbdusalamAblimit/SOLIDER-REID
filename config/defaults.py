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

# --- Pose-guided ReID settings ---
_C.MODEL.POSE_ENABLED = False
_C.MODEL.POSE_DATA_DIR = ''         # directory containing pose_data/{split}/
_C.MODEL.POSE_THRESHOLD = 0.3       # minimum keypoint score for part validity
_C.MODEL.POSE_PART_WEIGHT = 1.0     # weight for part losses
_C.MODEL.POSE_PART_TRI_WEIGHT = 1.0 # weight for part triplet loss
_C.MODEL.POSE_HEATMAP_SIZE = [96, 32]  # (H, W) heatmap size from dataloader
_C.MODEL.POSE_HEATMAP_NORM = 'spatial_softmax'  # 'sigmoid' or 'spatial_softmax'
_C.MODEL.POSE_TEMPERATURE = 1.0        # temperature for spatial softmax
_C.MODEL.POSE_TEST_FEAT = 'concat_scaled'  # 'concat_scaled', 'part_only', 'equal_concat', 'cvk_only', 'cvk_hybrid', 'maxsim'
_C.MODEL.POSE_PART_STAGE = -1              # which backbone stage for part pooling
# Pose Feature Modulation (PFM) — hidden dim shared with PSG encoder
_C.MODEL.POSE_PFM_ENABLED = False
_C.MODEL.POSE_PFM_HIDDEN = 64              # hidden dim in PSG encoder

# PSG (Pose Spatial Gate) — backbone injection
_C.MODEL.POSE_BACKBONE_PSG = False          # use PoseBackboneModel
_C.MODEL.POSE_PSG_PART = False              # PSG + Part Pooling combination
_C.MODEL.POSE_PSG_STAGES = [-1]             # which stages to inject PSG
_C.MODEL.POSE_PSG_SPATIAL = False           # use 3x3 depthwise conv in PSG gate

# ROA (Realistic Occlusion Augmentation)
_C.MODEL.POSE_ROA = False                   # paste VOC objects
_C.MODEL.POSE_ROA_PATH = 'data/VOCdevkit/VOC2012'
_C.MODEL.POSE_ROA_PROB = 0.5
_C.MODEL.POSE_ROA_POSE_AWARE = False
# PLBOA: Pose-guided Lower-Body Occlusion Augmentation
_C.MODEL.POSE_LOWER_BODY_OCC = False
_C.MODEL.POSE_LOWER_BODY_OCC_PROB = 0.5     # probability of applying
_C.MODEL.POSE_LOWER_BODY_OCC_RATIO = 0.5    # fraction of lower body to occlude (0.3-0.7)
_C.MODEL.POSE_LOWER_BODY_OCC_MODE = 'lower'  # 'lower' = hip下, 'gradient' = 从下到上概率递减
_C.MODEL.POSE_UPPER_BODY_OCC = False          # PGMPOA: additionally occlude random upper-body parts
_C.MODEL.POSE_UPPER_BODY_OCC_PROB = 0.3       # probability (applied AFTER lower-body occ)
_C.MODEL.POSE_PATCH_EMBED = False             # PAPE: parallel pose patch embedding at input
_C.MODEL.POSE_PATCH_EMBED_KS = 1             # PAPE kernel size (1=1x1, 3=3x3)

# PAA (Pose Additive Adapter)
_C.MODEL.POSE_ADDITIVE_ADAPTER = False
_C.MODEL.POSE_PAA_ROUTED = False
_C.MODEL.POSE_PAA_BOTTLENECK = 32
_C.MODEL.POSE_PAA_ADAPTIVE_GATE = False

# Stochastic Pose Dropout
_C.MODEL.POSE_DROPOUT_P = 0.0

# Global loss scale (0.5 simulates PDS list-loss effect)
_C.MODEL.GLOBAL_LOSS_SCALE = 1.0

# Skeleton GCN head
_C.MODEL.POSE_SKELETON_GCN = False
_C.MODEL.POSE_KEYPOINT_POOL_ONLY = False
_C.MODEL.POSE_GCN_LAYERS = 2
_C.MODEL.POSE_GCN_HIDDEN = 256
_C.MODEL.POSE_KP_WEIGHT_MODE = 'score'
_C.MODEL.POSE_KP_TRIPLET = False
_C.MODEL.POSE_KP_TRIPLET_WEIGHT = 1.0

# SPLADE: Learned Sparse Representation
_C.MODEL.POSE_SPLADE = False
_C.MODEL.POSE_SPLADE_DIM = 2048
_C.MODEL.POSE_SPLADE_REG = 0.01           # Sparsity regularization weight

# Evidential Deep Learning (Dirichlet classification head)
_C.MODEL.POSE_EVIDENTIAL = False           # Replace GCN branch CE with Evidential loss
_C.MODEL.POSE_EVIDENTIAL_KL_REG = 0.1     # KL regularization max weight
_C.MODEL.POSE_EVIDENTIAL_ANNEAL = 0.6     # Fraction of training to reach full KL weight

# PNIS: Pose-Normalized Identity Space
_C.MODEL.POSE_NORMALIZE = False           # Factor out pose from identity feature
_C.MODEL.POSE_NORMALIZE_HIDDEN = 256

# STD-PR: Structural Token Decomposition with Pose-guided Routing
_C.MODEL.POSE_STRUCTURAL_ROUTING = False
_C.MODEL.POSE_STR_NUM_PARTS = 6
_C.MODEL.POSE_STR_NUM_HEADS = 8
_C.MODEL.POSE_STR_NUM_LAYERS = 2
_C.MODEL.POSE_STR_PER_TOKEN = False       # Per-token classification (forces token diversity)
_C.MODEL.POSE_STR_SUPCON = False          # Replace per-token CE with SupCon loss
_C.MODEL.POSE_STR_SUPCON_TEMP = 0.07     # SupCon temperature
_C.MODEL.POSE_STR_SUPCON_ADDITIVE = False # SupCon additive to CE (not replacing)
_C.MODEL.POSE_STR_SUPCON_WEIGHT = 0.5    # Weight when additive
_C.MODEL.POSE_STR_SUPCON_VIS_WEIGHT = False  # Visibility-weighted per-token SupCon
_C.MODEL.POSE_STR_SUPCON_GLOBAL = False      # Also apply SupCon on global (pooled) feature
_C.MODEL.POSE_STR_SELF_ATTN = False       # DPTL: self-attention among part tokens (dual-path)
_C.MODEL.POSE_STR_PART_DROP = 0.0         # PLTD: part-level token dropout probability (0=disabled)

# MaxSim triplet: set-to-set metric learning
_C.MODEL.POSE_MAXSIM_TRIPLET = False
_C.MODEL.POSE_MAXSIM_TRIPLET_TEMP = 0.05
_C.MODEL.POSE_MAXSIM_TRIPLET_ADDITIVE = False  # If True, add to pooled triplet instead of replacing
_C.MODEL.POSE_MAXSIM_TRIPLET_WEIGHT = 0.25     # Weight when additive

# Parallel augmentation (3-view training)
_C.MODEL.POSE_PARALLEL_AUG = False
_C.MODEL.POSE_OA_SD = False               # Occlusion-Asymmetric Self-Distillation
_C.MODEL.POSE_OA_SD_WEIGHT = 1.0          # Distillation loss weight
_C.MODEL.POSE_OA_SD_EMA_DECAY = 0.999    # EMA teacher decay rate
_C.MODEL.POSE_OA_SD_GLOBAL_ONLY = False   # Only distill global feature (not per-token)

# OA-RD: Occlusion-Asymmetric Relational Distillation
_C.MODEL.POSE_OA_RD = False                # Enable relational distillation (distill pairwise similarity structure)
_C.MODEL.POSE_OA_RD_TEMP = 0.1            # Temperature for softmax on similarity matrix
_C.MODEL.POSE_OA_RD_WEIGHT = 1.0          # Weight of relational distillation loss

# KAMP: Keypoint-Anchored Multi-Scale Part features
_C.MODEL.POSE_MULTI_SCALE_KP = False      # Enable multi-scale keypoint sampling
_C.MODEL.POSE_MULTI_SCALE_STAGES = [2, 3] # Which stages to use (0-indexed, 3=last)

# PADPQ: Pose-Anchored Deformable Part Queries — learned offsets around keypoints
_C.MODEL.POSE_DEFORMABLE_SAMPLE = False    # Enable deformable keypoint sampling
_C.MODEL.POSE_DEFORMABLE_K = 4            # Number of offset sampling points per keypoint

# Per-body-part independent training (KPR-inspired)
_C.MODEL.POSE_GCN_PER_PART = False        # Split 17 keypoints into 6 body parts, each with own classifier

# PPA: Pose-Prompted Part-Assignment Head — end-to-end learnable part assignment
_C.MODEL.POSE_PPA = False                 # Enable PPA (replaces GCN Part branch)
_C.MODEL.POSE_PPA_NUM_PARTS = 5           # Number of body parts (5)
_C.MODEL.POSE_PPA_ASSIGN_WEIGHT = 0.5     # Assignment loss weight
_C.MODEL.POSE_PPA_GILT = False            # GiLt mode: Part triplet only, no Part CE

# LGPA: Language-Grounded Part Assignment — CLIP text prototypes + cross-attention + pose masks
_C.MODEL.POSE_LGPA = False                # Enable LGPA (replaces PPA / GCN Part branch)
_C.MODEL.POSE_LGPA_CLIP_DIM = 512        # CLIP text feature dimension (ViT-B-32 = 512)
_C.MODEL.POSE_LGPA_NUM_HEADS = 8         # Cross-attention heads
_C.MODEL.POSE_LGPA_POSE_TEMP = 1.0       # Pose mask temperature
_C.MODEL.POSE_LGPA_ASSIGN_WEIGHT = 0.5   # Assignment supervision loss weight
_C.MODEL.POSE_LGPA_DETACH = False         # Detach features before LGPA (no gradient to backbone)

# FSDC: Feature-Space Diffusion Completion — denoise occluded spatial tokens
_C.MODEL.POSE_FSDC = False                # Enable feature denoiser
_C.MODEL.POSE_FSDC_LAYERS = 2             # Denoiser transformer layers
_C.MODEL.POSE_FSDC_HEADS = 8              # Attention heads
_C.MODEL.POSE_FSDC_MASK_RATIO = 0.3       # Training mask ratio
_C.MODEL.POSE_FSDC_NOISE_STD = 0.1        # Gaussian noise std for masked tokens
_C.MODEL.POSE_FSDC_WEIGHT = 0.5           # Reconstruction loss weight

# GSPB: Gradient-Scaled Part Branch — partial gradient flow from Part to backbone
_C.MODEL.POSE_PART_GRAD_SCALE = 0.0       # 0.0 = detach (default), 1.0 = non-detach, 0.05 = scaled

# PACI: Pose-Anchored Compositional Identity — per-ID per-part prototype bank
_C.MODEL.POSE_PACI = False                # Enable PACI
_C.MODEL.POSE_PACI_WEIGHT = 0.5           # Consistency loss weight
_C.MODEL.POSE_PACI_MOMENTUM = 0.9         # EMA momentum for prototype update
_C.MODEL.POSE_PACI_MARGIN = 0.3           # Triplet margin for consistency loss
_C.MODEL.POSE_PACI_WARMUP = 5             # Epochs before consistency loss kicks in

# OERL: Occlusion-Equivariant Representation Learning
_C.MODEL.POSE_OERL = False                # Enable OERL Part Occlusion Invariance
_C.MODEL.POSE_OERL_WEIGHT = 1.0           # POI loss weight
_C.MODEL.POSE_OERL_OCC_RATIO = 0.5        # Fraction of keypoints to occlude (0.3-0.7)

# BA-PKC: Backbone-Aware Per-Keypoint Contrastive — gradients flow to backbone (not detached)
_C.MODEL.POSE_BA_PKC = False              # Enable BA-PKC
_C.MODEL.POSE_BA_PKC_WEIGHT = 0.1         # BA-PKC loss weight

# BT-PKD: Backbone-Through Per-Keypoint Distillation
# Distill per-keypoint features from EMA teacher (clean image) to student (occluded image)
# Uses NON-detached backbone features: smooth L2 gradient flows to backbone
# Key difference from BA-PKC: L2 distillation (smooth) vs SupCon (sharp, catastrophic)
_C.MODEL.POSE_BT_PKD = False              # Enable BT-PKD (requires OA-SD)
_C.MODEL.POSE_BT_PKD_WEIGHT = 0.01       # Loss weight (keep low: gradients flow to backbone)
_C.MODEL.POSE_BT_PKD_DECAY_EPOCH = 0     # Cosine decay BT-PKD weight to 0 by this epoch (0=no decay)

# MST: MaxSim Triplet loss — directly optimizes per-keypoint features for MaxSim distance
_C.MODEL.POSE_MST = False                 # Enable MaxSim Triplet
_C.MODEL.POSE_MST_WEIGHT = 0.5            # MST loss weight
_C.MODEL.POSE_MST_MARGIN = 0.3            # Triplet margin
_C.MODEL.POSE_MST_VIS_THR = 0.3           # Visibility threshold

# PKC: Per-Keypoint Contrastive loss on GCN keypoint features
_C.MODEL.POSE_PKC = False                 # Enable per-keypoint SupCon
_C.MODEL.POSE_PKC_WEIGHT = 0.5            # PKC loss weight
_C.MODEL.POSE_PKC_TEMP = 0.07             # SupCon temperature
_C.MODEL.POSE_PKC_VIS_THR = 0.3           # Visibility threshold for including keypoint

# Structural Token Mixup (STM)
_C.MODEL.POSE_STM = False                 # Enable token-level mixup within same ID
_C.MODEL.POSE_STM_NUM_SWAP = 2            # Number of body parts to swap per mixup
_C.MODEL.POSE_STM_PROB = 0.5              # Probability of applying mixup per sample
_C.MODEL.POSE_STM_WEIGHT = 0.5            # Weight of mixup loss (added to main loss)

# LTCS: Learn-to-Trust Common Support (pair-adaptive fusion)
_C.MODEL.POSE_LTCS = False
_C.MODEL.POSE_LTCS_WEIGHT = 0.5
_C.MODEL.POSE_LTCS_WARMUP = 20
_C.MODEL.POSE_LTCS_HIDDEN = 32
_C.MODEL.POSE_LTCS_ST_LOW_THR = 0.3
_C.MODEL.POSE_LTCS_ST_UPDATE_THR = 0.7
_C.MODEL.POSE_LTCS_ST_MOM = 0.9
_C.MODEL.POSE_LTCS_ST_MIN_COUNT = 1
_C.MODEL.POSE_LTCS_ST_UPDATE_STOP_EPOCH = -1

# LPCS: Learned Pair Correction Scorer
_C.MODEL.POSE_LPCS = False
_C.MODEL.POSE_LPCS_WEIGHT = 0.5
_C.MODEL.POSE_LPCS_WARMUP = 20
_C.MODEL.POSE_LPCS_HIDDEN = 32
_C.MODEL.POSE_LPCS_DELTA_SCALE = 0.5
_C.MODEL.POSE_LPCS_HEAD_MODE = 'residual'
_C.MODEL.POSE_LPCS_CONF_WEIGHT = 0.25
_C.MODEL.POSE_LPCS_PAIR_MODE = 'all'
_C.MODEL.POSE_LPCS_PAIR_TOP_RATIO = 1.0
_C.MODEL.POSE_LPCS_RANK_MODE = 'all'
_C.MODEL.POSE_LPCS_RANK_TOP_RATIO = 1.0
_C.MODEL.POSE_LPCS_RANK_TAU = 8.0
_C.MODEL.POSE_LPCS_CONTEXT_MODE = 'none'
_C.MODEL.POSE_LPCS_ST_LOW_THR = 0.3
_C.MODEL.POSE_LPCS_ST_UPDATE_THR = 0.7
_C.MODEL.POSE_LPCS_ST_MOM = 0.9
_C.MODEL.POSE_LPCS_ST_MIN_COUNT = 1
_C.MODEL.POSE_LPCS_ST_UPDATE_STOP_EPOCH = -1

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
_C.TEST.POWER_NORM = 0.0                # Power normalization exponent (0=disabled, 0.5=sqrt)

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
