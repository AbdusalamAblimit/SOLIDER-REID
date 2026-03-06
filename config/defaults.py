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
# Gradient checkpointing for transformer backbones (trade compute for memory)
_C.MODEL.WITH_CP = False

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
# Freeze backbone for first N epochs (useful for VPReID to let heads warm up)
_C.SOLVER.FREEZE_BACKBONE_EPOCHS = 0

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

# ---------------------------------------------------------------------------- #
# VPReID: Visibility-aware Pose-guided ReID
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# PosePart: Offline Pose-Guided Part Features
# ---------------------------------------------------------------------------- #
_C.MODEL.POSE_PART = CN()
_C.MODEL.POSE_PART.ENABLE = False
_C.MODEL.POSE_PART.N_PARTS = 5                    # number of body parts
_C.MODEL.POSE_PART.SIGMA = 2.0                    # Gaussian attention sigma in feature map space
_C.MODEL.POSE_PART.PART_ID_WEIGHT = 0.5           # per-part ID loss weight
_C.MODEL.POSE_PART.VIS_THRESHOLD = 0.3            # min visibility to include part loss

_C.MODEL.PCFC = CN()
_C.MODEL.PCFC.ENABLE = False
_C.MODEL.PCFC.SIGMA = 3.0                         # Gaussian sigma for visibility attention
_C.MODEL.PCFC.ALPHA_INIT = 0.5                    # initial attention strength
_C.MODEL.PCFC.USE_PART_LOSS = True                # also use part features for aux loss
_C.MODEL.PCFC.N_PARTS = 5
_C.MODEL.PCFC.PART_SIGMA = 2.0
_C.MODEL.PCFC.PART_ID_WEIGHT = 1.0                # part ID loss weight (best from exp004a)
_C.MODEL.PCFC.VIS_THRESHOLD = 0.3
_C.MODEL.PCFC.PART_TRIPLET_WEIGHT = 0.0              # GiLt-style part triplet loss weight
_C.MODEL.PCFC.OST_PROB = 0.0                         # Occlusion Simulation Training probability
_C.MODEL.PCFC.OST_MIN_PARTS = 1                      # Min body parts to occlude
_C.MODEL.PCFC.OST_MAX_PARTS = 3                      # Max body parts to occlude
_C.MODEL.PCFC.MS_PART_STAGE = -1                      # Multi-scale: use this stage for parts (-1=same as global)
_C.MODEL.PCFC.BPRE_PROB = 0.0                         # Body Part Random Erasing probability (image-level occlusion aug)
_C.MODEL.PCFC.BPRE_MAX_PARTS = 1                      # Max body parts to erase per image
_C.MODEL.PCFC.FREEZE_ALPHA = False                     # Freeze alpha (don't learn, use ALPHA_INIT as fixed value)

_C.MODEL.PVFM = CN()
_C.MODEL.PVFM.ENABLE = False
_C.MODEL.PVFM.SIGMA = 3.0                         # Gaussian sigma for visibility maps
_C.MODEL.PVFM.BETA_INIT = 0.3                     # initial modulation strength per stage
_C.MODEL.PVFM.ACTIVE_STAGES = (2, 3)              # which stages to apply modulation

_C.MODEL.KPE = CN()
_C.MODEL.KPE.ENABLE = False
_C.MODEL.KPE.SIGMA = 3.0                          # Gaussian sigma in patch grid space
_C.MODEL.KPE.INJECT_LAYER = 0                     # inject at this Swin stage (0 = before all stages)

_C.MODEL.VPREID = CN()
_C.MODEL.VPREID.ENABLE = False
_C.MODEL.VPREID.N_PARTS = 5
_C.MODEL.VPREID.PART_TEMP = 0.1              # softmax temperature for part attention
_C.MODEL.VPREID.VIS_THRESHOLD = 0.5          # part visibility threshold
_C.MODEL.VPREID.POSE_CFG = 'pose/config_vispredict.py'
_C.MODEL.VPREID.POSE_CKPT = 'pretrained/best_coco_AP_epoch_210.pth'
_C.MODEL.VPREID.ID_WEIGHT = 1.0              # global + fg ID loss weight
_C.MODEL.VPREID.TRI_WEIGHT = 1.0             # part-averaged triplet weight
_C.MODEL.VPREID.PART_ID_WEIGHT = 0.5         # per-part ID loss weight
_C.MODEL.VPREID.PUSH_WEIGHT = 0.1            # push diversity loss weight
