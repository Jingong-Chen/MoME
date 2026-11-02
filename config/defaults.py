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

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

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
# Parasitic settings for MoME v3 ablations:
#   PARASITIC_SCOPE: all / last6 / last3 / head_only
#   PARASITIC_TYPE: gamma_beta / gamma_only / beta_only
_C.MODEL.PARASITIC_SCOPE = "all"
_C.MODEL.PARASITIC_TYPE = "gamma_beta"
# Low-rank parasitic branches on the backbone Linear layers (qkv/proj/fc1/fc2). The
# normalization shadows can only rescale existing channels, which caps Full; a rank-r update
# mixes them. Off by default: every earlier result assumes normalization-only parasitics.
_C.MODEL.PARASITIC_LINEAR = False
_C.MODEL.PARASITIC_LORA_RANK = 8
_C.MODEL.PARASITIC_LORA_ALPHA = 16
# 'lora' keeps the rank-r wrapper; 'full' attaches full-capacity W_m shadows on qkv/proj/fc1/fc2
# (the difference of two SGD trajectories is high rank, so low-rank carriers cap Full ~82).
_C.MODEL.PARASITIC_LINEAR_MODE = "lora"
# Shadow the patch projection, positional embedding, class token, and SIE table as well.
_C.MODEL.PARASITIC_EMBED = False
# Evaluation-side switches for consolidated Full checkpoints: enable every head shadow
# (BNNeck affines, classifier columns) and the BN running-statistics switch at build time.
_C.MODEL.PARASITIC_HEAD_ON = False
_C.MODEL.PARASITIC_BN_STATS = False
_C.MODEL.FREEZE_BACKBONE = False

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
# Use synthetic D_r (distilled) instead of sampling real D_r from D_f
_C.DATASETS.USE_SYNTHETIC_DR = False
# Directory containing distilled D_r artifacts:
#   - bounding_box_train_syn/
#   - dr_syn_meta.pkl
_C.DATASETS.SYNTHETIC_DR_DIR = ''


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
# LR multiplier for parasitic parameters (weight_m / bias_m)
_C.SOLVER.PARASITIC_LR_MULT = 5.0
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
_C.SOLVER.WARMUP_METHOD = "linear"

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
# MOME
# ---------------------------------------------------------------------------- #
_C.MOME = CN()
_C.MOME.ENABLED = False
_C.MOME.ALPHA = 1.0
_C.MOME.BETA = 1.0
_C.MOME.MARGIN = 2.0
_C.MOME.DR_SAMPLES_PER_ID = 4
# Deletion granularity. 'per_id' keeps DR_SAMPLES_PER_ID images of every identity, so no
# identity is removed. 'sub_id' deletes a fraction of the identities outright, which is the
# right-to-be-forgotten setting: D_r holds every image of the retained identities and U holds
# every image of the deleted ones.
_C.MOME.FORGET_MODE = 'per_id'
_C.MOME.FORGET_ID_FRAC = 0.1
# Seed that fixes WHICH identities are deleted. Shadow models for the Forgetting-Rank
# certificate must delete the same identities as the target and differ only in training
# randomness, so this is kept separate from SOLVER.SEED. -1 means follow SOLVER.SEED.
_C.MOME.SPLIT_SEED = -1
# Combined protocol: identity-level deletion for privacy (U = every image of the deleted
# identities) plus per-identity thinning of the retain side: each RETAINED identity keeps only
# DR_K images in D_r. 0 keeps them all. A thin D_r weakens the retrain reference exactly where
# co-training with D_f helps the core, which is the regime the method was designed for.
_C.MOME.DR_K = 0
# Audit control, not a training option. Fraction of the identities requested for deletion that
# the core update silently keeps, simulating deletion requests that were not honoured. U is
# unchanged, so rho_U is asked the same question and its detection power can be measured.
# 0.0 is the real method.
_C.MOME.LEAK_ID_FRAC = 0.0
# Step 1 and Step 2 run once per iteration and the iteration count follows the LARGER loader,
# with the smaller one cycled. At 90% identity deletion D_f is 10.9x D_r, so Core takes ~10.9
# passes over its 1,209 images per epoch against the retrain reference's one, and overfits: Core
# lands 1.51 mAP under the oracle at that ratio but 0.25 above it at 50%, where the ratio is 2.0.
# True balances the core step down to one D_r pass per epoch. Default False preserves every
# earlier result.
_C.MOME.BALANCE_CORE_STEPS = False
# Explicit Step-1 period: one core update every N iterations, overriding BALANCE_CORE_STEPS.
# 0 keeps the legacy behaviour. Balancing sets N to the full ratio (about 32 on VeRi), which
# starves the core of updates; intermediate values trade core capacity against overfitting.
_C.MOME.CORE_EVERY = 0
# Restrict Step 1's softmax to the retained identities. True gives Core no trained column for a
# deleted identity. False keeps the full head: the deleted columns are never a target and never
# receive a positive gradient, so Core still never learns those identities, but they keep acting
# as negatives. That extra discriminative pressure is what the retrain reference gets, and
# removing it lets Core drive train accuracy to 0.97 while retrieval on unseen identities falls.
_C.MOME.RESTRICT_HEAD = True
# Skip Step 2 entirely. Diagnostic only: it makes Core's optimisation identical to the retrain
# reference apart from the head, which separates "the two steps interact badly" from
# "Step 1 itself is wrong".
_C.MOME.CORE_ONLY = False
# AMP dtype for the two-step loop. The reference trains in fp16 with a default-initialised
# GradScaler; MoME was switched to bf16 to escape a freeze that was actually caused by
# GradScaler(init_scale=1.0) with a backoff floor of 1.0, which makes an overflow unrecoverable.
# The scale was the bug, not the dtype, and bf16 carries two fewer mantissa bits, which starts
# to matter once the LR drops 10x at the first milestone. 'fp16' matches the reference exactly.
# fp16 is not usable here: every step overflows from iteration 1 (8967 consecutive skips),
# because this backbone needs bf16's fp32-range exponent. The earlier freeze was caused by
# GradScaler(init_scale=1.0), not by the dtype, but bf16 is still the right choice.
_C.MOME.AMP_DTYPE = "bf16"
# Evaluate Core retrieval over the training set at every eval. Diagnostic only (it is how the
# overfitting signature was caught), but on VeRi it is a ~37k x 37k distmat costing 9.5 minutes
# per eval, ~2 h per 120-epoch run. Off for production seeds; partition eval supplies the real
# numbers afterwards.
_C.MOME.EVAL_TRAIN_MAP = True
# v4: alignment standard is utility==Original, privacy==retrain.
# STEP2_DISTILL replaces Step 2's task loss with feature distillation from the Original
# checkpoint, so Full's objective IS alignment with Original from epoch 1.
# CORE_KD adds a distillation term to Step 1 computed ONLY on D_r images: the teacher's generic
# knowledge flows into the core through images it may legally train on, while U's images are
# never seen. This trades provable independence for empirical privacy; the four attacks and the
# certificate then decide whether the core stays at the reference level.
_C.MOME.STEP2_DISTILL = False
_C.MOME.CORE_KD = 0.0
# 'relational' matches the batch Gram matrix and was falsified twice (saturates at loss~0.003
# without moving retrieval; v4core 81.16 vs 80.89 baseline). 'feature' aligns the normalized
# retrieval feature directly; the full-capacity core has no adapter bottleneck, so the 0.955-
# cosine plateau that killed the Full-side feature distillation does not bind here.
_C.MOME.CORE_KD_TYPE = "relational"
# Path to a saved projector tensor V (D x k). When set, the teacher's KD feature is scrubbed
# before alignment: f_t <- normalize(f_t - V V^T f_t). V spans the teacher's U-discriminative
# subspace (computed offline from the teacher's U-class centroids), so the capability channel
# stays open while the deleted identities' structure is removed from the target. The core
# still never forward-passes a U image.
_C.MOME.CORE_KD_PROJ = ""
# Reference-world switch: train the co-training pipeline in a world where U never existed.
# The Step-2 D_f loader drops every image of the deleted identities; D_r is unchanged. Used
# to build same-procedure never-saw-U reference families for the attack panel (standard
# shadow-model methodology). sub_id mode only.
_C.MOME.DF_EXCLUDE_U = False
# Intermediate-layer hints for CORE_KD (FitNets-style): per-token cosine alignment of the
# listed backbone block outputs, teacher vs Core. JPM shares blocks[:-1], so layers <= 10.
_C.MOME.CORE_KD_HINT = 0.0
_C.MOME.CORE_KD_HINT_LAYERS = "3,7,10"
# Restricted-logit KD: KL between the teacher's retained-identity logits and the core's
# restricted head on D_r images, per branch head. Transfers class-boundary geometry that the
# feature cosine misses; deleted-identity columns are sliced away on both sides.
_C.MOME.CORE_KD_LOGIT = 0.0
_C.MOME.CORE_KD_LOGIT_T = 4.0
# Mixup-KD: an extra feature-alignment term on Beta(alpha,alpha) convex combinations of the
# D_r batch. The mixed queries stay inside the convex hull of retained images, so the channel
# is still D_r-only, but the teacher is sampled at unlimited new points instead of 1.5k.
_C.MOME.CORE_KD_MIX = 0.0
_C.MOME.CORE_KD_MIX_ALPHA = 0.4
# Weak-view teacher targets: the D_r loader additionally yields a val-transform view of each
# image and every KD teacher forward consumes it, so targets are computed on clean views while
# the student trains on the augmented ones.
_C.MOME.KD_WEAK_TEACHER = False
# The teacher checkpoint was trained under its own pixel normalization (the standard configs
# use mean/std 0.5) while the MoME pipeline uses ImageNet statistics. When set, teacher inputs
# are renormalized from the pipeline statistics to these before every teacher forward.
# Sample DR_K per camid group instead of per identity; camid encodes the modality for
# cross-modal datasets, so this keeps both modalities in every retained identity's quota.
_C.MOME.DR_K_STRATIFY_CAM = False
# Exponential moving average of the CORE parameters; evaluation and the saved checkpoint use
# the averaged weights. The EMA of a D_r-only trajectory is itself D_r-only, so the privacy
# argument is untouched. 0 disables.
_C.MOME.CORE_EMA = 0.0
_C.MOME.KD_TEACHER_MEAN = []
_C.MOME.KD_TEACHER_STD = []
_C.MOME.TEACHER_WEIGHT = ""
_C.MOME.TEACHER_CONFIG = ""
# Freeze BatchNorm running stats during Step-2 (parasitic update on D_f).
_C.MOME.FREEZE_BN_STEP2 = False
# Ablation only. True disables gradient isolation: neither step freezes the other side, so
# the core receives gradient from D_f and the by-construction guarantee no longer holds.
# This is the row that shows what the isolation buys; never set it for a deployed model.
_C.MOME.NO_ISOLATION = False
# 显存优化：True 时用两次 backward，峰值显存约与标准 TransReID 相当（L_dep 不反传到 para）
_C.MOME.MEMORY_EFFICIENT = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
