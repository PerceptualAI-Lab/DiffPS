# CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/prw_convnext.yaml 

from yacs.config import CfgNode as CN

_C = CN()

# -------------------------------------------------------------------------------------------------------------------- #
# WandB                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #
_C.WANDB_RUN = 'PRISM'
_C.WANDB_PROJECT = "ICCV preliminary TEST"
_C.WANDB_ENTITY = "PRISM_person_Search"
_C.OUTPUT_DIR = "timestep_0"

# -------------------------------------------------------------------------------------------------------------------- #
# FEATURE EXTRACTOR                                                                                                    #
# -------------------------------------------------------------------------------------------------------------------- #
_C.FEATURE_EXTRACTOR = CN()

_C.FEATURE_EXTRACTOR.DECOUPLE = False
_C.FEATURE_EXTRACTOR.IMAGE_SIZE = (1280, 1280) #(1280, 1280)
_C.FEATURE_EXTRACTOR.FEATURE_MAP_SIZE = (160, 160) #(160, 160)
_C.FEATURE_EXTRACTOR.VERSION = '2-1'
_C.FEATURE_EXTRACTOR.PROMPT = ['person','shoes','pants','shirts','head'] # 'a photo of a person'
_C.FEATURE_EXTRACTOR.TRAIN_UNET = False
_C.FEATURE_EXTRACTOR.PROMPT_TUNING = False
_C.FEATURE_EXTRACTOR.ATTENTION = None # up-cross
_C.FEATURE_EXTRACTOR.USE_IMAGE_ENCODER = False

# v1-5는 0, v2-1, xl은 1을 줘야됨
_C.FEATURE_EXTRACTOR.DECOUPLED_TIMESTEP = 1

_C.FEATURE_EXTRACTOR.DETECTION_AGGNET_FEATURE_DIMS = [320]
_C.FEATURE_EXTRACTOR.REID_AGGNET_FEATURE_DIMS = [320,320,320,320,640,640,1280]
_C.FEATURE_EXTRACTOR.AGGNET_VERSION = "v2" # v2,v3 ...
_C.FEATURE_EXTRACTOR.FRQ_VERSION = "v1"
_C.FEATURE_EXTRACTOR.DETECTION_LAYER = [
    "up-level3-repeat0-vit-block0-self-k",
    "up-level3-repeat0-vit-block0-cross-map",
]

_C.FEATURE_EXTRACTOR.REID_LAYER = [
    "up-level3-repeat1-vit-block0-self-k",
    "up-level3-repeat0-vit-block0-self-v",
    "up-level3-repeat0-vit-block0-self-k",
    "up-level3-repeat0-vit-block0-self-q",
    "up-level2-repeat2-vit-block0-self-q",
    "up-level2-repeat1-vit-block0-self-v",
    "up-level1-repeat2-vit-block0-self-q",
]

# ConvNeXt head이면 512, ResNet head이면 1024
_C.FEATURE_EXTRACTOR.AGGNET_OUTPUT_CHANNELS = 1024

# -------------------------------------------------------------------------------------------------------------------- #
# Dataset                                                                                                              #
# -------------------------------------------------------------------------------------------------------------------- #
_C.DATASET = CN()
# Choose one from {'CUHK_SYSU', 'PRW'} as the dataset
_C.DATASET.TYPE = 'CUHK_SYSU'
_C.DATASET.PATH = ''
# Number of images per batch
_C.DATASET.BATCH_SIZE = 5

# -------------------------------------------------------------------------------------------------------------------- #
# SOLVER                                                                                                               #
# -------------------------------------------------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.MAX_EPOCHS = 20

# Learning rate settings
_C.SOLVER.OPTIMIZER = 'Adam'
_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.BIAS_DECAY = 0.0
_C.SOLVER.SGD_MOMENTUM = 0.0

# The epoch milestones to decrease the learning rate by GAMMA
_C.SOLVER.LR_DECAY_MILESTONES = [8, 14]
_C.SOLVER.LR_DECAY_GAMMA = 0.1
_C.SOLVER.WARMUP_FACTOR = 0.001
_C.SOLVER.WARMUP_EPOCHS = 1

# Set to negative value to disable gradient clipping
_C.SOLVER.CLIP_GRADIENTS = 10.0
_C.SOLVER.AMP = True

# -------------------------------------------------------------------------------------------------------------------- #
# Evaluator                                                                                                            #
# -------------------------------------------------------------------------------------------------------------------- #
_C.EVAL = CN()
# The period to evaluate the model during training
_C.EVAL.PERIOD = 1
_C.EVAL.START = 10
_C.EVAL.DETECTION_IOU_THRESHOLD = 0.5
# If set to -1, the standard threshold will be used
_C.EVAL.SEARCH_IOU_THRESHOLD = 0.5
_C.EVAL.TOP_K = [1, 5, 10]

# -------------------------------------------------------------------------------------------------------------------- #
# Misc                                                                                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
# The device loading the model
_C.DEVICE = 'cuda'
# Set seed to negative to fully randomize everything
_C.SEED = 1

# -------------------------------------------------------------------------------------------------------------------- #
# Model                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #
_C.MODEL = CN()

_C.MODEL.TRANSFORM = CN()
_C.MODEL.TRANSFORM.MIN_SIZE = 900
_C.MODEL.TRANSFORM.MAX_SIZE = 1500
_C.MODEL.TRANSFORM.IMG_MEAN = [0.5, 0.5, 0.5]
_C.MODEL.TRANSFORM.IMG_STD = [0.5, 0.5, 0.5]

# Choose one from {'SOLIDER', 'ConvNeXt', 'ResNet'} as the backbone head
_C.MODEL.BACKBONE_HEAD = 'ConvNeXt'

_C.MODEL.RPN = CN()
_C.MODEL.RPN.ANCHOR_SIZE = ((32, 64, 128, 256, 512),)
_C.MODEL.RPN.ANCHOR_RATIO = ((0.5, 1.0, 2.0),)
# NMS threshold used on RoIs
_C.MODEL.RPN.NMS_THRESH = 0.7
# Number of anchors per image used to train RPN
_C.MODEL.RPN.BATCH_SIZE_TRAIN = 256
# Target fraction of foreground examples per RPN minibatch
_C.MODEL.RPN.POS_FRAC_TRAIN = 0.5
# Overlap threshold for an anchor to be considered foreground (if >= POS_THRESH_TRAIN)
_C.MODEL.RPN.POS_THRESH_TRAIN = 0.7
# Overlap threshold for an anchor to be considered background (if < NEG_THRESH_TRAIN)
_C.MODEL.RPN.NEG_THRESH_TRAIN = 0.3
# Number of top scoring RPN RoIs to keep before applying NMS
_C.MODEL.RPN.PRE_NMS_TOPN_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOPN_TEST = 6000
# Number of top scoring RPN RoIs to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOPN_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOPN_TEST = 300

_C.MODEL.DETECTION = CN()
_C.MODEL.DETECTION.SAMPLE = CN()
_C.MODEL.DETECTION.SAMPLE.POS_THRESH = 0.5
_C.MODEL.DETECTION.SAMPLE.NEG_THRESH = 0.5
_C.MODEL.DETECTION.SAMPLE.BATCH_SIZE = 128
_C.MODEL.DETECTION.SAMPLE.POS_FRAC = 0.5
_C.MODEL.DETECTION.FEAT_MAP_SIZE = (14, 14)
_C.MODEL.DETECTION.QUALITY = False
_C.MODEL.DETECTION.BOX_REG_LOSS_TYPE = 'smooth_l1'
_C.MODEL.DETECTION.QUALITY_LOSS_TYPE = 'iou'
_C.MODEL.DETECTION.POST_PROCESS = CN()
_C.MODEL.DETECTION.POST_PROCESS.SCORE_THRESH = 0.5
_C.MODEL.DETECTION.POST_PROCESS.NMS_THRESH = 0.4
_C.MODEL.DETECTION.POST_PROCESS.DETECTIONS_PER_IMAGE = 300

_C.MODEL.REID = CN()
_C.MODEL.REID.SAMPLE = CN()
_C.MODEL.REID.SAMPLE.POS_THRESH = 0.7
_C.MODEL.REID.SAMPLE.NEG_THRESH = 0.3
_C.MODEL.REID.SAMPLE.BATCH_SIZE = 128
_C.MODEL.REID.SAMPLE.POS_FRAC = 0.5
_C.MODEL.REID.FEAT_MAP_SIZE = (24, 12)
_C.MODEL.REID.DIM_IDENTITY = 1024
_C.MODEL.REID.FEAT_MAP_USED = 'OriginalAndDownsample'  # {'Downsample', 'OriginalAndDownsample'} feature map used
_C.MODEL.REID.EMBEDDING = 'MGE'  # {'MGE', 'GFE'} embedding type
_C.MODEL.REID.EMBEDDING_MGE = CN()
_C.MODEL.REID.EMBEDDING_MGE.NUM_BRANCHES = 3
_C.MODEL.REID.EMBEDDING_MGE.DROP_PATH = 0.1

_C.MODEL.REID.LOSS = CN()
_C.MODEL.REID.LOSS.LUT_SIZE = 5532
_C.MODEL.REID.LOSS.CQ_SIZE = 5000
_C.MODEL.REID.LOSS.MOMENTUM = 0.5
_C.MODEL.REID.LOSS.SCALAR = 30.0
_C.MODEL.REID.LOSS.MARGIN = 0.25

_C.MODEL.LOSS_WEIGHT = CN()
_C.MODEL.LOSS_WEIGHT.RPN_REG = 1.0
_C.MODEL.LOSS_WEIGHT.RPN_CLS = 1.0
_C.MODEL.LOSS_WEIGHT.PROPOSAL_REG = 10.0
_C.MODEL.LOSS_WEIGHT.PROPOSAL_CLS = 1.0
_C.MODEL.LOSS_WEIGHT.PROPOSAL_QLT = 1.0
_C.MODEL.LOSS_WEIGHT.BOX_REID = 2.0

# Choose one from {'v1', 'v2'}
_C.MODEL.PARAM_INIT = 'v1'

def get_default_cfg():
    return _C.clone()
