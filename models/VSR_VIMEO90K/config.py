from easydict import EasyDict as edict


class Config:
    model_version = 'RTA_CVPR2022'
    # dataset
    DATASET = edict()
    DATASET.DATASETS = '/home/hailangwu/zk/data/vimeo_septuplet'
    
    
    DATASET.SCALE = 4
    DATASET.PATCH_WIDTH = 64
    DATASET.PATCH_HEIGHT = 64


    DATASET.VALUE_RANGE = 255.0
    DATASET.NFRAME = 7
    DATASET.SEED = 0
    # dataloader
    DATALOADER = edict()
    DATALOADER.IMG_PER_GPU = 4
    DATALOADER.NUM_WORKERS = 4

    # model
    MODEL = edict()
    # feature extraction
    MODEL.EXT_BLOCKS = 8
    MODEL.IN_CHANNEL = 3

    # reconstruction
    MODEL.RECONS_BLOCKS = 40

    MODEL.N_CHANNEL = 128
    MODEL.INTERPOLATION = 'bicubic'
    MODEL.DEVICE = 'cuda'

    # solver
    SOLVER = edict()
    SOLVER.OPTIMIZER = 'Adam'
    SOLVER.BASE_LR = 5e-4
    SOLVER.WARM_UP_FACTOR = 0.1
    SOLVER.WARM_UP_ITER = 2000
    SOLVER.MAX_ITER = 900000
    SOLVER.WEIGHT_DECAY = 0
    SOLVER.MOMENTUM = 0
    SOLVER.BIAS_WEIGHT = 0.0

    # initialization

    INIT_MODEL = './VSR_VIMEO_Weights.pth'


    

    # log and save
    LOG_PERIOD = 10
    SAVE_PERIOD = 10000

    # validation
    VAL = edict()
    VAL.PERIOD = 1000
    VAL.IMG_PER_GPU = 1
    VAL.NUM_WORKERS = 1
    VAL.MAX_NUM = 100
    VAL.SAVE_IMG = True


config = Config()



