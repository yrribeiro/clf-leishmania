from pathlib import Path

import torch

SEED = 42

TIMESTAMP = '2023-11-28'
EXP_NAME = 'v1-ensble'
DATA_ID = 'D1D2'

MODEL_OUT_DIR = Path(f'models/{EXP_NAME}_{TIMESTAMP}')
DPLUS_DATA_DIR = Path('./data/merged/')
PLOTS_MODEL_EVAL_OUT = Path(F'plots/{EXP_NAME}_{TIMESTAMP}')
RAW_IMAGE_DIR = Path(f'./data/raw/imgs')
ENHANCED_IMG_DIR = Path(f'./data/data-enhanced/')
MASKS_FOLDER_PATH = './data/raw/masks'
LEISH_OUTPUT_PATH = './data/merged/leish/'  # algorithm does not support pathlib TODO
NO_LEISH_OUTPUT_PATH = './data/merged/no-leish/'

ENHANCER_FACTOR = 1.5

WITH_LEISH_STRIDE = 12
NO_LEISH_STRIDE = 96
ALPHA = 0.20
WINDOW_SIZE = 96

PATIENCE = 12
BATCH_SIZE = 32
NUM_IMGS = BATCH_SIZE
VALID_SIZE = .15
TRAIN_SIZE = .7
EPOCHS = 100
INIT_LEARNING_RATE = 0.001
NET_IMG_SIZE = (3, 96,96)
NUM_CLASSES = 2
EMBEDDING_SIZE = 128
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 6
LOSSES = ['Triplet', 'NPairs', 'Circle', 'MultiSimilarity']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')