import numpy as np

SPLIT = 11
POS_USER_RANGE = range(1, SPLIT)
NEG_USER_RANGE = range(SPLIT, 154)

NEG_RATE = 1

RAW_PATH = 'data/raw'
FEAT_PATH = 'data/'

BATCH_SIZE = 128
FEAT_EPOCHS = 20

RANDOM_STATE = np.random.seed(0)