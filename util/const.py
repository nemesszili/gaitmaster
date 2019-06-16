import numpy as np
import torch

S0_TRAIN_USER_RANGE = range(1, 12)
S0_TEST_USER_RANGE = range(12, 23)

NUM_USERS = 153
USER_RANGE = range(1, 154)

NEG_RATE = 1

PATH = 'data/'

BATCH_SIZE = 128

RANDOM_STATE = np.random.seed(0)
torch.manual_seed(0)
