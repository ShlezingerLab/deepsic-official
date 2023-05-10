from enum import Enum

HALF = 0.5
N_USERS = 4 # number of users
N_ANTS = 4 # number of antennas


class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'
