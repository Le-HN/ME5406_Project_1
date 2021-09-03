import numpy as np


class ENV_SETTINGS:
    FOUR_BY_FOUR             = np.array([[-1, -1, -1, -1, -1, -1],
                                         [-1,  0,  0,  0,  0, -1],
                                         [-1,  0, -1,  0, -1, -1],
                                         [-1,  0,  0,  0, -1, -1],
                                         [-1, -1,  0,  0,  1, -1],
                                         [-1, -1, -1, -1, -1, -1]])            # trap, walls: -1, goal: 1
    TEN_BY_TEN_TEST          = np.zeros((12, 12))
    MATRIX_SIZE              = 6                                              # map size + walls
    VALUE_ARRAY              = np.zeros((MATRIX_SIZE, MATRIX_SIZE))


class AGENT_ACTION:
    ACTION_SPACE             = [[0, 1], [1, 0], [0, -1], [-1, 0]]              # East, South, West, North
    ACTION_PROB_INT          = 0.25
    ACTION_REWARD            = 0.2
    DISCOUNT_FACTOR          = 0.9
    EPSILON                  = 0.8
