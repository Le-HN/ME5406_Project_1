import numpy as np


class ENV_SETTINGS:
    FOUR_BY_FOUR             = np.array([[-2, -2, -2, -2, -2, -2],
                                         [-2,  0,  0,  0,  0, -2],
                                         [-2,  0, -1,  0, -1, -2],
                                         [-2,  0,  0,  0, -1, -2],
                                         [-2, -1,  0,  0,  1, -2],
                                         [-2, -2, -2, -2, -2, -2]])            # trap, walls: -1, goal: 1
    TEN_BY_TEN               = np.array([[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
                                         [-2,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0, -2],
                                         [-2,  0, -1,  0, -1,  0,  0, -1,  0,  0, -1, -2],
                                         [-2,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0, -2],
                                         [-2,  0, -1,  0,  0, -1,  0, -1,  0, -1, -1, -2],
                                         [-2,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0, -2],
                                         [-2, -1, -1,  0,  0,  0, -1,  0, -1,  0,  0, -2],
                                         [-2,  0,  0, -1,  0, -1,  0,  0,  0, -1,  0, -2],
                                         [-2, -1,  0,  0,  0,  0,  0, -1,  0,  0,  0, -2],
                                         [-2,  0,  0, -1,  0, -1,  0,  0,  0, -1,  0, -2],
                                         [-2,  0,  0,  0,  0,  0,  0, -1,  0,  0,  1, -2],
                                         [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]])
    TEN_BY_TEN_TEST          = np.zeros((12, 12))
    MATRIX_SIZE              = 12                                              # walls + map size
    MATRIX_SIZE_SHOW         = 12
    VALUE_ARRAY              = np.zeros((MATRIX_SIZE_SHOW, MATRIX_SIZE_SHOW))
    STATE_ACTION_VALUE = np.zeros((MATRIX_SIZE, MATRIX_SIZE, 4))


class AGENT_ACTION:
    ACTION_SPACE             = [[0, 1], [1, 0], [0, -1], [-1, 0]]              # East, South, West, North
    ACTION_PROB_INT          = 0.25
    DISCOUNT_FACTOR          = 0.9
    EPSILON                  = 0.1
    L_RATE                   = 0.02
    ACTION_REWARD            = 0.02                                            # ACTION_REWARD is actually useless here,
                                                                               # just to avoid hit the wall.
                                                                               # Because if the ACTION_REWARD is added every step,
                                                                               # Q value will not converge.
