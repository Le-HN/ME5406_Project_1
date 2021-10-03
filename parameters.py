import numpy as np


class ENV_SETTINGS:
    # trap: -1, goal: 1, wall:-2
    # wall value will not be calculated, just a indication
    FOUR_BY_FOUR             = np.array([[-2, -2, -2, -2, -2, -2],
                                         [-2,  0,  0,  0,  0, -2],
                                         [-2,  0, -1,  0, -1, -2],
                                         [-2,  0,  0,  0, -1, -2],
                                         [-2, -1,  0,  0,  1, -2],
                                         [-2, -2, -2, -2, -2, -2]])
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
    # MATRIX_SIZE              = 12                                            # walls + map size
    # MATRIX_SIZE_SHOW         = 6
    # the max walls + map size is 12
    # VALUE_ARRAY              = np.zeros((12, 12))
    STATE_ACTION_VALUE       = np.zeros((12, 12, 4))


class AGENT_ACTION:
    ACTION_SPACE             = [[0, 1], [1, 0], [0, -1], [-1, 0]]              # East, South, West, North
    ACTION_PROB_INT          = 0.25                                            # initial probability for each direction in each state
    DISCOUNT_FACTOR          = 0.9                                             # GAMMA
    EPSILON                  = 0.1
    L_RATE                   = 0.02                                            # Learning rate

    ACTION_REWARD            = 0.02                                            # ACTION_REWARD is actually useless here,
                                                                               # just to avoid hit the wall.
                                                                               # Because if the ACTION_REWARD is added every step,
                                                                               # Q value will not converge.


# to calculate the number of success and failure
TEST = False
