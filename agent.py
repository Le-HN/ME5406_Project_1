import numpy as np
import parameters as param


class robot:
    def __init__(self):

        self.value = np.zeros((param.ENV_SETTINGS.MATRIX_SIZE, param.ENV_SETTINGS.MATRIX_SIZE))
        self.sample_num = np.zeros((param.ENV_SETTINGS.MATRIX_SIZE, param.ENV_SETTINGS.MATRIX_SIZE))

        self.value_Q = np.zeros((param.ENV_SETTINGS.MATRIX_SIZE, param.ENV_SETTINGS.MATRIX_SIZE, 4))
        self.sample_num_Q = np.zeros((param.ENV_SETTINGS.MATRIX_SIZE, param.ENV_SETTINGS.MATRIX_SIZE, 4))

        self.sample_list = [[1, 1]]
        self.action_list = []
        self.probs = np.zeros((param.ENV_SETTINGS.MATRIX_SIZE, param.ENV_SETTINGS.MATRIX_SIZE, 4))
        self.pos = []
        self.obser = None

        self.robot_init()

    def robot_init(self):
        for i in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
            for j in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
                for k in range(0, 4):
                    self.probs[i][j][k] = param.AGENT_ACTION.ACTION_PROB_INT
