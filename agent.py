import numpy as np
import parameters as param


class robot:
    def __init__(self):

        self.value = np.zeros((12, 12))
        self.sample_num = np.zeros((12, 12))

        self.value_Q = np.zeros((12, 12, 4))
        self.sample_num_Q = np.zeros((12, 12, 4))

        self.sample_list = [[1, 1]]
        self.action_list = []
        self.full_a_list =[]
        self.probs = np.zeros((12, 12, 4))
        self.pos = []
        self.obser = None

        self.robot_init()

    def robot_init(self):
        for i in range(0, 12):
            for j in range(0, 12):
                for k in range(0, 4):
                    self.probs[i][j][k] = param.AGENT_ACTION.ACTION_PROB_INT


class robot_size_4:
    def __init__(self):

        self.value = np.zeros((6, 6))
        self.sample_num = np.zeros((6, 6))

        self.value_Q = np.zeros((6, 6, 4))
        self.sample_num_Q = np.zeros((6, 6, 4))

        self.sample_list = [[1, 1]]
        self.action_list = []
        self.full_a_list = []
        self.probs = np.zeros((6, 6, 4))
        self.pos = []
        self.obser = None

        self.robot_init()

    def robot_init(self):
        for i in range(0, 6):
            for j in range(0, 6):
                for k in range(0, 4):
                    self.probs[i][j][k] = param.AGENT_ACTION.ACTION_PROB_INT
