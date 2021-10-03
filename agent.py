import numpy as np
import parameters as param


# agent in 10*10 map
class robot:
    def __init__(self):

        # used in Monte_Carlo_V, which doesn't matter
        self.value = np.zeros((12, 12))
        self.sample_num = np.zeros((12, 12))

        # store the action value
        self.value_Q = np.zeros((12, 12, 4))

        # store the number of traversing the particular action-state pairs
        self.sample_num_Q = np.zeros((12, 12, 4))

        # robot's positions in one episode
        self.sample_list = [[1, 1]]

        # robot's actions in one episode
        self.action_list = []

        # probability matrix
        self.probs = np.zeros((12, 12, 4))

        # current position of robot
        self.pos = []

        # observation of robot
        self.obser = None

        self.robot_init()

    # initialize the probability matrix
    def robot_init(self):
        for i in range(0, 12):
            for j in range(0, 12):
                for k in range(0, 4):
                    self.probs[i][j][k] = param.AGENT_ACTION.ACTION_PROB_INT


# agent in 4*4 map
class robot_size_4:
    def __init__(self):

        # used in Monte_Carlo_V, which doesn't matter
        self.value = np.zeros((6, 6))
        self.sample_num = np.zeros((6, 6))

        # store the action value
        self.value_Q = np.zeros((6, 6, 4))

        # store the number of traversing the particular action-state pairs
        self.sample_num_Q = np.zeros((6, 6, 4))

        # robot's positions in one episode
        self.sample_list = [[1, 1]]

        # robot's actions in one episode
        self.action_list = []

        # probability matrix
        self.probs = np.zeros((6, 6, 4))

        # current position of robot
        self.pos = []

        # observation of robot
        self.obser = None

        self.robot_init()

    # initialize the probability matrix
    def robot_init(self):
        for i in range(0, 6):
            for j in range(0, 6):
                for k in range(0, 4):
                    self.probs[i][j][k] = param.AGENT_ACTION.ACTION_PROB_INT
