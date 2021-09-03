import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import parameters as param
import sys


class LiRobot(gym.Env):
    def __init__(self):
        if param.ENV_SETTINGS.MATRIX_SIZE == 6:
            self.world = param.ENV_SETTINGS.FOUR_BY_FOUR
        else:
            # self.world = self.generate_ten_by_ten()
            self.world = param.ENV_SETTINGS.TEN_BY_TEN
        self.observation_space = self.world
        self.action_space = param.AGENT_ACTION.ACTION_SPACE
        self.agent_pos = [1, 1]

    def step(self, action_index):
        pre_pos = self.agent_pos
        self.agent_pos[0] += param.AGENT_ACTION.ACTION_SPACE[action_index][0]
        self.agent_pos[1] += param.AGENT_ACTION.ACTION_SPACE[action_index][1]
        if self.observation_space[self.agent_pos[0], self.agent_pos[1]] == 1 or \
                self.observation_space[self.agent_pos[0], self.agent_pos[1]] == -2 or \
                self.observation_space[self.agent_pos[0], self.agent_pos[1]] == -1:
            reward = self.observation_space[self.agent_pos[0], self.agent_pos[1]]
            done = True
            return done, self.agent_pos, reward
        else:
            done = False
            reward = param.AGENT_ACTION.ACTION_REWARD
            return done, self.agent_pos, reward

    def reset(self):
        self.agent_pos = [1, 1]
        return self.observation_space, self.agent_pos

    def generate_ten_by_ten(self):
        def connected_check(map):
            sys.setrecursionlimit(1000000)
            '''returns a list of tuples of connected squares to the given tile
            this is memoized with a dict'''
            visited = set()
            # x_limit, y_limit = map.shape[0] - 1, map.shape[1] - 1
            work_list = [(10, 10)]
            while len(work_list) > 0:
                (i, j) = work_list.pop()
                if map[i, j] == -1:
                    continue  # crashes
                if (i, j) in visited:
                    continue
                visited.add((i, j))
                work_list.append((i + 1, j))
                work_list.append((i, j + 1))
                work_list.append((i - 1, j))
                work_list.append((i, j - 1))
            if (1, 1) in visited:
                return True
            else:
                return False

        for i in range(0, 12):
            for j in range(0, 12):
                if (i == 0 or i == 11) or (j == 0 or j == 11):
                    param.ENV_SETTINGS.TEN_BY_TEN_TEST[i][j] = -1

        connected = False
        while not connected:
            for i in range(1, 11):
                for j in range(1, 11):
                    param.ENV_SETTINGS.TEN_BY_TEN_TEST[i][j] = 0
            i = 0
            while i < 25:
                cord_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                x = np.random.choice(cord_list, 1).item()
                y = np.random.choice(cord_list, 1).item()
                if (x == 10 and y == 10) or (param.ENV_SETTINGS.TEN_BY_TEN_TEST[x][y] == -1):
                    continue
                else:
                    param.ENV_SETTINGS.TEN_BY_TEN_TEST[x][y] = -1
                    i += 1
            param.ENV_SETTINGS.TEN_BY_TEN_TEST[10][10] = 1

            connected = connected_check(param.ENV_SETTINGS.TEN_BY_TEN_TEST)

        return param.ENV_SETTINGS.TEN_BY_TEN_TEST

    def observe(self):
        ...

    def reward(self):
        ...


if __name__ == '__main__':
    self = 0
    print(LiRobot.generate_ten_by_ten(self))
