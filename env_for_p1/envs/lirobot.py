import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
import parameters as param
import sys


class LiRobot(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self,):
        if param.ENV_SETTINGS.MATRIX_SIZE == 6:
            self.world = param.ENV_SETTINGS.FOUR_BY_FOUR
        else:
            # self.world = self.generate_ten_by_ten()
            self.world = param.ENV_SETTINGS.TEN_BY_TEN
        self.observation_space = self.world
        self.action_space = param.AGENT_ACTION.ACTION_SPACE
        self.agent_pos = [1, 1]
        self.viewer = None

    def step(self, action_index):
        pre_pos = self.agent_pos
        self.agent_pos[0] += param.AGENT_ACTION.ACTION_SPACE[action_index][0]
        self.agent_pos[1] += param.AGENT_ACTION.ACTION_SPACE[action_index][1]
        if self.observation_space[self.agent_pos[0], self.agent_pos[1]] == -2:
            self.agent_pos[0] -= param.AGENT_ACTION.ACTION_SPACE[action_index][0]
            self.agent_pos[1] -= param.AGENT_ACTION.ACTION_SPACE[action_index][1]
            reward = 0
            done = False
            return done, self.agent_pos, reward
        if self.observation_space[self.agent_pos[0], self.agent_pos[1]] == 1 or \
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

    def render_10(self, x, y, mode='human'):  # 可视化画图
        screen_width = 600
        screen_height = 600
        square_edge = 50
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)  # 调用rendering中的画图函数，#创建600*600的窗口
            # 创建网格世界，一共包括10条直线，事先算好每条直线的起点和终点坐标，然后绘制这些直线，代码如下：
            # 创建网格世界
            self.line1 = rendering.Line((50, 50), (550, 50))
            self.line2 = rendering.Line((50, 100), (550, 100))
            self.line3 = rendering.Line((50, 150), (550, 150))
            self.line4 = rendering.Line((50, 200), (550, 200))
            self.line5 = rendering.Line((50, 250), (550, 250))
            self.line6 = rendering.Line((50, 300), (550, 300))
            self.line7 = rendering.Line((50, 350), (550, 350))
            self.line8 = rendering.Line((50, 400), (550, 400))
            self.line9 = rendering.Line((50, 450), (550, 450))
            self.line10 = rendering.Line((50, 500), (550, 500))
            self.line11 = rendering.Line((50, 550), (550, 550))

            self.line12 = rendering.Line((50, 50), (50, 550))
            self.line13 = rendering.Line((100, 50), (100, 550))
            self.line14 = rendering.Line((150, 50), (150, 550))
            self.line15 = rendering.Line((200, 50), (200, 550))
            self.line16 = rendering.Line((250, 50), (250, 550))
            self.line17 = rendering.Line((300, 50), (300, 550))
            self.line18 = rendering.Line((350, 50), (350, 550))
            self.line19 = rendering.Line((400, 50), (400, 550))
            self.line20 = rendering.Line((450, 50), (450, 550))
            self.line21 = rendering.Line((500, 50), (500, 550))
            self.line22 = rendering.Line((550, 50), (550, 550))

            # 创建完之后，给11条直线设置颜色，并将这些创建的对象添加到几何中代码如下：
            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)

            self.line12.set_color(0, 0, 0)
            self.line13.set_color(0, 0, 0)
            self.line14.set_color(0, 0, 0)
            self.line15.set_color(0, 0, 0)
            self.line16.set_color(0, 0, 0)
            self.line17.set_color(0, 0, 0)
            self.line18.set_color(0, 0, 0)
            self.line19.set_color(0, 0, 0)
            self.line20.set_color(0, 0, 0)
            self.line21.set_color(0, 0, 0)
            self.line22.set_color(0, 0, 0)

            # create obstacles
            self.obstacle_1 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, square_edge / 2),
                                                       (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(75, 75))
            self.obstacle_1.add_attr(self.offset)

            self.obstacle_2 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, square_edge / 2),
                                                       (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(75, 175))
            self.obstacle_2.add_attr(self.offset)
            self.obstacle_2.set_color(0, 0, 0)

            self.obstacle_3 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, square_edge / 2),
                                                       (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(75, 275))
            self.obstacle_3.add_attr(self.offset)
            self.obstacle_3.set_color(0, 0, 0)

            self.obstacle_4 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, square_edge / 2),
                                                       (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(125, 475))
            self.obstacle_4.add_attr(self.offset)
            self.obstacle_4.set_color(0, 0, 0)

            self.obstacle_5 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, square_edge / 2),
                                                       (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(175, 125))
            self.obstacle_5.add_attr(self.offset)
            self.obstacle_5.set_color(0, 0, 0)

            self.obstacle_6 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, square_edge / 2),
                                                       (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(175, 175))
            self.obstacle_6.add_attr(self.offset)
            self.obstacle_6.set_color(0, 0, 0)

            self.obstacle_7 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, square_edge / 2),
                                                       (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(225, 325))
            self.obstacle_7.add_attr(self.offset)
            self.obstacle_7.set_color(0, 0, 0)

            self.obstacle_8 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, square_edge / 2),
                                                       (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(225, 375))
            self.obstacle_8.add_attr(self.offset)
            self.obstacle_8.set_color(0, 0, 0)

            self.obstacle_9 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, square_edge / 2),
                                                       (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(225, 425))
            self.obstacle_9.add_attr(self.offset)
            self.obstacle_9.set_color(0, 0, 0)

            self.obstacle_10 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, square_edge / 2),
                                                       (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(225, 475))
            self.obstacle_10.add_attr(self.offset)
            self.obstacle_10.set_color(0, 0, 0)

            self.obstacle_11 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, square_edge / 2),
                                                       (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(275, 125))
            self.obstacle_11.add_attr(self.offset)
            self.obstacle_11.set_color(0, 0, 0)

            self.obstacle_12 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, square_edge / 2),
                                                        (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(275, 225))
            self.obstacle_12.add_attr(self.offset)
            self.obstacle_12.set_color(0, 0, 0)

            self.obstacle_13 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, square_edge / 2),
                                                        (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(275, 375))
            self.obstacle_13.add_attr(self.offset)
            self.obstacle_13.set_color(0, 0, 0)

            self.obstacle_14 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, square_edge / 2),
                                                        (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(325, 275))
            self.obstacle_14.add_attr(self.offset)
            self.obstacle_14.set_color(0, 0, 0)

            self.obstacle_15 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, square_edge / 2),
                                                        (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(325, 525))
            self.obstacle_15.add_attr(self.offset)
            self.obstacle_15.set_color(0, 0, 0)

            self.obstacle_16 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, square_edge / 2),
                                                        (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(375, 75))
            self.obstacle_16.add_attr(self.offset)
            self.obstacle_16.set_color(0, 0, 0)

            self.obstacle_17 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, square_edge / 2),
                                                        (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(375, 175))
            self.obstacle_17.add_attr(self.offset)
            self.obstacle_17.set_color(0, 0, 0)

            self.obstacle_18 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, square_edge / 2),
                                                        (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(375, 375))
            self.obstacle_18.add_attr(self.offset)
            self.obstacle_18.set_color(0, 0, 0)

            self.obstacle_19 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, square_edge / 2),
                                                        (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(375, 475))
            self.obstacle_19.add_attr(self.offset)
            self.obstacle_19.set_color(0, 0, 0)

            self.obstacle_20 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, square_edge / 2),
                                                        (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(425, 275))
            self.obstacle_20.add_attr(self.offset)
            self.obstacle_20.set_color(0, 0, 0)

            self.obstacle_21 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, square_edge / 2),
                                                        (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(425, 325))
            self.obstacle_21.add_attr(self.offset)
            self.obstacle_21.set_color(0, 0, 0)

            self.obstacle_22 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, square_edge / 2),
                                                        (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(475, 175))
            self.obstacle_22.add_attr(self.offset)
            self.obstacle_22.set_color(0, 0, 0)

            self.obstacle_23 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, square_edge / 2),
                                                        (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(475, 225))
            self.obstacle_23.add_attr(self.offset)
            self.obstacle_23.set_color(0, 0, 0)

            self.obstacle_24 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, square_edge / 2),
                                                        (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(475, 375))
            self.obstacle_24.add_attr(self.offset)
            self.obstacle_24.set_color(0, 0, 0)

            self.obstacle_25 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, -square_edge / 2),
                                                        (square_edge / 2, square_edge / 2),
                                                        (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(525, 475))
            self.obstacle_25.add_attr(self.offset)
            self.obstacle_25.set_color(0, 0, 0)

            # set the goal
            self.goal = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                  (square_edge / 2, -square_edge / 2),
                                                  (square_edge / 2, square_edge / 2),
                                                  (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(525, 75))
            self.goal.add_attr(self.offset)
            self.goal.set_color(0, 1, 1)

            # set the initial pos
            self.robot = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                  (square_edge / 2, -square_edge / 2),
                                                  (square_edge / 2, square_edge / 2),
                                                  (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(25 + y * 50, 575 - x * 50))
            self.robot.add_attr(self.offset)
            self.robot.set_color(1, 0, 0)

            # 添加组件到Viewer中
            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.line12)
            self.viewer.add_geom(self.line13)
            self.viewer.add_geom(self.line14)
            self.viewer.add_geom(self.line15)
            self.viewer.add_geom(self.line16)
            self.viewer.add_geom(self.line17)
            self.viewer.add_geom(self.line18)
            self.viewer.add_geom(self.line19)
            self.viewer.add_geom(self.line20)
            self.viewer.add_geom(self.line21)
            self.viewer.add_geom(self.line22)

            self.viewer.add_geom(self.obstacle_1)
            self.viewer.add_geom(self.obstacle_2)
            self.viewer.add_geom(self.obstacle_3)
            self.viewer.add_geom(self.obstacle_4)
            self.viewer.add_geom(self.obstacle_5)
            self.viewer.add_geom(self.obstacle_6)
            self.viewer.add_geom(self.obstacle_7)
            self.viewer.add_geom(self.obstacle_8)
            self.viewer.add_geom(self.obstacle_9)
            self.viewer.add_geom(self.obstacle_10)
            self.viewer.add_geom(self.obstacle_11)
            self.viewer.add_geom(self.obstacle_12)
            self.viewer.add_geom(self.obstacle_13)
            self.viewer.add_geom(self.obstacle_14)
            self.viewer.add_geom(self.obstacle_15)
            self.viewer.add_geom(self.obstacle_16)
            self.viewer.add_geom(self.obstacle_17)
            self.viewer.add_geom(self.obstacle_18)
            self.viewer.add_geom(self.obstacle_19)
            self.viewer.add_geom(self.obstacle_20)
            self.viewer.add_geom(self.obstacle_21)
            self.viewer.add_geom(self.obstacle_22)
            self.viewer.add_geom(self.obstacle_23)
            self.viewer.add_geom(self.obstacle_24)
            self.viewer.add_geom(self.obstacle_25)
            self.viewer.add_geom(self.robot)
            self.viewer.add_geom(self.goal)
        else:
            # 创建机器人
            self.robot = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, square_edge / 2),
                                                       (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(25 + y * 50, 575 - x * 50))
            self.robot.add_attr(self.offset)
            self.robot.set_color(0, 0, 1)
            self.viewer.add_geom(self.robot)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render_4(self, x, y, mode='human'):
        screen_width = 600
        screen_height = 600
        square_edge = 100

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)  # 调用rendering中的画图函数，#创建600*600的窗口
            # 创建网格世界，一共包括10条直线，事先算好每条直线的起点和终点坐标，然后绘制这些直线，代码如下：
            # 创建网格世界
            self.line1 = rendering.Line((100, 100), (500, 100))
            self.line2 = rendering.Line((100, 200), (500, 200))
            self.line3 = rendering.Line((100, 300), (500, 300))
            self.line4 = rendering.Line((100, 400), (500, 400))
            self.line5 = rendering.Line((100, 500), (500, 500))
            self.line6 = rendering.Line((100, 100), (100, 500))
            self.line7 = rendering.Line((200, 100), (200, 500))
            self.line8 = rendering.Line((300, 100), (300, 500))
            self.line9 = rendering.Line((400, 100), (400, 500))
            self.line10 = rendering.Line((500, 100), (500, 500))

            # set the color of lines
            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)

            # create obstacles
            self.obstacle_1 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, square_edge / 2),
                                                       (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(150, 150))
            self.obstacle_1.add_attr(self.offset)

            self.obstacle_2 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, square_edge / 2),
                                                       (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(250, 350))
            self.obstacle_2.add_attr(self.offset)
            self.obstacle_2.set_color(0, 0, 0)

            self.obstacle_3 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, square_edge / 2),
                                                       (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(450, 250))
            self.obstacle_3.add_attr(self.offset)
            self.obstacle_3.set_color(0, 0, 0)

            self.obstacle_4 = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, -square_edge / 2),
                                                       (square_edge / 2, square_edge / 2),
                                                       (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(450, 350))
            self.obstacle_4.add_attr(self.offset)
            self.obstacle_4.set_color(0, 0, 0)

            # set the goal
            self.goal = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                 (square_edge / 2, -square_edge / 2),
                                                 (square_edge / 2, square_edge / 2),
                                                 (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(450, 150))
            self.goal.add_attr(self.offset)
            self.goal.set_color(0, 1, 1)

            # set the initial pos
            self.robot = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                  (square_edge / 2, -square_edge / 2),
                                                  (square_edge / 2, square_edge / 2),
                                                  (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(50 + y * 100, 550 - x * 100))
            self.robot.add_attr(self.offset)
            self.robot.set_color(1, 0, 0)

            # add the components to viewer
            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)

            self.viewer.add_geom(self.obstacle_1)
            self.viewer.add_geom(self.obstacle_2)
            self.viewer.add_geom(self.obstacle_3)
            self.viewer.add_geom(self.obstacle_4)

            self.viewer.add_geom(self.robot)
            self.viewer.add_geom(self.goal)
        else:
            # 创建机器人
            self.robot = rendering.FilledPolygon([(-square_edge / 2, -square_edge / 2),
                                                  (square_edge / 2, -square_edge / 2),
                                                  (square_edge / 2, square_edge / 2),
                                                  (-square_edge / 2, square_edge / 2)])
            self.offset = rendering.Transform(translation=(50 + y * 100, 550 - x * 100))
            self.robot.add_attr(self.offset)
            self.robot.set_color(0, 0, 1)
            self.viewer.add_geom(self.robot)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def observe(self):
        ...

    def reward(self):
        ...


if __name__ == '__main__':
    self = 0
    print(LiRobot.generate_ten_by_ten(self))
