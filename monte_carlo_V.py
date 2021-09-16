import env_for_p1.envs.lirobot as lr
import tkinter.messagebox
import parameters as param
import numpy as np
import time
import agent


def monte_carlo_v(iteration_lim):
    successful_num = 0
    shortest_num = 0
    robot_li = agent.robot_size_4()
    env = lr.LiRobot(size=4)

    return_saving = np.zeros((param.ENV_SETTINGS.MATRIX_SIZE_SHOW, param.ENV_SETTINGS.MATRIX_SIZE_SHOW))
    for iteration in range(0, iteration_lim):
        # prediction
        # robot initialization
        robot_li.obser, robot_li.pos = env.reset()
        G = 0
        reward = 0
        robot_li.sample_list = [[1, 1]]
        done = False
        while not done:
            # randomly choose the action according to the possibility of each state
            while reward == 0:
                index = np.random.choice([0, 1, 2, 3], 1, p=robot_li.probs[robot_li.pos[0]][robot_li.pos[1]]).item()
                done, robot_li.pos, reward = env.step(index)  # do a action and get the reward and state
            if robot_li.pos not in robot_li.sample_list:
                robot_li.sample_list.append(list(robot_li.pos))
            if reward != -1 and reward != 1:
                reward = 0
            G = reward
            reward = 0

        x = robot_li.sample_list[-1][0]
        y = robot_li.sample_list[-1][1]
        return_saving[x][y] += G
        robot_li.sample_num[x][y] += 1
        for j in range(len(robot_li.sample_list) - 2, -1, -1):
            x = robot_li.sample_list[j][0]
            y = robot_li.sample_list[j][1]
            G = param.AGENT_ACTION.ACTION_REWARD + param.AGENT_ACTION.DISCOUNT_FACTOR * G
            return_saving[x][y] += G
            robot_li.sample_num[x][y] += 1

        # calculate the expectation of the state value
        for i in range(0, param.ENV_SETTINGS.MATRIX_SIZE_SHOW):
            for j in range(0, param.ENV_SETTINGS.MATRIX_SIZE_SHOW):
                if not return_saving[i][j] == 0:
                    robot_li.value[i][j] = return_saving[i][j] / robot_li.sample_num[i][j]

        # print(iteration)
        # print(robot_li.value)
        # print(robot_li.sample_num)

        # control
        for i in range(1, param.ENV_SETTINGS.MATRIX_SIZE_SHOW - 1):
            for j in range(1, param.ENV_SETTINGS.MATRIX_SIZE_SHOW - 1):
                next_value_list = [robot_li.value[i][j + 1],
                                   robot_li.value[i + 1][j],
                                   robot_li.value[i][j - 1],
                                   robot_li.value[i - 1][j]]
                action_index = next_value_list.index(max(next_value_list))
                for k in range(0, 4):
                    if action_index == k:
                        robot_li.probs[i][j][k] = 1 - param.AGENT_ACTION.EPSILON + param.AGENT_ACTION.EPSILON / 4
                    else:
                        robot_li.probs[i][j][k] = param.AGENT_ACTION.EPSILON / 4

        # calculate the value
        for i in range(0, param.ENV_SETTINGS.MATRIX_SIZE_SHOW):
            for j in range(0, param.ENV_SETTINGS.MATRIX_SIZE_SHOW):
                if not robot_li.value[i][j] == 0:
                    param.ENV_SETTINGS.VALUE_ARRAY[i][j] = robot_li.value[i][j]
                else:
                    param.ENV_SETTINGS.VALUE_ARRAY[i][j] = 0
        np.set_printoptions(linewidth=400)
        print(param.ENV_SETTINGS.VALUE_ARRAY)
    print(env.world)

    robot_li.pos = [1, 1]
    route = []
    x = robot_li.pos[0]
    y = robot_li.pos[1]
    sum = 0
    success = True
    while env.world[x][y] == 0:
        robot_li.pos = [x, y]
        route.append((x, y))
        x = robot_li.pos[0]
        y = robot_li.pos[1]
        next_value_list = [robot_li.value[x][y + 1],
                           robot_li.value[x + 1][y],
                           robot_li.value[x][y - 1],
                           robot_li.value[x - 1][y]]
        action_index = next_value_list.index(max(next_value_list))

        x += param.AGENT_ACTION.ACTION_SPACE[action_index][0]
        y += param.AGENT_ACTION.ACTION_SPACE[action_index][1]

        sum += 1
        if sum > pow(param.ENV_SETTINGS.MATRIX_SIZE_SHOW - 2, 2):
            success = False
            break
    route.append((x, y))
    if env.world[x][y] != 1:
        success = False
    if success:
        print(route)
        print("Monte Carlo V: ", success)
        successful_num += 1
        if len(route) == (param.ENV_SETTINGS.MATRIX_SIZE_SHOW - 2) * 2 - 1:
            shortest_num += 1
    else:
        print("Monte Carlo V: ", success)
    return successful_num, shortest_num, env.world, route, env, success


if __name__ == '__main__':

    sc_n, st_n, world, route, env, result = monte_carlo_v(iteration_lim=500)
    if result:
        for pos in route:
            if pos != (param.ENV_SETTINGS.MATRIX_SIZE_SHOW - 2, param.ENV_SETTINGS.MATRIX_SIZE_SHOW - 2):
                env.render_4(pos[0], pos[1])
                time.sleep(0.5)
    else:
        tkinter.messagebox.showinfo(title='Note', message='Finding route failed!')