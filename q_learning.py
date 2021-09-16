import time
import env_for_p1.envs.lirobot as lr
import tkinter.messagebox
import parameters as param
import numpy as np
import agent


def q_learning(iteration_lim):
    successful_num = 0
    shortest_num = 0
    robot_li = agent.robot()
    env = lr.LiRobot(size=10)

    for iteration in range(0, iteration_lim):
        # prediction
        # robot initialization
        robot_li.obser, robot_li.pos = env.reset()
        reward = 0
        robot_li.sample_list = [[1, 1]]
        robot_li.action_list = []
        done = False
        while not done:
            # randomly choose the action according to the possibility of each state
            while reward == 0:
                index = np.random.choice([0, 1, 2, 3], 1, p=robot_li.probs[robot_li.pos[0]][robot_li.pos[1]]).item()
                done, robot_li.pos, reward = env.step(index)  # do a action and get the reward and state
            if reward != -1 and reward != 1:
                reward = 0
            if robot_li.pos not in robot_li.sample_list:
                robot_li.sample_list.append(list(robot_li.pos))
            x = robot_li.sample_list[-2][0]
            y = robot_li.sample_list[-2][1]
            # robot_li.action_list.append(index)
            q_current = max(robot_li.value_Q[robot_li.pos[0]][robot_li.pos[1]])
            if reward == 1 or reward == -1:
                q_current = 0
            robot_li.value_Q[x][y][index] += param.AGENT_ACTION.L_RATE * (reward + \
                                                                          param.AGENT_ACTION.DISCOUNT_FACTOR * q_current - robot_li.value_Q[x][y][index])
            reward = 0

        # for pos in robot_li.sample_list:
        #     if pos != (param.ENV_SETTINGS.MATRIX_SIZE - 2, param.ENV_SETTINGS.MATRIX_SIZE - 2):
        #         env.render_10(pos[0], pos[1])
        #         time.sleep(0.01)
        # env.viewer.close()

        # print(iteration)
        # print(robot_li.action_list)
        # print(return_saving)
        # print(robot_li.sample_num)

        # control
        for i in range(1, param.ENV_SETTINGS.MATRIX_SIZE - 1):
            for j in range(1, param.ENV_SETTINGS.MATRIX_SIZE - 1):
                next_value_list = [robot_li.value_Q[i][j][0],
                                   robot_li.value_Q[i][j][1],
                                   robot_li.value_Q[i][j][2],
                                   robot_li.value_Q[i][j][3]]
                if robot_li.obser[i][j+1] == -2:
                    robot_li.value_Q[i][j][0] = -10
                if robot_li.obser[i+1][j] == -2:
                    robot_li.value_Q[i][j][1] = -10
                if robot_li.obser[i][j-1] == -2:
                    robot_li.value_Q[i][j][2] = -10
                if robot_li.obser[i-1][j] == -2:
                    robot_li.value_Q[i][j][3] = -10
                # indexes = [k for k, x in enumerate(next_value_list) if x == max(next_value_list)]
                # prob_max = [0 for _ in range(len(indexes))]
                # for k in range(0, len(indexes)):
                #     prob_max[k] = 1 / len(indexes)
                # action_index = np.random.choice(indexes, 1, prob_max)
                action_index = next_value_list.index(max(next_value_list))

                # zero_counter = [k for k, x in enumerate(next_value_list) if x == 0]
                # if len(zero_counter) == 4:
                #     continue
                # else:
                for k in range(0, 4):
                    if action_index == k:
                        robot_li.probs[i][j][k] = 1 - param.AGENT_ACTION.EPSILON + param.AGENT_ACTION.EPSILON / 4
                    else:
                        robot_li.probs[i][j][k] = param.AGENT_ACTION.EPSILON / 4

        # calculate the value
        for i in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
            for j in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
                for k in range(0, 4):
                    param.ENV_SETTINGS.STATE_ACTION_VALUE[i][j][k] = robot_li.value_Q[i][j][k]

        # np.set_printoptions(linewidth=400)
        # for i in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
        #     print(param.ENV_SETTINGS.STATE_ACTION_VALUE[i])

    # test
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
        direction_value = max(param.ENV_SETTINGS.STATE_ACTION_VALUE[x][y])
        for i in range(0, 4):
            if param.ENV_SETTINGS.STATE_ACTION_VALUE[x][y][i] == direction_value:
                direction = i
                break

        x += param.AGENT_ACTION.ACTION_SPACE[direction][0]
        y += param.AGENT_ACTION.ACTION_SPACE[direction][1]

        sum += 1
        if sum > pow(param.ENV_SETTINGS.MATRIX_SIZE - 2, 2):
            success = False
            break
    route.append((param.ENV_SETTINGS.MATRIX_SIZE - 2, param.ENV_SETTINGS.MATRIX_SIZE - 2))
    if env.world[x][y] != 1:
        success = False
    if success:
        print(route)
        print("Q_Learning: ", success)
        successful_num += 1
        if len(route) == (param.ENV_SETTINGS.MATRIX_SIZE - 2) * 2 - 1:
            shortest_num += 1
    else:
        print("Q_Learning: ", success)
    return successful_num, shortest_num, env.world, route, env, success


if __name__ == '__main__':

    sc_n, st_n, world, route, env, result = q_learning(iteration_lim=500)
    if result:
        for pos in route:
            if pos != (param.ENV_SETTINGS.MATRIX_SIZE - 2, param.ENV_SETTINGS.MATRIX_SIZE - 2):
                env.render_10(pos[0], pos[1])
                time.sleep(0.5)
    else:
        tkinter.messagebox.showinfo(title='Note', message='Finding route failed!')
