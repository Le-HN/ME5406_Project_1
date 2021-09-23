import env_for_p1.envs.lirobot as lr
import matplotlib.pyplot as plt
import tkinter.messagebox
import parameters as param
import numpy as np
import time
import agent


def monte_carlo_q(iteration_lim):
    # variables to store the statics for plot
    # average q value
    average_q_value_list = []
    average_q_value = 0
    q_value_counter = 0

    # average reward
    average_reward_list = []
    average_reward = 0

    # episode
    episode_list = []
    episode = 0

    # instantiate the robot and environment
    robot_li = agent.robot()
    env = lr.LiRobot(size=10)

    # to save the return G of each state-action
    return_saving = np.zeros((param.ENV_SETTINGS.MATRIX_SIZE_SHOW, param.ENV_SETTINGS.MATRIX_SIZE_SHOW, 4))

    for iteration in range(0, iteration_lim):
        # prediction
        # robot initialization
        robot_li.obser, robot_li.pos = env.reset()
        # initialize parameters
        G = 0
        reward = 0
        robot_li.sample_list = [[1, 1]]
        robot_li.action_list = []
        done = False

        while not done:

            # the action was chose according to the possibility of each state in the possibility matrix
            while reward == 0:
                index = np.random.choice([0, 1, 2, 3], 1, p=robot_li.probs[robot_li.pos[0]][robot_li.pos[1]]).item()
                # do a action and get the reward and state
                done, robot_li.pos, reward = env.step(index)

            # at the beginning, there was a reward in each step, but it didn't work properly
            # so eliminate the effect of the step reward, except reaching the terminal state
            if reward != -1 and reward != 1:
                reward = 0
            # calculate the average reward
            else:
                average_reward += reward
                episode += 1
                if episode % 20 == 0:
                    average_reward_list.append(average_reward / episode)
                    episode_list.append(episode)

            # record the trajectory and the action
            if robot_li.pos not in robot_li.sample_list:
                robot_li.sample_list.append(list(robot_li.pos))
                robot_li.action_list.append(index)
            else:
                state_index = robot_li.sample_list.index(robot_li.pos)
                if robot_li.action_list[state_index-1] != index:
                    robot_li.sample_list.append(list(robot_li.pos))
                    robot_li.action_list.append(index)
            G = reward
            reward = 0

        # use return to update the q value according to the trajectory
        for j in range(len(robot_li.sample_list)-2, -1, -1):
            x = robot_li.sample_list[j][0]
            y = robot_li.sample_list[j][1]
            G = param.AGENT_ACTION.ACTION_REWARD + param.AGENT_ACTION.DISCOUNT_FACTOR * G
            return_saving[x][y][robot_li.action_list[j]] += G
            robot_li.sample_num_Q[x][y][robot_li.action_list[j]] += 1

        # calculate the expectation of the q value
        for i in range(0, param.ENV_SETTINGS.MATRIX_SIZE_SHOW):
            for j in range(0, param.ENV_SETTINGS.MATRIX_SIZE_SHOW):
                for k in range(0, 4):
                    if not return_saving[i][j][k] == 0:
                        robot_li.value_Q[i][j][k] = return_saving[i][j][k] / robot_li.sample_num_Q[i][j][k]

        # print the route to debug
        # for pos in robot_li.sample_list:
        #     if pos != (param.ENV_SETTINGS.MATRIX_SIZE - 2, param.ENV_SETTINGS.MATRIX_SIZE - 2):
        #         env.render_4(pos[0], pos[1])
        #         time.sleep(0.01)
        # env.viewer.close()
        if episode % 1000 == 0:
            print(episode)
        # print(robot_li.action_list)
        # print(return_saving)
        # print(robot_li.sample_num)

        # control
        for i in range(1, param.ENV_SETTINGS.MATRIX_SIZE_SHOW - 1):
            for j in range(1, param.ENV_SETTINGS.MATRIX_SIZE_SHOW - 1):
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

                action_index = next_value_list.index(max(next_value_list))
                for k in range(0, 4):
                    if action_index == k:
                        robot_li.probs[i][j][k] = 1 - param.AGENT_ACTION.EPSILON + param.AGENT_ACTION.EPSILON / 4
                    else:
                        robot_li.probs[i][j][k] = param.AGENT_ACTION.EPSILON / 4

        for i in range(0, param.ENV_SETTINGS.MATRIX_SIZE_SHOW):
            for j in range(0, param.ENV_SETTINGS.MATRIX_SIZE_SHOW):
                for k in range(0, 4):
                    param.ENV_SETTINGS.STATE_ACTION_VALUE[i][j][k] = robot_li.value_Q[i][j][k]
                    if robot_li.value_Q[i][j][k] != -10 and episode % 20 == 0:
                        average_q_value += robot_li.value_Q[i][j][k]
                        q_value_counter += 1
        if episode % 20 == 0:
            average_q_value_list.append(average_q_value / q_value_counter)
            average_q_value = 0
            q_value_counter = 0

        # np.set_printoptions(linewidth=400)
        # for i in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
        #     print(param.ENV_SETTINGS.STATE_ACTION_VALUE[i])

    # print(env.world)

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
        if sum > pow(param.ENV_SETTINGS.MATRIX_SIZE_SHOW - 2, 2):
            success = False
            break
    route.append((param.ENV_SETTINGS.MATRIX_SIZE_SHOW - 2, param.ENV_SETTINGS.MATRIX_SIZE_SHOW - 2))
    if env.world[x][y] != 1:
        success = False
    if success:
        print(route)
        print("Monte Carlo Q: ", success)
    else:
        print("Monte Carlo Q: ", success)
    return episode_list, average_reward_list, average_q_value_list, env.world, route, env, success


if __name__ == '__main__':

    # result = False
    # i = 0
    # while not result:
    e_list, ar_list, ar_q_list, world, route, env, result = monte_carlo_q(iteration_lim=400000)
        # i += 1
        # print(i)

    plt.plot(e_list, ar_list, label="Monte Carlo Q")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.show()

    plt.plot(e_list, ar_q_list, label="Monte Carlo Q")
    plt.xlabel("Episode")
    plt.ylabel("Average Q Value")
    plt.ylim(-0.2, 0.2)
    plt.legend()
    plt.show()

    if result:
        for pos in route:
            if pos != (param.ENV_SETTINGS.MATRIX_SIZE_SHOW - 2, param.ENV_SETTINGS.MATRIX_SIZE_SHOW - 2):
                env.render_10(pos[0], pos[1])
                time.sleep(1)
    else:
        tkinter.messagebox.showinfo(title='Note', message='Finding route failed!')
