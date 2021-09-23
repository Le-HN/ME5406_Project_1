import time
import env_for_p1.envs.lirobot as lr
import matplotlib.pyplot as plt
import tkinter.messagebox
import parameters as param
import numpy as np
import agent


def q_learning(iteration_lim):
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

    for iteration in range(0, iteration_lim):

        # prediction
        # robot initialization
        robot_li.obser, robot_li.pos = env.reset()
        reward = 0
        # initialize the trajectory list
        robot_li.sample_list = [[1, 1]]
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
            else:
                average_reward += reward
                episode += 1
                if episode % 20 == 0:
                    average_reward_list.append(average_reward / episode)
                    episode_list.append(episode)
            # save the trajectory
            robot_li.sample_list.append(list(robot_li.pos))

            # save the previous state position
            x = robot_li.sample_list[-2][0]
            y = robot_li.sample_list[-2][1]

            # choose the max q value in the current state
            q_current = max(robot_li.value_Q[robot_li.pos[0]][robot_li.pos[1]])

            # if terminal state is reached, q value will be 0
            if reward == 1 or reward == -1:
                q_current = 0

            # update the q value of the previous state
            robot_li.value_Q[x][y][index] += param.AGENT_ACTION.L_RATE * (reward + \
                                                                          param.AGENT_ACTION.DISCOUNT_FACTOR * q_current - robot_li.value_Q[x][y][index])
            reward = 0

        # print the route to debug
        # for pos in robot_li.sample_list:
        #     if pos != (param.ENV_SETTINGS.MATRIX_SIZE - 2, param.ENV_SETTINGS.MATRIX_SIZE - 2):
        #         env.render_10(pos[0], pos[1])
        #         time.sleep(0.01)
        # env.viewer.close()

        # print(iteration)

        # control
        for i in range(1, param.ENV_SETTINGS.MATRIX_SIZE - 1):
            for j in range(1, param.ENV_SETTINGS.MATRIX_SIZE - 1):
                next_value_list = [robot_li.value_Q[i][j][0],
                                   robot_li.value_Q[i][j][1],
                                   robot_li.value_Q[i][j][2],
                                   robot_li.value_Q[i][j][3]]
                # if the robot will hit the wall after the action,
                # the q value of this action under this state will be -10
                # so that it will not be choose
                if robot_li.obser[i][j+1] == -2:
                    robot_li.value_Q[i][j][0] = -10
                if robot_li.obser[i+1][j] == -2:
                    robot_li.value_Q[i][j][1] = -10
                if robot_li.obser[i][j-1] == -2:
                    robot_li.value_Q[i][j][2] = -10
                if robot_li.obser[i-1][j] == -2:
                    robot_li.value_Q[i][j][3] = -10

                # record the max q value direction (index) to update the possibility matrix
                action_index = next_value_list.index(max(next_value_list))

                # using epsilon-greedy method to update the possibilities of actions in each state and
                # save possibilities in the possibility matrix
                for k in range(0, 4):
                    if action_index == k:
                        robot_li.probs[i][j][k] = 1 - param.AGENT_ACTION.EPSILON + param.AGENT_ACTION.EPSILON / 4
                    else:
                        robot_li.probs[i][j][k] = param.AGENT_ACTION.EPSILON / 4

        # save the value to the q table and calculate the average q_value
        for i in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
            for j in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
                for k in range(0, 4):
                    param.ENV_SETTINGS.STATE_ACTION_VALUE[i][j][k] = robot_li.value_Q[i][j][k]
                    if robot_li.value_Q[i][j][k] != -10 and episode % 20 == 0:
                        average_q_value += robot_li.value_Q[i][j][k]
                        q_value_counter += 1
        if episode % 20 == 0:
            average_q_value_list.append(average_q_value / q_value_counter)
            average_q_value = 0
            q_value_counter = 0

        # print the q value to debug
        # np.set_printoptions(linewidth=400)
        # for i in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
        #     print(param.ENV_SETTINGS.STATE_ACTION_VALUE[i])

    # test
    # to find a route from start point to the destination
    # initialize the position
    robot_li.pos = [1, 1]

    # initialize the trajectory set
    route = []
    x = robot_li.pos[0]
    y = robot_li.pos[1]
    sum = 0
    success = True
    # find the route according to the q value
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
        # if the number of steps is out of the size of map, stop
        if sum > pow(param.ENV_SETTINGS.MATRIX_SIZE - 2, 2):
            success = False
            break
    route.append((param.ENV_SETTINGS.MATRIX_SIZE - 2, param.ENV_SETTINGS.MATRIX_SIZE - 2))
    # if the ending is not 1, path finding failed
    if env.world[x][y] != 1:
        success = False
    if success:
        print(route)
        print("Q_Learning: ", success)
    else:
        print("Q_Learning: ", success)

    return episode_list, average_reward_list, average_q_value_list, env.world, route, env, success


if __name__ == '__main__':

    e_list, ar_list, ar_q_list, world, route, env, result = q_learning(iteration_lim=10000)
    plt.plot(e_list, ar_list, label="Q-Learning")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.show()

    plt.plot(e_list, ar_q_list, label="Q-Learning")
    plt.xlabel("Episode")
    plt.ylabel("Average Q Value")
    plt.ylim(-0.1, 0.1)
    plt.legend()
    plt.show()
    # if success, render the route
    if result:
        for pos in route:
            if pos != (param.ENV_SETTINGS.MATRIX_SIZE - 2, param.ENV_SETTINGS.MATRIX_SIZE - 2):
                env.render_10(pos[0], pos[1])
                time.sleep(1)
    # if not success, show the failed result
    else:
        tkinter.messagebox.showinfo(title='Note', message='Finding route failed!')
