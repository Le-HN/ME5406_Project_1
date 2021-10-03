import env_for_p1.envs.lirobot as lr
import matplotlib.pyplot as plt
import parameters as param
import numpy as np
import agent
import time


def sarsa_q(map_size, iteration_lim):
    # variables to store the statics for plot
    average_q_value_list = []
    average_q_value = 0
    q_value_counter = 0
    average_reward_list = []
    average_reward = 0
    episode_list = []
    episode = 0

    # instantiate the robot and environment
    if map_size == 10:
        robot_li = agent.robot()
        env = lr.LiRobot(size=10)
    elif map_size == 4:
        robot_li = agent.robot_size_4()
        env = lr.LiRobot(size=4)

    map_plus_wall_size = map_size + 2

    for iteration in range(0, iteration_lim):

        if (iteration + 1) % 100 == 0:
            print("Training episode: ", iteration + 1)
        # prediction
        # robot initialization
        robot_li.obser, robot_li.pos = env.reset()
        reward = 0
        robot_li.sample_list = [[1, 1]]
        # robot_li.action_list = []
        done = False
        first_time = True
        current_action_index = 0
        action_value = 0
        index = 0
        # pre_pos = [1, 1]
        while not done:
            # control
            for i in range(1, map_plus_wall_size - 1):
                for j in range(1, map_plus_wall_size - 1):
                    next_value_list = [robot_li.value_Q[i][j][0],
                                       robot_li.value_Q[i][j][1],
                                       robot_li.value_Q[i][j][2],
                                       robot_li.value_Q[i][j][3]]
                    if robot_li.obser[i][j + 1] == -2:
                        robot_li.value_Q[i][j][0] = -10
                    if robot_li.obser[i + 1][j] == -2:
                        robot_li.value_Q[i][j][1] = -10
                    if robot_li.obser[i][j - 1] == -2:
                        robot_li.value_Q[i][j][2] = -10
                    if robot_li.obser[i - 1][j] == -2:
                        robot_li.value_Q[i][j][3] = -10
                    action_index = next_value_list.index(max(next_value_list))
                    for k in range(0, 4):
                        if action_index == k:
                            robot_li.probs[i][j][k] = 1 - param.AGENT_ACTION.EPSILON + param.AGENT_ACTION.EPSILON / 4
                        else:
                            robot_li.probs[i][j][k] = param.AGENT_ACTION.EPSILON / 4

            # randomly choose the action according to the possibility of each state
            if first_time:
                while reward == 0:
                    index = np.random.choice([0, 1, 2, 3], 1, p=robot_li.probs[robot_li.pos[0]][robot_li.pos[1]]).item()
                    done, robot_li.pos, reward = env.step(index)  # do a action and get the reward and state
                first_time = False
            else:
                index = current_action_index
                done, robot_li.pos, reward = env.step(index)  # do a action and get the reward and state
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
            # robot_li.action_list.append(index)
            x = robot_li.sample_list[-2][0]
            y = robot_li.sample_list[-2][1]

            valid_action = False
            while (not valid_action) and (reward == 0):
                action_value = np.random.choice(robot_li.value_Q[robot_li.pos[0]][robot_li.pos[1]], 1, p=robot_li.probs[robot_li.pos[0]][robot_li.pos[1]]).item()
                for k in range(0, 4):
                    if action_value == robot_li.value_Q[robot_li.pos[0]][robot_li.pos[1]][k]:
                        current_action_index = k
                        break
                assume_x = robot_li.pos[0] + param.AGENT_ACTION.ACTION_SPACE[current_action_index][0]
                assume_y = robot_li.pos[1] + param.AGENT_ACTION.ACTION_SPACE[current_action_index][1]
                if robot_li.obser[assume_x][assume_y] == -2:
                    valid_action = False
                else:
                    valid_action = True
            if reward == 1 or reward == -1:
                action_value = 0

            robot_li.value_Q[x][y][index] += param.AGENT_ACTION.L_RATE * (reward +
                                                                          param.AGENT_ACTION.DISCOUNT_FACTOR * action_value -
                                                                          robot_li.value_Q[x][y][index])

            reward = 0

        # for pos in robot_li.sample_list:
        #     if pos != (param.ENV_SETTINGS.MATRIX_SIZE - 2, param.ENV_SETTINGS.MATRIX_SIZE - 2):
        #         env.render_4(pos[0], pos[1])
        #         time.sleep(0.01)
        # env.viewer.close()
        # print(iteration)
        # print(robot_li.action_list)
        # print(return_saving)
        # print(robot_li.sample_num)

        # calculate the value
        for i in range(0, map_plus_wall_size):
            for j in range(0, map_plus_wall_size):
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
        if sum > pow(map_plus_wall_size - 2, 2):
            success = False
            break
    route.append((map_plus_wall_size - 2, map_plus_wall_size - 2))
    if env.world[x][y] != 1:
        success = False

    print(env.world)

    if success:
        print(route)
        print("SARSA: ", success)
    else:
        print("SARSA: ", success)
    return episode_list, average_reward_list, average_q_value_list, env.world, route, env, success


def successful_times_test():
    test_iteration = 100
    successful_num = 0
    for i in range(0, test_iteration):
        e_list, ar_list, ar_q_list, world, route, env, result = sarsa_q(map_size=10, iteration_lim=3000)
        if result:
            successful_num += 1
    failed_num = test_iteration - successful_num
    successful_list = [successful_num]
    failed_list = [failed_num]

    bar_width = 0.8
    bar_label = ("success", "failure")
    successful_index = 1
    failed_index = successful_index + bar_width
    plt.bar(successful_index, successful_list, width=bar_width, label='success')
    plt.bar(failed_index, failed_list, width=bar_width, label='failure')

    plt.legend()
    plt.xticks([successful_index, failed_index], bar_label)
    plt.ylim(0, test_iteration)
    plt.ylabel("Number")
    plt.show()


if __name__ == '__main__':

    map_plus_wall_size = 12

    if param.TEST:
        # count times of success then plot
        successful_times_test()
    else:
        e_list, ar_list, ar_q_list, world, route, env, result = sarsa_q(map_size=10, iteration_lim=10000)

        plt.plot(e_list, ar_list, label="SARSA")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.ylim(-1.5, 1.5)
        plt.legend()
        plt.show()

        plt.plot(e_list, ar_q_list, label="SARSA")
        plt.xlabel("Episode")
        plt.ylabel("Average Q Value")
        plt.ylim(-0.1, 0.1)
        plt.legend()
        plt.show()

        if result:
            for pos in route:
                if pos != (map_plus_wall_size - 2, map_plus_wall_size - 2):
                    if map_plus_wall_size - 2 == 10:
                        env.render_10(pos[0], pos[1])
                    elif map_plus_wall_size - 2 == 4:
                        env.render_4(pos[0], pos[1])
                    time.sleep(1)
        # else:
        #     tkinter.messagebox.showinfo(title='Note', message='Finding route failed!')
