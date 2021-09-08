import env_for_p1.envs.lirobot as lr
import parameters as param
import numpy as np
import agent


if __name__ == '__main__':
    env = lr.LiRobot()
    robot_li = agent.robot()

    return_saving = np.zeros((param.ENV_SETTINGS.MATRIX_SIZE, param.ENV_SETTINGS.MATRIX_SIZE, 4))
    for iteration in range(0, 200000):
        # prediction
        # robot initialization
        robot_li.obser, robot_li.pos = env.reset()
        G = 0
        reward = 0
        robot_li.sample_list = [[1, 1]]
        robot_li.action_list = []
        done = False
        while not done:
            # randomly choose the action according to the possibility of each state
            while reward == 0:
                index = np.random.choice([0, 1, 2, 3], 1, p=robot_li.probs[robot_li.pos[0]][robot_li.pos[1]]).item()
                done, robot_li.pos, reward = env.step(index)                # do a action and get the reward and state
            if robot_li.pos not in robot_li.sample_list:
                robot_li.sample_list.append(list(robot_li.pos))
            robot_li.action_list.append(index)
            G = reward
            reward = 0
        # x = robot_li.sample_list[-1][0]
        # y = robot_li.sample_list[-1][1]
        # return_saving[x][y] += G
        # robot_li.sample_num[x][y] += 1
        for j in range(len(robot_li.sample_list)-2, -1, -1):
            x = robot_li.sample_list[j][0]
            y = robot_li.sample_list[j][1]
            G = param.AGENT_ACTION.ACTION_REWARD + param.AGENT_ACTION.DISCOUNT_FACTOR * G
            return_saving[x][y][robot_li.action_list[j]] += G
            robot_li.sample_num_Q[x][y][robot_li.action_list[j]] += 1

        # calculate the expectation of the state value
        for i in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
            for j in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
                for k in range(0, 4):
                    if not return_saving[i][j][k] == 0:
                        robot_li.value_Q[i][j][k] = return_saving[i][j][k] / robot_li.sample_num_Q[i][j][k]

        print(iteration)
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
                action_index = next_value_list.index(max(next_value_list))
                for k in range(0, 4):
                    if action_index == k:
                        robot_li.probs[i][j][k] = 1 - param.AGENT_ACTION.EPSILON + param.AGENT_ACTION.EPSILON / 4
                    else:
                        robot_li.probs[i][j][k] = param.AGENT_ACTION.EPSILON / 4

        for i in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
            for j in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
                for k in range(0, 4):
                    param.ENV_SETTINGS.STATE_ACTION_VALUE[i][j][k] = robot_li.value_Q[i][j][k]

        # np.set_printoptions(linewidth=400)
        # for i in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
        #     print(param.ENV_SETTINGS.STATE_ACTION_VALUE[i])

    print(env.world)

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
        print("Monte Carlo Q: ", success)
    else:
        print("Monte Carlo Q: ", success)
