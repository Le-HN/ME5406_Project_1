import env_for_p1.envs.lirobot as lr
import parameters as param
import numpy as np
import agent


if __name__ == '__main__':
    env = lr.LiRobot()
    robot_li = agent.robot()

    return_saving = np.zeros((param.ENV_SETTINGS.MATRIX_SIZE, param.ENV_SETTINGS.MATRIX_SIZE))
    for iteration in range(0, 20000):
        # prediction
        # robot initialization
        robot_li.obser, robot_li.pos = env.reset()
        G = 0
        robot_li.sample_list = [[1, 1]]
        done = False
        while not done:
            # randomly choose the action according to the possibility of each state
            index = np.random.choice([0, 1, 2, 3], 1, p=robot_li.probs[robot_li.pos[0]][robot_li.pos[1]]).item()
            done, robot_li.pos, reward = env.step(index)                # do a action and get the reward and state
            if robot_li.pos not in robot_li.sample_list:
                robot_li.sample_list.append(list(robot_li.pos))
        G = reward
        x = robot_li.sample_list[-1][0]
        y = robot_li.sample_list[-1][1]
        return_saving[x][y] += G
        robot_li.sample_num[x][y] += 1
        for j in range(len(robot_li.sample_list)-2, -1, -1):
            x = robot_li.sample_list[j][0]
            y = robot_li.sample_list[j][1]
            G = param.AGENT_ACTION.ACTION_REWARD + param.AGENT_ACTION.DISCOUNT_FACTOR * G
            return_saving[x][y] += G
            robot_li.sample_num[x][y] += 1

        # calculate the expectation of the state value
        for i in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
            for j in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
                if not return_saving[i][j] == 0:
                    robot_li.value[i][j] = return_saving[i][j] / robot_li.sample_num[i][j]

        print(iteration)
        # print(robot_li.value)
        # print(robot_li.sample_num)

        # control
        for i in range(1, param.ENV_SETTINGS.MATRIX_SIZE - 1):
            for j in range(1, param.ENV_SETTINGS.MATRIX_SIZE - 1):
                next_value_list = [robot_li.value[i][j + 1],
                                   robot_li.value[i + 1][j],
                                   robot_li.value[i][j - 1],
                                   robot_li.value[i - 1][j]]
                action_index = next_value_list.index(max(next_value_list))
                if 0 not in next_value_list:
                    for k in range(0, 4):
                        if action_index == k:
                            robot_li.probs[i][j][k] = param.AGENT_ACTION.EPSILON
                        else:
                            robot_li.probs[i][j][k] = (1 - param.AGENT_ACTION.EPSILON) / 3
                else:
                    # continue
                    action_index = next_value_list.index(0)
                    for k in range(0, 4):
                        if action_index == k:
                            robot_li.probs[i][j][k] = param.AGENT_ACTION.EPSILON
                        else:
                            robot_li.probs[i][j][k] = (1 - param.AGENT_ACTION.EPSILON) / 3
                # for k in range(0, 4):
                #     if action_index == k:
                #         robot_li.probs[i][j][k] = param.AGENT_ACTION.EPSILON
                #     else:
                #         robot_li.probs[i][j][k] = (1 - param.AGENT_ACTION.EPSILON) / 3

        # calculate the value
        for i in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
            for j in range(0, param.ENV_SETTINGS.MATRIX_SIZE):
                if not robot_li.value[i][j] == 0:
                    param.ENV_SETTINGS.VALUE_ARRAY[i][j] = robot_li.value[i][j]
                else:
                    param.ENV_SETTINGS.VALUE_ARRAY[i][j] = 0
        np.set_printoptions(linewidth=400)
        print(param.ENV_SETTINGS.VALUE_ARRAY)
    print(env.world)

