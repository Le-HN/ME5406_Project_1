import os
import sys
import getopt
from q_learning import *
from sarsa_Q import *
from monte_carlo_Q import *


def main(argv):
    '''
    :param argv: the command or parameters user given
    :return: result, route, map_size, env to create a visualized route
    '''

    # 3 types: MONTE_CARLO, SARSA, Q_LEARNING, input by user
    algorithm_type = ""

    # 2 map size: 4, 10, input by user
    map_size = ""

    # the number of training iteration, input by user
    training_iteration = ""

    # receive the command and parameters from commandline and parse them, if not correct, exit
    try:
        opts, args = getopt.getopt(argv, "ht:s:i:", ["help", "algorithm_type=", "map_size=", "training_iteration="])
    except getopt.GetoptError:
        print('Error: main.py -t <algorithm_type> -s <map_size> -i <training_iteration>')
        print('   or: main.py --algorithm_type=<algorithm_type> --map_size=<map_size> --training_iteration=<training_iteration>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('main.py -t <algorithm_type> -s <map_size> -i <training_iteration>')
            print('or: main.py --algorithm_type=<algorithm_type> --map_size=<map_size> --training_iteration=<training_iteration>')
            sys.exit()
        elif opt in ("-t", "--algorithm_type"):
            algorithm_type = arg
        elif opt in ("-s", "--map_size"):
            map_size = arg
        elif opt in ("-i", "--training_iteration"):
            training_iteration = arg

    # remind the user what they are doing
    print('Algorithm type:', algorithm_type)
    print('Map size:', map_size)
    print('Training iteration:', training_iteration)
    print('Training process will be activated.')

    # choose the type according to the command
    if algorithm_type == "Q_LEARNING":
        e_l, ar_l, ar_q_l, world, route, env, result = q_learning(map_size=int(map_size), iteration_lim=int(training_iteration))
    elif algorithm_type == "SARSA":
        e_l, ar_l, ar_q_l, world, route, env, result = sarsa_q(map_size=int(map_size), iteration_lim=int(training_iteration))
    elif algorithm_type == "MONTE_CARLO":
        e_l, ar_l, ar_q_l, world, route, env, result = monte_carlo_q(map_size=int(map_size), iteration_lim=int(training_iteration))
    else:
        print("Wrong type!")
        sys.exit()

    # plot the average reward
    plt.plot(e_l, ar_l, label=algorithm_type)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.show()

    # plot the average q value
    plt.plot(e_l, ar_q_l, label=algorithm_type)
    plt.xlabel("Episode")
    plt.ylabel("Average Q Value")
    plt.ylim(-0.1, 0.1)
    plt.legend()
    plt.show()

    return result, route, map_size, env


if __name__ == "__main__":

    # get parameters from command line
    result, route, map_size, env = main(sys.argv[1:])

    # if success, render the route
    if result:
        for pos in route:
            if pos != (int(map_size), int(map_size)):
                if int(map_size) == 10:
                    env.render_10(pos[0], pos[1])
                elif int(map_size) == 4:
                    env.render_4(pos[0], pos[1])
                time.sleep(1)

