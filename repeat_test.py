from q_learning import *
from sarsa_Q import *


training_type = "SARSA"


if __name__ == '__main__':
    successful_num = 0
    shortest_num = 0
    for t_round in range(0, 100):
        print(t_round)
        if training_type == "Q_LEARNING":
            sc_n, st_n, world, route, env, result = q_learning(iteration_lim=500)
        elif training_type == "SARSA":
            sc_n, st_n, world, route, env, result = sarsa_q(iteration_lim=500)
        successful_num += sc_n
        shortest_num += st_n
    print(world)
    print(successful_num)
    print(shortest_num)
