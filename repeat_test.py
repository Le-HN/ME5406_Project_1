from q_learning import *
from sarsa_Q import *
from monte_carlo_Q import *
from monte_carlo_V import *
import matplotlib.pyplot as plt


training_type = "ALL"


if __name__ == '__main__':

    if training_type == "Q_LEARNING":
        e_l, ar_l, ar_q_l, world, route, env, result = q_learning(iteration_lim=3000)
    elif training_type == "SARSA":
        e_l, ar_l, ar_q_l, world, route, env, result = sarsa_q(iteration_lim=3000)
    elif training_type == "MONTE_CARLO_Q":
        e_l, ar_l, world, route, env, result = monte_carlo_q(iteration_lim=60000)
    elif training_type == "MONTE_CARLO_V":
        e_l, ar_l, world, route, env, result = monte_carlo_v(iteration_lim=50000)
    elif training_type == 'ALL':
        iteration_test = 10000
        e_l_ql, ar_l_ql, ar_q_l_ql, world_1, route_1, env_1, result_1 = q_learning(iteration_lim=iteration_test)
        e_l_sa, ar_l_sa, ar_q_l_sa, world_2, route_2, env_2, result_2 = sarsa_q(iteration_lim=iteration_test)
        e_l_mcq, ar_l_mcq, world_3, route_3, env_3, result_3 = monte_carlo_q(iteration_lim=iteration_test)
        e_l_mcv, ar_l_mcv, world_4, route_4, env_4, result_4 = monte_carlo_v(iteration_lim=iteration_test)
        # successful_num += sc_n
        # shortest_num += st_n
    # print(world)
    if training_type == 'ALL':
        plt.plot(e_l_ql, ar_l_ql, label="Q-Learning")
        plt.plot(e_l_sa, ar_l_sa, label="SARSA")
        plt.plot(e_l_mcq, ar_l_mcq, label="Monte Carlo Q")
        plt.plot(e_l_mcv, ar_l_mcv, label="Monte Carlo V")
    else:
        plt.plot(e_l, ar_l, label="")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.ylim(-1, 1)
    plt.legend()
    plt.show()
    # print(successful_num)
    # print(shortest_num)
