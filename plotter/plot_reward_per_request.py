"""
    Script for plotting the cumulative reward over the number of sent commands by the RL algorithm
"""

import matplotlib.pyplot as plt
import pylab as pl
from matplotlib.font_manager import FontProperties
from plotter.plot_moving_avg import print_cute_algo_name
from plotter.support_plotter import read_all_info_from_log, build_output_dir_from_path, get_font_family_and_size, \
    get_extension

font_family, font_size = get_font_family_and_size()

plt.rcParams["font.family"] = font_family
plt.rcParams['font.size'] = font_size

fontP = FontProperties()
fontP.set_size('x-small')
n_cols = 1

output_dir = './'


def retrieve_reward_per_request_single_run(date_to_retrieve, show_intermediate_graphs=False, color_index=0, algorithm="sarsa"):
    """
    Retrieve the reward per each time step (or command sent by the RL algorithm) from log file
    for 1 single execution
    """

    episodes, commands, reward, cum_rewards = read_all_info_from_log(date_to_retrieve)

    if show_intermediate_graphs:
        colors = ["#EB1E35", "#E37600", "#054AA6", "#038C02"]

        pl.plot(commands, cum_rewards, label=algorithm, color=colors[color_index])

        pl.xlabel("Number of sent commands $\mathregular{n_a}$")
        pl.ylabel("Cumulative reward $\mathregular{C(n_a)}$")
        pl.legend(loc='upper right')
        # pl.title('Cumulative reward over commands for ' + algorithm)
        pl.grid(True)
        plt.savefig('commands_plot_' + algorithm + get_extension())
        pl.tight_layout()
        plt.show()

    return commands, cum_rewards, len(commands)


def compute_avg_reward_per_request_multiple_runs(dates, algo, show_intermediate_graphs=False):
    """
    Compute the average reward per commands over multiple executions
    Note that multiple executions of the same algorithms are likely to have different numbers of commands sent/timesteps
    This number depends on the single run
    """
    commands = []
    cum_rewards = []
    min_length = -1

    for index, dat in enumerate(dates):
        com, cr, cl = retrieve_reward_per_request_single_run(dat)
        commands.append(com)
        cum_rewards.append(cr)
        if min_length == -1:
            min_length = cl
        if cl < min_length:
            min_length = cl
        if show_intermediate_graphs:
            pl.plot(com, cr, label=algo + "-run" + str(dates.index(dat)))  # single line

    # iterate over cum_rewards and min_length of commands to compute the average of cum_rewards
    avg_cum_reward = []
    avg_commands = []
    for i in range(min_length):
        total_sum = 0.0
        total_cnt = 0.0
        for index, dat in enumerate(dates):
            total_sum += cum_rewards[index][i]
            total_cnt += 1
        avg_cum_reward.append(total_sum / total_cnt)
        avg_commands.append(i)

    if show_intermediate_graphs:
        pl.xlabel("Number of sent commands $\mathregular{n_a}$")
        pl.ylabel("Cumulative reward $\mathregular{C(n_a)}$")
        pl.legend(loc='upper right')
        # pl.title('Cumulative reward over commands for ' + algo)
        pl.grid(True)
        pl.tight_layout()
        plt.savefig('all_commands_' + algo + get_extension())
        plt.show()

    return avg_cum_reward, avg_commands


def plot_cum_reward_per_command_multiple_algos_for_specified_path(rewards, commands, algorithms, path):
    """
    Generate plot with the cumulative reward average over multiple executions
    for all algorithms used for 1 single path
    """
    target_output_dir = build_output_dir_from_path(output_dir, path)

    for index, al in enumerate(algorithms):
        pl.plot(commands[index], rewards[index], label=print_cute_algo_name(al))  # single line

    pl.xlabel('Number of sent commands $\mathregular{n_a}$')
    pl.ylabel('Cumulative reward $\mathregular{C(n_a)}$')
    pl.legend(loc='upper left', prop=fontP, ncol=n_cols)
    # pl.title('Cumulative reward over commands for algos')
    pl.grid(True, color='gray', linestyle='dashed')
    pl.tight_layout()
    plt.savefig(target_output_dir + 'all_commands_all_algos' + get_extension())
    plt.show()


def main():
    algos = ["sarsa", "sarsa_lambda", "qlearning", "qlearning_lambda"]

    target_path = 1
    print("PATH ", target_path)

    from dates_for_graphs.date_for_graphs_path1 import sarsa_dates
    from dates_for_graphs.date_for_graphs_path1 import sarsa_lambda_dates
    from dates_for_graphs.date_for_graphs_path1 import qlearning_dates
    from dates_for_graphs.date_for_graphs_path1 import qlearning_lambda_dates

    all_cum_rewards = []
    all_avg_commands = []

    avg_cum_rew, avg_com = compute_avg_reward_per_request_multiple_runs(sarsa_dates, algos[0])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = compute_avg_reward_per_request_multiple_runs(sarsa_lambda_dates, algos[1])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = compute_avg_reward_per_request_multiple_runs(qlearning_dates, algos[2])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = compute_avg_reward_per_request_multiple_runs(qlearning_lambda_dates, algos[3])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)

    plot_cum_reward_per_command_multiple_algos_for_specified_path(all_cum_rewards, all_avg_commands, algos, target_path)
    target_path = 2
    print("PATH ", target_path)

    from dates_for_graphs.date_for_graphs_path2 import sarsa_dates
    from dates_for_graphs.date_for_graphs_path2 import sarsa_lambda_dates
    from dates_for_graphs.date_for_graphs_path2 import qlearning_dates
    from dates_for_graphs.date_for_graphs_path2 import qlearning_lambda_dates

    all_cum_rewards = []
    all_avg_commands = []

    avg_cum_rew, avg_com = compute_avg_reward_per_request_multiple_runs(sarsa_dates, algos[0])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = compute_avg_reward_per_request_multiple_runs(sarsa_lambda_dates, algos[1])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = compute_avg_reward_per_request_multiple_runs(qlearning_dates, algos[2])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = compute_avg_reward_per_request_multiple_runs(qlearning_lambda_dates, algos[3])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)

    plot_cum_reward_per_command_multiple_algos_for_specified_path(all_cum_rewards, all_avg_commands, algos, target_path)

    target_path = 3
    print("PATH ", target_path)

    from dates_for_graphs.date_for_graphs_path3 import sarsa_dates
    from dates_for_graphs.date_for_graphs_path3 import sarsa_lambda_dates
    from dates_for_graphs.date_for_graphs_path3 import qlearning_dates
    from dates_for_graphs.date_for_graphs_path3 import qlearning_lambda_dates

    all_cum_rewards = []
    all_avg_commands = []

    avg_cum_rew, avg_com = compute_avg_reward_per_request_multiple_runs(sarsa_dates, algos[0])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = compute_avg_reward_per_request_multiple_runs(sarsa_lambda_dates, algos[1])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = compute_avg_reward_per_request_multiple_runs(qlearning_dates, algos[2])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = compute_avg_reward_per_request_multiple_runs(qlearning_lambda_dates, algos[3])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)

    plot_cum_reward_per_command_multiple_algos_for_specified_path(all_cum_rewards, all_avg_commands, algos, target_path)

    target_path = 4
    print("PATH ", target_path)

    from dates_for_graphs.date_for_graphs_path4 import sarsa_dates
    from dates_for_graphs.date_for_graphs_path4 import sarsa_lambda_dates
    from dates_for_graphs.date_for_graphs_path4 import qlearning_dates
    from dates_for_graphs.date_for_graphs_path4 import qlearning_lambda_dates

    all_cum_rewards = []
    all_avg_commands = []

    avg_cum_rew, avg_com = compute_avg_reward_per_request_multiple_runs(sarsa_dates, algos[0])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = compute_avg_reward_per_request_multiple_runs(sarsa_lambda_dates, algos[1])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = compute_avg_reward_per_request_multiple_runs(qlearning_dates, algos[2])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = compute_avg_reward_per_request_multiple_runs(qlearning_lambda_dates, algos[3])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)

    plot_cum_reward_per_command_multiple_algos_for_specified_path(all_cum_rewards, all_avg_commands, algos, target_path)


if __name__ == '__main__':
    main()
