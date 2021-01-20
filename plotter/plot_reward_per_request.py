"""
    Script for plotting the cumulative reward over the number of sent commands by the RL algorithm
"""

import matplotlib.pyplot as plt
import pylab as pl
from matplotlib.font_manager import FontProperties

from plotter.plot_moving_avg import print_cute_algo_name

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 20
fontP = FontProperties()
fontP.set_size('x-small')
n_cols = 1

from config import FrameworkConfiguration


def plot_reward_per_request_single_run(date_to_retrieve='YY_mm_dd_HH_MM_SS', show_graphs=True, color_index=0, algorithm="sarsa"):
    directory = FrameworkConfiguration.directory + 'output/log/'
    log_file = directory + 'log_' + date_to_retrieve + '.log'

    print(log_file)

    # Each non empty line is a sent command
    # Command of power is substituted by episode finishing line
    # Minus last line that is the total time

    count = 0
    cum_reward = 0
    commands = []
    rewards = []
    cum_rewards = []
    episodes = []
    with open(log_file) as f:
        for line in f:
            if len(line.strip()) != 0:  # Not empty lines
                if line.startswith("Episode"):
                    episodes.append(count)
                if not line.startswith("Episode") and not line.startswith("Total"):
                    count += 1
                    commands.append(count)
                    tmp_reward = int(line.split()[5])
                    cum_reward += tmp_reward
                    rewards.append(tmp_reward)
                    cum_rewards.append(cum_reward)

    colors = ["#EB1E35", "#E37600", "#054AA6", "#038C02"]

    if show_graphs:
        # pl.plot(commands, rewards, label='reward')  # single line
        pl.plot(commands, cum_rewards, label=algorithm, color=colors[color_index])  # single line

        pl.xlabel('Number of sent commands')
        pl.ylabel('Cumulative reward')
        pl.legend(loc='upper right')
        pl.title('Cumulative reward over commands for ' + algorithm)
        pl.grid(True)
        plt.savefig('commands_plot_' + algorithm + '_lambda.png')
        plt.show()

    else:
        return commands, cum_rewards, len(commands)


# returns averages
def plot_reward_per_request_multiple_run(dates, algo, show_graphs=False):
    commands = []
    cum_rewards = []
    min_length = -1

    for index, dat in enumerate(dates):
        com, cr, cl = plot_reward_per_request_single_run(date_to_retrieve=dat, show_graphs=False)
        commands.append(com)
        cum_rewards.append(cr)
        if min_length == -1:
            min_length = cl
        if cl < min_length:
            min_length = cl
        if show_graphs:
            pl.plot(com, cr, label=algo + "-run" + str(dates.index(dat)))  # single line

    # iterate over cum_rewards and min_length of commands to compute the average of cum_rewards
    avg_cum_reward = []
    avg_commands = []
    for i in range(min_length):
        sum = 0.0
        cnt = 0.0
        for index, dat in enumerate(dates):
            sum += cum_rewards[index][i]
            cnt += 1
        avg_cum_reward.append(sum/cnt)
        avg_commands.append(i)
    if show_graphs:
        pl.xlabel('Number of sent commands')
        pl.ylabel('Cumulative reward')
        pl.legend(loc='upper right')
        pl.title('Cumulative reward over commands for ' + algo)
        pl.grid(True)
        plt.savefig('all_commands_' + algo + '.png')
        plt.show()

    return avg_cum_reward, avg_commands


def plot_reward_per_multiple_algo(dates, algorithms):
    colors = ["#EB1E35", "#E37600", "#054AA6", "#038C02"]
    commands = []
    cum_rewards = []
    for index, dat in enumerate(dates):
        com, cr, cl = plot_reward_per_request_single_run(date_to_retrieve=dat, show_graphs=False)
        commands.append(com)
        cum_rewards.append(cr)

        pl.plot(com, cr, label=algorithms[dates.index(dat)], color=colors[index])  # single line

    pl.xlabel('Number of sent commands')
    pl.ylabel('Cumulative reward')
    pl.legend(loc='upper right')
    pl.title('Cumulative reward over commands per different algorithms')
    pl.grid(True)
    plt.savefig('all_commands_all_algo.png')
    plt.show()


def plot_reward_per_multiple_algo_per_path(dates, algorithms, path):
    colors = ["#EB1E35", "#E37600", "#054AA6", "#038C02"]
    commands = []
    cum_rewards = []
    for index, dat in enumerate(dates):
        com, cr, cl = plot_reward_per_request_single_run(date_to_retrieve=dat, show_graphs=False)
        commands.append(com)
        cum_rewards.append(cr)

        pl.plot(com, cr, label=algorithms[dates.index(dat)], color=colors[index])  # single line

    pl.xlabel('Number of sent commands')
    pl.ylabel('Cumulative reward')
    pl.legend(loc='upper right')
    pl.title('Cumulative reward over commands per different algorithms')
    pl.grid(True)
    plt.savefig('all_commands_all_algo.png')
    plt.show()


def plot_reward_per_request_multiple_algos_all_paths(rewards, commands, algorithms, path):

    for index, al in enumerate(algorithms):
        pl.plot(commands[index], rewards[index], label=print_cute_algo_name(al))  # single line

    pl.xlabel('Number of sent commands')
    pl.ylabel('Cumulative reward')
    pl.legend(loc='upper left', prop=fontP, ncol=n_cols)
    # pl.title('Cumulative reward over commands for algos')
    pl.grid(True, color='gray', linestyle='dashed')
    pl.tight_layout()
    plt.savefig("../plot/path" + str(path) + "/" + 'all_commands_all_algos.png')
    plt.show()


if __name__ == '__main__':
    # from date_for_graphs_before_tuning_path2 import sarsa
    # from date_for_graphs_before_tuning_path2 import sarsa_lambda
    # from date_for_graphs_before_tuning_path2 import qlearning
    # from date_for_graphs_before_tuning_path2 import qlearning_lambda

    # algos = ["sarsa", "sarsa_lambda", "qlearning", "qlearning_lambda"]

    # plot_reward_per_request_single_run(date_to_retrieve=sarsa[1], show_graphs=True, color_index=0, algorithm=algos[0])
    # plot_reward_per_request_single_run(date_to_retrieve=sarsa_lambda[2], show_graphs=True, color_index=1, algorithm=algos[1])
    # plot_reward_per_request_single_run(date_to_retrieve=qlearning[4], show_graphs=True, color_index=2, algorithm=algos[2])
    # plot_reward_per_request_single_run(date_to_retrieve=qlearning_lambda[0], show_graphs=True, color_index=3, algorithm=algos[3])

    # plot_reward_per_multiple_algo([sarsa[1], sarsa_lambda[2], qlearning[4], qlearning_lambda[0]], algos)

    algos = ["sarsa", "sarsa_lambda", "qlearning", "qlearning_lambda"]

    target_path = 1
    print("PATH ", target_path)

    from date_for_graphs_path1 import sarsa_dates
    from date_for_graphs_path1 import sarsa_lambda_dates
    from date_for_graphs_path1 import qlearning_dates
    from date_for_graphs_path1 import qlearning_lambda_dates

    all_cum_rewards = []
    all_avg_commands = []

    avg_cum_rew, avg_com = plot_reward_per_request_multiple_run(sarsa_dates, algos[0])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = plot_reward_per_request_multiple_run(sarsa_lambda_dates, algos[1])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = plot_reward_per_request_multiple_run(qlearning_dates, algos[2])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = plot_reward_per_request_multiple_run(qlearning_lambda_dates, algos[3])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)

    plot_reward_per_request_multiple_algos_all_paths(all_cum_rewards, all_avg_commands, algos, target_path)
    target_path = 2
    print("PATH ", target_path)

    from date_for_graphs_path2 import sarsa_dates
    from date_for_graphs_path2 import sarsa_lambda_dates
    from date_for_graphs_path2 import qlearning_dates
    from date_for_graphs_path2 import qlearning_lambda_dates

    all_cum_rewards = []
    all_avg_commands = []

    avg_cum_rew, avg_com = plot_reward_per_request_multiple_run(sarsa_dates, algos[0])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = plot_reward_per_request_multiple_run(sarsa_lambda_dates, algos[1])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = plot_reward_per_request_multiple_run(qlearning_dates, algos[2])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = plot_reward_per_request_multiple_run(qlearning_lambda_dates, algos[3])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)

    plot_reward_per_request_multiple_algos_all_paths(all_cum_rewards, all_avg_commands, algos, target_path)

    target_path = 3
    print("PATH ", target_path)

    from date_for_graphs_path3 import sarsa_dates
    from date_for_graphs_path3 import sarsa_lambda_dates
    from date_for_graphs_path3 import qlearning_dates
    from date_for_graphs_path3 import qlearning_lambda_dates

    all_cum_rewards = []
    all_avg_commands = []

    avg_cum_rew, avg_com = plot_reward_per_request_multiple_run(sarsa_dates, algos[0])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = plot_reward_per_request_multiple_run(sarsa_lambda_dates, algos[1])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = plot_reward_per_request_multiple_run(qlearning_dates, algos[2])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)
    avg_cum_rew, avg_com = plot_reward_per_request_multiple_run(qlearning_lambda_dates, algos[3])
    all_cum_rewards.append(avg_cum_rew)
    all_avg_commands.append(avg_com)

    plot_reward_per_request_multiple_algos_all_paths(all_cum_rewards, all_avg_commands, algos, target_path)
