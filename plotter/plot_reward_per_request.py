import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
plt.rcParams["font.family"] = "Times New Roman"

from config import GlobalVar


def plot_reward_per_request_single_run(date_to_retrieve='YY_mm_dd_HH_MM_SS', show_graphs=True, color_index=0, algorithm="sarsa"):
    directory = GlobalVar.directory + 'output/log/'
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
        return commands, cum_rewards


def plot_reward_per_request_multiple_run(dates, algo):
    commands = []
    cum_rewards = []
    for index, dat in enumerate(dates):
        com, cr = plot_reward_per_request_single_run(date_to_retrieve=dat, show_graphs=False)
        commands.append(com)
        cum_rewards.append(cr)

        pl.plot(com, cr, label=algo + "-run" + str(dates.index(dat)))  # single line

    pl.xlabel('Number of sent commands')
    pl.ylabel('Cumulative reward')
    pl.legend(loc='upper right')
    pl.title('Cumulative reward over commands for ' + algo)
    pl.grid(True)
    plt.savefig('all_commands_' + algo + '.png')
    plt.show()


def plot_reward_per_multiple_algo(dates, algorithms):
    colors = ["#EB1E35", "#E37600", "#054AA6", "#038C02"]
    commands = []
    cum_rewards = []
    for index, dat in enumerate(dates):
        com, cr = plot_reward_per_request_single_run(date_to_retrieve=dat, show_graphs=False)
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


if __name__ == '__main__':
    sarsa = ['2020_11_05_03_27_46',
             '2020_11_05_04_07_23',
             '2020_11_05_04_48_59',
             '2020_11_05_05_30_35',
             '2020_11_05_06_10_02', ]

    sarsa_lambda = ['2020_11_05_06_47_59',
                    '2020_11_05_07_33_31',
                    '2020_11_05_08_04_47',
                    '2020_11_05_08_48_46',
                    '2020_11_05_09_35_46', ]

    qlearning = ['2020_11_05_10_24_34',
                 '2020_11_05_11_05_37',
                 '2020_11_05_11_48_23',
                 '2020_11_05_12_33_03',
                 '2020_11_05_13_16_54', ]

    qlearning_lambda = ['2020_11_05_13_54_50',
                        '2020_11_05_14_37_02',
                        '2020_11_05_15_10_00',
                        '2020_11_05_15_49_28',
                        '2020_11_05_16_27_15', ]

    algos = ["sarsa", "sarsa_lambda", "qlearning", "qlearning_lambda"]

    plot_reward_per_request_single_run(date_to_retrieve=sarsa[1], show_graphs=True, color_index=0, algorithm=algos[0])
    plot_reward_per_request_single_run(date_to_retrieve=sarsa_lambda[2], show_graphs=True, color_index=1, algorithm=algos[1])
    plot_reward_per_request_single_run(date_to_retrieve=qlearning[4], show_graphs=True, color_index=2, algorithm=algos[2])
    plot_reward_per_request_single_run(date_to_retrieve=qlearning_lambda[0], show_graphs=True, color_index=3, algorithm=algos[3])

    plot_reward_per_request_multiple_run(sarsa, algos[0])
    plot_reward_per_request_multiple_run(sarsa_lambda, algos[1])
    plot_reward_per_request_multiple_run(qlearning, algos[2])
    plot_reward_per_request_multiple_run(qlearning_lambda, algos[3])

    plot_reward_per_multiple_algo([sarsa[1], sarsa_lambda[2], qlearning[4], qlearning_lambda[0]], algos)
