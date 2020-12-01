import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from matplotlib.font_manager import FontProperties

from plotter.plot_moving_avg import print_cute_algo_name

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 20
fontP = FontProperties()
fontP.set_size('x-small')
n_cols = 1

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
    # sarsa = ['2020_11_05_03_27_46',
    #          '2020_11_05_04_07_23',
    #          '2020_11_05_04_48_59',
    #          '2020_11_05_05_30_35',
    #          '2020_11_05_06_10_02', ]
    #
    # sarsa_lambda = ['2020_11_05_06_47_59',
    #                 '2020_11_05_07_33_31',
    #                 '2020_11_05_08_04_47',
    #                 '2020_11_05_08_48_46',
    #                 '2020_11_05_09_35_46', ]
    #
    # qlearning = ['2020_11_05_10_24_34',
    #              '2020_11_05_11_05_37',
    #              '2020_11_05_11_48_23',
    #              '2020_11_05_12_33_03',
    #              '2020_11_05_13_16_54', ]
    #
    # qlearning_lambda = ['2020_11_05_13_54_50',
    #                     '2020_11_05_14_37_02',
    #                     '2020_11_05_15_10_00',
    #                     '2020_11_05_15_49_28',
    #                     '2020_11_05_16_27_15', ]

    # algos = ["sarsa", "sarsa_lambda", "qlearning", "qlearning_lambda"]

    # plot_reward_per_request_single_run(date_to_retrieve=sarsa[1], show_graphs=True, color_index=0, algorithm=algos[0])
    # plot_reward_per_request_single_run(date_to_retrieve=sarsa_lambda[2], show_graphs=True, color_index=1, algorithm=algos[1])
    # plot_reward_per_request_single_run(date_to_retrieve=qlearning[4], show_graphs=True, color_index=2, algorithm=algos[2])
    # plot_reward_per_request_single_run(date_to_retrieve=qlearning_lambda[0], show_graphs=True, color_index=3, algorithm=algos[3])

    # plot_reward_per_multiple_algo([sarsa[1], sarsa_lambda[2], qlearning[4], qlearning_lambda[0]], algos)

    algos = ["sarsa", "sarsa_lambda", "qlearning", "qlearning_lambda"]

    target_path = 1
    print("PATH ", target_path)
    sarsa_dates = [
        '2020_11_25_12_48_15',
        '2020_11_25_13_37_48',
        '2020_11_25_14_29_17',
        '2020_11_25_15_35_12',
        '2020_11_25_16_36_56',
        '2020_11_25_17_24_57',
        '2020_11_25_18_14_50',
        '2020_11_25_19_16_07',
        '2020_11_25_20_05_06',
        '2020_11_25_21_03_28',
    ]

    sarsa_lambda_dates = [
        '2020_11_25_21_54_44',
        '2020_11_25_22_49_27',
        '2020_11_25_23_40_38',
        '2020_11_26_00_37_06',
        '2020_11_26_01_30_01',
        '2020_11_26_02_25_21',
        '2020_11_26_03_22_27',
        '2020_11_26_04_12_40',
        '2020_11_26_05_05_32',
        '2020_11_26_05_52_05',
    ]

    qlearning_dates = [
        '2020_11_22_02_19_21',
        '2020_11_22_03_29_10',
        '2020_11_22_04_18_21',
        '2020_11_22_05_13_56',
        '2020_11_22_06_10_46',
        '2020_11_22_07_24_00',
        '2020_11_22_08_21_59',
        '2020_11_22_09_36_02',
        '2020_11_22_10_25_36',
        '2020_11_22_11_14_52',
    ]

    qlearning_lambda_dates = [
        '2020_11_22_12_02_03',
        '2020_11_22_12_50_14',
        '2020_11_22_13_55_12',
        '2020_11_22_14_52_53',
        '2020_11_22_15_45_11',
        '2020_11_22_16_39_57',
        '2020_11_22_17_35_14',
        '2020_11_22_18_43_39',
        '2020_11_22_19_49_24',
        '2020_11_22_20_46_00',
    ]

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

    sarsa_dates = [
        '2020_11_24_15_03_34',
        '2020_11_24_15_51_12',
        '2020_11_24_16_51_27',
        '2020_11_24_17_45_45',
        '2020_11_24_18_38_37',
        '2020_11_24_19_34_52',
        '2020_11_24_20_26_29',
        '2020_11_24_21_19_02',
        '2020_11_24_22_12_30',
        '2020_11_24_23_03_13',
    ]

    sarsa_lambda_dates = [
        '2020_11_25_00_05_17',
        '2020_11_25_01_00_18',
        '2020_11_25_02_01_42',
        '2020_11_25_03_02_56',
        '2020_11_25_03_55_24',
        '2020_11_25_04_47_54',
        '2020_11_25_05_45_45',
        '2020_11_25_06_46_58',
        '2020_11_25_07_39_08',
        '2020_11_25_08_34_32',
    ]

    qlearning_dates = [
        '2020_11_19_23_20_35',
        '2020_11_20_00_45_39',
        '2020_11_20_01_43_25',
        '2020_11_20_02_58_24',
        '2020_11_20_04_02_19',
        '2020_11_20_04_57_58',
        '2020_11_20_05_52_01',
        '2020_11_20_06_51_41',
        '2020_11_20_07_50_17',
        '2020_11_20_08_48_08',
    ]

    qlearning_lambda_dates = [
        '2020_11_20_09_41_30',
        '2020_11_20_10_36_29',
        '2020_11_20_11_27_28',
        '2020_11_20_12_27_46',
        '2020_11_20_13_35_56',
        '2020_11_20_14_39_49',
        '2020_11_20_15_45_16',
        '2020_11_20_16_43_07',
        '2020_11_20_17_39_35',
        '2020_11_20_18_44_51',
    ]

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

    sarsa_dates = [
        '2020_11_22_22_20_40',
        '2020_11_22_23_44_50',
        '2020_11_23_01_14_07',
        '2020_11_23_03_01_07',
        '2020_11_23_04_33_00',
        '2020_11_23_05_50_31',
        '2020_11_23_07_13_52',
        '2020_11_23_08_35_52',
        '2020_11_23_09_57_28',
        '2020_11_23_11_36_57',
    ]

    sarsa_lambda_dates = [
        '2020_11_23_18_01_20',
        '2020_11_23_19_27_16',
        '2020_11_23_20_40_12',
        '2020_11_23_21_54_42',
        '2020_11_23_23_24_23',
        '2020_11_24_00_41_00',
        '2020_11_24_02_15_22',
        '2020_11_24_03_27_53',
        '2020_11_24_04_40_32',
        '2020_11_24_05_59_40',
    ]

    qlearning_dates = [
        '2020_11_20_19_55_08',
        '2020_11_20_21_25_32',
        '2020_11_20_22_52_02',
        '2020_11_21_00_14_24',
        '2020_11_21_01_42_09',
        '2020_11_21_03_01_06',
        '2020_11_21_04_21_03',
        '2020_11_21_05_36_05',
        '2020_11_21_07_01_27',
        '2020_11_21_08_22_04',
    ]

    qlearning_lambda_dates = [
        '2020_11_21_09_39_37',
        '2020_11_21_11_09_13',
        '2020_11_21_12_30_19',
        '2020_11_21_13_58_09',
        '2020_11_21_15_29_01',
        '2020_11_21_16_49_23',
        '2020_11_21_18_13_51',
        '2020_11_21_19_54_03',
        '2020_11_21_21_43_57',
        '2020_11_21_23_07_58',
    ]

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
