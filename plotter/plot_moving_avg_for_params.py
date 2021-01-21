"""
    Script for plotting the moving average of the reward and timesteps over episodes for different params configurations
"""

import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import pylab as pl
from matplotlib.font_manager import FontProperties

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 20

fontP = FontProperties()
fontP.set_size('x-small')
n_cols = 2

from config import FrameworkConfiguration

# Functions for plotting the moving average for multiple runs and multiple configurations of params

target_dir = "../plot/robustness/"
complete_target_dir = ""


def return_greek_letter(par):
    if par == "epsilon":
        return "ε"
    elif par == "alpha":
        return "α"
    elif par == "gamma":
        return "γ"
    elif par == "lambda":
        return "λ"
    else:
        return "invalid"


def plot_single_configuration_multiple_runs(date_array, param):
    param_value = ""
    x_all = []
    y_all_reward = []
    y_all_cum_reward = []
    y_all_timesteps = []

    # retrieve data for all dates
    for dat in date_array:
        directory = FrameworkConfiguration.directory + 'output/output_Q_parameters'
        file_parameters = 'output_parameters_' + dat + '.csv'

        with open(directory + '/' + file_parameters, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

        algorithm = parameters['algorithm_used']
        print("RL ALGORITHM:", algorithm)
        param_value = parameters[param]

        directory = FrameworkConfiguration.directory + 'output/output_csv'
        filename = 'output_' + algorithm + '_' + dat + '.csv'

        x = []
        y_reward = []
        y_cum_reward = []
        y_timesteps = []
        with open(directory + '/' + filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            next(reader, None)
            for row in reader:
                x.append(int(row[0]))
                y_reward.append(int(row[1]))
                y_cum_reward.append(int(row[2]))
                y_timesteps.append(int(row[3]))

        # TODO remove this check only for robustness
        # if len(x) > 100:
        #     x = x[0:100]
        #     y_reward = y_reward[0:100]
        #     y_cum_reward = y_cum_reward[0:100]
        #     y_timesteps = y_timesteps[0:100]
        x_all.append(x)
        y_all_reward.append(y_reward)
        y_all_cum_reward.append(y_cum_reward)
        y_all_timesteps.append(y_timesteps)

    # compute average over multiple runs
    y_final_reward = []
    y_final_cum_reward = []
    y_final_timesteps = []
    for array_index in range(0, len(x_all[0])):
        sum_r = 0
        sum_cr = 0
        sum_t = 0
        count = 0
        for date_index in range(0, len(date_array)):  # compute average
            sum_r += y_all_reward[date_index][array_index]
            sum_cr += y_all_cum_reward[date_index][array_index]
            sum_t += y_all_timesteps[date_index][array_index]
            count += 1
        y_final_reward.append(sum_r / float(count))
        y_final_cum_reward.append(sum_cr / float(count))
        y_final_timesteps.append(sum_t / float(count))

    df_final_avg_over_n_runs = pd.DataFrame(
        {'x': x_all[0], 'y1': y_final_reward, 'y2': y_final_timesteps, 'y3': y_final_cum_reward})

    # color = ('#77FF82', '#47CC99', '#239DBA', '#006586')

    window_size = 10

    # Compute the smoothed moving average
    weights = np.repeat(1.0, window_size) / window_size
    yMA = np.convolve(df_final_avg_over_n_runs['y1'], weights, 'valid')
    yMA_timesteps = np.convolve(df_final_avg_over_n_runs['y2'], weights, 'valid')

    # Compute also the global avg reward and the number of rewards >= average in percentage
    # Compute also the global avg timesteps and the number of rewards <= average in percentage
    global_avg_reward = np.mean(y_final_reward)
    global_std_dev_reward = np.std(y_final_reward)
    global_avg_timesteps = np.mean(y_final_timesteps)
    global_std_dev_timesteps = np.std(y_final_timesteps)

    global_n_reward = sum(i >= global_avg_reward for i in y_final_reward) * 100 / float(len(y_final_reward))
    global_n_timesteps = sum(i <= global_avg_timesteps for i in y_final_timesteps) * 100 / float(len(y_final_timesteps))

    # plot results
    # pl.plot(df_single_run['x'], df_single_run['y1'], label='single run', color=color[0])  # single line
    # pl.plot(df_final_avg_over_n_runs['x'], df_final_avg_over_n_runs['y1'],
    #         label="avg over " + str(len(date_array)) + " run", color=color[1])  # avg line
    # pl.plot(df_final_avg_over_n_runs['x'], yMA[0:np.array(df_final_avg_over_n_runs['x']).shape[0]], 'r',
    #         label='moving average')  # moving avg line
    #
    # pl.xlabel('Episodes')
    # pl.ylabel('Reward')
    # pl.legend(loc='lower right')
    # pl.title('Reward for ' + algorithm + ' algorithm over episodes')
    # pl.grid(True)
    # plt.savefig('all_reward_plot.png')
    # plt.show()

    # plot results
    # pl.plot(df_single_run['x'], df_single_run['y2'], label='single run', color=color[0])  # single line
    # pl.plot(df_final_avg_over_n_runs['x'], df_final_avg_over_n_runs['y2'],
    #         label="avg over " + str(len(date_array)) + " run", color=color[1])  # avg line
    # pl.plot(df_final_avg_over_n_runs['x'], yMA_timesteps[0:np.array(df_final_avg_over_n_runs['x']).shape[0]], 'r',
    #         label='moving average')  # moving avg line
    #
    # pl.xlabel('Episodes')
    # pl.ylabel('Number of steps')
    # pl.legend(loc='lower right')
    # pl.title('Steps for ' + algorithm + ' algorithm over episodes')
    # pl.grid(True)
    # plt.savefig('all_timesteps_plot.png')
    # plt.show()

    return param_value, x_all[
        0], yMA, yMA_timesteps, global_avg_reward, global_avg_timesteps, global_n_reward, global_n_timesteps, global_std_dev_reward, global_std_dev_timesteps


def plot_multiple_configuration_moving_avg(algorithm, param, param_values_target, episodes_target,
                                           moving_average_rewards_target,
                                           moving_average_timesteps_target):
    # 3 colors for params
    # color = ('#A66066', '#F2A172', '#858C4A')

    for i in range(0, len(param_values_target)):
        if i == 0 and param == "lambda":
            pl.plot(episodes_target[i][
                    np.array(episodes_target[i]).shape[0] - np.array(moving_average_rewards_target[i]).shape[0]:],
                    moving_average_rewards_target[i],
                    label=return_greek_letter(param) + r'$=$' + param_values_target[i], )
        else:
            pl.plot(episodes_target[i][
                    np.array(episodes_target[i]).shape[0] - np.array(moving_average_rewards_target[i]).shape[0]:],
                    moving_average_rewards_target[i],
                    label=return_greek_letter(param) + r'$=$' + param_values_target[i].lstrip('0'), )  # color=color[i])

    pl.xlabel('Episode')
    pl.ylabel('Final reward')
    pl.legend(loc='lower right', prop=fontP, ncol=n_cols)
    # pl.title('Moving average of reward over episodes for ' + algorithm)
    pl.grid(True, color='gray', linestyle='dashed')
    pl.tight_layout()
    plt.savefig(complete_target_dir + 'mavg_reward_params.png')
    plt.show()

    for i in range(0, len(param_values_target)):
        if i == 0 and param == "lambda":
            pl.plot(episodes_target[i][
                    np.array(episodes_target[i]).shape[0] - np.array(moving_average_timesteps_target[i]).shape[0]:],
                    moving_average_timesteps_target[i],
                    label=return_greek_letter(param) + r'$=$' + param_values_target[i], )  # color=color[i])
        else:
            pl.plot(episodes_target[i][
                    np.array(episodes_target[i]).shape[0] - np.array(moving_average_timesteps_target[i]).shape[0]:],
                    moving_average_timesteps_target[i],
                    label=return_greek_letter(param) + r'$=$' + param_values_target[i].lstrip('0'), )

    pl.xlabel('Episode')
    pl.ylabel('Number of time steps')
    pl.legend(loc='upper right', prop=fontP, ncol=n_cols)
    # pl.title('Moving average of number of steps over episodes for ' + algorithm)
    pl.grid(True, color='gray', linestyle='dashed')
    pl.tight_layout()
    plt.savefig(complete_target_dir + 'mavg_timesteps_params.png')
    plt.show()


def plot_multiple_configuration_rewards_timesteps(algo, param, param_values, avg_rew, avg_steps, n_rew, n_steps,
                                                  std_dev_rew, std_dev_steps):
    fig, ax = plt.subplots()
    param_labels = []
    cnt = 0
    for i in param_values:
        if cnt == 0 and param == "lambda":
            param_labels.append(return_greek_letter(param) + "=" + i)
            cnt += 1
        else:
            param_labels.append(return_greek_letter(param) + "=" + i.lstrip('0'))
    col = ax.bar(param_labels,
                 avg_rew,
                 align='center')
    # color=('#A66066', '#F2A172', '#858C4A'))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg reward for episode')
    # ax.set_title('Avg reward for different configurations of ' + param)
    plt.axhline(0, color='black', lw=.3)

    fig.tight_layout()

    plt.savefig(complete_target_dir + 'avg_rewards_for_' + param + '.png')

    fig, ax = plt.subplots()
    param_labels = []
    cnt = 0
    for i in param_values:
        if cnt == 0 and param == "lambda":
            param_labels.append(return_greek_letter(param) + "=" + i)
            cnt += 1
        else:
            param_labels.append(return_greek_letter(param) + "=" + i.lstrip('0'))
    col = ax.bar(param_labels,
                 avg_steps, align='center', )
    # color=('#A66066', '#F2A172', '#858C4A'))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg time steps for episode')
    # ax.set_title('Avg steps for different configurations of ' + param)

    fig.tight_layout()

    plt.savefig(complete_target_dir + 'avg_steps_for_' + param + '.png')

    fig, ax = plt.subplots()
    col = ax.bar(param_values,
                 n_rewards, align='center', )  # color=('#A66066', '#F2A172', '#858C4A'))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage of rewards upon avg (%)')
    ax.set_title('Percentage of rewards over avg for different configurations of ' + param)

    fig.tight_layout()

    plt.savefig(complete_target_dir + 'n_rewards_for_' + param + '.png')

    fig, ax = plt.subplots()
    col = ax.bar(param_values,
                 n_steps,
                 align='center', )  # color=('#A66066', '#F2A172', '#858C4A'))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage of steps below avg (%)')
    ax.set_title('Percentage of steps below avg for different configurations of ' + param)

    fig.tight_layout()

    plt.savefig(complete_target_dir + 'n_steps_for_' + param + '.png')


def boxplot_multiple_configurations_rewards_timesteps_last_episodes(algor, param, values_of_param, last_20_rewards,
                                                                    last_20_timesteps):
    fig, ax = plt.subplots()

    # non sto più facendo una media, sto mettendo tutti i punti del reward medio
    # last 20 episodes rewards of 5 run -> 100 punti per box
    # [    run 1     run 2   run 3   run 4    run 5
    #     [ep 90]    ...     ...
    #     [ep 91]
    #     ...
    #     [ep 100]
    # ]
    col = ax.boxplot(last_20_rewards)

    ax.set_xticklabels(values_of_param)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg reward')
    ax.set_title('Avg reward in 5 runs of last 20 episodes per config of ' + param + ' for algo ' + algor)

    fig.tight_layout()

    plt.savefig('boxplot_param_reward_last_20.png')

    fig, ax = plt.subplots()

    col = ax.boxplot(last_20_timesteps)  # , ["SARSA", "SARSA(λ)", "Q-learning", "Q(λ)"])

    ax.set_xticklabels(values_of_param)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg time steps')
    ax.set_title('Avg time steps in 5 runs of last 20 episodes per config of ' + param + ' for algo ' + algor)

    fig.tight_layout()

    plt.savefig('boxplot_param_timestep_last_20.png')


if __name__ == '__main__':
    # I could pass a list of dates, then do the average of these dates
    # Then put multiple lines inside 1 multiline plot
    changing_param_values = []
    episodes = []
    moving_avgs_rewards = []
    moving_avgs_timesteps = []
    avg_rewards = []
    avg_timesteps = []
    n_rewards = []
    n_timesteps = []
    std_dev_rewards = []
    std_dev_timesteps = []

    boxplot_last_rewards = []
    boxplot_last_timesteps = []

    changing_param = "gamma"
    algo = "qlearning"
    complete_target_dir = target_dir + changing_param + "/" + algo + "/"
    print("ALGO SHOULD BE", algo, "FOR ALL RESULTS")
    value_of_gamma = [[
        # gam=0.35
        '2020_12_02_11_40_06',
        '2020_12_02_12_52_42',
        '2020_12_02_13_52_01',
        '2020_12_02_14_57_50',
        '2020_12_02_15_50_34',],
        [  # eps = 0.2   alp = 0.1   gam = 0.55
        '2020_11_20_19_55_08',
        '2020_11_20_21_25_32',
        '2020_11_20_22_52_02',
        '2020_11_21_00_14_24',
        '2020_11_21_01_42_09',],[
        # gam = 0.75
        '2020_12_02_16_58_01',
        '2020_12_02_17_50_35',
        '2020_12_02_18_51_47',
        '2020_12_02_20_10_51',
        '2020_12_02_21_07_42',]
    ]

    for val in value_of_gamma:
        p, ep, ma, mats, ar, at, nr, nt, sdr, sdt = plot_single_configuration_multiple_runs(date_array=val,
                                                                                            param=changing_param)
        changing_param_values.append(p)
        episodes.append(ep)
        moving_avgs_rewards.append(ma)
        moving_avgs_timesteps.append(mats)
        avg_rewards.append(ar)
        avg_timesteps.append(at)
        n_rewards.append(nr)
        n_timesteps.append(nt)
        std_dev_rewards.append(sdr)
        std_dev_timesteps.append(sdt)

    plot_multiple_configuration_moving_avg(algo, changing_param, changing_param_values, episodes, moving_avgs_rewards,
                                           moving_avgs_timesteps)

    plot_multiple_configuration_rewards_timesteps(algo, changing_param, changing_param_values, avg_rewards,
                                                  avg_timesteps, n_rewards, n_timesteps, std_dev_rewards,
                                                  std_dev_timesteps)
